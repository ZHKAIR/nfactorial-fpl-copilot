"""Цитируемый ответ на стратегический вопрос: KBRetriever -> gpt-4o-mini (structured output, T=0)
-> детерминированная проверка ссылок [n] и цитат -> {answer, citations, model, usage}.

Промпт — rag/kb/prompts/<version>/strategy_answer.system.md (версионируется файлами, как и
prompts/ новостного RAG). Правила промпта: отвечать только по документам, ссылки [n], «not covered»
при отсутствии ответа, правила (тег rules) авторитетнее советов, без ставок, без выдуманных чисел
о конкретном составе (числа считают инструменты). Код проверяет то, чему нельзя доверять модели:

- каждая ссылка [n] в тексте указывает на существующий документ, иначе удаляется;
- каждая цитата — verbatim-подстрока своего документа (rag.extract.verbatim_quote: пробелы и
  точка в конце прощаются; почти-verbatim выравнивается align_quote, иначе отбрасывается);
- документ, на который ссылается текст, обязан иметь валидную цитату — иначе ссылка снимается;
- covered=false или ни одной валидной цитаты -> фиксированный ответ NOT_COVERED без цитат.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from fplcopilot.config import settings
from fplcopilot.rag.extract import MAX_QUOTE_CHARS, align_quote, verbatim_quote
from fplcopilot.rag.kb.retrieve import DEFAULT_K, KBChunk, KBMode, KBRetriever, chunk_to_dict
from fplcopilot.rag.llm import configure_tracing, get_openai_client

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_NAME = "strategy_answer"
PROMPT_VERSION = "v1"
NOT_COVERED = "Not covered by the strategy knowledge base."
DEFAULT_MODEL = settings.rag_llm_model  # gpt-4o-mini
MAX_COMPLETION_TOKENS = 600
# list prices, USD per 1M tokens — только для оценки стоимости в результатах
PRICE_PER_1M: dict[str, tuple[float, float]] = {"gpt-4o-mini": (0.15, 0.60)}

_REF = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


# ---------- промпт ----------


@lru_cache(maxsize=4)
def load_system_prompt(version: str = PROMPT_VERSION) -> str:
    path = PROMPTS_DIR / version / f"{PROMPT_NAME}.system.md"
    if not path.is_file():
        available = sorted(p.parent.name for p in PROMPTS_DIR.glob(f"v*/{PROMPT_NAME}.system.md"))
        raise ValueError(f"prompt version {version!r} не найдена; есть: {available}")
    return path.read_text(encoding="utf-8").strip()


def _escape(value: str) -> str:
    return value.replace('"', "'").replace("<", "‹").replace(">", "›")


def render_documents(chunks: Sequence[KBChunk]) -> str:
    """Нумерованные <document n=…> блоки; текст чанка — ДАННЫЕ (вложенные теги нейтрализуются)."""
    parts = []
    for n, c in enumerate(chunks, start=1):
        body = c.text.replace("</document", "</document ").replace("<document", "<document ")
        parts.append(
            f'<document n="{n}" source="{_escape(c.source)}" tags="{",".join(c.tags)}" '
            f'title="{_escape(c.title)}">\n{body}\n</document>'
        )
    return "\n\n".join(parts) if parts else "(no documents retrieved)"


def build_messages(
    question: str, chunks: Sequence[KBChunk], *, prompt_version: str = PROMPT_VERSION
) -> list[dict[str, str]]:
    user = f"Question: {question.strip()}\n\nDocuments ({len(chunks)}):\n{render_documents(chunks)}"
    return [
        {"role": "system", "content": load_system_prompt(prompt_version)},
        {"role": "user", "content": user},
    ]


# ---------- схема ответа модели ----------


class CitationDraft(BaseModel):
    n: int = Field(description="document number as shown in <document n=...>")
    quote: str = Field(description="verbatim substring of that document's text, <= 200 characters")


class StrategyAnswerDraft(BaseModel):
    """Порядок полей намеренный: модель сначала выбирает цитаты, потом пишет ответ по ним."""

    covered: bool = Field(description="false when the documents do not answer the question")
    citations: list[CitationDraft] = Field(
        description="one verbatim quote per document you rely on; empty when covered=false"
    )
    answer: str = Field(description="answer with [n] markers, or the exact not-covered sentence")


# ---------- детерминированная проверка ----------


@dataclass
class Citation:
    n: int
    chunk_id: int
    title: str
    url: str
    source: str
    tags: list[str]
    quote: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "chunk_id": self.chunk_id,
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "tags": list(self.tags),
            "quote": self.quote,
        }


@dataclass
class ValidatedAnswer:
    answer: str
    covered: bool
    citations: list[Citation] = field(default_factory=list)
    fixes: int = 0
    dropped_citations: int = 0
    removed_refs: list[int] = field(default_factory=list)
    aligned_quotes: int = 0
    forced_not_covered: bool = False

    def validation_dict(self) -> dict[str, Any]:
        return {
            "fixes": self.fixes,
            "dropped_citations": self.dropped_citations,
            "removed_refs": self.removed_refs,
            "aligned_quotes": self.aligned_quotes,
            "forced_not_covered": self.forced_not_covered,
        }


def parse_refs(answer: str) -> list[int]:
    """Номера [n] в порядке появления (поддерживается и «[1, 3]»); без дубликатов."""
    out: list[int] = []
    for m in _REF.finditer(answer or ""):
        for part in m.group(1).split(","):
            n = int(part.strip())
            if n not in out:
                out.append(n)
    return out


def strip_refs(answer: str, remove: set[int]) -> str:
    """Убрать из текста ссылки на номера remove; «[1, 3]» с одним плохим номером -> «[1]»."""
    if not remove:
        return answer

    def repl(m: re.Match[str]) -> str:
        keep = [p.strip() for p in m.group(1).split(",") if int(p.strip()) not in remove]
        return "".join(f"[{k}]" for k in keep)

    text = _REF.sub(repl, answer)
    return re.sub(r"\s+([.,;:!?])", r"\1", re.sub(r"[ \t]{2,}", " ", text)).strip()


def validate_answer(draft: StrategyAnswerDraft, chunks: Sequence[KBChunk]) -> ValidatedAnswer:
    """Правила из докстринга модуля; возвращает исправленный ответ и валидные цитаты."""
    n_docs = len(chunks)
    if not draft.covered:
        forced = draft.answer.strip() != NOT_COVERED or bool(draft.citations)
        return ValidatedAnswer(
            answer=NOT_COVERED, covered=False, fixes=int(forced), removed_refs=parse_refs(draft.answer)
        )  # fmt: skip

    out = ValidatedAnswer(answer=draft.answer.strip(), covered=True)
    citations: dict[int, Citation] = {}
    for cit in draft.citations:
        if not 1 <= cit.n <= n_docs:
            out.dropped_citations += 1
            out.fixes += 1
            continue
        chunk = chunks[cit.n - 1]
        quote = verbatim_quote(cit.quote[:MAX_QUOTE_CHARS], chunk.text)
        if quote is None:
            aligned = align_quote(cit.quote, chunk.text)
            out.fixes += 1
            if aligned is None:
                out.dropped_citations += 1
                log.debug("kb citation dropped (doc %d): %r", cit.n, cit.quote)
                continue
            out.aligned_quotes += 1
            quote = aligned
        if cit.n in citations:
            continue  # одна цитата на документ
        citations[cit.n] = Citation(
            n=cit.n,
            chunk_id=chunk.chunk_id,
            title=chunk.title,
            url=chunk.url,
            source=chunk.source,
            tags=list(chunk.tags),
            quote=quote,
        )

    refs = parse_refs(out.answer)
    bad = {n for n in refs if not 1 <= n <= n_docs or n not in citations}
    if bad:
        out.removed_refs = sorted(bad)
        out.fixes += len(bad)
        out.answer = strip_refs(out.answer, bad)

    if not citations:
        return ValidatedAnswer(
            answer=NOT_COVERED,
            covered=False,
            fixes=out.fixes + 1,
            dropped_citations=out.dropped_citations,
            removed_refs=out.removed_refs,
            aligned_quotes=out.aligned_quotes,
            forced_not_covered=True,
        )
    out.citations = [citations[n] for n in sorted(citations)]
    return out


# ---------- LLM ----------


def call_llm(
    messages: list[dict[str, str]], *, model: str = DEFAULT_MODEL
) -> tuple[StrategyAnswerDraft, dict[str, int]]:
    client = get_openai_client()
    completion = client.chat.completions.parse(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        response_format=StrategyAnswerDraft,
        temperature=0,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )
    msg = completion.choices[0].message
    usage = completion.usage
    tokens = {
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
    }
    if msg.parsed is None:
        log.warning("kb answer: refusal/empty parse: %s", msg.refusal)
        return StrategyAnswerDraft(covered=False, answer=NOT_COVERED, citations=[]), tokens
    return msg.parsed, tokens


def estimate_cost(usage: dict[str, int], model: str) -> float | None:
    price = PRICE_PER_1M.get(model)
    if price is None:
        return None
    return (usage["prompt_tokens"] * price[0] + usage["completion_tokens"] * price[1]) / 1e6


def answer_strategy_question(
    query: str,
    *,
    k: int = DEFAULT_K,
    tags: Sequence[str] | None = None,
    mode: KBMode = "hybrid_rerank",
    retriever: KBRetriever | None = None,
    model: str = DEFAULT_MODEL,
    prompt_version: str = PROMPT_VERSION,
) -> dict[str, Any]:
    """Полный конвейер. Возвращает dict: answer, covered, citations[{title,url,quote,…}], model,
    usage, cost_usd, retrieved, validation, timings."""
    configure_tracing()
    started = time.perf_counter()
    retriever = retriever or KBRetriever()
    chunks = retriever.search(query, tags=tags, k=k, mode=mode)
    retrieve_ms = (time.perf_counter() - started) * 1000

    t = time.perf_counter()
    if chunks:
        draft, usage = call_llm(
            build_messages(query, chunks, prompt_version=prompt_version), model=model
        )
    else:  # пустой корпус / слишком узкий фильтр тегов — LLM не нужен
        draft, usage = (
            StrategyAnswerDraft(covered=False, answer=NOT_COVERED, citations=[]),
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
        )
    llm_ms = (time.perf_counter() - t) * 1000
    validated = validate_answer(draft, chunks)
    return {
        "question": query,
        "answer": validated.answer,
        "covered": validated.covered,
        "citations": [c.to_dict() for c in validated.citations],
        "model": model,
        "prompt_version": prompt_version,
        "usage": usage,
        "cost_usd": estimate_cost(usage, model),
        "llm_calls": 1 if chunks else 0,
        "mode": mode,
        "tags": list(tags) if tags else None,
        "k": k,
        "retrieved": [chunk_to_dict(c) for c in chunks],
        "raw_answer": draft.answer,
        "raw_citations": [c.model_dump() for c in draft.citations],
        "validation": validated.validation_dict(),
        "timings": {
            "retrieve_ms": retrieve_ms,
            "llm_ms": llm_ms,
            "total_ms": (time.perf_counter() - started) * 1000,
            **{f"retrieve_{k_}": v for k_, v in retriever.last_timings.items()},
        },
    }


def format_answer(result: dict[str, Any]) -> str:
    lines = [result["answer"], ""]
    for c in result["citations"]:
        rules = " (rules)" if "rules" in c["tags"] else ""
        lines.append(f'[{c["n"]}] {c["title"]}{rules} — {c["url"]}\n    "{c["quote"]}"')
    u = result["usage"]
    cost = result.get("cost_usd")
    lines.append(
        f"\nmodel={result['model']} tokens={u['prompt_tokens']}+{u['completion_tokens']} "
        f"cost=${cost:.5f} "
        if cost is not None
        else f"\nmodel={result['model']} "
    )
    lines[-1] += (
        f"retrieve={result['timings']['retrieve_ms']:.0f}ms llm={result['timings']['llm_ms']:.0f}ms "
        f"fixes={result['validation']['fixes']}"
    )
    return "\n".join(lines)
