"""LLM-as-judge for faithfulness: is every claim in `summary` supported by the cited quotes?

gpt-4o-mini, temperature 0, structured output (JudgeVerdict). Verdict -> score {1, 0.5, 0}.
The judge sees only the summary, the official FPL status line the extractor was given, and the
verbatim evidence quotes — never the full documents — so it measures what a reader of the
signal could verify from the citations alone. Prompt: evals/prompts/faithfulness_judge.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from functools import lru_cache
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from evals import PROMPTS_DIR

JUDGE_MODEL = "gpt-4o-mini"
JUDGE_PROMPT_VERSION = "v1"
MAX_EVIDENCE_QUOTES = 6

Verdict = Literal["supported", "partially_supported", "unsupported"]
VERDICT_SCORE: dict[str, float] = {
    "supported": 1.0,
    "partially_supported": 0.5,
    "unsupported": 0.0,
}


class JudgeVerdict(BaseModel):
    """Structured output requested from the judge model."""

    verdict: Verdict
    reason: str = Field(description="one sentence, <= 200 characters")


@dataclass(frozen=True)
class JudgeResult:
    score: float | None  # None when the judge refused / returned nothing parseable
    verdict: str | None
    reason: str
    prompt_tokens: int
    completion_tokens: int
    model: str


class EvidenceLike(Protocol):
    source: str
    published_at: Any
    quote: str


@lru_cache(maxsize=1)
def load_judge_prompt() -> str:
    return (PROMPTS_DIR / "faithfulness_judge.md").read_text(encoding="utf-8").strip()


def format_fpl_status(status: str, chance_next: int | None, news: str) -> str:
    chance = "null" if chance_next is None else str(chance_next)
    return f'status={status}; chance_next={chance}; news="{news or ""}"'


def build_judge_messages(
    summary: str, evidence: Sequence[EvidenceLike], *, fpl_status: str
) -> list[dict[str, str]]:
    """System = judge prompt; user = data blocks only (no instructions inside the data)."""
    lines = []
    for i, ev in enumerate(evidence[:MAX_EVIDENCE_QUOTES], start=1):
        published = ev.published_at
        stamp = f"{published.astimezone(UTC):%Y-%m-%d}" if hasattr(published, "astimezone") else ""
        quote = " ".join(ev.quote.split())
        lines.append(f'{i}. [{ev.source} {stamp}] "{quote}"')
    evidence_block = "\n".join(lines) if lines else "(none)"
    user = (
        f"SUMMARY:\n{summary.strip()}\n\n"
        f"OFFICIAL FPL STATUS:\n{fpl_status}\n\n"
        f"EVIDENCE:\n{evidence_block}"
    )
    return [{"role": "system", "content": load_judge_prompt()}, {"role": "user", "content": user}]


def verdict_score(verdict: JudgeVerdict | str | None) -> float | None:
    """Map a verdict (model object or raw string) to {1, 0.5, 0}; None for refusals/unknown."""
    if verdict is None:
        return None
    name = verdict.verdict if isinstance(verdict, JudgeVerdict) else str(verdict).strip().lower()
    return VERDICT_SCORE.get(name)


def parse_judge_completion(completion: Any) -> tuple[JudgeVerdict | None, dict[str, int]]:
    """Extract (parsed verdict or None, token usage) from a chat.completions.parse() response."""
    usage = getattr(completion, "usage", None)
    tokens = {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }
    choices = getattr(completion, "choices", None) or []
    if not choices:
        return None, tokens
    msg = choices[0].message
    parsed = getattr(msg, "parsed", None)
    if parsed is None:
        return None, tokens
    if not isinstance(parsed, JudgeVerdict):
        parsed = JudgeVerdict.model_validate(parsed)
    return parsed, tokens


def judge_faithfulness(
    summary: str,
    evidence: Sequence[EvidenceLike],
    *,
    fpl_status: str,
    model: str = JUDGE_MODEL,
    client: Any | None = None,
) -> JudgeResult:
    """One judge call. `client` defaults to the project's OpenAI client (LangSmith-wrapped)."""
    if client is None:
        from fplcopilot.rag.llm import get_openai_client

        client = get_openai_client()
    messages = build_judge_messages(summary, evidence, fpl_status=fpl_status)
    completion = client.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=JudgeVerdict,
        temperature=0,
        max_completion_tokens=200,
    )
    verdict, tokens = parse_judge_completion(completion)
    return JudgeResult(
        score=verdict_score(verdict),
        verdict=verdict.verdict if verdict else None,
        reason=verdict.reason.strip() if verdict else "judge returned no structured verdict",
        prompt_tokens=tokens["prompt_tokens"],
        completion_tokens=tokens["completion_tokens"],
        model=model,
    )
