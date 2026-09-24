"""Разбиение статьи на чанки (parent = статья, child = абзац или склейка коротких абзацев).

Правила:
- Статья без полного текста (google_news — только заголовок; fpl_api; RSS без trafilatura)
  становится ОДНИМ чанком: заголовок + тело, если тело добавляет информацию к заголовку.
- Полный текст режется по абзацам; короткие абзацы жадно склеиваются, пока чанк влезает
  в max_tokens; абзац длиннее лимита делится по предложениям. Хвост короче MIN_TAIL_TOKENS
  приклеивается к предыдущему чанку.
- Каждый чанк начинается со строки заголовка статьи — абзац «he trained fully on Thursday»
  без заголовка не найти ни эмбеддингом, ни BM25.
- Боилерплейт BBC/Guardian («- Published», «2 days ago») выбрасывается.

Чистые функции без БД и сети — покрыты unit-тестами. Подсчёт токенов — tiktoken cl100k_base
(llm.count_tokens); в тестах передаётся дешёвый счётчик.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from fplcopilot.rag.llm import count_tokens

MIN_TAIL_TOKENS = 40  # хвост короче — приклеиваем к предыдущему чанку
MIN_BODY_BUDGET = 80  # даже при длинном заголовке телу чанка оставляем не меньше
MIN_PARAGRAPH_WORDS = 3

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[A-Z\"'“(\[])")
_BOILERPLATE = re.compile(  # «- Published1 day ago», «Image source, Getty Images», заглушки видео BBC
    r"^[\W\d_]*(published|updated|image source|getty images|image caption|"
    r"this content is not available|there was an error|follow bbc|listen to|watch:|sign up)(?![a-z])",
    re.IGNORECASE,
)
_WORD = re.compile(r"[A-Za-z]{2,}")


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    n_tokens: int


TokenCounter = Callable[[str], int]


def normalize_ws(text: str | None) -> str:
    return " ".join((text or "").split())


def split_paragraphs(content: str) -> list[str]:
    """Абзацы по переносам строк; пустые и боилерплейт выбрасываем."""
    out: list[str] = []
    for raw in re.split(r"\n+", content):
        p = normalize_ws(raw)
        if not p or _BOILERPLATE.match(p) or len(_WORD.findall(p)) < MIN_PARAGRAPH_WORDS:
            continue
        out.append(p)
    return out


def split_long_paragraph(paragraph: str, max_tokens: int, count: TokenCounter) -> list[str]:
    """Абзац длиннее лимита -> куски по предложениям (предложение длиннее лимита -> по словам)."""
    sentences = [s for s in _SENTENCE_SPLIT.split(paragraph) if s]
    pieces: list[str] = []
    for s in sentences:
        if count(s) <= max_tokens:
            pieces.append(s)
            continue
        words = s.split()
        cur: list[str] = []
        for w in words:
            cur.append(w)
            if count(" ".join(cur)) > max_tokens and len(cur) > 1:
                pieces.append(" ".join(cur[:-1]))
                cur = [w]
        if cur:
            pieces.append(" ".join(cur))
    return _merge(pieces, max_tokens, count)


def _merge(pieces: list[str], budget: int, count: TokenCounter) -> list[str]:
    """Жадная склейка соседних кусков, пока сумма токенов <= budget; короткий хвост — к предыдущему."""
    groups: list[list[str]] = []
    cur: list[str] = []
    cur_tokens = 0
    for piece in pieces:
        t = count(piece)
        if cur and cur_tokens + t > budget:
            groups.append(cur)
            cur, cur_tokens = [piece], t
        else:
            cur.append(piece)
            cur_tokens += t
    if cur:
        if groups and cur_tokens < MIN_TAIL_TOKENS:
            groups[-1].extend(cur)
        else:
            groups.append(cur)
    return ["\n".join(g) for g in groups]


def _covers(a: str, b: str) -> bool:
    """b не добавляет информации к a (одно содержится в другом без учёта регистра)."""
    return b.lower() in a.lower()


def single_chunk_text(title: str, body: str | None) -> str:
    title, body = normalize_ws(title), normalize_ws(body)
    if not body or _covers(title, body):
        return title
    if _covers(body, title):
        return body
    return f"{title}\n{body}"


def chunk_article(
    *,
    title: str,
    summary: str | None,
    content: str | None,
    fulltext: bool,
    max_tokens: int = 300,
    count: TokenCounter = count_tokens,
) -> list[Chunk]:
    """Чанки статьи в порядке следования. Всегда >= 1 чанк (заголовок NOT NULL)."""
    title = normalize_ws(title)
    if not fulltext or not content:
        text = single_chunk_text(title, content or summary)
        return [Chunk(0, text, count(text))]

    paragraphs = [p for p in split_paragraphs(content) if p.lower() != title.lower()]
    if not paragraphs:
        text = single_chunk_text(title, summary)
        return [Chunk(0, text, count(text))]

    budget = max(MIN_BODY_BUDGET, max_tokens - count(title) - 1)
    pieces: list[str] = []
    for p in paragraphs:
        if count(p) > budget:
            pieces.extend(split_long_paragraph(p, budget, count))
        else:
            pieces.append(p)

    chunks: list[Chunk] = []
    for i, body in enumerate(_merge(pieces, budget, count)):
        text = f"{title}\n{body}"
        chunks.append(Chunk(i, text, count(text)))
    return chunks
