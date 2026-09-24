"""Чанкинг документов KB с учётом заголовков разделов.

Отличие от новостного чанкера (rag/chunking.py): документ KB — длинный гайд с разделами
(markdown-заголовки `#…` от trafilatura / reddit или строки-«шапки» `**Squad Size**`, как в
официальных правилах FPL). Абзац «It cannot be cancelled once confirmed» без раздела «Free Hit»
бесполезен и для эмбеддинга, и для BM25, поэтому каждый чанк начинается с заголовка документа
и пути разделов («Wildcard Strategy > When to Play Your Wildcard»). Списки и таблицы markdown
сохраняются целиком (пункт «- 2 Goalkeepers» новостной фильтр «< 3 слов» выбросил бы — здесь это
правило). Внутри раздела абзацы склеиваются тем же жадным алгоритмом (_merge), длинные режутся по
предложениям (split_long_paragraph); крошечные разделы (< MIN_TAIL_TOKENS) приклеиваются к
соседнему. Чистые функции без БД/сети — unit-тесты tests/test_kb_chunking.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from fplcopilot.rag.chunking import (
    _BOILERPLATE,
    MIN_BODY_BUDGET,
    MIN_TAIL_TOKENS,
    Chunk,
    TokenCounter,
    _merge,
    normalize_ws,
    split_long_paragraph,
    split_paragraphs,
)
from fplcopilot.rag.llm import count_tokens

_MD_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_BOLD_HEADING = re.compile(r"^\s*(?:\*\*|__)(.+?)(?:\*\*|__)\s*:?\s*$")
_UNDERLINE = re.compile(r"^\s*(={3,}|-{3,})\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+•]|\d{1,3}[.)])\s+\S")
_LIST_MARKER = re.compile(r"^\s*(?:[-*+•]|\d{1,3}[.)])\s+")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_SITE_SUFFIX_MIN_WORDS = 5  # «… - Best FPL Tips, Advice, … from Fantasy Football Scout» -> долой
MAX_HEADING_WORDS = 12
BOLD_HEADING_LEVEL = 3
HEADING_SEP = " > "


@dataclass
class Section:
    path: tuple[str, ...]  # иерархия заголовков (последний — ближайший)
    paragraphs: list[str] = field(default_factory=list)

    @property
    def heading(self) -> str:
        return HEADING_SEP.join(self.path)


def clean_title(title: str) -> str:
    """Срезаем хвосты сайтов: «… | FPLWatch | FPLWatch», «… - Best FPL Tips, … Fantasy Football Scout»."""
    t = normalize_ws(title).split(" | ")[0].strip()
    head, sep, tail = t.rpartition(" - ")
    if sep and len(tail.split()) >= _SITE_SUFFIX_MIN_WORDS and head:
        t = head.strip()
    return t


def parse_heading(line: str) -> tuple[int, str] | None:
    """`## Free Hit` -> (2, 'Free Hit'); `**Squad Size**` -> (3, 'Squad Size'); иначе None."""
    m = _MD_HEADING.match(line)
    if m:
        title = normalize_ws(m.group(2)).strip("*_ ")
        return (len(m.group(1)), title) if title else None
    m = _BOLD_HEADING.match(line)
    if m:
        title = normalize_ws(m.group(1)).rstrip(":").strip()
        if title and len(title.split()) <= MAX_HEADING_WORDS and not title.endswith("."):
            return BOLD_HEADING_LEVEL, title
    return None


def split_kb_paragraphs(block: str) -> list[str]:
    """Абзацы блока текста: обычные — через новостной split_paragraphs (боилерплейт, < 3 слов —
    долой), а подряд идущие пункты списка / строки таблицы — одним абзацем без фильтра длины."""
    out: list[str] = []
    plain: list[str] = []
    group: list[str] = []
    group_kind: str | None = None

    def flush_plain() -> None:
        if plain:
            out.extend(split_paragraphs("\n".join(plain)))
            plain.clear()

    def flush_group() -> None:
        nonlocal group_kind
        if group:
            out.append("\n".join(normalize_ws(g) for g in group))
            group.clear()
        group_kind = None

    for raw in block.splitlines():
        if not raw.strip():
            flush_group()
            plain.append("")
            continue
        kind = "list" if _LIST_ITEM.match(raw) else "table" if _TABLE_ROW.match(raw) else None
        if kind is None:
            flush_group()
            plain.append(raw)
            continue
        if kind == "table" and set(raw.replace("|", "").strip()) <= set("-: "):
            continue  # разделитель шапки таблицы |---|---|
        if kind == "list" and _BOILERPLATE.match(_LIST_MARKER.sub("", raw, count=1).strip()):
            continue  # «- Published2 days ago» в виде пункта списка
        flush_plain()
        if group_kind not in (None, kind):
            flush_group()
        group_kind = kind
        group.append(raw)
    flush_group()
    flush_plain()
    return out


def split_sections(text: str) -> list[Section]:
    """Текст с заголовками -> разделы с путём заголовков (глубина <= 2, H1 документа в путь не входит,
    если есть вложенные уровни). Заголовки-подчёркивания (Title\\n=====) тоже понимаются.
    Текст до первого заголовка — раздел с пустым путём; разделы без абзацев не создаются."""
    lines = (text or "").splitlines()
    stack: list[tuple[int, str]] = []
    sections: list[Section] = [Section(path=())]
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            sections[-1].paragraphs.extend(split_kb_paragraphs("\n".join(buffer)))
            buffer.clear()

    def open_section(level: int, title: str) -> None:
        nonlocal stack
        stack = [(lvl, t) for lvl, t in stack if lvl < level]
        stack.append((level, title))
        titles = [t for lvl, t in stack if lvl > 1] or [t for _, t in stack]
        path = tuple(titles[-2:])  # не глубже двух уровней: «Chips > Free Hit»
        if not sections[-1].paragraphs:
            sections.pop()  # заголовок за заголовком / пустой пролог — пустой раздел не нужен
        sections.append(Section(path=path))

    i = 0
    while i < len(lines):
        line = lines[i]
        parsed = parse_heading(line)
        if (
            parsed is None
            and line.strip()
            and i + 1 < len(lines)
            and _UNDERLINE.match(lines[i + 1])
            and not _TABLE_ROW.match(line)
        ):
            parsed = (1 if "=" in lines[i + 1] else 2, normalize_ws(line))
            i += 1
        if parsed is not None:
            flush()
            open_section(*parsed)
        else:
            buffer.append(line)
        i += 1
    flush()
    return [s for s in sections if s.paragraphs]


def chunk_document(
    *,
    title: str,
    text: str,
    max_tokens: int = 300,
    count: TokenCounter = count_tokens,
) -> list[Chunk]:
    """Чанки документа по разделам; каждый начинается с `title\\n<heading path>\\n`."""
    title = clean_title(title)
    sections = split_sections(text)
    if not sections:
        body = normalize_ws(text)
        chunk_text = f"{title}\n{body}" if body and body.lower() != title.lower() else title
        return [Chunk(0, chunk_text, count(chunk_text))]

    # крошечные разделы приклеиваем к соседнему (их заголовок остаётся в тексте абзаца)
    merged: list[Section] = []
    for s in sections:
        tokens = sum(count(p) for p in s.paragraphs)
        if tokens < MIN_TAIL_TOKENS and merged:
            inline = f"{s.path[-1]}: " if s.path else ""
            merged[-1].paragraphs.extend([inline + s.paragraphs[0], *s.paragraphs[1:]])
        else:
            merged.append(Section(path=s.path, paragraphs=list(s.paragraphs)))

    chunks: list[Chunk] = []
    for s in merged:
        heading = s.heading
        if heading and heading.lower() == title.lower():
            heading = ""
        prefix = f"{title}\n{heading}\n" if heading else f"{title}\n"
        budget = max(MIN_BODY_BUDGET, max_tokens - count(prefix.strip()) - 1)
        pieces: list[str] = []
        for p in s.paragraphs:
            if p.lower() == title.lower():
                continue
            pieces.extend(split_long_paragraph(p, budget, count) if count(p) > budget else [p])
        for body in _merge(pieces, budget, count):
            chunk_text = prefix + body
            chunks.append(Chunk(len(chunks), chunk_text, count(chunk_text)))
    if not chunks:
        return [Chunk(0, title, count(title))]
    return chunks
