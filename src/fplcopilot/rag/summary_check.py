"""Детерминированная проверка свободного текста LLM (summary сигнала, summary / claim дайджеста
клуба) против того, на что он опирается.

Цитаты проверяет validate_draft (verbatim), а summary раньше не проверялся: на живом сигнале
Cherki (23.09) модель написала «… scoring in the last match against Sunderland (BBC, 2026-09-20)»,
хотя ни одна процитированная статья этого не содержит. Правило: каждое предложение summary может
называть только
  - даты (2026-09-20, «20 Sep», «September 20»), которые есть в поддержке: даты публикации
    процитированных документов и даты внутри их текста, FPL prior, календарь туров;
  - числа, которые встречаются в тексте поддержки;
  - собственные имена (слова с заглавной буквы: игроки, клубы, соперники, источники), которые есть в
    тексте поддержки, среди имён игрока / клуба / источников или являются обычными словами
    (частотный словарь rag/entity_matcher + короткий список слов предметной области).
Предложение с неподтверждённым — выбрасывается; если не осталось ни одного — нейтральный шаблон
из полей. Обычные слова не трогаются, поэтому правило не вырезает «He trained fully on Thursday».
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date

from fplcopilot.rag.entity_matcher import common_english_words, strip_accents

_MONTHS = {
    m: i
    for i, m in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
        start=1,
    )
}
_MON = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_ISO_DATE = re.compile(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})(?:[T ][0-9:]+Z?)?\b")
_DAY_MON = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_MON}(?:\s+20\d{{2}})?\b", re.IGNORECASE)
_MON_DAY = re.compile(
    rf"\b{_MON}\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+20\d{{2}})?\b", re.IGNORECASE
)
_DD_MM = re.compile(r"\b(\d{1,2})\.(\d{1,2})\b(?!\.\d)")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")
_CAP_WORD = re.compile(r"(?<![\w'’-])[A-ZÀ-ÖØ-Þ][\w'’-]*")
_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9«\"(])")
MONTH_WORDS = frozenset(
    [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
    ]
)
DOMAIN_WORDS = frozenset(
    [
        "fpl",
        "api",
        "gw",
        "gameweek",
        "gameweeks",
        "premier",
        "league",
        "status",
        "official",
        "news",
        "source",
        "sources",
        "available",
        "doubtful",
        "injured",
        "suspended",
        "unavailable",
        "fit",
        "unknown",
        "chance",
        "rotation",
        "manager",
        "boss",
        "head",
        "coach",
        "press",
        "conference",
        "training",
        "squad",
        "team",
        "club",
        "match",
        "matches",
        "game",
        "games",
        "fixture",
        "fixtures",
        "season",
        "international",
        "break",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]
)


def _fold(s: str) -> str:
    return strip_accents(s).lower()


@dataclass
class Support:
    tokens: set[str] = field(default_factory=set)
    numbers: set[str] = field(default_factory=set)
    dates: set[tuple[int, int]] = field(default_factory=set)  # (месяц, день)


def _dates_in(text: str) -> list[tuple[int, int, tuple[int, int]]]:
    """(start, end, (month, day)) для всех дат в тексте."""
    out: list[tuple[int, int, tuple[int, int]]] = []
    for m in _ISO_DATE.finditer(text):
        out.append((m.start(), m.end(), (int(m.group(2)), int(m.group(3)))))
    for m in _DAY_MON.finditer(text):
        out.append((m.start(), m.end(), (_MONTHS[m.group(2).lower()[:3]], int(m.group(1)))))
    for m in _MON_DAY.finditer(text):
        out.append((m.start(), m.end(), (_MONTHS[m.group(1).lower()[:3]], int(m.group(2)))))
    return out


def build_support(
    texts: Iterable[str], *, names: Iterable[str] = (), dates: Iterable[date] = ()
) -> Support:
    sup = Support()
    for t in [*texts, *names]:
        if not t:
            continue
        sup.tokens.update(_WORD.findall(_fold(t)))
        sup.numbers.update(m.group() for m in _NUMBER.finditer(t))
        sup.dates.update(md for _, _, md in _dates_in(t))
        for m in _DD_MM.finditer(t):
            sup.dates.add((int(m.group(2)), int(m.group(1))))
    for d in dates:
        sup.dates.add((d.month, d.day))
    return sup


def unsupported_in(sentence: str, sup: Support) -> list[str]:
    """Что в предложении не подтверждено поддержкой (пусто — предложение в порядке)."""
    problems: list[str] = []
    spans = _dates_in(sentence)
    for start, end, md in spans:
        if md not in sup.dates:
            problems.append(f"date '{sentence[start:end]}'")
    rest = sentence
    for start, end, _ in sorted(spans, reverse=True):
        rest = rest[:start] + " " + rest[end:]
    for m in _NUMBER.finditer(rest):
        tok = m.group()
        if tok not in sup.numbers and tok.rstrip("0").rstrip(".") not in sup.numbers:
            problems.append(f"number '{tok}'")
    common = common_english_words() | DOMAIN_WORDS | MONTH_WORDS
    for m in _CAP_WORD.finditer(rest):
        for part in re.split(r"[-'’]", m.group()):
            key = _fold(part)
            if len(key) < 3 or not key.isalpha():
                continue
            if key in sup.tokens or key in common:
                continue
            problems.append(f"name '{part}'")
    return list(dict.fromkeys(problems))


@dataclass
class SummaryCheck:
    summary: str
    dropped: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    replaced: bool = False

    @property
    def fixes(self) -> int:
        return len(self.dropped)


def check_summary(summary: str, sup: Support, *, fallback: str) -> SummaryCheck:
    """Выбросить предложения с неподтверждёнными датами / числами / именами; пусто -> fallback."""
    text = " ".join((summary or "").split())
    if not text:
        return SummaryCheck(summary=fallback, replaced=True)
    kept: list[str] = []
    res = SummaryCheck(summary=text)
    for sent in _SENTENCE.split(text):
        problems = unsupported_in(sent, sup)
        if problems:
            res.dropped.append(sent)
            res.reasons.append(f"{sent[:80]!r}: " + ", ".join(problems))
        else:
            kept.append(sent)
    if not res.dropped:
        return res
    if kept:
        res.summary = " ".join(kept)
    else:
        res.summary, res.replaced = fallback, True
    return res
