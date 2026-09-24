"""Структурированный сигнал о доступности игрока: retrieval -> gpt-4o-mini (structured outputs)
-> детерминированная пост-валидация -> player_signals.

Конвейер extract_signal(player_id, as_of):
  1. запрос = детерминированное расширение по игроку (retrieve.expand_player_query);
  2. Retriever.search(..., as_of=as_of) — только документы, опубликованные до as_of;
  2a. abstention ДО LLM (v2, RAG_ABSTAIN): если среди кандидатов нет официального fpl_api-чанка
     игрока и (ни один выданный чанк не упоминает игрока ИЛИ max score ниже порога) —
     сигнал unknown/0 без вызова модели (abstained=True, llm_calls=0);
  3. промпт prompts/<version>/signal_extraction.*.md (v2 по умолчанию: календарь туров,
     правила про цитаты/чужие новости/форму, return_date вместо угадывания GW): документы как
     ДАННЫЕ в <document> тегах, статус FPL API как авторитетный prior;
  4. LLM отдаёт SignalDraft (v1) или SignalDraftV2 (pydantic, temperature 0);
  4a. v2 пост-обработка кодом: return_gw из даты («Expected back 18 Sep», «Suspended until
     17 Oct» — сначала официальный текст FPL, затем return_date модели) через календарь туров
     и фикстуры клуба; expected_minutes по формуле от availability/start_probability/позиции;
  5. validate_draft(): правила (b)-(e) из ТЗ проверяются кодом, а не доверяются модели —
     цитаты должны быть подстроками документов, пустые доказательства => unknown/0,
     injured/suspended => start_probability 0, статус i/s из FPL нельзя перебить «fit»;
     плюс (f): цитата засчитывается, только если её чанк упоминает целевого игрока
     (границы слов; «White» требует имени/инициала/клуба в том же тексте);
  6. PlayerSignal сохраняется в player_signals (evidence как jsonb).

LangSmith: клиент OpenAI обёрнут wrap_openai, стадии — @traceable; включается только ключом.
TODO(step 5+): TeamSignal (ротация/пресс-конференции на уровне клуба) — здесь не нужен.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal

from langsmith import traceable
from pydantic import BaseModel, Field
from sqlalchemy import text

from fplcopilot.config import settings
from fplcopilot.data import Bootstrap, Event, Fixture, FPLClient, Player, Team
from fplcopilot.db import session_scope
from fplcopilot.prompts import Prompt, load_prompt
from fplcopilot.rag.entity_matcher import (
    TEAM_ALIASES,
    EntityMatcher,
    strip_accents,
    text_mentions_player,
)
from fplcopilot.rag.llm import configure_tracing, get_openai_client
from fplcopilot.rag.quote_kind import quote_kind
from fplcopilot.rag.retrieve import (
    Mode,
    RetrievalConfig,
    RetrievedChunk,
    Retriever,
    expand_player_query,
)
from fplcopilot.rag.summary_check import SummaryCheck, build_support, check_summary

log = logging.getLogger(__name__)

PROMPT_NAME = "signal_extraction"
PROMPT_VERSION = settings.rag_prompt_version  # дефолт (RAG_PROMPT_VERSION); v1 доступен для A/B
MAX_QUOTE_CHARS = 200
ALIGN_MIN_RATIO = 0.85  # порог схожести для выравнивания почти-verbatim цитаты на документ
LIVE_PRIOR_WINDOW = timedelta(minutes=10)  # as_of «почти сейчас» -> статус из живого bootstrap
UNKNOWN_START_PROBABILITY = 0.5  # нейтральные значения при availability=unknown
UNKNOWN_EXPECTED_MINUTES = 45
CALENDAR_GWS_IN_PROMPT = 8  # сколько ближайших туров показываем модели
ABSTAIN_SUMMARY = "No player-specific news found."

# expected_minutes (v2) считает код, не модель:
#   E[min] = p_start · START_MINUTES[pos] + (1 − p_start) · BENCH_MINUTES[pos]
# START_MINUTES — средние минуты при выходе в старте по позициям, player_gw_history 2026/27 GW1–4
# (GKP 90.0, DEF 84.7, MID 78.4, FWD 79.4; n=80/345/374/81 стартов). BENCH_MINUTES — ожидание
# минут при невыходе в старте: P(выйдет на замену | не в старте) ≈ 0.5 для игрока основной
# обоймы × ~18 мин средней замены (вратари на замену не выходят). injured/suspended/unavailable → 0,
# unknown → нейтральные 45.
START_MINUTES = {"GKP": 90, "DEF": 85, "MID": 78, "FWD": 79}
BENCH_MINUTES = {"GKP": 0, "DEF": 9, "MID": 9, "FWD": 9}

Availability = Literal["fit", "doubtful", "injured", "suspended", "unavailable", "unknown"]
RotationRisk = Literal["low", "medium", "high", "unknown"]

STATUS_LABELS = {
    "a": "available",
    "d": "doubtful",
    "i": "injured",
    "s": "suspended",
    "u": "unavailable",
    "n": "not in squad",
}
# Статус FPL, который нельзя перебить «fit» (правило b): что ставим вместо fit.
HARD_STATUS_TO_AVAILABILITY: dict[str, Availability] = {
    "i": "injured",
    "s": "suspended",
    "u": "unavailable",
}
ZERO_MINUTES_AVAILABILITY = frozenset({"injured", "suspended", "unavailable"})


# ---------- схемы ----------


class EvidenceDraft(BaseModel):
    chunk_id: int = Field(description="id of the <document> the quote is copied from")
    quote: str = Field(description="verbatim substring of that document, <= 200 characters")


class SignalDraft(BaseModel):
    """Ровно то, что просим у LLM в промпте v1 (strict structured output). Остальное дописывает код.

    Также внутренняя «нормализованная» форма для validate_draft: v2-черновик приводится к ней
    кодом (expected_minutes по формуле, return_gw из даты).
    """

    availability: Availability
    start_probability: float = Field(description="0..1, probability of starting the next match")
    expected_minutes: int = Field(description="0..90 expected minutes in the next match")
    rotation_risk: RotationRisk
    return_gw: int | None = Field(description="gameweek of expected return, or null")
    confidence: float = Field(description="0..1 confidence in availability/start_probability")
    summary: str = Field(description="<= 2 sentences, plain English")
    evidence: list[EvidenceDraft]


class SignalDraftV2(BaseModel):
    """Промпт v2: без expected_minutes (считает код), с return_date (дата вместо номера тура)."""

    availability: Availability
    start_probability: float = Field(description="0..1, probability of starting the next match")
    rotation_risk: RotationRisk
    return_date: str | None = Field(
        description="ISO date (YYYY-MM-DD) the player is expected to be available again, or null"
    )
    return_gw: int | None = Field(
        description="gameweek number ONLY if a document literally names it, else null"
    )
    confidence: float = Field(description="0..1 confidence in availability/start_probability")
    summary: str = Field(description="<= 2 sentences, plain English")
    evidence: list[EvidenceDraft]


FormKind = Literal["form", "role", "position", "set_pieces"]
EvidenceAbout = Literal[
    "fitness", "injury", "training", "selection", "rotation", "suspension", "transfer"
]


class EvidenceDraftV4(BaseModel):
    """v4: категория раньше цитаты — модель сначала называет, о каком факте доступности цитата."""

    about: EvidenceAbout = Field(description="which availability fact the quote itself states")
    chunk_id: int = Field(description="id of the <document> the quote is copied from")
    quote: str = Field(description="verbatim substring of that document, <= 200 characters")


class FormNoteDraft(BaseModel):
    kind: FormKind
    text: str = Field(description="<= 1 short sentence, only facts stated in the quote")
    chunk_id: int = Field(description="id of the <document> the quote is copied from")
    quote: str = Field(
        description="verbatim substring of that document that names the player, <= 200 characters"
    )


class SignalDraftV4(BaseModel):
    """Промпт v4: evidence — только fitness / selection / suspension; форма, роль, позиция и
    стандарты — отдельным списком form_notes в конце (контекст, не доказательство доступности).

    Порядок полей = порядок генерации structured output, как у v2/v3 (вердикт -> evidence):
    черновик v4 с form_notes первым полем раскладывал в «форму» и травмы / баны / «night off»
    (11 из 14 заметок, A/B #4 docs/EVALS.md §4.6), и точность падала 25 -> 21 из 27.
    """

    availability: Availability
    start_probability: float = Field(description="0..1, probability of starting the next match")
    rotation_risk: RotationRisk
    return_date: str | None = Field(
        description="ISO date (YYYY-MM-DD) the player is expected to be available again, or null"
    )
    return_gw: int | None = Field(
        description="gameweek number ONLY if a document literally names it, else null"
    )
    confidence: float = Field(description="0..1 confidence in availability, from evidence only")
    summary: str = Field(description="<= 2 sentences about availability, plain English")
    evidence: list[EvidenceDraftV4] = Field(
        description="availability facts only (fitness, selection, suspension); [] if none"
    )
    form_notes: list[FormNoteDraft] = Field(
        description="form / role / position / set-piece context about the player; [] if none"
    )


DRAFT_SCHEMAS: dict[str, type[BaseModel]] = {
    "v1": SignalDraft,
    "v2": SignalDraftV2,
    "v3": SignalDraftV2,
    "v4": SignalDraftV4,
}
FORM_NOTES_VERSIONS = frozenset({"v4"})  # версии промпта, которые возвращают form_notes
MAX_FORM_NOTES = 3
# v4, правило 12: статус FPL a + вердикт fit только по косвенным цитатам (метки модели
# selection / rotation: «rested», «that was planned») — confidence не выше этого значения
INDIRECT_EVIDENCE = frozenset({"selection", "rotation"})
INDIRECT_FIT_MAX_CONFIDENCE = 0.8


def draft_schema(prompt_version: str) -> type[BaseModel]:
    """Схема structured output для версии промпта (неизвестная версия -> v2-схема)."""
    return DRAFT_SCHEMAS.get(prompt_version, SignalDraftV2)


class Evidence(BaseModel):
    chunk_id: int
    source: str
    url: str
    published_at: datetime
    quote: str


class FormNote(BaseModel):
    """Форма / роль / позиция / стандарты (v4). Контекст для пользователя и объяснителя: не
    доказательство доступности, модель минут / xPts его не читает."""

    kind: FormKind
    text: str
    quote: str
    chunk_id: int
    source: str
    url: str
    published_at: datetime


class PlayerSignal(BaseModel):
    player_id: int
    player_name: str
    as_of: datetime
    availability: Availability
    start_probability: float = Field(ge=0, le=1)
    expected_minutes: int = Field(ge=0, le=90)
    rotation_risk: RotationRisk
    return_gw: int | None = None
    confidence: float = Field(ge=0, le=1)
    summary: str
    evidence: list[Evidence]
    fpl_status: str
    fpl_chance_next: int | None = None
    model: str
    prompt_version: str
    retrieved_chunk_ids: list[int]
    mode: str
    validation_fixes: int = 0
    abstained: bool = False  # v2: сигнал выдан без вызова LLM (см. should_abstain)
    form_notes: list[FormNote] = Field(default_factory=list)  # v4; v1–v3 — []


@dataclass(frozen=True)
class FPLPrior:
    status: str
    chance_next: int | None
    news: str
    news_added: datetime | None

    @property
    def label(self) -> str:
        return STATUS_LABELS.get(self.status, self.status)


# ---------- prior из FPL API с учётом as_of ----------

_PLAYER_SNAPSHOTS = text(
    """
    SELECT status, chance_next, news, news_added, snapshot_at
    FROM player_status_snapshots
    WHERE player_id = :pid
    ORDER BY snapshot_at, id
    """
)


@dataclass(frozen=True)
class StatusSnapshot:
    status: str
    chance_next: int | None
    news: str
    news_added: datetime | None
    snapshot_at: datetime  # когда ингест увидел это состояние (время наблюдения)


def prior_from_snapshots(
    snapshots: Sequence[StatusSnapshot],
    as_of: datetime,
    *,
    known_before: datetime | None = None,
) -> FPLPrior:
    """Статус FPL на as_of по снимкам игрока (в порядке snapshot_at).

    Время события — snapshot_at, а не news_added: FPL не обновляет news_added, когда меняет
    chance/текст или снимает новость (Caicedo: «Expected back 18 Sep» -> «50% chance» с тем же
    news_added 30.08; Doku: i -> a с news_added 18.08) — фильтр по news_added пропускал
    снимки, сделанные после as_of. Если до as_of снимков нет (ингест ещё не работал) —
    приближение по первому наблюдению, если его новость опубликована до as_of; иначе «доступен
    без новостей». known_before (evals, операционный replay) — учитывать только снимки,
    сделанные до этого момента.
    """
    seen = [s for s in snapshots if known_before is None or s.snapshot_at <= known_before]
    before = [s for s in seen if s.snapshot_at <= as_of]
    if before:
        s = before[-1]
    elif seen and seen[0].news_added is not None and seen[0].news_added <= as_of:
        s = seen[0]
    else:
        return FPLPrior("a", None, "", None)
    added = s.news_added.astimezone(UTC) if s.news_added else None
    return FPLPrior(s.status, s.chance_next, s.news, added)


def fpl_prior(
    player: Player,
    as_of: datetime,
    *,
    now: datetime | None = None,
    known_before: datetime | None = None,
) -> FPLPrior:
    """Статус FPL на момент as_of: живой bootstrap для «сейчас», иначе prior_from_snapshots."""
    now = now or datetime.now(UTC)
    if as_of >= now - LIVE_PRIOR_WINDOW and known_before is None:
        return FPLPrior(
            player.status, player.chance_of_playing_next_round, player.news, player.news_added
        )
    with session_scope() as s:
        rows = s.execute(_PLAYER_SNAPSHOTS, {"pid": player.id}).all()
    snapshots = [
        StatusSnapshot(status, chance, news or "", added, snap_at.astimezone(UTC))
        for status, chance, news, added, snap_at in rows
    ]
    return prior_from_snapshots(snapshots, as_of, known_before=known_before)


def next_event_as_of(bs: Bootstrap, as_of: datetime) -> Event | None:
    """Ближайший тур, дедлайн которого ещё не прошёл на момент as_of."""
    upcoming = [e for e in bs.events if e.deadline_time > as_of]
    return min(upcoming, key=lambda e: e.deadline_time) if upcoming else None


# ---------- календарь туров: дата возврата -> return_gw (v2) ----------


@dataclass(frozen=True)
class GWEntry:
    gw: int
    deadline: datetime
    first_kickoff: datetime | None
    last_kickoff: datetime | None


@dataclass(frozen=True)
class GWCalendar:
    """Туры с дедлайнами и окнами матчей + фикстуры клубов (team_id -> [(kickoff, gw)]).

    Правило (совпадает с протоколом golden, README rule 6, и с тем, как FPL пишет даты):
    return_gw = тур, в котором у КЛУБА игрока первый матч с датой >= даты возврата; без фикстур
    клуба — первый тур, окно матчей которого не закончилось до этой даты. Проверено на всех 8
    return_date-строках golden: «Expected back 18 Sep» (Caicedo) = BRE–CHE 18.09 (GW5);
    «Suspended until 19 Oct» (Awoniyi) = TOT–COV 19.10 (GW7); «Suspended until 17 Oct» (Foden) =
    MCI–IPS 17.10 (GW7) — «until» у FPL включительно: это дата первого матча, где игрок доступен.
    Для формулировок прессы вида «banned until <день до возврата>» есть exclusive=True (+1 день).
    """

    entries: tuple[GWEntry, ...]
    team_fixtures: dict[int, tuple[tuple[datetime, int], ...]]

    @classmethod
    def from_fpl(cls, bs: Bootstrap, fixtures: Sequence[Fixture] = ()) -> GWCalendar:
        kickoffs: dict[int, list[datetime]] = {}
        by_team: dict[int, list[tuple[datetime, int]]] = {}
        for f in fixtures:
            if f.event is None or f.kickoff_time is None:
                continue
            ko = f.kickoff_time.astimezone(UTC)
            kickoffs.setdefault(f.event, []).append(ko)
            for team_id in (f.team_h, f.team_a):
                by_team.setdefault(team_id, []).append((ko, f.event))
        entries = tuple(
            GWEntry(
                gw=e.id,
                deadline=e.deadline_time.astimezone(UTC),
                first_kickoff=min(kickoffs[e.id]) if e.id in kickoffs else None,
                last_kickoff=max(kickoffs[e.id]) if e.id in kickoffs else None,
            )
            for e in sorted(bs.events, key=lambda e: e.deadline_time)
        )
        return cls(
            entries=entries,
            team_fixtures={t: tuple(sorted(v)) for t, v in by_team.items()},
        )

    def upcoming(self, as_of: datetime, n: int = CALENDAR_GWS_IN_PROMPT) -> list[GWEntry]:
        return [e for e in self.entries if e.deadline > as_of][:n]

    def gw_for_date(
        self, when: date, *, team_id: int | None = None, exclusive: bool = False
    ) -> int | None:
        """Первый тур, в котором игрок может сыграть, если доступен с даты `when`.

        exclusive=True — «until <дата>» в смысле «доступен со следующего дня».
        """
        if exclusive:
            when = when + timedelta(days=1)
        for ko, gw in self.team_fixtures.get(team_id or -1, ()):
            if ko.date() >= when:
                return gw
        # нет фикстур клуба (или они кончились раньше даты) — по окнам туров
        for e in self.entries:
            end = e.last_kickoff or e.deadline
            if end.date() >= when:
                return e.gw
        return None

    def render(self, as_of: datetime, *, team: Team | None = None, n: int | None = None) -> str:
        """Блок календаря для промпта: дедлайн, окно матчей, дата матча клуба."""
        lines = []
        for e in self.upcoming(as_of, n or CALENDAR_GWS_IN_PROMPT):
            line = f"GW{e.gw}: deadline {e.deadline:%Y-%m-%d %H:%M}Z"
            if e.first_kickoff and e.last_kickoff:
                line += f"; matches {e.first_kickoff:%Y-%m-%d} to {e.last_kickoff:%Y-%m-%d}"
            if team is not None:
                own = [ko for ko, gw in self.team_fixtures.get(team.id, ()) if gw == e.gw]
                if own:
                    line += f"; {team.name} plays " + ", ".join(f"{ko:%Y-%m-%d}" for ko in own)
                else:
                    line += f"; {team.name}: no fixture (blank)"
            lines.append(line)
        return "\n".join(lines) if lines else "(no upcoming gameweeks)"


_MONTHS = {
    m: i
    for i, m in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
        start=1,
    )
}
_FPL_RETURN = re.compile(
    r"\b(?P<kind>Expected back|Suspended until|Unavailable until|Out until|until)\s+"
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+(?P<mon>[A-Za-z]{3,9})\.?(?:\s+(?P<year>20\d{2}))?",
    re.IGNORECASE,
)
_DAY_MONTH = re.compile(
    r"^(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+(?P<mon>[A-Za-z]{3,9})\.?(?:\s+(?P<year>20\d{2}))?$"
)


def _infer_year(day: int, month: int, as_of: datetime, year: int | None) -> date | None:
    """Дата без года -> ближайшая по календарю к as_of вперёд (декабрь->январь переносится)."""
    try:
        if year is not None:
            return date(year, month, day)
        candidate = date(as_of.year, month, day)
    except ValueError:
        return None
    if candidate < as_of.date() - timedelta(days=180):
        return date(as_of.year + 1, month, day)
    return candidate


def parse_fpl_return_date(news: str, as_of: datetime) -> tuple[date, str] | None:
    """«Calf injury - Expected back 18 Sep» -> (2026-09-18, 'expected_back');
    «Suspended until 17 Oct» -> (2026-10-17, 'suspended_until'). None — даты нет
    («Unknown return date», «75% chance of playing»)."""
    m = _FPL_RETURN.search(news or "")
    if not m:
        return None
    month = _MONTHS.get(m.group("mon")[:3].lower())
    if month is None:
        return None
    year = int(m.group("year")) if m.group("year") else None
    when = _infer_year(int(m.group("day")), month, as_of, year)
    if when is None:
        return None
    kind = m.group("kind").lower().replace(" ", "_")
    if kind == "until":
        kind = "suspended_until"
    return when, kind


def parse_return_date(value: str | None, as_of: datetime) -> date | None:
    """return_date модели: ISO «2026-09-18» или «18 Sep»/«18 September 2026»; иначе None."""
    if not value:
        return None
    s = value.strip()
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        pass
    m = _DAY_MONTH.match(s)
    if m and (month := _MONTHS.get(m.group("mon")[:3].lower())):
        year = int(m.group("year")) if m.group("year") else None
        return _infer_year(int(m.group("day")), month, as_of, year)
    parsed = parse_fpl_return_date(s, as_of)
    return parsed[0] if parsed else None


def expected_minutes_for(availability: str, start_probability: float, position: str) -> int:
    """Детерминированная формула (см. START_MINUTES/BENCH_MINUTES выше)."""
    if availability in ZERO_MINUTES_AVAILABILITY:
        return 0
    if availability == "unknown":
        return UNKNOWN_EXPECTED_MINUTES
    p = min(1.0, max(0.0, float(start_probability)))
    start = START_MINUTES.get(position, START_MINUTES["MID"])
    bench = BENCH_MINUTES.get(position, BENCH_MINUTES["MID"])
    return round(p * start + (1.0 - p) * bench)


# ---------- промпт ----------


def render_documents(chunks: Sequence[RetrievedChunk]) -> str:
    parts = []
    for c in chunks:
        body = c.text.replace("</document", "</document ").replace("<document", "<document ")
        parts.append(
            f'<document id="{c.chunk_id}" source="{c.source}" '
            f'published_at="{c.published_at.astimezone(UTC):%Y-%m-%dT%H:%MZ}">\n{body}\n</document>'
        )
    return "\n\n".join(parts) if parts else "(no documents retrieved)"


def build_messages(
    prompt: Prompt,
    *,
    player: Player,
    team: Team,
    prior: FPLPrior,
    chunks: Sequence[RetrievedChunk],
    as_of: datetime,
    next_event: Event | None,
    calendar: GWCalendar | None = None,
) -> list[dict[str, str]]:
    user = prompt.render_user(
        player_name=player.full_name,
        player_id=player.id,
        team_name=team.name,
        position=player.position.short,
        as_of=f"{as_of.astimezone(UTC):%Y-%m-%dT%H:%MZ}",
        next_gw=next_event.id if next_event else "?",
        deadline=f"{next_event.deadline_time:%Y-%m-%d %H:%MZ}" if next_event else "unknown",
        fpl_status=prior.status,
        fpl_status_label=prior.label,
        fpl_chance_next=prior.chance_next if prior.chance_next is not None else "null",
        fpl_news=prior.news or "",
        fpl_news_added=f"{prior.news_added:%Y-%m-%dT%H:%MZ}" if prior.news_added else "null",
        n_docs=len(chunks),
        documents=render_documents(chunks),
        # v2: блок календаря; v1-шаблон плейсхолдера не имеет и kwarg игнорирует
        gw_calendar=(
            calendar.render(as_of, team=team) if calendar is not None else "(calendar unavailable)"
        ),
    )
    return [{"role": "system", "content": prompt.system}, {"role": "user", "content": user}]


# ---------- пост-валидация (детерминированная) ----------


def _norm_ws(s: str) -> str:
    return " ".join(s.split())


def quote_in_text(quote: str, doc_text: str) -> bool:
    """Точная подстрока; допускаем лишь расхождения в пробелах/переносах."""
    q = quote.strip()
    if not q:
        return False
    return q in doc_text or _norm_ws(q).lower() in _norm_ws(doc_text).lower()


def verbatim_quote(quote: str, doc_text: str) -> str | None:
    """Цитата в том виде, в котором она реально есть в документе, или None.

    Модель любит добавлять точку к заголовку без точки — такую цитату принимаем без точки.
    """
    q = quote.strip()
    if quote_in_text(q, doc_text):
        return q
    stripped = q.rstrip(".!?…").rstrip()
    if stripped and quote_in_text(stripped, doc_text):
        return stripped
    return None


_SENTENCES = re.compile(r"\n+|(?<=[.!?])\s+")


def align_quote(quote: str, doc_text: str, *, min_ratio: float = ALIGN_MIN_RATIO) -> str | None:
    """Почти-verbatim цитату (модель убрала «(£7.8m)», поправила опечатку) заменяем на
    реальный фрагмент документа: лучшая по difflib.ratio цепочка из 1-3 соседних предложений.
    None — если похожего фрагмента нет; тогда цитата отбрасывается."""
    q = _norm_ws(quote)
    if not q:
        return None
    sentences = [s for s in _SENTENCES.split(doc_text) if s.strip()]
    best, best_ratio = None, 0.0
    for i in range(len(sentences)):
        for j in range(i, min(i + 3, len(sentences))):
            span = " ".join(sentences[i : j + 1]).strip()
            if len(span) > 3 * len(q) + 40:
                break
            ratio = difflib.SequenceMatcher(None, span.lower(), q.lower(), autojunk=False).ratio()
            if ratio > best_ratio:
                best, best_ratio = span, ratio
    if best is None or best_ratio < min_ratio:
        return None
    return best[:MAX_QUOTE_CHARS] if len(best) > MAX_QUOTE_CHARS else best


def get_matcher(bs: Bootstrap) -> EntityMatcher:
    """EntityMatcher по bootstrap, кэш на объекте (та же логика, что тегирует статьи при ингесте)."""
    matcher = bs.__dict__.get("_entity_matcher")
    if matcher is None:
        matcher = EntityMatcher.from_bootstrap(bs)
        bs.__dict__["_entity_matcher"] = matcher
    return matcher


def chunk_mentions_player(
    chunk: RetrievedChunk,
    player: Player,
    team: Team | None = None,
    matcher: EntityMatcher | None = None,
) -> bool:
    """(f) доказательство должно быть про целевого игрока: тег сущности или имя в тексте.

    Имя ищется по границам слов: с EntityMatcher (тот же, что тегирует статьи: неоднозначные
    фамилии, «Enzo Maresca» ≠ Enzo, обычные слова требуют контекста) или, без bootstrap,
    облегчённой text_mentions_player. «Egan» не находится в «Keegan»; «White» требует имени,
    инициала или клуба в том же чанке.
    """
    if player.id in chunk.players:
        return True
    if matcher is not None:
        return player.id in matcher.match(chunk.text).players
    return text_mentions_player(chunk.text, player, team)


def validate_draft(
    draft: SignalDraft,
    chunks: Sequence[RetrievedChunk],
    prior: FPLPrior,
    *,
    next_gw: int | None = None,
    player: Player | None = None,
    team: Team | None = None,
    matcher: EntityMatcher | None = None,
) -> tuple[SignalDraft, list[Evidence], int]:
    """Правила (b)-(f) кодом. Возвращает (исправленный draft, evidence, число исправлений)."""
    fixes = 0
    by_id = {c.chunk_id: c for c in chunks}
    data = draft.model_dump()

    # (d) цитаты — только из выданных документов и только verbatim: точная подстрока,
    # иначе выравниваем на реальный фрагмент (align_quote, считается исправлением), иначе долой.
    # (f) чанк должен быть про целевого игрока — общие «Spurs injury boost» не доказательство.
    evidence: list[Evidence] = []
    seen: set[tuple[int, str]] = set()
    for ev in draft.evidence:
        chunk = by_id.get(ev.chunk_id)
        if chunk is None or (player is not None and not chunk_mentions_player(chunk, player, team)):
            fixes += 1
            log.debug("evidence dropped (chunk %s): not about the player", ev.chunk_id)
            continue
        quote = verbatim_quote(ev.quote[:MAX_QUOTE_CHARS], chunk.text)
        if quote is None:
            aligned = align_quote(ev.quote, chunk.text)
            fixes += 1
            if aligned is None:
                log.debug("quote dropped (chunk %d): %r", ev.chunk_id, ev.quote)
                continue
            log.debug("quote aligned (chunk %d): %r -> %r", ev.chunk_id, ev.quote, aligned)
            quote = aligned
        key = (ev.chunk_id, _norm_ws(quote).lower())
        if key in seen:
            continue
        seen.add(key)
        evidence.append(
            Evidence(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                url=chunk.url,
                published_at=chunk.published_at,
                quote=quote,
            )
        )

    # (c) нет доказательств => unknown / 0 / нейтральные значения; summary модели тоже нельзя
    # оставлять — она пишет «is available» без единого документа
    if not evidence:
        who = player.full_name if player is not None else "the player"
        forced = {
            "availability": "unknown",
            "confidence": 0.0,
            "start_probability": UNKNOWN_START_PROBABILITY,
            "expected_minutes": UNKNOWN_EXPECTED_MINUTES,
            "rotation_risk": "unknown",
            "return_gw": None,
            "summary": (
                f"No evidence about {who} in the retrieved documents; "
                f"FPL API status {prior.status} ({prior.label}) is the only signal."
            ),
        }
        for key, value in forced.items():
            if data[key] != value:
                data[key] = value
                fixes += 1

    # (b) статус FPL i/s/u нельзя перебить «fit»
    hard = HARD_STATUS_TO_AVAILABILITY.get(prior.status)
    if hard and data["availability"] == "fit":
        data["availability"] = hard
        fixes += 1

    # (e) согласованность чисел с availability + клампы
    if data["availability"] in ZERO_MINUTES_AVAILABILITY:
        for key in ("start_probability", "expected_minutes"):
            if data[key] != 0:
                data[key] = 0
                fixes += 1
    for key, lo, hi in (("start_probability", 0.0, 1.0), ("confidence", 0.0, 1.0)):
        clamped = min(hi, max(lo, float(data[key])))
        if clamped != data[key]:
            data[key] = clamped
            fixes += 1
    clamped_min = min(90, max(0, int(data["expected_minutes"])))
    if clamped_min != data["expected_minutes"]:
        data["expected_minutes"] = clamped_min
        fixes += 1
    if data["return_gw"] is not None and (
        data["availability"] == "fit" or (next_gw is not None and data["return_gw"] < next_gw)
    ):
        data["return_gw"] = None
        fixes += 1

    data["evidence"] = [{"chunk_id": e.chunk_id, "quote": e.quote} for e in evidence]
    return SignalDraft.model_validate(data), evidence, fixes


def _quote_key(quote: str) -> str:
    return _norm_ws(quote).lower().rstrip(".!?… ")


def split_form_quotes(
    draft: SignalDraft, notes: Sequence[FormNoteDraft]
) -> tuple[SignalDraft, list[FormNoteDraft], int]:
    """v4: цитата evidence со словами о форме и без единого слова о доступности
    (rag/quote_kind: «Rayan Cherki scored», «back to his dangerous best») — не доказательство
    доступности: переносится в form_notes (kind form), без доказательств дальше действует (c).
    Возвращает (draft, заметки модели + перенесённые, число перенесённых)."""
    kept: list[EvidenceDraft] = []
    moved: list[FormNoteDraft] = []
    for e in draft.evidence:
        if quote_kind(e.quote) == "form":
            moved.append(
                FormNoteDraft(kind="form", text=e.quote, chunk_id=e.chunk_id, quote=e.quote)
            )
        else:
            kept.append(e)
    if not moved:
        return draft, list(notes), 0
    return draft.model_copy(update={"evidence": kept}), [*notes, *moved], len(moved)


def cap_indirect_fit_confidence(
    draft: SignalDraft,
    evidence: Sequence[Evidence],
    raw_evidence: Sequence[BaseModel],
    prior: FPLPrior,
) -> tuple[SignalDraft, int]:
    """v4, правило 12 кодом: статус FPL a, вердикт fit, а все оставшиеся цитаты — косвенные
    (метка модели selection / rotation, не fpl_api) -> confidence не выше 0.8."""
    if prior.status != "a" or draft.availability != "fit" or not evidence:
        return draft, 0
    about = {int(getattr(e, "chunk_id", -1)): getattr(e, "about", None) for e in raw_evidence}
    labels = {about.get(e.chunk_id) for e in evidence}
    if any(e.source == "fpl_api" for e in evidence) or not labels <= INDIRECT_EVIDENCE:
        return draft, 0
    if draft.confidence <= INDIRECT_FIT_MAX_CONFIDENCE:
        return draft, 0
    return draft.model_copy(update={"confidence": INDIRECT_FIT_MAX_CONFIDENCE}), 1


def quote_mentions_player(
    quote: str, player: Player, team: Team | None = None, matcher: EntityMatcher | None = None
) -> bool:
    """Сама цитата называет игрока (те же правила границ слов / обычных фамилий, что и (f))."""
    if matcher is not None:
        return player.id in matcher.match(quote).players
    return text_mentions_player(quote, player, team)


_CONTENT_WORD = re.compile(r"[a-z]{4,}")
NOTE_TEXT_MIN_OVERLAP = 0.6  # доля содержательных слов текста заметки, которые есть в цитате
_NOTE_STOPWORDS = frozenset(
    {
        "have", "has", "been", "being", "with", "that", "this", "from", "into", "their", "they",
        "them", "will", "would", "could", "should", "after", "over", "also", "about", "than",
        "then", "when", "were", "what", "which", "while", "very", "more", "most",
    }
)  # fmt: skip


def _content_words(s: str) -> set[str]:
    return set(_CONTENT_WORD.findall(strip_accents(s).lower())) - _NOTE_STOPWORDS


def _form_note_text(
    text: str, quote: str, chunk: RetrievedChunk, player: Player, team: Team | None
) -> SummaryCheck:
    """Текст заметки: даты / числа / имена — только из её цитаты (summary_check), и сам текст —
    пересказ этой цитаты (>= 60 % содержательных слов из неё: «scoring in the first four
    matches» к цитате про прогноз очков не проходит); иначе — сама цитата."""
    names = [player.full_name, player.web_name, chunk.source.replace("_", " ")]
    if team is not None:
        names += [team.name, team.short_name, *TEAM_ALIASES.get(team.short_name.upper(), ())]
    published = chunk.published_at.astimezone(UTC).date()
    sup = build_support([quote], names=names, dates=[published])
    checked = check_summary(text, sup, fallback=quote)
    words = _content_words(checked.summary)
    if words and len(words & _content_words(quote)) / len(words) < NOTE_TEXT_MIN_OVERLAP:
        checked.dropped.append(checked.summary)
        checked.reasons.append(f"{checked.summary[:80]!r}: not a restatement of its quote")
        checked.summary, checked.replaced = quote, True
    return checked


def validate_form_notes(
    notes: Sequence[FormNoteDraft],
    chunks: Sequence[RetrievedChunk],
    *,
    player: Player,
    team: Team | None = None,
    matcher: EntityMatcher | None = None,
    evidence: Sequence[Evidence] = (),
) -> tuple[list[FormNote], int]:
    """form_notes (v4) кодом: (d) verbatim / выравнивание, как у evidence; (f) чанк про игрока
    И сама цитата называет его — в сводках FFScout чанк упоминает 20 игроков, «X scored» про
    одноклубника не должно стать заметкой о форме цели. Текст заметки проверяет summary_check
    по её цитате. Цитата, уже взятая в evidence, не дублируется (без штрафа); заметка со словами
    о травме / бане / выборе состава (rag/quote_kind) — неверно разложенная, отбрасывается.
    Возвращает (заметки, число исправлений)."""
    by_id = {c.chunk_id: c for c in chunks}
    taken = {(e.chunk_id, _quote_key(e.quote)) for e in evidence}
    out: list[FormNote] = []
    fixes = 0
    for note in notes:
        if len(out) >= MAX_FORM_NOTES:
            break
        chunk = by_id.get(note.chunk_id)
        if chunk is None or not chunk_mentions_player(chunk, player, team, matcher):
            fixes += 1
            continue
        quote = verbatim_quote(note.quote[:MAX_QUOTE_CHARS], chunk.text)
        if quote is None:
            fixes += 1
            quote = align_quote(note.quote, chunk.text)
            if quote is None:
                continue
        if not quote_mentions_player(quote, player, team, matcher):
            fixes += 1
            continue
        key = (chunk.chunk_id, _quote_key(quote))
        if key in taken:
            continue
        if quote_kind(quote) == "availability":
            fixes += 1
            continue
        taken.add(key)
        checked = _form_note_text(note.text.strip() or quote, quote, chunk, player, team)
        fixes += checked.fixes
        out.append(
            FormNote(
                kind=note.kind,
                text=checked.summary,
                quote=quote,
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                url=chunk.url,
                published_at=chunk.published_at,
            )
        )
    return out, fixes


def no_availability_news_summary(player: Player, prior: FPLPrior) -> str:
    """(c) при v4, когда новости об игроке есть, но только о форме / роли."""
    return (
        f"No fitness or selection news about {player.full_name} in the retrieved documents; "
        f"FPL API status {prior.status} ({prior.label}) is the only availability signal. "
        "Form and role notes are listed separately."
    )


# ---------- LLM ----------


def _unknown_draft(reason: str) -> SignalDraft:
    return SignalDraft(
        availability="unknown",
        start_probability=UNKNOWN_START_PROBABILITY,
        expected_minutes=UNKNOWN_EXPECTED_MINUTES,
        rotation_risk="unknown",
        return_gw=None,
        confidence=0.0,
        summary=reason,
        evidence=[],
    )


@traceable(name="signal_llm", run_type="chain")
def call_llm(
    messages: list[dict[str, str]],
    *,
    model: str,
    schema: type[BaseModel] = SignalDraft,
    temperature: float = 0.0,
    top_p: float | None = None,
    max_completion_tokens: int = 800,
) -> tuple[BaseModel, dict[str, Any]]:
    """Structured output по схеме (SignalDraft для v1, SignalDraftV2 для v2); отказ -> unknown.

    Сам вызов OpenAI трейсится wrap_openai (дочерний run), здесь — обёртка с токенами.
    temperature / top_p / max_completion_tokens — ручки для evals (docs/EVALS.md «Hyperparameters»);
    продакшен-дефолты: T = 0, top_p не передаётся (= 1.0 у API), 800 токенов.
    """
    client = get_openai_client()
    sampling: dict[str, Any] = {"temperature": temperature}
    if top_p is not None:
        sampling["top_p"] = top_p
    completion = client.chat.completions.parse(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        response_format=schema,
        max_completion_tokens=max_completion_tokens,
        **sampling,
    )
    msg = completion.choices[0].message
    usage = completion.usage
    tokens: dict[str, Any] = {
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "finish_reason": completion.choices[0].finish_reason,
    }
    if msg.parsed is None:
        log.warning("LLM refusal/empty parse: %s", msg.refusal)
        return _unknown_draft("model returned no structured answer"), tokens
    return msg.parsed, tokens


# ---------- v2: abstention до LLM и нормализация черновика ----------


@dataclass(frozen=True)
class AbstainDecision:
    abstain: bool
    reason: str
    has_official: bool
    any_mention: bool
    top_score: float | None


def should_abstain(
    chunks: Sequence[RetrievedChunk],
    candidates: Sequence[RetrievedChunk],
    player: Player,
    team: Team | None,
    *,
    mode: str,
    rerank_threshold: float | None = None,
    dense_threshold: float | None = None,
    matcher: EntityMatcher | None = None,
) -> AbstainDecision:
    """Отказ до LLM: нет официального fpl_api-чанка игрока среди кандидатов И
    (ни один выданный чанк не упоминает игрока ИЛИ максимальный score кандидатов ниже порога).

    Пороги score (0 = выключен; дефолт после A/B #2): hybrid_rerank — «сырой» rerank_score
    (RAG_ABSTAIN_SCORE; 0.2 из A/B #1 на запросе extract_signal дал 4 ложных отказа — Raya, Gabriel,
    Emersonn, Vicario: новости про игрока есть, но не про травму, cross-encoder ставит 0.01–0.16);
    dense — косинус (RAG_ABSTAIN_DENSE_SCORE; на golden косинус классы не разделяет: no_coverage
    0.50–0.59, покрытые игроки без официального чанка 0.42–0.61). Работает триггер «нет упоминаний»:
    4/4 no_coverage, 0 ложных. Он эквивалентен исходу валидации: без чанка про игрока правило (f)
    всё равно обнулило бы evidence, а вызов LLM был бы потрачен зря.
    """
    rerank_threshold = settings.rag_abstain_score if rerank_threshold is None else rerank_threshold
    dense_threshold = (
        settings.rag_abstain_dense_score if dense_threshold is None else dense_threshold
    )
    has_official = any(c.source == "fpl_api" and player.id in c.players for c in candidates)
    any_mention = any(chunk_mentions_player(c, player, team, matcher) for c in chunks)
    if mode == "hybrid_rerank":
        scores = [c.rerank_score for c in candidates if c.rerank_score is not None]
        threshold: float | None = rerank_threshold
    elif mode == "dense":
        scores = [c.dense_score for c in candidates if c.dense_score is not None]
        threshold = dense_threshold
    else:
        scores, threshold = [], None
    top = max(scores) if scores else None
    if has_official:
        return AbstainDecision(
            False, "official fpl_api item among candidates", True, any_mention, top
        )
    if not any_mention:
        return AbstainDecision(True, "no retrieved chunk mentions the player", False, False, top)
    if threshold is not None and threshold > 0 and top is not None and top < threshold:
        return AbstainDecision(
            True, f"top {mode} score {top:.3f} < {threshold}", False, any_mention, top
        )
    return AbstainDecision(False, "player-specific candidates present", False, any_mention, top)


def abstained_signal(
    player: Player,
    prior: FPLPrior,
    chunks: Sequence[RetrievedChunk],
    *,
    as_of: datetime,
    mode: str,
    prompt_version: str,
    model: str | None = None,
) -> PlayerSignal:
    return PlayerSignal(
        player_id=player.id,
        player_name=player.full_name,
        as_of=as_of,
        availability="unknown",
        start_probability=UNKNOWN_START_PROBABILITY,
        expected_minutes=UNKNOWN_EXPECTED_MINUTES,
        rotation_risk="unknown",
        return_gw=None,
        confidence=0.0,
        summary=f"{ABSTAIN_SUMMARY} FPL API status {prior.status} ({prior.label}) is the only signal.",
        evidence=[],
        fpl_status=prior.status,
        fpl_chance_next=prior.chance_next,
        model=model or settings.rag_llm_model,
        prompt_version=prompt_version,
        retrieved_chunk_ids=[c.chunk_id for c in chunks],
        mode=mode,
        validation_fixes=0,
        abstained=True,
    )


def resolve_return_gw(
    *,
    prior: FPLPrior,
    llm_return_gw: int | None,
    llm_return_date: str | None,
    calendar: GWCalendar | None,
    team_id: int | None,
    as_of: datetime,
) -> tuple[int | None, str | None]:
    """return_gw для v2: официальная дата FPL («Expected back 18 Sep») -> дата модели ->
    номер тура от модели. Возвращает (gw, источник) — источник пишется в timings для evals."""
    if calendar is not None:
        official = parse_fpl_return_date(prior.news, as_of)
        if official is not None:
            gw = calendar.gw_for_date(official[0], team_id=team_id)
            if gw is not None:
                return gw, "fpl_news"
        when = parse_return_date(llm_return_date, as_of)
        if when is not None:
            gw = calendar.gw_for_date(when, team_id=team_id)
            if gw is not None:
                return gw, "llm_date"
    if llm_return_gw is not None:
        return llm_return_gw, "llm_gw"
    return None, None


def normalise_draft(
    raw: BaseModel,
    *,
    player: Player,
    prior: FPLPrior,
    calendar: GWCalendar | None,
    as_of: datetime,
) -> tuple[SignalDraft, str | None]:
    """Любой черновик -> SignalDraft для validate_draft. v2: return_gw из даты, минуты по формуле."""
    if isinstance(raw, SignalDraft):
        return raw, ("llm_gw" if raw.return_gw is not None else None)
    d = raw.model_dump()
    return_gw, source = resolve_return_gw(
        prior=prior,
        llm_return_gw=d.get("return_gw"),
        llm_return_date=d.get("return_date"),
        calendar=calendar,
        team_id=player.team,
        as_of=as_of,
    )
    draft = SignalDraft(
        availability=d["availability"],
        start_probability=d["start_probability"],
        expected_minutes=expected_minutes_for(
            d["availability"], d["start_probability"], player.position.short
        ),
        rotation_risk=d["rotation_risk"],
        return_gw=return_gw,
        confidence=d["confidence"],
        summary=d["summary"],
        evidence=[EvidenceDraft.model_validate(e) for e in d["evidence"]],
    )
    return draft, source


def neutral_summary(
    player: Player, availability: str, evidence: Sequence[Evidence], prior: FPLPrior
) -> str:
    """Шаблон из полей, когда ни одно предложение summary модели не подтвердилось."""
    srcs = ", ".join(f"{e.source} {e.published_at.astimezone(UTC):%d.%m}" for e in evidence[:3])
    return (
        f"{player.full_name}: {availability} according to {len(evidence)} news item(s) "
        f"({srcs}); FPL status {prior.status} ({prior.label})."
    )


def check_signal_summary(
    summary: str,
    evidence: Sequence[Evidence],
    chunks: Sequence[RetrievedChunk],
    *,
    player: Player,
    team: Team,
    prior: FPLPrior,
    availability: str,
    calendar_text: str = "",
) -> SummaryCheck:
    """summary: даты / числа / имена — только из процитированных документов (текст, источник,
    дата публикации), FPL prior, календаря туров, имени игрока и клуба (rag/summary_check.py)."""
    cited = {e.chunk_id for e in evidence}
    docs = [c for c in chunks if c.chunk_id in cited]
    sup = build_support(
        [c.text for c in docs] + [prior.news or "", calendar_text],
        names=[
            player.full_name,
            player.web_name,
            team.name,
            team.short_name,
            *TEAM_ALIASES.get(team.short_name.upper(), ()),
            *(c.source.replace("_", " ") for c in docs),
            f"chance {prior.chance_next}" if prior.chance_next is not None else "",
        ],
        dates=[c.published_at.astimezone(UTC).date() for c in docs]
        + ([prior.news_added.date()] if prior.news_added else []),
    )
    return check_summary(
        summary, sup, fallback=neutral_summary(player, availability, evidence, prior)
    )


# ---------- конвейер ----------


@traceable(name="retrieve_for_player", run_type="retriever")
def retrieve_for_player(
    retriever: Retriever,
    player: Player,
    team: Team,
    *,
    as_of: datetime,
    k: int,
    mode: Mode,
    config: RetrievalConfig | None = None,
    corpus_cutoff: datetime | None = None,
) -> list[RetrievedChunk]:
    query = expand_player_query(player, team)
    return retriever.search(
        query,
        player_ids=[player.id],
        as_of=as_of,
        k=k,
        mode=mode,
        config=config,
        corpus_cutoff=corpus_cutoff,
    )


def load_calendar(bs: Bootstrap, fixtures: Sequence[Fixture] | None = None) -> GWCalendar:
    """Календарь туров из bootstrap + fixtures (по умолчанию — живой FPLClient().fixtures())."""
    if fixtures is None:
        try:
            fixtures = FPLClient().fixtures()
        except Exception:  # без фикстур календарь работает по окнам туров (дедлайны)
            log.warning("fixtures unavailable — calendar without team fixtures", exc_info=True)
            fixtures = []
    return GWCalendar.from_fpl(bs, fixtures)


@traceable(name="extract_signal", run_type="chain")
def extract_signal(
    player_id: int,
    as_of: datetime,
    *,
    mode: Mode = "hybrid_rerank",
    k: int = 8,
    retriever: Retriever | None = None,
    bs: Bootstrap | None = None,
    prompt_version: str | None = None,
    abstain: bool | None = None,
    retrieval_config: RetrievalConfig | None = None,
    calendar: GWCalendar | None = None,
    fixtures: Sequence[Fixture] | None = None,
    save: bool = True,
    timings: dict[str, Any] | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    top_p: float | None = None,
    corpus_cutoff: datetime | None = None,
) -> PlayerSignal:
    """Полный конвейер для одного игрока. timings (мс по стадиям, токены, llm_calls,
    abstain_reason, return_gw_source) заполняется, если передан dict.

    prompt_version: v1 — прежний конвейер (без календаря/формулы/abstention-логики v2);
    v2 (дефолт, RAG_PROMPT_VERSION) — см. докстринг модуля. abstain: RAG_ABSTAIN по умолчанию.
    model / temperature / top_p — переопределения для evals (по умолчанию RAG_LLM_MODEL, T = 0,
    top_p не передаётся); продакшен-вызовы их не задают.
    corpus_cutoff (evals): только статьи и снимки статусов FPL, собранные до этого момента
    (fetched_at / snapshot_at) — воспроизводимый корпус прошлого прогона; продакшен не задаёт.
    """
    if as_of.tzinfo is None:
        raise ValueError("as_of должен быть timezone-aware (UTC)")
    as_of = as_of.astimezone(UTC)
    configure_tracing()
    timings = timings if timings is not None else {}
    started = time.perf_counter()
    prompt_version = prompt_version or settings.rag_prompt_version
    abstain = settings.rag_abstain if abstain is None else abstain
    model = model or settings.rag_llm_model
    v2 = prompt_version != "v1"

    bs = bs or FPLClient().bootstrap()
    player = bs.player(player_id)
    team = bs.team(player.team)
    prior = fpl_prior(player, as_of, known_before=corpus_cutoff)
    next_event = next_event_as_of(bs, as_of)
    prompt = load_prompt(PROMPT_NAME, prompt_version)
    matcher = get_matcher(bs)
    if v2 and calendar is None:
        calendar = load_calendar(bs, fixtures)

    retriever = retriever or Retriever()
    chunks = retrieve_for_player(
        retriever,
        player,
        team,
        as_of=as_of,
        k=k,
        mode=mode,
        config=retrieval_config,
        corpus_cutoff=corpus_cutoff,
    )
    timings.update({f"retrieve_{k_}": v for k_, v in retriever.last_timings.items()})
    timings["llm_calls"] = 0
    timings["prompt_tokens"] = 0
    timings["completion_tokens"] = 0

    if abstain:
        decision = should_abstain(
            chunks, retriever.last_candidates, player, team, mode=mode, matcher=matcher
        )
        timings["abstain_top_score"] = decision.top_score if decision.top_score is not None else -1
        if decision.abstain:
            timings["abstain_reason"] = decision.reason
            signal = abstained_signal(
                player,
                prior,
                chunks,
                as_of=as_of,
                mode=mode,
                prompt_version=prompt.version,
                model=model,
            )
            if save:
                save_signal(signal)
            timings["total_ms"] = (time.perf_counter() - started) * 1000
            log.info(
                "%s: abstained (%s) (%.0f ms)",
                player.web_name,
                decision.reason,
                timings["total_ms"],
            )
            return signal

    messages = build_messages(
        prompt,
        player=player,
        team=team,
        prior=prior,
        chunks=chunks,
        as_of=as_of,
        next_event=next_event,
        calendar=calendar if v2 else None,
    )
    t = time.perf_counter()
    raw, tokens = call_llm(
        messages,
        model=model,
        schema=draft_schema(prompt.version),
        temperature=temperature,
        top_p=top_p,
    )
    timings["llm_ms"] = (time.perf_counter() - t) * 1000
    timings["llm_calls"] = 1
    timings.update(tokens)

    draft, return_source = normalise_draft(
        raw, player=player, prior=prior, calendar=calendar if v2 else None, as_of=as_of
    )
    with_notes = prompt.version in FORM_NOTES_VERSIONS
    raw_notes: list[FormNoteDraft] = list(getattr(raw, "form_notes", None) or [])
    moved = 0
    if with_notes:
        draft, raw_notes, moved = split_form_quotes(draft, raw_notes)
        timings["form_quotes_moved"] = moved
    draft, evidence, fixes = validate_draft(
        draft,
        chunks,
        prior,
        next_gw=next_event.id if next_event else None,
        player=player,
        team=team,
        matcher=matcher,
    )
    fixes += moved
    form_notes: list[FormNote] = []
    if with_notes:
        draft, capped = cap_indirect_fit_confidence(
            draft, evidence, getattr(raw, "evidence", None) or [], prior
        )
        fixes += capped
        timings["confidence_capped"] = capped
        form_notes, note_fixes = validate_form_notes(
            raw_notes,
            chunks,
            player=player,
            team=team,
            matcher=matcher,
            evidence=evidence,
        )
        fixes += note_fixes
        timings["form_note_fixes"] = note_fixes
    summary = draft.summary.strip()
    if not evidence and form_notes:
        summary = no_availability_news_summary(player, prior)
    if v2 and evidence and settings.rag_summary_check:
        checked = check_signal_summary(
            summary,
            evidence,
            chunks,
            player=player,
            team=team,
            prior=prior,
            availability=draft.availability,
            calendar_text=calendar.render(as_of, team=team) if calendar is not None else "",
        )
        if checked.dropped:
            fixes += checked.fixes
            timings["summary_fixes"] = checked.reasons
            log.info("%s: summary fixed: %s", player.web_name, "; ".join(checked.reasons))
        summary = checked.summary
    expected_minutes = draft.expected_minutes
    if v2:  # после валидации availability могла измениться (fit -> injured, unknown) — пересчёт
        expected_minutes = expected_minutes_for(
            draft.availability, draft.start_probability, player.position.short
        )
    if draft.return_gw is None:
        return_source = None
    if return_source:
        timings["return_gw_source"] = return_source
    signal = PlayerSignal(
        player_id=player.id,
        player_name=player.full_name,
        as_of=as_of,
        availability=draft.availability,
        start_probability=draft.start_probability,
        expected_minutes=expected_minutes,
        rotation_risk=draft.rotation_risk,
        return_gw=draft.return_gw,
        confidence=draft.confidence,
        summary=summary,
        evidence=evidence,
        fpl_status=prior.status,
        fpl_chance_next=prior.chance_next,
        model=model,
        prompt_version=prompt.version,
        retrieved_chunk_ids=[c.chunk_id for c in chunks],
        mode=mode,
        validation_fixes=fixes,
        abstained=False,
        form_notes=form_notes,
    )
    if save:
        t = time.perf_counter()
        save_signal(signal)
        timings["persist_ms"] = (time.perf_counter() - t) * 1000
    timings["total_ms"] = (time.perf_counter() - started) * 1000
    log.info(
        "%s: %s p_start=%.2f conf=%.2f evidence=%d fixes=%d return_gw=%s (%.0f ms)",
        player.web_name,
        signal.availability,
        signal.start_probability,
        signal.confidence,
        len(signal.evidence),
        fixes,
        signal.return_gw,
        timings["total_ms"],
    )
    return signal


_INSERT_SIGNAL = text(
    """
    INSERT INTO player_signals
        (player_id, as_of, availability, start_probability, expected_minutes, rotation_risk,
         return_gw, confidence, summary, evidence, fpl_status, fpl_chance_next, model,
         prompt_version, mode, retrieved_chunk_ids, validation_fixes, abstained, form_notes)
    VALUES
        (:player_id, :as_of, :availability, :start_probability, :expected_minutes, :rotation_risk,
         :return_gw, :confidence, :summary, CAST(:evidence AS jsonb), :fpl_status, :fpl_chance_next,
         :model, :prompt_version, :mode, :retrieved_chunk_ids, :validation_fixes, :abstained,
         CAST(:form_notes AS jsonb))
    RETURNING id
    """
)


def save_signal(signal: PlayerSignal) -> int:
    """Нужна миграция 010_signal_form_notes.sql (колонка form_notes)."""
    params = signal.model_dump(mode="json", exclude={"player_name"})
    params["as_of"] = signal.as_of
    params["evidence"] = json.dumps(params["evidence"], ensure_ascii=False)
    params["form_notes"] = json.dumps(params["form_notes"], ensure_ascii=False)
    with session_scope() as s:
        return int(s.execute(_INSERT_SIGNAL, params).scalar_one())
