"""Командный дайджест новостей клуба (TeamSignal): retrieval по team_ids с as_of -> gpt-4o-mini
(structured output, T = 0) -> детерминированная проверка цитат -> team_news_digests.

Зачем отдельный конвейер: сигнал игрока (rag/extract.py) по правилу 2 промпта v2 намеренно не
считает доказательством новости одноклубников и клубные round-up заголовки — A/B #1 показал, что
модель цитировала «Spurs injury update: …» как доказательство про конкретного игрока. Но для решения
«кого взять» клубный контекст полезен: травмы конкурентов за место, слова тренера о ротации, форма.
Поэтому он живёт отдельно (своя таблица, свой промпт) и в player_signals.evidence не попадает никогда.

Две части с разными источниками истины:
  - club_absences(): кто из клуба недоступен / под вопросом и ДО КАКОГО ТУРА — детерминированно из
    статусов FPL на as_of (fpl_prior: bootstrap / player_status_snapshots), дата — из официального
    текста FPL (parse_fpl_return_date), тур — первый матч клуба в эту дату или позже (GWCalendar),
    плюс сохранённый сигнал одноклубника. Статьи дают срок плохо: «next two matches» от 18.09 к
    23.09 уже наполовину сыгран. Считается при каждом чтении (дёшево), не хранится;
  - extract_team_news(): LLM-дайджест только мягкого контекста — слова тренера, ротация, форма.

extract_team_news(team_id, as_of):
  1. запрос = expand_team_query(team) («{name} {short} team news injuries press conference squad»);
  2. Retriever.search(team_ids=[клуб], player_ids=[игроки клуба], as_of=as_of) — HARD RULE as_of
     соблюдает Retriever (SQL dense + индекс BM25); здесь фильтр published_at <= as_of повторяется,
     плюс «чанк про клуб»: тег клуба / название или алиас клуба в тексте / тег игрока клуба;
  3. нет чанков про клуб -> «No club news found.» без вызова LLM (abstained=True);
  4. промпт prompts/v1/team_news.*.md: документы — ДАННЫЕ в <document> (prompt-injection guard),
     виды пунктов rotation / manager_quote / form_context, одна verbatim-цитата на пункт,
     относительные сроки («this weekend») — от даты публикации документа;
  5. validate_team_draft(): цитата — подстрока своего документа (verbatim_quote, иначе align_quote,
     иначе пункт долой); документ — про клуб; цитата не про чужой клуб; названные игроки — игроки
     этого клуба, упомянутые в документе (иначе имя снимается); claim и summary проверяются
     rag/summary_check (даты / числа / имена только из процитированных документов);
  6. TeamNewsDigest -> team_news_digests (миграция 009_team_news.sql).
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from langsmith import traceable
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from fplcopilot.config import settings
from fplcopilot.data import Bootstrap, Event, FPLClient, Player, Team
from fplcopilot.db import session_scope
from fplcopilot.prompts import Prompt, load_prompt
from fplcopilot.rag.entity_matcher import TEAM_ALIASES, EntityMatcher, strip_accents
from fplcopilot.rag.extract import (
    MAX_QUOTE_CHARS,
    STATUS_LABELS,
    FPLPrior,
    GWCalendar,
    align_quote,
    fpl_prior,
    get_matcher,
    next_event_as_of,
    parse_fpl_return_date,
    render_documents,
    verbatim_quote,
)
from fplcopilot.rag.llm import configure_tracing, get_openai_client
from fplcopilot.rag.retrieve import (
    Mode,
    RetrievalConfig,
    RetrievedChunk,
    Retriever,
    expand_team_query,
)
from fplcopilot.rag.summary_check import build_support, check_summary

log = logging.getLogger(__name__)

PROMPT_NAME = "team_news"
PROMPT_VERSION = "v1"  # версия промпта дайджеста (своя линейка, не RAG_PROMPT_VERSION сигналов)
NO_NEWS_SUMMARY = "No club news found."
MAX_ITEMS = 5
MAX_CLAIM_WORDS = 25
MAX_SUMMARY_CHARS = 400
OVERFETCH = 3  # k × 3 кандидатов из поиска -> фильтр «про клуб» и свежести -> k документов
MAX_DOC_AGE_DAYS = 14  # мягкий контекст (ротация, форма) старше двух недель не актуален
ABSENT_STATUSES = frozenset({"d", "i", "s", "u", "n"})
_DEPARTED = re.compile(
    r"\b(joined|on loan|permanent(ly)?|left the club|departed|transferred)\b", re.IGNORECASE
)

# Только мягкий контекст: кто недоступен и до какого тура — из данных FPL (club_absences), а не
# из текста статей («next two matches» от 18.09 к 23.09 уже наполовину сыгран).
ItemKind = Literal["rotation", "manager_quote", "form_context"]
ITEM_KINDS: tuple[str, ...] = ("rotation", "manager_quote", "form_context")

_TOKEN = re.compile(r"[a-z0-9]+")


# ---------- схемы ----------


class TeamNewsItemDraft(BaseModel):
    kind: ItemKind
    players: list[str] = Field(
        description="players of the target club the item is about, as written; [] if none"
    )
    claim: str = Field(description="<= 25 words, plain English, only what the quote says")
    chunk_id: int = Field(description="id of the <document> the quote is copied from")
    quote: str = Field(description="verbatim substring of that document, <= 200 characters")


class TeamNewsDraft(BaseModel):
    """Ровно то, что просим у LLM (strict structured output); остальное дописывает код."""

    items: list[TeamNewsItemDraft]
    summary: str = Field(description="<= 2 sentences built only from the items")


class TeamNewsItem(BaseModel):
    kind: ItemKind
    claim: str
    player_ids: list[int] = Field(default_factory=list)  # проверены кодом: игроки клуба в документе
    players: list[str] = Field(default_factory=list)  # их web_name
    chunk_id: int
    source: str
    url: str
    published_at: datetime
    quote: str


class TeamNewsDigest(BaseModel):
    team_id: int
    team_name: str
    as_of: datetime
    summary: str
    items: list[TeamNewsItem]
    model: str
    prompt_version: str
    mode: str
    retrieved_chunk_ids: list[int]
    validation_fixes: int = 0
    abstained: bool = False  # нет документов про клуб -> без вызова LLM


# ---------- «про клуб» и игроки клуба (детерминированно) ----------


def _tokens(s: str) -> tuple[str, ...]:
    return tuple(t for t in _TOKEN.findall(strip_accents(s).lower()) if len(t) > 1)


def _contains(seq: tuple[str, ...], sub: tuple[str, ...]) -> bool:
    n = len(sub)
    return 0 < n <= len(seq) and any(seq[i : i + n] == sub for i in range(len(seq) - n + 1))


def club_aliases(team: Team) -> list[tuple[str, ...]]:
    """Название клуба и прозвища (TEAM_ALIASES) токенами без диакритики."""
    names = [team.name, *TEAM_ALIASES.get(team.short_name.upper(), ())]
    return [t for t in dict.fromkeys(_tokens(n) for n in names) if t]


def text_mentions_club(text_: str, team: Team) -> bool:
    toks = _tokens(text_)
    return any(_contains(toks, alias) for alias in club_aliases(team))


def squad_of(bs: Bootstrap, team_id: int) -> dict[int, Player]:
    return {p.id: p for p in bs.elements if p.team == team_id}


def chunk_about_team(
    chunk: RetrievedChunk,
    team: Team,
    *,
    squad: dict[int, Player] | None = None,
    matcher: EntityMatcher | None = None,
) -> bool:
    """Документ про клуб: тег клуба, название / алиас в тексте или тег игрока этого клуба."""
    if team.id in chunk.teams:
        return True
    if squad and set(chunk.players) & squad.keys():
        return True
    if matcher is not None and team.id in matcher.match(chunk.text).teams:
        return True
    return text_mentions_club(chunk.text, team)


def _player_matches(name_toks: tuple[str, ...], player: Player) -> bool:
    return _contains(_tokens(player.web_name), name_toks) or _contains(
        _tokens(player.full_name), name_toks
    )


def verify_players(
    names: Sequence[str],
    chunk: RetrievedChunk,
    quote: str,
    squad: dict[int, Player],
    matcher: EntityMatcher | None = None,
) -> list[int]:
    """Имена модели -> id игроков клуба, которые реально упомянуты в документе (тег или матчер).
    Имя без подтверждения снимается. Если модель никого не назвала — игроки клуба из самой цитаты."""
    present = set(chunk.players) & squad.keys()
    if matcher is not None:
        present |= set(matcher.match(chunk.text).players) & squad.keys()
    out: list[int] = []
    for name in names:
        toks = _tokens(name)
        hits = [pid for pid in sorted(present) if toks and _player_matches(toks, squad[pid])]
        if len(hits) == 1 and hits[0] not in out:
            out.append(hits[0])
    if not out and not [n for n in names if n.strip()] and matcher is not None:
        out = [pid for pid in matcher.match(quote).players if pid in squad]
    return out


def quote_about_other_club(
    quote: str, team: Team, squad: dict[int, Player], matcher: EntityMatcher | None
) -> bool:
    """Цитата называет другой клуб и не называет ни этот клуб, ни его игроков — чужая новость
    из round-up статьи («Chelsea boss Maresca said …» в дайджесте Arsenal)."""
    if matcher is None:
        return False
    found = matcher.match(quote)
    if team.id in found.teams or text_mentions_club(quote, team):
        return False
    if set(found.players) & squad.keys():
        return False
    return bool(found.teams)


def _trim_words(s: str, n: int) -> str:
    words = " ".join(s.split()).split(" ")
    return " ".join(words[:n]) + (" …" if len(words) > n else "")


def validate_team_draft(
    draft: TeamNewsDraft,
    chunks: Sequence[RetrievedChunk],
    team: Team,
    *,
    squad: dict[int, Player],
    matcher: EntityMatcher | None = None,
    max_items: int = MAX_ITEMS,
) -> tuple[list[TeamNewsItem], str, int]:
    """Проверка кодом. Возвращает (пункты, summary, число исправлений)."""
    fixes = 0
    by_id = {c.chunk_id: c for c in chunks}
    items: list[TeamNewsItem] = []
    seen: set[tuple[int, str]] = set()
    for it in draft.items:
        chunk = by_id.get(it.chunk_id)
        if chunk is None or not chunk_about_team(chunk, team, squad=squad, matcher=matcher):
            fixes += 1
            continue
        quote = verbatim_quote(it.quote[:MAX_QUOTE_CHARS], chunk.text)
        if quote is None:
            quote = align_quote(it.quote, chunk.text)
            fixes += 1
            if quote is None:
                continue
        if quote_about_other_club(quote, team, squad, matcher):
            fixes += 1
            continue
        ids = verify_players(it.players, chunk, quote, squad, matcher)
        named = [n for n in it.players if n.strip()]
        if named and len(ids) < len(named):
            fixes += 1  # часть имён не подтвердилась документом — снята
        key = (chunk.chunk_id, " ".join(quote.split()).lower())
        if key in seen:
            continue
        seen.add(key)
        claim = check_summary(
            _trim_words(it.claim, MAX_CLAIM_WORDS),
            _support([chunk], team, squad),
            fallback=_trim_words(quote, MAX_CLAIM_WORDS),
        )
        fixes += claim.fixes
        items.append(
            TeamNewsItem(
                kind=it.kind,
                claim=claim.summary,
                player_ids=ids,
                players=[squad[p].web_name for p in ids],
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                url=chunk.url,
                published_at=chunk.published_at,
                quote=quote,
            )
        )
        if len(items) >= max_items:
            break
    if not items:
        if draft.summary.strip() != NO_NEWS_SUMMARY:
            fixes += 1
        return [], NO_NEWS_SUMMARY, fixes
    kept = [by_id[i.chunk_id] for i in items]
    checked = check_summary(
        " ".join(draft.summary.split())[:MAX_SUMMARY_CHARS],
        _support(kept, team, squad),
        fallback=fallback_summary(team, items),
    )
    if checked.dropped:
        log.info("%s: digest summary fixed: %s", team.short_name, "; ".join(checked.reasons))
    return items, checked.summary, fixes + checked.fixes


def _support(chunks: Sequence[RetrievedChunk], team: Team, squad: dict[int, Player]) -> Any:
    return build_support(
        [c.text for c in chunks],
        names=[
            team.name,
            team.short_name,
            *TEAM_ALIASES.get(team.short_name.upper(), ()),
            *(p.full_name for p in squad.values()),
            *(p.web_name for p in squad.values()),
            *(c.source.replace("_", " ") for c in chunks),
        ],
        dates=[c.published_at.astimezone(UTC).date() for c in chunks],
    )


def fallback_summary(team: Team, items: Sequence[TeamNewsItem]) -> str:
    kinds = {"rotation": "rotation", "manager_quote": "manager's words", "form_context": "form"}
    parts = [
        f"{kinds.get(i.kind, i.kind)} ({i.source} {i.published_at.astimezone(UTC):%d.%m})"
        for i in items[:3]
    ]
    return f"{team.name} club news: " + ", ".join(parts) + "."


# ---------- отсутствующие игроки клуба: из данных FPL, не из текста статей ----------

_TEAM_SIGNALS = text(
    """
    SELECT DISTINCT ON (player_id) player_id, as_of, availability
    FROM player_signals
    WHERE player_id = ANY(:ids) AND as_of <= :as_of
    ORDER BY player_id, as_of DESC, id DESC
    """
)


def saved_signals(ids: Sequence[int], as_of: datetime) -> dict[int, tuple[datetime, str]]:
    """player_id -> (as_of, availability) последнего сохранённого сигнала; БД недоступна — {}."""
    if not ids:
        return {}
    try:
        with session_scope() as s:
            rows = s.execute(_TEAM_SIGNALS, {"ids": list(ids), "as_of": as_of}).all()
    except SQLAlchemyError as exc:
        log.warning("player_signals unavailable for club absences: %s", exc)
        return {}
    return {int(pid): (dt.astimezone(UTC), str(av)) for pid, dt, av in rows}


def club_absences(
    bs: Bootstrap,
    team: Team,
    as_of: datetime,
    *,
    calendar: GWCalendar | None = None,
    signals: dict[int, tuple[datetime, str]] | None = None,
    prior: Callable[[Player, datetime], FPLPrior] = fpl_prior,
) -> list[dict[str, Any]]:
    """Игроки клуба со статусом FPL d/i/s/u/n на as_of (fpl_prior: живой bootstrap для «сейчас»,
    снимок player_status_snapshots для прошлого) и тур возвращения: официальная дата FPL
    («Suspended until 17 Oct») -> первый матч клуба в эту дату или позже (GWCalendar). Новостной
    сигнал одноклубника — дополнительное поле; срок из текста статей здесь не используется."""
    squad = squad_of(bs, team.id)
    sigs = signals if signals is not None else saved_signals(sorted(squad), as_of)
    out: list[dict[str, Any]] = []
    for p in sorted(squad.values(), key=lambda x: (-float(x.selected_by_percent or 0), x.id)):
        pr = prior(p, as_of)
        if pr.status not in ABSENT_STATUSES:
            continue
        if pr.status in ("u", "n") and _DEPARTED.search(pr.news or ""):
            continue  # ушёл из клуба (аренда / продажа) — не отсутствие в составе
        parsed = parse_fpl_return_date(pr.news, as_of)
        return_gw = None
        if parsed is not None and calendar is not None:
            return_gw = calendar.gw_for_date(parsed[0], team_id=team.id)
        sig = sigs.get(p.id)
        out.append(
            {
                "player_id": p.id,
                "player": p.web_name,
                "position": p.position.short,
                "status": pr.status,
                "status_label": STATUS_LABELS.get(pr.status, pr.status),
                "chance_next": pr.chance_next,
                "fpl_news": pr.news or "",
                "return_date": parsed[0].isoformat() if parsed else None,
                "return_gw": return_gw,
                "news_availability": sig[1] if sig else None,
                "news_as_of": sig[0].isoformat() if sig else None,
            }
        )
    return out


# ---------- retrieval ----------


@traceable(name="retrieve_for_team", run_type="retriever")
def retrieve_for_team(
    retriever: Retriever,
    team: Team,
    *,
    as_of: datetime,
    k: int,
    mode: Mode,
    squad: dict[int, Player],
    matcher: EntityMatcher | None = None,
    config: RetrievalConfig | None = None,
) -> list[RetrievedChunk]:
    """Поиск по клубу. as_of — HARD RULE: Retriever фильтрует dense (SQL) и BM25, здесь фильтр
    повторяется (защита в глубину), затем остаются только документы про клуб."""
    if as_of.tzinfo is None:
        raise ValueError("as_of должен быть timezone-aware (UTC)")
    raw = retriever.search(
        expand_team_query(team),
        player_ids=sorted(squad),
        team_ids=[team.id],
        as_of=as_of,
        k=k * OVERFETCH,
        mode=mode,
        config=config,
    )
    oldest = as_of - timedelta(days=MAX_DOC_AGE_DAYS)
    kept = [
        c
        for c in raw
        if oldest <= c.published_at <= as_of
        and chunk_about_team(c, team, squad=squad, matcher=matcher)
    ]
    # полные статьи раньше заголовков google_news (в заголовке нечего цитировать про ротацию)
    kept.sort(key=lambda c: c.source == "google_news")
    return kept[:k]


# ---------- промпт и LLM ----------


def build_team_messages(
    prompt: Prompt,
    *,
    team: Team,
    squad: dict[int, Player],
    chunks: Sequence[RetrievedChunk],
    as_of: datetime,
    next_event: Event | None,
) -> list[dict[str, str]]:
    names = ", ".join(sorted({p.web_name for p in squad.values()}))
    user = prompt.render_user(
        team_name=team.name,
        team_short=team.short_name,
        team_id=team.id,
        as_of=f"{as_of.astimezone(UTC):%Y-%m-%dT%H:%MZ}",
        next_gw=next_event.id if next_event else "?",
        deadline=f"{next_event.deadline_time:%Y-%m-%d %H:%MZ}" if next_event else "unknown",
        squad_names=names or "(unknown)",
        n_docs=len(chunks),
        documents=render_documents(chunks),
    )
    return [{"role": "system", "content": prompt.system}, {"role": "user", "content": user}]


@traceable(name="team_news_llm", run_type="chain")
def call_team_llm(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float = 0.0,
    max_completion_tokens: int = 900,
) -> tuple[TeamNewsDraft, dict[str, Any]]:
    """Structured output TeamNewsDraft; отказ / пустой разбор -> пустой дайджест."""
    client = get_openai_client()
    completion = client.chat.completions.parse(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        response_format=TeamNewsDraft,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
    )
    msg = completion.choices[0].message
    usage = completion.usage
    tokens: dict[str, Any] = {
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "finish_reason": completion.choices[0].finish_reason,
    }
    if msg.parsed is None:
        log.warning("team news LLM refusal/empty parse: %s", msg.refusal)
        return TeamNewsDraft(items=[], summary=NO_NEWS_SUMMARY), tokens
    return msg.parsed, tokens


# ---------- конвейер ----------


def empty_digest(
    team: Team,
    chunks: Sequence[RetrievedChunk],
    *,
    as_of: datetime,
    mode: str,
    model: str,
    abstained: bool = True,
) -> TeamNewsDigest:
    return TeamNewsDigest(
        team_id=team.id,
        team_name=team.name,
        as_of=as_of,
        summary=NO_NEWS_SUMMARY,
        items=[],
        model=model,
        prompt_version=PROMPT_VERSION,
        mode=mode,
        retrieved_chunk_ids=[c.chunk_id for c in chunks],
        abstained=abstained,
    )


@traceable(name="extract_team_news", run_type="chain")
def extract_team_news(
    team_id: int,
    as_of: datetime,
    *,
    mode: Mode = "hybrid_rerank",
    k: int = 8,
    retriever: Retriever | None = None,
    bs: Bootstrap | None = None,
    retrieval_config: RetrievalConfig | None = None,
    save: bool = True,
    timings: dict[str, Any] | None = None,
    model: str | None = None,
    temperature: float = 0.0,
) -> TeamNewsDigest:
    """Полный конвейер для одного клуба. timings: мс по стадиям, токены, llm_calls, abstain_reason."""
    if as_of.tzinfo is None:
        raise ValueError("as_of должен быть timezone-aware (UTC)")
    as_of = as_of.astimezone(UTC)
    configure_tracing()
    timings = timings if timings is not None else {}
    started = time.perf_counter()
    model = model or settings.rag_llm_model
    bs = bs or FPLClient().bootstrap()
    team = bs.team(team_id)
    squad = squad_of(bs, team.id)
    matcher = get_matcher(bs)
    retriever = retriever or Retriever()
    chunks = retrieve_for_team(
        retriever,
        team,
        as_of=as_of,
        k=k,
        mode=mode,
        squad=squad,
        matcher=matcher,
        config=retrieval_config,
    )
    timings.update({f"retrieve_{k_}": v for k_, v in retriever.last_timings.items()})
    timings.update({"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
    if not chunks:
        timings["abstain_reason"] = "no retrieved chunk is about the club"
        digest = empty_digest(team, chunks, as_of=as_of, mode=mode, model=model)
    else:
        prompt = load_prompt(PROMPT_NAME, PROMPT_VERSION)
        messages = build_team_messages(
            prompt,
            team=team,
            squad=squad,
            chunks=chunks,
            as_of=as_of,
            next_event=next_event_as_of(bs, as_of),
        )
        t = time.perf_counter()
        draft, tokens = call_team_llm(messages, model=model, temperature=temperature)
        timings["llm_ms"] = (time.perf_counter() - t) * 1000
        timings["llm_calls"] = 1
        timings.update(tokens)
        items, summary, fixes = validate_team_draft(
            draft, chunks, team, squad=squad, matcher=matcher
        )
        digest = TeamNewsDigest(
            team_id=team.id,
            team_name=team.name,
            as_of=as_of,
            summary=summary,
            items=items,
            model=model,
            prompt_version=prompt.version,
            mode=mode,
            retrieved_chunk_ids=[c.chunk_id for c in chunks],
            validation_fixes=fixes,
            abstained=False,
        )
    if save:
        t = time.perf_counter()
        save_digest(digest)
        timings["persist_ms"] = (time.perf_counter() - t) * 1000
    timings["total_ms"] = (time.perf_counter() - started) * 1000
    log.info(
        "%s: team news %d item(s), fixes=%d, abstained=%s (%.0f ms)",
        team.short_name,
        len(digest.items),
        digest.validation_fixes,
        digest.abstained,
        timings["total_ms"],
    )
    return digest


# ---------- хранение ----------

_INSERT_DIGEST = text(
    """
    INSERT INTO team_news_digests
        (team_id, as_of, summary, items, model, prompt_version, mode, retrieved_chunk_ids,
         validation_fixes, abstained)
    VALUES
        (:team_id, :as_of, :summary, CAST(:items AS jsonb), :model, :prompt_version, :mode,
         :retrieved_chunk_ids, :validation_fixes, :abstained)
    RETURNING id
    """
)

_LATEST_DIGEST = text(
    """
    SELECT as_of, summary, items, model, prompt_version, mode, abstained, validation_fixes
    FROM team_news_digests
    WHERE team_id = :team_id AND as_of <= :as_of
    ORDER BY as_of DESC, id DESC
    LIMIT 1
    """
)


def save_digest(digest: TeamNewsDigest) -> int:
    params = digest.model_dump(mode="json", exclude={"team_name"})
    params["as_of"] = digest.as_of
    params["items"] = json.dumps(params["items"], ensure_ascii=False)
    with session_scope() as s:
        return int(s.execute(_INSERT_DIGEST, params).scalar_one())


def latest_digest(team_id: int, as_of: datetime) -> dict[str, Any] | None:
    """Последний сохранённый дайджест клуба с as_of <= as_of (SQLAlchemyError — наружу)."""
    with session_scope() as s:
        row = s.execute(_LATEST_DIGEST, {"team_id": team_id, "as_of": as_of}).first()
    if row is None:
        return None
    keys = (
        "as_of",
        "summary",
        "items",
        "model",
        "prompt_version",
        "mode",
        "abstained",
        "validation_fixes",
    )
    return dict(zip(keys, row, strict=True))
