"""LLM-вызовы агента: роутер, объяснение, грейдер. Все — через rag/llm.get_openai_client
(wrap_openai при включённом LangSmith) и structured outputs по pydantic-схемам.

LLM здесь маршрутизируют, извлекают и объясняют; ни одного числа они не считают — факты
приходят готовыми из инструментов (tools.py), уже округлёнными. Промпты — файлы
agent/prompts/<version>/*.md (версионирование как у rag: эволюция v1 -> v2 показывается diff-ом);
версия по умолчанию — `AGENT_PROMPT_VERSION` (v2), файл, которого нет в версии, берётся из v1
(grader и rules_digest не менялись). v2: роутер отдаёт `language` (RouterOutputV2), объяснитель
пишет на языке вопроса, таблицы по явной схеме, Sources только по обсуждаемым игрокам,
цитирует `rules_context` из стратегической KB.
"""

from __future__ import annotations

import json
import logging
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from fplcopilot.config import settings
from fplcopilot.rag.llm import get_openai_client

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
BASE_PROMPT_VERSION = "v1"  # версия-источник для файлов, которых нет в более новой
PROMPT_VERSION = settings.agent_prompt_version  # активная версия (AGENT_PROMPT_VERSION)
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_LATIN = re.compile(r"[A-Za-z]")


def available_prompt_versions() -> list[str]:
    return sorted(p.name for p in PROMPTS_DIR.iterdir() if p.is_dir() and p.name.startswith("v"))


def detect_language(query: str) -> str:
    """Детерминированный детектор языка запроса (ISO 639-1): кириллица -> ru, иначе en.
    Страховка для v1-роутера (без поля language) и для ответов LLM без кода языка."""
    cyr = len(_CYRILLIC.findall(query or ""))
    lat = len(_LATIN.findall(query or ""))
    return "ru" if cyr > 0 and cyr >= lat * 0.5 else "en"


def normalize_language(code: str | None, query: str) -> str:
    """Код языка от роутера (v2) с проверкой формы; иначе — детектор по тексту."""
    if code:
        c = code.strip().lower()[:2]
        if len(c) == 2 and c.isalpha():
            return c
    return detect_language(query)


# USD за 1M токенов (вход, выход); источник — прайс OpenAI на момент написания (сентябрь 2026).
# Неизвестная модель оценивается по gpt-4o-mini и помечается в LLMUsage.price_known=False.
PRICES_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "o4-mini": (1.10, 4.40),
    "text-embedding-3-small": (0.02, 0.0),
}
DEFAULT_PRICE = PRICES_PER_1M["gpt-4o-mini"]


def price_for(model: str) -> tuple[tuple[float, float], bool]:
    if model in PRICES_PER_1M:
        return PRICES_PER_1M[model], True
    for name, price in PRICES_PER_1M.items():  # gpt-4o-mini-2024-07-18 -> gpt-4o-mini
        if model.startswith(name):
            return price, True
    return DEFAULT_PRICE, False


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    (p_in, p_out), _ = price_for(model)
    return round((prompt_tokens * p_in + completion_tokens * p_out) / 1_000_000, 6)


@lru_cache(maxsize=32)
def load_agent_prompt(name: str, version: str | None = None) -> str:
    """Промпт `name` версии `version` (по умолчанию активная); файл, отсутствующий в версии,
    берётся из v1 — так v2 переопределяет только router/explain."""
    version = version or PROMPT_VERSION
    for v in _fallback_chain(version):
        path = PROMPTS_DIR / v / f"{name}.md"
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"agent prompt {name!r} v{version} not found under {PROMPTS_DIR}")


def _vnum(version: str) -> int | None:
    return int(version[1:]) if version[:1] == "v" and version[1:].isdigit() else None


def _fallback_chain(version: str) -> list[str]:
    """v3 -> [v3, v2, v1]: файл, которого нет в версии, берётся из ближайшей более старой."""
    n = _vnum(version)
    if n is None:
        return [version, BASE_PROMPT_VERSION]
    older = sorted(
        (v for v in available_prompt_versions() if (_vnum(v) or 0) < n),
        key=lambda v: -(_vnum(v) or 0),
    )
    return [version, *older]


def news_block_enabled(version: str) -> bool:
    """Блок NEWS в сообщении объяснителя — с v3 (v1 / v2 рендерятся как раньше, для A/B)."""
    n = _vnum(version)
    return n is not None and n >= 3


class LLMUsage(BaseModel):
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    price_known: bool = True
    prompt_version: str = PROMPT_VERSION


TRANSIENT_ERRORS = frozenset({"RateLimitError", "APIConnectionError", "APITimeoutError"})
TRANSIENT_RETRIES = 2
TRANSIENT_BACKOFF_S = 2.0


def _parse(
    *,
    model: str,
    messages: list[dict[str, str]],
    schema: type[BaseModel],
    temperature: float,
    max_tokens: int,
    prompt_version: str = PROMPT_VERSION,
) -> tuple[BaseModel, LLMUsage]:
    client = get_openai_client()
    started = time.perf_counter()
    for attempt in range(TRANSIENT_RETRIES + 1):
        try:
            completion = client.chat.completions.parse(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                response_format=schema,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            break
        except Exception as exc:  # 429 / обрыв соединения — короткий повтор, остальное наверх
            if attempt >= TRANSIENT_RETRIES or type(exc).__name__ not in TRANSIENT_ERRORS:
                raise
            log.warning("%s from %s, retry %d", type(exc).__name__, model, attempt + 1)
            time.sleep(TRANSIENT_BACKOFF_S * (attempt + 1))
    latency = (time.perf_counter() - started) * 1000
    msg = completion.choices[0].message
    usage = completion.usage
    p_tok = usage.prompt_tokens if usage else 0
    c_tok = usage.completion_tokens if usage else 0
    _, known = price_for(model)
    info = LLMUsage(
        model=model,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        latency_ms=round(latency),
        cost_usd=estimate_cost(model, p_tok, c_tok),
        price_known=known,
        prompt_version=prompt_version,
    )
    if msg.parsed is None:
        raise RuntimeError(f"LLM returned no structured answer: {msg.refusal or 'empty'}")
    return msg.parsed, info


# ---------- роутер ----------

IntentLiteral = Literal[
    "player_status",
    "compare_players",
    "transfer",
    "captain",
    "lineup",
    "plan",
    "what_if",
    "player_ranking",
    "strategy_question",
    "off_topic",
    "betting",
]
RankingMetric = Literal[
    "points_per_million", "total_points", "consistency", "budget_consistency", "form"
]


class RankingDraft(BaseModel):
    """player_ranking (v2): что и как ранжировать по всей лиге. Код проверяет значения и
    страхует regex-парсером по тексту вопроса (позиция, цена, окно туров), если поле пустое.
    Окно туров: gw_from / gw_to как написано («gw4 и gw5» -> 4 / 5, «в GW5» -> 5 / 5) либо
    last_n_gws («последние 3 тура» -> 3) — какой именно тур «последний», решает код."""

    metric: RankingMetric | None = Field(
        description="points_per_million for cheap / budget / value-for-money / 'дешёвый' "
        "questions; consistency for steady / reliable / 'стабильный' / 'надёжный' players; "
        "budget_consistency when cheap AND consistent / stable are asked together ('самый "
        "дешёвый но стабильный'); total_points for 'most points'; form for recent form; null "
        "if unclear"
    )
    position: Literal["GKP", "DEF", "MID", "FWD"] | None = Field(
        description="position filter if the user names one (defenders / защитники -> DEF, "
        "midfielders / полузащитники -> MID, forwards / strikers / нападающие -> FWD, "
        "goalkeepers / вратари -> GKP); null otherwise"
    )
    max_price: float | None = Field(
        description="upper price bound in £m if stated ('under 6.0' -> 6.0, 'дешевле 4.5' -> 4.5, "
        "'до 5' -> 5.0); null otherwise"
    )
    min_price: float | None = Field(description="lower price bound in £m if stated; null otherwise")
    limit: int | None = Field(
        description="how many players the user asks for ('top 5' -> 5); null if not stated"
    )
    gw_from: int | None = Field(
        default=None,
        description="first gameweek of an explicit past window, copied as written: 'in GW5' / "
        "'в GW5' / 'в 5 туре' -> 5, 'gw4 и gw5' / 'GW4-GW5' -> 4; null when no gameweek "
        "number is named (do NOT compute it from 'last N gameweeks')",
    )
    gw_to: int | None = Field(
        default=None,
        description="last gameweek of that explicit window: 'in GW5' -> 5, 'gw4 и gw5' -> 5; "
        "null when no gameweek number is named",
    )
    last_n_gws: int | None = Field(
        default=None,
        description="N for 'the last / previous N gameweeks' / 'последние N туров' -> N; "
        "'last gameweek' / 'в прошлом туре' -> 1; null otherwise. Never convert it into "
        "gw_from / gw_to yourself",
    )


class ScenarioDraft(BaseModel):
    sell: list[str] = Field(description="players the user wants to sell / get rid of (as written)")
    buy: list[str] = Field(description="players the user wants to bring in (as written)")
    keep: list[str] = Field(description="players the user explicitly wants to keep")
    allow_hit: bool | None = Field(
        description="true if the user explicitly accepts a points hit (-4, 'take a hit'); "
        "false if they refuse hits; null if not mentioned"
    )
    use_wildcard: bool | None = Field(
        description="true if the user wants to (or asks what if they) play the Wildcard; "
        "false if they rule it out; null if not mentioned"
    )


class RouterOutput(BaseModel):
    intent: IntentLiteral
    player_mentions: list[str] = Field(
        description="player names exactly as written in the query (surface forms), no clubs"
    )
    horizon: int | None = Field(
        description="number of gameweeks the user asks about (e.g. 'next 5 gameweeks' -> 5), "
        "null if not stated"
    )
    scenario: ScenarioDraft
    needs_squad: bool = Field(
        description="true if answering requires the user's own squad (transfers, captain, "
        "lineup, plan, what-if, 'my team')"
    )
    reason: str = Field(description="one short sentence explaining the routing decision")


class RouterOutputV2(RouterOutput):
    """v2: + язык запроса и параметры ранжирования игроков (интент player_ranking).
    Схема v1 не меняется, чтобы A/B сравнивал ровно промпты."""

    language: str = Field(
        description="ISO 639-1 code of the language the query is written in: 'en', 'ru', 'de', "
        "... Player and club names do not count — judge by the surrounding words"
    )
    ranking: RankingDraft | None = Field(
        default=None,
        description="only for intent player_ranking: metric, position and price filters, limit, "
        "gameweek window (gw_from / gw_to as written, or last_n_gws); null for every other intent",
    )


_LATIN_NAMES = (
    "Latin script as written; a Cyrillic or declined name in its original Latin FPL spelling, "
    "nominative ('Саки' -> 'Saka')"
)


class ScenarioDraftV3(ScenarioDraft):
    sell: list[str] = Field(
        description=f"players the user wants to sell / replace ({_LATIN_NAMES})"
    )
    buy: list[str] = Field(description=f"players the user wants to bring in ({_LATIN_NAMES})")
    keep: list[str] = Field(
        description=f"players the user explicitly wants to keep ({_LATIN_NAMES})"
    )


class RouterOutputV3(RouterOutputV2):
    """v3: имена — латиницей в именительном падеже (промпт v3); v2 копирует их как написаны."""

    player_mentions: list[str] = Field(
        description=f"player names mentioned in the query, no clubs ({_LATIN_NAMES})"
    )
    scenario: ScenarioDraftV3


IntentLiteralV4 = Literal[
    "player_status",
    "compare_players",
    "transfer",
    "captain",
    "lineup",
    "plan",
    "what_if",
    "player_ranking",
    "strategy_question",
    "squad_review",
    "fixtures",
    "chips",
    "general_fpl",
    "gw_review",
    "off_topic",
    "betting",
]
RankingMetricV4 = Literal[
    "points_per_million",
    "total_points",
    "consistency",
    "budget_consistency",
    "form",
    "xpts",
    "xpts_per_million",
]


class RankingDraftV4(RankingDraft):
    """v4: + прогнозные метрики («кто БУДЕТ лучшим» -> xpts / xpts_per_million на горизонт) и
    фильтр владения для дифференциалов. Прошлое окно (gw_from / gw_to / last_n_gws) — только
    для прошедших туров."""

    metric: RankingMetricV4 | None = Field(
        description="FUTURE questions (will / next N gameweeks / upcoming / 'будет', 'на "
        "ближайшие', 'в следующих') -> xpts, or xpts_per_million when cheap / value is asked; "
        "PAST questions (was / scored / last N / season so far) -> points_per_million for cheap / "
        "value, consistency for steady, budget_consistency for cheap AND steady, total_points "
        "for most points, form for recent form; null if unclear"
    )
    horizon_gws: int | None = Field(
        default=None,
        description="for xpts / xpts_per_million: how many upcoming gameweeks ('next 3' -> 3); "
        "null if not stated",
    )
    max_ownership: float | None = Field(
        default=None,
        description="ownership cap in % for differentials ('differential' / 'low ownership' / "
        "'дифференциал' -> 10.0, or the number the user states); null otherwise",
    )


class ChipDraft(BaseModel):
    chip: Literal["bboost", "3xc", "wildcard", "freehit"] = Field(
        description="bboost = Bench Boost (BB, 'бенч буст'), 3xc = Triple Captain (TC, 'трипл "
        "капитан'), wildcard (WC, 'вайлдкард'), freehit (FH, 'фри хит')"
    )
    gw: int | None = Field(
        description="gameweek the user plans or asks to play it in ('BB in GW7' -> 7); null if "
        "no gameweek is named"
    )


class RouterOutputV4(RouterOutputV3):
    """v4: новые интенты, самостоятельная формулировка уточнения по истории, целевой тур,
    фишки по турам, клубы, прогнозный рейтинг."""

    intent: IntentLiteralV4
    standalone_query: str = Field(
        description="the question rewritten as a self-contained question in the user's language, "
        "resolving pronouns, ellipsis and follow-ups from CONVERSATION (players, gameweek, "
        "chips, constraints of the previous turn); the query itself when it is already complete"
    )
    target_gw: int | None = Field(
        description="the specific gameweek the answer is about when the user names one ('на "
        "gw7', 'in GW8', 'к 7 туру'); null when none is named"
    )
    chips: list[ChipDraft] = Field(
        description="chips the user plans / asks about, with their gameweek; empty if none"
    )
    team_mentions: list[str] = Field(
        description="clubs mentioned (English FPL names: 'Арсенал' -> 'Arsenal', 'Ман Сити' -> "
        "'Man City'); empty if none"
    )
    ranking: RankingDraftV4 | None = Field(
        default=None,
        description="only for intent player_ranking; null for every other intent",
    )


def router_schema(version: str) -> type[RouterOutput]:
    n = _vnum(version)
    if version == BASE_PROMPT_VERSION:
        return RouterOutput
    if n is not None and n >= 4:
        return RouterOutputV4
    return RouterOutputV3 if n is not None and n >= 3 else RouterOutputV2


class RouterRequest(BaseModel):
    query: str
    as_of: str
    gw: int | None
    deadline: str | None
    has_manager: bool
    strategy: str
    history: str = ""  # v4: agent.chat.render_history — предыдущие реплики и meta их ходов


def router_llm(
    req: RouterRequest, *, model: str | None = None, version: str | None = None
) -> tuple[RouterOutput, LLMUsage]:
    model = model or settings.agent_router_model
    version = version or PROMPT_VERSION
    system = load_agent_prompt("router.system", version)
    user = (
        f"Query: {req.query}\n"
        f"Context: data as of {req.as_of}; next gameweek GW{req.gw} "
        f"(deadline {req.deadline}); manager squad {'available' if req.has_manager else 'NOT provided'}; "
        f"strategy preset {req.strategy}."
    )
    if req.history and (_vnum(version) or 0) >= 4:
        user = (
            "CONVERSATION (previous turns, oldest first; use it only to complete the current "
            f"query):\n{req.history}\n\n" + user
        )
    out, usage = _parse(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        schema=router_schema(version),
        temperature=0,
        max_tokens=400,
        prompt_version=version,
    )
    return out, usage  # type: ignore[return-value]


# ---------- объяснение ----------


class ExplainRequest(BaseModel):
    query: str
    intent: str
    strategy: str
    as_of: str
    facts: dict[str, Any]
    evidence: list[dict[str, Any]]
    caveats: list[str]
    feedback: str | None = None  # нарушения валидатора при регенерации
    language: str = "en"  # ISO 639-1: язык ответа (v2), имена и числа не переводятся
    history: str = ""  # v4: предыдущие реплики (для «почему»-вопросов и уточнений)
    original_query: str | None = None  # v4: как спросил пользователь, если роутер переписал


class ExplainOutput(BaseModel):
    answer_markdown: str = Field(description="the full answer in Markdown, following the format")


LANGUAGE_NAMES: dict[str, str] = {"en": "English", "ru": "Russian", "de": "German", "es": "Spanish"}


def _render_rules_context(rc: dict[str, Any]) -> list[str]:
    """v2: блок RULES CONTEXT с готовой строкой цитаты — модель копирует [source], не выдумывая."""
    excerpts = list(rc.get("excerpts") or [])
    if not excerpts:
        return []
    first = excerpts[0]
    lines = [
        "",
        (
            f"RULES CONTEXT (strategy knowledge base, about: {rc.get('about')}). REQUIRED: add "
            f"exactly ONE bullet to Why that states the rule behind the {rc.get('about')} decision "
            f"and ends with the citation [{first.get('source')}]; list "
            f'"{first.get("source")} — {first.get("title")}, {first.get("url")}" under Sources. '
            "The numbers of the decision still come from FACTS."
        ),
    ]
    for i, x in enumerate(excerpts, start=1):
        kind = "official rules" if x.get("official_rules") else "community guide"
        lines.append(
            f'{i}. cite as [{x.get("source")}] — {x.get("title")} ({kind}): "{x.get("text")}"'
        )
    return lines


def _render_kb_answer(sa: dict[str, Any]) -> list[str]:
    """v2: блок KB ANSWER — текст с маркерами [n] и готовый блок Sources для копирования."""
    lines = [
        "",
        (
            "KB ANSWER (strategy knowledge base; covered="
            f"{sa.get('covered')}). Reproduce its claims in LANGUAGE keeping every [n] marker "
            "exactly where it stands (a marker is the citation — never replace it with a source "
            "name); copy the SOURCES BLOCK below verbatim under the Sources heading:"
        ),
        str(sa.get("answer") or ""),
        "",
        "SOURCES BLOCK (verbatim):",
        str(sa.get("sources_markdown") or "(none)"),
    ]
    return lines


def _cites(evidence: list[dict[str, Any]], *, player: str | None = None, club: str | None = None):
    out: list[str] = []
    for e in evidence:
        # scope=form (форма / роль, сигнал v4) — не цитата для строки о доступности
        own = e.get("scope") not in ("club", "form")
        if (player is not None and own and e.get("player") == player) or (
            club is not None and e.get("scope") == "club" and e.get("team") == club
        ):
            c = f"[{e.get('source')}, {e.get('date')}]"
            if c not in out:
                out.append(c)
    return out[:2]


def render_news_block(facts: dict[str, Any], evidence: list[dict[str, Any]]) -> list[str]:
    """v3: блок NEWS — новости рекомендованных / ранжированных игроков и клубов с готовыми
    цитатами (как RULES CONTEXT: внутри FACTS JSON модель их игнорировала)."""
    cn = (facts.get("candidate_news") or {}).get("players") or {}
    tn = facts.get("team_news") or {}
    if not cn and not tn:
        return []
    lines = [
        "",
        (
            "NEWS (from FACTS.candidate_news / FACTS.team_news). REQUIRED in Why: for the "
            "recommended incoming / top ranked players below (at most 3, in this order) state what "
            "the player news says and end with the citation shown; 'no player-specific news' when "
            "there is none. Then ONE bullet per club listed below that has unavailable players or "
            "a digest (at most 2 clubs): who is out and until which GW (FPL data, no citation) and "
            "the digest line with its citation — phrased as club news, never as evidence about a "
            "player's own fitness. Numbers still come from FACTS. Paraphrase news without "
            "quotation marks (no translated 'quotes'); player names in Latin script as written."
        ),
    ]
    for name, p in list(cn.items())[:6]:
        sig = p.get("news_signal") or {}
        cites = _cites(evidence, player=name.split(" (")[0])
        line = f"- {name} ({p.get('role')}): news {sig.get('availability')}"
        if sig.get("availability") not in (None, "n/a", "unknown"):
            line += f", confidence {sig.get('confidence')}, signal {sig.get('signal_as_of')}"
        line += f" — cite {' '.join(cites)}" if cites else " — no player-specific news, no citation"
        lines.append(line)
    for code, club in tn.items():
        out = [
            f"{a.get('player')} {a.get('status_label')}"
            + (f" until GW{a['return_gw']}" if a.get("return_gw") else "")
            for a in club.get("unavailable_or_doubtful_players") or []
        ][:6]
        cites = _cites(evidence, club=code)
        lines.append(
            f"- club news {club.get('club')} ({code}): unavailable/doubtful by FPL data: "
            + (", ".join(out) if out else "none")
            + (f"; digest: {club.get('summary')}" if club.get("summary") else "")
            + (f" — cite {' '.join(cites)}" if cites else "")
        )
    return lines


NEWS_HEADER_V4 = (
    "NEWS (from FACTS.candidate_news / FACTS.team_news, with ready-made citations). Use it in the "
    "answer only for the recommended incoming / top ranked players (at most 3, one short clause "
    "each: what the player news says + the citation shown; 'no player-specific news' when none) "
    "and at most ONE club-news bullet when it matters for a recommended player. Numbers still "
    "come from FACTS; player names in Latin script as written."
)


def render_explain_user(req: ExplainRequest, version: str | None = None) -> str:
    """Пользовательское сообщение объяснителя. v1 — ровно как в шаге 7 (без строки LANGUAGE и
    дополнительных блоков), чтобы A/B сравнивал версии как есть; v2 добавляет LANGUAGE и явные
    блоки RULES CONTEXT / KB ANSWER поверх тех же FACTS."""
    version = version or PROMPT_VERSION
    v2 = version != BASE_PROMPT_VERSION
    lang = LANGUAGE_NAMES.get(req.language, req.language)
    titles = (
        "keep the English section titles Verdict / Why / Sources / Caveats"
        if req.language == "en"
        else "translate the section titles too"
    )
    parts = [f"QUESTION: {req.query}", f"INTENT: {req.intent}"]
    if (_vnum(version) or 0) >= 4:
        if req.original_query and req.original_query.strip() != req.query.strip():
            parts.append(f"USER'S OWN WORDS: {req.original_query}")
        if req.history:
            parts.append(
                "CONVERSATION SO FAR (context only — numbers still come from FACTS):\n"
                + req.history
            )
    if v2:
        parts.append(
            f"LANGUAGE: {req.language} ({lang}) — write the whole answer in this language "
            f"({titles}); keep player and club names as written in FACTS and copy every number "
            "exactly (decimal point, same digits)"
        )
    parts += [
        f"STRATEGY PRESET: {req.strategy}",
        f"DATA AS OF: {req.as_of}",
        "",
        "FACTS (JSON, computed by deterministic models — use ONLY these numbers and names):",
        json.dumps(req.facts, ensure_ascii=False, indent=1, default=str),
        "",
        "EVIDENCE (news quotes; cite as [source, dd.mm]; list URLs under Sources):",
        json.dumps(req.evidence, ensure_ascii=False, indent=1, default=str)
        if req.evidence
        else "(none — say that no player-specific news evidence was found)",
    ]
    if v2 and isinstance(req.facts.get("rules_context"), dict):
        parts += _render_rules_context(req.facts["rules_context"])
    if v2 and isinstance(req.facts.get("strategy_answer"), dict):
        parts += _render_kb_answer(req.facts["strategy_answer"])
    if news_block_enabled(version):
        block = render_news_block(req.facts, req.evidence)
        if block and (_vnum(version) or 0) >= 4:
            # v4: новости — нюанс к рекомендации, а не обязательные пункты (клубный шум в v3)
            block[1] = NEWS_HEADER_V4
        parts += block
    parts += [
        "",
        "CAVEATS TO INCLUDE:",
        "\n".join(f"- {c}" for c in req.caveats) if req.caveats else "- (none)",
    ]
    if req.feedback:
        parts += [
            "",
            "VALIDATION FEEDBACK ON YOUR PREVIOUS ANSWER (fix these, keep everything else):",
            req.feedback,
        ]
    if (_vnum(version) or 0) >= 4:
        # последняя строка сообщения — язык и заголовки (в v4 модель сползала в русские
        # заголовки в английском ответе: примеры ru в системном промпте)
        heads = (
            "Why / Sources / Caveats, 'Data as of'"
            if req.language == "en"
            else "Почему / Источники / Оговорки, «Данные на»"
            if req.language == "ru"
            else "in that language"
        )
        parts += [
            "",
            (
                f"WRITE THE WHOLE ANSWER IN {lang.upper()} — prose, caveats, headings ({heads}). "
                "Skip any heading you have nothing for (never write '(none)')."
            ),
        ]
    return "\n".join(parts)


def explain_llm(
    req: ExplainRequest,
    *,
    model: str | None = None,
    temperature: float | None = None,
    version: str | None = None,
) -> tuple[ExplainOutput, LLMUsage]:
    """temperature по умолчанию — `AGENT_EXPLAIN_TEMPERATURE` (0; docs/LLM_CHOICE.md §4.2)."""
    model = model or settings.agent_explain_model
    temperature = settings.agent_explain_temperature if temperature is None else temperature
    version = version or PROMPT_VERSION
    system = load_agent_prompt("explain.system", version)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": render_explain_user(req, version)},
    ]
    # v3: + пункты о новостях кандидатов и клуба; русский ответ с таблицей упирался в 1400
    max_tokens = 2000 if news_block_enabled(version) else 1400
    try:
        out, usage = _parse(
            model=model,
            messages=messages,
            schema=ExplainOutput,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_version=version,
        )
    except Exception as exc:
        # v4: ответ не уместился в лимит (модель «зациклилась» на списке) — один повтор с
        # требованием краткости вместо фолбэка без объяснения
        if type(exc).__name__ != "LengthFinishReasonError" or (_vnum(version) or 0) < 4:
            raise
        log.warning("explain hit max_tokens (%s) — retry once, concise", max_tokens)
        out, usage = _parse(
            model=model,
            messages=[
                *messages,
                {
                    "role": "user",
                    "content": "Your previous answer was cut off by the length limit. Answer "
                    "again in at most 200 words: the answer first, 2–4 bullets, no long lists.",
                },
            ],
            schema=ExplainOutput,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_version=version,
        )
    return out, usage  # type: ignore[return-value]


# ---------- грейдер достаточности сигнала ----------


class GradeRequest(BaseModel):
    player: str
    fpl_status: str
    fpl_chance: int | None
    fpl_news: str
    availability: str
    confidence: float
    summary: str
    evidence: list[dict[str, Any]]
    next_gw: int | None


class GradeOutput(BaseModel):
    sufficient: bool = Field(
        description="true if the evidence answers whether the player is available for the next "
        "gameweek well enough to make a decision"
    )
    reason: str = Field(description="one short sentence")


def grader_llm(
    req: GradeRequest, *, model: str | None = None, version: str | None = None
) -> tuple[GradeOutput, LLMUsage]:
    model = model or settings.agent_router_model
    version = version or PROMPT_VERSION
    system = load_agent_prompt("grader.system", version)  # v2 не переопределяет -> файл v1
    user = json.dumps(req.model_dump(), ensure_ascii=False, indent=1)
    out, usage = _parse(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        schema=GradeOutput,
        temperature=0,
        max_tokens=120,
        prompt_version=version,
    )
    return out, usage  # type: ignore[return-value]
