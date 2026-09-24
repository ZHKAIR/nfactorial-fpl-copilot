"""LangGraph-агент FPL Copilot (шаг 7, v2): маршрутизация -> сигналы (с ограниченным циклом
повторного поиска) -> детерминированные вычисления (+ стратегическая KB) -> подтверждение
человеком -> объяснение -> проверка ответа.

    load_context -> router -+-> refuse -----------------------------------------------> END
                            +-> clarify -> resolve_clarification (interrupt_before) -+-> clarify (ещё имя)
                            |                 (cancel -> END)                        +-> ensure_signals
                            +-> ensure_signals -> grade_signals <-> rewrite_retry (<= 2 раз)
                                                        |
                                   compute -> candidate_news -> check_action -+-> explain
                                      ^   +--- (1 пересчёт) --+                |
                                      +-------------- confirm_action <--------+  (interrupt_before)
                                                             (reject -> compute, confirm -> explain)

`candidate_news` (после расчёта, когда кандидаты известны): сигналы новостей игроков, которых
рекомендует / ранжирует ответ, и дайджесты их клубов в пределах бюджета LLM-извлечений на запрос;
рекомендованная покупка, недоступная по свежей новости, -> один пересчёт без неё.
                                                     explain <-> validate_answer (1 регенерация) -> END

Два вида human-in-the-loop: подтверждение платного действия (`confirm_action`: хит / Wildcard) и
выбор игрока при неоднозначном имени (`resolve_clarification`: «Gabriel» -> кандидаты).
`compute` для интента strategy_question берёт цитируемый ответ стратегической KB
(`answer_strategy_question`), для player_ranking («самый дешёвый, но стабильный игрок?», «best
value midfielder under 6.0», «кто был лучшим в gw4 и gw5») — детерминированный рейтинг всей лиги
`rank_players` (bootstrap + player_gw_history; параметры и окно туров из роутера v2 `ranking` с
regex-страховкой `ranking_params`; окно считается только по истории),
а для transfer / plan / what_if с хитом или чипом подшивает к фактам
1–2 чанка KB (`rules_context`, теги hits / chips) — объяснитель цитирует их как [source]; числа
по-прежнему только из оптимизатора.

Принцип проекта: LLM маршрутизируют, извлекают и объясняют; считают — core/ (xPts, MILP) через
инструменты tools.py. Все зависимости (инструменты, три LLM-вызова, часы, резолвер имён)
инжектируются через `Deps`, поэтому граф тестируется фейками без сети/БД/LLM.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from fplcopilot.agent import chat, tracing
from fplcopilot.agent.llm import (
    ExplainOutput,
    ExplainRequest,
    GradeOutput,
    GradeRequest,
    LLMUsage,
    RouterOutput,
    RouterRequest,
    estimate_cost,
    load_agent_prompt,
    normalize_language,
)
from fplcopilot.agent.resolve import Resolution
from fplcopilot.agent.state import (
    INTENTS,
    NEWS_INTENTS,
    SIGNAL_SQUAD_INTENTS,
    SQUAD_INTENTS,
    AgentState,
    initial_state,
)
from fplcopilot.agent.tools import (
    KB_TAGS_CHIPS,
    KB_TAGS_HITS,
    MAX_GW,
    POSITIONS,
    RANK_LIMIT_MAX,
    RANKING_METRICS,
    SEASON_WINDOW_LABEL,
    AgentTools,
    BuildPlanInput,
    ChipsStatusInput,
    ComparePlayersInput,
    DiagnoseSquadInput,
    ForecastRankInput,
    GameweekContextInput,
    GWReviewInput,
    KBSearchInput,
    OptimizeTeamInput,
    PlanChipIn,
    PlayerRiskInput,
    PredictPlayerInput,
    RankPlayersInput,
    RecommendTransfersInput,
    SimulateScenarioInput,
    SquadLike,
    SquadOverride,
    StrategyAnswerInput,
    TeamFixturesInput,
    TeamNewsInput,
    TransferTrendsInput,
    as_override,
)
from fplcopilot.agent.translit import fix_mixed_script
from fplcopilot.agent.validate import (
    fix_name_typos,
    not_in_squad_lines,
    prune_undiscussed_sources,
    restore_latin_names,
    strip_bad_citations,
    strip_lines,
    strip_ratings,
    validate_answer,
)
from fplcopilot.config import settings
from fplcopilot.rag.form_notes import (
    FORM_ABOUT,
    form_evidence,
    form_notes_brief,
    load_form_notes,
)

log = logging.getLogger(__name__)

DEFAULT_HORIZON: dict[str, int] = {
    "player_status": 3,
    "compare_players": 3,
    "transfer": 3,
    "captain": 1,
    "lineup": 1,
    "plan": 5,
    "what_if": 3,
    "player_ranking": 1,
    "strategy_question": 1,
    "off_topic": 1,
    "betting": 1,
    "squad_review": 3,
    "fixtures": 3,
    "chips": 5,
    "general_fpl": 1,
    "gw_review": 1,
}
# v4: «кто БУДЕТ лучшим» — прогнозный рейтинг (LiveTools.rank_forecast), не история
FORECAST_METRICS = frozenset({"xpts", "xpts_per_million"})
MAX_HORIZON = 8
# Интенты, не привязанные к составу пользователя: блок менеджера в факты не идёт.
LEAGUE_WIDE_INTENTS = frozenset({"strategy_question", "player_ranking"})
# Страховка роутера для player_ranking: фильтры и метрика по тексту вопроса (RU / EN), если LLM
# не заполнил поле `ranking` (v1-схема или пропуск).
RANK_MAX_PRICE = re.compile(
    r"(?:under|below|up\s+to|max(?:imum)?|cheaper\s+than|less\s+than|at\s+most|<=?|"
    r"дешевле|не\s+дороже|до|максимум|ниже|меньше)\s*£?\s*(\d{1,2}(?:[.,]\d)?)\s*(?:m|£|млн)?\b",
    re.IGNORECASE,
)
RANK_MIN_PRICE = re.compile(
    r"(?:over|above|more\s+than|at\s+least|>=?|дороже|не\s+дешевле|от|минимум|выше)\s*£?\s*"
    r"(\d{1,2}(?:[.,]\d)?)\s*(?:m|£|млн)?\b",
    re.IGNORECASE,
)
RANK_LIMIT = re.compile(
    r"\b(?:top|топ)[\s-]*(\d{1,2})\b|\b(\d{1,2})\s+(?:best|players?|игрок\w*|лучш\w*)\b",
    re.IGNORECASE,
)


def _words(pattern: str) -> re.Pattern[str]:
    return re.compile(rf"\b({pattern})\b", re.IGNORECASE)


RANK_POSITION_WORDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("GKP", _words(r"goalkeepers?|keepers?|gks?|gkp|вратар\w*|голкипер\w*|кипер\w*")),
    ("DEF", _words(r"defenders?|defence|def|защитник\w*|защит\w*|оборон\w*")),
    ("MID", _words(r"midfielders?|mids?|полузащитник\w*|хав\w*|мидфилдер\w*")),
    ("FWD", _words(r"forwards?|strikers?|fwd|нападающ\w*|форвард\w*|страйкер\w*")),
)
# Порядок = приоритет при одном совпадении; «дешёвый» + «стабильный» вместе -> budget_consistency
# (см. ranking_params). Границы цены («дешевле 4.5», «under 6.0») вырезаются до поиска слов.
RANK_METRIC_WORDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "points_per_million",
        _words(
            r"cheap\w*|budget|value|bargain\w*|per\s+million|per\s+£?m|дешев\w*|дешёв\w*|"
            r"бюджет\w*|недорог\w*|за\s+свои\s+деньги|окупа\w*"
        ),
    ),
    (
        "consistency",
        _words(
            r"consisten\w*|stable|stabl\w*|steady|reliab\w*|стабильн\w*|надежн\w*|надёжн\w*|"
            r"регулярн\w*|без\s+провалов"
        ),
    ),
    ("form", _words(r"in\s+form|form|hot|форм\w*")),
    (
        "total_points",
        _words(
            r"most\s+points|by\s+(?:total\s+)?points|top\s+scor\w*|highest\s+scor\w*|"
            r"больше\s+всего\s+очков|самы\w*\s+результативн\w*|по\s+(?:общим\s+)?очкам"
        ),
    ),
)
RANK_DEFAULT_METRIC = "points_per_million"
RANK_DEFAULT_LIMIT = 8  # строк в таблице объяснителя по умолчанию (лимит слов ответа)
# Окно туров для player_ranking (страховка роутера, как для метрики): «кто был лучшим в gw4 и
# gw5», «top scorers in the last 3 gameweeks», «лучшие защитники в GW5», «GW3–GW5», «с 3 по 5
# тур». «Последние N» остаются N — какой тур последний завершённый, решает инструмент по bootstrap.
_GW_WORD = r"(?:gws?|gameweeks?|game\s*weeks?|гв|тур\w*)"
RANK_GW_RANGE = re.compile(
    rf"\b{_GW_WORD}\s*(\d{{1,2}})\s*[-–—]\s*{_GW_WORD}?\s*(\d{{1,2}})\b"
    rf"|\b(?:с|from)\s+{_GW_WORD}?\s*(\d{{1,2}})(?:-?\w{{1,2}})?\s+(?:по|to|through|until)\s+"
    rf"{_GW_WORD}?\s*(\d{{1,2}})\b",
    re.IGNORECASE,
)
RANK_GW = re.compile(
    r"\b(?:gws?|gameweeks?|game\s*weeks?|rounds?|гв|тур[аеы]?)\s*-?\s*(\d{1,2})\b"  # gw4, тур 5
    r"|\b(\d{1,2})(?:st|nd|rd|th|-?(?:й|м|го|ом|ый|ой|ем))\s+(?:gw|gameweek|round|тур\w*)\b"
    r"|\b(\d{1,2})\s+тур[еу]\b",  # «в 5 туре» (а «3 тура» / «5 туров» — период, не номер)
    re.IGNORECASE,
)
RANK_LAST_N = re.compile(
    r"\b(?:last|past|previous|recent)\s+(\d{1,2})\s+"
    r"(?:gws?|gameweeks?|game\s*weeks?|rounds?|weeks?|matches|games)\b"
    r"|\b(?:последн\w*|прошл\w*|предыдущ\w*|за)\s+(\d{1,2})\s+(?:тур\w*|гв|gws?|матч\w*|игр\w*|недел\w*)\b",
    re.IGNORECASE,
)
RANK_LAST_ONE = re.compile(
    r"\b(?:last|past|previous)\s+(?:gw|gameweek|game\s*week|round|week)\b"
    r"|\b(?:последн\w*|прошл\w*|предыдущ\w*)\s+(?:тур\w*|гв|недел\w*)\b",
    re.IGNORECASE,
)
HORIZON_HINT = re.compile(
    r"\b\d+\s*(gws?|gameweeks?|weeks?|games?|matches|rounds?|тур(а|ов)?|недел[иьюя]?)\b"
    r"|\bnext\s+\d+\b|\bgw\s?\d+\b|\bgameweek\s?\d+\b|\bthis\s+(week|gameweek|round)\b"
    r"|\blong[- ]term\b|\bmonth\b|\bseason\b|\bhorizon\b"
    r"|\bследующ\w*\s+\d+\b|\bэт(от|ом)\s+тур\w*\b|\bна\s+\d+\s+тур\w*\b|\bсезон\w*\b",
    re.IGNORECASE,
)
# Альтернативные стратегии поиска для цикла rewrite_retry: (mode, k) по номеру попытки.
RETRY_STRATEGIES: tuple[tuple[str, int], ...] = (("dense", 8), ("hybrid_rerank", 12))
SUSPICIOUS_STATUSES = frozenset({"d", "i", "s", "u", "n"})
LOW_CONFIDENCE = 0.4
ISSUE_KINDS_FOR_SIGNALS = frozenset(
    {"injured", "suspended", "unavailable", "doubtful", "not_playing"}
)
MAX_EXPLAIN_ATTEMPTS = 2
FORCED_SALE_TOLERANCE = 0.5  # xPts за горизонт: меньше — «примерно поровну»
# Стратегическая KB в фактах: k чанков для rules_context и длина выдержки для объяснителя.
RULES_CONTEXT_K = 2
RULES_CONTEXT_EXCERPT_CHARS = 420
KB_ANSWER_K = 6
NOT_COVERED = "Not covered by the strategy knowledge base."
CLARIFY_DECISIONS = ("choose", "cancel")

RouterFn = Callable[[RouterRequest], tuple[RouterOutput, LLMUsage]]
ExplainFn = Callable[[ExplainRequest], tuple[ExplainOutput, LLMUsage]]
GraderFn = Callable[[GradeRequest], tuple[GradeOutput, LLMUsage]]


class ResolverLike:
    """Интерфейс резолвера имён (agent/resolve.PlayerResolver или фейк в тестах)."""

    def resolve_all(
        self, mentions: Iterable[str], query: str
    ) -> tuple[list[dict[str, Any]], list[Resolution], list[Resolution], list[str]]: ...


@dataclass
class Deps:
    tools: AgentTools
    router_llm: RouterFn
    explain_llm: ExplainFn
    resolver: Callable[[list[int]], ResolverLike]  # squad ids -> резолвер
    grader_llm: GraderFn | None = None
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    signal_max_age_h: float = field(default_factory=lambda: settings.agent_signal_max_age_h)
    max_retries: int = 2
    max_signal_players: int = 8
    router_model: str = field(default_factory=lambda: settings.agent_router_model)
    explain_model: str = field(default_factory=lambda: settings.agent_explain_model)
    extra_known_names: list[str] = field(default_factory=list)  # клубы и т.п. для валидатора
    prompt_version: str = field(default_factory=lambda: settings.agent_prompt_version)
    rules_context: bool = True  # подшивать чанки KB (hits / chips) к фактам при хите / чипе
    # candidate_news (после compute): новости по рекомендованным / ранжированным игрокам и клубам
    candidate_news: bool = field(default_factory=lambda: settings.agent_candidate_news)
    candidate_max_players: int = field(
        default_factory=lambda: settings.agent_candidate_max_players
    )
    candidate_routes: int = field(default_factory=lambda: settings.agent_candidate_routes)
    candidate_ranking_top: int = field(
        default_factory=lambda: settings.agent_candidate_ranking_top
    )
    team_news_max_clubs: int = field(default_factory=lambda: settings.agent_team_news_max_clubs)
    news_max_llm_calls: int = field(default_factory=lambda: settings.agent_news_max_llm_calls)
    news_mode: str = field(default_factory=lambda: settings.agent_news_mode)
    news_reoptimize: bool = field(default_factory=lambda: settings.agent_news_reoptimize)
    news_exclude_min_confidence: float = field(
        default_factory=lambda: settings.agent_news_exclude_min_confidence
    )


# ---------- утилиты ----------


def new_thread_id() -> str:
    return uuid.uuid4().hex[:12]


def chat_v4(deps: Deps) -> bool:
    """Поведение чата v4 (новые интенты, general_fpl вместо отказа) — с промптов v4."""
    v = str(deps.prompt_version or "")
    return v[:1] == "v" and v[1:].isdigit() and int(v[1:]) >= 4


def chat_target_gw(raw: Any, gw: Any) -> int | None:
    """Целевой тур из роутера v4: только ближайший и будущие туры сезона."""
    t = _gw_number(raw)
    now = _gw_number(gw) or 1
    return t if t is not None and now <= t <= MAX_GW else None


def chat_chips(raw: Iterable[Any], gw: Any) -> list[dict[str, Any]]:
    """Фишки из роутера v4 -> [{chip, gw}] (коды bboost / 3xc / wildcard / freehit)."""
    now = _gw_number(gw) or 1
    out: list[dict[str, Any]] = []
    for c in raw:
        get = c.get if isinstance(c, dict) else (lambda k, c=c: getattr(c, k, None))
        code = chat.chip_code(get("chip"))
        g = _gw_number(get("gw"))
        if code is None:
            continue
        item = {"chip": code, "gw": g if g is not None and now <= g <= MAX_GW else None}
        if item not in out:
            out.append(item)
    return out


def _iso(dt: datetime | None) -> str | None:
    return None if dt is None else dt.astimezone(UTC).isoformat(timespec="seconds")


def _parse_dt(s: str | None) -> datetime:
    if not s:
        return datetime.now(UTC)
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _fmt_dt(s: str | None) -> str:
    if not s:
        return "n/a"
    return f"{_parse_dt(s):%Y-%m-%d %H:%M}Z"


def _compact_args(inp: Any) -> dict[str, Any]:
    if isinstance(inp, BaseModel):
        data = inp.model_dump(mode="json")
    elif isinstance(inp, dict):
        data = dict(inp)
    else:
        return {"value": str(inp)}
    return {k: v for k, v in data.items() if v not in (None, [], {}, "")}


def _price(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw.replace(",", "."))
    except ValueError:
        return None
    return value if 3.0 <= value <= 20.0 else None  # цены FPL: £3.5–£15+


def _gw_number(raw: Any) -> int | None:
    """Номер тура / число туров от роутера: int в [1, MAX_GW], иначе None (bool — не число)."""
    if isinstance(raw, bool) or not isinstance(raw, int | float | str):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if 1 <= value <= MAX_GW else None


def ranking_window_from_text(query: str) -> dict[str, int | None]:
    """Окно туров по тексту вопроса (RU / EN): диапазон «GW3–GW5» / «с 3 по 5 тур», «последние
    N туров» / «last N gameweeks» (-> last_n_gws), «в прошлом туре» (-> 1), номера «gw4 и gw5»
    (-> от min до max). Чистая функция; всё None — окна нет (весь сезон)."""
    none: dict[str, int | None] = {"gw_from": None, "gw_to": None, "last_n_gws": None}
    m = RANK_GW_RANGE.search(query)
    if m:
        nums = [int(g) for g in m.groups() if g]
        lo, hi = min(nums), max(nums)
        if 1 <= lo and hi <= MAX_GW:
            return {"gw_from": lo, "gw_to": hi, "last_n_gws": None}
    m = RANK_LAST_N.search(query)
    if m:
        n = int(m.group(1) or m.group(2))
        return {**none, "last_n_gws": n} if 1 <= n <= MAX_GW else none
    if RANK_LAST_ONE.search(query):
        return {**none, "last_n_gws": 1}
    gws = sorted(
        {
            int(g)
            for mm in RANK_GW.finditer(query)
            for g in mm.groups()
            if g and 1 <= int(g) <= MAX_GW
        }
    )
    if gws:
        return {"gw_from": gws[0], "gw_to": gws[-1], "last_n_gws": None}
    return none


def ranking_params(draft: Any, query: str) -> dict[str, Any]:
    """Параметры player_ranking: поля роутера v2 (`ranking`) проверяются кодом, пустые —
    достраиваются regex-страховкой по тексту вопроса (RU / EN). Чистая функция.
    Окно туров: явное gw_from / gw_to главнее last_n_gws; «последние N» не пересчитываются в
    номера здесь — это делает инструмент по последнему завершённому туру bootstrap."""
    get = (
        (lambda k: getattr(draft, k, None))
        if draft is not None and not isinstance(draft, dict)
        else (lambda k: (draft or {}).get(k))
    )
    metric = get("metric")
    gw_from, gw_to = _gw_number(get("gw_from")), _gw_number(get("gw_to"))
    last_n = _gw_number(get("last_n_gws"))
    if gw_from is None and gw_to is None and last_n is None:
        window = ranking_window_from_text(query)
        gw_from, gw_to, last_n = window["gw_from"], window["gw_to"], window["last_n_gws"]
    if gw_from is not None or gw_to is not None:  # явное окно главнее «последних N»
        lo = gw_from if gw_from is not None else gw_to
        hi = gw_to if gw_to is not None else gw_from
        gw_from, gw_to, last_n = min(lo, hi), max(lo, hi), None  # type: ignore[type-var]
    windowed = gw_from is not None or last_n is not None
    # «дешевле 4.5» — граница цены, а не просьба о дешёвых игроках: вырезаем до поиска слов
    stripped = RANK_MIN_PRICE.sub(" ", RANK_MAX_PRICE.sub(" ", query))
    words = {m for m, rx in RANK_METRIC_WORDS if rx.search(stripped)}
    if metric not in RANKING_METRICS:
        # «кто был лучшим в gw4 и gw5» без слов о цене/стабильности — это очки за окно, а не £/pts
        default = "total_points" if windowed else RANK_DEFAULT_METRIC
        metric = next((m for m, _ in RANK_METRIC_WORDS if m in words), default)
    if {"points_per_million", "consistency"} <= words and metric in (
        "points_per_million",
        "consistency",
    ):
        # «дешёвый, но стабильный»: обе оси названы явно -> составная метрика, даже если LLM
        # выбрал одну из них (аналог HORIZON_HINT: текст вопроса — детерминированная страховка)
        metric = "budget_consistency"
    position = str(get("position") or "").upper() or None
    if position not in POSITIONS:
        position = next((p for p, rx in RANK_POSITION_WORDS if rx.search(query)), None)
    max_price = _price(str(get("max_price"))) if get("max_price") is not None else None
    if max_price is None:
        m = RANK_MAX_PRICE.search(query)
        max_price = _price(m.group(1)) if m else None
    min_price = _price(str(get("min_price"))) if get("min_price") is not None else None
    if min_price is None:
        m = RANK_MIN_PRICE.search(query)
        min_price = _price(m.group(1)) if m else None
    if min_price is not None and max_price is not None and min_price > max_price:
        min_price = None
    limit = get("limit")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        m = RANK_LIMIT.search(query)
        limit = int(m.group(1) or m.group(2)) if m else RANK_DEFAULT_LIMIT
    limit = max(1, min(RANK_LIMIT_MAX, int(limit)))
    return {
        "metric": metric,
        "position": position,
        "max_price": max_price,
        "min_price": min_price,
        "limit": limit,
        "gw_from": gw_from,
        "gw_to": gw_to,
        "last_n_gws": last_n,
    }


def call_tool(
    entries: list[dict[str, Any]], node: str, name: str, fn: Callable[[Any], Any], inp: Any
) -> Any:
    """Вызов инструмента с записью в tool_log (латентность, ок/ошибка). Ошибка -> None."""
    started = time.perf_counter()
    try:
        out = fn(inp)
        ok, note = True, None
    except Exception as exc:  # инструмент не должен валить весь граф: оговорка в ответе
        log.warning("tool %s failed in %s: %s", name, node, exc, exc_info=True)
        out, ok, note = None, False, f"{type(exc).__name__}: {exc}"[:300]
    entries.append(
        {
            "node": node,
            "tool": name,
            "args": _compact_args(inp),
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "ok": ok,
            "note": note,
        }
    )
    return out


def llm_entry(node: str, usage: LLMUsage, *, purpose: str) -> dict[str, Any]:
    return {
        "node": node,
        "purpose": purpose,
        "model": usage.model,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "latency_ms": usage.latency_ms,
        "cost_usd": usage.cost_usd,
        "prompt_version": usage.prompt_version,
    }


def _dump(model: BaseModel | None) -> dict[str, Any] | None:
    return None if model is None else model.model_dump(mode="json", by_alias=True)


def squad_ids(state: AgentState) -> list[int]:
    summary = state.get("squad_summary") or {}
    return [int(p["id"]) for p in (summary.get("squad") or [])]


def squad_tool(fn: Callable[..., Any], state: AgentState) -> Callable[[Any], Any]:
    """Инструмент уровня состава с учётом `squad_override` из состояния (скриншот вместо picks).
    Без override возвращает fn как есть — сигнатура фейков в тестах не меняется."""
    raw = state.get("squad_override")
    if not raw:
        return fn
    override = SquadOverride.model_validate(raw)
    return lambda inp: fn(inp, squad_override=override)


# ---------- узлы ----------


def make_nodes(deps: Deps) -> dict[str, Callable[[AgentState], dict[str, Any]]]:
    tools = deps.tools

    # 1. контекст тура и состав
    def load_context(state: AgentState) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        caveats = list(state.get("caveats") or [])
        ctx = call_tool(
            entries,
            "load_context",
            "get_gameweek_context",
            squad_tool(tools.get_gameweek_context, state),
            GameweekContextInput(manager_id=state.get("manager_id"), strategy=state["strategy"]),
        )
        if ctx is None and (state.get("manager_id") is not None or state.get("squad_override")):
            # Состав требует БД (входы оптимизатора), а тур и дедлайн — только bootstrap (дисковый
            # кэш FPL API): повторяем без менеджера, чтобы compute не работал с gw=1 и рейтинг
            # лиги считал начавшиеся туры правильно даже при лежащем Postgres.
            note = entries[-1]["note"] or "context unavailable"
            ctx = call_tool(
                entries,
                "load_context",
                "get_gameweek_context",
                tools.get_gameweek_context,
                GameweekContextInput(manager_id=None, strategy=state["strategy"]),
            )
            if ctx is not None:
                entries[-1]["note"] = "fallback without squad: gameweek context from bootstrap only"
                kind = "database" if "OperationalError" in note or "SQLAlchemy" in note else "data"
                caveat = (
                    f"Squad unavailable ({kind} error): squad-level facts are not shown; the "
                    "gameweek context comes from the FPL API."
                )
                return {
                    "gw": ctx.gw,
                    "current_gw": ctx.current_gw,
                    "deadline": _iso(ctx.deadline),
                    "as_of": _iso(ctx.as_of),
                    "squad_summary": None,
                    "squad_note": f"squad unavailable ({note}); gameweek context from the FPL API",
                    "issues": [],
                    # Оговорка без сырого текста исключения: имена классов ошибок валидатор
                    # принимает за неизвестных игроков; техническая причина — в squad_note/tool_log
                    "caveats": caveats + [caveat],
                    "tool_log": entries,
                }
        if ctx is None:
            return {
                "as_of": _iso(deps.clock()),
                "squad_summary": None,
                "squad_note": "gameweek context unavailable (FPL API/DB error)",
                "caveats": caveats + [entries[-1]["note"] or "context unavailable"],
                "tool_log": entries,
            }
        issues = list(ctx.issues)
        summary = None
        if ctx.squad is not None:
            diag = call_tool(
                entries,
                "load_context",
                "diagnose_squad",
                squad_tool(tools.diagnose_squad, state),
                DiagnoseSquadInput(
                    manager_id=int(state.get("manager_id") or 0),
                    gw=ctx.gw,
                    strategy=state["strategy"],
                ),
            )
            if diag is not None:
                issues = list(diag.issues)
            summary = {
                "manager_id": state.get("manager_id"),
                "squad_gw": ctx.squad_gw,
                "bank": ctx.bank,
                "free_transfers": ctx.free_transfers,
                "chips_available": ctx.chips_available,
                "squad": [p.model_dump(mode="json") for p in ctx.squad],
            }
            if ctx.squad_note:
                caveats.append(ctx.squad_note)
        return {
            "gw": ctx.gw,
            "current_gw": ctx.current_gw,
            "deadline": _iso(ctx.deadline),
            "as_of": _iso(ctx.as_of),
            "squad_summary": summary,
            "squad_note": ctx.squad_note,
            "issues": issues,
            "caveats": caveats,
            "tool_log": entries,
        }

    # 2. маршрутизация: LLM классифицирует, код разрешает имена
    def router(state: AgentState) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        started = time.perf_counter()
        req = RouterRequest(
            query=state["query"],
            as_of=_fmt_dt(state.get("as_of")),
            gw=state.get("gw"),
            deadline=_fmt_dt(state.get("deadline")),
            has_manager=state.get("squad_summary") is not None,
            strategy=state["strategy"],
            history=chat.render_history(state.get("history")),
        )
        try:
            out, usage = deps.router_llm(req)
        except Exception as exc:
            log.exception("router LLM failed")
            entries.append(
                {
                    "node": "router",
                    "tool": "router_llm",
                    "args": {"query": state["query"]},
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                    "ok": False,
                    "note": f"{type(exc).__name__}: {exc}"[:300],
                }
            )
            return {
                "intent": "error",
                "answer": (
                    "Sorry — the request could not be routed "
                    f"({type(exc).__name__}). Please try again."
                ),
                "tool_log": entries,
            }
        entries.append(
            {
                "node": "router",
                "tool": "router_llm",
                "args": {"query": state["query"], "model": usage.model},
                "latency_ms": usage.latency_ms,
                "ok": True,
                "note": out.reason,
            }
        )
        # v4: незнакомый интент — частичный ответ (general_fpl), а не отказ; v1–v3 — как раньше
        fallback_intent = "general_fpl" if chat_v4(deps) else "off_topic"
        intent = out.intent if out.intent in INTENTS else fallback_intent
        # v4: уточнение по истории переписано роутером в самостоятельный вопрос — имена, горизонт
        # и фильтры ищем в нём (исходная реплика «а он сколько наберёт?» имён не содержит)
        query = str(getattr(out, "standalone_query", "") or "").strip() or state["query"]
        resolver = deps.resolver(squad_ids(state))
        mentions = list(dict.fromkeys(out.player_mentions))
        players, ambiguous, unknown, notes = resolver.resolve_all(mentions, query)
        scenario: dict[str, Any] = {
            "sell": [],
            "buy": [],
            "keep": [],
            "allow_hit": out.scenario.allow_hit,
            "use_wildcard": out.scenario.use_wildcard,
        }
        roles: dict[str, list[str]] = {}  # неоднозначное упоминание -> роли sell/buy/keep
        for key in ("sell", "buy", "keep"):
            for mention in getattr(out.scenario, key):
                res_players, res_amb, res_unk, _ = resolver.resolve_all([mention], mention)
                if res_players:
                    pid = res_players[0]["id"]
                    if pid not in scenario[key]:
                        scenario[key].append(pid)
                    if pid not in {p["id"] for p in players}:
                        players.append(res_players[0])
                for a in res_amb:
                    roles.setdefault(a.mention, [])
                    if key not in roles[a.mention]:
                        roles[a.mention].append(key)
                ambiguous += [a for a in res_amb if a.mention not in {x.mention for x in ambiguous}]
                unknown += [u for u in res_unk if u.mention not in {x.mention for x in unknown}]
        caveats = list(state.get("caveats") or [])
        caveats += [n for n in notes if n not in caveats]
        not_in_squad: list[str] = []
        if chat_v4(deps) and state.get("squad_summary") is not None:
            # «не продавать X» / «продать X» применяются только к игрокам состава; иначе —
            # оговорка на языке вопроса, и X не считается ни игроком состава, ни темой ответа
            own = set(squad_ids(state))
            lang_now = normalize_language(getattr(out, "language", None), state["query"])
            for role in ("keep", "sell"):
                for pid in [x for x in scenario[role] if x not in own]:
                    scenario[role].remove(pid)
                    name = next((p["name"] for p in players if p["id"] == pid), str(pid))
                    not_in_squad.append(name)
                    caveats.append(chat.not_in_squad_note(name, role, lang_now))
                    if pid not in scenario["buy"]:
                        players = [p for p in players if p["id"] != pid]
        for u in unknown:
            caveats.append(u.note or f"'{u.mention}' not found")
        # Горизонт от LLM принимается только если в вопросе он действительно назван (число или
        # слова про период); иначе — дефолт интента (роутер v1 склонен писать 1 вместо null,
        # v2 обязан отдавать null — регулярка остаётся страховкой).
        horizon = DEFAULT_HORIZON.get(intent, 3)
        target_gw = chat_target_gw(getattr(out, "target_gw", None), state.get("gw"))
        if intent == "gw_review":
            # разбор — только завершённый тур: названный пользователем или последний завершённый
            raw = _gw_number(getattr(out, "target_gw", None))
            last = _gw_number(state.get("current_gw"))
            target_gw = raw if raw is not None and last is not None and raw <= last else last
        chips_asked = chat_chips(getattr(out, "chips", None) or [], state.get("gw"))
        if chat_v4(deps) and intent == "lineup" and any(
            c["chip"] in chat.PLAN_CHIPS and c.get("gw") for c in chips_asked
        ):
            # состав под Bench Boost / Triple Captain в туре N — это план с фишкой, не XI без неё
            intent = "plan"
            horizon = DEFAULT_HORIZON["plan"]
        if out.horizon and HORIZON_HINT.search(query):
            horizon = int(out.horizon)
        # «в gw7» — номер тура, а не период «1 тур»: горизонт должен дойти до целевого тура и до
        # тура каждой фишки (BB в GW7 при ближайшем GW6 -> минимум 2 тура)
        gw_now = int(state.get("gw") or 1)
        reach = [g for g in [target_gw, *(c.get("gw") for c in chips_asked)] if g]
        if reach and intent in ("plan", "transfer", "what_if", "chips", "fixtures"):
            if target_gw and out.horizon == 1:
                horizon = DEFAULT_HORIZON.get(intent, 3)
            explicit = bool(out.horizon and out.horizon > 1 and HORIZON_HINT.search(query))
            if intent == "plan" and not explicit:
                horizon = max(reach) - gw_now + 1  # «состав к GW7» = план ровно до GW7
            horizon = max(horizon, max(reach) - gw_now + 1)
        horizon = max(1, min(MAX_HORIZON, horizon))
        # Язык ответа: код от роутера v2 (поле language), иначе детектор по тексту вопроса.
        language = normalize_language(getattr(out, "language", None), state["query"])
        # Параметры ранжирования лиги: поле роутера v2 + regex-страховка по тексту.
        ranking = (
            ranking_params(getattr(out, "ranking", None), query)
            if intent == "player_ranking"
            else None
        )
        draft = getattr(out, "ranking", None)
        if ranking is not None and getattr(draft, "metric", None) in FORECAST_METRICS:
            # «кто БУДЕТ лучшим»: прогноз xPts на горизонт; окно прошлых туров не применяется
            ranking.update(
                {
                    "metric": draft.metric,
                    "forecast": True,
                    "horizon_gws": max(
                        1,
                        min(
                            MAX_HORIZON,
                            int(getattr(draft, "horizon_gws", None) or out.horizon or 3),
                        ),
                    ),
                    "gw_from": None,
                    "gw_to": None,
                    "last_n_gws": None,
                }
            )
        if ranking is not None and getattr(draft, "max_ownership", None):
            ranking["max_ownership"] = float(draft.max_ownership)
        clarification = None
        if ambiguous and intent not in ("betting", "off_topic"):
            clarification = {
                "mentions": [
                    {
                        "mention": a.mention,
                        "note": a.note,
                        "candidates": a.candidates,
                        "roles": roles.get(a.mention, []),
                    }
                    for a in ambiguous
                ]
            }
        tracing.tag_root_run(
            [f"intent:{intent}", f"lang:{language}"],
            {"intent": intent, "horizon": horizon, "language": language},
        )
        return {
            "intent": intent,
            "router_raw": out.model_dump(mode="json"),
            "players": players,
            "unresolved": [u.mention for u in unknown],
            "clarification": clarification,
            "horizon": horizon,
            "language": language,
            "scenario": scenario,
            "ranking": ranking,
            "needs_squad": bool(out.needs_squad or intent in SQUAD_INTENTS),
            "standalone_query": query if query != state["query"] else None,
            "target_gw": target_gw,
            "chips_asked": chips_asked,
            "team_mentions": list(getattr(out, "team_mentions", None) or []),
            "not_in_squad": not_in_squad,
            "caveats": caveats,
            "tool_log": entries,
            "llm_calls": [llm_entry("router", usage, purpose="route")],
        }

    # 3. отказ: доменный guardrail (ставки) и off-topic — без LLM
    def refuse(state: AgentState) -> dict[str, Any]:
        intent = state.get("intent")
        if state.get("answer"):
            return {}
        # На языке вопроса (роутер v2+ отдаёт language, иначе детектор по тексту)
        answer = chat.refusal_text(
            intent, state.get("language") or normalize_language(None, state["query"])
        )
        return {
            "answer": answer,
            "tool_log": [
                {
                    "node": "refuse",
                    "tool": "-",
                    "args": {"intent": intent},
                    "latency_ms": 0,
                    "ok": True,
                    "note": "deterministic refusal, no LLM",
                }
            ],
        }

    # 3b. уточнение неоднозначного имени — HITL: вопрос с кандидатами, прерывание перед
    #     resolve_clarification; resume(thread_id, player_id=...) продолжает граф с выбранным игроком
    def clarify(state: AgentState) -> dict[str, Any]:
        clar = state.get("clarification") or {}
        mentions = list(clar.get("mentions") or [])
        if not mentions:  # нечего уточнять (страховка): пустой pending -> resolve пропустит
            return {"pending_clarification": None, "clarification_choice": None}
        item = mentions[0]
        candidates = [
            {
                "player_id": int(c["id"]),
                "name": c.get("name"),
                "full_name": c.get("full_name"),
                "team": c.get("team"),
                "position": c.get("position"),
                "price": c.get("price"),
                "ownership": c.get("ownership"),
                "status": c.get("status"),
                "in_squad": bool(c.get("in_squad")),
            }
            for c in item.get("candidates", [])[:6]
        ]
        pending = {
            "mention": item["mention"],
            "note": item.get("note"),
            "roles": list(item.get("roles") or []),
            "candidates": candidates,
        }
        return {
            "pending_clarification": pending,
            "clarification_choice": None,
            "answer": clarification_text(pending, state.get("language")),
            "tool_log": [
                {
                    "node": "clarify",
                    "tool": "-",
                    "args": {"mention": item["mention"], "candidates": len(candidates)},
                    "latency_ms": 0,
                    "ok": True,
                    "note": "ambiguous player name -> waiting for the user's choice (no LLM)",
                }
            ],
        }

    def resolve_clarification(state: AgentState) -> dict[str, Any]:
        """Узел-«человек» №2: сюда попадаем после resume с `clarification_choice`."""
        pending = state.get("pending_clarification") or {}
        choice = state.get("clarification_choice") or {}
        decision = choice.get("decision") or ("choose" if choice.get("player_id") else "cancel")
        clar = dict(state.get("clarification") or {})
        remaining = [
            m for m in clar.get("mentions") or [] if m.get("mention") != pending.get("mention")
        ]
        history = list(state.get("clarification_history") or [])
        caveats = list(state.get("caveats") or [])
        players = list(state.get("players") or [])
        scenario = {
            k: list(v) if isinstance(v, list) else v
            for k, v in (state.get("scenario") or {}).items()
        }
        picked: dict[str, Any] | None = None
        if decision == "choose":
            pid = int(choice.get("player_id") or 0)
            picked = next(
                (c for c in pending.get("candidates") or [] if c["player_id"] == pid), None
            )
            if picked is None:  # resume проверяет кандидата заранее; здесь — страховка
                decision = "cancel"
        history.append(
            {
                "mention": pending.get("mention"),
                "decision": decision,
                "player_id": choice.get("player_id"),
            }
        )
        if decision != "choose" or picked is None:
            return {
                "pending_clarification": None,
                "clarification": None,
                "clarification_choice": {"decision": "cancel"},
                "clarification_history": history,
                "answer": (state.get("answer") or "")
                + "\n\n"
                + (
                    "Уточнение отменено — задайте вопрос с полным именем игрока."
                    if state.get("language") == "ru"
                    else "Clarification cancelled — ask again with the player's full name."
                ),
                "tool_log": [
                    {
                        "node": "resolve_clarification",
                        "tool": "human",
                        "args": {"mention": pending.get("mention"), "decision": "cancel"},
                        "latency_ms": 0,
                        "ok": True,
                        "note": "cancelled by the user",
                    }
                ],
            }
        player = {
            "id": picked["player_id"],
            "name": picked.get("name"),
            "full_name": picked.get("full_name"),
            "team": picked.get("team"),
            "position": picked.get("position"),
            "price": picked.get("price"),
            "status": picked.get("status"),
            "ownership": picked.get("ownership"),
            "in_squad": picked.get("in_squad"),
        }
        if player["id"] not in {p["id"] for p in players}:
            players.append(player)
        for role in pending.get("roles") or []:
            scenario.setdefault(role, [])
            if player["id"] not in scenario[role]:
                scenario[role].append(player["id"])
        caveats.append(
            f"'{pending.get('mention')}' clarified by you as {player.get('full_name') or player['name']}"
            f" ({player.get('team')})."
        )
        return {
            "players": players,
            "scenario": scenario,
            "pending_clarification": None,
            "clarification": {"mentions": remaining} if remaining else None,
            "clarification_choice": {"decision": "choose", "player_id": player["id"]},
            "clarification_history": history,
            "caveats": caveats,
            "answer": None,  # вопрос-уточнение больше не ответ: граф продолжает к сигналам
            "tool_log": [
                {
                    "node": "resolve_clarification",
                    "tool": "human",
                    "args": {"mention": pending.get("mention"), "player_id": player["id"]},
                    "latency_ms": 0,
                    "ok": True,
                    "note": f"{pending.get('mention')} -> {player.get('full_name') or player['name']}"
                    + (f"; {len(remaining)} more to clarify" if remaining else ""),
                }
            ],
        }

    # 4. сигналы новостей: свежие из БД, иначе извлечение
    def _relevant_players(state: AgentState) -> list[int]:
        ids: list[int] = [int(p["id"]) for p in state.get("players") or []]
        sc = state.get("scenario") or {}
        for key in ("sell", "buy", "keep"):
            ids += [int(x) for x in sc.get(key) or []]
        if state.get("intent") in SIGNAL_SQUAD_INTENTS:
            issues = sorted(state.get("issues") or [], key=lambda i: -int(i.get("severity", 0)))
            ids += [int(i["player_id"]) for i in issues if i.get("kind") in ISSUE_KINDS_FOR_SIGNALS]
        return list(dict.fromkeys(ids))[: deps.max_signal_players]

    def _flatten_evidence(signals: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for sig in signals.values():
            out.extend(sig.get("evidence") or [])
            out.extend(form_evidence(sig.get("form_notes") or [], str(sig.get("player", ""))))
        return out

    def _risk_calls(
        node: str, pids: Iterable[int], state: AgentState, *, force: bool, mode: str, k: int
    ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], bool]:
        entries: list[dict[str, Any]] = []
        llm_calls: list[dict[str, Any]] = []
        signals = dict(state.get("signals") or {})
        fresh = False
        as_of = _parse_dt(state.get("as_of"))
        for pid in pids:
            risk = call_tool(
                entries,
                node,
                "analyze_player_risk",
                tools.analyze_player_risk,
                PlayerRiskInput(
                    player_id=pid,
                    as_of=as_of,
                    max_age_h=deps.signal_max_age_h,
                    force=force,
                    mode=mode,
                    k=k,
                ),
            )
            if risk is None:
                continue
            entries[-1]["note"] = (
                f"{risk.origin}: {risk.availability} conf {risk.confidence:.2f} "
                f"evidence {len(risk.evidence)}" + (f" ({risk.note})" if risk.note else "")
            )
            signals[str(pid)] = risk.model_dump(mode="json")
            signals[str(pid)]["form_notes"] = load_form_notes(pid, risk.signal_as_of)
            if risk.origin == "extracted":
                fresh = True
            if risk.llm_calls:
                llm_calls.append(
                    {
                        "node": node,
                        "purpose": "extract_signal",
                        "model": risk.model or settings.rag_llm_model,
                        "prompt_tokens": risk.prompt_tokens,
                        "completion_tokens": risk.completion_tokens,
                        "latency_ms": risk.latency_ms,
                        "cost_usd": estimate_cost(
                            risk.model or settings.rag_llm_model,
                            risk.prompt_tokens,
                            risk.completion_tokens,
                        ),
                        "prompt_version": "rag",
                    }
                )
        return signals, entries, llm_calls, fresh

    def ensure_signals(state: AgentState) -> dict[str, Any]:
        relevant = _relevant_players(state)
        signals, entries, llm_calls, fresh = _risk_calls(
            "ensure_signals", relevant, state, force=False, mode="hybrid_rerank", k=8
        )
        return {
            "relevant_players": relevant,
            "signals": signals,
            "evidence": _flatten_evidence(signals),
            "fresh_signals": bool(state.get("fresh_signals")) or fresh,
            "tool_log": entries,
            "llm_calls": llm_calls,
        }

    # 5. грейдер достаточности сигнала: детерминированное правило + (опционально) LLM
    def grade_signals(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        signals = state.get("signals") or {}
        insufficient: list[int] = []
        reasons: dict[str, str] = {}
        llm_calls: list[dict[str, Any]] = []
        for pid in state.get("relevant_players") or []:
            sig = signals.get(str(pid))
            if sig is None:
                continue
            if sig.get("fpl_status") not in SUSPICIOUS_STATUSES:
                continue  # доступный по FPL игрок: сигнал желателен, но не обязателен
            unknown = sig.get("availability") == "unknown" or bool(sig.get("abstained"))
            weak = float(sig.get("confidence") or 0.0) < LOW_CONFIDENCE
            if not unknown and not weak:
                continue
            if unknown or not sig.get("evidence") or deps.grader_llm is None:
                insufficient.append(pid)
                reasons[str(pid)] = (
                    "no evidence / availability unknown"
                    if unknown or not sig.get("evidence")
                    else f"confidence {float(sig.get('confidence') or 0):.2f} < {LOW_CONFIDENCE}"
                )
                continue
            try:
                verdict, usage = deps.grader_llm(
                    GradeRequest(
                        player=sig.get("player", str(pid)),
                        fpl_status=sig.get("fpl_status", "a"),
                        fpl_chance=sig.get("fpl_chance"),
                        fpl_news=sig.get("fpl_news", ""),
                        availability=sig.get("availability", "unknown"),
                        confidence=float(sig.get("confidence") or 0.0),
                        summary=sig.get("summary", ""),
                        evidence=[
                            {k: e.get(k) for k in ("source", "date", "quote")}
                            for e in sig.get("evidence") or []
                        ],
                        next_gw=state.get("gw"),
                    )
                )
                llm_calls.append(llm_entry("grade_signals", usage, purpose="grade"))
            except Exception as exc:
                log.warning("grader failed for %s: %s", pid, exc, exc_info=True)
                insufficient.append(pid)
                reasons[str(pid)] = f"grader error: {type(exc).__name__}"
                continue
            if not verdict.sufficient:
                insufficient.append(pid)
                reasons[str(pid)] = f"grader: {verdict.reason}"
        retries = int(state.get("retries") or 0)
        grade = {
            "iteration": retries,
            "insufficient": insufficient,
            "reasons": reasons,
            "sufficient": not insufficient,
            "will_retry": bool(insufficient) and retries < deps.max_retries,
        }
        names = {str(p["id"]): p["name"] for p in state.get("players") or []}
        return {
            "grade": grade,
            "tool_log": [
                {
                    "node": "grade_signals",
                    "tool": "grader" if llm_calls else "deterministic_rule",
                    "args": {
                        "iteration": retries,
                        "checked": len(state.get("relevant_players") or []),
                    },
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                    "ok": True,
                    "note": (
                        "insufficient: "
                        + ", ".join(
                            f"{names.get(str(p), p)} ({reasons[str(p)]})" for p in insufficient
                        )
                        if insufficient
                        else "all signals sufficient"
                    ),
                }
            ],
            "llm_calls": llm_calls,
        }

    # 5b. повторный поиск с альтернативной стратегией (ограниченный цикл)
    def rewrite_retry(state: AgentState) -> dict[str, Any]:
        retries = int(state.get("retries") or 0)
        mode, k = RETRY_STRATEGIES[min(retries, len(RETRY_STRATEGIES) - 1)]
        targets = [int(p) for p in (state.get("grade") or {}).get("insufficient", [])]
        signals, entries, llm_calls, fresh = _risk_calls(
            "rewrite_retry", targets, state, force=True, mode=mode, k=k
        )
        for e in entries:
            e["args"] = {**e["args"], "retry": retries + 1, "strategy": f"{mode} k={k}"}
        return {
            "signals": signals,
            "evidence": _flatten_evidence(signals),
            "fresh_signals": bool(state.get("fresh_signals")) or fresh,
            "retries": retries + 1,
            "tool_log": entries
            or [
                {
                    "node": "rewrite_retry",
                    "tool": "-",
                    "args": {"retry": retries + 1, "strategy": f"{mode} k={k}"},
                    "latency_ms": 0,
                    "ok": True,
                    "note": "nothing to retry",
                }
            ],
            "llm_calls": llm_calls,
        }

    # 6. детерминированные вычисления по интенту
    def _player_facts(pred: dict[str, Any], sig: dict[str, Any] | None) -> dict[str, Any]:
        p = pred["player"]
        by_gw = pred.get("by_gw") or []
        nxt = by_gw[0] if by_gw else None
        facts: dict[str, Any] = {
            "team": p["team"],
            "position": p["position"],
            "price": p["price"],
            "fpl_status": p["status"],
            "fpl_chance_pct": p.get("chance"),
            "fpl_news": p.get("news") or "",
            "ownership_pct": round(float(p.get("ownership") or 0.0), 1),
            "xpts_by_gw": {f"GW{g['gw']}": g["xpts"] for g in by_gw},
            f"total_xpts_{len(by_gw)}gw": pred.get("total_xpts"),
            "fixtures": {
                f"GW{g['gw']}": " ".join(
                    f"{'v' if f['is_home'] else '@'}{f['opponent']} (FSI {f['fsi']})"
                    for f in g.get("fixtures") or []
                )
                or "blank"
                for g in by_gw
            },
        }
        if nxt is not None:
            facts["next_gw"] = {
                "gw": nxt["gw"],
                "xpts": nxt["xpts"],
                "sd": nxt["sd"],
                "p_start": nxt["p_start"],
                "exp_minutes": nxt["exp_minutes"],
                "components": nxt.get("components"),
                "fpl_ep_next": nxt.get("ep_next"),
                "notes": nxt.get("notes"),
            }
        if sig is not None:
            facts["news_signal"] = {
                "availability": sig.get("availability"),
                "start_probability": sig.get("start_probability"),
                "expected_minutes": sig.get("expected_minutes"),
                "rotation_risk": sig.get("rotation_risk"),
                "return_gw": sig.get("return_gw"),
                "confidence": sig.get("confidence"),
                "summary": sig.get("summary"),
                "signal_as_of": sig.get("signal_as_of"),
                "origin": sig.get("origin"),
                "evidence_count": len(sig.get("evidence") or []),
            }
            if sig.get("form_notes"):
                facts["form_and_context"] = {
                    "about": FORM_ABOUT,
                    "notes": form_notes_brief(sig["form_notes"]),
                }
        return facts

    def _with_club_pos(names: list[str]) -> list[str]:
        """«Raya» -> «Raya (ARS, GKP)» по bootstrap (у фейков тестов bootstrap может не быть)."""
        bs = getattr(tools, "bootstrap", None)
        if bs is None:
            return list(names)
        by_name: dict[str, Any] = {}
        for p in bs.elements:
            by_name.setdefault(p.web_name, p)
        out = []
        for n in names:
            p = by_name.get(n)
            out.append(f"{n} ({bs.team(p.team).short_name}, {p.position.short})" if p else n)
        return out

    def _team_xpts(entries: list[dict[str, Any]], names: list[str], gw: int) -> dict[str, Any]:
        """{«Raya (ARS, GKP)»: «4.39 xPts, @NFO (FSI 1)»} на тур gw — прогноз модели, без LLM."""
        bs = getattr(tools, "bootstrap", None)
        if bs is None or not names:
            return {}
        started = time.perf_counter()
        by_name: dict[str, Any] = {}
        for p in bs.elements:
            by_name.setdefault(p.web_name, p)
        out: dict[str, Any] = {}
        for n in names:
            p = by_name.get(n)
            if p is None:
                continue
            try:
                pred = tools.predict_player(PredictPlayerInput(player_id=p.id, gw=gw, horizon=1))
            except Exception:  # один игрок без прогноза не ломает ответ
                log.debug("team xpts: no prediction for %s GW%s", n, gw, exc_info=True)
                continue
            if pred.by_gw:
                g = pred.by_gw[0]
                fx = ", ".join(
                    f"{'v' if f.is_home else '@'}{f.opponent} (FSI {f.fsi})" for f in g.fixtures
                ) or "blank"
                out[f"{n} ({bs.team(p.team).short_name}, {p.position.short})"] = (
                    f"{g.xpts} xPts, {fx}"
                )
        entries.append(
            {
                "node": "compute",
                "tool": "predict_player",
                "args": {"players": len(names), "gw": gw},
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "ok": True,
                "note": f"xPts of the plan's squad for GW{gw}: {len(out)} players",
            }
        )
        return out

    def _v4_tool(entries: list[dict[str, Any]], name: str, inp: Any) -> Any:
        """Инструмент чата v4, если он есть у набора инструментов (у фейков тестов — нет)."""
        fn = getattr(tools, name, None)
        if fn is None:
            return None
        return call_tool(entries, "compute", name, fn, inp)

    def _chips_status(
        entries: list[dict[str, Any]], state: AgentState, gw: int
    ) -> dict[str, Any] | None:
        if not state.get("manager_id"):
            return None
        out = _v4_tool(
            entries, "chips_status", ChipsStatusInput(manager_id=int(state["manager_id"]), gw=gw)
        )
        return chat.chips_facts(out) if out is not None else None

    def compute(state: AgentState) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        caveats = list(state.get("caveats") or [])
        intent = state.get("intent") or "strategy_question"
        gw = int(state.get("gw") or 1)
        horizon = int(state.get("horizon") or DEFAULT_HORIZON.get(intent, 3))
        strategy = state["strategy"]
        manager_id = state.get("manager_id")
        summary = state.get("squad_summary")
        sc = state.get("scenario") or {}
        signals = state.get("signals") or {}
        # покупки, исключённые узлом candidate_news после новости injured / suspended / unavailable
        news_exclude = [int(x) for x in state.get("news_exclude") or []]
        targets: list[dict[str, Any]] = []  # кого ответ рекомендует / ранжирует -> candidate_news
        if state.get("fresh_signals"):
            tools.invalidate()  # свежие сигналы должны попасть в модель минут / прогнозы
        facts: dict[str, Any] = {
            "question": state["query"],
            "intent": intent,
            "strategy": strategy,
            "gw": gw,
            "deadline": _fmt_dt(state.get("deadline")),
            "as_of": _fmt_dt(state.get("as_of")),
        }
        # Общий вопрос о правилах/стратегии или рейтинг всей лиги — не про состав: без блока
        # менеджера в фактах, чтобы объяснитель не подмешивал проблемы конкретного состава.
        if summary is not None and intent not in LEAGUE_WIDE_INTENTS:
            facts["manager"] = {
                "id": manager_id,
                "squad_from_gw": summary.get("squad_gw"),
                "bank": summary.get("bank"),
                "free_transfers": summary.get("free_transfers"),
                "chips_available": summary.get("chips_available"),
                "squad": [
                    f"{p['name']} ({p['team']}, {p['position']}, £{p['price']:.1f}"
                    + (", C" if p.get("is_captain") else "")
                    + (", bench" if not p.get("is_starting", True) else "")
                    + ")"
                    for p in summary.get("squad") or []
                ],
            }
            if state.get("issues"):
                facts["squad_issues"] = [
                    f"{i.get('name')}: {i.get('kind')} (severity {i.get('severity')}) — {i.get('detail')}"
                    for i in state["issues"]
                ]
        needs_squad = bool(state.get("needs_squad")) and intent in SQUAD_INTENTS
        squad_ok = summary is not None
        if needs_squad and not squad_ok:
            caveats.append(
                "Squad-level advice needs a manager id with at least one played gameweek "
                "(screenshot input of your team is coming in a later step); "
                f"{state.get('squad_note') or 'no squad available'}. "
                "Only the player-level part of the question is answered."
            )

        # игроки из запроса: прогноз на горизонт всегда
        predictions = dict(state.get("predictions") or {})
        facts_players: dict[str, Any] = {}
        for p in state.get("players") or []:
            pred = call_tool(
                entries,
                "compute",
                "predict_player",
                tools.predict_player,
                PredictPlayerInput(player_id=int(p["id"]), gw=gw, horizon=horizon),
            )
            if pred is None:
                caveats.append(f"prediction unavailable for {p['name']}")
                continue
            data = pred.model_dump(mode="json")
            predictions[str(p["id"])] = data
            facts_players[p["name"]] = _player_facts(data, signals.get(str(p["id"])))
        if facts_players and chat_v4(deps):
            # v4: компоненты прогноза — очки по источникам, а не голы / ассисты (объяснитель
            # подписывал «2.29 гола»): ключи переименованы без изменения чисел
            for pf in facts_players.values():
                nxt = pf.get("next_gw") or {}
                if isinstance(nxt.get("components"), dict):
                    nxt["xpts_components"] = {
                        f"xpts_from_{k}": v for k, v in nxt.pop("components").items()
                    }
        if facts_players:
            facts["players"] = facts_players
        # сигналы игроков состава с проблемами (не упомянутых в вопросе) — коротко
        others = {
            sig["player"]: {
                "fpl_status": sig.get("fpl_status"),
                "availability": sig.get("availability"),
                "confidence": sig.get("confidence"),
                "summary": sig.get("summary"),
            }
            for pid, sig in signals.items()
            if pid not in predictions
        }
        if others:
            facts["squad_news_signals"] = others

        lineup = state.get("lineup")
        routes = list(state.get("routes") or [])
        plan = state.get("plan")
        scenario_result = state.get("scenario_result")

        if intent == "compare_players" and len(state.get("players") or []) >= 2:
            cmp = call_tool(
                entries,
                "compute",
                "compare_players",
                tools.compare_players,
                ComparePlayersInput(
                    player_ids=[int(p["id"]) for p in state["players"]], gw=gw, horizon=horizon
                ),
            )
            if cmp is not None:
                facts["comparison_ranking"] = cmp.ranking
        if intent in ("compare_players", "player_status"):
            targets += [news_target(p["id"], p["name"], "asked") for p in state.get("players") or []]

        manager_id = manager_id if manager_id is not None else 0  # override без id менеджера
        if squad_ok and intent in ("captain", "lineup", "squad_review"):
            # v4: «состав на GW7» — XI текущего состава на целевой тур (без трансферов)
            xi_gw = int(state.get("target_gw") or gw) if intent != "squad_review" else gw
            lu = call_tool(
                entries,
                "compute",
                "optimize_team",
                squad_tool(tools.optimize_team, state),
                OptimizeTeamInput(manager_id=int(manager_id), gw=max(gw, xi_gw), strategy=strategy),
            )
            if state.get("chips_asked") and intent in ("captain", "lineup"):
                cs = _chips_status(entries, state, gw)
                if cs is not None:
                    facts["chips_status"] = cs
            if lu is not None:
                lineup = _dump(lu)
                facts["best_xi"] = {
                    "gw": lu.gw,
                    "formation": lu.formation,
                    "expected_points": lu.expected_points,
                    "captain": lu.captain,
                    "vice": lu.vice,
                    "current_captain_in_picks": lu.current_captain,
                    "current_xi_points": lu.current_xi_points,
                    "starters": [
                        f"{s.name} ({s.team}, {s.position}) {s.xpts} xPts, p_start {s.p_start}, "
                        f"{s.fixture}"
                        for s in lu.starters
                    ],
                    "bench": [f"{s.name} ({s.position}) {s.xpts}" for s in lu.bench],
                }
                facts["captain_options"] = [
                    {
                        "name": o.name,
                        "team": o.team,
                        "xpts": o.xpts,
                        "captain_points": o.captain_points,
                        "sd": o.sd,
                        "ownership_pct": o.ownership,
                        "tag": o.tag,
                        "fixture": o.fixture,
                        "p_start": o.p_start,
                    }
                    for o in lu.captain_options
                ]
                facts["captain_tag_rule"] = (
                    ">30% owned = safe, 10–30% = balanced, <10% = differential"
                )

        if intent == "transfer" and (state.get("target_gw") or 0) > gw:
            caveats.append(
                f"Transfers are made before the GW{gw} deadline; the route is evaluated over "
                f"GW{gw}–GW{gw + horizon - 1} (it includes GW{state['target_gw']})."
            )
        if squad_ok and intent in ("transfer", "squad_review"):
            rt = call_tool(
                entries,
                "compute",
                "recommend_transfers",
                squad_tool(tools.recommend_transfers, state),
                RecommendTransfersInput(
                    manager_id=int(manager_id),
                    gw=gw,
                    horizon=horizon,
                    strategy=strategy,
                    allow_hit=sc.get("allow_hit"),
                    sell=[int(x) for x in sc.get("sell") or []],
                    buy=[int(x) for x in sc.get("buy") or []],
                    keep=[int(x) for x in sc.get("keep") or []],
                    exclude=news_exclude,
                ),
            )
            if rt is not None:
                targets += route_targets(
                    rt.routes,
                    rt.recommended_rank,
                    forced_buy={int(x) for x in sc.get("buy") or []},
                    n_routes=deps.candidate_routes,
                    alternative=rt.alternative,
                )
                routes = [r.model_dump(mode="json", by_alias=True) for r in rt.routes]
                facts["transfers"] = {
                    "free_transfers": rt.free_transfers,
                    "bank": rt.bank,
                    "horizon_gws": horizon,
                    "allow_hit": sc.get("allow_hit"),
                    "baseline_xi_points_no_transfer": rt.baseline_xi_points,
                    "routes": [
                        {
                            "rank": r.rank,
                            "out": r.out,
                            "in": r.in_,
                            "hit_cost": r.hit_cost,
                            "gain_next_gw": r.gain_next_gw,
                            "gain_horizon": r.gain_horizon,
                            "gain_discounted": r.gain_discounted,
                            "hit_marginal_gain": r.hit_marginal_gain,
                            "xi_points_after_next_gw": r.xi_points_after,
                            "new_bank": r.new_bank,
                            "verdict": r.verdict,
                            "risk_note": r.risk_note,
                        }
                        for r in rt.routes
                    ],
                    "recommendation": rt.recommendation,
                    "constrained_by_question": rt.constrained,
                    "notes": rt.notes,
                }
                if rt.constrained:
                    forced = [p["name"] for p in state["players"] if p["id"] in sc.get("sell", [])]
                    who = ", ".join(forced) or "the named player"
                    if rt.alternative:
                        facts["transfers"]["best_route_without_forcing_the_sale"] = {
                            "keeps": forced,
                            "out": rt.alternative.out,
                            "in": rt.alternative.in_,
                            "hit_cost": rt.alternative.hit_cost,
                            "gain_next_gw": rt.alternative.gain_next_gw,
                            "gain_horizon": rt.alternative.gain_horizon,
                            "xi_points_after_next_gw": rt.alternative.xi_points_after,
                            "verdict": rt.alternative.verdict,
                            "note": f"optimizer's best use of the free transfers if {who} is kept",
                        }
                        # Детерминированный вердикт по принудительной продаже: сравнение выигрышей.
                        if rt.routes:
                            forced_gain = rt.routes[0].gain_horizon
                            alt_gain = rt.alternative.gain_horizon
                            diff = round(alt_gain - forced_gain, 2)
                            if diff > FORCED_SALE_TOLERANCE:
                                verdict = (
                                    f"keep {who}: the unconstrained route gains more "
                                    f"({alt_gain:+} vs {forced_gain:+} xPts over the horizon, "
                                    f"diff {diff:+})"
                                )
                            elif diff < -FORCED_SALE_TOLERANCE:
                                verdict = (
                                    f"sell {who}: the forced route gains more "
                                    f"({forced_gain:+} vs {alt_gain:+} xPts over the horizon)"
                                )
                            else:
                                verdict = (
                                    f"roughly a tie ({forced_gain:+} vs {alt_gain:+} over the "
                                    f"horizon, diff {diff:+}): selling {who} is fine but not "
                                    "required; the optimizer's free choice keeps them"
                                )
                            facts["transfers"]["verdict_on_forced_sale"] = verdict
                    else:
                        facts["transfers"]["best_route_without_forcing_the_sale"] = {
                            "note": "the optimizer's unconstrained best route makes the same "
                            "buys — the forced sale is also the free choice"
                        }
                        facts["transfers"]["verdict_on_forced_sale"] = (
                            f"sell {who}: the optimizer's free choice sells them too"
                        )

        if squad_ok and intent == "plan":
            # v4: фишки по турам из вопроса (BB / TC), запрет платных трансферов, «не продавать»
            plan_chips = [
                PlanChipIn(gw=int(c["gw"]), chip=c["chip"])
                for c in state.get("chips_asked") or []
                if c.get("chip") in chat.PLAN_CHIPS and c.get("gw")
            ]
            plan_inp = BuildPlanInput(
                manager_id=int(manager_id),
                gw=gw,
                horizon=horizon,
                strategy=strategy,
                use_wildcard=sc.get("use_wildcard"),
                keep=[int(x) for x in sc.get("keep") or []],
                exclude=news_exclude,
                allow_hits=sc.get("allow_hit") is not False,
                chips=plan_chips,
            )
            pl = call_tool(
                entries,
                "compute",
                "build_gameweek_plan",
                squad_tool(tools.build_gameweek_plan, state),
                plan_inp,
            )
            if pl is None and plan_chips and "ChipPlanError" in str(entries[-1].get("note")):
                # Фишку в этом туре сыграть нельзя (уже сыграна / окно / вне горизонта): причина —
                # в ответ пользователю, план считается без неё
                reason = str(entries[-1]["note"]).split(": ", 1)[-1]
                facts["chip_plan"] = {
                    "requested": [f"{chat.CHIP_NAMES[c.chip]} GW{c.gw}" for c in plan_chips],
                    "feasible": False,
                    "reason": reason,
                    "summary": f"requested {', '.join(f'{chat.CHIP_NAMES[c.chip]} GW{c.gw}' for c in plan_chips)} is NOT possible: {reason}",
                }
                caveats.append(
                    "The requested chip cannot be played — the plan below is computed without it."
                )
                pl = call_tool(
                    entries,
                    "compute",
                    "build_gameweek_plan",
                    squad_tool(tools.build_gameweek_plan, state),
                    plan_inp.model_copy(update={"chips": []}),
                )
            if plan_chips or sc.get("use_wildcard") or state.get("chips_asked"):
                cs = _chips_status(entries, state, gw)
                if cs is not None:
                    facts["chips_status"] = cs
            if pl is not None:
                if pl.chips:
                    facts.setdefault(
                        "chip_plan",
                        {
                            "feasible": True,
                            "summary": "; ".join(
                                f"{c.name} in GW{c.gw}: +{c.points} xPts"
                                + (f" ({', '.join(c.players)})" if c.players else "")
                                for c in pl.chips
                            ),
                        },
                    )
                    facts["chip_plan"]["chips"] = [
                        {"chip": c.name, "gw": f"GW{c.gw}", "chip_points": c.points,
                         "players": c.players}
                        for c in pl.chips
                    ]
                    last = pl.chips[-1]
                    # команда тура фишки после запланированных трансферов (не текущий состав)
                    facts["chip_plan"]["team_for_chip_gw"] = {
                        "gw": f"GW{last.gw}",
                        "squad_after_planned_transfers": (
                            pl.target_squad if last.gw == pl.from_gw + pl.horizon - 1 else None
                        ),
                        "captain": pl.captain_by_gw.get(str(last.gw)),
                        "bench_counted_by_bench_boost": (
                            last.players if last.chip == "bboost" else None
                        ),
                        "xi_xpts_with_chip": pl.xi_points_by_gw.get(str(last.gw)),
                    }
                targets += plan_targets(pl, gw)
                plan = _dump(pl)
                facts["plan"] = {
                    "gws": f"GW{pl.from_gw}–GW{pl.from_gw + pl.horizon - 1}",
                    "moves_by_gw": {
                        g: [
                            f"{m.out} (£{m.price_out:.1f}"
                            + (f", {m.out_problem}" if m.out_problem else "")
                            + f") -> {m.in_} (£{m.price_in:.1f}) Δ{m.delta_xpts_horizon:+.2f}"
                            + (" PAID -4" if m.paid else "")
                            for m in ms
                        ]
                        or ["roll (no transfer)"]
                        for g, ms in pl.moves_by_gw.items()
                    },
                    "hits_by_gw": pl.hits_by_gw,
                    "free_transfers_by_gw": pl.ft_by_gw,
                    "bank_by_gw": pl.bank_by_gw,
                    "xi_points_by_gw": pl.xi_points_by_gw,
                    "captain_by_gw": pl.captain_by_gw,
                    "expected_total": pl.expected_total,
                    "baseline_total_no_transfers": pl.baseline_total,
                    "gain_vs_baseline": round(pl.expected_total - pl.baseline_total, 2),
                    "recommendation": pl.recommendation,
                    "wildcard_alternative": pl.wildcard,
                    "diff_vs_previous_plan": pl.diff_vs_previous,
                    "previous_plan_at": pl.previous_plan_at,
                    "solver": pl.solver,
                    "runtime_s": pl.runtime_s,
                    "paid_transfers_allowed": pl.allow_hits,
                    "kept_as_asked": [
                        p["name"] for p in state.get("players") or [] if p["id"] in (sc.get("keep") or [])
                    ],
                    "squad_at_end_of_plan": _with_club_pos(pl.target_squad),
                    "squad_at_end_of_plan_is_for": f"GW{pl.from_gw + pl.horizon - 1}",
                }
                if chat_v4(deps) and (state.get("target_gw") or state.get("chips_asked")):
                    # xPts каждого игрока итогового состава на целевой тур: без них объяснитель
                    # заполнял таблицу «состава на GW7» чужими числами
                    team_xpts = _team_xpts(entries, pl.target_squad, pl.from_gw + pl.horizon - 1)
                    if team_xpts:
                        facts["plan"]["xpts_of_squad_at_end_of_plan"] = team_xpts
                caveats.append(
                    "Only the next gameweek's move is actionable; later moves are directional "
                    "and are re-planned every week."
                )

        if squad_ok and intent == "what_if":
            sr = call_tool(
                entries,
                "compute",
                "simulate_scenario",
                squad_tool(tools.simulate_scenario, state),
                SimulateScenarioInput(
                    manager_id=int(manager_id),
                    gw=gw,
                    horizon=horizon,
                    strategy=strategy,
                    sell=[int(x) for x in sc.get("sell") or []],
                    buy=[int(x) for x in sc.get("buy") or []],
                    keep=[int(x) for x in sc.get("keep") or []],
                    allow_hit=sc.get("allow_hit"),
                    use_wildcard=sc.get("use_wildcard"),
                    exclude=news_exclude,
                ),
            )
            if sr is not None:
                if sr.primary is not None:
                    targets += route_targets(
                        [sr.primary],
                        sr.primary.rank,
                        forced_buy={int(x) for x in sc.get("buy") or []},
                        n_routes=1,
                    )
                scenario_result = _dump(sr)
                facts["scenario"] = {
                    "asked": {
                        "sell": [
                            p["name"] for p in state["players"] if p["id"] in sc.get("sell", [])
                        ],
                        "buy": [
                            p["name"] for p in state["players"] if p["id"] in sc.get("buy", [])
                        ],
                        "allow_hit": sc.get("allow_hit"),
                        "use_wildcard": sc.get("use_wildcard"),
                    },
                    "feasible": sr.feasible,
                    "reason": sr.reason,
                    "free_transfers": sr.free_transfers,
                    "bank": sr.bank,
                    "transfers_cap": sr.transfers_cap,
                    "hold_xi_points": sr.hold_xi_points,
                    "primary": (
                        {
                            "out": sr.primary.out,
                            "in": sr.primary.in_,
                            "transfers": len(sr.primary.in_),
                            "hit_cost": sr.primary.hit_cost,
                            "gain_next_gw": sr.primary.gain_next_gw,
                            "gain_horizon": sr.primary.gain_horizon,
                            "gain_discounted": sr.primary.gain_discounted,
                            "hit_marginal_gain": sr.primary.hit_marginal_gain,
                            "xi_points_after_next_gw": sr.primary.xi_points_after,
                            "new_bank": sr.primary.new_bank,
                            "verdict": sr.primary.verdict,
                            "risk_note": sr.primary.risk_note,
                        }
                        if sr.primary
                        else None
                    ),
                    "free_alternative": (
                        {
                            "out": sr.free_alternative.out,
                            "in": sr.free_alternative.in_,
                            "hit_cost": sr.free_alternative.hit_cost,
                            "gain_next_gw": sr.free_alternative.gain_next_gw,
                            "gain_horizon": sr.free_alternative.gain_horizon,
                            "xi_points_after_next_gw": sr.free_alternative.xi_points_after,
                            "verdict": sr.free_alternative.verdict,
                        }
                        if sr.free_alternative
                        else None
                    ),
                    "hit_alternative_as_asked": (
                        {
                            "out": sr.hit_alternative.out,
                            "in": sr.hit_alternative.in_,
                            "transfers": len(sr.hit_alternative.in_),
                            "hit_cost": sr.hit_alternative.hit_cost,
                            "gain_next_gw": sr.hit_alternative.gain_next_gw,
                            "gain_horizon": sr.hit_alternative.gain_horizon,
                            "hit_marginal_gain_vs_free": sr.hit_alternative.hit_marginal_gain,
                            "verdict": sr.hit_alternative.verdict,
                            "risk_note": sr.hit_alternative.risk_note,
                        }
                        if sr.hit_alternative
                        else None
                    ),
                    "other_routes": [
                        {
                            "out": r.out,
                            "in": r.in_,
                            "hit_cost": r.hit_cost,
                            "gain_horizon": r.gain_horizon,
                        }
                        for r in sr.alternatives
                    ],
                    "wildcard": sr.wildcard,
                    "verdict": sr.verdict,
                    "notes": sr.notes,
                }

        # ---- v4: оценка состава, календарь, фишки, общий вопрос, прогнозный рейтинг ----
        if intent == "squad_review" and squad_ok:
            facts.pop("captain_options", None)
            facts.pop("captain_tag_rule", None)
            tr = facts.get("transfers") or {}
            rec = str(tr.get("recommendation") or "hold")
            routes_f = tr.get("routes") or []
            best = None
            if rec.startswith("route"):
                idx = int(rec.split()[-1]) - 1
                if 0 <= idx < len(routes_f):
                    best = routes_f[idx]
            xi = facts.get("best_xi") or {}
            facts["squad_review"] = {
                "issues": facts.get("squad_issues") or [],
                "best_xi_xpts": xi.get("expected_points"),
                "current_xi_xpts": xi.get("current_xi_points"),
                "best_xi_formation": xi.get("formation"),
                "free_transfers": (summary or {}).get("free_transfers"),
                "bank": (summary or {}).get("bank"),
                "horizon_gws": tr.get("horizon_gws"),
                "best_fix": (
                    {k: best.get(k) for k in ("out", "in", "gain_next_gw", "gain_horizon", "hit_cost")}
                    if best
                    else None
                ),
                "note": "a review from the model's numbers; there is no single 'team rating' score",
            }
            if xi.get("starters"):
                facts["squad_review"]["xi_by_xpts"] = xi["starters"]
        if intent == "fixtures" or (
            chat_v4(deps) and state.get("team_mentions") and intent == "general_fpl"
        ):
            teams = list(getattr(getattr(tools, "bootstrap", None), "teams", None) or [])
            team_ids, missing = chat.resolve_teams(state.get("team_mentions") or [], teams)
            # клуб игрока из вопроса («с кем играет команда Bogle»)
            for pl_ in state.get("players") or []:
                tid = next((t.id for t in teams if t.short_name == pl_.get("team")), None)
                if tid is not None and tid not in team_ids:
                    team_ids.append(tid)
            fx = _v4_tool(
                entries, "team_fixtures", TeamFixturesInput(gw=gw, horizon=horizon, team_ids=[])
            )
            if fx is not None:
                facts["fixtures"] = chat.fixtures_facts(fx, team_ids)
            for m in missing:
                caveats.append(f"club '{m}' not recognised")
        if intent in ("chips",) and squad_ok:
            cs = _chips_status(entries, state, gw)
            if cs is not None:
                facts["chips_status"] = cs
            asked = [
                c for c in state.get("chips_asked") or []
                if c.get("chip") in chat.PLAN_CHIPS and c.get("gw")
            ]
            if asked:
                # «можно ли BB в GW7?» — проверка через план с этой фишкой (правила — в оптимизаторе)
                pl_chk = call_tool(
                    entries,
                    "compute",
                    "build_gameweek_plan",
                    squad_tool(tools.build_gameweek_plan, state),
                    BuildPlanInput(
                        manager_id=int(manager_id or 0),
                        gw=gw,
                        horizon=max(horizon, max(int(c["gw"]) for c in asked) - gw + 1),
                        strategy=strategy,
                        allow_hits=sc.get("allow_hit") is not False,
                        keep=[int(x) for x in sc.get("keep") or []],
                        chips=[PlanChipIn(gw=int(c["gw"]), chip=c["chip"]) for c in asked],
                        save=False,
                    ),
                )
                if pl_chk is not None:
                    facts["chip_plan"] = {
                        "feasible": True,
                        "summary": "; ".join(
                            f"{c.name} in GW{c.gw}: +{c.points} xPts"
                            + (f" ({', '.join(c.players)})" if c.players else "")
                            for c in pl_chk.chips
                        ),
                        "plan_expected_total": pl_chk.expected_total,
                    }
                else:
                    reason = str(entries[-1].get("note") or "").split(": ", 1)[-1]
                    facts["chip_plan"] = {
                        "feasible": False,
                        "reason": reason,
                        "summary": f"requested chip is NOT possible: {reason}",
                    }
        if intent == "gw_review" and state.get("manager_id"):
            review_gw = int(state.get("target_gw") or state.get("current_gw") or max(1, gw - 1))
            rv = _v4_tool(
                entries,
                "review_gameweek",
                GWReviewInput(manager_id=int(state["manager_id"]), gw=review_gw),
            )
            if rv is not None:
                facts["gw_review"] = chat.gw_review_facts(rv)
            else:
                facts["gw_review"] = {
                    "gw": f"GW{review_gw}",
                    "available": False,
                    "reason": str(entries[-1].get("note") or "review data unavailable")[:200],
                }
            # разбор прошлого тура — не про текущий состав: без проблем состава и трансферов
            facts.pop("squad_issues", None)
        if intent == "general_fpl":
            facts["gameweek"] = chat.gameweek_facts(state)
            facts["price_changes"] = chat.price_facts()
            trends = _v4_tool(entries, "transfer_trends", TransferTrendsInput(limit=8))
            if trends is not None:
                facts["transfer_trends"] = chat.trends_facts(trends)
            if squad_ok:
                cs = _chips_status(entries, state, gw)
                if cs is not None:
                    facts["chips_status"] = cs
            facts["what_i_can_compute"] = (
                "player fitness and expected points, transfers with hit verdicts, captain, "
                "starting XI (also for a future gameweek), multi-gameweek plans with chips, club "
                "fixtures (FSI), league rankings (past points or forecast xPts), FPL rules"
            )

        rk_now = state.get("ranking") or {}
        if intent == "player_ranking" and rk_now.get("forecast"):
            fr = _v4_tool(
                entries,
                "rank_forecast",
                ForecastRankInput(
                    gw=gw,
                    horizon=int(rk_now.get("horizon_gws") or 3),
                    metric=rk_now.get("metric"),
                    position=rk_now.get("position"),
                    max_price=rk_now.get("max_price"),
                    min_price=rk_now.get("min_price"),
                    max_ownership=rk_now.get("max_ownership"),
                    limit=int(rk_now.get("limit") or RANK_DEFAULT_LIMIT),
                    squad_ids=squad_ids(state),
                ),
            )
            if fr is not None:
                facts["forecast_ranking"] = chat.forecast_ranking_facts(fr)
                targets += [
                    news_target(r.id, r.name, f"ranked #{r.rank}")
                    for r in fr.rows[: deps.candidate_ranking_top]
                ]
                caveats.append(
                    "Ranking is a FORECAST of the xPts model for the upcoming gameweeks, not "
                    "past points."
                )
        if intent == "player_ranking" and not rk_now.get("forecast"):
            # Рейтинг всей лиги: детерминированный расчёт по bootstrap + player_gw_history
            # (rank_players), параметры — из роутера v2 / regex-страховки (state.ranking).
            rk = state.get("ranking") or ranking_params(None, state["query"])
            ranking_out = call_tool(
                entries,
                "compute",
                "rank_players",
                tools.rank_players,
                RankPlayersInput(
                    gw=gw,
                    metric=rk.get("metric") or RANK_DEFAULT_METRIC,
                    position=rk.get("position"),
                    max_price=rk.get("max_price"),
                    min_price=rk.get("min_price"),
                    limit=int(rk.get("limit") or RANK_DEFAULT_LIMIT),
                    squad_ids=squad_ids(state),
                    gw_from=rk.get("gw_from"),
                    gw_to=rk.get("gw_to"),
                    last_n_gws=rk.get("last_n_gws"),
                ),
            )
            if ranking_out is not None:
                facts["player_ranking"] = ranking_facts(ranking_out)
                targets += [
                    news_target(r.id, r.name, f"ranked #{r.rank}")
                    for r in ranking_out.rows[: deps.candidate_ranking_top]
                ]
                entries[-1]["note"] = (
                    f"{ranking_out.metric}, {ranking_out.window_label}: top "
                    f"{len(ranking_out.rows)} of {ranking_out.candidates} candidates"
                    + (
                        f"; history through GW{ranking_out.history_through_gw}"
                        if ranking_out.history_through_gw
                        else ""
                    )
                )
                caveats += ranking_out.notes
                min_minutes = (ranking_out.filters or {}).get("min_minutes")
                scope = (
                    "season totals to date"
                    if ranking_out.gw_from is None
                    else f"points scored in {ranking_out.window_label} only"
                )
                caveats.append(
                    f"Ranking uses {scope} (past points, not a forecast); consistency = share of "
                    "played gameweeks with >= 5 points and the spread of points; "
                    + (
                        f"only players with at least {min_minutes} minutes played are included."
                        if min_minutes is not None
                        else "players with few minutes are filtered out."
                    )
                )
            else:
                reason = str(entries[-1].get("note") or "FPL API / DB error").split(": ", 1)[-1]
                caveats.append(f"Player ranking unavailable ({reason}) — no numbers were invented.")

        llm_calls: list[dict[str, Any]] = []
        if intent in ("strategy_question", "general_fpl"):
            # Стратегическая KB (rag/kb): цитируемый ответ с маркерами [n]; единственный
            # LLM-вызов внутри инструмента учитывается в llm_calls как kb_answer.
            kb_answer = call_tool(
                entries,
                "compute",
                "answer_strategy_question",
                tools.answer_strategy_question,
                StrategyAnswerInput(
                    query=state.get("standalone_query") or state["query"], k=KB_ANSWER_K
                ),
            )
            if kb_answer is not None:
                facts["strategy_answer"] = strategy_answer_facts(kb_answer)
                entries[-1]["note"] = (
                    f"{'covered' if kb_answer.covered else 'not covered'}, "
                    f"{len(kb_answer.citations)} citations, {kb_answer.retrieved} chunks"
                )
                if kb_answer.llm_calls:
                    llm_calls.append(
                        {
                            "node": "compute",
                            "purpose": "kb_answer",
                            "model": kb_answer.model or settings.rag_llm_model,
                            "prompt_tokens": kb_answer.prompt_tokens,
                            "completion_tokens": kb_answer.completion_tokens,
                            "latency_ms": kb_answer.latency_ms,
                            "cost_usd": (
                                kb_answer.cost_usd
                                if kb_answer.cost_usd is not None
                                else estimate_cost(
                                    kb_answer.model or settings.rag_llm_model,
                                    kb_answer.prompt_tokens,
                                    kb_answer.completion_tokens,
                                )
                            ),
                            "prompt_version": f"kb:{kb_answer.prompt_version or 'v1'}",
                        }
                    )
                if not kb_answer.covered and intent == "general_fpl":
                    facts.pop("strategy_answer", None)  # ответ — из данных тура, не из KB
                elif not kb_answer.covered:
                    caveats.append(
                        "The strategy knowledge base does not cover this question; no rule was "
                        "invented — ask about your squad for tool-based numbers."
                    )
                else:
                    caveats.append(
                        "Strategy answer comes from the cited knowledge base (official rules pages "
                        "are tagged rules; the rest is community guidance)."
                    )
            else:  # KB недоступна (БД / OpenAI): встроенная выжимка правил как запасной вариант
                facts["rules_digest"] = load_agent_prompt("rules_digest", deps.prompt_version)
                caveats.append(
                    "Strategy knowledge base unavailable — answered from the built-in rules "
                    "digest (no citations)."
                )

        # Контекст правил для решений о хите / чипе: 1–2 чанка KB (теги hits / chips) в фактах —
        # объяснитель цитирует их как [source]; числа по-прежнему только из оптимизатора.
        if deps.rules_context and intent in ("transfer", "plan", "what_if"):
            kind = rules_context_kind(intent, facts, sc, gw)
            if kind is not None:
                query, tags = rules_context_query(kind, state)
                kb_hits = call_tool(
                    entries,
                    "compute",
                    "search_strategy_kb",
                    tools.search_strategy_kb,
                    KBSearchInput(query=query, tags=list(tags), k=RULES_CONTEXT_K),
                )
                if kb_hits is not None and kb_hits.chunks:
                    facts["rules_context"] = rules_context_facts(kind, query, kb_hits)
                    entries[-1]["note"] = f"{kind}: {len(kb_hits.chunks)} chunk(s) " + ", ".join(
                        sorted({c.source for c in kb_hits.chunks})
                    )

        if not entries:
            entries.append(
                {
                    "node": "compute",
                    "tool": "-",
                    "args": {"intent": intent},
                    "latency_ms": 0,
                    "ok": True,
                    "note": "no deterministic tool for this intent",
                }
            )
        if chat_v4(deps) and state.get("unresolved"):
            facts["players_not_found"] = list(state["unresolved"])
        if state.get("not_in_squad"):
            facts["not_in_your_squad"] = list(state["not_in_squad"])
        facts["headline"] = headline(intent, facts, state)
        if facts.get("not_in_your_squad"):
            names = ", ".join(facts["not_in_your_squad"])
            facts["headline"] = (
                f"NOTE: {names} is NOT in your squad — the keep / sell condition was not applied; "
                "never present him as a squad member. " + facts["headline"]
            )
        if facts.get("players_not_found"):
            names = ", ".join(f"'{m}'" for m in facts["players_not_found"])
            facts["headline"] = (
                f"NOTE: {names} is not in the FPL player list this season — no data for them. "
                + facts["headline"]
            )
        if intent == "squad_review" and facts.get("squad_review"):
            facts["headline"] += (
                f" (review of the squad for the upcoming GW{gw}; reviews of past gameweeks are not "
                "available)"
            )
        if facts.get("chip_plan") and intent == "plan":
            facts["headline"] += f"; CHIP: {facts['chip_plan']['summary']}"
        if intent == "plan" and isinstance(facts.get("manager"), dict):
            # план меняет состав: текущий — только как «до плана», чтобы его не выдали за итог
            facts["manager"]["current_squad_before_plan"] = facts["manager"].pop("squad", [])
        return {
            "predictions": predictions,
            "lineup": lineup,
            "routes": routes,
            "plan": plan,
            "scenario_result": scenario_result,
            "facts": facts,
            "caveats": list(dict.fromkeys(caveats)),
            "fresh_signals": False,
            "news_targets": dedupe_targets(targets),
            "news_reoptimize": False,
            "tool_log": entries,
            "llm_calls": llm_calls,
        }

    # 6b. новости по игрокам, которых рекомендует / ранжирует / сравнивает ответ, и по их клубам.
    #     Кандидаты известны только после compute; бюджет LLM-извлечений на запрос ограничен.
    def _extract_entry(node: str, purpose: str, out: Any) -> dict[str, Any]:
        model = out.model or settings.rag_llm_model
        return {
            "node": node,
            "purpose": purpose,
            "model": model,
            "prompt_tokens": out.prompt_tokens,
            "completion_tokens": out.completion_tokens,
            "latency_ms": out.latency_ms,
            "cost_usd": estimate_cost(model, out.prompt_tokens, out.completion_tokens),
            "prompt_version": "rag",
        }

    def candidate_news(state: AgentState) -> dict[str, Any]:
        node = "candidate_news"
        intent = state.get("intent")
        targets = list(state.get("news_targets") or [])
        if not deps.candidate_news or intent not in NEWS_INTENTS or not targets:
            return {
                "news_reoptimize": False,
                "tool_log": [
                    {
                        "node": node,
                        "tool": "-",
                        "args": {"intent": intent},
                        "latency_ms": 0,
                        "ok": True,
                        "note": "no recommended / ranked players to fetch news for",
                    }
                ],
            }
        as_of = _parse_dt(state.get("as_of"))
        gw = int(state.get("gw") or 1)
        horizon = int(state.get("horizon") or DEFAULT_HORIZON.get(intent or "", 3))
        signals = state.get("signals") or {}
        cand = dict(state.get("candidate_signals") or {})
        clubs = dict(state.get("team_news") or {})
        used = int(state.get("news_llm_used") or 0)
        budget = deps.news_max_llm_calls
        entries: list[dict[str, Any]] = []
        llm_calls: list[dict[str, Any]] = []
        fresh = False
        not_refreshed: list[str] = []
        team_fn = getattr(tools, "team_news", None)  # фейки без метода — без клубного контекста
        fetched = 0
        for t in targets:
            key = str(t["id"])
            sig = signals.get(key) or cand.get(key)
            if sig is None and fetched < deps.candidate_max_players:
                fetched += 1
                cached_only = used >= budget
                risk = call_tool(
                    entries,
                    node,
                    "analyze_player_risk",
                    tools.analyze_player_risk,
                    PlayerRiskInput(
                        player_id=int(t["id"]),
                        as_of=as_of,
                        max_age_h=deps.signal_max_age_h,
                        cached_only=cached_only,
                        mode=deps.news_mode,
                        k=8,
                    ),
                )
                if risk is not None:
                    entries[-1]["note"] = (
                        f"{t['role']}: {risk.origin}: {risk.availability} conf "
                        f"{risk.confidence:.2f} evidence {len(risk.evidence)}"
                        + (" (budget: saved only)" if cached_only else "")
                    )
                    sig = cand[key] = risk.model_dump(mode="json")
                    sig["form_notes"] = load_form_notes(int(t["id"]), risk.signal_as_of)
                    if risk.llm_calls:
                        used += 1
                        llm_calls.append(_extract_entry(node, "extract_signal", risk))
                    fresh = fresh or risk.origin == "extracted"
                    stale = risk.origin != "cached" or (risk.age_h or 0) > deps.signal_max_age_h
                    if cached_only and stale:
                        not_refreshed.append(str(t["name"]))
            tid = (sig or {}).get("team_id")
            if (
                t.get("club_news")
                and tid is not None
                and team_fn is not None
                and str(tid) not in clubs
                and len(clubs) < deps.team_news_max_clubs
            ):
                cached_only = used >= budget
                club = call_tool(
                    entries,
                    node,
                    "team_news",
                    team_fn,
                    TeamNewsInput(
                        team_id=int(tid),
                        as_of=as_of,
                        max_age_h=deps.signal_max_age_h,
                        cached_only=cached_only,
                        mode=deps.news_mode,
                    ),
                )
                if club is not None:
                    entries[-1]["note"] = (
                        f"{club.team}: {club.origin}, {len(club.items)} item(s), "
                        f"{len(club.absences)} absence(s)"
                        + (" (budget: saved only)" if cached_only else "")
                    )
                    clubs[str(tid)] = club.model_dump(mode="json")
                    if club.llm_calls:
                        used += 1
                        llm_calls.append(_extract_entry(node, "team_news", club))
        preds: dict[str, dict[str, Any]] = {}
        known_preds = state.get("predictions") or {}
        for t in targets:
            if len(preds) >= deps.candidate_max_players:
                break
            # рейтинг — прошлые очки, не прогноз: xPts только для покупок оптимизатора
            if not str(t["role"]).startswith("buy") or str(t["id"]) in known_preds:
                continue
            pred = call_tool(
                entries,
                node,
                "predict_player",
                tools.predict_player,
                PredictPlayerInput(player_id=int(t["id"]), gw=gw, horizon=horizon),
            )
            if pred is not None:
                preds[str(t["id"])] = pred.model_dump(mode="json")

        all_sigs = {**cand, **signals}
        facts = dict(state.get("facts") or {})
        cn = candidate_news_facts(targets, all_sigs, preds, facts)
        if cn["players"]:
            facts["candidate_news"] = cn
        if clubs:
            facts["team_news"] = team_news_facts(clubs)
        caveats = list(state.get("caveats") or [])
        caveats += news_caveats(targets, all_sigs, not_refreshed, budget)
        bad = unavailable_primary_buys(targets, all_sigs, deps.news_exclude_min_confidence)
        exclude = [int(x) for x in state.get("news_exclude") or []]
        reopt = (
            bool(bad)
            and deps.news_reoptimize
            and not state.get("news_reoptimized")
            and intent in ("transfer", "plan")
        )
        for t in bad:
            sig = all_sigs[str(t["id"])]
            what = (
                f"{t['name']} ({t['role']}): the news signal says {sig.get('availability')} "
                f"(confidence {sig.get('confidence')}, signal {_fmt_dt(sig.get('signal_as_of'))})"
            )
            if reopt:
                exclude.append(int(t["id"]))
                caveats.append(f"{what} — excluded, the optimizer re-ran without him.")
            else:
                caveats.append(f"{what} — treat this recommendation with caution.")
        evidence = _flatten_evidence(signals)
        for key, sig in cand.items():
            if key not in signals:  # ≤ 2 цитаты на кандидата: ответ и промпт не раздуваются
                evidence.extend((sig.get("evidence") or [])[:CANDIDATE_EVIDENCE_MAX])
                notes = sig.get("form_notes") or []
                evidence.extend(form_evidence(notes, str(sig.get("player", "")))[:1])
        evidence += club_evidence(clubs)
        entries.append(
            {
                "node": node,
                "tool": "-",
                "args": {"targets": len(targets), "budget": budget},
                "latency_ms": 0,
                "ok": True,
                "note": f"news extractions used {used}/{budget}"
                + (f"; re-optimize without {[t['name'] for t in bad]}" if reopt else ""),
            }
        )
        return {
            "candidate_signals": cand,
            "team_news": clubs,
            "news_llm_used": used,
            "facts": facts,
            "evidence": evidence,
            "caveats": list(dict.fromkeys(caveats)),
            "fresh_signals": bool(state.get("fresh_signals")) or fresh,
            "news_exclude": sorted(set(exclude)),
            "news_reoptimize": reopt,
            "news_reoptimized": bool(state.get("news_reoptimized")) or reopt,
            "tool_log": entries,
            "llm_calls": llm_calls,
        }

    # 7. действие, требующее подтверждения человеком (хит / wildcard) — см. action_for()
    def check_action(state: AgentState) -> dict[str, Any]:
        action = action_for(
            state.get("intent"),
            state.get("facts") or {},
            state.get("scenario") or {},
            state.get("gw"),
        )
        caveats = list(state.get("caveats") or [])
        pending = None
        if action is not None and state.get("user_decision") is None:
            pending = action
            note = (
                f"needs confirmation: {action['kind']} ({action['detail']}, cost {action['cost']})"
            )
        elif action is not None:
            note = f"{action['kind']} remains after decision '{state.get('user_decision')}'"
            if state.get("user_decision") == "reject":
                caveats.append(
                    "You rejected paid transfers / wildcard, but the requested scenario still "
                    f"requires a {action['kind']} ({action['detail']}); shown as-is."
                )
        else:
            note = "no hit / wildcard in the primary recommendation"
        return {
            "pending_action": pending,
            "caveats": caveats,
            "tool_log": [
                {
                    "node": "check_action",
                    "tool": "-",
                    "args": {"intent": state.get("intent")},
                    "latency_ms": 0,
                    "ok": True,
                    "note": note,
                }
            ],
        }

    def confirm_action(state: AgentState) -> dict[str, Any]:
        """Узел-«человек»: граф прерывается ПЕРЕД ним; сюда попадаем после resume с решением."""
        decision = state.get("user_decision") or "reject"  # без решения — консервативно
        action = state.get("pending_action") or {}
        sc = dict(state.get("scenario") or {})
        caveats = list(state.get("caveats") or [])
        history = list(state.get("action_history") or [])
        history.append({"action": action, "decision": decision})
        if decision == "reject":
            sc["allow_hit"] = False
            sc["use_wildcard"] = False
            caveats.append(
                f"You rejected the {action.get('kind', 'action')} "
                f"({action.get('detail', '')}); recomputed without paid transfers / wildcard."
            )
        else:
            caveats.append(
                f"You confirmed the {action.get('kind', 'action')}: {action.get('detail', '')}"
                + (f" (cost {action.get('cost')} points)" if action.get("cost") else "")
                + "."
            )
        return {
            "user_decision": decision,
            "scenario": sc,
            "action_history": history,
            "caveats": caveats,
            "tool_log": [
                {
                    "node": "confirm_action",
                    "tool": "human",
                    "args": {"decision": decision, "action": action},
                    "latency_ms": 0,
                    "ok": True,
                    "note": "recompute without hit/wildcard" if decision == "reject" else "kept",
                }
            ],
        }

    # 8. объяснение
    def explain(state: AgentState) -> dict[str, Any]:
        attempts = int(state.get("explain_attempts") or 0)
        validation = state.get("validation") or {}
        feedback = validation.get("feedback") if attempts > 0 else None
        facts = dict(state.get("facts") or {})
        if state.get("user_decision"):
            history = state.get("action_history") or []
            sc = state.get("scenario") or {}
            facts["user_decision"] = {
                "decision": state["user_decision"],
                "on": history[-1] if history else None,
                "scenario_after_decision": {
                    "allow_hit": sc.get("allow_hit"),
                    "use_wildcard": sc.get("use_wildcard"),
                },
            }
        req = ExplainRequest(
            query=state.get("standalone_query") or state["query"],
            original_query=state["query"],
            history=chat.render_history(state.get("history")),
            intent=state.get("intent") or "unknown",
            strategy=state["strategy"],
            as_of=_fmt_dt(state.get("as_of")),
            facts=facts,
            evidence=state.get("evidence") or [],
            caveats=state.get("caveats") or [],
            feedback=feedback,
            language=state.get("language") or "en",
        )
        started = time.perf_counter()
        try:
            out, usage = deps.explain_llm(req)
        except Exception as exc:
            log.exception("explain LLM failed")
            # Без LLM: вердикт кода и оговорки на языке вопроса (сырые факты — в футере UI)
            fallback = chat.explain_fallback_text(
                (state.get("facts") or {}).get("headline"),
                state.get("caveats") or [],
                state.get("language"),
            )
            return {
                "answer": fallback,
                "explain_attempts": attempts + 1,
                "tool_log": [
                    {
                        "node": "explain",
                        "tool": "explain_llm",
                        "args": {"attempt": attempts + 1},
                        "latency_ms": round((time.perf_counter() - started) * 1000),
                        "ok": False,
                        "note": f"{type(exc).__name__}: {exc}"[:300],
                    }
                ],
            }
        return {
            "answer": out.answer_markdown.strip(),
            "explain_attempts": attempts + 1,
            "tool_log": [
                {
                    "node": "explain",
                    "tool": "explain_llm",
                    "args": {
                        "attempt": attempts + 1,
                        "model": usage.model,
                        "feedback": bool(feedback),
                    },
                    "latency_ms": usage.latency_ms,
                    "ok": True,
                    "note": f"{len(out.answer_markdown)} chars",
                }
            ],
            "llm_calls": [llm_entry("explain", usage, purpose="explain")],
        }

    # 9. детерминированная проверка ответа
    def validate_answer_node(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        answer = fix_mixed_script(state.get("answer") or "")  # «О'Shea» с кириллической «О»
        names = [p["name"] for p in state.get("players") or []] + [
            p.get("full_name", "") for p in state.get("players") or []
        ]
        summary = state.get("squad_summary") or {}
        names += [p["name"] for p in summary.get("squad") or []]
        names += deps.extra_known_names
        result = validate_answer(
            answer,
            state.get("facts") or {},
            # оговорки собирает код (squad = picks GW5 и т.п.) — их туры и числа допустимы
            [*(state.get("evidence") or []), {"caveats": list(state.get("caveats") or [])}],
            extra_names=names,
        )
        attempts = int(state.get("explain_attempts") or 0)
        history = list((state.get("validation") or {}).get("history") or [])
        history.append(result.as_dict())
        validation: dict[str, Any] = {
            **result.as_dict(),
            "attempts": attempts,
            "feedback": result.feedback if not result.passed else None,
            "history": history,
            "regenerate": (not result.passed) and attempts < MAX_EXPLAIN_ATTEMPTS,
        }
        out: dict[str, Any] = {"validation": validation}
        if answer != (state.get("answer") or ""):
            out["answer"] = answer
        final_pass = result.passed or attempts >= MAX_EXPLAIN_ATTEMPTS
        if final_pass:
            # Детерминированная зачистка «Sources» от новостей игроков, которых ответ не
            # обсуждает (слабость промпта v1 №1; в A/B v2 промпт один раз её повторил).
            pruned, removed = prune_undiscussed_sources(
                answer, state.get("evidence") or [], state.get("facts") or {}
            )
            if removed:
                answer = pruned
                validation["pruned_sources"] = removed
                out["answer"] = answer
        if not result.passed and attempts >= MAX_EXPLAIN_ATTEMPTS:
            final = (
                strip_bad_citations(answer, result.unknown_refs) if result.bad_citations else answer
            )
            final = restore_latin_names(final, result.cyrillic_names)
            ratings = [n for n in result.unknown_numbers if "10" in n and not n.replace(".", "").isdigit()]
            if ratings:  # «5 out of 10» — оценку не считал ни один инструмент: убрать фразу
                final = strip_ratings(final, ratings)
            final, typos = fix_name_typos(final, result.unknown_names, names)
            if result.not_in_squad_claims:  # «X в составе», хотя X в составе нет — убрать утверждение
                final = strip_lines(
                    final,
                    not_in_squad_lines(final, state.get("facts") or {})[1],
                    result.not_in_squad_claims,
                )
            flagged = [n for n in result.unknown_names if n not in typos] + [
                n for n in result.unknown_numbers if n not in ratings
            ] + list(getattr(result, "unknown_gws", []) or []) + [
                f"{m['player']} {m['number']}" for m in result.misattributed_numbers
            ]
            if flagged:
                # Оговорка на языке вопроса; технические детали — в футере «Как получен ответ»
                final += chat.validation_note(state.get("language"), flagged)
            out["answer"] = final
        out["tool_log"] = [
            {
                "node": "validate_answer",
                "tool": "validate_answer",
                "args": {"attempt": attempts},
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "ok": True,
                "note": (
                    "passed"
                    if result.passed
                    else f"violations: names {result.unknown_names} numbers {result.unknown_numbers}"
                    f" citations {result.bad_citations}"
                    + (" kb markers missing" if result.missing_refs else "")
                    + (" hit mismatch" if result.hit_mismatch else "")
                    + (f" cyrillic {result.cyrillic_names}" if result.cyrillic_names else "")
                    + (
                        " misattributed "
                        + str([f"{m['player']} {m['number']}" for m in result.misattributed_numbers])
                        if result.misattributed_numbers
                        else ""
                    )
                    + (" -> regenerate" if validation["regenerate"] else " -> caveat added")
                )
                + (
                    f"; pruned {len(validation['pruned_sources'])} undiscussed source line(s)"
                    if validation.get("pruned_sources")
                    else ""
                ),
            }
        ]
        return out

    return {
        "load_context": load_context,
        "router": router,
        "refuse": refuse,
        "clarify": clarify,
        "resolve_clarification": resolve_clarification,
        "ensure_signals": ensure_signals,
        "grade_signals": grade_signals,
        "rewrite_retry": rewrite_retry,
        "compute": compute,
        "candidate_news": candidate_news,
        "check_action": check_action,
        "confirm_action": confirm_action,
        "explain": explain,
        "validate_answer": validate_answer_node,
    }


# ---------- HITL: платное действие и уточнение имени (чистые функции) ----------


def action_for(
    intent: str | None, facts: dict[str, Any], scenario: dict[str, Any], gw: int | None
) -> dict[str, Any] | None:
    """Хит или Wildcard в ОСНОВНОЙ рекомендации -> {kind, detail, cost}; иначе None.
    Используется и check_action (нужно ли подтверждение), и compute (нужен ли rules_context)."""
    if intent == "transfer":
        tr = facts.get("transfers") or {}
        rec = tr.get("recommendation", "hold")
        if rec.startswith("route"):
            idx = int(rec.split()[-1]) - 1
            routes = tr.get("routes") or []
            if 0 <= idx < len(routes) and routes[idx].get("hit_cost", 0) > 0:
                r = routes[idx]
                return {
                    "kind": "hit",
                    "detail": f"{', '.join(r['out'])} -> {', '.join(r['in'])} (GW{gw})",
                    "cost": r["hit_cost"],
                }
    elif intent == "what_if":
        scn = facts.get("scenario") or {}
        primary = scn.get("primary")
        if primary and primary.get("hit_cost", 0) > 0:
            return {
                "kind": "hit",
                "detail": f"{', '.join(primary['out'])} -> {', '.join(primary['in'])} (GW{gw})",
                "cost": primary["hit_cost"],
            }
        wc = scn.get("wildcard")
        if wc and scenario.get("use_wildcard"):
            return {
                "kind": "wildcard",
                "detail": f"play the Wildcard in GW{gw} ({len(wc.get('moves') or [])} changes)",
                "cost": 0,
            }
    elif intent == "plan":
        pl = facts.get("plan") or {}
        if pl.get("recommendation") == "wildcard":
            return {"kind": "wildcard", "detail": f"play the Wildcard in GW{gw}", "cost": 0}
        hit = int((pl.get("hits_by_gw") or {}).get(str(gw), 0) or 0)
        if hit > 0:
            moves = (pl.get("moves_by_gw") or {}).get(str(gw), [])
            return {"kind": "hit", "detail": f"GW{gw}: " + "; ".join(moves), "cost": hit}
    return None


def rules_context_kind(
    intent: str, facts: dict[str, Any], scenario: dict[str, Any], gw: int | None
) -> str | None:
    """О чём нужен контекст правил: 'hit' / 'wildcard', если основная рекомендация содержит хит
    или чип ЛИБО пользователь явно спросил про хит / Wildcard (тогда объяснение «хит не нужен»
    тоже опирается на правило); иначе None."""
    action = action_for(intent, facts, scenario, gw)
    if action is not None:
        return str(action["kind"])
    if scenario.get("use_wildcard") is True or (facts.get("plan") or {}).get(
        "wildcard_alternative"
    ):
        return "wildcard"
    if scenario.get("allow_hit") is True or (facts.get("scenario") or {}).get(
        "hit_alternative_as_asked"
    ):
        return "hit"
    return None


def rules_context_query(kind: str, state: AgentState) -> tuple[str, tuple[str, ...]]:
    """Запрос к KB, собранный из ситуации (не из текста пользователя), и теги фильтра."""
    if kind == "wildcard":
        serious = [i for i in state.get("issues") or [] if int(i.get("severity", 0)) >= 2]
        if len(serious) >= 3:
            return "when to play the wildcard with many injured players in the squad", KB_TAGS_CHIPS
        return "when is the right time to play the wildcard", KB_TAGS_CHIPS
    sc = state.get("scenario") or {}
    n_moves = max(len(sc.get("sell") or []), len(sc.get("buy") or []))
    if n_moves >= 2:
        return "when is a -4 points hit worth it for two transfers", KB_TAGS_HITS
    return "when is a -4 points hit worth it", KB_TAGS_HITS


def _excerpt(text: str, limit: int = RULES_CONTEXT_EXCERPT_CHARS) -> str:
    body = " ".join(text.split())
    if len(body) <= limit:
        return body
    cut = body[:limit]
    dot = cut.rfind(". ")
    return (cut[: dot + 1] if dot > limit // 2 else cut.rstrip()) + " …"


def rules_context_facts(kind: str, query: str, kb: Any) -> dict[str, Any]:
    """Выдержки для объяснителя. Порядок цитирования: официальные внешние страницы правил
    (premierleague) -> внутренний дайджест правил (internal) -> гайды сообщества; внутри группы —
    порядок ранжирования KB."""
    chunks = sorted(
        kb.chunks,
        key=lambda c: (c.source == "internal", "rules" not in c.tags),
    )
    return {
        "about": kind,
        "how_to_cite": "cite one excerpt as [source] (e.g. [premierleague]); rules-tagged "
        "sources are official, others are community guides; numbers still come from the optimizer",
        "kb_query": query,
        "excerpts": [
            {
                "source": c.source,
                "title": c.title,
                "url": c.url,
                "tags": list(c.tags),
                "official_rules": "rules" in c.tags,
                "text": _excerpt(c.text),
            }
            for c in chunks
        ],
    }


def strategy_answer_facts(kb: Any) -> dict[str, Any]:
    """Ответ KB для объяснителя: текст с [n], цитаты и готовый блок Sources (копируется verbatim)."""
    lines = []
    for c in kb.citations:
        kind = "rules" if "rules" in c.tags else "guide"
        lines.append(f"- [{c.n}] {c.title} ({c.source}, {kind}) — {c.url}")
    return {
        "answer": kb.answer,
        "covered": kb.covered,
        "citations": [
            {
                "n": c.n,
                "title": c.title,
                "url": c.url,
                "source": c.source,
                "tags": list(c.tags),
                "official_rules": "rules" in c.tags,
                "quote": c.quote,
            }
            for c in kb.citations
        ],
        "sources_markdown": "\n".join(lines),
        "retrieved_chunks": kb.retrieved,
        "kb_model": kb.model,
    }


def ranking_facts(out: Any) -> dict[str, Any]:
    """Рейтинг лиги для объяснителя: компактные строки (все числа уже округлены инструментом).
    Окно (`window_label`) и смысл фильтров даны словами, чтобы объяснитель не переформулировал
    их («не менее 45 минут», а не «менее»)."""
    windowed = out.gw_from is not None
    label = out.window_label
    return {
        "metric": out.metric,
        "metric_label": out.metric_label,
        "window_label": label,
        "gw_from": out.gw_from,
        "gw_to": out.gw_to,
        "points_definition": (
            f"total_points, minutes, points_per_game, points_per_million and the consistency "
            f"columns cover {label} ONLY (from per-gameweek match history); "
            "season_total_points is the whole season to date, for context"
            if windowed
            else "total_points, minutes, points_per_game, points_per_million are season totals "
            "to date (FPL API); the consistency columns come from the local match history"
        ),
        "filters_definition": (
            "min_minutes = only players with AT LEAST this many minutes played "
            f"({label}) are included — never say 'less than'; exclude_unavailable = players "
            "currently injured / suspended / unavailable are left out; max_price / min_price = "
            "bounds on the current price in £m"
        ),
        "filters": dict(out.filters),
        "candidates_after_filters": out.candidates,
        "history_through_gw": out.history_through_gw,
        "consistency_definition": (
            "gws_5plus / share_5plus_pct = played gameweeks with >= 5 points; "
            "gws_2plus = played gameweeks with >= 2 points; gws_dnp = team matches the player "
            "did not play (0 minutes); std_points = standard deviation of points over played "
            "gameweeks (lower = steadier); points_by_gw = points in each played gameweek"
        ),
        "order": "rows are already ranked by the metric — row 1 is the answer; never re-rank",
        "rows": [
            {
                "rank": r.rank,
                "player": r.name,
                "full_name": r.full_name or r.name,
                "team": r.team,
                "position": r.position,
                "price": r.price,
                "fpl_status": r.status,
                "ownership_pct": r.ownership,
                "total_points": r.total_points,
                **(
                    {"season_total_points": r.season_total_points}
                    if r.season_total_points is not None
                    else {}
                ),
                "minutes": r.minutes,
                "points_per_game": r.points_per_game,
                "points_per_million": r.points_per_million,
                "form": r.form,
                "gws_played": r.gws_played,
                "gws_dnp": r.gws_dnp,
                "gws_in_history": r.gws_in_history,
                "gws_5plus": r.gws_5plus,
                "share_5plus_pct": r.share_5plus_pct,
                "gws_2plus": r.gws_2plus,
                "std_points": r.std_points,
                "min_max_points": (
                    f"{r.min_points}–{r.max_points}" if r.min_points is not None else "n/a"
                ),
                "points_by_gw": r.points_by_gw,
                "in_your_squad": r.in_squad,
            }
            for r in out.rows
        ],
        "notes": list(out.notes),
    }


# ---------- новости кандидатов (узел candidate_news): чистые функции ----------

NEWS_RULES_OUT = frozenset({"injured", "suspended", "unavailable"})
NEWS_SIGNAL_MAX_AGE_H = 7 * 24  # как core.signals.SIGNAL_MAX_AGE: старше — не повод исключать
CANDIDATE_EVIDENCE_MAX = 2  # цитат на кандидата в EVIDENCE
CLUB_EVIDENCE_MAX = 2  # цитат дайджеста на клуб в EVIDENCE
CANDIDATE_NEWS_ABOUT = (
    "news for the players this answer recommends (buy / sell in the optimizer's routes or plan) "
    "or ranks; xpts_* and p_start_model come from the xPts model, news_signal comes from "
    "player-specific news (availability, start_probability, confidence, signal date); "
    "FACTS.team_news is club context and never evidence about one player's own fitness"
)


def news_target(
    pid: int,
    name: str,
    role: str,
    *,
    primary: bool = False,
    forced: bool = False,
    club_news: bool = True,
) -> dict[str, Any]:
    return {
        "id": int(pid),
        "name": name,
        "role": role,
        "primary": primary,
        "forced": forced,
        "club_news": club_news,
    }


def route_targets(
    routes: Iterable[Any],
    recommended_rank: int | None,
    *,
    forced_buy: set[int] | frozenset[int] = frozenset(),
    n_routes: int = 3,
    alternative: Any = None,
) -> list[dict[str, Any]]:
    """Покупки (и продажи) лучших маршрутов: рекомендованный первым, покупки его — primary
    (их недоступность по новостям — повод пересчитать), затем покупки остальных маршрутов."""
    ordered = sorted(routes, key=lambda r: (r.rank != recommended_rank, r.rank))[: max(1, n_routes)]
    out: list[dict[str, Any]] = []
    for r in ordered:
        rec = r.rank == recommended_rank
        tag = f"route {r.rank}" + (" (recommended)" if rec else "")
        for pid, name in zip(r.in_ids, r.in_, strict=False):
            out.append(
                news_target(pid, name, f"buy, {tag}", primary=rec, forced=int(pid) in forced_buy)
            )
        if rec:
            out += [
                news_target(pid, name, f"sell, {tag}", club_news=False)
                for pid, name in zip(r.out_ids, r.out, strict=False)
            ]
    if alternative is not None:
        out += [
            news_target(pid, name, "buy, best route without the forced sale")
            for pid, name in zip(alternative.in_ids, alternative.in_, strict=False)
        ]
    return out


def plan_targets(pl: Any, gw: int) -> list[dict[str, Any]]:
    """Ходы плана: покупки ближайшего тура — primary, затем покупки следующего тура."""
    out: list[dict[str, Any]] = []
    keys = sorted(pl.moves_by_gw, key=int)
    first = str(gw) if str(gw) in pl.moves_by_gw else (keys[0] if keys else None)
    for key in [k for k in keys if k == first] + [k for k in keys if k != first][:1]:
        now = key == first
        for m in pl.moves_by_gw[key]:
            if m.in_id is not None:
                out.append(news_target(m.in_id, m.in_, f"buy, GW{m.gw} plan", primary=now))
            if now and m.out_id is not None:
                out.append(news_target(m.out_id, m.out, f"sell, GW{m.gw} plan", club_news=False))
    return out


def dedupe_targets(targets: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[int] = set()
    out = []
    for t in targets:
        if int(t["id"]) not in seen:
            seen.add(int(t["id"]))
            out.append(t)
    return out


def signal_brief(sig: dict[str, Any] | None) -> dict[str, Any]:
    if not sig:
        return {"availability": "n/a", "note": "no news signal available"}
    brief = {
        "availability": sig.get("availability"),
        "start_probability": sig.get("start_probability"),
        "confidence": sig.get("confidence"),
        "rotation_risk": sig.get("rotation_risk"),
        "return_gw": sig.get("return_gw"),
        "summary": sig.get("summary"),
        "signal_as_of": _fmt_dt(sig.get("signal_as_of")) if sig.get("signal_as_of") else None,
        "signal_age_h": sig.get("age_h"),
        "origin": sig.get("origin"),
        "evidence_count": len(sig.get("evidence") or []),
    }
    if sig.get("note"):
        brief["note"] = sig["note"]
    if sig.get("form_notes"):  # v4: форма / роль — рядом, но не часть вердикта доступности
        brief["form_and_context"] = form_notes_brief(sig["form_notes"])
    return brief


def candidate_news_facts(
    targets: Iterable[dict[str, Any]],
    sigs: dict[str, dict[str, Any]],
    preds: dict[str, dict[str, Any]],
    facts: dict[str, Any],
) -> dict[str, Any]:
    """FACTS.candidate_news: роль, xPts модели и новостной сигнал рядом (игроки из вопроса уже
    в FACTS.players — сюда не дублируются)."""
    asked = set((facts.get("players") or {}).keys())
    players: dict[str, Any] = {}
    for t in targets:
        if t["role"] == "asked" or t["name"] in asked:
            continue
        sig = sigs.get(str(t["id"]))
        pred = preds.get(str(t["id"]))
        entry: dict[str, Any] = {"role": t["role"]}
        if pred is not None:
            p = pred["player"]
            by_gw = pred.get("by_gw") or []
            entry.update(
                {
                    "team": p.get("team"),
                    "position": p.get("position"),
                    "price": p.get("price"),
                    "fpl_status": p.get("status"),
                    "xpts_by_gw": {f"GW{g['gw']}": g["xpts"] for g in by_gw},
                    f"total_xpts_{len(by_gw)}gw": pred.get("total_xpts"),
                    "p_start_model_next_gw": by_gw[0]["p_start"] if by_gw else None,
                }
            )
        elif sig is not None:
            entry["fpl_status"] = sig.get("fpl_status")
        entry["news_signal"] = signal_brief(sig)
        name = t["name"] if t["name"] not in players else f"{t['name']} ({entry.get('team')})"
        players[name] = entry
    return {"about": CANDIDATE_NEWS_ABOUT, "players": players}


def team_news_facts(clubs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """FACTS.team_news по коду клуба: недоступные игроки клуба из данных FPL (статус, тур
    возвращения по календарю) + мягкий контекст дайджеста (тренер, ротация, форма)."""
    out: dict[str, Any] = {}
    for club in clubs.values():
        out[club.get("team") or str(club.get("team_id"))] = {
            "club": club.get("team_name"),
            "about": "club context — not evidence about any single player's own fitness",
            "unavailable_or_doubtful_players": [
                {
                    k: v
                    for k, v in a.items()
                    if k in ("player", "position", "status_label", "chance_next", "return_gw",
                             "return_date", "fpl_news", "news_availability")
                    and v not in (None, "")
                }
                for a in club.get("absences") or []
            ],
            "digest_as_of": _fmt_dt(club.get("digest_as_of")) if club.get("digest_as_of") else None,
            "summary": club.get("summary") or "",
            "context_items": [
                {
                    "kind": it.get("kind"),
                    "claim": it.get("claim"),
                    "players": it.get("players") or [],
                    "source": it.get("source"),
                    "date": it.get("date"),
                }
                for it in club.get("items") or []
            ],
            **({"note": club["note"]} if club.get("note") else {}),
        }
    return out


def club_evidence(clubs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Цитаты дайджестов клубов для EVIDENCE — с пометкой scope=club (в отличие от цитат игроков)."""
    out: list[dict[str, Any]] = []
    for club in clubs.values():
        for it in (club.get("items") or [])[:CLUB_EVIDENCE_MAX]:
            out.append(
                {
                    "scope": "club",
                    "club": club.get("team_name"),
                    "team": club.get("team"),
                    "about": f"club news ({club.get('team_name')}): context, not evidence about "
                    "any single player's own fitness",
                    "kind": it.get("kind"),
                    "players": it.get("players") or [],
                    "source": it.get("source"),
                    "url": it.get("url"),
                    "published_at": it.get("published_at"),
                    "date": it.get("date"),
                    "quote": it.get("quote"),
                }
            )
    return out


def unavailable_primary_buys(
    targets: Iterable[dict[str, Any]], sigs: dict[str, dict[str, Any]], min_confidence: float
) -> list[dict[str, Any]]:
    """Рекомендованные покупки (не навязанные вопросом), которые новость делает недоступными."""
    bad = []
    for t in targets:
        if not t.get("primary") or t.get("forced") or not str(t["role"]).startswith("buy"):
            continue
        sig = sigs.get(str(t["id"])) or {}
        age = sig.get("age_h")
        if (
            sig.get("availability") in NEWS_RULES_OUT
            and float(sig.get("confidence") or 0.0) >= min_confidence
            and (age is None or float(age) <= NEWS_SIGNAL_MAX_AGE_H)
        ):
            bad.append(t)
    return bad


def news_caveats(
    targets: Iterable[dict[str, Any]],
    sigs: dict[str, dict[str, Any]],
    not_refreshed: list[str],
    budget: int,
) -> list[str]:
    out = []
    for t in targets:
        sig = sigs.get(str(t["id"])) or {}
        if str(t["role"]).startswith(("buy", "ranked")) and sig.get("availability") == "doubtful":
            out.append(
                f"{t['name']} ({t['role']}): the news signal says doubtful (confidence "
                f"{sig.get('confidence')}) — check the team news before the deadline."
            )
    if not_refreshed:
        out.append(
            f"News for {', '.join(not_refreshed)} was not refreshed within this request's budget "
            f"({budget} news extractions); saved signals are shown with their age."
        )
    return out


def clarification_text(pending: dict[str, Any], language: str | None = None) -> str:
    """Текст вопроса-уточнения (детерминированный, без LLM), на языке вопроса."""
    ru = language == "ru"
    mention = pending.get("mention")
    lines = [
        f"Какого **{mention}** вы имеете в виду?" if ru else f"Which **{mention}** do you mean?"
    ]
    for i, c in enumerate(pending.get("candidates") or [], start=1):
        flag = (" — в вашем составе" if ru else " — in your squad") if c.get("in_squad") else ""
        price = f"£{float(c['price']):.1f}" if c.get("price") is not None else "n/a"
        own = f"{float(c['ownership']):.1f}%" if c.get("ownership") is not None else "n/a"
        lines.append(
            f"{i}. {c.get('full_name') or c.get('name')} ({c.get('team')}, {c.get('position')}, "
            f"{price}, {own} {'владение' if ru else 'owned'}, {'статус' if ru else 'status'} "
            f"{c.get('status')}){flag}"
        )
    lines.append("")
    lines.append(
        "Выберите игрока кнопкой ниже — ответ продолжится с этим игроком."
        if ru
        else "Pick the player with a button below — the answer continues with that player."
    )
    return "\n".join(lines).strip()


# ---------- детерминированный заголовок фактов ----------


def _signed(x: Any) -> str:
    return "n/a" if x is None else f"{float(x):+}"


def _route_line(r: dict[str, Any]) -> str:
    hit = f"hit -{r['hit_cost']}" if r.get("hit_cost") else "no hit"
    n = len(r.get("in") or [])
    return (
        f"{', '.join(r.get('out') or [])} -> {', '.join(r.get('in') or [])} "
        f"({n} transfer{'s' if n != 1 else ''}, {hit}): {_signed(r.get('gain_next_gw'))} xPts "
        f"next GW, {_signed(r.get('gain_horizon'))} over the horizon; verdict {r.get('verdict')}"
    )


def headline(intent: str, facts: dict[str, Any], state: AgentState) -> str:
    """Одна фраза-вердикт, собранная кодом из чисел: объяснитель обязан опираться на неё.

    Это главный барьер против «−4», которого нет в фактах, и против переворота вердикта.
    """
    gw = facts.get("gw")
    players = facts.get("players") or {}
    v4 = chat.headline_v4(intent, facts)
    if v4:
        return v4
    if intent == "player_status" and players:
        parts = []
        for name, p in players.items():
            nxt = p.get("next_gw") or {}
            sig = p.get("news_signal") or {}
            status = f"FPL status {p.get('fpl_status')}"
            if p.get("fpl_chance_pct") is not None:
                status += f" ({p['fpl_chance_pct']}% chance)"
            news = (
                f"news signal {sig.get('availability')} (confidence {sig.get('confidence')})"
                if sig
                else "no news signal"
            )
            parts.append(
                f"{name}: {status}; {news}; GW{nxt.get('gw', gw)} xPts {nxt.get('xpts')}, "
                f"p(start) {nxt.get('p_start')}"
            )
        return " | ".join(parts)
    if intent == "compare_players" and facts.get("comparison_ranking"):
        ranked = " > ".join(f"{r['name']} ({r['total_xpts']})" for r in facts["comparison_ranking"])
        return f"Ranking by total xPts over the horizon: {ranked}"
    if intent == "transfer" and facts.get("transfers"):
        tr = facts["transfers"]
        rec = tr.get("recommendation", "hold")
        if rec.startswith("route"):
            idx = int(rec.split()[-1]) - 1
            routes = tr.get("routes") or []
            if 0 <= idx < len(routes):
                scope = (
                    "under the question's constraint"
                    if tr.get("constrained_by_question")
                    else "(optimizer, free choice)"
                )
                line = f"Best route GW{gw} {scope}: {_route_line(routes[idx])}"
                if tr.get("allow_hit") and not routes[idx].get("hit_cost"):
                    # Пользователь спросил про хит, а лучший маршрут бесплатный: сказать это явно —
                    # главный барьер против вердикта «take a -4», которого нет в фактах.
                    line += (
                        ". NO hit needed — the recommended route uses free transfers only; a paid "
                        "transfer (-4) is NOT recommended"
                    )
                alt = tr.get("best_route_without_forcing_the_sale")
                if alt and alt.get("in"):
                    line += (
                        f". Alternative keeping {', '.join(alt.get('keeps') or [])}: "
                        f"{_route_line(alt)}"
                    )
                if tr.get("verdict_on_forced_sale"):
                    line += f". VERDICT: {tr['verdict_on_forced_sale']}"
                return line
        return (
            f"Recommended GW{gw}: hold (roll the free transfer) — no route beats keeping the "
            "transfer; best XI without transfers "
            f"{tr.get('baseline_xi_points_no_transfer')} xPts"
        )
    if intent in ("captain", "lineup") and facts.get("best_xi"):
        xi = facts["best_xi"]
        opts = facts.get("captain_options") or []
        top = opts[0] if opts else None
        cap = (
            f"captain {top['name']} ({top['xpts']} xPts, x2 = {top['captain_points']}, "
            f"{top['ownership_pct']}% owned, {top['tag']})"
            if top
            else f"captain {xi.get('captain')}"
        )
        return (
            f"Best XI GW{xi.get('gw') or gw} ({xi.get('formation')}): {xi.get('expected_points')} xPts; {cap}; "
            f"vice {xi.get('vice')}"
        )
    if intent == "plan" and facts.get("plan"):
        pl = facts["plan"]
        first = (pl.get("moves_by_gw") or {}).get(str(gw)) or []
        wc = pl.get("wildcard_alternative")
        line = (
            f"Plan {pl.get('gws')}: recommendation {pl.get('recommendation')}; GW{gw}: "
            + "; ".join(first)
            + f"; expected total {pl.get('expected_total')} vs {pl.get('baseline_total_no_transfers')}"
            " without transfers"
        )
        if wc:
            line += (
                f"; wildcard alternative {wc.get('expected_total')} "
                f"({_signed(wc.get('delta_vs_plan'))} vs plan)"
            )
        return line
    if intent == "what_if" and facts.get("scenario"):
        sc = facts["scenario"]
        if not sc.get("feasible"):
            return f"Scenario NOT feasible: {sc.get('reason')}"
        primary = sc.get("primary")
        if primary:
            n_paid = primary["hit_cost"] // 4 if primary.get("hit_cost") else 0
            hit_txt = (
                f"needs {n_paid} paid transfer(s) (hit -{primary['hit_cost']})"
                if n_paid
                else "NO hit needed — free transfers cover it"
            )
            line = f"Scenario feasible, {hit_txt}: {_route_line(primary)}"
            if sc.get("free_alternative"):
                line += f". Free alternative: {_route_line(sc['free_alternative'])}"
            if sc.get("hit_alternative_as_asked"):
                line += f". The -4 version you asked about: {_route_line(sc['hit_alternative_as_asked'])}"
            return line
        if sc.get("wildcard"):
            wc = sc["wildcard"]
            return (
                f"Wildcard now: expected total {wc.get('expected_total')} vs "
                f"{wc.get('plan_without_wildcard_total')} without it ({_signed(wc.get('delta_vs_plan'))}); "
                f"optimizer recommendation {wc.get('recommendation')}"
            )
    if intent == "player_ranking":
        pr = facts.get("player_ranking")
        if pr is None:
            return "Player ranking unavailable — no numbers were computed"
        rows = pr.get("rows") or []
        flt = pr.get("filters") or {}
        window = pr.get("window_label") or SEASON_WINDOW_LABEL
        scope = ", ".join(
            s
            for s in (
                flt.get("position"),
                f"<= £{flt['max_price']}" if flt.get("max_price") is not None else None,
                f">= £{flt['min_price']}" if flt.get("min_price") is not None else None,
                (
                    f"players with at least {flt['min_minutes']} minutes played"
                    if flt.get("min_minutes") is not None
                    else None
                ),
            )
            if s
        )
        if not rows:
            return f"No player passes the filters ({scope or 'none'}, {window}) — relax them"
        top = "; ".join(
            f"{r['rank']}. {r['player']} ({r['team']}, {r['position']}, £{r['price']}, "
            f"{r['total_points']} pts, {r['points_per_million']} pts/£m, >=5 pts in "
            f"{r['gws_5plus']} of {r['gws_played']} GWs)"
            for r in rows[:3]
        )
        return (
            f"Top {len(rows)} of {pr.get('candidates_after_filters')} players by "
            f"{pr.get('metric_label')}, {window}" + (f" [{scope}]" if scope else "") + f": {top}"
        )
    if intent == "strategy_question":
        sa = facts.get("strategy_answer")
        if sa is None:
            return "Answer from the built-in FPL 2026/27 rules digest (no squad computation)"
        if not sa.get("covered"):
            return f"Strategy KB: {NOT_COVERED} No rule is invented; no squad computation."
        cits = sa.get("citations") or []
        n_rules = sum(1 for c in cits if c.get("official_rules"))
        first = str(sa.get("answer") or "").split(". ")[0].strip()
        return (
            f"Strategy KB answer with {len(cits)} cited source(s) ({n_rules} official rules): "
            f"{first}"
        )
    if state.get("needs_squad") and facts.get("manager") is None:
        return "Squad-level advice unavailable without a manager squad; player-level facts only"
    return "See facts"


# ---------- условные рёбра (ветвление / цикл / HITL) ----------


def after_router(state: AgentState) -> str:
    intent = state.get("intent")
    if intent in ("betting", "off_topic", "error"):
        return "refuse"
    if state.get("clarification"):
        return "clarify"
    return "ensure_signals"


def after_resolve_clarification(state: AgentState) -> str:
    choice = state.get("clarification_choice") or {}
    if choice.get("decision") == "cancel":
        return END
    if state.get("clarification"):  # осталось ещё неоднозначное имя -> следующий вопрос
        return "clarify"
    return "ensure_signals"


def after_grade(state: AgentState) -> str:
    grade = state.get("grade") or {}
    return "rewrite_retry" if grade.get("will_retry") else "compute"


def after_candidate_news(state: AgentState) -> str:
    """Покупка из рекомендации недоступна по свежей новости -> один пересчёт без неё."""
    return "compute" if state.get("news_reoptimize") else "check_action"


def after_check(state: AgentState) -> str:
    return "confirm_action" if state.get("pending_action") else "explain"


def after_confirm(state: AgentState) -> str:
    return "compute" if state.get("user_decision") == "reject" else "explain"


def after_validate(state: AgentState) -> str:
    validation = state.get("validation") or {}
    return "explain" if validation.get("regenerate") else END


def build_graph(deps: Deps, checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    nodes = make_nodes(deps)
    g: StateGraph = StateGraph(AgentState)
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.add_edge(START, "load_context")
    g.add_edge("load_context", "router")
    g.add_conditional_edges("router", after_router, ["refuse", "clarify", "ensure_signals"])
    g.add_edge("refuse", END)
    g.add_edge("clarify", "resolve_clarification")
    g.add_conditional_edges(
        "resolve_clarification", after_resolve_clarification, ["clarify", "ensure_signals", END]
    )
    g.add_edge("ensure_signals", "grade_signals")
    g.add_conditional_edges("grade_signals", after_grade, ["rewrite_retry", "compute"])
    g.add_edge("rewrite_retry", "grade_signals")
    g.add_edge("compute", "candidate_news")
    g.add_conditional_edges("candidate_news", after_candidate_news, ["compute", "check_action"])
    g.add_conditional_edges("check_action", after_check, ["confirm_action", "explain"])
    g.add_conditional_edges("confirm_action", after_confirm, ["compute", "explain"])
    g.add_edge("explain", "validate_answer")
    g.add_conditional_edges("validate_answer", after_validate, ["explain", END])
    return g.compile(
        checkpointer=checkpointer or MemorySaver(),
        interrupt_before=["confirm_action", "resolve_clarification"],
    )


# ---------- запуск, стриминг, resume ----------


def status_line(node: str, delta: dict[str, Any] | None, *, after: str | None = None) -> str:
    """Строка статуса узла для stderr / UI; `after` — предыдущий узел (для __interrupt__)."""
    d = delta or {}
    if node == "__interrupt__":
        if after == "clarify":
            return "waiting for the user's choice of player (--choose <player_id> | cancel)"
        return "waiting for human decision (confirm | reject)"
    if node == "load_context":
        s = d.get("squad_summary")
        squad = (
            f"squad {len(s.get('squad') or [])} players from GW{s.get('squad_gw')}, "
            f"bank £{s.get('bank')}, FT {s.get('free_transfers')}"
            if s
            else f"no squad ({d.get('squad_note')})"
        )
        return f"GW{d.get('gw')} deadline {_fmt_dt(d.get('deadline'))}; {squad}; issues {len(d.get('issues') or [])}"
    if node == "router":
        names = [p["name"] for p in d.get("players") or []]
        sc = d.get("scenario") or {}
        extra = []
        if sc.get("allow_hit") is not None:
            extra.append(f"allow_hit={sc['allow_hit']}")
        if sc.get("use_wildcard") is not None:
            extra.append(f"wildcard={sc['use_wildcard']}")
        if d.get("clarification"):
            extra.append("ambiguous -> clarify")
        if d.get("target_gw"):
            extra.append(f"target=GW{d['target_gw']}")
        if d.get("chips_asked"):
            extra.append(
                "chips=" + ",".join(f"{c['chip']}@GW{c.get('gw') or '?'}" for c in d["chips_asked"])
            )
        if d.get("standalone_query"):
            extra.append(f"rewritten={str(d['standalone_query'])[:90]!r}")
        if d.get("ranking"):
            rk = d["ranking"]
            extra.append("ranking=" + ",".join(f"{k}={v}" for k, v in rk.items() if v is not None))
        return (
            f"intent={d.get('intent')} players={names} horizon={d.get('horizon')} "
            f"lang={d.get('language')}" + (" " + " ".join(extra) if extra else "")
        )
    if node == "refuse":
        return "deterministic answer (no LLM)"
    if node == "clarify":
        p = d.get("pending_clarification") or {}
        return (
            f"ambiguous '{p.get('mention')}': {len(p.get('candidates') or [])} candidates "
            "(no LLM) -> interrupt"
        )
    if node == "resolve_clarification":
        c = d.get("clarification_choice") or {}
        if c.get("decision") == "cancel":
            return "cancelled by the user"
        picked = next((p for p in d.get("players") or [] if p.get("id") == c.get("player_id")), {})
        return f"player_id={c.get('player_id')} -> {picked.get('full_name') or picked.get('name')}"
    if node in ("ensure_signals", "rewrite_retry"):
        logs = [t for t in d.get("tool_log") or [] if t.get("tool") == "analyze_player_risk"]
        origins = [str(t.get("note") or "").split(":")[0] for t in logs]
        counts = {o: origins.count(o) for o in dict.fromkeys(origins)}
        prefix = f"retry {d.get('retries')}: " if node == "rewrite_retry" else ""
        return prefix + (f"{len(logs)} signals {counts}" if logs else "no players need signals")
    if node == "grade_signals":
        g = d.get("grade") or {}
        return (
            "all signals sufficient"
            if g.get("sufficient")
            else f"insufficient {g.get('insufficient')} -> {'retry' if g.get('will_retry') else 'proceed'}"
        )
    if node == "compute":
        tools_called = [t["tool"] for t in d.get("tool_log") or [] if t.get("tool") != "-"]
        facts = d.get("facts") or {}
        extra = ""
        if facts.get("rules_context"):
            extra = f"; rules_context: {facts['rules_context'].get('about')} x{len(facts['rules_context'].get('excerpts') or [])}"
        if facts.get("strategy_answer"):
            sa = facts["strategy_answer"]
            extra = f"; kb answer: covered={sa.get('covered')} citations={len(sa.get('citations') or [])}"
        return f"tools {tools_called}; facts: {sorted(facts.keys())}{extra}"
    if node == "candidate_news":
        logs = d.get("tool_log") or []
        sigs = [t for t in logs if t.get("tool") == "analyze_player_risk"]
        clubs = [t for t in logs if t.get("tool") == "team_news"]
        if not sigs and not clubs:
            return str((logs[-1] if logs else {}).get("note") or "no candidates")
        return (
            f"{len(sigs)} candidate signal(s), {len(clubs)} club digest(s); "
            f"news extractions {d.get('news_llm_used')}"
            + (" -> re-optimize" if d.get("news_reoptimize") else "")
        )
    if node == "check_action":
        p = d.get("pending_action")
        return f"pending {p['kind']} ({p['detail']})" if p else "no confirmation needed"
    if node == "confirm_action":
        return f"decision={d.get('user_decision')}"
    if node == "explain":
        return f"attempt {d.get('explain_attempts')}: {len(d.get('answer') or '')} chars"
    if node == "validate_answer":
        v = d.get("validation") or {}
        pruned = (
            f"; pruned {len(v['pruned_sources'])} undiscussed source line(s)"
            if v.get("pruned_sources")
            else ""
        )
        return (
            "passed" + pruned
            if v.get("passed")
            else (
                f"violations names={v.get('unknown_names')} numbers={v.get('unknown_numbers')} "
                f"citations={v.get('bad_citations')}"
                + (" kb_markers_missing" if v.get("missing_refs") else "")
                + (" hit_mismatch" if v.get("hit_mismatch") else "")
                + (f" cyrillic={list(v['cyrillic_names'])}" if v.get("cyrillic_names") else "")
                + (f" gws={v.get('unknown_gws')}" if v.get("unknown_gws") else "")
                + (f" gw_uncovered={v.get('gw_uncovered')}" if v.get("gw_uncovered") else "")
                + (f" captain!={v.get('captain_mismatch')}" if v.get("captain_mismatch") else "")
                + (
                    f" not_in_squad={v.get('not_in_squad_claims')}"
                    if v.get("not_in_squad_claims")
                    else ""
                )
                + (
                    " misattributed="
                    + str([f"{m['player']} {m['number']}" for m in v["misattributed_numbers"]])
                    if v.get("misattributed_numbers")
                    else ""
                )
                + (" -> regenerate" if v.get("regenerate") else " -> caveat")
            )
        )
    return str(d)[:120]


class Agent:
    """Обёртка над скомпилированным графом: run / stream / resume + снимок состояния."""

    def __init__(self, deps: Deps, checkpointer: BaseCheckpointSaver | None = None) -> None:
        self.deps = deps
        self.checkpointer = checkpointer or MemorySaver()
        self.graph = build_graph(deps, self.checkpointer)

    def config(self, thread_id: str, *, intent: str | None = None, **metadata: Any) -> dict:
        return {
            "configurable": {"thread_id": thread_id},
            "tags": tracing.run_config_tags(
                strategy=str(metadata.get("strategy", "balanced")),
                model=self.deps.explain_model,
                intent=intent,
            ),
            "metadata": {"thread_id": thread_id, **metadata},
            "run_name": "fpl_copilot_agent",
        }

    def close(self) -> None:
        """Освободить ресурсы инструментов (см. LiveTools.close) и собрать мусор до выхода."""
        import gc

        close = getattr(self.deps.tools, "close", None)
        if callable(close):
            close()
        gc.collect()

    def snapshot(self, thread_id: str) -> dict[str, Any]:
        snap = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        values: dict[str, Any] = dict(snap.values)
        values["thread_id"] = thread_id
        values["interrupted"] = bool(snap.next)
        values["next_nodes"] = list(snap.next)
        # Вид прерывания для UI/CLI: action (confirm_action) | clarification (resolve_clarification)
        values["interrupt_kind"] = (
            "clarification"
            if "resolve_clarification" in snap.next
            else "action"
            if "confirm_action" in snap.next
            else None
        )
        values["cost_estimate"] = round(
            sum(float(c.get("cost_usd") or 0.0) for c in values.get("llm_calls") or []), 6
        )
        values["turn_meta"] = chat.turn_meta(values)  # контекст этого хода для следующего
        return values

    @staticmethod
    def _initial(
        query: str,
        *,
        thread_id: str,
        manager_id: int | None,
        strategy: str,
        squad_override: SquadLike | None,
        history: list[dict[str, Any]] | None = None,
    ) -> AgentState:
        override = as_override(squad_override)
        return initial_state(
            query,
            thread_id=thread_id,
            manager_id=manager_id,
            strategy=strategy,
            squad_override=None if override is None else override.model_dump(mode="json"),
            history=history,
        )

    def run(
        self,
        query: str,
        *,
        manager_id: int | None = None,
        strategy: str = "balanced",
        thread_id: str | None = None,
        squad_override: SquadLike | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """squad_override — состав не из API (скриншот): `Squad` или `SquadOverride`; None —
        picks последнего завершённого тура (поведение по умолчанию)."""
        thread_id = thread_id or new_thread_id()
        state = self._initial(
            query,
            thread_id=thread_id,
            manager_id=manager_id,
            strategy=strategy,
            squad_override=squad_override,
            history=history,
        )
        cfg = self.config(thread_id, strategy=strategy, manager_id=manager_id)
        self.graph.invoke(state, cfg)
        return self.snapshot(thread_id)

    def stream(
        self,
        query: str,
        *,
        manager_id: int | None = None,
        strategy: str = "balanced",
        thread_id: str | None = None,
        squad_override: SquadLike | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> Iterator[tuple[str, str]]:
        """Генератор (узел, строка статуса); итоговое состояние — self.snapshot(thread_id)."""
        thread_id = thread_id or new_thread_id()
        state = self._initial(
            query,
            thread_id=thread_id,
            manager_id=manager_id,
            strategy=strategy,
            squad_override=squad_override,
            history=history,
        )
        cfg = self.config(thread_id, strategy=strategy, manager_id=manager_id)
        yield "start", f"thread {thread_id}"
        yield from self._stream_updates(state, cfg)

    def _stream_updates(self, inp: Any, cfg: dict) -> Iterator[tuple[str, str]]:
        last: str | None = None
        for update in self.graph.stream(inp, cfg, stream_mode="updates"):
            for node, delta in update.items():
                yield (
                    node,
                    status_line(node, delta if isinstance(delta, dict) else None, after=last),
                )
                if node != "__interrupt__":
                    last = node

    def resume(
        self,
        thread_id: str,
        decision: str | None = None,
        *,
        player_id: int | None = None,
    ) -> dict[str, Any]:
        """Продолжить прерванный тред. Ожидание подтверждения (`confirm_action`): decision =
        confirm | reject. Ожидание уточнения имени (`resolve_clarification`): player_id одного
        из кандидатов (decision опционально 'choose') либо decision = 'cancel'."""
        for _ in self.stream_resume(thread_id, decision, player_id=player_id):
            pass
        return self.snapshot(thread_id)

    def stream_resume(
        self,
        thread_id: str,
        decision: str | None = None,
        *,
        player_id: int | None = None,
    ) -> Iterator[tuple[str, str]]:
        cfg_base = {"configurable": {"thread_id": thread_id}}
        snap = self.graph.get_state(cfg_base)
        if not snap.values:
            raise ValueError(f"unknown thread {thread_id}")
        values = snap.values
        cfg = self.config(
            thread_id,
            intent=values.get("intent"),
            strategy=values.get("strategy", "balanced"),
            manager_id=values.get("manager_id"),
            resumed=True,
        )
        if "resolve_clarification" in snap.next:
            pending = values.get("pending_clarification") or {}
            ids = {int(c["player_id"]) for c in pending.get("candidates") or []}
            if decision == "cancel" or (decision in (None, "reject") and player_id is None):
                choice: dict[str, Any] = {"decision": "cancel"}
            else:
                if decision not in (None, "choose"):
                    raise ValueError(
                        f"thread {thread_id} waits for a player choice: pass player_id (one of "
                        f"{sorted(ids)}) or decision='cancel', not {decision!r}"
                    )
                if player_id is None or int(player_id) not in ids:
                    raise ValueError(
                        f"player_id must be one of the candidates {sorted(ids)}, got {player_id!r}"
                    )
                choice = {"decision": "choose", "player_id": int(player_id)}
            # Выбор записывается как выход узла clarify; ребро clarify -> resolve_clarification
            # безусловное, поэтому граф продолжает узлом-«человеком» без повторного прерывания.
            self.graph.update_state(cfg_base, {"clarification_choice": choice}, as_node="clarify")
            yield "resume", f"thread {thread_id}: clarification {choice}"
            yield from self._stream_updates(None, cfg)
            return
        if "confirm_action" not in snap.next:
            raise ValueError(f"thread {thread_id} is not waiting for a decision (next={snap.next})")
        if decision not in ("confirm", "reject"):
            raise ValueError("decision must be 'confirm' or 'reject'")
        # Решение записывается как выход последнего узла (check_action); ребро -> confirm_action
        # зависит только от pending_action, поэтому граф продолжает с узла-«человека».
        self.graph.update_state(cfg_base, {"user_decision": decision}, as_node="check_action")
        yield "resume", f"thread {thread_id}: decision={decision}"
        yield from self._stream_updates(None, cfg)


# ---------- живой агент ----------


def live_deps(
    *, grader: bool = True, tools: Any | None = None, prompt_version: str | None = None
) -> Deps:
    """tools — готовый LiveTools (UI делит один экземпляр между страницами и агентом);
    prompt_version — версия промптов роутера/объяснителя (по умолчанию AGENT_PROMPT_VERSION)."""
    from fplcopilot.agent import llm as agent_llm
    from fplcopilot.agent.resolve import PlayerResolver
    from fplcopilot.agent.tools import LiveTools
    from fplcopilot.rag.entity_matcher import TEAM_ALIASES

    version = prompt_version or settings.agent_prompt_version
    tools = tools if tools is not None else LiveTools()
    teams = tools.bootstrap.teams
    known = [t.name for t in teams] + [t.short_name for t in teams]
    for aliases in TEAM_ALIASES.values():
        known.extend(aliases)
    return Deps(
        tools=tools,
        router_llm=lambda req: agent_llm.router_llm(
            req, model=settings.agent_router_model, version=version
        ),
        explain_llm=lambda req: agent_llm.explain_llm(
            req, model=settings.agent_explain_model, version=version
        ),
        grader_llm=(
            (
                lambda req: agent_llm.grader_llm(
                    req, model=settings.agent_router_model, version=version
                )
            )
            if grader
            else None
        ),
        resolver=lambda squad: PlayerResolver(tools.bootstrap, squad_ids=squad),
        extra_known_names=known,
        prompt_version=version,
    )


def sqlite_checkpointer(path: str | None = None) -> BaseCheckpointSaver:
    import sqlite3
    from pathlib import Path

    from langgraph.checkpoint.sqlite import SqliteSaver

    p = Path(path or settings.agent_checkpoint_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(p), check_same_thread=False))


def live_agent(
    *,
    checkpoint_path: str | None = None,
    grader: bool = True,
    tools: Any | None = None,
    prompt_version: str | None = None,
) -> Agent:
    return Agent(
        live_deps(grader=grader, tools=tools, prompt_version=prompt_version),
        sqlite_checkpointer(checkpoint_path),
    )
