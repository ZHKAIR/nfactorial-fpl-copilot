"""Инструменты агента: тонкие in-process функции над core/ и rag/ с pydantic-схемами входа/выхода.

Ровно этот набор выставляет MCP-сервер: get_gameweek_context, predict_player, compare_players,
analyze_player_risk, optimize_team, recommend_transfers, build_gameweek_plan, simulate_scenario,
diagnose_squad, search_strategy_kb (10-й: стратегическая KB, docs/strategy_kb.md), rank_players
(11-й: ранжирование игроков всей лиги по очкам за £1m / стабильности / очкам / форме из bootstrap
и player_gw_history — интент player_ranking; окно туров gw_from/gw_to/last_n_gws считается только
по player_gw_history, без истории — HistoryUnavailable, а не тихий откат к сезону). LLM здесь не
считают ничего: LLM-вызовы есть только внутри rag.extract.extract_signal (analyze_player_risk)
и rag.kb.answer.answer_strategy_question (обёртка `answer_strategy_question` для интента
strategy_question — цитируемый ответ, не число). Каждый вызов из графа логируется в tool_log с
латентностью (graph.call_tool). `team_news` (дайджест новостей клуба, rag/team_news.py; единственный
LLM-вызов внутри extract_team_news) — обёртка для узла candidate_news и карточки игрока, в MCP не
выставлена и в протокол `AgentTools` не входит (граф вызывает её, только если она есть).

Протокол `AgentTools` позволяет тестам подменять реализацию фейком; `LiveTools` — реальная,
с кэшем входов оптимизатора на время одного запроса (`invalidate()` после новых извлечений,
чтобы свежий сигнал попал в прогнозы).

Состав не из API (скриншот «Pick Team», docs/vision.md): инструменты уровня состава принимают
необязательный `squad_override` (`Squad` или компактный `SquadOverride`); по умолчанию None —
picks последнего завершённого тура, как раньше.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import time
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol

import httpx
import pulp
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from fplcopilot.core.candidates import Candidate, PredictionStore, SquadIssue, problem_players
from fplcopilot.core.optimizer import (
    BENCH_BOOST,
    CHIP_TITLES,
    TRIPLE_CAPTAIN,
    ChipPlanError,
    InfeasibleError,
    LineupResult,
    ManagerInputs,
    Model,
    ModelSpec,
    Solution,
    TransferPlan,
    TransferRoute,
    best_xi,
    load_inputs,
    plan_transfers,
    recommend_route,
    single_transfer,
)
from fplcopilot.core.plan import diff_plans, inputs_hash, latest_plan, save_plan
from fplcopilot.core.strategy import HIT_COST, get_strategy
from fplcopilot.core.xpts import XPtsBreakdown
from fplcopilot.data import Bootstrap, FPLClient, Player, Squad
from fplcopilot.data.schemas import Pick, SquadPlayer
from fplcopilot.db import session_scope

log = logging.getLogger(__name__)

Origin = Literal["cached", "extracted", "unavailable"]
CaptainTag = Literal["safe", "balanced", "differential"]
# Теги капитанских опций по владению (selected_by_percent), см. docs/agent.md.
SAFE_OWNERSHIP = 30.0
DIFFERENTIAL_OWNERSHIP = 10.0
DIAGNOSIS_HORIZON = 3  # туров для diagnose_squad и лучших 11
STATUS_LABEL = {
    "a": "available",
    "d": "doubtful",
    "i": "injured",
    "s": "suspended",
    "u": "unavailable",
    "n": "not in squad",
}


def r2(x: float | None) -> float | None:
    return None if x is None else round(float(x), 2)


# ---------- общие схемы ----------


class PlayerRef(BaseModel):
    id: int
    name: str
    full_name: str = ""
    team: str
    position: str
    price: float
    status: str = "a"
    chance: int | None = None
    ownership: float = 0.0
    news: str = ""


class SquadPlayerRow(PlayerRef):
    is_starting: bool = True
    is_captain: bool = False
    is_vice: bool = False
    xpts_next: float | None = None


class GameweekContextInput(BaseModel):
    manager_id: int | None = None
    strategy: str = "balanced"


class GameweekContext(BaseModel):
    gw: int
    current_gw: int | None
    deadline: datetime
    as_of: datetime
    squad_gw: int | None = None
    bank: float | None = None
    free_transfers: int | None = None
    chips_available: list[str] = Field(default_factory=list)
    squad: list[SquadPlayerRow] | None = None
    squad_note: str | None = None
    issues: list[dict[str, Any]] = Field(default_factory=list)


class FixtureBrief(BaseModel):
    opponent: str
    is_home: bool
    fsi: int  # fixture strength index 1 (very easy) .. 5 (very hard)
    xg_for: float
    xg_against: float
    clean_sheet_prob: float
    fsi_source: str = "team_rating"  # 'odds' | 'team_rating' — чем посчитана сложность (подсказка UI)

    def label(self) -> str:
        return f"{'v' if self.is_home else '@'}{self.opponent} (FSI {self.fsi})"


class GWPrediction(BaseModel):
    gw: int
    xpts: float
    sd: float
    p_start: float
    exp_minutes: float
    components: dict[str, float]
    fixtures: list[FixtureBrief]
    ep_next: float | None = None
    notes: list[str] = Field(default_factory=list)


class PredictPlayerInput(BaseModel):
    player_id: int
    gw: int
    horizon: int = 1


class PlayerPrediction(BaseModel):
    player: PlayerRef
    by_gw: list[GWPrediction]
    total_xpts: float
    signal_used: dict[str, Any] | None = None


class ComparePlayersInput(BaseModel):
    player_ids: list[int]
    gw: int
    horizon: int = 3


class ComparePlayersOutput(BaseModel):
    players: list[PlayerPrediction]
    ranking: list[dict[str, Any]]  # [{id, name, total_xpts}] по убыванию


class EvidenceItem(BaseModel):
    player_id: int
    player: str
    source: str
    url: str
    published_at: str  # ISO
    date: str  # dd.mm — форма для цитирования в ответе
    quote: str


class PlayerRiskInput(BaseModel):
    player_id: int
    as_of: datetime
    max_age_h: float = 12.0
    force: bool = False  # игнорировать кэш и извлекать заново
    mode: str = "hybrid_rerank"
    k: int = 8
    cached_only: bool = False  # только сохранённый сигнал (любого возраста), без LLM-извлечения


class PlayerRisk(BaseModel):
    player_id: int
    player: str
    team_id: int | None = None  # клуб игрока: ключ для дайджеста новостей клуба (team_news)
    fpl_status: str
    fpl_chance: int | None
    fpl_news: str = ""
    availability: str = "unknown"
    start_probability: float = 0.5
    expected_minutes: int = 45
    rotation_risk: str = "unknown"
    return_gw: int | None = None
    confidence: float = 0.0
    summary: str = ""
    evidence: list[EvidenceItem] = Field(default_factory=list)
    signal_as_of: str | None = None
    age_h: float | None = None
    origin: Origin = "unavailable"
    abstained: bool = False
    mode: str = "hybrid_rerank"
    k: int = 8
    note: str | None = None
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    model: str | None = None


class TeamNewsInput(BaseModel):
    """Дайджест новостей клуба (rag/team_news.py): сохранённый моложе max_age_h, иначе извлечение."""

    team_id: int
    as_of: datetime
    max_age_h: float = 12.0
    force: bool = False
    cached_only: bool = False  # только сохранённый дайджест (любого возраста), без LLM
    mode: str = "hybrid_rerank"
    k: int = 8


class TeamNewsItemOut(BaseModel):
    kind: str  # absence | return | rotation | manager_quote | form_context
    claim: str
    players: list[str] = Field(default_factory=list)  # игроки клуба, проверенные кодом
    player_ids: list[int] = Field(default_factory=list)
    source: str
    url: str
    published_at: str  # ISO
    date: str  # dd.mm — форма для цитирования
    quote: str


class ClubAbsenceOut(BaseModel):
    """Игрок клуба со статусом FPL d/i/s/u/n на as_of — из структурированных данных, не из текста
    статей: тур возвращения по официальной дате FPL и календарю клуба."""

    player_id: int
    player: str
    position: str
    status: str
    status_label: str
    chance_next: int | None = None
    fpl_news: str = ""
    return_date: str | None = None  # ISO, из текста FPL («Suspended until 17 Oct»)
    return_gw: int | None = None  # первый тур клуба с матчем в эту дату или позже
    news_availability: str | None = None  # сохранённый новостной сигнал одноклубника, если есть
    news_as_of: str | None = None


class TeamNewsOut(BaseModel):
    """Контекст клуба — НЕ доказательство доступности конкретного игрока (в PlayerRisk.evidence
    не попадает)."""

    team_id: int
    team: str  # short_name
    team_name: str
    absences: list[ClubAbsenceOut] = Field(default_factory=list)  # детерминированно (FPL)
    summary: str = ""  # мягкий контекст дайджеста LLM: тренер, ротация, форма
    items: list[TeamNewsItemOut] = Field(default_factory=list)
    digest_as_of: str | None = None
    age_h: float | None = None
    origin: Origin = "unavailable"
    abstained: bool = False
    mode: str = "hybrid_rerank"
    note: str | None = None
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    model: str | None = None


class OptimizeTeamInput(BaseModel):
    manager_id: int
    gw: int
    strategy: str = "balanced"


class LineupPlayer(BaseModel):
    id: int
    name: str
    team: str
    position: str
    price: float
    xpts: float
    sd: float
    p_start: float
    ownership: float
    fixture: str
    status: str = "a"


class CaptainOption(BaseModel):
    id: int
    name: str
    team: str
    xpts: float
    captain_points: float  # 2 · xPts
    sd: float
    ownership: float
    tag: CaptainTag
    fixture: str
    p_start: float


class LineupOut(BaseModel):
    gw: int
    formation: str
    starters: list[LineupPlayer]
    bench: list[LineupPlayer]
    captain: str
    vice: str
    expected_points: float
    captain_options: list[CaptainOption]
    current_captain: str | None = None
    current_xi_points: float | None = None  # очки нынешних 11 (picks) с нынешним капитаном


class RouteOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    rank: int
    out: list[str]
    in_: list[str] = Field(alias="in")
    out_ids: list[int]
    in_ids: list[int]
    hit_cost: int
    gain_next_gw: float
    gain_horizon: float
    gain_discounted: float
    objective_gain: float
    hit_marginal_gain: float | None
    new_bank: float
    risk_note: str
    verdict: str
    xi_points_after: float


class RecommendTransfersInput(BaseModel):
    manager_id: int
    gw: int
    horizon: int = 3
    strategy: str = "balanced"
    allow_hit: bool | None = None
    sell: list[int] = Field(default_factory=list)  # обязательно продать
    buy: list[int] = Field(default_factory=list)  # обязательно купить
    keep: list[int] = Field(default_factory=list)  # не продавать
    exclude: list[int] = Field(default_factory=list)  # не покупать


class RoutesOut(BaseModel):
    gw: int
    horizon: int
    strategy: str
    free_transfers: int
    bank: float
    routes: list[RouteOut]
    recommended_rank: int | None
    recommendation: str  # "route N" | "hold"
    baseline_xi_points: float
    constrained: bool = False
    alternative: RouteOut | None = None  # лучший маршрут без ограничений сценария
    notes: list[str] = Field(default_factory=list)


class PlanMoveOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    gw: int
    out: str
    in_: str = Field(alias="in")
    price_out: float
    price_in: float
    delta_xpts_horizon: float
    out_problem: str | None = None
    paid: bool = False
    out_id: int | None = None
    in_id: int | None = None


class PlanChipIn(BaseModel):
    """Условие «фишка в туре»: chip = bboost (Bench Boost) | 3xc (Triple Captain)."""

    gw: int
    chip: str


class PlanChipOut(BaseModel):
    gw: int
    chip: str  # bboost | 3xc
    name: str  # Bench Boost | Triple Captain
    points: float  # вклад фишки: BB — Σ xPts скамейки, TC — ещё раз xPts капитана
    players: list[str] = Field(default_factory=list)  # BB — скамейка тура, TC — капитан


class BuildPlanInput(BaseModel):
    manager_id: int
    gw: int
    horizon: int = 5
    strategy: str = "balanced"
    use_wildcard: bool | None = None
    keep: list[int] = Field(default_factory=list)
    exclude: list[int] = Field(default_factory=list)
    save: bool = True
    allow_hits: bool = True  # False — ни одного платного трансфера в горизонте (HITL reject)
    chips: list[PlanChipIn] = Field(default_factory=list)  # пусто — план без фишек


class PlanOut(BaseModel):
    from_gw: int
    horizon: int
    strategy: str
    moves_by_gw: dict[str, list[PlanMoveOut]]
    hits_by_gw: dict[str, int]
    ft_by_gw: dict[str, int]
    bank_by_gw: dict[str, float]
    xi_points_by_gw: dict[str, float]
    captain_by_gw: dict[str, str]
    expected_total: float
    baseline_total: float
    recommendation: str
    target_squad: list[str] = Field(default_factory=list)  # состав к концу горизонта (имена)
    # лучшие 11 из нынешних 15 по турам без трансферов (сумма = baseline_total)
    baseline_xi_points_by_gw: dict[str, float] = Field(default_factory=dict)
    wildcard: dict[str, Any] | None = None
    issues: list[dict[str, Any]] = Field(default_factory=list)
    diff_vs_previous: list[dict[str, Any]] | None = None
    previous_plan_at: str | None = None
    saved_id: int | None = None
    solver: str = ""
    runtime_s: float = 0.0
    time_limit_hit: bool = False
    allow_hits: bool = True
    chips: list[PlanChipOut] = Field(default_factory=list)  # xi_points_by_gw уже с их вкладом
    notes: list[str] = Field(default_factory=list)


class SimulateScenarioInput(BaseModel):
    manager_id: int
    gw: int
    horizon: int = 3
    strategy: str = "balanced"
    sell: list[int] = Field(default_factory=list)
    buy: list[int] = Field(default_factory=list)
    keep: list[int] = Field(default_factory=list)
    exclude: list[int] = Field(default_factory=list)
    allow_hit: bool | None = None
    use_wildcard: bool | None = None


class ScenarioOut(BaseModel):
    gw: int
    horizon: int
    strategy: str
    feasible: bool
    reason: str | None = None
    free_transfers: int
    bank: float
    transfers_cap: int
    primary: RouteOut | None = None  # лучший маршрут, удовлетворяющий сценарию
    free_alternative: RouteOut | None = None  # то же без хита (если отличается)
    hit_alternative: RouteOut | None = None  # «версия с хитом», если её просили, а оптимум — без
    alternatives: list[RouteOut] = Field(default_factory=list)
    hold_xi_points: float | None = None
    wildcard: dict[str, Any] | None = None
    verdict: str = ""
    notes: list[str] = Field(default_factory=list)


class DiagnoseSquadInput(BaseModel):
    manager_id: int
    gw: int
    horizon: int = DIAGNOSIS_HORIZON
    strategy: str = "balanced"


class IssuesOut(BaseModel):
    gw: int
    issues: list[dict[str, Any]]
    problem_players: list[str]


class KBSearchInput(BaseModel):
    """Поиск по стратегической KB (rag/kb): вопрос + необязательный фильтр тегов."""

    query: str
    tags: list[str] = Field(default_factory=list)  # rules, chips, hits, transfers, captaincy, ...
    k: int = 6
    mode: str = "hybrid_rerank"


class KBChunkOut(BaseModel):
    chunk_id: int
    doc_id: int
    title: str
    url: str
    source: str
    tags: list[str] = Field(default_factory=list)
    text: str
    score: float | None = None  # итоговая оценка ранжирования (rerank / rrf / dense)


class KBSearchOutput(BaseModel):
    query: str
    tags: list[str] = Field(default_factory=list)
    mode: str = "hybrid_rerank"
    chunks: list[KBChunkOut] = Field(default_factory=list)
    latency_ms: float = 0.0


class StrategyAnswerInput(BaseModel):
    query: str
    tags: list[str] = Field(default_factory=list)
    k: int = 6


class KBCitation(BaseModel):
    n: int
    chunk_id: int
    title: str
    url: str
    source: str
    tags: list[str] = Field(default_factory=list)
    quote: str


class StrategyAnswerOutput(BaseModel):
    """`rag.kb.answer.answer_strategy_question`: ответ с маркерами [n], проверенные цитаты и
    учёт единственного LLM-вызова (для llm_calls графа)."""

    query: str
    answer: str
    covered: bool
    citations: list[KBCitation] = Field(default_factory=list)
    retrieved: int = 0
    model: str | None = None
    prompt_version: str | None = None
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    latency_ms: float = 0.0
    validation: dict[str, Any] = Field(default_factory=dict)


# ---------- ранжирование игроков всей лиги (интент player_ranking) ----------

RankingMetric = Literal[
    "points_per_million", "total_points", "consistency", "budget_consistency", "form"
]
RANKING_METRICS: tuple[str, ...] = (
    "points_per_million",
    "total_points",
    "consistency",
    "budget_consistency",
    "form",
)
# Окно (сезон / GW4–GW5) в метку не входит — оно идёт отдельно (RankPlayersOutput.window_label).
RANKING_METRIC_LABEL: dict[str, str] = {
    "points_per_million": "points per £1m (total points / current price)",
    "total_points": "total points",
    "consistency": (
        "consistency (share of played GWs with >= 5 pts, then fewer GWs missed, then lowest "
        "std of points)"
    ),
    "budget_consistency": (
        "cheap and consistent (share of played GWs with >= 5 pts, then fewer GWs missed, then "
        "the lower price)"
    ),
    "form": "FPL form (average points over the last 30 days)",
}
POSITIONS: tuple[str, ...] = ("GKP", "DEF", "MID", "FWD")
# ---------- v4 чата: календарь клубов, прогнозный рейтинг, фишки, тренды трансферов ----------


class TeamFixturesInput(BaseModel):
    gw: int
    horizon: int = 3
    team_ids: list[int] = Field(default_factory=list)  # пусто — все клубы (рейтинг календарей)


class TeamFixtureRow(BaseModel):
    rank: int = 0  # 1 = самый лёгкий календарь (наименьший средний FSI)
    team_id: int
    team: str  # код клуба (ARS)
    team_name: str
    fixtures: dict[str, str]  # "GW6": "vLEE (FSI 1)"; DGW — через запятую; без матча — "blank"
    matches: int
    mean_fsi: float | None


class TeamFixturesOut(BaseModel):
    gw_from: int
    gw_to: int
    teams: list[TeamFixtureRow]
    ranking_rule: str = "mean FSI over the window, lower = easier (FSI 1 very easy … 5 very hard)"
    notes: list[str] = Field(default_factory=list)


ForecastMetric = Literal["xpts", "xpts_per_million"]


class ForecastRankInput(BaseModel):
    """Рейтинг «кто БУДЕТ лучшим»: сумма прогноза xPts на горизонт (или xPts на £1m)."""

    gw: int
    horizon: int = 3
    metric: ForecastMetric = "xpts"
    position: str | None = None
    max_price: float | None = None
    min_price: float | None = None
    max_ownership: float | None = None  # дифференциалы: владение не выше, %
    min_p_start: float = 0.5  # p(start) ближайшего тура — отсекает запасных с «дешёвыми» xPts
    limit: int = 8
    squad_ids: list[int] = Field(default_factory=list)
    exclude_unavailable: bool = True


class ForecastRow(BaseModel):
    rank: int = 0
    id: int
    name: str
    full_name: str = ""
    team: str
    position: str
    price: float
    ownership: float
    status: str = "a"
    xpts_total: float
    xpts_per_million: float
    xpts_by_gw: dict[str, float] = Field(default_factory=dict)
    p_start_next: float | None = None
    fixtures: dict[str, str] = Field(default_factory=dict)
    in_squad: bool = False


class ForecastRankOut(BaseModel):
    gw_from: int
    gw_to: int
    metric: str
    metric_label: str
    filters: dict[str, Any] = Field(default_factory=dict)
    filters_definition: str = ""
    candidates: int = 0
    rows: list[ForecastRow] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ChipsStatusInput(BaseModel):
    manager_id: int
    gw: int


class ChipState(BaseModel):
    chip: str  # wildcard | freehit | bboost | 3xc
    name: str
    available_now: bool  # доступна на дедлайн gw
    played_gws: list[int] = Field(default_factory=list)  # туры, где менеджер её сыграл
    window_now: str | None = None  # «GW1–GW19» — окно, в которое попадает gw
    next_available_gw: int | None = None  # если сейчас недоступна: начало следующего окна


class ChipsStatusOut(BaseModel):
    gw: int
    chips: list[ChipState]
    available_now: list[str]  # названия
    notes: list[str] = Field(default_factory=list)


class TransferTrendsInput(BaseModel):
    limit: int = 8
    position: str | None = None


class TrendRow(BaseModel):
    name: str
    team: str
    position: str
    price: float
    ownership: float
    transfers_in_event: int
    transfers_out_event: int
    net_transfers_event: int
    price_change_event: float  # £m за текущий тур (cost_change_event / 10)


class TransferTrendsOut(BaseModel):
    gw: int  # тур, к которому относятся трансферы (ближайший дедлайн)
    most_transferred_in: list[TrendRow]
    most_transferred_out: list[TrendRow]
    notes: list[str] = Field(default_factory=list)


class GWReviewInput(BaseModel):
    manager_id: int
    gw: int  # завершённый тур


class GWReviewRow(BaseModel):
    name: str
    team: str
    position: str
    role: str  # C | VC | XI | bench
    multiplier: int  # 0 скамейка, 1, 2 капитан, 3 TC
    points: int | None  # фактические очки игрока за тур (player_gw_history), None — нет строки
    counted_points: int | None  # очки в зачёт менеджера = points × multiplier
    minutes: int | None
    xpts_forecast: float | None  # последний прогноз модели до дедлайна тура (xpts_predictions)
    diff_vs_forecast: float | None  # points - xpts_forecast


class GWReviewOut(BaseModel):
    gw: int
    finished: bool
    manager_points: int | None = None  # очки менеджера за тур (entry history, с учётом хитов нет)
    transfers_cost: int | None = None
    points_on_bench: int | None = None
    average_points: int | None = None  # средний результат тура по FPL
    active_chip: str | None = None
    rows: list[GWReviewRow] = Field(default_factory=list)
    captain: str | None = None
    captain_points: int | None = None  # в зачёт (× множитель)
    best_starter: str | None = None  # больше всех очков среди стартовых
    best_starter_points: int | None = None
    best_bench: str | None = None
    best_bench_points: int | None = None
    forecast_available: bool = False
    history_available: bool = False
    notes: list[str] = Field(default_factory=list)


_FORECAST_BEFORE_DEADLINE = text(
    """
    SELECT DISTINCT ON (player_id) player_id, xpts
    FROM xpts_predictions
    WHERE gw = :gw AND model_version = :mv AND as_of <= :deadline
    ORDER BY player_id, as_of DESC, id DESC
    """
)


CHIP_ORDER = ("wildcard", "freehit", "bboost", "3xc")


RETURN_POINTS = 5  # «вернул очки» за тур: >= 5
FLOOR_POINTS = 2  # «сыграл без провала»: >= 2 (60+ минут без отрицательных событий)
MIN_MINUTES_PER_GW = 45  # дефолтный фильтр минут: 45 на каждый завершённый тур (в окне)
RANK_LIMIT_MAX = 25
MAX_GW = 38
SEASON_WINDOW_LABEL = "season to date"
UNAVAILABLE_STATUSES = frozenset({"i", "s", "u", "n"})


class HistoryUnavailable(RuntimeError):
    """Окно туров запрошено, а player_gw_history недоступна или не покрывает окно: явная ошибка
    вместо тихого отката к сезонным итогам bootstrap."""


class RankPlayersInput(BaseModel):
    """Ранжирование по всей лиге: фильтры позиции / цены / минут, метрика, top-N, окно туров.

    Окно: `gw_from` / `gw_to` (одно из них -> один тур: «в GW5»; «gw4 и gw5» -> 4..5) либо
    `last_n_gws` («последние 3 тура» — относительно последнего завершённого тура, разрешает
    код по bootstrap). При окне очки / минуты / pts/£m / стабильность считаются ТОЛЬКО по
    player_gw_history за эти туры; без окна — сезонные итоги bootstrap + стабильность по всей
    локальной истории."""

    gw: int  # предстоящий тур: история берётся только по раундам < gw (leakage guard)
    metric: RankingMetric = "points_per_million"
    position: str | None = None  # GKP | DEF | MID | FWD
    max_price: float | None = None  # £m включительно
    min_price: float | None = None
    min_minutes: int | None = None  # None -> MIN_MINUTES_PER_GW * туров в окне (или начавшихся)
    exclude_unavailable: bool = True  # статусы i / s / u / n не показываем
    limit: int = 10
    squad_ids: list[int] = Field(default_factory=list)  # флаг in_squad в строках
    gw_from: int | None = None  # окно туров (включительно); оба None и last_n_gws None -> сезон
    gw_to: int | None = None
    last_n_gws: int | None = None  # «последние N туров» до последнего завершённого включительно


class RankedPlayerRow(BaseModel):
    rank: int = 0
    id: int
    name: str
    full_name: str = ""  # объяснитель может назвать полное имя — валидатор его знает
    team: str
    position: str
    price: float
    status: str = "a"
    ownership: float = 0.0
    total_points: int  # сезон (bootstrap) либо сумма за окно туров (player_gw_history)
    minutes: int  # то же окно, что и total_points
    points_per_game: float
    points_per_million: float  # total_points / текущая цена
    form: float | None = None  # FPL form (30 дней) — окно туров не учитывает
    season_total_points: int | None = None  # только при окне: сезонные очки для контекста
    gws_played: int = 0  # туров с минутами > 0 в локальной истории
    gws_dnp: int = 0  # матчей команды, где игрок не вышел (0 минут)
    gws_in_history: int = 0  # завершённых туров в окне истории
    gws_5plus: int = 0
    share_5plus_pct: int = 0  # % сыгранных туров с >= RETURN_POINTS очков
    gws_2plus: int = 0
    share_2plus_pct: int = 0
    std_points: float | None = None  # стандартное отклонение очков по сыгранным турам
    min_points: int | None = None
    max_points: int | None = None
    points_by_gw: dict[str, int] = Field(default_factory=dict)  # "GW3": 6 (только сыгранные)
    in_squad: bool = False


class RankPlayersOutput(BaseModel):
    gw: int
    metric: str
    metric_label: str
    window_label: str = SEASON_WINDOW_LABEL  # «season to date» | «GW5» | «GW4–GW5»
    gw_from: int | None = None  # окно туров (включительно); None — весь сезон
    gw_to: int | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    history_through_gw: int | None = None  # последний раунд в player_gw_history
    history_available: bool = True
    candidates: int = 0  # игроков после фильтров (до обрезки limit)
    rows: list[RankedPlayerRow] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def window_label(gw_from: int | None, gw_to: int | None) -> str:
    """«GW5» / «GW4–GW5» / «season to date» — одна форма для headline, facts и оговорок."""
    if gw_from is None or gw_to is None:
        return SEASON_WINDOW_LABEL
    return f"GW{gw_from}" if gw_from == gw_to else f"GW{gw_from}–GW{gw_to}"


def resolve_window(
    gw_from: int | None,
    gw_to: int | None,
    last_n_gws: int | None,
    *,
    upcoming_gw: int,
    last_finished_gw: int | None,
) -> tuple[int, int] | None:
    """Окно туров рейтинга -> (от, до) включительно или None (весь сезон). Чистая функция.

    Явное окно главнее «последних N»; одно заданное число -> один тур; перепутанные границы
    меняются местами. «Последние N» отсчитываются от последнего завершённого тура (bootstrap
    `events[].finished`), не от предстоящего. Окно обязано закончиться до предстоящего тура
    (leakage guard) — иначе ValueError с понятным текстом для оговорки."""
    if gw_from is None and gw_to is None:
        if last_n_gws is None:
            return None
        if last_n_gws < 1:
            raise ValueError(f"last_n_gws must be >= 1, got {last_n_gws}")
        if last_finished_gw is None:
            raise ValueError("no gameweek has finished yet — 'last N gameweeks' is empty")
        gw_to = last_finished_gw
        gw_from = max(1, gw_to - last_n_gws + 1)
    lo = gw_from if gw_from is not None else gw_to
    hi = gw_to if gw_to is not None else gw_from
    assert lo is not None and hi is not None
    if lo > hi:
        lo, hi = hi, lo
    if lo < 1 or hi > MAX_GW:
        raise ValueError(f"gameweek window {window_label(lo, hi)} is outside GW1–GW{MAX_GW}")
    if hi >= upcoming_gw:
        raise ValueError(
            f"GW{hi} has not been played yet (the upcoming gameweek is GW{upcoming_gw}); a "
            "ranking window must end before it"
        )
    return lo, hi


def consistency_stats(
    by_round: dict[int, tuple[int, int]], rounds_in_history: Sequence[int]
) -> dict[str, Any]:
    """Стабильность по строкам player_gw_history одного игрока: round -> (очки, минуты) (DGW
    уже просуммирован). Считаются только туры с минутами > 0. Чистая функция без БД."""
    played = {r: pts for r, (pts, mins) in sorted(by_round.items()) if mins > 0}
    pts = list(played.values())
    n = len(pts)
    std: float | None = None
    if n:
        mean = sum(pts) / n
        std = round(math.sqrt(sum((x - mean) ** 2 for x in pts) / n), 2)
    gws_5plus = sum(1 for x in pts if x >= RETURN_POINTS)
    gws_2plus = sum(1 for x in pts if x >= FLOOR_POINTS)
    return {
        "gws_played": n,
        "gws_dnp": sum(1 for _, mins in by_round.values() if mins <= 0),
        "gws_in_history": len(rounds_in_history),
        "gws_5plus": gws_5plus,
        "share_5plus_pct": round(100 * gws_5plus / n) if n else 0,
        "gws_2plus": gws_2plus,
        "share_2plus_pct": round(100 * gws_2plus / n) if n else 0,
        "std_points": std,
        "min_points": min(pts) if n else None,
        "max_points": max(pts) if n else None,
        "points_by_gw": {f"GW{r}": p for r, p in played.items()},
    }


def rank_player_rows(rows: Iterable[RankedPlayerRow], metric: str) -> list[RankedPlayerRow]:
    """Детерминированная сортировка по метрике (ключи описаны в RANKING_METRIC_LABEL) + rank."""
    if metric not in RANKING_METRICS:
        raise ValueError(f"unknown ranking metric {metric!r}; use one of {RANKING_METRICS}")
    inf = float("inf")

    def key(r: RankedPlayerRow) -> tuple:
        if metric == "points_per_million":
            return (-r.points_per_million, -r.total_points, r.price, r.id)
        if metric == "total_points":
            return (-r.total_points, -r.points_per_million, r.id)
        if metric == "form":
            return (-(r.form or 0.0), -r.total_points, r.id)
        std = r.std_points if r.std_points is not None else inf
        if metric == "budget_consistency":  # «дешёвый, но стабильный»: стабильность, потом цена
            return (-r.share_5plus_pct, r.gws_dnp, r.price, std, -r.points_per_game, r.id)
        return (-r.share_5plus_pct, r.gws_dnp, std, -r.points_per_game, -r.total_points, r.id)

    ranked = sorted(rows, key=key)
    return [r.model_copy(update={"rank": i}) for i, r in enumerate(ranked, start=1)]


class SquadUnavailable(RuntimeError):
    """Состав менеджера недоступен (нет публичных picks / менеджер не найден)."""


class ScenarioInfeasible(RuntimeError):
    pass


# ---------- состав не из API (скриншот) ----------


class SquadOverride(BaseModel):
    """Компактное описание состава, полученного не из публичного API (скриншот «Pick Team»,
    `vision.to_squad`): picks + банк + FT. JSON-совместимо — лежит в состоянии агента и в
    чекпоинтах; полный `Squad` восстанавливается по живому bootstrap (`to_squad`)."""

    source: str = "screenshot"
    gw: int
    picks: list[Pick]
    bank: float
    team_value: float
    free_transfers: int | None = None
    chips_used: list[str] = Field(default_factory=list)

    @classmethod
    def from_squad(cls, squad: Squad, *, source: str = "screenshot") -> SquadOverride:
        picks = [p.pick for p in sorted(squad.players, key=lambda p: p.pick.position)]
        return cls(
            source=source,
            gw=squad.gw,
            picks=picks,
            bank=squad.bank,
            team_value=squad.team_value,
            free_transfers=squad.free_transfers,
            chips_used=list(squad.chips_used),
        )

    def to_squad(self, bs: Bootstrap, manager_id: int = 0) -> Squad:
        players = [
            SquadPlayer(
                pick=pk, player=bs.player(pk.element), team=bs.team(bs.player(pk.element).team)
            )
            for pk in self.picks
        ]
        return Squad(
            manager_id=manager_id,
            gw=self.gw,
            players=players,
            bank=self.bank,
            team_value=self.team_value,
            active_chip=None,
            free_transfers=self.free_transfers,
            chips_used=list(self.chips_used),
        )

    def fingerprint(self) -> str:
        payload = {
            "gw": self.gw,
            "picks": [(p.element, p.position, p.multiplier) for p in self.picks],
            "bank": round(self.bank, 1),
            "ft": self.free_transfers,
            "chips": sorted(self.chips_used),
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


SquadLike = Squad | SquadOverride


def as_override(squad: SquadLike | None, *, source: str = "screenshot") -> SquadOverride | None:
    if squad is None or isinstance(squad, SquadOverride):
        return squad
    return SquadOverride.from_squad(squad, source=source)


class _ClientWithSquad:
    """Прокси FPLClient: `.squad()` отдаёт заданный состав, всё остальное — у настоящего клиента.
    Так `optimizer.load_inputs` получает состав со скриншота без изменений в core/."""

    def __init__(self, client: FPLClient, squad: Squad) -> None:
        self._client = client
        self._squad = squad

    def squad(self, manager_id: int, gw: int | None = None) -> Squad:
        return self._squad

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


# ---------- протокол ----------


class AgentTools(Protocol):
    def get_gameweek_context(self, inp: GameweekContextInput) -> GameweekContext: ...

    def predict_player(self, inp: PredictPlayerInput) -> PlayerPrediction: ...

    def compare_players(self, inp: ComparePlayersInput) -> ComparePlayersOutput: ...

    def analyze_player_risk(self, inp: PlayerRiskInput) -> PlayerRisk: ...

    def optimize_team(self, inp: OptimizeTeamInput) -> LineupOut: ...

    def recommend_transfers(self, inp: RecommendTransfersInput) -> RoutesOut: ...

    def build_gameweek_plan(self, inp: BuildPlanInput) -> PlanOut: ...

    def simulate_scenario(self, inp: SimulateScenarioInput) -> ScenarioOut: ...

    def diagnose_squad(self, inp: DiagnoseSquadInput) -> IssuesOut: ...

    def search_strategy_kb(self, inp: KBSearchInput) -> KBSearchOutput: ...

    def answer_strategy_question(self, inp: StrategyAnswerInput) -> StrategyAnswerOutput: ...

    def rank_players(self, inp: RankPlayersInput) -> RankPlayersOutput: ...

    # v4 чата (фейки тестов могут их не иметь — граф проверяет через getattr)
    def team_fixtures(self, inp: TeamFixturesInput) -> TeamFixturesOut: ...

    def rank_forecast(self, inp: ForecastRankInput) -> ForecastRankOut: ...

    def chips_status(self, inp: ChipsStatusInput) -> ChipsStatusOut: ...

    def transfer_trends(self, inp: TransferTrendsInput) -> TransferTrendsOut: ...

    def review_gameweek(self, inp: GWReviewInput) -> GWReviewOut: ...

    def invalidate(self) -> None: ...


TOOL_NAMES: tuple[str, ...] = (
    "get_gameweek_context",
    "predict_player",
    "compare_players",
    "analyze_player_risk",
    "optimize_team",
    "recommend_transfers",
    "build_gameweek_plan",
    "simulate_scenario",
    "diagnose_squad",
    "search_strategy_kb",
    "rank_players",
)
# Теги KB, которыми граф заземляет решения о хите / чипе (facts.rules_context), docs/strategy_kb.md.
KB_TAGS_HITS: tuple[str, ...] = ("hits",)
KB_TAGS_CHIPS: tuple[str, ...] = ("chips",)


# ---------- детерминированные помощники (без БД, тестируются отдельно) ----------


def captain_tag(ownership: float) -> CaptainTag:
    if ownership > SAFE_OWNERSHIP:
        return "safe"
    if ownership >= DIFFERENTIAL_OWNERSHIP:
        return "balanced"
    return "differential"


def fixture_label(pred: XPtsBreakdown | None) -> str:
    if pred is None or not pred.fixtures:
        return "blank"
    return " ".join(
        f"{'v' if f.is_home else '@'}{f.opponent} (FSI {f.fixture_strength_index})"
        for f in pred.fixtures
    )


def route_out(route: TransferRoute, rank: int) -> RouteOut:
    return RouteOut(
        rank=rank,
        out=list(route.out_names),
        in_=list(route.in_names),
        out_ids=list(route.out),
        in_ids=list(route.in_),
        hit_cost=route.hit_cost,
        gain_next_gw=r2(route.expected_gain_next_gw) or 0.0,
        gain_horizon=r2(route.expected_gain_horizon) or 0.0,
        gain_discounted=r2(route.expected_gain_discounted) or 0.0,
        objective_gain=r2(route.objective_gain) or 0.0,
        hit_marginal_gain=r2(route.hit_marginal_gain),
        new_bank=r2(route.new_bank) or 0.0,
        risk_note=route.risk_note,
        verdict=route.verdict,
        xi_points_after=r2(route.lineup_after.expected_points) or 0.0,
    )


def risk_note(
    pool: dict[int, Candidate],
    gw: int,
    out: Sequence[int],
    inn: Sequence[int],
    issues: Sequence[SquadIssue] | None,
) -> str:
    """Короткая заметка о рисках маршрута (та же логика, что в optimizer._risk_note)."""
    notes: list[str] = []
    by_player: dict[int, SquadIssue] = {}
    for i in issues or []:
        by_player.setdefault(i.player_id, i)
    for p in out:
        if p in by_player:
            notes.append(f"out {pool[p].name}: {by_player[p].kind} ({by_player[p].detail})")
    for p in inn:
        c = pool[p]
        if c.status != "a":
            notes.append(f"in {c.name}: FPL status {c.status}")
        sd = math.sqrt(max(0.0, c.variance(gw)))
        if sd >= 3.5:
            notes.append(f"in {c.name}: high variance (sd {sd:.1f})")
        if c.ownership and c.ownership < 5:
            notes.append(f"in {c.name}: differential ({c.ownership:.1f}% owned)")
        elif c.ownership >= 40:
            notes.append(f"in {c.name}: template ({c.ownership:.0f}% owned)")
    return "; ".join(notes)


def constrained_routes(
    inputs: ManagerInputs,
    gw: int,
    horizon: int,
    strategy: str,
    *,
    force_in: Sequence[int] = (),
    force_out: Sequence[int] = (),
    keep: Iterable[int] = (),
    exclude: Iterable[int] = (),
    allow_hit: bool | None = None,
    top: int = 3,
    min_transfers: int | None = None,
) -> tuple[list[TransferRoute], dict[str, Any]]:
    """Top-`top` маршрутов трансфера в туре gw при обязательных покупках/продажах.

    Та же MILP, что optimizer.single_transfer (ModelSpec/Model), плюс ограничения
    t_in[p, gw] = 1 для force_in и t_out[p, gw] = 1 для force_out; альтернативы — no-good cuts.
    Лимит трансферов: 1 (2 при FT >= 2), +1 при allow_hit; если сценарий требует больше ходов,
    лимит поднимается до их числа (хиты неизбежны) — кроме allow_hit=False (ScenarioInfeasible).
    min_transfers — «версия с хитом» по просьбе пользователя (Σ t_in >= free_cap + 1).
    Вердикт по хиту — от лучшего бесплатного варианта при тех же ограничениях, как в оптимизаторе.
    """
    S = get_strategy(strategy)
    gws = list(range(gw, gw + horizon))
    pool = dict(inputs.pool)
    for p in force_in:
        pool.setdefault(p, inputs.cands[p])
    free_cap = 1 if inputs.free_transfers < 2 else 2
    n_forced = max(len(force_in), len(force_out))
    cap = free_cap + (1 if allow_hit else 0)
    if n_forced > cap:
        if allow_hit is False:
            raise ScenarioInfeasible(
                f"scenario needs {n_forced} transfers but only {free_cap} free transfer(s) "
                "and paid transfers were rejected"
            )
        cap = n_forced
    keep_set = frozenset(keep) - set(force_out)
    exclude_set = frozenset(exclude) - set(force_in)

    def spec_for(c: int, *, transfers: bool = True) -> ModelSpec:
        return ModelSpec(
            gws=gws,
            pool=pool,
            initial_squad=list(inputs.squad),
            bank=inputs.bank,
            free_transfers=inputs.free_transfers,
            strategy=S,
            max_transfers={w: ((c if w == gw else 0) if transfers else 0) for w in gws},
            keep=keep_set,
            exclude=exclude_set,
        )

    def build(c: int, *, at_least: int | None = None) -> Model:
        m = Model(spec_for(c))
        for p in force_in:
            m.prob += m.v.tin[(p, gw)] == 1, f"force_in_{p}"
        for p in force_out:
            m.prob += m.v.tout[(p, gw)] == 1, f"force_out_{p}"
        m.require_transfer(gw)
        if at_least is not None and at_least > 1:
            m.prob += (
                pulp.lpSum(v for (p, w), v in m.v.tin.items() if w == gw) >= at_least,
                f"min_transfers_{gw}",
            )
        return m

    baseline, _ = Model(spec_for(0, transfers=False)).solve()

    def points_d(sol: Solution) -> float:
        return sol.horizon_points(decay=S.decay_base, from_gw=gw)

    ref_obj, ref_gain_d = baseline.objective, 0.0
    free_sol: Solution | None = None
    if cap > free_cap and n_forced <= free_cap:
        try:
            free_sol, _ = build(free_cap).solve()
        except InfeasibleError:
            free_sol = None
        if free_sol is not None and free_sol.objective > ref_obj:
            ref_obj = free_sol.objective
            ref_gain_d = points_d(free_sol) - points_d(baseline)

    def to_route(sol: Solution) -> TransferRoute:
        out, inn = sol.transfers[gw]
        hit = HIT_COST * sol.paid[gw]
        gain_h = sol.horizon_points() - baseline.horizon_points()
        gain_d = points_d(sol) - points_d(baseline)
        gain_next = sol.lineups[gw].expected_points - baseline.lineups[gw].expected_points
        obj_gain = sol.objective - baseline.objective
        worth_hit: bool | None = None
        marginal: float | None = None
        if hit:
            marginal = gain_d - ref_gain_d - hit
            worth_hit = sol.objective > ref_obj and marginal >= S.hit_threshold
            verdict = "go" if worth_hit else "hit_not_worth"
        else:
            verdict = "go" if obj_gain > 0 else "hold"
        return TransferRoute(
            gw=gw,
            out=out,
            in_=inn,
            out_names=[pool[p].name for p in out],
            in_names=[pool[p].name for p in inn],
            hit_cost=hit,
            expected_gain_horizon=round(gain_h, 3),
            expected_gain_discounted=round(gain_d, 3),
            expected_gain_next_gw=round(gain_next, 3),
            objective_gain=round(obj_gain, 3),
            hit_marginal_gain=None if marginal is None else round(marginal, 3),
            new_bank=sol.itb[gw],
            risk_note=risk_note(pool, gw, out, inn, inputs.issues),
            worth_hit=worth_hit,
            verdict=verdict,
            lineup_after=sol.lineups[gw],
        )

    model = build(cap, at_least=min_transfers)
    routes: list[TransferRoute] = []
    for _ in range(top):
        try:
            sol, _ = model.solve()
        except InfeasibleError:
            break
        out, inn = sol.transfers[gw]
        if not inn:
            break
        routes.append(to_route(sol))
        model.add_no_good_cut(gw, out, inn)
    meta = {
        "cap": cap,
        "free_cap": free_cap,
        "baseline_xi_points": r2(baseline.lineups[gw].expected_points),
        "free_route": to_route(free_sol) if free_sol is not None else None,
    }
    return routes, meta


# ---------- реальная реализация ----------

_LATEST_SIGNAL = text(
    """
    SELECT as_of, availability, start_probability, expected_minutes, rotation_risk, return_gw,
           confidence, summary, evidence, fpl_status, fpl_chance_next, abstained, mode, model
    FROM player_signals
    WHERE player_id = :pid AND as_of <= :as_of
    ORDER BY as_of DESC, id DESC
    LIMIT 1
    """
)


class LiveTools:
    """Инструменты над живыми данными: FPL API (дисковый кэш), Postgres, xPts v0, HiGHS, RAG."""

    def __init__(self, client: FPLClient | None = None, *, now: datetime | None = None) -> None:
        self.client = client or FPLClient()
        self._now = now
        self._inputs: dict[tuple[Any, ...], ManagerInputs] = {}
        self._store: PredictionStore | None = None
        self._retriever: Any = None
        self._kb_retriever: Any = None
        self._squads: dict[tuple[int, int], Any] = {}

    # ---- служебное ----

    def now(self) -> datetime:
        return self._now or datetime.now(UTC)

    @property
    def bootstrap(self) -> Bootstrap:
        return self.client.bootstrap()

    @property
    def store(self) -> PredictionStore:
        if self._store is None:
            self._store = PredictionStore(self.client)
        return self._store

    def invalidate(self) -> None:
        """Сбросить кэш прогнозов/входов: новые сигналы должны попасть в модель минут."""
        self._inputs.clear()
        self._store = None

    def close(self) -> None:
        """Освободить тяжёлые ресурсы (ONNX-сессия reranker'а, кэши, HTTP-клиент) до завершения
        интерпретатора: onnxruntime, разрушаемый при teardown после остановки потоков, падает с
        `recursive_mutex lock failed` на macOS."""
        self._retriever = None
        self._kb_retriever = None
        self.invalidate()
        self._squads.clear()
        try:
            self.client.close()
        except Exception:  # закрытие best-effort, ответ уже напечатан
            log.debug("FPLClient.close failed", exc_info=True)
        gc.collect()

    def next_gw(self) -> int:
        bs = self.bootstrap
        if bs.next_event is not None:
            return bs.next_event.id
        return bs.current_event.id if bs.current_event else 1

    def player_ref(self, player: Player) -> PlayerRef:
        bs = self.bootstrap
        return PlayerRef(
            id=player.id,
            name=player.web_name,
            full_name=player.full_name,
            team=bs.team(player.team).short_name,
            position=player.position.short,
            price=player.price,
            status=player.status,
            chance=player.chance_of_playing_next_round,
            ownership=float(player.selected_by_percent or 0.0),
            news=player.news or "",
        )

    def override_squad(self, manager_id: int, squad_override: SquadLike | None) -> Squad | None:
        """`Squad` / `SquadOverride` -> полный `Squad` по живому bootstrap (None -> None)."""
        if squad_override is None:
            return None
        if isinstance(squad_override, SquadOverride):
            return squad_override.to_squad(self.bootstrap, manager_id)
        return squad_override

    def squad_override_from(
        self, squad: Squad, *, manager_id: int | None = None, source: str = "screenshot"
    ) -> SquadOverride:
        """Состав из `vision.to_squad` -> `SquadOverride`. Сыгранные чипы (нужны правилу Wildcard)
        со скриншота не видны — берём из публичной истории менеджера, если она есть."""
        chips = list(squad.chips_used)
        if not chips and manager_id:
            try:
                chips = [c.name for c in self.client.history(manager_id).chips]
            except Exception:  # менеджер без истории / API недоступен — считаем WC доступным
                log.debug("history unavailable for %s", manager_id, exc_info=True)
        return SquadOverride.from_squad(squad, source=source).model_copy(
            update={"chips_used": chips}
        )

    def inputs(
        self,
        manager_id: int,
        gw: int,
        horizon: int,
        strategy: str,
        *,
        exclude: Iterable[int] = (),
        wildcard: bool | None = None,
        squad_override: SquadLike | None = None,
    ) -> ManagerInputs:
        override = as_override(squad_override)
        fp = override.fingerprint() if override is not None else None
        key = (manager_id, gw, horizon, strategy, tuple(sorted(exclude)), wildcard, fp)
        if key not in self._inputs:
            client: Any = self.client
            if override is not None:
                client = _ClientWithSquad(
                    self.client, override.to_squad(self.bootstrap, manager_id)
                )
            try:
                self._inputs[key] = load_inputs(
                    manager_id,
                    gw,
                    horizon,
                    strategy,
                    client=client,
                    exclude=exclude,
                    wildcard=wildcard,
                )
            except httpx.HTTPStatusError as exc:
                raise SquadUnavailable(self._squad_note(manager_id, gw, exc)) from exc
        return self._inputs[key]

    def _squad_note(self, manager_id: int, gw: int, exc: httpx.HTTPStatusError) -> str:
        code = exc.response.status_code
        if code != 404:
            return f"FPL API returned {code} for manager {manager_id}"
        try:
            entry = self.client.entry(manager_id)
        except httpx.HTTPStatusError:
            return f"manager {manager_id} not found in FPL API"
        started = entry.started_event
        if started is not None and started >= gw:
            return (
                f"manager {manager_id} ('{entry.name}') has no public picks yet: the team starts "
                f"in GW{started}, picks become visible after that deadline"
            )
        return f"no public picks for manager {manager_id} before GW{gw}"

    def _squad(
        self, manager_id: int, squad_gw: int, squad_override: SquadLike | None = None
    ) -> Any:
        if squad_override is not None:
            return self.override_squad(manager_id, squad_override)
        key = (manager_id, squad_gw)
        if key not in self._squads:
            self._squads[key] = self.client.squad(manager_id, squad_gw)
        return self._squads[key]

    # ---- инструменты ----

    def get_gameweek_context(
        self, inp: GameweekContextInput, *, squad_override: SquadLike | None = None
    ) -> GameweekContext:
        bs = self.bootstrap
        gw = self.next_gw()
        event = next(e for e in bs.events if e.id == gw)
        ctx = GameweekContext(
            gw=gw,
            current_gw=bs.current_event.id if bs.current_event else None,
            deadline=event.deadline_time,
            as_of=self.now(),
        )
        if inp.manager_id is None and squad_override is None:
            ctx.squad_note = "no manager id given: squad-level tools are unavailable"
            return ctx
        manager_id = inp.manager_id if inp.manager_id is not None else 0
        try:
            inputs = self.inputs(
                manager_id, gw, DIAGNOSIS_HORIZON, inp.strategy, squad_override=squad_override
            )
        except SquadUnavailable as exc:
            ctx.squad_note = str(exc)
            return ctx
        squad = self._squad(manager_id, inputs.squad_gw, squad_override)
        preds = inputs.store.get(gw)
        rows = []
        for sp in sorted(squad.players, key=lambda s: s.pick.position):
            ref = self.player_ref(sp.player)
            pred = preds.get(sp.player.id)
            rows.append(
                SquadPlayerRow(
                    **ref.model_dump(),
                    is_starting=sp.pick.is_starting,
                    is_captain=sp.pick.is_captain,
                    is_vice=sp.pick.is_vice_captain,
                    xpts_next=r2(pred.xpts) if pred else None,
                )
            )
        ctx.squad_gw = inputs.squad_gw
        ctx.bank = round(inputs.bank, 1)
        ctx.free_transfers = inputs.free_transfers
        ctx.chips_available = list(inputs.chips_available)
        ctx.squad = rows
        ctx.squad_note = (
            f"squad = picks of GW{inputs.squad_gw} (public API shows transfers of the current "
            "gameweek only after its deadline)"
        )
        if squad_override is not None:
            ctx.squad_gw = squad.gw
            ft = squad.free_transfers if squad.free_transfers is not None else "unknown"
            ctx.squad_note = (
                f"squad = uploaded screenshot (draft for GW{squad.gw}: bank £{squad.bank:.1f}, "
                f"FT {ft}; not the public GW{inputs.squad_gw} picks)"
            )
        ctx.issues = [i.model_dump() for i in inputs.issues]
        return ctx

    def diagnose_squad(
        self, inp: DiagnoseSquadInput, *, squad_override: SquadLike | None = None
    ) -> IssuesOut:
        inputs = self.inputs(
            inp.manager_id, inp.gw, inp.horizon, inp.strategy, squad_override=squad_override
        )
        problems = problem_players(inputs.issues)
        return IssuesOut(
            gw=inp.gw,
            issues=[i.model_dump() for i in inputs.issues],
            problem_players=[inputs.cands[p].name for p in sorted(problems)],
        )

    def _gw_prediction(self, pred: XPtsBreakdown) -> GWPrediction:
        return GWPrediction(
            gw=pred.gw,
            xpts=r2(pred.xpts) or 0.0,
            sd=r2(pred.sd) or 0.0,
            p_start=r2(pred.p_start) or 0.0,
            exp_minutes=round(pred.exp_minutes),
            components={k: round(v, 2) for k, v in pred.components.model_dump().items()},
            fixtures=[
                FixtureBrief(
                    opponent=f.opponent,
                    is_home=f.is_home,
                    fsi=f.fixture_strength_index,
                    xg_for=r2(f.xg_for) or 0.0,
                    xg_against=r2(f.xg_against) or 0.0,
                    clean_sheet_prob=r2(f.clean_sheet_prob) or 0.0,
                    fsi_source=f.fsi_source,
                )
                for f in pred.fixtures
            ],
            ep_next=pred.ep_next,
            notes=list(pred.notes),
        )

    def predict_player(self, inp: PredictPlayerInput) -> PlayerPrediction:
        player = self.bootstrap.player(inp.player_id)
        by_gw: list[GWPrediction] = []
        signal_used = None
        for g in range(inp.gw, inp.gw + max(1, inp.horizon)):
            pred = self.store.get(g).get(inp.player_id)
            if pred is None:
                continue
            by_gw.append(self._gw_prediction(pred))
            if signal_used is None and pred.signal:
                signal_used = pred.signal
        return PlayerPrediction(
            player=self.player_ref(player),
            by_gw=by_gw,
            total_xpts=round(sum(p.xpts for p in by_gw), 2),
            signal_used=signal_used,
        )

    def compare_players(self, inp: ComparePlayersInput) -> ComparePlayersOutput:
        preds = [
            self.predict_player(PredictPlayerInput(player_id=pid, gw=inp.gw, horizon=inp.horizon))
            for pid in inp.player_ids
        ]
        ranking = sorted(
            ({"id": p.player.id, "name": p.player.name, "total_xpts": p.total_xpts} for p in preds),
            key=lambda r: -r["total_xpts"],
        )
        return ComparePlayersOutput(players=preds, ranking=ranking)

    # ---- ранжирование игроков всей лиги ----

    def _history_by_player(self, gw: int) -> tuple[dict[int, dict[int, tuple[int, int]]], bool]:
        """player_id -> {round: (очки, минуты)} по раундам < gw (DGW суммируется).
        БД недоступна -> ({}, False): ранжирование продолжается по bootstrap без стабильности."""
        from fplcopilot.core.history import load_history

        out: dict[int, dict[int, tuple[int, int]]] = {}
        try:
            rows = load_history(before_gw=gw)
        except SQLAlchemyError as exc:
            log.warning("player_gw_history unavailable: %s", exc)
            return out, False
        for r in rows:
            per_round = out.setdefault(r.element, {})
            pts, mins = per_round.get(r.round, (0, 0))
            per_round[r.round] = (pts + int(r.total_points), mins + int(r.minutes))
        return out, True

    def rank_players(self, inp: RankPlayersInput) -> RankPlayersOutput:
        """Детерминированный рейтинг игроков лиги: bootstrap (total_points, now_cost, minutes,
        form, ppg) + player_gw_history (доля туров с >= 5 / >= 2 очков, std очков). Без LLM.

        С окном туров (gw_from / gw_to / last_n_gws) очки, минуты, pts/£m, ppg и стабильность
        считаются только по player_gw_history за туры окна; порог минут — 45 × туров окна.
        Нет истории или она не покрывает окно -> HistoryUnavailable (без отката к сезону)."""
        bs = self.bootstrap
        now = self.now()
        # Туры, чей дедлайн уже прошёл (сыграны или идут): окно сезонных сумм bootstrap.
        started = sorted(e.id for e in bs.events if e.id < inp.gw and e.deadline_time <= now)
        finished = [e.id for e in bs.events if e.id < inp.gw and e.finished]
        last_finished = max(finished) if finished else None
        window = resolve_window(
            inp.gw_from,
            inp.gw_to,
            inp.last_n_gws,
            upcoming_gw=inp.gw,
            last_finished_gw=last_finished,
        )
        history, hist_ok = self._history_by_player(inp.gw)
        rounds = sorted({r for per_round in history.values() for r in per_round})
        notes: list[str] = []
        window_rounds: list[int] = []
        label = SEASON_WINDOW_LABEL
        if window is not None:
            lo, hi = window
            label = window_label(lo, hi)
            if not hist_ok:
                raise HistoryUnavailable(
                    f"a gameweek window ({label}) needs player_gw_history, but the database is "
                    "unavailable — season totals are not a substitute for per-gameweek points"
                )
            missing = [g for g in range(lo, hi + 1) if g not in rounds]
            if missing:
                covered = f"GW1–GW{rounds[-1]}" if rounds else "no gameweek"
                raise HistoryUnavailable(
                    f"a gameweek window ({label}) needs match history for "
                    f"{', '.join(f'GW{g}' for g in missing)}, but player_gw_history covers "
                    f"{covered} — run `python -m fplcopilot.core.history --sync`"
                )
            window_rounds = list(range(lo, hi + 1))
            if last_finished is not None and hi > last_finished:
                notes.append(f"GW{hi} is still in progress: its points and minutes are partial")
            if inp.metric == "form":
                notes.append(
                    "FPL form is a 30-day average and ignores the gameweek window; the other "
                    f"columns cover {label} only"
                )
        min_minutes = (
            inp.min_minutes
            if inp.min_minutes is not None
            else MIN_MINUTES_PER_GW * max(1, len(window_rounds) if window else len(started))
        )
        position = inp.position.upper() if inp.position else None
        if position is not None and position not in POSITIONS:
            raise ValueError(f"unknown position {position!r}; use one of {POSITIONS}")
        limit = max(1, min(RANK_LIMIT_MAX, int(inp.limit)))
        squad = set(inp.squad_ids)
        rows: list[RankedPlayerRow] = []
        for p in bs.elements:
            if inp.exclude_unavailable and p.status in UNAVAILABLE_STATUSES:
                continue
            if position is not None and p.position.short != position:
                continue
            if inp.max_price is not None and p.price > inp.max_price + 1e-9:
                continue
            if inp.min_price is not None and p.price < inp.min_price - 1e-9:
                continue
            per_round = history.get(p.id, {})
            season_total: int | None = None
            if window is not None:
                # Только туры окна: очки/минуты/ppg из истории, сезонные итоги — для контекста
                per_round = {r: v for r, v in per_round.items() if lo <= r <= hi}
                total = sum(pts for pts, _ in per_round.values())
                minutes = sum(m for _, m in per_round.values())
                played = sum(1 for _, m in per_round.values() if m > 0)
                ppg = round(total / played, 1) if played else 0.0
                season_total = int(p.total_points)
                stats = consistency_stats(per_round, window_rounds)
            else:
                total, minutes = int(p.total_points), int(p.minutes)
                ppg = round(float(p.points_per_game or 0.0), 1)
                stats = consistency_stats(per_round, rounds)
            if minutes < min_minutes:
                continue
            rows.append(
                RankedPlayerRow(
                    id=p.id,
                    name=p.web_name,
                    full_name=p.full_name,
                    team=bs.team(p.team).short_name,
                    position=p.position.short,
                    price=p.price,
                    status=p.status,
                    ownership=round(float(p.selected_by_percent or 0.0), 1),
                    total_points=total,
                    minutes=minutes,
                    points_per_game=ppg,
                    points_per_million=round(total / p.price, 2) if p.price else 0.0,
                    form=None if p.form is None else round(float(p.form), 1),
                    season_total_points=season_total,
                    in_squad=p.id in squad,
                    **stats,
                )
            )
        ranked = rank_player_rows(rows, inp.metric)
        if not hist_ok:
            notes.append(
                "player_gw_history unavailable: consistency columns are empty, ranking uses "
                "bootstrap totals only"
            )
        elif window is None and rounds and started and rounds[-1] < started[-1]:
            notes.append(
                f"local match history (consistency columns) covers GW1–GW{rounds[-1]}; season "
                f"totals (points, minutes, form) are from the FPL API through GW{started[-1]}"
            )
        if not rows:
            notes.append("no player passes the filters — relax the price / position / minutes")
        return RankPlayersOutput(
            gw=inp.gw,
            metric=inp.metric,
            metric_label=RANKING_METRIC_LABEL[inp.metric],
            window_label=label,
            gw_from=window[0] if window else None,
            gw_to=window[1] if window else None,
            filters={
                k: v
                for k, v in {
                    "position": position,
                    "max_price": inp.max_price,
                    "min_price": inp.min_price,
                    "min_minutes": min_minutes,
                    "exclude_unavailable": inp.exclude_unavailable,
                    "window": label,
                    "gws_in_window": len(window_rounds) if window else None,
                    "gws_started": len(started),
                }.items()
                if v is not None
            },
            history_through_gw=rounds[-1] if rounds else None,
            history_available=hist_ok,
            candidates=len(rows),
            rows=ranked[:limit],
            notes=notes,
        )

    # ---- сигналы ----

    @property
    def retriever(self) -> Any:
        if self._retriever is None:
            from fplcopilot.rag.retrieve import Retriever

            self._retriever = Retriever()
        return self._retriever

    def _latest_signal(self, player_id: int, as_of: datetime) -> dict[str, Any] | None:
        try:
            with session_scope() as s:
                row = s.execute(_LATEST_SIGNAL, {"pid": player_id, "as_of": as_of}).first()
        except SQLAlchemyError as exc:  # БД недоступна — работаем только по статусу FPL
            log.warning("player_signals unavailable: %s", exc)
            return None
        if row is None:
            return None
        keys = (
            "as_of",
            "availability",
            "start_probability",
            "expected_minutes",
            "rotation_risk",
            "return_gw",
            "confidence",
            "summary",
            "evidence",
            "fpl_status",
            "fpl_chance_next",
            "abstained",
            "mode",
            "model",
        )
        return dict(zip(keys, row, strict=True))

    def _evidence_items(self, player: Player, raw: Iterable[Any]) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        for ev in raw or []:
            data = ev.model_dump() if isinstance(ev, BaseModel) else dict(ev)
            published = data.get("published_at")
            if isinstance(published, str):
                published_dt = datetime.fromisoformat(published)
            else:
                published_dt = published
            if published_dt is not None and published_dt.tzinfo is None:
                published_dt = published_dt.replace(tzinfo=UTC)
            items.append(
                EvidenceItem(
                    player_id=player.id,
                    player=player.web_name,
                    source=str(data.get("source", "")),
                    url=str(data.get("url", "")),
                    published_at=published_dt.astimezone(UTC).isoformat() if published_dt else "",
                    date=f"{published_dt.astimezone(UTC):%d.%m}" if published_dt else "",
                    quote=" ".join(str(data.get("quote", "")).split()),
                )
            )
        return items

    def analyze_player_risk(self, inp: PlayerRiskInput) -> PlayerRisk:
        bs = self.bootstrap
        player = bs.player(inp.player_id)
        as_of = inp.as_of if inp.as_of.tzinfo else inp.as_of.replace(tzinfo=UTC)
        base = PlayerRisk(
            player_id=player.id,
            player=player.web_name,
            team_id=player.team,
            fpl_status=player.status,
            fpl_chance=player.chance_of_playing_next_round,
            fpl_news=player.news or "",
            mode=inp.mode,
            k=inp.k,
        )
        cached = None if inp.force else self._latest_signal(player.id, as_of)
        if cached is not None:
            sig_as_of = cached["as_of"].astimezone(UTC)
            age_h = (as_of - sig_as_of).total_seconds() / 3600
            if age_h <= inp.max_age_h or inp.cached_only:
                return base.model_copy(
                    update={
                        "availability": cached["availability"],
                        "start_probability": float(cached["start_probability"]),
                        "expected_minutes": int(cached["expected_minutes"]),
                        "rotation_risk": cached["rotation_risk"] or "unknown",
                        "return_gw": cached["return_gw"],
                        "confidence": float(cached["confidence"]),
                        "summary": cached["summary"] or "",
                        "evidence": self._evidence_items(player, cached["evidence"] or []),
                        "signal_as_of": sig_as_of.isoformat(),
                        "age_h": round(age_h, 1),
                        "origin": "cached",
                        "abstained": bool(cached.get("abstained")),
                        "mode": cached.get("mode") or inp.mode,
                        "model": cached.get("model"),
                        "note": (
                            None
                            if age_h <= inp.max_age_h
                            else f"stale signal ({age_h:.1f} h > {inp.max_age_h:g} h), not refreshed"
                        ),
                    }
                )
        if inp.cached_only:  # UI-таблицы: никаких LLM-вызовов «по дороге»
            return base.model_copy(
                update={
                    "origin": "unavailable",
                    "note": "no saved news signal (cached_only)",
                    "summary": f"FPL status {player.status} is the only signal.",
                }
            )
        from fplcopilot.rag.extract import extract_signal

        timings: dict[str, Any] = {}
        started = time.perf_counter()
        try:
            sig = extract_signal(
                player.id,
                as_of,
                mode=inp.mode,  # type: ignore[arg-type]
                k=inp.k,
                retriever=self.retriever,
                bs=bs,
                save=True,
                timings=timings,
            )
        except Exception as exc:  # сеть/OpenAI/БД: агент отвечает по статусу FPL с пометкой
            log.warning("extract_signal failed for %s: %s", player.web_name, exc, exc_info=True)
            note = f"signal extraction failed: {type(exc).__name__}: {exc}"[:200]
            if cached is not None:  # устаревший сигнал лучше, чем ничего — но с пометкой
                sig_as_of = cached["as_of"].astimezone(UTC)
                return base.model_copy(
                    update={
                        "availability": cached["availability"],
                        "start_probability": float(cached["start_probability"]),
                        "confidence": float(cached["confidence"]),
                        "summary": cached["summary"] or "",
                        "evidence": self._evidence_items(player, cached["evidence"] or []),
                        "signal_as_of": sig_as_of.isoformat(),
                        "age_h": round((as_of - sig_as_of).total_seconds() / 3600, 1),
                        "origin": "cached",
                        "note": note + " (stale cached signal used)",
                        "latency_ms": round((time.perf_counter() - started) * 1000),
                    }
                )
            return base.model_copy(
                update={
                    "origin": "unavailable",
                    "note": note,
                    "summary": f"FPL status {player.status} is the only signal.",
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                }
            )
        return base.model_copy(
            update={
                "availability": sig.availability,
                "start_probability": sig.start_probability,
                "expected_minutes": sig.expected_minutes,
                "rotation_risk": sig.rotation_risk,
                "return_gw": sig.return_gw,
                "confidence": sig.confidence,
                "summary": sig.summary,
                "evidence": self._evidence_items(player, sig.evidence),
                "signal_as_of": sig.as_of.isoformat(),
                "age_h": 0.0,
                "origin": "extracted",
                "abstained": bool(getattr(sig, "abstained", False)),
                "mode": sig.mode,
                "model": sig.model,
                "llm_calls": int(timings.get("llm_calls", 0)),
                "prompt_tokens": int(timings.get("prompt_tokens", 0)),
                "completion_tokens": int(timings.get("completion_tokens", 0)),
                "latency_ms": round(
                    float(timings.get("total_ms", (time.perf_counter() - started) * 1000))
                ),
            }
        )

    # ---- новости клуба (rag/team_news.py) ----

    def _team_items(self, raw: Iterable[Any]) -> list[TeamNewsItemOut]:
        items: list[TeamNewsItemOut] = []
        for it in raw or []:
            data = it.model_dump() if isinstance(it, BaseModel) else dict(it)
            published = data.get("published_at")
            dt = datetime.fromisoformat(published) if isinstance(published, str) else published
            if dt is not None and dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            items.append(
                TeamNewsItemOut(
                    kind=str(data.get("kind", "")),
                    claim=str(data.get("claim", "")),
                    players=[str(p) for p in data.get("players") or []],
                    player_ids=[int(p) for p in data.get("player_ids") or []],
                    source=str(data.get("source", "")),
                    url=str(data.get("url", "")),
                    published_at=dt.astimezone(UTC).isoformat() if dt else "",
                    date=f"{dt.astimezone(UTC):%d.%m}" if dt else "",
                    quote=" ".join(str(data.get("quote", "")).split()),
                )
            )
        return items

    def _calendar(self) -> Any:
        if getattr(self, "_gw_calendar", None) is None:
            from fplcopilot.rag.extract import load_calendar

            try:
                fixtures = self.client.fixtures()
            except Exception:  # без фикстур календарь работает по окнам туров
                log.warning("fixtures unavailable for the club calendar", exc_info=True)
                fixtures = []
            self._gw_calendar = load_calendar(self.bootstrap, fixtures)
        return self._gw_calendar

    def team_news(self, inp: TeamNewsInput) -> TeamNewsOut:
        """Новости клуба: недоступные игроки — из статусов FPL на as_of (всегда, без LLM) + мягкий
        дайджест (сохранённый моложе max_age_h, cached_only — любого возраста, иначе извлечение)."""
        from fplcopilot.rag import team_news as tn

        out = self._team_digest(inp)
        as_of = inp.as_of if inp.as_of.tzinfo else inp.as_of.replace(tzinfo=UTC)
        try:
            rows = tn.club_absences(
                self.bootstrap, self.bootstrap.team(inp.team_id), as_of, calendar=self._calendar()
            )
        except Exception as exc:  # снимки статусов недоступны — дайджест без списка
            log.warning("club_absences failed for %s: %s", inp.team_id, exc, exc_info=True)
            return out
        return out.model_copy(update={"absences": [ClubAbsenceOut(**r) for r in rows]})

    def _team_digest(self, inp: TeamNewsInput) -> TeamNewsOut:
        """Дайджест клуба: сохранённый моложе max_age_h (cached_only — любого возраста), иначе
        extract_team_news. Ошибка извлечения -> устаревший сохранённый с пометкой / unavailable."""
        from fplcopilot.rag import team_news as tn

        team = self.bootstrap.team(inp.team_id)
        as_of = inp.as_of if inp.as_of.tzinfo else inp.as_of.replace(tzinfo=UTC)
        base = TeamNewsOut(
            team_id=team.id, team=team.short_name, team_name=team.name, mode=inp.mode
        )
        cached = None
        if not inp.force:
            try:
                cached = tn.latest_digest(team.id, as_of)
            except SQLAlchemyError as exc:
                log.warning("team_news_digests unavailable: %s", exc)

        def from_cached(row: dict[str, Any], note: str | None) -> TeamNewsOut:
            row_as_of = row["as_of"].astimezone(UTC)
            return base.model_copy(
                update={
                    "summary": row["summary"] or "",
                    "items": self._team_items(row["items"] or []),
                    "digest_as_of": row_as_of.isoformat(),
                    "age_h": round((as_of - row_as_of).total_seconds() / 3600, 1),
                    "origin": "cached",
                    "abstained": bool(row.get("abstained")),
                    "mode": row.get("mode") or inp.mode,
                    "model": row.get("model"),
                    "note": note,
                }
            )

        if cached is not None:
            age_h = (as_of - cached["as_of"].astimezone(UTC)).total_seconds() / 3600
            if age_h <= inp.max_age_h:
                return from_cached(cached, None)
            if inp.cached_only:
                return from_cached(
                    cached, f"stale digest ({age_h:.1f} h > {inp.max_age_h:g} h), not refreshed"
                )
        if inp.cached_only:
            return base.model_copy(update={"note": "no saved club news digest (cached_only)"})
        timings: dict[str, Any] = {}
        started = time.perf_counter()
        try:
            digest = tn.extract_team_news(
                team.id,
                as_of,
                mode=inp.mode,  # type: ignore[arg-type]
                k=inp.k,
                retriever=self.retriever,
                bs=self.bootstrap,
                save=True,
                timings=timings,
            )
        except Exception as exc:  # сеть / OpenAI / БД: без клубного контекста, с пометкой
            log.warning("extract_team_news failed for %s: %s", team.short_name, exc, exc_info=True)
            note = f"club news extraction failed: {type(exc).__name__}: {exc}"[:200]
            if cached is not None:
                return from_cached(cached, note + " (stale saved digest used)")
            return base.model_copy(
                update={"note": note, "latency_ms": round((time.perf_counter() - started) * 1000)}
            )
        return base.model_copy(
            update={
                "summary": digest.summary,
                "items": self._team_items(digest.items),
                "digest_as_of": digest.as_of.isoformat(),
                "age_h": 0.0,
                "origin": "extracted",
                "abstained": digest.abstained,
                "mode": digest.mode,
                "model": digest.model,
                "llm_calls": int(timings.get("llm_calls", 0)),
                "prompt_tokens": int(timings.get("prompt_tokens", 0)),
                "completion_tokens": int(timings.get("completion_tokens", 0)),
                "latency_ms": round(
                    float(timings.get("total_ms", (time.perf_counter() - started) * 1000))
                ),
            }
        )

    # ---- стратегическая KB (rag/kb) ----

    @property
    def kb_retriever(self) -> Any:
        """KBRetriever c общим reranker'ом новостного Retriever'а (одна ONNX-сессия на процесс)."""
        if self._kb_retriever is None:
            from fplcopilot.rag.kb.retrieve import KBRetriever

            shared = getattr(self._retriever, "reranker", None)
            self._kb_retriever = (
                KBRetriever(reranker=shared) if shared is not None else KBRetriever()
            )
        return self._kb_retriever

    def search_strategy_kb(self, inp: KBSearchInput) -> KBSearchOutput:
        started = time.perf_counter()
        tags = [t for t in inp.tags if t] or None
        chunks = self.kb_retriever.search(
            inp.query,
            tags=tags,
            k=max(1, inp.k),
            mode=inp.mode,  # type: ignore[arg-type]
        )
        return KBSearchOutput(
            query=inp.query,
            tags=list(tags or []),
            mode=inp.mode,
            chunks=[
                KBChunkOut(
                    chunk_id=c.chunk_id,
                    doc_id=c.article_id,
                    title=c.title,
                    url=c.url,
                    source=c.source,
                    tags=list(c.tags),
                    text=c.text,
                    score=r2(c.final_score),
                )
                for c in chunks
            ],
            latency_ms=round((time.perf_counter() - started) * 1000),
        )

    def answer_strategy_question(self, inp: StrategyAnswerInput) -> StrategyAnswerOutput:
        from fplcopilot.rag.kb.answer import answer_strategy_question

        started = time.perf_counter()
        tags = [t for t in inp.tags if t] or None
        res = answer_strategy_question(
            inp.query, k=max(1, inp.k), tags=tags, retriever=self.kb_retriever
        )
        usage = res.get("usage") or {}
        return StrategyAnswerOutput(
            query=inp.query,
            answer=res["answer"],
            covered=bool(res["covered"]),
            citations=[KBCitation.model_validate(c) for c in res.get("citations") or []],
            retrieved=len(res.get("retrieved") or []),
            model=res.get("model"),
            prompt_version=res.get("prompt_version"),
            llm_calls=int(res.get("llm_calls") or 0),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            cost_usd=res.get("cost_usd"),
            latency_ms=round((time.perf_counter() - started) * 1000),
            validation=dict(res.get("validation") or {}),
        )

    def kb_stats(self) -> dict[str, Any]:
        """Размер стратегической KB для ресурса MCP `fpl://kb/stats` (без LLM)."""
        from fplcopilot.rag.kb.ingest import corpus_stats

        stats = corpus_stats()
        with session_scope() as s:
            by_tag = s.execute(
                text(
                    "SELECT t, count(DISTINCT doc_id), count(*) FROM kb_chunks, unnest(tags) AS t "
                    "GROUP BY t ORDER BY 3 DESC"
                )
            ).all()
        stats["tags"] = {str(t): {"docs": int(d), "chunks": int(c)} for t, d, c in by_tag}
        return stats

    # ---- оптимизатор ----

    def _lineup_player(self, pid: int, gw: int, inputs: ManagerInputs) -> LineupPlayer:
        c = inputs.cands[pid]
        pred = inputs.store.get(gw).get(pid)
        return LineupPlayer(
            id=pid,
            name=c.name,
            team=c.team,
            position=["", "GKP", "DEF", "MID", "FWD"][c.position],
            price=c.price,
            xpts=r2(c.xpts(gw)) or 0.0,
            sd=round(math.sqrt(max(0.0, c.variance(gw))), 1),
            p_start=r2(pred.p_start) if pred else 0.0,
            ownership=round(c.ownership, 1),
            fixture=fixture_label(pred),
            status=c.status,
        )

    def optimize_team(
        self, inp: OptimizeTeamInput, *, squad_override: SquadLike | None = None
    ) -> LineupOut:
        # Будущий тур (GW7 при ближайшем GW6): публичных picks на него ещё нет — берём входы
        # ближайшего тура с горизонтом до целевого, состав текущий (без трансферов).
        base_gw = min(inp.gw, self.next_gw())
        inputs = self.inputs(
            inp.manager_id,
            base_gw,
            max(DIAGNOSIS_HORIZON, inp.gw - base_gw + 1),
            inp.strategy,
            squad_override=squad_override,
        )
        squad_cands = [inputs.cands[p] for p in inputs.squad]
        lu: LineupResult = best_xi(squad_cands, inp.gw, inp.strategy)
        starters = [self._lineup_player(p, inp.gw, inputs) for p in lu.starters]
        bench = [self._lineup_player(p, inp.gw, inputs) for p in lu.bench_order]
        options = [
            CaptainOption(
                id=s.id,
                name=s.name,
                team=s.team,
                xpts=s.xpts,
                captain_points=round(2 * s.xpts, 2),
                sd=s.sd,
                ownership=s.ownership,
                tag=captain_tag(s.ownership),
                fixture=s.fixture,
                p_start=s.p_start,
            )
            for s in sorted(starters, key=lambda s: -s.xpts)[:3]
        ]
        squad = self._squad(inp.manager_id, inputs.squad_gw, squad_override)
        cap = squad.captain
        current_xi = sum(inputs.cands[p.player.id].xpts(inp.gw) for p in squad.starting_xi)
        if cap is not None:
            current_xi += inputs.cands[cap.player.id].xpts(inp.gw)
        return LineupOut(
            gw=inp.gw,
            formation=lu.formation,
            starters=starters,
            bench=bench,
            captain=inputs.cands[lu.captain].name,
            vice=inputs.cands[lu.vice].name,
            expected_points=r2(lu.expected_points) or 0.0,
            captain_options=options,
            current_captain=cap.player.web_name if cap else None,
            current_xi_points=r2(current_xi),
        )

    def recommend_transfers(
        self, inp: RecommendTransfersInput, *, squad_override: SquadLike | None = None
    ) -> RoutesOut:
        inputs = self.inputs(
            inp.manager_id,
            inp.gw,
            inp.horizon,
            inp.strategy,
            exclude=inp.exclude,
            squad_override=squad_override,
        )
        notes: list[str] = []
        squad = set(inputs.squad)
        sell = [p for p in inp.sell if p in squad]
        for p in inp.sell:
            if p not in squad:
                notes.append(f"{inputs.cands[p].name} is not in the squad — cannot be sold")
        buy = [p for p in inp.buy if p not in squad]
        for p in inp.buy:
            if p in squad:
                notes.append(f"{inputs.cands[p].name} is already in the squad")
        constrained = bool(sell or buy)
        alternative: RouteOut | None = None
        if constrained:
            routes, meta = constrained_routes(
                inputs,
                inp.gw,
                inp.horizon,
                inp.strategy,
                force_in=buy,
                force_out=sell,
                keep=inp.keep,
                exclude=inp.exclude,
                allow_hit=inp.allow_hit,
            )
            baseline_pts = meta["baseline_xi_points"] or 0.0
            free = single_transfer(
                inputs.squad,
                inputs.bank,
                inputs.free_transfers,
                inputs.pool,
                inp.gw,
                inp.horizon,
                inp.strategy,
                exclude=inp.exclude,
                keep=inp.keep,
                allow_hit=inp.allow_hit,
                top=1,
                issues=inputs.issues,
            )
            pick = recommend_route(free)
            if pick is not None and (not routes or set(pick.in_) != set(routes[0].in_)):
                alternative = route_out(pick, 0)
        else:
            routes = single_transfer(
                inputs.squad,
                inputs.bank,
                inputs.free_transfers,
                inputs.pool,
                inp.gw,
                inp.horizon,
                inp.strategy,
                exclude=inp.exclude,
                keep=inp.keep,
                allow_hit=inp.allow_hit,
                issues=inputs.issues,
            )
            squad_cands = [inputs.cands[p] for p in inputs.squad]
            baseline_pts = r2(best_xi(squad_cands, inp.gw, inp.strategy).expected_points) or 0.0
        pick = recommend_route(routes)
        rank = routes.index(pick) + 1 if pick is not None else None
        return RoutesOut(
            gw=inp.gw,
            horizon=inp.horizon,
            strategy=inp.strategy,
            free_transfers=inputs.free_transfers,
            bank=round(inputs.bank, 1),
            routes=[route_out(r, i + 1) for i, r in enumerate(routes)],
            recommended_rank=rank,
            recommendation=f"route {rank}" if rank else "hold",
            baseline_xi_points=baseline_pts,
            constrained=constrained,
            alternative=alternative,
            notes=notes,
        )

    def _plan_out(self, plan: TransferPlan, pool: dict[int, Candidate]) -> PlanOut:
        moves = {
            str(gw): [
                PlanMoveOut(
                    gw=m.gw,
                    out=m.out_name,
                    in_=m.in_name,
                    price_out=m.price_out,
                    price_in=m.price_in,
                    delta_xpts_horizon=r2(m.delta_xpts_horizon) or 0.0,
                    out_problem=m.out_problem,
                    paid=m.paid,
                    out_id=m.out,
                    in_id=m.in_,
                )
                for m in ms
            ]
            for gw, ms in sorted(plan.moves_by_gw.items())
        }
        wc = None
        if plan.wildcard_alternative is not None:
            w = plan.wildcard_alternative
            wc = {
                "gw": w.gw,
                "expected_total": r2(w.expected_total),
                "delta_vs_plan": r2(w.expected_total - plan.expected_total),
                "squad": list(w.squad_names),
            }
        chips = [
            PlanChipOut(
                gw=g,
                chip=lu.chip,
                name=CHIP_TITLES.get(lu.chip, lu.chip),
                points=r2(lu.chip_points) or 0.0,
                players=[
                    pool[p].name
                    for p in (lu.bench_order if lu.chip == BENCH_BOOST else [lu.captain])
                    if p in pool
                ],
            )
            for g, lu in sorted(plan.lineups_by_gw.items())
            if lu.chip in (BENCH_BOOST, TRIPLE_CAPTAIN)
        ]
        return PlanOut(
            from_gw=plan.from_gw,
            horizon=plan.horizon,
            strategy=plan.strategy,
            moves_by_gw=moves,
            hits_by_gw={str(g): v for g, v in sorted(plan.hits_by_gw.items())},
            ft_by_gw={str(g): v for g, v in sorted(plan.ft_by_gw.items())},
            bank_by_gw={str(g): r2(v) or 0.0 for g, v in sorted(plan.bank_by_gw.items())},
            xi_points_by_gw={
                str(g): r2(lu.expected_points) or 0.0
                for g, lu in sorted(plan.lineups_by_gw.items())
            },
            captain_by_gw={
                str(g): pool[lu.captain].name if lu.captain in pool else str(lu.captain)
                for g, lu in sorted(plan.lineups_by_gw.items())
            },
            expected_total=r2(plan.expected_total) or 0.0,
            baseline_total=r2(plan.baseline_total) or 0.0,
            baseline_xi_points_by_gw={
                str(g): r2(v) or 0.0 for g, v in sorted(plan.baseline_points_by_gw.items())
            },
            recommendation=plan.recommendation,
            target_squad=list(plan.target_squad_names),
            wildcard=wc,
            issues=[i.model_dump() for i in plan.issues],
            solver=plan.solver,
            runtime_s=plan.runtime_s,
            time_limit_hit=plan.time_limit_hit,
            allow_hits=plan.allow_hits,
            chips=chips,
            notes=list(plan.notes),
        )

    def build_gameweek_plan(
        self, inp: BuildPlanInput, *, squad_override: SquadLike | None = None
    ) -> PlanOut:
        chips = [(c.gw, c.chip) for c in inp.chips]
        clash = next((c for g, c in chips if g == inp.gw), None)
        if inp.use_wildcard and clash is not None:
            raise ChipPlanError(
                f"GW{inp.gw}: Wildcard и {CHIP_TITLES.get(clash, clash)} в одном туре нельзя — "
                "одна фишка за тур"
            )
        inputs = self.inputs(
            inp.manager_id,
            inp.gw,
            inp.horizon,
            inp.strategy,
            exclude=inp.exclude,
            wildcard=inp.use_wildcard,
            squad_override=squad_override,
        )
        previous: TransferPlan | None = None
        try:
            previous = latest_plan(inp.manager_id, inp.gw, inp.strategy)
        except SQLAlchemyError as exc:
            log.warning("latest_plan unavailable: %s", exc)
        plan = plan_transfers(
            inputs.squad,
            inputs.bank,
            inputs.free_transfers,
            inputs.chips_available,
            inp.gw,
            inp.horizon,
            inp.strategy,
            pool=inputs.pool,
            issues=inputs.issues,
            keep=inp.keep,
            exclude=inp.exclude,
            manager_id=inp.manager_id,
            allow_hits=inp.allow_hits,
            chips=chips,
            chips_available_by_gw=inputs.chips_by_gw or None,
        )
        out = self._plan_out(plan, inputs.pool)
        if previous is not None:
            out.diff_vs_previous = [c.model_dump() for c in diff_plans(previous, plan)]
            out.previous_plan_at = previous.created_at.astimezone(UTC).isoformat()
        if inp.save:
            try:
                digest = inputs_hash(
                    inputs.squad,
                    inputs.bank,
                    inputs.free_transfers,
                    inp.strategy,
                    inp.horizon,
                    inp.gw,
                    inputs.pool,
                    chips=plan.chips_by_gw,
                )
                out.saved_id = save_plan(plan, inputs_hash=digest)
            except SQLAlchemyError as exc:
                log.warning("save_plan failed: %s", exc)
        return out

    def simulate_scenario(
        self, inp: SimulateScenarioInput, *, squad_override: SquadLike | None = None
    ) -> ScenarioOut:
        notes: list[str] = []
        if inp.use_wildcard:
            inputs = self.inputs(
                inp.manager_id,
                inp.gw,
                max(inp.horizon, 5),
                inp.strategy,
                wildcard=True,
                squad_override=squad_override,
            )
            plan = plan_transfers(
                inputs.squad,
                inputs.bank,
                inputs.free_transfers,
                ["wildcard"],
                inp.gw,
                max(inp.horizon, 5),
                inp.strategy,
                pool=inputs.pool,
                issues=inputs.issues,
                keep=inp.keep,
                exclude=inp.exclude,
                manager_id=inp.manager_id,
            )
            wc = plan.wildcard_alternative
            return ScenarioOut(
                gw=inp.gw,
                horizon=plan.horizon,
                strategy=inp.strategy,
                feasible=wc is not None,
                free_transfers=inputs.free_transfers,
                bank=round(inputs.bank, 1),
                transfers_cap=0,
                wildcard=(
                    {
                        "expected_total": r2(wc.expected_total),
                        "plan_without_wildcard_total": r2(plan.expected_total),
                        "delta_vs_plan": r2(wc.expected_total - plan.expected_total),
                        "baseline_total": r2(plan.baseline_total),
                        "squad": list(wc.squad_names),
                        "moves": [
                            f"{m.out_name} -> {m.in_name} ({m.delta_xpts_horizon:+.1f})"
                            for m in wc.moves
                        ],
                        "recommendation": plan.recommendation,
                        "problem_players": len(problem_players(inputs.issues)),
                    }
                    if wc
                    else None
                ),
                verdict=plan.recommendation,
                notes=notes,
            )
        inputs = self.inputs(
            inp.manager_id,
            inp.gw,
            inp.horizon,
            inp.strategy,
            exclude=inp.exclude,
            squad_override=squad_override,
        )
        squad = set(inputs.squad)
        buy = [p for p in inp.buy if p not in squad]
        for p in inp.buy:
            if p in squad:
                notes.append(f"{inputs.cands[p].name} is already in the squad")
        sell = [p for p in inp.sell if p in squad]
        for p in inp.sell:
            if p not in squad:
                notes.append(f"{inputs.cands[p].name} is not in the squad")
        try:
            routes, meta = constrained_routes(
                inputs,
                inp.gw,
                inp.horizon,
                inp.strategy,
                force_in=buy,
                force_out=sell,
                keep=inp.keep,
                exclude=inp.exclude,
                allow_hit=inp.allow_hit,
            )
        except ScenarioInfeasible as exc:
            return ScenarioOut(
                gw=inp.gw,
                horizon=inp.horizon,
                strategy=inp.strategy,
                feasible=False,
                reason=str(exc),
                free_transfers=inputs.free_transfers,
                bank=round(inputs.bank, 1),
                transfers_cap=0,
                notes=notes,
            )
        if not routes:
            return ScenarioOut(
                gw=inp.gw,
                horizon=inp.horizon,
                strategy=inp.strategy,
                feasible=False,
                reason="no feasible squad satisfies the scenario (budget, club limit or quotas)",
                free_transfers=inputs.free_transfers,
                bank=round(inputs.bank, 1),
                transfers_cap=meta["cap"],
                hold_xi_points=meta["baseline_xi_points"],
                notes=notes,
            )
        primary = route_out(routes[0], 1)
        free_alt = None
        if meta["free_route"] is not None and set(meta["free_route"].in_) != set(routes[0].in_):
            free_alt = route_out(meta["free_route"], 0)
        hit_alt = None
        if inp.allow_hit and primary.hit_cost == 0 and meta["cap"] > meta["free_cap"]:
            # Пользователь спросил именно про хит, а оптимум обошёлся без него: покажем
            # «версию с хитом» (>= free_cap + 1 трансферов) для честного сравнения.
            try:
                hit_routes, _ = constrained_routes(
                    inputs,
                    inp.gw,
                    inp.horizon,
                    inp.strategy,
                    force_in=buy,
                    force_out=sell,
                    keep=inp.keep,
                    exclude=inp.exclude,
                    allow_hit=True,
                    top=1,
                    min_transfers=meta["free_cap"] + 1,
                )
            except (ScenarioInfeasible, InfeasibleError):
                hit_routes = []
            if hit_routes:
                hit_alt = route_out(hit_routes[0], 0)
                notes.append(
                    "the requested hit is not needed: the free transfers cover the scenario; "
                    "hit_alternative shows the best version with a paid transfer"
                )
        return ScenarioOut(
            gw=inp.gw,
            horizon=inp.horizon,
            strategy=inp.strategy,
            feasible=True,
            free_transfers=inputs.free_transfers,
            bank=round(inputs.bank, 1),
            transfers_cap=meta["cap"],
            primary=primary,
            free_alternative=free_alt,
            hit_alternative=hit_alt,
            alternatives=[route_out(r, i + 1) for i, r in enumerate(routes[1:], start=1)],
            hold_xi_points=meta["baseline_xi_points"],
            verdict=primary.verdict,
            notes=notes,
        )

    # ---- v4 чата ----

    def team_fixtures(self, inp: TeamFixturesInput) -> TeamFixturesOut:
        """Матчи клубов на горизонт с FSI из прогнозов xPts (тот же индекс, что в ответах про
        игроков): один игрок клуба с прогнозом на тур даёт соперников клуба в этом туре."""
        bs = self.bootstrap
        gws = list(range(inp.gw, min(MAX_GW, inp.gw + max(1, inp.horizon) - 1) + 1))
        wanted = set(inp.team_ids) or {t.id for t in bs.teams}
        per_team: dict[int, dict[str, list[tuple[str, int]]]] = {t: {} for t in wanted}
        for g in gws:
            seen: set[int] = set()
            for pid, pred in self.store.get(g).items():
                team_id = bs.player(pid).team
                if team_id not in wanted or team_id in seen or not pred.fixtures:
                    continue
                seen.add(team_id)
                per_team[team_id][f"GW{g}"] = [
                    (f"{'v' if f.is_home else '@'}{f.opponent}", int(f.fixture_strength_index))
                    for f in pred.fixtures
                ]
        rows = []
        for team_id, by_gw in per_team.items():
            team = bs.team(team_id)
            fsis = [fsi for fx in by_gw.values() for _, fsi in fx]
            rows.append(
                TeamFixtureRow(
                    team_id=team_id,
                    team=team.short_name,
                    team_name=team.name,
                    fixtures={
                        f"GW{g}": ", ".join(f"{o} (FSI {s})" for o, s in by_gw[f"GW{g}"])
                        if f"GW{g}" in by_gw
                        else "blank"
                        for g in gws
                    },
                    matches=len(fsis),
                    mean_fsi=round(sum(fsis) / len(fsis), 2) if fsis else None,
                )
            )
        rows.sort(key=lambda r: (r.mean_fsi is None, r.mean_fsi or 0.0, -r.matches, r.team))
        for i, r in enumerate(rows, start=1):
            r.rank = i
        return TeamFixturesOut(gw_from=gws[0], gw_to=gws[-1], teams=rows)

    def rank_forecast(self, inp: ForecastRankInput) -> ForecastRankOut:
        """Прогнозный рейтинг лиги: сумма xPts модели по турам gw..gw+h-1 (или на £1m)."""
        bs = self.bootstrap
        gws = list(range(inp.gw, min(MAX_GW, inp.gw + max(1, inp.horizon) - 1) + 1))
        preds = {g: self.store.get(g) for g in gws}
        squad = set(inp.squad_ids)
        rows: list[ForecastRow] = []
        for p in bs.elements:
            if inp.position and p.position.short != inp.position:
                continue
            if inp.max_price is not None and p.price > inp.max_price + 1e-9:
                continue
            if inp.min_price is not None and p.price < inp.min_price - 1e-9:
                continue
            own = float(p.selected_by_percent or 0.0)
            if inp.max_ownership is not None and own > inp.max_ownership:
                continue
            if inp.exclude_unavailable and p.status in UNAVAILABLE_STATUSES:
                continue
            by_gw = {g: preds[g].get(p.id) for g in gws}
            first = by_gw[gws[0]]
            if first is None or float(first.p_start) < inp.min_p_start:
                continue
            total = sum(float(x.xpts) for x in by_gw.values() if x is not None)
            rows.append(
                ForecastRow(
                    id=p.id,
                    name=p.web_name,
                    full_name=p.full_name,
                    team=bs.team(p.team).short_name,
                    position=p.position.short,
                    price=p.price,
                    ownership=round(own, 1),
                    status=p.status,
                    xpts_total=round(total, 2),
                    xpts_per_million=round(total / p.price, 2) if p.price else 0.0,
                    xpts_by_gw={
                        f"GW{g}": r2(x.xpts) or 0.0 for g, x in by_gw.items() if x is not None
                    },
                    p_start_next=r2(first.p_start),
                    fixtures={f"GW{g}": fixture_label(x) for g, x in by_gw.items()},
                    in_squad=p.id in squad,
                )
            )
        key = "xpts_per_million" if inp.metric == "xpts_per_million" else "xpts_total"
        rows.sort(key=lambda r: (-getattr(r, key), -r.xpts_total, r.price))
        top = rows[: max(1, min(RANK_LIMIT_MAX, inp.limit))]
        for i, r in enumerate(top, start=1):
            r.rank = i
        window = f"GW{gws[0]}" if len(gws) == 1 else f"GW{gws[0]}–GW{gws[-1]}"
        flt = {
            k: v
            for k, v in {
                "position": inp.position,
                "max_price": inp.max_price,
                "min_price": inp.min_price,
                "max_ownership_pct": inp.max_ownership,
                "min_p_start_next_gw": inp.min_p_start,
            }.items()
            if v is not None
        }
        return ForecastRankOut(
            gw_from=gws[0],
            gw_to=gws[-1],
            metric=inp.metric,
            metric_label=(
                f"forecast xPts per £1m over {window}"
                if inp.metric == "xpts_per_million"
                else f"forecast xPts over {window}"
            ),
            filters=flt,
            filters_definition=(
                "players with a model p(start) of at least "
                f"{inp.min_p_start} in GW{gws[0]}; injured / suspended / unavailable excluded"
                + (f"; ownership at most {inp.max_ownership}%" if inp.max_ownership else "")
            ),
            candidates=len(rows),
            rows=top,
            notes=[
                (
                    "Forecast of the xPts model (expected points), not past points; it is "
                    "re-computed when news and fixtures change."
                )
            ],
        )

    def chips_status(self, inp: ChipsStatusInput) -> ChipsStatusOut:
        """Фишки менеджера: доступна ли на дедлайн gw, в каких турах сыграна, когда откроется
        снова (окна фишек — bootstrap.chips, сыгранные — публичная история менеджера)."""
        bs = self.bootstrap
        notes: list[str] = []
        try:
            played = [(c.name, int(c.event)) for c in self.client.history(inp.manager_id).chips]
        except Exception as exc:  # история недоступна (скриншот без id) — окна без сыгранных
            log.debug("history unavailable for %s", inp.manager_id, exc_info=True)
            played = []
            notes.append(f"manager history unavailable ({type(exc).__name__}): played chips unknown")
        out: list[ChipState] = []
        for chip in CHIP_ORDER:
            windows = sorted((c.start_event, c.stop_event) for c in bs.chips if c.name == chip)
            now = next(((a, b) for a, b in windows if a <= inp.gw <= b), None)
            gws_played = sorted(g for name, g in played if name == chip)
            used_now = now is not None and any(now[0] <= g <= now[1] for g in gws_played)
            available = now is not None and not used_now
            nxt = None
            if not available:
                nxt = next((a for a, _ in windows if a > inp.gw), None)
            out.append(
                ChipState(
                    chip=chip,
                    name=CHIP_TITLES.get(chip, chip),
                    available_now=available,
                    played_gws=gws_played,
                    window_now=f"GW{now[0]}–GW{now[1]}" if now else None,
                    next_available_gw=nxt,
                )
            )
        return ChipsStatusOut(
            gw=inp.gw,
            chips=out,
            available_now=[c.name for c in out if c.available_now],
            notes=notes
            + ["Each chip can be played twice a season: once in GW1–GW19 and once in GW20–GW38."],
        )

    def transfer_trends(self, inp: TransferTrendsInput) -> TransferTrendsOut:
        """Самые покупаемые / продаваемые игроки ближайшего тура (bootstrap FPL API)."""
        bs = self.bootstrap
        rows = []
        for p in bs.elements:
            if inp.position and p.position.short != inp.position:
                continue
            rows.append(
                TrendRow(
                    name=p.web_name,
                    team=bs.team(p.team).short_name,
                    position=p.position.short,
                    price=p.price,
                    ownership=round(float(p.selected_by_percent or 0.0), 1),
                    transfers_in_event=int(p.transfers_in_event),
                    transfers_out_event=int(p.transfers_out_event),
                    net_transfers_event=int(p.transfers_in_event) - int(p.transfers_out_event),
                    price_change_event=round(int(p.cost_change_event) / 10, 1),
                )
            )
        n = max(1, min(RANK_LIMIT_MAX, inp.limit))
        return TransferTrendsOut(
            gw=self.next_gw(),
            most_transferred_in=sorted(rows, key=lambda r: -r.net_transfers_event)[:n],
            most_transferred_out=sorted(rows, key=lambda r: r.net_transfers_event)[:n],
            notes=[
                (
                    "FPL price changes are driven by net transfers, but the thresholds are not "
                    "public: this is transfer activity, not a price-change prediction."
                )
            ],
        )


    def review_gameweek(self, inp: GWReviewInput) -> GWReviewOut:
        """Разбор завершённого тура по фактам: picks менеджера этого тура, фактические очки
        (player_gw_history), последний прогноз модели до дедлайна (xpts_predictions), очки
        менеджера и средний результат тура (FPL API). Без LLM; чего нет в данных — пусто + note."""
        from fplcopilot.core.history import actual_points
        from fplcopilot.core.xpts import MODEL_VERSION

        bs = self.bootstrap
        event = next((e for e in bs.events if e.id == inp.gw), None)
        if event is None:
            raise ValueError(f"GW{inp.gw} not found in bootstrap")
        out = GWReviewOut(gw=inp.gw, finished=bool(event.finished))
        out.average_points = event.average_entry_score
        if not event.finished:
            out.notes.append(f"GW{inp.gw} is not finished yet — nothing to review")
            return out
        squad = self.client.squad(inp.manager_id, inp.gw)
        out.active_chip = squad.active_chip
        try:
            actual = actual_points(inp.gw)
        except SQLAlchemyError as exc:
            log.warning("player_gw_history unavailable: %s", exc)
            actual = {}
        out.history_available = bool(actual)
        if not actual:
            out.notes.append(f"actual points for GW{inp.gw} are not in the local history yet")
        forecast: dict[int, float] = {}
        try:
            with session_scope() as s:
                rows = s.execute(
                    _FORECAST_BEFORE_DEADLINE,
                    {"gw": inp.gw, "mv": MODEL_VERSION, "deadline": event.deadline_time},
                ).all()
            forecast = {int(r[0]): float(r[1]) for r in rows}
        except SQLAlchemyError as exc:
            log.warning("xpts_predictions unavailable: %s", exc)
        out.forecast_available = bool(forecast)
        if not forecast:
            out.notes.append(f"no model forecast saved before the GW{inp.gw} deadline")
        try:
            row = next(
                (h for h in self.client.history(inp.manager_id).current if h.event == inp.gw), None
            )
        except Exception:  # история менеджера недоступна — очки тура из суммы по игрокам не считаем
            log.debug("manager history unavailable", exc_info=True)
            row = None
        if row is not None:
            out.manager_points = row.points
            out.transfers_cost = row.event_transfers_cost
            out.points_on_bench = row.points_on_bench
        for sp in sorted(squad.players, key=lambda x: x.pick.position):
            pid = sp.player.id
            pts, mins = actual.get(pid, (None, None))
            fc = forecast.get(pid)
            role = (
                "C" if sp.pick.is_captain
                else "VC" if sp.pick.is_vice_captain and sp.pick.is_starting
                else "XI" if sp.pick.is_starting
                else "bench"
            )
            out.rows.append(
                GWReviewRow(
                    name=sp.player.web_name,
                    team=sp.team.short_name,
                    position=sp.player.position.short,
                    role=role,
                    multiplier=sp.pick.multiplier,
                    points=pts,
                    counted_points=None if pts is None else pts * sp.pick.multiplier,
                    minutes=mins,
                    xpts_forecast=r2(fc),
                    diff_vs_forecast=None if pts is None or fc is None else r2(pts - fc),
                )
            )
        cap = next((r for r in out.rows if r.role == "C"), None)
        if cap is not None:
            out.captain, out.captain_points = cap.name, cap.counted_points
        starters = [r for r in out.rows if r.role != "bench" and r.points is not None]
        bench = [r for r in out.rows if r.role == "bench" and r.points is not None]
        if starters:
            best = max(starters, key=lambda r: r.points or 0)
            out.best_starter, out.best_starter_points = best.name, best.points
        if bench:
            b = max(bench, key=lambda r: r.points or 0)
            out.best_bench, out.best_bench_points = b.name, b.points
        return out

def signal_age_hours(signal_as_of: str | None, as_of: datetime) -> float | None:
    if not signal_as_of:
        return None
    dt = datetime.fromisoformat(signal_as_of)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return (as_of - dt) / timedelta(hours=1)
