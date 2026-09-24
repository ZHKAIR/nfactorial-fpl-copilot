"""Компонентный xPts v0: ожидаемые очки игрока на тур как сумма компонентов правил FPL.

    uv run python -m fplcopilot.core.xpts --gw 5 --top 25 [--position MID] [--save]
    uv run python -m fplcopilot.core.xpts --player Haaland --horizon 3
    uv run python -m fplcopilot.core.xpts --explain "João Pedro" [--gw 5]

Для игрока p, матча f (команда t против o) на момент дедлайна тура g:
  appearance     = p_appear·1 + p60·1
  goals          = xG90·m·att_f·pts_goal[pos]        m = exp_minutes/90,
  assists        = xA90·m·att_f·3                    att_f = xg_for(f) / базовый xg_for команды
  clean_sheet    = p60·exp(-xg_against(f))·pts_cs[pos]
  goals_conceded = -p60·E[floor(Poisson(xg_against(f))/2)]          (GKP/DEF)
  saves          = saves90·m·def_f/3                 def_f = xg_against(f) / базовый xg_against
  defcon         = 2·[p_start·P(Pois(rate90·min_start/90) >= T) + (1-p_start)·p_sub·P(Pois(rate90·min_sub/90) >= T)]
  bonus          = 1.2·pct(bps90 внутри позиции)^3·m
  cards          = -yellow90·m
Ставки per-90 сжаты к приору игрока (прошлый сезон, вес m/(m+900) с приором позиции) с весом
minutes/(minutes+450). Blank GW -> все компоненты 0; double GW -> сумма по матчам.
LLM здесь не участвуют. Подробно — docs/xpts.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import text

from fplcopilot.config import settings
from fplcopilot.core import scoring
from fplcopilot.core.ext_stats import (
    ExtIndex,
    blend_rates_with_understat,
    load_ext_index,
    setpiece_bonus_xpts,
    setpiece_phrase,
)
from fplcopilot.core.fixtures import (
    FixtureStrengthProvider,
    FixtureView,
    fixture_views,
    get_provider,
)
from fplcopilot.core.history import (
    SeasonTotals,
    group_by_player,
    load_history,
    load_past_seasons,
    rows_before,
    team_match_stats,
)
from fplcopilot.core.minutes import MinutesEstimate, estimate_minutes
from fplcopilot.core.signals import SignalLite, latest_signals
from fplcopilot.core.stats import expected_floor_div, negbin_sf, percentile_rank, shrink
from fplcopilot.data import Bootstrap, Fixture, FPLClient, Player, Position
from fplcopilot.data.schemas import PlayerGWHistory
from fplcopilot.db import session_scope

log = logging.getLogger(__name__)

MODEL_VERSION = "v0"
EXT_MODEL_VERSION = "v0-ext"  # снимок с Understat/стандартами; v0 (GW5/GW6) не перезаписываем
SHRINK_MINUTES = 450.0  # псевдо-минуты приора для per-90 ставок (5 полных матчей)
PAST_SEASON_K_MINUTES = 900.0  # прошлый сезон весит m/(m+900) против приора позиции
PENALTY_TAKER_XG_BONUS = 0.05  # +xG/90 первому пенальтисту (penalties_order == 1)
# Ассисты FPL шире Opta xA (пенальти «заработан», рикошеты, вторая передача):
# GW1–4 2026/27 — 26 ассистов за тур против 18.2 суммарного xA -> ×1.43; берём консервативно.
ASSIST_XA_MULTIPLIER = 1.35
# bonus = BONUS_MAX[pos] · percentile(bps90 внутри позиции)^3 · minutes/90.
# BONUS_MAX откалиброван так, чтобы сумма по позиции ≈ фактическим бонусам за тур GW1–4
# (GKP 4.8, DEF 18.8, MID 29, FWD 11.3 — у FWD недобор: их бонусы почти целиком от голов).
BONUS_MAX_PER_MATCH: dict[Position, float] = {
    Position.GKP: 0.9,
    Position.DEF: 0.85,
    Position.MID: 1.2,
    Position.FWD: 1.5,
}
BONUS_EXPONENT = 3.0  # медианный игрок позиции ~0.1–0.15 за матч, топ-10% ~0.6–0.9
# Счётчики CBIT/CBIRT сверхдисперсны: pooled Var/mean по матчам >= 60 мин GW1–4 = 1.28
# (DEF 1.48, MID 1.14) -> отрицательный биномиал с Var = 1.3 · mean вместо Пуассона.
DEFCON_DISPERSION = 1.3
MIN_POSITION_MINUTES = 900.0  # меньше — приор позиции берётся из FALLBACK_PRIORS
LIVE_WINDOW = timedelta(minutes=10)  # as_of «почти сейчас» -> статусы из живого bootstrap


@dataclass(frozen=True)
class Rates:
    """Ставки per 90 минут."""

    xg: float
    xa: float
    saves: float
    bps: float
    cbit: float
    cbirt: float
    yellow: float

    def defcon(self, pos: Position) -> float:
        return self.cbirt if scoring.defcon_uses_recoveries(pos) else self.cbit

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in self.__dict__.items()}


# Приоры позиции per 90, если окно данных пустое (GW1) или слишком мало минут.
# Значения — агрегаты GW1–4 сезона 2026/27 (см. docs/xpts.md).
FALLBACK_PRIORS: dict[Position, Rates] = {
    Position.GKP: Rates(xg=0.0, xa=0.005, saves=2.9, bps=15.5, cbit=1.3, cbirt=10.0, yellow=0.03),
    Position.DEF: Rates(xg=0.07, xa=0.06, saves=0.0, bps=13.5, cbit=7.5, cbirt=11.0, yellow=0.18),
    Position.MID: Rates(xg=0.16, xa=0.13, saves=0.0, bps=20.0, cbit=3.9, cbirt=8.1, yellow=0.20),
    Position.FWD: Rates(xg=0.45, xa=0.06, saves=0.0, bps=19.5, cbit=1.7, cbirt=4.1, yellow=0.19),
}


@dataclass(frozen=True)
class Totals:
    minutes: float = 0.0
    xg: float = 0.0
    xa: float = 0.0
    saves: float = 0.0
    bps: float = 0.0
    cbit: float = 0.0
    cbirt: float = 0.0
    yellow: float = 0.0

    @classmethod
    def from_rows(cls, rows: Iterable[PlayerGWHistory]) -> Totals:
        t = defaultdict(float)
        for r in rows:
            t["minutes"] += r.minutes
            t["xg"] += r.expected_goals or 0.0
            t["xa"] += r.expected_assists or 0.0
            t["saves"] += r.saves
            t["bps"] += r.bps
            t["cbit"] += scoring.cbit(r)
            t["cbirt"] += scoring.cbirt(r)
            t["yellow"] += r.yellow_cards
        return cls(**t)

    @classmethod
    def from_past(cls, past: SeasonTotals) -> Totals:
        cbit = past.clearances_blocks_interceptions + past.tackles
        return cls(
            minutes=past.minutes,
            xg=past.expected_goals or 0.0,
            xa=past.expected_assists or 0.0,
            saves=past.saves,
            bps=past.bps,
            cbit=cbit,
            cbirt=cbit + past.recoveries,
            yellow=past.yellow_cards,
        )

    def per90(self) -> Rates | None:
        if self.minutes <= 0:
            return None
        f = 90.0 / self.minutes
        return Rates(
            xg=self.xg * f,
            xa=self.xa * f,
            saves=self.saves * f,
            bps=self.bps * f,
            cbit=self.cbit * f,
            cbirt=self.cbirt * f,
            yellow=self.yellow * f,
        )


def blend_rates(a: Rates, b: Rates, w: float) -> Rates:
    """w·a + (1-w)·b покомпонентно."""
    return Rates(
        *(
            w * x + (1 - w) * y
            for x, y in zip(a.__dict__.values(), b.__dict__.values(), strict=True)
        )
    )


def shrink_rates(totals: Totals, prior: Rates, k_minutes: float = SHRINK_MINUTES) -> Rates:
    """rate90 = (сумма + prior·k/90) / (minutes + k) · 90 — сжатие в единицах минут."""
    own = totals.per90()
    if own is None:
        return prior
    return Rates(
        *(
            shrink(x, p, totals.minutes, k_minutes)
            for x, p in zip(own.__dict__.values(), prior.__dict__.values(), strict=True)
        )
    )


def position_priors(
    history: dict[int, list[PlayerGWHistory]], players: dict[int, Player]
) -> dict[Position, Rates]:
    """Средние per-90 по позиции в окне данных (эмпирический приор); fallback — константы."""
    totals: dict[Position, list[PlayerGWHistory]] = defaultdict(list)
    for pid, rows in history.items():
        player = players.get(pid)
        if player is not None:
            totals[player.position].extend(rows)
    out: dict[Position, Rates] = {}
    for pos in Position:
        t = Totals.from_rows(totals.get(pos, []))
        rates = t.per90() if t.minutes >= MIN_POSITION_MINUTES else None
        out[pos] = rates or FALLBACK_PRIORS[pos]
    return out


def player_prior(past: SeasonTotals | None, pos_prior: Rates) -> Rates:
    """Приор игрока: прошлый сезон с весом m/(m+900), остальное — приор позиции."""
    if past is None or past.minutes <= 0:
        return pos_prior
    own = Totals.from_past(past).per90()
    if own is None:
        return pos_prior
    w = past.minutes / (past.minutes + PAST_SEASON_K_MINUTES)
    return blend_rates(own, pos_prior, w)


# ---------- схемы результата ----------


class XPtsComponents(BaseModel):
    appearance: float = 0.0
    goals: float = 0.0
    assists: float = 0.0
    clean_sheet: float = 0.0
    goals_conceded: float = 0.0
    saves: float = 0.0
    defcon: float = 0.0
    bonus: float = 0.0
    cards: float = 0.0
    setpiece: float = 0.0  # аддитивный бонус пенальтиста/штатника (XPTS_SETPIECE_WEIGHT)

    def total(self) -> float:
        return sum(self.model_dump().values())

    def add(self, other: XPtsComponents) -> XPtsComponents:
        return XPtsComponents(**{k: v + getattr(other, k) for k, v in self.model_dump().items()})


class FixtureInput(BaseModel):
    fixture_id: int
    opponent: str
    is_home: bool
    kickoff_time: datetime | None = None
    xg_for: float
    xg_against: float
    clean_sheet_prob: float
    fixture_strength_index: int
    attack_factor: float
    defence_factor: float
    # чем посчитана сложность: 'odds' (букмекеры) | 'team_rating' (рейтинги команд); только метка,
    # сами коэффициенты/вероятности исходов наружу не выходят
    fsi_source: str = "team_rating"
    # внешняя стата: крутим одним числом в config (XPTS_SETPIECE_WEIGHT / XPTS_UNDERSTAT_BLEND)
    setpiece_bonus: float = 0.0
    understat_blend: float = 0.0


class XPtsBreakdown(BaseModel):
    player_id: int
    web_name: str
    team: str
    position: str
    price: float
    gw: int
    as_of: datetime
    model_version: str = MODEL_VERSION
    xpts: float
    variance: float
    p_start: float
    p_appear: float
    p60: float
    exp_minutes: float
    components: XPtsComponents
    fixtures: list[FixtureInput] = Field(default_factory=list)
    rates: dict[str, float] = Field(default_factory=dict)
    bonus_per_match: float = 0.0
    ep_next: float | None = None
    status: str = "a"
    chance: int | None = None
    signal: dict[str, Any] | None = None
    notes: list[str] = Field(default_factory=list)
    setpiece_bonus: float = 0.0
    understat_blend: float = 0.0

    @property
    def sd(self) -> float:
        return math.sqrt(max(0.0, self.variance))


# ---------- контекст ----------


@dataclass
class ModelContext:
    gw: int
    as_of: datetime  # момент «знания»: история строго до него
    live: bool  # статусы/сигналы из живого bootstrap (иначе допущение status='a')
    bs: Bootstrap
    fixtures: list[Fixture]
    history: dict[int, list[PlayerGWHistory]]
    past: dict[int, SeasonTotals]
    signals: dict[int, SignalLite]
    provider: FixtureStrengthProvider
    team_fixtures: dict[int, list[FixtureView]]
    team_baseline: dict[int, tuple[float, float]]  # team -> (средний xg_for, средний xg_against)
    priors: dict[Position, Rates]
    rates: dict[int, Rates]
    bps_sorted: dict[Position, list[float]]
    n_rounds: int
    # Статусы для не-live контекста (бэктест): player_id -> (status, chance); иначе ('a', None).
    status_overrides: dict[int, tuple[str, int | None]] = field(default_factory=dict)
    ext: ExtIndex | None = None  # Understat + маппинг имён; None в unit-тестах (без сети)

    def status_for(self, player: Player) -> tuple[str, int | None]:
        if self.live:
            return player.status, player.chance_of_playing_next_round
        return self.status_overrides.get(player.id, ("a", None))

    @property
    def is_next_gw(self) -> bool:
        nxt = self.bs.next_event
        return nxt is not None and nxt.id == self.gw

    @property
    def status_gw(self) -> int:
        """Тур, к которому относится статус FPL: ближайший (live) или сам gw (бэктест)."""
        nxt = self.bs.next_event
        return nxt.id if self.live and nxt is not None else self.gw

    def bonus_per_match(self, player: Player) -> float:
        pct = percentile_rank(self.bps_sorted[player.position], self.rates[player.id].bps)
        return BONUS_MAX_PER_MATCH[player.position] * pct**BONUS_EXPONENT


def team_baseline_xg(
    provider: FixtureStrengthProvider, fixtures: list[Fixture]
) -> dict[int, tuple[float, float]]:
    """Средние (xg_for, xg_against) команды по всем её матчам сезона глазами провайдера.

    Нужны, чтобы фактор фикстуры не удваивал силу команды: xG90 игрока уже содержит атаку
    его клуба, поэтому att_f = xg_for(f) / базовый xg_for команды, а не / средний по лиге.
    """
    acc: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for f in fixtures:
        try:
            h, a = provider.expected_goals(f)
        except (KeyError, NotImplementedError):
            continue
        acc[f.team_h].append((h, a))
        acc[f.team_a].append((a, h))
    return {
        t: (sum(x for x, _ in v) / len(v), sum(y for _, y in v) / len(v)) for t, v in acc.items()
    }


def make_context(
    gw: int,
    *,
    bs: Bootstrap,
    fixtures: list[Fixture],
    history_rows: Sequence[PlayerGWHistory],
    past: dict[int, SeasonTotals] | None = None,
    signals: dict[int, SignalLite] | None = None,
    as_of: datetime | None = None,
    provider: FixtureStrengthProvider | None = None,
    live: bool | None = None,
    now: datetime | None = None,
    status_overrides: dict[int, tuple[str, int | None]] | None = None,
    ext: ExtIndex | None = None,
) -> ModelContext:
    """Собирает всё нужное для прогноза на тур gw без обращений к БД (тесты, бэктест)."""
    now = now or datetime.now(UTC)
    event = next((e for e in bs.events if e.id == gw), None)
    deadline = event.deadline_time if event else None
    if as_of is None:
        as_of = now if deadline is None else min(now, deadline)
    elif deadline is not None:
        as_of = min(as_of, deadline)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=UTC)
    if live is None:
        live = as_of >= now - LIVE_WINDOW

    rows = rows_before(list(history_rows), gw, as_of)
    history = group_by_player(rows)
    players = bs._players_by_id
    if provider is None:
        stats = team_match_stats(gw - 1, rows=rows, fixtures=fixtures, deadline=as_of)
        provider = get_provider(teams=bs.teams, match_stats=stats, calibration_fixtures=fixtures)
    team_fixtures = fixture_views(provider, fixtures, gw)
    baseline = team_baseline_xg(provider, fixtures)

    priors = position_priors(history, players)
    past = past or {}
    rates: dict[int, Rates] = {}
    bps: dict[Position, list[float]] = defaultdict(list)
    bps_all: dict[Position, list[float]] = defaultdict(list)
    for pid, player in players.items():
        totals = Totals.from_rows(history.get(pid, []))
        prior = player_prior(past.get(pid), priors[player.position])
        rates[pid] = shrink_rates(totals, prior)
        bps_all[player.position].append(rates[pid].bps)
        if totals.minutes > 0:
            bps[player.position].append(rates[pid].bps)
        if ext is not None:
            rates[pid] = blend_rates_with_understat(rates[pid], ext.player_row(pid))
    bps_sorted = {pos: sorted(bps.get(pos) or bps_all.get(pos) or [0.0]) for pos in Position}

    return ModelContext(
        gw=gw,
        as_of=as_of,
        live=live,
        bs=bs,
        fixtures=fixtures,
        history=history,
        past=past,
        signals=signals or {},
        provider=provider,
        team_fixtures=team_fixtures,
        team_baseline=baseline,
        priors=priors,
        rates=rates,
        bps_sorted=bps_sorted,
        n_rounds=max((r.round for r in rows), default=0),
        status_overrides=status_overrides or {},
        ext=ext,
    )


def build_context(
    gw: int,
    as_of: datetime | None = None,
    *,
    client: FPLClient | None = None,
    provider_name: str | None = None,
) -> ModelContext:
    """Контекст из БД (player_gw_history, player_season_history, player_signals) и FPL API."""
    client = client or FPLClient()
    bs = client.bootstrap()
    fixtures = client.fixtures()
    event = next((e for e in bs.events if e.id == gw), None)
    if event is None:
        raise ValueError(f"тур {gw} не найден в bootstrap")
    now = datetime.now(UTC)
    cutoff = min(as_of or now, event.deadline_time)
    rows = load_history(before_gw=gw, deadline=cutoff)
    if not rows and gw > 1:
        log.warning("player_gw_history пуста: uv run python -m fplcopilot.core.history --sync")
    provider = None
    if provider_name:
        stats = team_match_stats(gw - 1, rows=rows, fixtures=fixtures, deadline=cutoff)
        provider = get_provider(
            provider_name, teams=bs.teams, match_stats=stats, calibration_fixtures=fixtures
        )
    ext = None
    if settings.understat_enabled:
        ext = load_ext_index(bs)
    return make_context(
        gw,
        bs=bs,
        fixtures=fixtures,
        history_rows=rows,
        past=load_past_seasons(),
        signals=latest_signals(cutoff),
        as_of=cutoff,
        provider=provider,
        now=now,
        ext=ext,
    )


# ---------- ядро ----------


def fixture_components(
    pos: Position,
    rates: Rates,
    mins: MinutesEstimate,
    fv: FixtureView,
    *,
    baseline: tuple[float, float],
    bonus_per_match: float,
    penalty_taker: bool = False,
) -> tuple[XPtsComponents, float, FixtureInput]:
    """Компоненты, дисперсия и входы для одного матча."""
    m = mins.exp_minutes / 90.0
    base_for, base_against = baseline
    att = fv.xg_for / base_for if base_for > 0 else 1.0
    dfn = fv.xg_against / base_against if base_against > 0 else 1.0

    xg90 = rates.xg + (PENALTY_TAKER_XG_BONUS if penalty_taker else 0.0)
    lam_goals = xg90 * m * att
    lam_assists = rates.xa * ASSIST_XA_MULTIPLIER * m * att
    p_cs = mins.p60 * fv.clean_sheet_prob
    # E[floor(G/2)] по Пуассону точнее наивного 0.5·λ (0.49 против 0.73 при λ=1.45)
    gc = (
        expected_floor_div(fv.xg_against, scoring.GOALS_CONCEDED_PER_POINT)
        if scoring.concedes_penalty(pos)
        else 0.0
    )
    lam_saves = rates.saves * m * dfn if pos == Position.GKP else 0.0
    save_pts = expected_floor_div(lam_saves, scoring.SAVES_PER_POINT) if lam_saves else 0.0

    p_dc = 0.0
    thr = scoring.defcon_threshold(pos)
    if thr is not None:
        rate = rates.defcon(pos)
        p_dc = mins.p_start * negbin_sf(
            thr, rate * mins.exp_minutes_if_start / 90.0, DEFCON_DISPERSION
        ) + (1 - mins.p_start) * mins.p_sub * negbin_sf(
            thr, rate * mins.sub_minutes / 90.0, DEFCON_DISPERSION
        )

    goal_pts = scoring.goal_points(pos)
    cs_pts = scoring.clean_sheet_points(pos)
    comps = XPtsComponents(
        appearance=mins.p_appear * scoring.PTS_APPEARANCE + mins.p60 * scoring.PTS_APPEARANCE_60,
        goals=lam_goals * goal_pts,
        assists=lam_assists * scoring.PTS_ASSIST,
        clean_sheet=p_cs * cs_pts,
        goals_conceded=-mins.p60 * gc,
        saves=save_pts,
        defcon=scoring.DEFCON_POINTS * p_dc,
        bonus=bonus_per_match * m,
        cards=rates.yellow * m * scoring.PTS_YELLOW,
    )
    # Эвристика дисперсии: биномиальные члены для вероятностей, пуассоновские для счётчиков.
    variance = (
        mins.p_appear * (1 - mins.p_appear)
        + mins.p60 * (1 - mins.p60)
        + goal_pts**2 * lam_goals
        + scoring.PTS_ASSIST**2 * lam_assists
        + cs_pts**2 * p_cs * (1 - p_cs)
        + mins.p60 * gc
        + save_pts
        + scoring.DEFCON_POINTS**2 * p_dc * (1 - p_dc)
        + bonus_per_match * m
        + rates.yellow * m
    )
    inputs = FixtureInput(
        fixture_id=fv.fixture_id,
        opponent="",
        is_home=fv.is_home,
        kickoff_time=fv.kickoff_time,
        xg_for=round(fv.xg_for, 3),
        xg_against=round(fv.xg_against, 3),
        clean_sheet_prob=round(fv.clean_sheet_prob, 4),
        fixture_strength_index=fv.fixture_strength_index,
        attack_factor=round(att, 3),
        defence_factor=round(dfn, 3),
        fsi_source=fv.fsi_source,
    )
    return comps, variance, inputs


def predict_player(
    player_id: int, gw: int, as_of: datetime | None = None, *, ctx: ModelContext | None = None
) -> XPtsBreakdown:
    ctx = ctx or build_context(gw, as_of)
    player = ctx.bs.player(player_id)
    pos = player.position
    rows = ctx.history.get(player_id, [])
    status, chance = ctx.status_for(player)
    signal = ctx.signals.get(player_id)
    mins = estimate_minutes(
        rows,
        position=pos,
        status=status,
        chance=chance,
        signal=signal,
        past=ctx.past.get(player_id),
        gw=ctx.gw,
        status_gw=ctx.status_gw,
    )
    rates = ctx.rates[player_id]
    bonus = ctx.bonus_per_match(player)
    baseline = ctx.team_baseline.get(player.team, (0.0, 0.0))

    comps = XPtsComponents()
    variance = 0.0
    inputs: list[FixtureInput] = []
    raw_setpiece = setpiece_bonus_xpts(player)
    minutes_scale = min(1.0, mins.exp_minutes / 90.0) if mins.exp_minutes > 0 else 0.0
    applied_setpiece = raw_setpiece * minutes_scale
    blend_w = (
        settings.xpts_understat_blend
        if ctx.ext is not None and ctx.ext.player_row(player.id) is not None
        else 0.0
    )
    for fv in ctx.team_fixtures.get(player.team, []):
        c, v, inp = fixture_components(
            pos,
            rates,
            mins,
            fv,
            baseline=baseline,
            bonus_per_match=bonus,
            penalty_taker=player.penalties_order == 1,
        )
        inp.opponent = ctx.bs.team(fv.opponent_id).short_name
        inp.setpiece_bonus = applied_setpiece
        inp.understat_blend = blend_w
        comps = comps.add(c)
        variance += v
        inputs.append(inp)
    if inputs and applied_setpiece:
        comps = comps.add(XPtsComponents(setpiece=applied_setpiece))

    notes = list(mins.notes)
    if not inputs:
        notes.append("blank gameweek: no fixture")
    if applied_setpiece:
        notes.append(f"setpiece_bonus={applied_setpiece:.2f}")
        phrase = setpiece_phrase(player, ctx.bs.team(player.team))
        if phrase:
            notes.append(phrase)
    if blend_w:
        notes.append(f"understat_blend={blend_w:.2f}")
    return XPtsBreakdown(
        player_id=player.id,
        web_name=player.web_name,
        team=ctx.bs.team(player.team).short_name,
        position=pos.short,
        price=player.price,
        gw=ctx.gw,
        as_of=ctx.as_of,
        model_version=EXT_MODEL_VERSION,
        xpts=comps.total(),
        variance=variance,
        p_start=mins.p_start,
        p_appear=mins.p_appear,
        p60=mins.p60,
        exp_minutes=mins.exp_minutes,
        components=comps,
        fixtures=inputs,
        rates=rates.as_dict(),
        bonus_per_match=round(bonus, 3),
        ep_next=player.ep_next if ctx.is_next_gw else None,
        status=status,
        chance=chance,
        signal=(
            {
                "availability": signal.availability,
                "start_probability": signal.start_probability,
                "confidence": signal.confidence,
                "as_of": signal.as_of.isoformat(),
                "used": signal.usable and mins.signal_weight > 0,
                "rotation_risk": signal.rotation_risk,
                "return_gw": signal.return_gw,
            }
            if signal
            else None
        ),
        notes=notes,
        setpiece_bonus=applied_setpiece,
        understat_blend=blend_w,
    )


def predict_all(
    gw: int, as_of: datetime | None = None, *, ctx: ModelContext | None = None
) -> list[XPtsBreakdown]:
    ctx = ctx or build_context(gw, as_of)
    preds = [predict_player(p.id, gw, ctx=ctx) for p in ctx.bs.elements]
    preds.sort(key=lambda b: -b.xpts)
    return preds


def predict_horizon(
    player_id: int,
    gws: Sequence[int],
    as_of: datetime | None = None,
    *,
    client: FPLClient | None = None,
) -> list[XPtsBreakdown]:
    client = client or FPLClient()
    return [
        predict_player(player_id, gw, ctx=build_context(gw, as_of, client=client)) for gw in gws
    ]


# ---------- персистентность ----------

_UPSERT_PREDICTION = text(
    """
    INSERT INTO xpts_predictions
        (player_id, gw, as_of, model_version, xpts, p_start, exp_minutes, components, variance,
         ep_next_snapshot)
    VALUES
        (:player_id, :gw, :as_of, :model_version, :xpts, :p_start, :exp_minutes,
         CAST(:components AS jsonb), :variance, :ep_next_snapshot)
    ON CONFLICT (player_id, gw, model_version, as_of) DO UPDATE SET
        xpts = EXCLUDED.xpts, p_start = EXCLUDED.p_start, exp_minutes = EXCLUDED.exp_minutes,
        components = EXCLUDED.components, variance = EXCLUDED.variance,
        ep_next_snapshot = EXCLUDED.ep_next_snapshot
    """
)


def save_predictions(preds: Sequence[XPtsBreakdown]) -> int:
    """Пишет прогнозы в xpts_predictions (ep_next_snapshot — ep_next из bootstrap для next GW)."""
    params = [
        {
            "player_id": p.player_id,
            "gw": p.gw,
            "as_of": p.as_of,
            "model_version": p.model_version,
            "xpts": p.xpts,
            "p_start": p.p_start,
            "exp_minutes": p.exp_minutes,
            "components": json.dumps(p.components.model_dump()),
            "variance": p.variance,
            "ep_next_snapshot": p.ep_next,
        }
        for p in preds
    ]
    if not params:
        return 0
    with session_scope() as s:
        s.execute(_UPSERT_PREDICTION, params)
    return len(params)


# ---------- вывод ----------

COMPONENT_COLUMNS = (
    ("app", "appearance"),
    ("gls", "goals"),
    ("ast", "assists"),
    ("cs", "clean_sheet"),
    ("gc", "goals_conceded"),
    ("sav", "saves"),
    ("dc", "defcon"),
    ("bon", "bonus"),
    ("crd", "cards"),
)


def format_table(preds: Sequence[XPtsBreakdown], *, title: str | None = None) -> str:
    head = (
        f"{'#':>3} {'player':<16} {'team':<4} {'pos':<3} {'£':>5} {'xPts':>5} {'±':>4} {'ep':>4} "
        f"{'pS':>4} {'min':>4} " + " ".join(f"{c:>4}" for c, _ in COMPONENT_COLUMNS) + "  fixture"
    )
    lines = [title, head, "-" * len(head)] if title else [head, "-" * len(head)]
    for i, p in enumerate(preds, start=1):
        comps = p.components.model_dump()
        fx = (
            " ".join(
                f"{'v' if f.is_home else '@'}{f.opponent}({f.fixture_strength_index})"
                for f in p.fixtures
            )
            or "blank"
        )
        ep = f"{p.ep_next:4.1f}" if p.ep_next is not None else "   -"
        lines.append(
            f"{i:>3} {p.web_name[:16]:<16} {p.team:<4} {p.position:<3} {p.price:>5.1f} "
            f"{p.xpts:>5.2f} {p.sd:>4.1f} {ep} {p.p_start:>4.2f} {p.exp_minutes:>4.0f} "
            + " ".join(f"{comps[key]:>4.2f}" for _, key in COMPONENT_COLUMNS)
            + f"  {fx}"
        )
    return "\n".join(lines)


def explain(pred: XPtsBreakdown) -> str:
    comps = pred.components.model_dump()
    ep = f"   FPL ep_next = {pred.ep_next}" if pred.ep_next is not None else ""
    lines = [
        (
            f"{pred.web_name} ({pred.team}, {pred.position}, £{pred.price}) — GW{pred.gw} "
            f"as of {pred.as_of:%Y-%m-%d %H:%M}Z, model {pred.model_version}"
        ),
        f"  xPts = {pred.xpts:.2f} (sd {pred.sd:.2f}){ep}",
        "  components: " + ", ".join(f"{k}={v:+.2f}" for k, v in comps.items()),
        (
            f"  minutes: p_start={pred.p_start:.2f} p_appear={pred.p_appear:.2f} "
            f"p60={pred.p60:.2f} exp_minutes={pred.exp_minutes:.0f}  "
            f"status={pred.status} chance={pred.chance}"
        ),
        "  rates/90: "
        + ", ".join(f"{k}={v:.3f}" for k, v in pred.rates.items())
        + f", bonus/match={pred.bonus_per_match:.2f}",
    ]
    for f in pred.fixtures:
        lines.append(
            f"  fixture {'vs' if f.is_home else 'at'} {f.opponent}: xg_for={f.xg_for:.2f} "
            f"xg_against={f.xg_against:.2f} cs={f.clean_sheet_prob:.2f} "
            f"FSI={f.fixture_strength_index} ({f.fsi_source}) att_factor={f.attack_factor:.2f} "
            f"def_factor={f.defence_factor:.2f}"
        )
    if pred.signal:
        s = pred.signal
        ret = f" return_gw={s['return_gw']}" if s.get("return_gw") is not None else ""
        lines.append(
            f"  news signal: {s['availability']} p_start={s['start_probability']:.2f} "
            f"conf={s['confidence']:.2f} as_of={s['as_of']} used={s['used']}{ret}"
        )
    else:
        lines.append("  news signal: none")
    for n in pred.notes:
        lines.append(f"  note: {n}")
    return "\n".join(lines)


def _fold(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    ).lower()


def find_player(bs: Bootstrap, query: str) -> Player:
    """Поиск по имени без учёта диакритики; несколько совпадений -> предпочитаем точное web_name."""
    q = _fold(query)
    found = [p for p in bs.elements if q in _fold(p.web_name) or q in _fold(p.full_name)]
    if not found:
        raise LookupError(f"игрок {query!r} не найден")
    exact = [p for p in found if _fold(p.web_name) == q]
    if exact:
        return exact[0]
    if len(found) > 1:
        found.sort(key=lambda p: -p.minutes)
        log.info("несколько совпадений для %r: %s", query, [p.web_name for p in found[:5]])
    return found[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FPL Copilot: компонентный xPts v0")
    ap.add_argument("--gw", type=int, help="тур (по умолчанию следующий)")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--position", choices=[p.name for p in Position])
    ap.add_argument("--player", help="имя игрока: прогноз на горизонт")
    ap.add_argument("--horizon", type=int, default=1, help="число туров начиная с --gw")
    ap.add_argument("--explain", help="имя игрока: разбор компонентов и входов")
    ap.add_argument("--save", action="store_true", help="записать в xpts_predictions")
    ap.add_argument("--as-of", dest="as_of", help="ISO-время (по умолчанию сейчас)")
    ap.add_argument("--provider", choices=["team_rating", "odds"])
    ap.add_argument(
        "--model-version",
        dest="model_version",
        default=EXT_MODEL_VERSION,
        help=(
            f"метка model_version для --save (по умолчанию {EXT_MODEL_VERSION}; "
            f"снимок эксперимента GW5/GW6 — {MODEL_VERSION}, его не перезаписывать; "
            "odds-вариант — v0-odds)"
        ),
    )
    ap.add_argument("--json", action="store_true", help="JSON вместо таблицы")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    client = FPLClient()
    bs = client.bootstrap()
    gw = args.gw or client.next_gw() or 1
    as_of = None
    if args.as_of:
        as_of = datetime.fromisoformat(args.as_of)
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=UTC)

    if args.player or args.explain:
        try:
            player = find_player(bs, args.player or args.explain)
        except LookupError as exc:
            print(f"ошибка: {exc}", file=sys.stderr)
            return 2
        gws = list(range(gw, gw + (args.horizon if args.player else 1)))
        preds = []
        for g in gws:
            ctx = build_context(g, as_of, client=client, provider_name=args.provider)
            preds.append(predict_player(player.id, g, ctx=ctx))
        for p in preds:
            p.model_version = args.model_version
        if args.json:
            print(
                json.dumps([p.model_dump(mode="json") for p in preds], indent=2, ensure_ascii=False)
            )
        elif args.explain:
            print(explain(preds[0]))
        else:
            print(format_table(preds, title=f"{player.web_name}: GW{gws[0]}–GW{gws[-1]}"))
            print(f"total xPts over horizon: {sum(p.xpts for p in preds):.2f}")
        if args.save:
            print(f"saved {save_predictions(preds)} rows")
        return 0

    ctx = build_context(gw, as_of, client=client, provider_name=args.provider)
    preds = predict_all(gw, ctx=ctx)
    for p in preds:
        p.model_version = args.model_version
    shown = [p for p in preds if not args.position or p.position == args.position][: args.top]
    if args.json:
        print(json.dumps([p.model_dump(mode="json") for p in shown], indent=2, ensure_ascii=False))
    else:
        sources = sorted({f.fsi_source for p in preds for f in p.fixtures})
        title = (
            f"xPts {args.model_version} GW{gw} as of {ctx.as_of:%Y-%m-%d %H:%M}Z "
            f"(provider={ctx.provider.name}, fsi_source={'+'.join(sources) or '-'}, "
            f"rounds used={ctx.n_rounds}, live={ctx.live})"
        )
        print(format_table(shown, title=title))
    if args.save:
        print(f"saved {save_predictions(preds)} rows into xpts_predictions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
