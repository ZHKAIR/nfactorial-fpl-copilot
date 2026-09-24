"""MILP-оптимизатор состава: лучшие 11, один трансфер (top-3 маршрутов), многотуровый план с
сравнением Wildcard и запланированными Bench Boost / Triple Captain. PuLP + HiGHS (highspy);
CBC из комплекта PuLP — запасной вариант.

    uv run python -m fplcopilot.core.optimizer --manager 895045 --gw 5 --strategy balanced --xi
    uv run python -m fplcopilot.core.optimizer --manager 895045 --gw 5 --transfer [--allow-hit] [--horizon 3]
    uv run python -m fplcopilot.core.optimizer --manager 895045 --gw 5 --plan --horizon 5 [--save]
    uv run python -m fplcopilot.core.optimizer --manager 895045 --gw 5 --plan --chip 7:bboost
    uv run python -m fplcopilot.core.optimizer --manager 895045 --gw 5 --diagnose

Одна модель на всё (build_model): переменные по игроку p и туру w — squad, start, captain,
bench-слот (13–15; запасной вратарь линейно = squad − start), t_in, t_out; по туру — paid,
ft, itb. best_xi = модель на один тур без трансферов; single_transfer = трансферы только в
from_gw (top-3 через no-good cuts); plan_transfers = трансферы в каждом туре горизонта, FT
копятся до 5, второй solve с Wildcard в from_gw. Фишка тура (ModelSpec.chips) меняет только
цель этого тура: Bench Boost — очки всех 15, Triple Captain — капитан ×3. Формулировка,
константы и ограничения — docs/optimizer.md. LLM здесь нет: стратегия меняет коэффициенты,
не текст.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import pulp
from pydantic import BaseModel, ConfigDict, Field

from fplcopilot.config import settings
from fplcopilot.core.candidates import (
    Candidate,
    PredictionStore,
    SquadIssue,
    build_candidates,
    build_pool,
    chips_available,
    diagnose_squad,
    problem_players,
    squad_state,
    wildcard_available,
)
from fplcopilot.core.strategy import (
    FIRST_HALF_LAST_GW,
    FORMATION_BOUNDS,
    HIT_COST,
    MAX_FREE_TRANSFERS,
    MAX_PER_CLUB,
    POSITION_QUOTA,
    SQUAD_SIZE,
    STARTERS,
    Strategy,
    get_strategy,
)
from fplcopilot.data import FPLClient

log = logging.getLogger(__name__)

GK = 1
BENCH_SLOTS = (13, 14, 15)  # полевые запасные; слот 12 — вратарь
Recommendation = Literal["transfers", "wildcard", "hold"]
Verdict = Literal["go", "hold", "hit_not_worth"]

# Фишки, которые план умеет ставить на тур (имена FPL API); Free Hit не моделируется.
BENCH_BOOST = "bboost"
TRIPLE_CAPTAIN = "3xc"
PLANNABLE_CHIPS = (BENCH_BOOST, TRIPLE_CAPTAIN)
CHIP_TITLES = {
    "wildcard": "Wildcard",
    "freehit": "Free Hit",
    BENCH_BOOST: "Bench Boost",
    TRIPLE_CAPTAIN: "Triple Captain",
}


# ---------- результаты ----------


class LineupResult(BaseModel):
    gw: int
    formation: str
    starters: list[int]
    bench_order: list[int]  # слот 12 (вратарь), затем полевые по убыванию xPts
    captain: int
    vice: int
    expected_points: float  # Σ xPts стартеров + xPts капитана (удвоение) + chip_points
    objective: float  # вклад тура в цель: + скамейка·веса − λ·дисперсия ± владение
    chip: str | None = None  # bboost | 3xc — фишка тура
    chip_points: float = 0.0  # вклад фишки: BB — Σ xPts скамейки, TC — ещё раз xPts капитана


class TransferRoute(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    gw: int
    out: list[int]
    in_: list[int] = Field(alias="in")
    out_names: list[str] = Field(default_factory=list)
    in_names: list[str] = Field(default_factory=list)
    hit_cost: int
    expected_gain_horizon: (
        float  # Σ по горизонту (XI + капитан) против «без трансферов», без дисконта
    )
    expected_gain_discounted: float  # то же с decay^(w − gw); база для вердикта по хиту
    expected_gain_next_gw: float
    objective_gain: float  # разность целевых функций (учитывает ft_value, скамейку, λ, владение)
    hit_marginal_gain: float | None = None  # (Δdisc сверх лучшего бесплатного маршрута) − хит
    new_bank: float
    risk_note: str
    worth_hit: bool | None  # None — хита нет
    verdict: Verdict
    lineup_after: LineupResult


class PlannedMove(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    gw: int
    out: int
    in_: int = Field(alias="in")
    out_name: str = ""
    in_name: str = ""
    position: int = 0
    price_out: float = 0.0
    price_in: float = 0.0
    delta_xpts_horizon: float = 0.0  # xPts(in) − xPts(out) с тура трансфера до конца горизонта
    out_problem: str | None = (
        None  # injured | suspended | unavailable | doubtful | not_playing | fixtures | low_xpts | blank
    )
    paid: bool = False  # трансфер платный (−4) в этом туре


class WildcardAlternative(BaseModel):
    gw: int
    squad: list[int]
    squad_names: list[str] = Field(default_factory=list)
    expected_total: float
    objective: float
    moves: list[PlannedMove] = Field(default_factory=list)
    lineups_by_gw: dict[int, LineupResult] = Field(default_factory=dict)


class TransferPlan(BaseModel):
    manager_id: int
    from_gw: int
    horizon: int
    strategy: str
    moves_by_gw: dict[int, list[PlannedMove]]
    hits_by_gw: dict[int, int]
    ft_by_gw: dict[int, int]  # FT на дедлайн тура (последний ключ — после горизонта)
    bank_by_gw: dict[int, float]
    lineups_by_gw: dict[int, LineupResult]
    target_squad: list[int]
    target_squad_names: list[str] = Field(default_factory=list)
    expected_total: float  # Σ_w (XI + капитан + вклад фишки − 4·paid) без дисконта
    baseline_total: float  # без трансферов, FT копятся, те же фишки
    # очки лучших 11 из нынешних 15 по турам без трансферов (та же baseline-модель, те же фишки)
    baseline_points_by_gw: dict[int, float] = Field(default_factory=dict)
    objective: float
    baseline_objective: float
    wildcard_alternative: WildcardAlternative | None = None
    recommendation: Recommendation
    issues: list[SquadIssue] = Field(default_factory=list)
    solver: str
    runtime_s: float
    time_limit_hit: bool = False
    allow_hits: bool = True  # False — план построен без платных трансферов (paid[w] = 0)
    chips_by_gw: dict[int, str] = Field(default_factory=dict)  # запланированные BB / TC
    notes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------- солвер ----------


def make_solver(
    *, time_limit: float | None = None, gap: float | None = None, msg: bool = False
) -> tuple[pulp.LpSolver, str]:
    """HiGHS через highspy (auto/highs) или CBC из PuLP (cbc). См. settings.optimizer_solver."""
    name = settings.optimizer_solver
    time_limit = settings.optimizer_time_limit_s if time_limit is None else time_limit
    gap = settings.optimizer_mip_gap if gap is None else gap
    if name in ("auto", "highs"):
        highs = pulp.HiGHS(msg=msg, timeLimit=time_limit, gapRel=gap)
        if highs.available():
            return highs, "HiGHS"
        if name == "highs":
            raise RuntimeError("HiGHS недоступен: uv add highspy")
    cbc = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit, gapRel=gap)
    if cbc.available():
        return cbc, "CBC"
    raise RuntimeError("нет доступного MILP-солвера (HiGHS/CBC)")


# ---------- модель ----------


@dataclass
class ModelSpec:
    gws: list[int]
    pool: dict[int, Candidate]
    initial_squad: list[int]
    bank: float
    free_transfers: int
    strategy: Strategy
    max_transfers: dict[int, int | None]  # по туру; None — без лимита, 0 — запрет
    wildcard_gw: int | None = None
    keep: frozenset[int] = frozenset()
    exclude: frozenset[int] = frozenset()  # нельзя купить
    allow_hits: bool = True  # False — paid[w] = 0 в каждом туре: только бесплатные трансферы
    chips: Mapping[int, str] = field(default_factory=dict)  # тур -> bboost | 3xc

    def validate(self) -> None:
        missing = [p for p in self.initial_squad if p not in self.pool]
        if missing:
            raise ValueError(f"игроки состава отсутствуют в пуле: {missing}")
        if len(set(self.initial_squad)) != SQUAD_SIZE:
            raise ValueError(
                f"в составе должно быть {SQUAD_SIZE} игроков, есть {len(self.initial_squad)}"
            )
        for w, chip in self.chips.items():
            if chip not in PLANNABLE_CHIPS or w not in self.gws:
                raise ChipPlanError(f"фишка {chip!r} в GW{w} вне модели (горизонт {self.gws})")
        if self.wildcard_gw is not None and self.wildcard_gw in self.chips:
            raise ChipPlanError(
                f"GW{self.wildcard_gw}: Wildcard и {CHIP_TITLES[self.chips[self.wildcard_gw]]} "
                "в одном туре нельзя — одна фишка за тур"
            )


@dataclass
class Solution:
    status: str
    objective: float
    squads: dict[int, list[int]]
    lineups: dict[int, LineupResult]
    transfers: dict[int, tuple[list[int], list[int]]]  # gw -> (out, in)
    paid: dict[int, int]
    ft: dict[int, int]  # FT на дедлайн тура; ключ gws[-1] + 1 — после горизонта
    itb: dict[int, float]
    runtime_s: float
    time_limit_hit: bool = False

    def expected_total(self) -> float:
        return sum(
            lu.expected_points - HIT_COST * self.paid.get(gw, 0) for gw, lu in self.lineups.items()
        )

    def horizon_points(self, *, decay: float = 1.0, from_gw: int | None = None) -> float:
        g0 = from_gw if from_gw is not None else min(self.lineups)
        return sum(decay ** (gw - g0) * lu.expected_points for gw, lu in self.lineups.items())


@dataclass
class _Vars:
    sq: dict[tuple[int, int], pulp.LpVariable]
    st: dict[tuple[int, int], pulp.LpVariable]
    cp: dict[tuple[int, int], pulp.LpVariable]
    bench: dict[tuple[int, int, int], pulp.LpVariable]
    tin: dict[tuple[int, int], pulp.LpVariable]
    tout: dict[tuple[int, int], pulp.LpVariable]
    paid: dict[int, pulp.LpVariable]
    ft: dict[int, pulp.LpVariable | float]
    itb: dict[int, pulp.LpVariable]
    cuts: int = 0


@dataclass
class Model:
    spec: ModelSpec
    prob: pulp.LpProblem = field(init=False)
    v: _Vars = field(init=False)

    def __post_init__(self) -> None:
        self.spec.validate()
        self.prob, self.v = build_model(self.spec)

    def add_no_good_cut(self, gw: int, out: Sequence[int], inn: Sequence[int]) -> None:
        """Запретить ровно этот набор покупок в туре gw: Σ_{p∈S} in_p − Σ_{p∉S} in_p <= |S| − 1
        (канонический no-good cut: S запрещён, надмножества и другие наборы разрешены).
        Top-k альтернатив различаются тем, кого берём; смена продаваемого «энейблера» при тех же
        покупках — не новый маршрут. Без покупок режем набор продаж; пустой — требуем трансфер."""
        self.v.cuts += 1
        vars_gw = {p: v for (p, w), v in self.v.tin.items() if w == gw}
        chosen = set(inn) & set(vars_gw)
        if not chosen:
            vars_gw = {p: v for (p, w), v in self.v.tout.items() if w == gw}
            chosen = set(out) & set(vars_gw)
        if not chosen:
            self.prob += (
                pulp.lpSum(v for (p, w), v in self.v.tin.items() if w == gw) >= 1,
                f"cut_{self.v.cuts}",
            )
            return
        inside = pulp.lpSum(vars_gw[p] for p in chosen)
        outside = pulp.lpSum(v for p, v in vars_gw.items() if p not in chosen)
        self.prob += inside - outside <= len(chosen) - 1, f"cut_{self.v.cuts}"

    def require_transfer(self, gw: int) -> None:
        self.prob += (
            pulp.lpSum(v for (p, w), v in self.v.tin.items() if w == gw) >= 1,
            f"require_transfer_{gw}",
        )

    def solve(
        self, *, time_limit: float | None = None, gap: float | None = None, msg: bool = False
    ) -> tuple[Solution, str]:
        solver, name = make_solver(time_limit=time_limit, gap=gap, msg=msg)
        started = time.perf_counter()
        self.prob.solve(solver)
        runtime = time.perf_counter() - started
        status = pulp.LpStatus[self.prob.status]
        if self.prob.status != pulp.LpStatusOptimal:
            raise InfeasibleError(f"solver status {status}: модель без допустимого решения")
        limit_hit = self.prob.sol_status == pulp.LpSolutionIntegerFeasible
        return extract_solution(self.spec, self.v, self.prob, runtime, limit_hit), name


class InfeasibleError(RuntimeError):
    pass


class ChipPlanError(ValueError):
    """Невалидное условие «фишка в туре N»: правила FPL, горизонт плана, доступность фишки."""


def build_model(spec: ModelSpec) -> tuple[pulp.LpProblem, _Vars]:
    S = spec.strategy
    pool = spec.pool
    gws = list(spec.gws)
    g0 = gws[0]
    end = gws[-1] + 1
    init = set(spec.initial_squad)
    players = sorted(pool)
    outfield = [p for p in players if pool[p].position != GK]
    keepers = [p for p in players if pool[p].position == GK]

    prob = pulp.LpProblem("fpl", pulp.LpMaximize)
    B = "Binary"
    var = prob.add_variable  # API PuLP 4.0 (без DeprecationWarning на прямой LpVariable)
    sq = {(p, w): var(f"sq_{p}_{w}", cat=B) for p in players for w in gws}
    st = {(p, w): var(f"st_{p}_{w}", cat=B) for p in players for w in gws}
    cp = {(p, w): var(f"cp_{p}_{w}", cat=B) for p in players for w in gws}
    bench = {
        (p, s, w): var(f"bn_{p}_{s}_{w}", cat=B) for p in outfield for s in BENCH_SLOTS for w in gws
    }
    transfers_allowed = {w: spec.max_transfers.get(w, None) != 0 for w in gws}
    tin = {
        (p, w): var(f"in_{p}_{w}", cat=B)
        for p in players
        if p not in spec.exclude
        for w in gws
        if transfers_allowed[w]
    }
    tout = {
        (p, w): var(f"out_{p}_{w}", cat=B)
        for p in players
        if p not in spec.keep
        for w in gws
        if transfers_allowed[w]
    }
    paid = {w: var(f"paid_{w}", lowBound=0) for w in gws}
    ft: dict[int, pulp.LpVariable | float] = {
        g0: float(min(spec.free_transfers, MAX_FREE_TRANSFERS))
    }
    for w in gws[1:] + [end]:
        ft[w] = var(f"ft_{w}", lowBound=1, upBound=MAX_FREE_TRANSFERS, cat="Integer")
    itb = {w: var(f"itb_{w}", lowBound=0) for w in gws}

    obj = pulp.LpAffineExpression()
    prev_sq: dict[int, Any] = {p: (1.0 if p in init else 0.0) for p in players}
    prev_itb: Any = float(spec.bank)
    for w in gws:
        d = S.decay(w, g0)
        # --- состав ---
        prob += pulp.lpSum(sq[p, w] for p in players) == SQUAD_SIZE, f"squad_size_{w}"
        for pos, quota in POSITION_QUOTA.items():
            prob += (
                pulp.lpSum(sq[p, w] for p in players if pool[p].position == pos) == quota,
                f"quota_{pos}_{w}",
            )
        teams: dict[int, list[int]] = {}
        for p in players:
            teams.setdefault(pool[p].team_id, []).append(p)
        for team, members in teams.items():
            if len(members) > MAX_PER_CLUB:
                prob += pulp.lpSum(sq[p, w] for p in members) <= MAX_PER_CLUB, f"club_{team}_{w}"
        # --- стартовые 11, капитан ---
        for p in players:
            prob += st[p, w] <= sq[p, w], f"st_le_sq_{p}_{w}"
            prob += cp[p, w] <= st[p, w], f"cp_le_st_{p}_{w}"
        prob += pulp.lpSum(st[p, w] for p in players) == STARTERS, f"starters_{w}"
        prob += pulp.lpSum(cp[p, w] for p in players) == 1, f"captain_{w}"
        for pos, (lo, hi) in FORMATION_BOUNDS.items():
            expr = pulp.lpSum(st[p, w] for p in players if pool[p].position == pos)
            prob += expr >= lo, f"form_lo_{pos}_{w}"
            prob += expr <= hi, f"form_hi_{pos}_{w}"
        # --- скамейка: полевые по слотам 13–15, вратарь = squad − start ---
        for p in outfield:
            prob += (
                pulp.lpSum(bench[p, s, w] for s in BENCH_SLOTS) == sq[p, w] - st[p, w],
                f"bench_p_{p}_{w}",
            )
        for s in BENCH_SLOTS:
            prob += pulp.lpSum(bench[p, s, w] for p in outfield) == 1, f"bench_slot_{s}_{w}"
        # --- трансферы, непрерывность, бюджет ---
        for p in players:
            expr = prev_sq[p]
            if (p, w) in tin:
                expr = expr + tin[p, w]
            if (p, w) in tout:
                expr = expr - tout[p, w]
            prob += sq[p, w] == expr, f"cont_{p}_{w}"
            if (p, w) in tin and (p, w) in tout:
                prob += tin[p, w] + tout[p, w] <= 1, f"in_xor_out_{p}_{w}"
        used = pulp.lpSum(v for (p, ww), v in tin.items() if ww == w)
        sold = pulp.lpSum(v for (p, ww), v in tout.items() if ww == w)
        cap = spec.max_transfers.get(w, None)
        if transfers_allowed[w] and cap is not None and w != spec.wildcard_gw:
            prob += used <= cap, f"max_transfers_{w}"
        price_in = pulp.lpSum(pool[p].price * v for (p, ww), v in tin.items() if ww == w)
        price_out = pulp.lpSum(pool[p].price * v for (p, ww), v in tout.items() if ww == w)
        prob += itb[w] == prev_itb + price_out - price_in, f"budget_{w}"
        # --- хиты и динамика FT (open-fpl-solver: «<=» точен, т.к. FT в цели только в плюс) ---
        nxt = gws[gws.index(w) + 1] if w != gws[-1] else end
        if w == spec.wildcard_gw:
            prob += paid[w] == 0, f"wc_free_{w}"
            prob += ft[nxt] <= ft[w] + 1, f"ft_next_{w}"
        else:
            prob += paid[w] >= used - ft[w], f"paid_{w}"
            prob += ft[nxt] <= ft[w] - used + paid[w] + 1, f"ft_next_{w}"
        if not spec.allow_hits:
            prob += paid[w] == 0, f"no_hits_{w}"  # трансферы только в пределах FT
        if transfers_allowed[w]:
            prob += sold - used == 0, f"in_eq_out_{w}"
        # --- цель тура: BB — очки (и дисперсия) всех 15 вместо весов скамейки, TC — капитан ×3 ---
        chip = spec.chips.get(w)
        scoring = sq if chip == BENCH_BOOST else st
        cap_extra = 2 if chip == TRIPLE_CAPTAIN else 1
        pts = pulp.lpSum(pool[p].xpts(w) * (scoring[p, w] + cap_extra * cp[p, w]) for p in players)
        bench_pts: Any = 0
        if chip != BENCH_BOOST:
            bench_pts = pulp.lpSum(
                S.bench_weight(s) * pool[p].xpts(w) * bench[p, s, w]
                for p in outfield
                for s in BENCH_SLOTS
            ) + pulp.lpSum(
                S.bench_weight(12) * pool[p].xpts(w) * (sq[p, w] - st[p, w]) for p in keepers
            )
        risk = pulp.lpSum(pool[p].variance(w) * scoring[p, w] for p in players)
        own = pulp.lpSum(pool[p].ownership * sq[p, w] for p in players)
        # Как в open-fpl-solver: ценность FT начисляется в туре, когда она появляется
        # (ft[next] − ft[w]), деньги в банке — за каждый тур (itb_value за £1m), всё с дисконтом.
        obj += d * (
            pts
            + bench_pts
            - S.variance_penalty * risk
            + S.ownership_weight * own
            - HIT_COST * paid[w]
            + S.itb_value * itb[w]
            + S.ft_value * (ft[nxt] - ft[w])
        )
        prev_sq = {p: sq[p, w] for p in players}
        prev_itb = itb[w]
    prob += obj
    return prob, _Vars(sq, st, cp, bench, tin, tout, paid, ft, itb)


def _val(v: Any) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    x = v.varValue
    return 0.0 if x is None else float(x)


def formation_of(pool: dict[int, Candidate], starters: Iterable[int]) -> str:
    counts = {pos: 0 for pos in POSITION_QUOTA}
    for p in starters:
        counts[pool[p].position] += 1
    return f"{counts[2]}-{counts[3]}-{counts[4]}"


def best_starters(pool: dict[int, Candidate], gw: int, squad: Sequence[int]) -> list[int]:
    """Лучшие 11 состава по xPts: минимумы формации — лучшими по позиции, остальные места —
    лучшими из оставшихся в пределах максимумов (разбиение на матроид, жадность точна). Нужна
    туру с Bench Boost: там очки 15 не зависят от выбора 11, и солвер ставит старт произвольно."""
    order = sorted(squad, key=lambda p: (-pool[p].xpts(gw), p))
    starters = []
    counts = dict.fromkeys(FORMATION_BOUNDS, 0)
    for pos, (lo, _) in FORMATION_BOUNDS.items():
        picked = [p for p in order if pool[p].position == pos][:lo]
        starters += picked
        counts[pos] = len(picked)
    for p in order:
        pos = pool[p].position
        if len(starters) == STARTERS:
            break
        if p not in starters and counts[pos] < FORMATION_BOUNDS[pos][1]:
            starters.append(p)
            counts[pos] += 1
    return starters


def lineup_from_sets(
    pool: dict[int, Candidate],
    gw: int,
    squad: Sequence[int],
    starters: Sequence[int],
    captain: int,
    strategy: Strategy,
    chip: str | None = None,
) -> LineupResult:
    """Собирает LineupResult: вице — лучший стартер после капитана, скамейка — вратарь, затем
    полевые по убыванию xPts. objective — вклад тура в цель модели (без paid/ft/itb).
    chip: BB — скамейка входит в очки целиком (вместо весов), TC — капитан считается трижды."""
    xp = {p: pool[p].xpts(gw) for p in squad}
    starters = sorted(starters, key=lambda p: (pool[p].position, -xp[p], p))
    vice_pool = [p for p in starters if p != captain]
    vice = max(vice_pool, key=lambda p: (xp[p], -p)) if vice_pool else captain
    bench_gk = [p for p in squad if p not in starters and pool[p].position == GK]
    bench_out = sorted(
        (p for p in squad if p not in starters and pool[p].position != GK),
        key=lambda p: (-xp[p], p),
    )
    bench = bench_gk + bench_out
    chip_points = 0.0
    if chip == BENCH_BOOST:
        chip_points = sum(xp[p] for p in bench)
    elif chip == TRIPLE_CAPTAIN:
        chip_points = xp[captain]
    expected = sum(xp[p] for p in starters) + xp[captain] + chip_points
    objective = expected
    if chip == BENCH_BOOST:
        objective -= strategy.variance_penalty * sum(pool[p].variance(gw) for p in squad)
    else:
        objective += sum(strategy.bench_weight(12 + i) * xp[p] for i, p in enumerate(bench))
        objective -= strategy.variance_penalty * sum(pool[p].variance(gw) for p in starters)
    objective += strategy.ownership_weight * sum(pool[p].ownership for p in squad)
    return LineupResult(
        gw=gw,
        formation=formation_of(pool, starters),
        starters=list(starters),
        bench_order=bench,
        captain=captain,
        vice=vice,
        expected_points=round(expected, 4),
        objective=round(objective, 4),
        chip=chip,
        chip_points=round(chip_points, 4),
    )


def extract_solution(
    spec: ModelSpec, v: _Vars, prob: pulp.LpProblem, runtime: float, limit_hit: bool
) -> Solution:
    pool = spec.pool
    players = sorted(pool)
    squads: dict[int, list[int]] = {}
    lineups: dict[int, LineupResult] = {}
    transfers: dict[int, tuple[list[int], list[int]]] = {}
    paid: dict[int, int] = {}
    ft: dict[int, int] = {}
    itb: dict[int, float] = {}
    bank = float(spec.bank)
    ft_now = min(spec.free_transfers, MAX_FREE_TRANSFERS)
    for w in spec.gws:
        squad = [p for p in players if _val(v.sq[p, w]) > 0.5]
        starters = [p for p in squad if _val(v.st[p, w]) > 0.5]
        captain = max(starters, key=lambda p: (_val(v.cp[p, w]), pool[p].xpts(w)))
        if spec.chips.get(w) == BENCH_BOOST:
            # та же цель (Σ 15 + лучший капитан), но старт и скамейка — как их поставит менеджер
            starters = best_starters(pool, w, squad)
            captain = max(starters, key=lambda p: (pool[p].xpts(w), _val(v.cp[p, w]), -p))
        squads[w] = squad
        lineups[w] = lineup_from_sets(
            pool, w, squad, starters, captain, spec.strategy, spec.chips.get(w)
        )
        out = sorted(p for (p, ww), var in v.tout.items() if ww == w and _val(var) > 0.5)
        inn = sorted(p for (p, ww), var in v.tin.items() if ww == w and _val(var) > 0.5)
        transfers[w] = (out, inn)
        bank += sum(pool[p].price for p in out) - sum(pool[p].price for p in inn)
        itb[w] = round(bank, 2)
        ft[w] = ft_now
        # FT/хиты пересчитываем детерминированно из трансферов (см. build_model про «<=»)
        if w == spec.wildcard_gw:
            paid[w] = 0
            ft_now = min(MAX_FREE_TRANSFERS, ft_now + 1)
        else:
            paid[w] = max(0, len(inn) - ft_now)
            ft_now = min(MAX_FREE_TRANSFERS, ft_now - len(inn) + paid[w] + 1)
    ft[spec.gws[-1] + 1] = ft_now
    return Solution(
        status=pulp.LpStatus[prob.status],
        objective=round(float(pulp.value(prob.objective) or 0.0), 4),
        squads=squads,
        lineups=lineups,
        transfers=transfers,
        paid=paid,
        ft=ft,
        itb=itb,
        runtime_s=round(runtime, 3),
        time_limit_hit=limit_hit,
    )


# ---------- публичные функции ----------


def best_xi(
    squad_candidates: Sequence[Candidate],
    gw: int,
    strategy: str | Strategy = "balanced",
    *,
    time_limit: float | None = None,
) -> LineupResult:
    """Лучшие 11 + капитан/вице + порядок скамейки для состава из 15 на тур gw."""
    S = get_strategy(strategy)
    pool = {c.player_id: c for c in squad_candidates}
    spec = ModelSpec(
        gws=[gw],
        pool=pool,
        initial_squad=list(pool),
        bank=0.0,
        free_transfers=1,
        strategy=S,
        max_transfers={gw: 0},
    )
    sol, _ = Model(spec).solve(time_limit=time_limit)
    return sol.lineups[gw]


def _baseline(spec: ModelSpec, *, time_limit: float | None) -> Solution:
    base = ModelSpec(
        gws=spec.gws,
        pool=spec.pool,
        initial_squad=spec.initial_squad,
        bank=spec.bank,
        free_transfers=spec.free_transfers,
        strategy=spec.strategy,
        max_transfers={w: 0 for w in spec.gws},
        chips=spec.chips,
    )
    sol, _ = Model(base).solve(time_limit=time_limit)
    return sol


def _risk_note(
    pool: dict[int, Candidate],
    gw: int,
    out: Sequence[int],
    inn: Sequence[int],
    issues: Sequence[SquadIssue] | None,
) -> str:
    notes: list[str] = []
    by_player: dict[int, SquadIssue] = {}
    for i in issues or []:
        if i.player_id not in by_player:
            by_player[i.player_id] = i
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


def single_transfer(
    squad: Sequence[int],
    bank: float,
    free_transfers: int,
    pool: dict[int, Candidate],
    gw: int,
    horizon: int = 3,
    strategy: str | Strategy = "balanced",
    *,
    exclude: Iterable[int] = (),
    keep: Iterable[int] = (),
    allow_hit: bool | None = None,
    top: int = 3,
    issues: Sequence[SquadIssue] | None = None,
    time_limit: float | None = None,
) -> list[TransferRoute]:
    """Top-`top` маршрутов трансфера в туре gw: одна MILP с t_in/t_out, альтернативы —
    no-good cuts. Лимит трансферов: 1 (2 при FT >= 2), +1 платный при allow_hit.

    Вердикт по хиту считается от лучшей бесплатной альтернативы (не от «без трансферов»):
    hit_marginal_gain = Δdisc(маршрут) − Δdisc(лучший бесплатный) − 4 >= hit_threshold."""
    S = get_strategy(strategy)
    gws = list(range(gw, gw + horizon))
    free_cap = 1 if free_transfers < 2 else 2
    max_t = free_cap + (1 if allow_hit else 0)

    def spec_for(cap: int) -> ModelSpec:
        return ModelSpec(
            gws=gws,
            pool=pool,
            initial_squad=list(squad),
            bank=bank,
            free_transfers=free_transfers,
            strategy=S,
            max_transfers={w: (cap if w == gw else 0) for w in gws},
            keep=frozenset(keep),
            exclude=frozenset(exclude),
        )

    spec = spec_for(max_t)
    baseline = _baseline(spec, time_limit=time_limit)

    def points_d(sol: Solution) -> float:
        return sol.horizon_points(decay=S.decay_base, from_gw=gw)

    # Точка отсчёта для хита: лучшее из «ничего не делать» и лучшего бесплатного маршрута.
    ref_obj, ref_gain_d = baseline.objective, 0.0
    if allow_hit:
        free_model = Model(spec_for(free_cap))
        free_model.require_transfer(gw)
        try:
            free_sol, _ = free_model.solve(time_limit=time_limit)
        except InfeasibleError:
            free_sol = None
        if free_sol is not None and free_sol.objective > ref_obj:
            ref_obj = free_sol.objective
            ref_gain_d = points_d(free_sol) - points_d(baseline)

    model = Model(spec)
    model.require_transfer(gw)
    routes: list[TransferRoute] = []
    for _ in range(top):
        try:
            sol, _ = model.solve(time_limit=time_limit)
        except InfeasibleError:
            break
        out, inn = sol.transfers[gw]
        if not inn:
            break
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
            verdict: Verdict = "go" if worth_hit else "hit_not_worth"
        else:
            verdict = "go" if obj_gain > 0 else "hold"
        routes.append(
            TransferRoute(
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
                risk_note=_risk_note(pool, gw, out, inn, issues),
                worth_hit=worth_hit,
                verdict=verdict,
                lineup_after=sol.lineups[gw],
            )
        )
        model.add_no_good_cut(gw, out, inn)
    return routes


def recommend_route(routes: Sequence[TransferRoute]) -> TransferRoute | None:
    """Первый маршрут с вердиктом go (маршруты идут по убыванию целевой функции модели);
    None — держать трансфер(ы)."""
    return next((r for r in routes if r.verdict == "go"), None)


def _pair_moves(
    pool: dict[int, Candidate],
    gw: int,
    gws: Sequence[int],
    out: Sequence[int],
    inn: Sequence[int],
    paid_count: int,
    issues: Sequence[SquadIssue] | None,
) -> list[PlannedMove]:
    """Пары out->in по позиции (квоты состава фиксированы, значит по позиции #in == #out);
    внутри позиции — по убыванию цены. Платными считаем последние paid_count пар."""
    problem: dict[int, str] = {}
    for i in sorted(issues or [], key=lambda i: -i.severity):
        problem.setdefault(i.player_id, i.kind)
    rest = [g for g in gws if g >= gw]
    moves: list[PlannedMove] = []
    for pos in POSITION_QUOTA:
        outs = sorted((p for p in out if pool[p].position == pos), key=lambda p: -pool[p].price)
        ins = sorted((p for p in inn if pool[p].position == pos), key=lambda p: -pool[p].price)
        for o, i in zip(outs, ins, strict=False):
            moves.append(
                PlannedMove(
                    gw=gw,
                    out=o,
                    in_=i,
                    out_name=pool[o].name,
                    in_name=pool[i].name,
                    position=pos,
                    price_out=pool[o].price,
                    price_in=pool[i].price,
                    delta_xpts_horizon=round(
                        pool[i].horizon_xpts(rest) - pool[o].horizon_xpts(rest), 3
                    ),
                    out_problem=problem.get(o),
                )
            )
    moves.sort(key=lambda m: -m.delta_xpts_horizon)
    if paid_count:
        for m in moves[len(moves) - paid_count :]:
            m.paid = True
    return moves


def validate_chip_plan(
    chips: Mapping[int, str] | Iterable[tuple[int, str]],
    gws: Sequence[int],
    available: Mapping[int, Iterable[str]],
) -> dict[int, str]:
    """Условия «фишка в туре N» -> {тур: фишка}; нарушение правил — ChipPlanError, не пропуск.

    Можно только Bench Boost и Triple Captain (Free Hit не моделируется, Wildcard — отдельная
    альтернатива плана); тур внутри горизонта; одна фишка за тур; фишка есть у менеджера в этом
    туре (available — чипы по туру: окно bootstrap минус сыгранные); одна и та же фишка — не
    больше раза за половину сезона (первый набор до GW19, второй с GW20)."""
    items = list(chips.items()) if isinstance(chips, Mapping) else list(chips)
    plan: dict[int, str] = {}
    for gw, chip in items:
        title = CHIP_TITLES.get(chip, chip)
        if chip == "freehit":
            raise ChipPlanError(
                "Free Hit оптимизатор не моделирует (состав возвращается после тура); "
                "в плане можно Bench Boost и Triple Captain"
            )
        if chip == "wildcard":
            raise ChipPlanError(
                "Wildcard не задаётся условием по туру: план сравнивается с Wildcard в первом "
                "туре горизонта автоматически, если он доступен"
            )
        if chip not in PLANNABLE_CHIPS:
            raise ChipPlanError(
                f"неизвестная фишка {chip!r}; можно bboost (Bench Boost) и 3xc (Triple Captain)"
            )
        if gw not in gws:
            raise ChipPlanError(f"{title} в GW{gw}: тур вне горизонта плана GW{gws[0]}–GW{gws[-1]}")
        if gw in plan:
            raise ChipPlanError(
                f"GW{gw}: две фишки в одном туре ({CHIP_TITLES[plan[gw]]} и {title}) — "
                "по правилам одна фишка за тур"
            )
        if chip not in set(available.get(gw, ())):
            raise ChipPlanError(
                f"{title} недоступен менеджеру в GW{gw}: уже сыгран в этой половине сезона "
                "или окно фишки закрыто"
            )
        half = gw <= FIRST_HALF_LAST_GW
        twin = next(
            (g for g, c in plan.items() if c == chip and (g <= FIRST_HALF_LAST_GW) == half), None
        )
        if twin is not None:
            raise ChipPlanError(
                f"{title} дважды в одной половине сезона (GW{twin} и GW{gw}): "
                f"фишка одна на GW1–{FIRST_HALF_LAST_GW} и одна на GW{FIRST_HALF_LAST_GW + 1}–38"
            )
        plan[gw] = chip
    return dict(sorted(plan.items()))


def plan_transfers(
    squad: Sequence[int],
    bank: float,
    free_transfers: int,
    chips_available: Iterable[str],
    from_gw: int,
    horizon: int = 5,
    strategy: str | Strategy = "balanced",
    *,
    pool: dict[int, Candidate],
    issues: Sequence[SquadIssue] | None = None,
    keep: Iterable[int] = (),
    exclude: Iterable[int] = (),
    max_transfers_per_gw: int = 4,
    manager_id: int = 0,
    time_limit: float | None = None,
    allow_hits: bool = True,
    chips: Mapping[int, str] | Iterable[tuple[int, str]] = (),
    chips_available_by_gw: Mapping[int, Iterable[str]] | None = None,
) -> TransferPlan:
    """Многотуровый план (solve_multi_period в духе open-fpl-solver) + альтернатива с Wildcard
    в from_gw. Рекомендация: hold — ноль трансферов в from_gw; wildcard — WC доступен, план с WC
    лучше на wc_margin очков и проблемных игроков >= 3; иначе transfers.

    allow_hits=False — платные трансферы запрещены в каждом туре горизонта (paid[w] = 0): ходы
    возможны только в пределах накопленных FT (пользователь отказался от хита в HITL).

    chips — условия «фишка в туре» (bboost / 3xc), проверяются validate_chip_plan по
    chips_available_by_gw (None — chips_available в каждом туре горизонта). Фишки входят в план,
    baseline и WC-альтернативу; фишка в from_gw исключает WC-альтернативу (одна фишка за тур)."""
    S = get_strategy(strategy)
    gws = list(range(from_gw, from_gw + horizon))
    started = time.perf_counter()
    chips_now = list(chips_available)
    available = (
        chips_available_by_gw if chips_available_by_gw is not None else {w: chips_now for w in gws}
    )
    chip_plan = validate_chip_plan(chips, gws, available)
    spec = ModelSpec(
        gws=gws,
        pool=pool,
        initial_squad=list(squad),
        bank=bank,
        free_transfers=free_transfers,
        strategy=S,
        max_transfers={w: max_transfers_per_gw for w in gws},
        keep=frozenset(keep),
        exclude=frozenset(exclude),
        allow_hits=allow_hits,
        chips=chip_plan,
    )
    baseline = _baseline(spec, time_limit=time_limit)
    sol, solver_name = Model(spec).solve(time_limit=time_limit)
    limit_hit = sol.time_limit_hit

    notes: list[str] = []
    wc: WildcardAlternative | None = None
    if "wildcard" in set(chips_now) and from_gw in chip_plan:
        notes.append(
            f"Альтернатива Wildcard в GW{from_gw} не строилась: в этом туре запланирован "
            f"{CHIP_TITLES[chip_plan[from_gw]]}, а фишка за тур одна"
        )
    elif "wildcard" in set(chips_now):
        wc_spec = ModelSpec(
            gws=gws,
            pool=pool,
            initial_squad=list(squad),
            bank=bank,
            free_transfers=free_transfers,
            strategy=S,
            max_transfers={w: (None if w == from_gw else max_transfers_per_gw) for w in gws},
            wildcard_gw=from_gw,
            keep=frozenset(keep),
            exclude=frozenset(exclude),
            allow_hits=allow_hits,
            chips=chip_plan,
        )
        wc_sol, _ = Model(wc_spec).solve(time_limit=time_limit)
        limit_hit = limit_hit or wc_sol.time_limit_hit
        out, inn = wc_sol.transfers[from_gw]
        wc = WildcardAlternative(
            gw=from_gw,
            squad=wc_sol.squads[from_gw],
            squad_names=[pool[p].name for p in wc_sol.squads[from_gw]],
            expected_total=round(wc_sol.expected_total(), 3),
            objective=wc_sol.objective,
            moves=_pair_moves(pool, from_gw, gws, out, inn, 0, issues),
            lineups_by_gw=wc_sol.lineups,
        )

    moves_by_gw = {
        w: _pair_moves(pool, w, gws, *sol.transfers[w], sol.paid[w], issues) for w in gws
    }
    n_problem = len(problem_players(issues or []))
    total = sol.expected_total()
    if wc is not None and wc.expected_total - total > S.wc_margin and n_problem >= 3:
        recommendation: Recommendation = "wildcard"
    elif not sol.transfers[from_gw][1]:
        recommendation = "hold"
    else:
        recommendation = "transfers"
    target = sol.squads[gws[-1]]
    return TransferPlan(
        manager_id=manager_id,
        from_gw=from_gw,
        horizon=horizon,
        strategy=S.name,
        moves_by_gw=moves_by_gw,
        hits_by_gw={w: HIT_COST * sol.paid[w] for w in gws},
        ft_by_gw=sol.ft,
        bank_by_gw=sol.itb,
        lineups_by_gw=sol.lineups,
        target_squad=target,
        target_squad_names=[pool[p].name for p in target],
        expected_total=round(total, 3),
        baseline_total=round(baseline.expected_total(), 3),
        baseline_points_by_gw={
            w: round(lu.expected_points, 3) for w, lu in sorted(baseline.lineups.items())
        },
        objective=sol.objective,
        baseline_objective=baseline.objective,
        wildcard_alternative=wc,
        recommendation=recommendation,
        issues=list(issues or []),
        solver=solver_name,
        runtime_s=round(time.perf_counter() - started, 3),
        time_limit_hit=limit_hit,
        allow_hits=allow_hits,
        chips_by_gw=chip_plan,
        notes=notes,
    )


# ---------- сборка входов из реальных данных ----------


@dataclass
class ManagerInputs:
    manager_id: int
    from_gw: int
    gws: list[int]
    squad: list[int]
    bank: float
    free_transfers: int
    chips_available: list[str]
    cands: dict[int, Candidate]
    pool: dict[int, Candidate]
    issues: list[SquadIssue]
    store: PredictionStore
    squad_gw: int  # тур, за который взяты picks (последний завершённый)
    chips_by_gw: dict[int, list[str]] = field(default_factory=dict)  # доступные чипы по туру


def load_inputs(
    manager_id: int,
    from_gw: int,
    horizon: int,
    strategy: str | Strategy = "balanced",
    *,
    client: FPLClient | None = None,
    pool_size: int | None = None,
    exclude: Iterable[int] = (),
    wildcard: bool | None = None,
) -> ManagerInputs:
    """Состав менеджера (picks последнего завершённого тура), прогнозы по горизонту, пул,
    диагностика, доступные чипы (на from_gw и по каждому туру горизонта).
    wildcard=True/False — принудительно считать WC доступным/сыгранным в from_gw."""
    client = client or FPLClient()
    store = PredictionStore(client)
    bs = store.bootstrap
    squad_gw = max(1, from_gw - 1)
    sq = client.squad(manager_id, squad_gw)
    ids, bank, ft = squad_state(sq)
    gws = list(range(from_gw, from_gw + horizon))
    preds = store.horizon(gws)
    cands = build_candidates(bs, preds, ids)
    pool = build_pool(
        cands,
        squad_ids=ids,
        gws=gws,
        size=pool_size or settings.optimizer_pool_size,
        bank=bank,
        exclude=exclude,
    )
    xi = best_xi([cands[p] for p in ids], from_gw, strategy)
    issues = diagnose_squad(ids, preds, cands, from_gw, starters=xi.starters)
    wc = wildcard_available(bs, sq.chips_used, from_gw) if wildcard is None else wildcard
    chips_by_gw = {g: chips_available(bs, sq.chips_used, g) for g in gws}
    chips_now = (["wildcard"] if wc else []) + [c for c in chips_by_gw[from_gw] if c != "wildcard"]
    return ManagerInputs(
        manager_id=manager_id,
        from_gw=from_gw,
        gws=gws,
        squad=ids,
        bank=bank,
        free_transfers=ft,
        chips_available=chips_now,
        cands=cands,
        pool=pool,
        issues=issues,
        store=store,
        squad_gw=squad_gw,
        chips_by_gw=chips_by_gw,
    )


# ---------- вывод ----------


def format_lineup(lu: LineupResult, pool: dict[int, Candidate], *, title: str | None = None) -> str:
    def row(p: int, tag: str) -> str:
        c = pool[p]
        mark = " (C)" if p == lu.captain else " (V)" if p == lu.vice else ""
        return (
            f"  {tag:<3} {c.name[:16]:<16} {c.team:<4} {['', 'GKP', 'DEF', 'MID', 'FWD'][c.position]} "
            f"£{c.price:>4.1f} {c.xpts(lu.gw):>5.2f} ±{math.sqrt(max(0.0, c.variance(lu.gw))):.1f}"
            f"{mark}"
        )

    lines = [title or f"GW{lu.gw} best XI ({lu.formation})"]
    lines += [row(p, "XI") for p in lu.starters]
    lines += [row(p, f"B{i + 1}") for i, p in enumerate(lu.bench_order)]
    lines.append(
        f"  expected XI points: {lu.expected_points:.2f} (objective {lu.objective:.2f}, "
        f"captain {pool[lu.captain].name}, vice {pool[lu.vice].name})"
    )
    return "\n".join(lines)


def format_routes(routes: Sequence[TransferRoute], pool: dict[int, Candidate]) -> str:
    if not routes:
        return "no transfer routes (model infeasible or no eligible candidates)"
    head = (
        f"{'#':>2} {'out':<28} {'in':<28} {'hit':>3} {'Δnext':>6} {'Δhor':>6} {'Δdisc':>6} "
        f"{'Δobj':>6} {'hitΔ':>6} {'bank':>5}  verdict"
    )
    lines = [head]
    for i, r in enumerate(routes, start=1):
        out = ", ".join(r.out_names)[:28]
        inn = ", ".join(r.in_names)[:28]
        hit_m = f"{r.hit_marginal_gain:>+6.2f}" if r.hit_marginal_gain is not None else "     -"
        lines.append(
            f"{i:>2} {out:<28} {inn:<28} {r.hit_cost:>3} {r.expected_gain_next_gw:>+6.2f} "
            f"{r.expected_gain_horizon:>+6.2f} {r.expected_gain_discounted:>+6.2f} "
            f"{r.objective_gain:>+6.2f} {hit_m} {r.new_bank:>5.1f}  {r.verdict}"
        )
        if r.risk_note:
            lines.append(f"     note: {r.risk_note}")
    return "\n".join(lines)


def format_plan(plan: TransferPlan, pool: dict[int, Candidate]) -> str:
    lines = [
        f"Plan GW{plan.from_gw}–GW{plan.from_gw + plan.horizon - 1} strategy={plan.strategy} "
        f"solver={plan.solver} runtime={plan.runtime_s:.1f}s"
        + (" (time limit hit)" if plan.time_limit_hit else "")
    ]
    for gw in sorted(plan.lineups_by_gw):
        lu = plan.lineups_by_gw[gw]
        moves = plan.moves_by_gw.get(gw, [])
        ft = plan.ft_by_gw.get(gw, 0)
        chip = ""
        if lu.chip == BENCH_BOOST:
            bench = ", ".join(pool[p].name for p in lu.bench_order)
            chip = f" [Bench Boost +{lu.chip_points:.2f}: {bench}]"
        elif lu.chip == TRIPLE_CAPTAIN:
            chip = f" [Triple Captain +{lu.chip_points:.2f}: {pool[lu.captain].name}]"
        lines.append(
            f"GW{gw}: FT={ft} hit={plan.hits_by_gw.get(gw, 0)} bank={plan.bank_by_gw.get(gw, 0):.1f} "
            f"XI={lu.expected_points:.2f} ({lu.formation}, C {pool[lu.captain].name}){chip}"
        )
        for m in moves:
            flag = f" [{m.out_problem}]" if m.out_problem else ""
            paid = " (-4)" if m.paid else ""
            lines.append(
                f"    {m.out_name} £{m.price_out:.1f}{flag} -> {m.in_name} £{m.price_in:.1f} "
                f"Δhorizon {m.delta_xpts_horizon:+.2f}{paid}"
            )
        if not moves:
            lines.append("    roll")
    end_ft = plan.ft_by_gw.get(plan.from_gw + plan.horizon)
    lines.append(
        f"expected total {plan.expected_total:.2f} vs baseline (no transfers) "
        f"{plan.baseline_total:.2f}; FT after horizon {end_ft}; "
        f"objective {plan.objective:.2f} vs {plan.baseline_objective:.2f}"
    )
    lines.append("target squad: " + ", ".join(plan.target_squad_names))
    if plan.wildcard_alternative:
        wc = plan.wildcard_alternative
        lines.append(
            f"wildcard alternative (GW{wc.gw}): expected total {wc.expected_total:.2f} "
            f"({wc.expected_total - plan.expected_total:+.2f} vs plan); squad: "
            + ", ".join(wc.squad_names)
        )
    elif plan.from_gw in plan.chips_by_gw:
        lines.append(f"wildcard alternative: not compared (chip in GW{plan.from_gw})")
    else:
        lines.append("wildcard alternative: not available")
    lines += [f"note: {n}" for n in plan.notes]
    if plan.issues:
        lines.append(
            "issues: " + "; ".join(f"{i.name}: {i.kind} ({i.detail})" for i in plan.issues)
        )
    lines.append(f"recommendation: {plan.recommendation}")
    return "\n".join(lines)


def format_issues(issues: Sequence[SquadIssue]) -> str:
    if not issues:
        return "no issues detected"
    return "\n".join(f"  sev {i.severity} {i.name:<16} {i.kind:<12} {i.detail}" for i in issues)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FPL Copilot: MILP-оптимизатор состава")
    ap.add_argument("--manager", type=int, default=settings.fpl_manager_id, required=False)
    ap.add_argument("--gw", type=int, help="тур дедлайна (по умолчанию следующий)")
    ap.add_argument(
        "--strategy", default="balanced", choices=["conservative", "balanced", "aggressive"]
    )
    ap.add_argument("--horizon", type=int, help="туров в горизонте (transfer: 3, plan: 5)")
    ap.add_argument("--xi", action="store_true", help="лучшие 11 без трансферов")
    ap.add_argument("--transfer", action="store_true", help="top-3 маршрутов одного трансфера")
    ap.add_argument("--allow-hit", dest="allow_hit", action="store_true")
    ap.add_argument("--plan", action="store_true", help="многотуровый план + Wildcard")
    ap.add_argument(
        "--no-hits",
        dest="allow_hits",
        action="store_false",
        help="план без платных трансферов (paid = 0 в каждом туре)",
    )
    ap.add_argument(
        "--chip",
        dest="chips",
        action="append",
        default=[],
        metavar="GW:CHIP",
        help="фишка в туре плана: 7:bboost (Bench Boost) или 7:3xc (Triple Captain); можно повторять",
    )
    ap.add_argument("--diagnose", action="store_true", help="проблемы состава")
    ap.add_argument("--save", action="store_true", help="сохранить план в plan_snapshots")
    ap.add_argument(
        "--save-xpts",
        dest="save_xpts",
        action="store_true",
        help="сохранить прогнозы горизонта в xpts_predictions (только туры без строк v0)",
    )
    ap.add_argument("--pool", type=int, help="размер пула кандидатов вне состава")
    ap.add_argument("--exclude", help="id игроков через запятую, которых не покупать")
    ap.add_argument("--keep", help="id игроков через запятую, которых не продавать")
    wc = ap.add_mutually_exclusive_group()
    wc.add_argument("--wc", dest="wildcard", action="store_true", default=None)
    wc.add_argument("--no-wc", dest="wildcard", action="store_false")
    ap.add_argument("--time-limit", dest="time_limit", type=float)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    if args.manager is None:
        ap.error("--manager обязателен (или FPL_MANAGER_ID в .env)")
    if not (args.xi or args.transfer or args.plan or args.diagnose):
        args.xi = True
    chips: list[tuple[int, str]] = []
    for raw in args.chips:
        gw_s, _, name = raw.partition(":")
        if not gw_s.strip().isdigit() or not name.strip():
            ap.error(f"--chip {raw!r}: ожидается GW:CHIP, например 7:bboost")
        chips.append((int(gw_s), name.strip()))

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    client = FPLClient()
    gw = args.gw or client.next_gw() or 1
    horizon = args.horizon or (5 if args.plan else 3)
    exclude = [int(x) for x in args.exclude.split(",")] if args.exclude else []
    keep = [int(x) for x in args.keep.split(",")] if args.keep else []
    t0 = time.perf_counter()
    inputs = load_inputs(
        args.manager,
        gw,
        horizon,
        args.strategy,
        client=client,
        pool_size=args.pool,
        exclude=exclude,
        wildcard=args.wildcard,
    )
    _, solver_name = make_solver()
    print(
        f"manager {inputs.manager_id}: squad from GW{inputs.squad_gw} picks, bank £{inputs.bank:.1f}, "
        f"FT {inputs.free_transfers}, chips {inputs.chips_available or '-'}, pool {len(inputs.pool)} "
        f"players, horizon GW{inputs.gws[0]}–GW{inputs.gws[-1]}, strategy {args.strategy}, "
        f"solver {solver_name}, inputs built in {time.perf_counter() - t0:.1f}s"
    )
    out: dict[str, Any] = {}
    squad_cands = [inputs.cands[p] for p in inputs.squad]

    if args.diagnose or args.plan:
        print("\nsquad issues:")
        print(format_issues(inputs.issues))
        out["issues"] = [i.model_dump() for i in inputs.issues]

    if args.xi:
        t = time.perf_counter()
        lu = best_xi(squad_cands, gw, args.strategy, time_limit=args.time_limit)
        print(f"\n{format_lineup(lu, inputs.cands)}\n  solved in {time.perf_counter() - t:.2f}s")
        out["xi"] = lu.model_dump()

    if args.transfer:
        t = time.perf_counter()
        routes = single_transfer(
            inputs.squad,
            inputs.bank,
            inputs.free_transfers,
            inputs.pool,
            gw,
            horizon,
            args.strategy,
            exclude=exclude,
            keep=keep,
            allow_hit=args.allow_hit,
            issues=inputs.issues,
            time_limit=args.time_limit,
        )
        print(f"\ntransfer routes GW{gw} (horizon {horizon}, allow_hit={bool(args.allow_hit)}):")
        print(format_routes(routes, inputs.pool))
        pick = recommend_route(routes)
        if pick is None:
            print("recommended: hold (roll the free transfer)")
        else:
            idx = routes.index(pick) + 1
            print(
                f"recommended: route {idx} — {', '.join(pick.out_names)} -> "
                f"{', '.join(pick.in_names)}"
            )
            print(format_lineup(pick.lineup_after, inputs.pool, title=f"XI after route {idx}:"))
        print(f"  solved in {time.perf_counter() - t:.2f}s ({solver_name})")
        out["routes"] = [r.model_dump(by_alias=True) for r in routes]
        out["recommended_route"] = None if pick is None else routes.index(pick) + 1

    if args.plan:
        try:
            plan = plan_transfers(
                inputs.squad,
                inputs.bank,
                inputs.free_transfers,
                inputs.chips_available,
                gw,
                horizon,
                args.strategy,
                pool=inputs.pool,
                issues=inputs.issues,
                keep=keep,
                exclude=exclude,
                manager_id=inputs.manager_id,
                time_limit=args.time_limit,
                allow_hits=args.allow_hits,
                chips=chips,
                chips_available_by_gw=inputs.chips_by_gw,
            )
        except ChipPlanError as exc:
            print(f"\nchip plan error: {exc}", file=sys.stderr)
            return 2
        print("\n" + format_plan(plan, inputs.pool))
        out["plan"] = plan.model_dump(by_alias=True, mode="json")
        if args.save:
            from fplcopilot.core.plan import inputs_hash, save_plan

            digest = inputs_hash(
                inputs.squad,
                inputs.bank,
                inputs.free_transfers,
                args.strategy,
                horizon,
                gw,
                inputs.pool,
                chips=plan.chips_by_gw,
            )
            plan_id = save_plan(plan, inputs_hash=digest)
            print(f"saved plan_snapshots id={plan_id} inputs_hash={digest[:12]}")

    if args.save_xpts:
        n = inputs.store.persist(inputs.gws)
        print(f"saved {n} xpts rows (only GWs without existing v0 rows)")

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
