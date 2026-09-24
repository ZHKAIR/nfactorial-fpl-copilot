"""MCP-сервер `fpl-intelligence` (официальный Python SDK `mcp`, класс MCPServer — бывший FastMCP).

Инструменты — тонкие обёртки над `LiveTools` (agent/tools.py): имена игроков -> id через
детерминированный резолвер агента, вход -> pydantic-схема инструмента, выход ->
`model_dump()` с округлением до 2 знаков и обрезкой списков. Ресурсы `fpl://...` отдают JSON,
промпт `pre_deadline_review` — последовательность вызовов (зеркало Skill'а fpl-transfer-analyst).
Docstring каждого инструмента — его описание для LLM-клиента.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from fplcopilot.agent.tools import (
    MAX_GW,
    POSITIONS,
    RANK_LIMIT_MAX,
    RANKING_METRICS,
    BuildPlanInput,
    ComparePlayersInput,
    DiagnoseSquadInput,
    GameweekContextInput,
    KBSearchInput,
    OptimizeTeamInput,
    PlanChipIn,
    PlayerRiskInput,
    PredictPlayerInput,
    RankingMetric,
    RankPlayersInput,
    RecommendTransfersInput,
    RouteOut,
    SimulateScenarioInput,
)
from fplcopilot.config import settings
from fplcopilot.core.ext_stats import load_ext_index, player_advanced_stats, team_defensive_profile
from fplcopilot.core.optimizer import PLANNABLE_CHIPS, ChipPlanError
from fplcopilot.core.plan import latest_plan
from fplcopilot.mcp_server.runtime import (
    ChipPlanRejected,
    Runtime,
    StrategyName,
    ToolInputError,
    check_horizon,
    check_manager_id,
    check_strategy,
    compact,
    dump,
    iso,
    resolve_players,
)
from fplcopilot.rag.kb.registry import KNOWN_TAGS

log = logging.getLogger("fplcopilot.mcp")

SERVER_NAME = "fpl-intelligence"
SERVER_VERSION = "0.1.0"  # = pyproject version
FORCED_SALE_TOLERANCE = 0.5  # xPts за горизонт: меньше — «примерно поровну» (как agent.graph)

INSTRUCTIONS = """FPL Copilot intelligence server: deterministic Fantasy Premier League numbers
(xPts model v0, MILP optimizer, news signals from the RAG index) plus a cited strategy
knowledge base (rules, chips, hits, transfers, captaincy). The client explains; the server
computes — never re-add or re-derive points yourself, quote the numbers as returned.
Typical order: get_gameweek_context -> diagnose_squad -> (predict_player / analyze_player_risk
for named players) -> recommend_transfers | optimize_team | build_gameweek_plan |
simulate_scenario. General "how does FPL work / when to use a chip / is a hit worth it in
general" questions: search_strategy_kb and answer only from the returned chunks, citing them —
never invent rules. "Best value / cheapest consistent / most points / in-form players" with no
player named, or "who was the best in GW4 and GW5 / in the last 3 gameweeks": rank_players
(league-wide, season to date or a past gameweek window, not a forecast).
"Plan with Bench Boost / Triple Captain in GW N": build_gameweek_plan(chips=[{gw, chip}]).
"Why did X score / was it luck", xG-xA-set pieces, "does team T concede little":
get_player_points_breakdown, get_player_advanced_stats, get_team_defensive_profile (by id).
Player names resolve deterministically; an {"error": "ambiguous"} reply
lists candidates — ask the user, do not guess. Squad = picks of the last finished gameweek
(public API). No betting advice, no bookmaker odds.
"""
KB_MAX_K = 12
PositionName = Literal["GKP", "DEF", "MID", "FWD"]
PlannableChip = Literal["bboost", "3xc"]


class PlanChip(BaseModel):
    """Play `chip` in gameweek `gw` of the plan (Bench Boost or Triple Captain)."""

    gw: int = Field(ge=1, le=MAX_GW, description="gameweek inside the plan horizon")
    chip: PlannableChip = Field(description="bboost = Bench Boost, 3xc = Triple Captain")


RUNTIME = Runtime()

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=True)
WRITES_CACHE = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=True)


@asynccontextmanager
async def _lifespan(server: MCPServer) -> AsyncIterator[Runtime]:
    log.info("%s starting (mcp server pid ready; LiveTools is created lazily)", SERVER_NAME)
    try:
        yield RUNTIME
    finally:
        log.info("%s shutting down after %d tool calls", SERVER_NAME, RUNTIME.calls)
        RUNTIME.close()


mcp: MCPServer = MCPServer(
    SERVER_NAME,
    title="FPL Copilot intelligence",
    instructions=INSTRUCTIONS,
    version=SERVER_VERSION,
    lifespan=_lifespan,
)


# ---------- вспомогательное ----------


def _names(field: str, mentions: list[str], resolver: Any) -> tuple[list[int], list[str]]:
    ids, notes, err = resolve_players(resolver, mentions, field=field)
    if err is not None:
        raise _Resolved(err)
    return ids, notes


class _Resolved(Exception):
    """Сигнал «вернуть структурированную ошибку резолвера как результат инструмента»."""

    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(payload.get("error"))
        self.payload = payload


def _run(name: str, args: dict[str, Any], fn: Any) -> dict[str, Any]:
    def body() -> dict[str, Any]:
        try:
            return fn()
        except _Resolved as exc:
            return exc.payload

    out = RUNTIME.call(name, args, body)
    return out


def _with_notes(payload: dict[str, Any], notes: list[str]) -> dict[str, Any]:
    if notes:
        payload["resolution_notes"] = notes
    return payload


def _forced_sale_verdict(routes: list[RouteOut], alternative: RouteOut | None, who: str) -> str:
    """Детерминированный вердикт по принудительной продаже (та же логика, что в agent.graph):
    сравнение выигрыша за горизонт маршрута с продажей и лучшего свободного маршрута."""
    if not routes:
        return f"no feasible route sells {who}"
    if alternative is None:
        return f"sell {who}: the optimizer's unconstrained best route sells them too"
    forced_gain = routes[0].gain_horizon
    alt_gain = alternative.gain_horizon
    diff = round(alt_gain - forced_gain, 2)
    if diff > FORCED_SALE_TOLERANCE:
        return (
            f"keep {who}: the unconstrained route gains more ({alt_gain:+} vs {forced_gain:+} "
            f"xPts over the horizon, diff {diff:+})"
        )
    if diff < -FORCED_SALE_TOLERANCE:
        return f"sell {who}: the forced route gains more ({forced_gain:+} vs {alt_gain:+})"
    return (
        f"roughly a tie ({forced_gain:+} vs {alt_gain:+} over the horizon, diff {diff:+}): "
        f"selling {who} is fine but not required"
    )


# ---------- инструменты ----------


@mcp.tool(annotations=READ_ONLY)
def get_gameweek_context(
    manager_id: int | None = None,
    gw: int | None = None,
    strategy: StrategyName = "balanced",
) -> dict[str, Any]:
    """Start here. Upcoming gameweek number, its deadline (UTC) and the data as-of timestamp;
    with `manager_id` also the manager's squad (15 rows: name, team, position, price in £m,
    FPL status, ownership %, starter/captain flags, next-GW xPts), bank (£m), estimated free
    transfers, available chips and squad issues (injured/doubtful/not_playing/low_xpts/...).
    The squad is the picks of the last finished gameweek (public API limitation, see
    `squad_note`). `gw` is optional and must equal the upcoming gameweek (only the next deadline
    is modelled). Cheap: no LLM, ~0.5–2 s on a warm cache.
    """

    def body() -> dict[str, Any]:
        check_strategy(strategy)
        tools = RUNTIME.tools
        current = tools.next_gw()
        if gw is not None and gw != current:
            raise ToolInputError(
                f"only the upcoming gameweek GW{current} is modelled (deadline context); "
                f"gw={gw} is not available"
            )
        if manager_id is not None:
            check_manager_id(manager_id)
        ctx = tools.get_gameweek_context(
            GameweekContextInput(manager_id=manager_id, strategy=strategy)
        )
        out = dump(ctx) or {}
        out["squad_is_last_finished_gw_picks"] = ctx.squad is not None
        return out

    return _run("get_gameweek_context", {"manager_id": manager_id, "gw": gw}, body)


@mcp.tool(annotations=READ_ONLY)
def predict_player(player: str, horizon: int = 1) -> dict[str, Any]:
    """Expected FPL points (xPts model v0) for one player over the next `horizon` gameweeks
    (1–8, default 1). Returns per-GW xPts, standard deviation, start probability, expected
    minutes, components (appearance, goals, assists, clean sheet, DefCon, bonus, saves...),
    fixtures (opponent, home/away, FSI 1 easy…5 hard, team xG for/against, clean-sheet prob),
    plus `total_xpts` and the news signal used by the minutes model. `player` is a web name
    ("Haaland"), full name ("João Pedro") or numeric FPL id; ambiguous names return
    {"error": "ambiguous", "candidates": [...]} — ask the user. No LLM.
    """

    def body() -> dict[str, Any]:
        check_horizon(horizon)
        tools = RUNTIME.tools
        ids, notes = _names("player", [player], RUNTIME.resolver())
        pred = tools.predict_player(
            PredictPlayerInput(player_id=ids[0], gw=tools.next_gw(), horizon=horizon)
        )
        return _with_notes(dump(pred) or {}, notes)

    return _run("predict_player", {"player": player, "horizon": horizon}, body)


@mcp.tool(annotations=READ_ONLY)
def compare_players(players: list[str], horizon: int = 3) -> dict[str, Any]:
    """Rank 2+ players by total xPts over `horizon` gameweeks (default 3) — for "X or Y?" and
    "who to bring in" questions. Returns each player's per-GW prediction (as predict_player)
    and `ranking` [{id, name, total_xpts}] in descending order. Names resolve deterministically;
    any ambiguous or unknown name aborts with a structured error listing candidates. No LLM.
    """

    def body() -> dict[str, Any]:
        check_horizon(horizon)
        if len(players) < 2:
            raise ToolInputError("compare_players needs at least two player names")
        tools = RUNTIME.tools
        ids, notes = _names("players", players, RUNTIME.resolver())
        if len(ids) < 2:
            raise ToolInputError("the names resolve to the same player; give two different ones")
        out = tools.compare_players(
            ComparePlayersInput(player_ids=ids, gw=tools.next_gw(), horizon=horizon)
        )
        return _with_notes(dump(out) or {}, notes)

    return _run("compare_players", {"players": players, "horizon": horizon}, body)


@mcp.tool(annotations=WRITES_CACHE)
def analyze_player_risk(player: str) -> dict[str, Any]:
    """Fitness / availability / rotation signal for one player from the news index and the
    official FPL status. Returns availability (available|doubtful|injured|suspended|unknown),
    start_probability, expected_minutes, rotation_risk, return_gw, confidence, a one-line
    summary and `evidence` quotes with source, date (dd.mm) and url — cite them as
    [source, dd.mm]. `origin` = cached (signal younger than AGENT_SIGNAL_MAX_AGE_H) | extracted
    (fresh LLM extraction over retrieved news, ~3–5 s, gpt-4o-mini) | unavailable (FPL status
    only). This is the only tool that may call an LLM. Use for "is X fit/injured/starting?".
    """

    def body() -> dict[str, Any]:
        tools = RUNTIME.tools
        ids, notes = _names("player", [player], RUNTIME.resolver())
        risk = tools.analyze_player_risk(
            PlayerRiskInput(
                player_id=ids[0],
                as_of=RUNTIME.now(),
                max_age_h=settings.agent_signal_max_age_h,
            )
        )
        return _with_notes(dump(risk) or {}, notes)

    return _run("analyze_player_risk", {"player": player}, body)


@mcp.tool(annotations=READ_ONLY)
def optimize_team(manager_id: int, strategy: StrategyName = "balanced") -> dict[str, Any]:
    """Best starting XI, bench order, captain and vice-captain for the manager's current squad
    in the upcoming gameweek (MILP over xPts; strategy conservative|balanced|aggressive changes
    the risk penalty). Returns formation, starters and bench (xPts, sd, p_start, ownership,
    fixture), `expected_points` of the XI incl. the doubled captain, top-3 `captain_options`
    with `captain_points` (2·xPts) and an ownership tag (safe >30%, balanced 10–30%,
    differential <10%), plus the current picks' captain and XI points for comparison. Use for
    "who to captain" and "starting XI / bench order". No LLM, <1 s.
    """

    def body() -> dict[str, Any]:
        check_manager_id(manager_id)
        check_strategy(strategy)
        tools = RUNTIME.tools
        out = tools.optimize_team(
            OptimizeTeamInput(manager_id=manager_id, gw=tools.next_gw(), strategy=strategy)
        )
        return dump(out) or {}

    return _run("optimize_team", {"manager_id": manager_id, "strategy": strategy}, body)


@mcp.tool(annotations=READ_ONLY)
def recommend_transfers(
    manager_id: int,
    strategy: StrategyName = "balanced",
    horizon: int = 3,
    allow_hit: bool | None = None,
    exclude: list[str] | None = None,
    keep: list[str] | None = None,
    sell: list[str] | None = None,
    buy: list[str] | None = None,
) -> dict[str, Any]:
    """Top-3 transfer routes for the upcoming gameweek from the MILP optimizer (horizon 1–8 GWs,
    default 3). Each route: players out/in, `hit_cost` (0 or 4 per extra transfer),
    `gain_next_gw`, `gain_horizon` (xPts vs. making no transfer), `hit_marginal_gain`
    (extra gain over the best free route minus 4 — the hit verdict basis), `new_bank`,
    `risk_note` and `verdict` go|hold|hit_not_worth; `recommendation` is "route N" or "hold"
    (roll the free transfer). `allow_hit=true` lets the optimizer consider one paid transfer;
    `sell`/`buy` force a named player out/in (then `alternative` is the best unconstrained
    route and `verdict_on_forced_sale` compares them — use for "should I sell X?");
    `keep` blocks sales, `exclude` blocks buys. If the recommended route has a hit, confirm with
    the user before presenting it as the decision. No LLM, ~1–3 s.
    """

    def body() -> dict[str, Any]:
        check_manager_id(manager_id)
        check_strategy(strategy)
        check_horizon(horizon)
        tools = RUNTIME.tools
        gw = tools.next_gw()
        resolver = RUNTIME.resolver(RUNTIME.squad_ids(manager_id, gw, strategy))
        notes: list[str] = []
        resolved: dict[str, list[int]] = {}
        for field, mentions in (
            ("sell", sell or []),
            ("buy", buy or []),
            ("keep", keep or []),
            ("exclude", exclude or []),
        ):
            ids, field_notes = _names(field, mentions, resolver)
            resolved[field] = ids
            notes += field_notes
        out = tools.recommend_transfers(
            RecommendTransfersInput(
                manager_id=manager_id,
                gw=gw,
                horizon=horizon,
                strategy=strategy,
                allow_hit=allow_hit,
                sell=resolved["sell"],
                buy=resolved["buy"],
                keep=resolved["keep"],
                exclude=resolved["exclude"],
            )
        )
        payload = dump(out) or {}
        if out.constrained and resolved["sell"]:
            who = ", ".join(tools.bootstrap.player(p).web_name for p in resolved["sell"])
            payload["verdict_on_forced_sale"] = _forced_sale_verdict(
                out.routes, out.alternative, who
            )
        return _with_notes(payload, notes)

    return _run(
        "recommend_transfers",
        {
            "manager_id": manager_id,
            "strategy": strategy,
            "horizon": horizon,
            "allow_hit": allow_hit,
            "sell": sell,
            "buy": buy,
            "keep": keep,
            "exclude": exclude,
        },
        body,
    )


def _plannable_chips_by_gw(
    manager_id: int, gw: int, horizon: int, strategy: str
) -> dict[str, list[str]] | None:
    """BB / TC, которые есть у менеджера в каждом туре горизонта (inputs уже в кэше LiveTools)."""
    try:
        inputs = RUNTIME.tools.inputs(manager_id, gw, horizon, strategy)
    except Exception:  # noqa: BLE001 — подсказка к ошибке фишки не должна её подменять
        return None
    return {
        str(g): [c for c in chips if c in PLANNABLE_CHIPS]
        for g, chips in sorted(inputs.chips_by_gw.items())
    }


@mcp.tool(annotations=WRITES_CACHE)
def build_gameweek_plan(
    manager_id: int,
    horizon: int = 5,
    strategy: StrategyName = "balanced",
    allow_hits: bool = True,
    chips: list[PlanChip] | None = None,
) -> dict[str, Any]:
    """Multi-gameweek transfer plan (default 5 GWs) from the MILP optimizer, with free
    transfers banking up to 5 and a Wildcard alternative when the chip is available. Returns
    `moves_by_gw` (out -> in, prices, Δ xPts over the horizon, `out_problem`, `paid` flag),
    `hits_by_gw`, `ft_by_gw`, `bank_by_gw`, `xi_points_by_gw`, `captain_by_gw`,
    `expected_total` vs `baseline_total` (no transfers), `recommendation`
    transfers|wildcard|hold, `wildcard` alternative (expected total, delta, squad) and
    `diff_vs_previous` against the last saved plan. `allow_hits=false` forbids paid transfers
    in every gameweek. `chips` (optional, only when the user names the gameweek): list of
    {gw, chip} with chip bboost (Bench Boost — all 15 score) or 3xc (Triple Captain — captain
    ×3); the transfers are optimised for it, `chips` in the result gives each chip's xPts
    contribution and players, `xi_points_by_gw` already includes it, `notes` explains side
    effects (a chip in the first GW disables the Wildcard alternative). Free Hit is not
    modelled. A chip the manager no longer has, outside the horizon or two in one GW ->
    {"error": "invalid_chip_plan", "chips_available_by_gw": {...}}. Without `chips` the plan is
    chip-free. Only the next gameweek's move is actionable; later moves are directional.
    Saves a plan snapshot to Postgres. No LLM, ~5–15 s (solver).
    """

    def body() -> dict[str, Any]:
        check_manager_id(manager_id)
        check_strategy(strategy)
        check_horizon(horizon, lo=2)
        tools = RUNTIME.tools
        gw = tools.next_gw()
        try:
            out = tools.build_gameweek_plan(
                BuildPlanInput(
                    manager_id=manager_id,
                    gw=gw,
                    horizon=horizon,
                    strategy=strategy,
                    allow_hits=allow_hits,
                    chips=[PlanChipIn(gw=c.gw, chip=c.chip) for c in chips or []],
                )
            )
        except ChipPlanError as exc:
            available = _plannable_chips_by_gw(manager_id, gw, horizon, strategy)
            raise ChipPlanRejected(str(exc), available) from exc
        return dump(out) or {}

    return _run(
        "build_gameweek_plan",
        {
            "manager_id": manager_id,
            "horizon": horizon,
            "strategy": strategy,
            "allow_hits": allow_hits,
            "chips": [c.model_dump() for c in chips or []],
        },
        body,
    )


@mcp.tool(annotations=READ_ONLY)
def simulate_scenario(
    manager_id: int,
    sell: list[str] | None = None,
    buy: list[str] | None = None,
    keep: list[str] | None = None,
    allow_hit: bool | None = None,
    use_wildcard: bool | None = None,
    strategy: StrategyName = "balanced",
    horizon: int = 3,
) -> dict[str, Any]:
    """What-if for the upcoming gameweek: force `sell` and/or `buy` named players (or `keep`
    them), optionally allow a -4 hit, or evaluate playing the Wildcard now
    (`use_wildcard=true`). Returns `feasible` with `reason` when not, the best `primary` route
    satisfying the scenario (hit_cost, gain_next_gw, gain_horizon, hit_marginal_gain, verdict),
    `free_alternative` (same scenario without a hit, if different), `hit_alternative` (the -4
    version when the user asked about a hit but free transfers suffice), other `alternatives`,
    `hold_xi_points`, and for the Wildcard: expected total with/without it, the WC squad and
    moves. A hit or Wildcard in the result needs the user's confirmation before it becomes the
    recommendation. No LLM, ~1–3 s (Wildcard ~10 s).
    """

    def body() -> dict[str, Any]:
        check_manager_id(manager_id)
        check_strategy(strategy)
        check_horizon(horizon)
        tools = RUNTIME.tools
        gw = tools.next_gw()
        resolver = RUNTIME.resolver(RUNTIME.squad_ids(manager_id, gw, strategy))
        notes: list[str] = []
        resolved: dict[str, list[int]] = {}
        for field, mentions in (("sell", sell or []), ("buy", buy or []), ("keep", keep or [])):
            ids, field_notes = _names(field, mentions, resolver)
            resolved[field] = ids
            notes += field_notes
        if not (resolved["sell"] or resolved["buy"] or use_wildcard or allow_hit):
            raise ToolInputError(
                "give a scenario: sell/buy names, allow_hit=true or use_wildcard=true "
                "(for the plain recommendation call recommend_transfers)"
            )
        out = tools.simulate_scenario(
            SimulateScenarioInput(
                manager_id=manager_id,
                gw=gw,
                horizon=horizon,
                strategy=strategy,
                sell=resolved["sell"],
                buy=resolved["buy"],
                keep=resolved["keep"],
                allow_hit=allow_hit,
                use_wildcard=use_wildcard,
            )
        )
        return _with_notes(dump(out) or {}, notes)

    return _run(
        "simulate_scenario",
        {
            "manager_id": manager_id,
            "sell": sell,
            "buy": buy,
            "keep": keep,
            "allow_hit": allow_hit,
            "use_wildcard": use_wildcard,
            "strategy": strategy,
            "horizon": horizon,
        },
        body,
    )


@mcp.tool(annotations=READ_ONLY)
def diagnose_squad(manager_id: int, strategy: StrategyName = "balanced") -> dict[str, Any]:
    """Problems in the manager's squad for the upcoming gameweek: `issues` with kind
    (injured|suspended|unavailable|doubtful|not_playing|fixtures|low_xpts|blank), severity
    (3 will not play, 2 serious, 1 mild) and a detail string, plus `problem_players`
    (severity ≥ 2 — the Wildcard rule input). Call before recommend_transfers to know who the
    optimizer wants out and why. No LLM, ~0.5 s on a warm cache.
    """

    def body() -> dict[str, Any]:
        check_manager_id(manager_id)
        check_strategy(strategy)
        tools = RUNTIME.tools
        out = tools.diagnose_squad(
            DiagnoseSquadInput(manager_id=manager_id, gw=tools.next_gw(), strategy=strategy)
        )
        return dump(out) or {}

    return _run("diagnose_squad", {"manager_id": manager_id, "strategy": strategy}, body)


@mcp.tool(annotations=READ_ONLY)
def search_strategy_kb(query: str, tags: list[str] | None = None, k: int = 6) -> dict[str, Any]:
    """Search the evergreen FPL strategy knowledge base (official premierleague.com rules pages
    tagged `rules`, plus community guides: LiveFPL, FPLWatch, FPL Pilot, FFScout, ...) for
    "how does FPL work" and "how to play" questions: free transfers and banking, points hits
    (-4), chips (Wildcard / Free Hit / Bench Boost / Triple Captain) and their timing, squad
    structure, captaincy and effective ownership, price changes and selling price, DefCon.
    Hybrid retrieval (pgvector + BM25 -> RRF -> cross-encoder rerank, max 2 chunks per document).
    `tags` narrows the search (any of: rules, chips, transfers, hits, captaincy, structure, rank,
    prices, fixtures, defcon, beginner); `k` = number of chunks (1–12, default 6). Returns
    `chunks` [{title, url, source, tags, text, score}] — answer ONLY from these texts, cite each
    claim with the chunk's source/title/url, treat `rules`-tagged chunks as authoritative and the
    rest as advice, and say "not covered" when nothing relevant comes back. Never invent rules
    and never use it for numbers about a specific squad (use the optimizer tools). No LLM, ~1 s.
    """

    def body() -> dict[str, Any]:
        text_q = (query or "").strip()
        if not text_q:
            raise ToolInputError("query must be a non-empty question or topic")
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= KB_MAX_K:
            raise ToolInputError(f"k must be an integer in [1, {KB_MAX_K}], got {k!r}")
        clean_tags = [t.strip().lower() for t in (tags or []) if t and t.strip()]
        unknown = sorted(set(clean_tags) - KNOWN_TAGS)
        if unknown:
            raise ToolInputError(f"unknown tags {unknown}; allowed: {sorted(KNOWN_TAGS)}")
        out = RUNTIME.tools.search_strategy_kb(KBSearchInput(query=text_q, tags=clean_tags, k=k))
        payload = dump(out) or {}
        if not out.chunks:
            payload["note"] = (
                "no chunk matched the query"
                + (f" with tags {clean_tags}" if clean_tags else "")
                + " — say the topic is not covered by the knowledge base instead of guessing"
            )
        return payload

    return _run("search_strategy_kb", {"query": query, "tags": tags, "k": k}, body)


@mcp.tool(annotations=READ_ONLY)
def rank_players(
    metric: RankingMetric = "points_per_million",
    position: PositionName | None = None,
    max_price: float | None = None,
    min_price: float | None = None,
    limit: int = 10,
    min_minutes: int | None = None,
    gw_from: int | None = None,
    gw_to: int | None = None,
    last_n_gws: int | None = None,
) -> dict[str, Any]:
    """League-wide player ranking from season data to date or over a window of past gameweeks —
    for "best value / cheapest but consistent / most points / in-form players" and "who was the
    best in GW4 and GW5 / in the last 3 gameweeks" questions with no specific player named.
    `metric`: points_per_million (points / price — use for cheap, budget, value, and
    "cheap AND consistent"), consistency (share of played gameweeks with >= 5 points, then
    lowest std of points), total_points, form (FPL 30-day form, ignores the window). Window:
    `gw_from` / `gw_to` (inclusive; one of them -> a single gameweek) or `last_n_gws` (counted
    back from the last finished gameweek); without a window the season to date is used. With a
    window, points / minutes / pts per £m / consistency cover ONLY those gameweeks (from local
    match history; `history_unavailable` error if it is missing — season totals are never
    substituted). Filters: `position` GKP|DEF|MID|FWD, `max_price` / `min_price` in £m,
    `min_minutes` = only players with AT LEAST that many minutes played are included (default 45 ×
    gameweeks in the window — drops bench fodder), `limit` 1–25. Returns `window_label`
    ("season to date" | "GW4–GW5"), `rows` [{rank, name, team, position, price, total_points,
    season_total_points (window only), minutes, points_per_game, points_per_million, form,
    gws_played, gws_5plus, share_5plus_pct, gws_2plus, std_points, min/max points,
    points_by_gw}], `candidates` after filters, `history_through_gw` and `notes`. Past points
    are not a forecast — for expected points use predict_player / compare_players. No LLM, ~1 s.
    """

    def body() -> dict[str, Any]:
        if metric not in RANKING_METRICS:
            raise ToolInputError(f"unknown metric {metric!r}; use one of {RANKING_METRICS}")
        if position is not None and position.upper() not in POSITIONS:
            raise ToolInputError(f"unknown position {position!r}; use one of {POSITIONS}")
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= RANK_LIMIT_MAX
        ):
            raise ToolInputError(
                f"limit must be an integer in [1, {RANK_LIMIT_MAX}], got {limit!r}"
            )
        for name, value in (("max_price", max_price), ("min_price", min_price)):
            if value is not None and not 3.0 <= float(value) <= 20.0:
                raise ToolInputError(f"{name} must be a price in £m between 3.0 and 20.0")
        for name, value in (("gw_from", gw_from), ("gw_to", gw_to), ("last_n_gws", last_n_gws)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_GW
            ):
                raise ToolInputError(f"{name} must be an integer in [1, {MAX_GW}], got {value!r}")
        tools = RUNTIME.tools
        out = tools.rank_players(
            RankPlayersInput(
                gw=tools.next_gw(),
                metric=metric,
                position=position.upper() if position else None,
                max_price=max_price,
                min_price=min_price,
                min_minutes=min_minutes,
                limit=limit,
                gw_from=gw_from,
                gw_to=gw_to,
                last_n_gws=last_n_gws,
            )
        )
        return compact(dump(out) or {}, max_list=RANK_LIMIT_MAX)

    return _run(
        "rank_players",
        {
            "metric": metric,
            "position": position,
            "max_price": max_price,
            "min_price": min_price,
            "limit": limit,
            "min_minutes": min_minutes,
            "gw_from": gw_from,
            "gw_to": gw_to,
            "last_n_gws": last_n_gws,
        },
        body,
    )


@mcp.tool(annotations=READ_ONLY)
def get_player_advanced_stats(player_id: int) -> dict[str, Any]:
    """Advanced shooting / set-piece / goalkeeper numbers for one FPL player id: FPL bootstrap
    (penalties_order, direct free-kicks, corners, saves, xG/xA) plus Understat 2026/27
    (xG90, xA90, npxG, shots) when the name matched. Returns sources and a one-line set-piece
    phrase. No bookmaker odds, no model coefficients — raw stats and the small xPts knobs
    (setpiece_bonus, understat_blend) so a host can quote them. Unmatched -> FPL fields only.
    `player_id` = `id` from predict_player / get_gameweek_context; `team_id` in the reply feeds
    get_team_defensive_profile.
    """

    def body() -> dict[str, Any]:
        if not isinstance(player_id, int) or isinstance(player_id, bool) or player_id <= 0:
            raise ToolInputError(f"player_id must be a positive integer, got {player_id!r}")
        bs = RUNTIME.tools.bootstrap
        try:
            player = bs.player(player_id)
        except KeyError:
            raise ToolInputError(f"no player with id {player_id} in FPL bootstrap") from None
        index = load_ext_index(bs)
        return compact(player_advanced_stats(player, bs, index=index) | {"team_id": player.team})

    return _run("get_player_advanced_stats", {"player_id": player_id}, body)


@mcp.tool(annotations=READ_ONLY)
def get_player_points_breakdown(player_id: int, last_n: int = 3) -> dict[str, Any]:
    """Last-N gameweek FPL points by category (appearance, goals, assists, clean sheets,
    saves, bonus, defensive contribution, cards and other minuses) plus a luck / unlucky /
    repeatable label. Category sum equals total_points. Explanation only — not fed into xPts.
    last_n defaults to 3 (max 8). No history in the database -> empty categories and null label.
    """

    def body() -> dict[str, Any]:
        if not isinstance(player_id, int) or isinstance(player_id, bool) or player_id <= 0:
            raise ToolInputError(f"player_id must be a positive integer, got {player_id!r}")
        n = 3 if last_n is None else int(last_n)
        if n < 1 or n > 8:
            raise ToolInputError(f"last_n must be 1..8, got {last_n!r}")
        bs = RUNTIME.tools.bootstrap
        try:
            player = bs.player(player_id)
        except KeyError:
            raise ToolInputError(f"no player with id {player_id} in FPL bootstrap") from None
        gw = next((e.id for e in bs.events if getattr(e, "is_next", False)), 99)
        try:
            from fplcopilot.core.history import load_history

            rows = load_history(before_gw=gw, player_ids=[player_id])
        except Exception:  # noqa: BLE001 — без истории инструмент отвечает пусто
            rows = []
        from fplcopilot.core.points_form import points_breakdown

        window = points_breakdown(player_id, n, history=rows, player=player)
        if window is None:
            return {
                "player_id": player_id,
                "last_n": n,
                "n_rounds": 0,
                "total_points": 0,
                "categories": {},
                "label": None,
                "label_ru": None,
                "sums_to_total": True,
            }
        return compact(window.as_dict())

    return _run(
        "get_player_points_breakdown",
        {"player_id": player_id, "last_n": last_n},
        body,
    )


@mcp.tool(annotations=READ_ONLY)
def get_team_defensive_profile(team_id: int) -> dict[str, Any]:
    """How little a Premier League team concedes: Understat team xG / xGA per game for 2026/27
    and a `concedes_little` flag (xGA below the league median). Also FPL goalkeeper clean-sheet
    totals and penalties won/conceded from Understat xG−npxG (one penalty ≈ 0.76 xG).
    Fouls and opponent set pieces are not included — Understat does not publish them
    stably; FBref is the next source. No bookmaker odds. `team_id` is the FPL team id
    (`team_id` in get_player_advanced_stats).
    """

    def body() -> dict[str, Any]:
        if not isinstance(team_id, int) or isinstance(team_id, bool) or team_id <= 0:
            raise ToolInputError(f"team_id must be a positive integer, got {team_id!r}")
        bs = RUNTIME.tools.bootstrap
        try:
            team = bs.team(team_id)
        except KeyError:
            raise ToolInputError(f"no team with id {team_id} in FPL bootstrap") from None
        index = load_ext_index(bs)
        return compact(team_defensive_profile(team, bs, index=index))

    return _run("get_team_defensive_profile", {"team_id": team_id}, body)


# ---------- ресурсы ----------


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=1, default=str)


def _int_param(name: str, raw: str) -> int:
    if not str(raw).isdigit() or int(raw) <= 0:
        raise ToolInputError(f"{name} must be a positive integer, got {raw!r}")
    return int(raw)


@mcp.resource(
    "fpl://gameweek/current",
    name="gameweek_current",
    description="Upcoming gameweek: number, deadline (UTC), current GW, data as-of (JSON).",
    mime_type="application/json",
)
def gameweek_current() -> str:
    def body() -> dict[str, Any]:
        ctx = RUNTIME.tools.get_gameweek_context(GameweekContextInput())
        return dump(ctx) or {}

    return _json(_run("resource:gameweek/current", {}, body))


@mcp.resource(
    "fpl://manager/{manager_id}/squad",
    name="manager_squad",
    description="Manager's squad (picks of the last finished GW), bank, free transfers, chips, "
    "issues (JSON).",
    mime_type="application/json",
)
def manager_squad(manager_id: str) -> str:
    def body() -> dict[str, Any]:
        mid = _int_param("manager_id", manager_id)
        ctx = RUNTIME.tools.get_gameweek_context(GameweekContextInput(manager_id=mid))
        return dump(ctx) or {}

    return _json(_run("resource:manager/squad", {"manager_id": manager_id}, body))


@mcp.resource(
    "fpl://manager/{manager_id}/plan",
    name="manager_plan",
    description="Latest saved multi-GW transfer plan for the upcoming gameweek "
    "(plan_snapshots) or null (JSON).",
    mime_type="application/json",
)
def manager_plan(manager_id: str) -> str:
    def body() -> dict[str, Any] | None:
        mid = _int_param("manager_id", manager_id)
        tools = RUNTIME.tools
        gw = tools.next_gw()
        plan = latest_plan(mid, gw)
        if plan is None:
            return None
        bs = tools.bootstrap

        def name(pid: int) -> str:
            try:
                return bs.player(pid).web_name
            except KeyError:
                return str(pid)

        data = plan.model_dump(mode="json", by_alias=True)
        lineups = data.pop("lineups_by_gw", {})
        data["xi_points_by_gw"] = {g: lu.get("expected_points") for g, lu in lineups.items()}
        data["captain_by_gw"] = {g: name(int(lu["captain"])) for g, lu in lineups.items()}
        data.pop("target_squad", None)
        if data.get("wildcard_alternative"):
            data["wildcard_alternative"].pop("lineups_by_gw", None)
            data["wildcard_alternative"].pop("squad", None)
        data["created_at"] = iso(plan.created_at)
        return compact(data)

    return _json(_run("resource:manager/plan", {"manager_id": manager_id}, body))


@mcp.resource(
    "fpl://player/{player_id}/signal",
    name="player_signal",
    description="Latest stored news signal (player_signals) for a player id: availability, "
    "start probability, evidence quotes; null if none (JSON). Never triggers extraction.",
    mime_type="application/json",
)
def player_signal(player_id: str) -> str:
    def body() -> dict[str, Any] | None:
        pid = _int_param("player_id", player_id)
        tools = RUNTIME.tools
        try:
            player = tools.bootstrap.player(pid)
        except KeyError:
            raise ToolInputError(f"no player with id {pid} in FPL bootstrap") from None
        # приватные помощники LiveTools того же пакета: чтение последнего сигнала без извлечения
        row = tools._latest_signal(pid, RUNTIME.now())
        if row is None:
            return None
        row = dict(row)
        sig_as_of = row.pop("as_of")
        row["evidence"] = [
            e.model_dump() for e in tools._evidence_items(player, row.get("evidence") or [])
        ]
        return compact(
            {
                "player": dump(tools.player_ref(player)),
                "signal_as_of": iso(sig_as_of),
                "age_h": round((RUNTIME.now() - sig_as_of).total_seconds() / 3600, 1),
                **row,
            }
        )

    return _json(_run("resource:player/signal", {"player_id": player_id}, body))


@mcp.resource(
    "fpl://kb/stats",
    name="kb_stats",
    description="Strategy knowledge base size: documents, chunks (embedded), tokens, documents "
    "per source, docs/chunks per tag, last fetch time (JSON). No LLM.",
    mime_type="application/json",
)
def kb_stats() -> str:
    def body() -> dict[str, Any]:
        return compact(RUNTIME.tools.kb_stats(), max_list=60)

    return _json(_run("resource:kb/stats", {}, body))


# ---------- промпт ----------


@mcp.prompt(
    name="pre_deadline_review",
    description="Pre-deadline review workflow for one manager: the exact tool sequence and "
    "output rules (mirrors the fpl-transfer-analyst Skill).",
)
def pre_deadline_review(manager_id: int, strategy: str = "balanced") -> str:
    """Возвращает пользовательское сообщение с последовательностью вызовов инструментов."""
    mid = int(manager_id)
    strat = check_strategy(str(strategy))
    return f"""Run a pre-deadline review for FPL manager {mid} (strategy: {strat}) using the
fpl-intelligence tools, in this order, and report only numbers the tools returned:

1. get_gameweek_context(manager_id={mid}, strategy="{strat}") — deadline, as-of time, squad,
   bank, free transfers, chips, issues. State the as-of time and that the squad is the picks of
   the last finished gameweek.
2. diagnose_squad(manager_id={mid}) — who has problems and why (kind, severity).
3. For every player with an issue of severity ≥ 2 (max 4): analyze_player_risk(player) —
   cite evidence as [source, dd.mm]; predict_player(player, horizon=3) for the numbers.
4. recommend_transfers(manager_id={mid}, strategy="{strat}", horizon=3) — routes, hit
   verdicts, recommendation. If the recommended route has hit_cost > 0, ASK the user to confirm
   the -4 before presenting it as the decision; offer the best free route as the alternative.
5. optimize_team(manager_id={mid}, strategy="{strat}") — starting XI, bench order, captain
   options with ownership tags.
6. Optional: build_gameweek_plan(manager_id={mid}, horizon=5) — direction for the next
   gameweeks (only the next move is actionable); add chips=[{{"gw": N, "chip": "bboost"|"3xc"}}]
   only if the user named a Bench Boost / Triple Captain gameweek; read
   fpl://manager/{mid}/plan for the previous saved plan and mention what changed.
7. If the recommendation involves a points hit or a chip: search_strategy_kb(query, tags=["hits"]
   or ["chips"], k=2) and cite ONE returned excerpt as the rule behind the decision — the
   numbers still come from the optimizer.

Answer format: one verdict line; an options table (option, xPts next GW, Δ vs hold, hit cost,
risk note); "Why" bullets with citations; "Sources"; "Caveats" (data as-of, squad = last
finished GW picks, model v0). Never compute sums or deltas yourself, never give betting advice
or mention bookmaker odds. If any tool returns {{"error": "ambiguous"}}, ask which player is meant."""
