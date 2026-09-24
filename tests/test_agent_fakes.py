"""Фейки для unit-тестов агента: синтетический bootstrap, инструменты и LLM без сети/БД.

Не содержит тестов; импортируется из tests/test_agent_*.py (pytest добавляет tests/ в sys.path).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel

from fplcopilot.agent.llm import (
    ExplainOutput,
    ExplainRequest,
    GradeOutput,
    GradeRequest,
    LLMUsage,
    RouterOutput,
    RouterRequest,
    ScenarioDraft,
)
from fplcopilot.agent.tools import (
    CaptainOption,
    ComparePlayersOutput,
    EvidenceItem,
    FixtureBrief,
    GameweekContext,
    GWPrediction,
    HistoryUnavailable,
    IssuesOut,
    KBChunkOut,
    KBCitation,
    KBSearchOutput,
    LineupOut,
    LineupPlayer,
    PlanMoveOut,
    PlanOut,
    PlayerPrediction,
    PlayerRef,
    PlayerRisk,
    PlayerRiskInput,
    RankedPlayerRow,
    RankPlayersInput,
    RankPlayersOutput,
    RouteOut,
    RoutesOut,
    ScenarioOut,
    SquadPlayerRow,
    StrategyAnswerOutput,
    consistency_stats,
    rank_player_rows,
    resolve_window,
    window_label,
)
from fplcopilot.data.schemas import Bootstrap, Event, Player, Team

NOW = datetime(2026, 9, 17, 10, 0, tzinfo=UTC)
DEADLINE = datetime(2026, 9, 18, 17, 30, tzinfo=UTC)
SQUAD_IDS = [154, 165, 4, 411, 388]  # Palmer, João Pedro, Gabriel, Haaland, Guéhi


def make_bootstrap() -> Bootstrap:
    events = [
        Event(
            id=4,
            name="Gameweek 4",
            deadline_time=DEADLINE - timedelta(days=7),
            finished=True,
            is_current=True,
        ),
        Event(id=5, name="Gameweek 5", deadline_time=DEADLINE, is_next=True),
        Event(id=6, name="Gameweek 6", deadline_time=DEADLINE + timedelta(days=7)),
        Event(id=7, name="Gameweek 7", deadline_time=DEADLINE + timedelta(days=14)),
    ]
    teams = [
        Team(id=1, name="Arsenal", short_name="ARS"),
        Team(id=2, name="Chelsea", short_name="CHE"),
        Team(id=3, name="Man City", short_name="MCI"),
        Team(id=4, name="Ipswich", short_name="IPS"),
        Team(id=5, name="Leeds", short_name="LEE"),
        Team(id=6, name="Spurs", short_name="TOT"),
    ]

    def pl(pid, web, first, second, team, pos, cost, own, status="a", chance=None):
        return Player(
            id=pid,
            web_name=web,
            first_name=first,
            second_name=second,
            team=team,
            element_type=pos,
            now_cost=cost,
            status=status,
            chance_of_playing_next_round=chance,
            selected_by_percent=own,
        )

    players = [
        pl(154, "Palmer", "Cole", "Palmer", 2, 3, 97, 26.8),
        pl(301, "Palmer", "Alex", "Palmer", 4, 1, 45, 4.0),
        pl(12, "Saka", "Bukayo", "Saka", 1, 3, 95, 12.5),
        pl(611, "Wan-Bissaka", "Aaron", "Wan-Bissaka", 6, 2, 45, 0.2),
        pl(388, "Guéhi", "Marc", "Guéhi", 3, 2, 60, 18.5),
        pl(165, "João Pedro", "João Pedro", "Junqueira de Jesus", 2, 4, 78, 74.5, "d", 75),
        pl(156, "Neto", "Pedro", "Lomba Neto", 2, 3, 70, 1.4),
        pl(499, "Pedro Porro", "Pedro", "Porro Sauceda", 6, 2, 55, 8.8),
        pl(4, "Gabriel", "Gabriel", "dos Santos Magalhães", 1, 2, 80, 23.3),
        pl(18, "Martinelli", "Gabriel", "Martinelli Silva", 1, 3, 65, 0.1, "u"),
        pl(27, "G.Jesus", "Gabriel Fernando", "de Jesus", 1, 4, 68, 0.2, "u"),
        pl(331, "Gudmundsson", "Gabriel", "Gudmundsson", 5, 2, 45, 0.1),
        pl(411, "Haaland", "Erling", "Haaland", 3, 4, 155, 72.6),
    ]
    return Bootstrap(events=events, teams=teams, elements=players)


BS = make_bootstrap()


def ref(pid: int) -> PlayerRef:
    p = BS.player(pid)
    return PlayerRef(
        id=p.id,
        name=p.web_name,
        full_name=p.full_name,
        team=BS.team(p.team).short_name,
        position=p.position.short,
        price=p.price,
        status=p.status,
        chance=p.chance_of_playing_next_round,
        ownership=float(p.selected_by_percent or 0.0),
    )


XPTS = {154: 4.76, 165: 3.3, 4: 4.58, 411: 7.13, 388: 5.94, 12: 6.2, 301: 3.0}


def usage(model: str = "fake-model") -> LLMUsage:
    return LLMUsage(
        model=model, prompt_tokens=100, completion_tokens=20, latency_ms=1, cost_usd=0.0
    )


def router_output(
    intent: str,
    mentions: list[str] | None = None,
    *,
    horizon: int | None = None,
    sell: list[str] | None = None,
    buy: list[str] | None = None,
    keep: list[str] | None = None,
    allow_hit: bool | None = None,
    use_wildcard: bool | None = None,
    needs_squad: bool | None = None,
) -> RouterOutput:
    squad_intents = {"transfer", "captain", "lineup", "plan", "what_if"}
    return RouterOutput(
        intent=intent,  # type: ignore[arg-type]
        player_mentions=mentions or [],
        horizon=horizon,
        scenario=ScenarioDraft(
            sell=sell or [],
            buy=buy or [],
            keep=keep or [],
            allow_hit=allow_hit,
            use_wildcard=use_wildcard,
        ),
        needs_squad=intent in squad_intents if needs_squad is None else needs_squad,
        reason="fake",
    )


class FakeRouter:
    def __init__(self, output: RouterOutput) -> None:
        self.output = output
        self.calls: list[RouterRequest] = []

    def __call__(self, req: RouterRequest) -> tuple[RouterOutput, LLMUsage]:
        self.calls.append(req)
        return self.output, usage()


class FakeExplain:
    """Отдаёт заранее заданные ответы по очереди (последний — повторяется)."""

    def __init__(self, answers: list[str] | None = None) -> None:
        self.answers = answers or ["**Verdict:** ok.\n\n**Why**\n- computed."]
        self.calls: list[ExplainRequest] = []

    def __call__(self, req: ExplainRequest) -> tuple[ExplainOutput, LLMUsage]:
        self.calls.append(req)
        idx = min(len(self.calls) - 1, len(self.answers) - 1)
        return ExplainOutput(answer_markdown=self.answers[idx]), usage()


class FakeGrader:
    def __init__(self, sufficient: bool = False) -> None:
        self.sufficient = sufficient
        self.calls: list[GradeRequest] = []

    def __call__(self, req: GradeRequest) -> tuple[GradeOutput, LLMUsage]:
        self.calls.append(req)
        return GradeOutput(sufficient=self.sufficient, reason="fake"), usage()


def fit_risk(inp: PlayerRiskInput) -> PlayerRisk:
    p = BS.player(inp.player_id)
    return PlayerRisk(
        player_id=p.id,
        player=p.web_name,
        fpl_status=p.status,
        fpl_chance=p.chance_of_playing_next_round,
        availability="fit",
        start_probability=0.9,
        expected_minutes=80,
        confidence=0.8,
        summary=f"{p.web_name} trained fully.",
        evidence=[
            EvidenceItem(
                player_id=p.id,
                player=p.web_name,
                source="bbc",
                url="https://bbc.example/x",
                published_at="2026-09-16T10:00:00+00:00",
                date="16.09",
                quote=f"{p.web_name} trained fully on Tuesday.",
            )
        ],
        signal_as_of=NOW.isoformat(),
        age_h=1.0,
        origin="cached",
        mode=inp.mode,
        k=inp.k,
    )


def unknown_risk(inp: PlayerRiskInput) -> PlayerRisk:
    p = BS.player(inp.player_id)
    return PlayerRisk(
        player_id=p.id,
        player=p.web_name,
        fpl_status=p.status,
        fpl_chance=p.chance_of_playing_next_round,
        availability="unknown",
        confidence=0.0,
        summary="No player-specific news found.",
        origin="extracted",
        abstained=True,
        mode=inp.mode,
        k=inp.k,
        llm_calls=0,
    )


def low_conf_risk(inp: PlayerRiskInput) -> PlayerRisk:
    r = fit_risk(inp)
    return r.model_copy(
        update={"availability": "doubtful", "confidence": 0.3, "origin": "extracted"}
    )


def route(
    rank: int, out: list[str], inn: list[str], *, hit: int = 0, verdict: str = "go"
) -> RouteOut:
    return RouteOut(
        rank=rank,
        out=out,
        in_=inn,
        out_ids=[BS.elements[0].id],
        in_ids=[BS.elements[1].id],
        hit_cost=hit,
        gain_next_gw=1.5,
        gain_horizon=4.03,
        gain_discounted=3.5,
        objective_gain=2.1,
        hit_marginal_gain=1.2 if hit else None,
        new_bank=0.3,
        risk_note="",
        verdict=verdict,
        xi_points_after=57.6,
    )


# ---------- стратегическая KB (rag/kb) — канонические чанки и цитируемый ответ ----------

KB_HIT_CHUNKS = [
    KBChunkOut(
        chunk_id=901,
        doc_id=41,
        title="FPL Rules Copilot",
        url="https://www.premierleague.com/en/news/4661029",
        source="premierleague",
        tags=["rules", "transfers", "hits"],
        text="FPL Rules Copilot\nTransfers\nEach additional transfer beyond your free transfers "
        "will deduct 4 points from your total score.",
        score=0.99,
    ),
    KBChunkOut(
        chunk_id=902,
        doc_id=17,
        title="Points Hits Guide",
        url="https://fplwatch.example/hits",
        source="fplwatch",
        tags=["hits", "transfers"],
        text="Points Hits Guide\nWhen a -4 is worth it\nA hit pays off only when the incoming "
        "player is expected to outscore the outgoing one by more than four points over the "
        "next few gameweeks.",
        score=0.97,
    ),
]
KB_CHIP_CHUNKS = [
    KBChunkOut(
        chunk_id=903,
        doc_id=12,
        title="FPL Chips Strategy Guide",
        url="https://fplwatch.example/chips",
        source="fplwatch",
        tags=["chips"],
        text="FPL Chips Strategy Guide\nWhen to use your Wildcard\nPlay the Wildcard when three "
        "or more of your players are injured or have terrible fixtures.",
        score=0.98,
    ),
]
KB_ANSWER = StrategyAnswerOutput(
    query="When should I play my wildcard?",
    answer="Play the Wildcard when several players are injured or fixtures turn [1]. The first "
    "Wildcard expires at the GW19 deadline [2].",
    covered=True,
    citations=[
        KBCitation(
            n=1,
            chunk_id=903,
            title="FPL Chips Strategy Guide",
            url="https://fplwatch.example/chips",
            source="fplwatch",
            tags=["chips"],
            quote="Play the Wildcard when three or more of your players are injured",
        ),
        KBCitation(
            n=2,
            chunk_id=904,
            title="FPL Rules Copilot",
            url="https://www.premierleague.com/en/news/4661029",
            source="premierleague",
            tags=["rules", "chips"],
            quote="The first Wildcard will be available until the Gameweek 19 deadline",
        ),
    ],
    retrieved=6,
    model="fake-kb-model",
    prompt_version="v1",
    llm_calls=1,
    prompt_tokens=900,
    completion_tokens=80,
    cost_usd=0.0002,
    latency_ms=5,
)


# ---------- рейтинг лиги (player_ranking): синтетическая история 4 туров ----------

# player_id -> {round: (очки, минуты)}; Guéhi — стабильный дешёвый защитник, Haaland — дорогой
# и «взрывной», Saka — ровный, но дорогой, Wan-Bissaka — мало минут (отсекается фильтром).
RANK_HISTORY: dict[int, dict[int, tuple[int, int]]] = {
    388: {1: (6, 90), 2: (6, 90), 3: (8, 90), 4: (5, 90)},  # Guéhi £6.0: 25 pts, 4/4 >= 5
    411: {1: (13, 90), 2: (2, 90), 3: (17, 90), 4: (1, 65)},  # Haaland £15.5: 33 pts, 2/4
    12: {1: (5, 90), 2: (7, 88), 3: (2, 90), 4: (9, 90)},  # Saka £9.5: 23 pts, 3/4
    154: {1: (2, 90), 2: (12, 90), 3: (2, 90), 4: (3, 90)},  # Palmer £9.7: 19 pts, 1/4
    611: {1: (0, 0), 2: (1, 12), 3: (0, 0), 4: (0, 0)},  # Wan-Bissaka £4.5: 1 pt, 12 min
}
RANK_ROUNDS = [1, 2, 3, 4]
RANK_LAST_FINISHED = 4  # в фейковом bootstrap GW4 завершён, GW5 — предстоящий


def rank_rows_from_history(
    history: dict[int, dict[int, tuple[int, int]]] = RANK_HISTORY,
    *,
    min_minutes: int = 180,
    position: str | None = None,
    max_price: float | None = None,
    window: tuple[int, int] | None = None,
) -> list[RankedPlayerRow]:
    """Строки рейтинга из BS + RANK_HISTORY тем же способом, что LiveTools.rank_players
    (фильтры минут / позиции / цены, статистика стабильности, окно туров) — без БД."""
    rows: list[RankedPlayerRow] = []
    rounds = list(range(window[0], window[1] + 1)) if window else RANK_ROUNDS
    for pid, by_round in history.items():
        p = BS.player(pid)
        season_total = sum(pts for pts, _ in by_round.values())
        if window:
            by_round = {r: v for r, v in by_round.items() if window[0] <= r <= window[1]}
        total = sum(pts for pts, _ in by_round.values())
        minutes = sum(m for _, m in by_round.values())
        if minutes < min_minutes:
            continue
        if position and p.position.short != position:
            continue
        if max_price is not None and p.price > max_price:
            continue
        rows.append(
            RankedPlayerRow(
                id=pid,
                name=p.web_name,
                full_name=p.full_name,
                team=BS.team(p.team).short_name,
                position=p.position.short,
                price=p.price,
                status=p.status,
                ownership=float(p.selected_by_percent or 0.0),
                total_points=total,
                minutes=minutes,
                points_per_game=round(total / max(1, sum(1 for _, m in by_round.values() if m)), 1),
                points_per_million=round(total / p.price, 2),
                form=round(season_total / 4, 1),
                season_total_points=season_total if window else None,
                **consistency_stats(by_round, rounds),
            )
        )
    return rows


class FakeTools:
    def __init__(
        self,
        *,
        squad: bool = True,
        risk: Callable[[PlayerRiskInput], PlayerRisk] = fit_risk,
        routes: list[RouteOut] | None = None,
        recommended_rank: int | None = 1,
        scenario: ScenarioOut | None = None,
        plan: PlanOut | None = None,
        issues: list[dict[str, Any]] | None = None,
        kb_answer: StrategyAnswerOutput | None = None,
        kb_chunks: dict[str, list[KBChunkOut]] | None = None,
        history: bool = True,  # False — player_gw_history недоступна (окно туров -> ошибка)
    ) -> None:
        self.squad = squad
        self.risk = risk
        self.history = history
        self.routes = routes if routes is not None else [route(1, ["Palmer"], ["Saka"])]
        self.recommended_rank = recommended_rank
        self.scenario = scenario
        self.plan = plan
        self.kb_answer = kb_answer if kb_answer is not None else KB_ANSWER
        # тег -> чанки; поиск без тега отдаёт всё
        self.kb_chunks = (
            kb_chunks if kb_chunks is not None else {"hits": KB_HIT_CHUNKS, "chips": KB_CHIP_CHUNKS}
        )
        self.issues = (
            issues
            if issues is not None
            else [
                {
                    "player_id": 165,
                    "name": "João Pedro",
                    "kind": "doubtful",
                    "severity": 1,
                    "detail": "FPL doubtful 75%",
                    "gw": 5,
                }
            ]
        )
        self.calls: list[tuple[str, BaseModel]] = []
        self.invalidated = 0

    def _rec(self, name: str, inp: BaseModel) -> None:
        self.calls.append((name, inp))

    def called(self, name: str) -> list[BaseModel]:
        return [inp for n, inp in self.calls if n == name]

    def invalidate(self) -> None:
        self.invalidated += 1

    def get_gameweek_context(self, inp):
        self._rec("get_gameweek_context", inp)
        ctx = GameweekContext(gw=5, current_gw=4, deadline=DEADLINE, as_of=NOW)
        if not self.squad or inp.manager_id is None:
            ctx.squad_note = "manager 1 has no public picks yet: the team starts in GW5"
            return ctx
        ctx.squad_gw = 4
        ctx.bank = 0.1
        ctx.free_transfers = 2
        ctx.squad = [
            SquadPlayerRow(**ref(pid).model_dump(), is_captain=(pid == 411), xpts_next=XPTS[pid])
            for pid in SQUAD_IDS
        ]
        ctx.squad_note = "squad = picks of GW4"
        ctx.issues = list(self.issues)
        return ctx

    def diagnose_squad(self, inp):
        self._rec("diagnose_squad", inp)
        return IssuesOut(gw=5, issues=list(self.issues), problem_players=[])

    def predict_player(self, inp):
        self._rec("predict_player", inp)
        by_gw = [
            GWPrediction(
                gw=g,
                xpts=XPTS.get(inp.player_id, 2.0),
                sd=3.2,
                p_start=0.93,
                exp_minutes=80,
                components={"goals": 1.1, "assists": 0.8, "appearance": 1.9},
                fixtures=[
                    FixtureBrief(
                        opponent="BRE",
                        is_home=False,
                        fsi=3,
                        xg_for=1.5,
                        xg_against=1.44,
                        clean_sheet_prob=0.24,
                    )
                ],
            )
            for g in range(inp.gw, inp.gw + inp.horizon)
        ]
        return PlayerPrediction(
            player=ref(inp.player_id), by_gw=by_gw, total_xpts=round(sum(g.xpts for g in by_gw), 2)
        )

    def compare_players(self, inp):
        self._rec("compare_players", inp)
        ranking = sorted(
            (
                {"id": pid, "name": BS.player(pid).web_name, "total_xpts": XPTS.get(pid, 2.0)}
                for pid in inp.player_ids
            ),
            key=lambda r: -r["total_xpts"],
        )
        return ComparePlayersOutput(players=[], ranking=ranking)

    def analyze_player_risk(self, inp):
        self._rec("analyze_player_risk", inp)
        return self.risk(inp)

    def search_strategy_kb(self, inp):
        self._rec("search_strategy_kb", inp)
        if inp.tags:
            chunks = [c for t in inp.tags for c in self.kb_chunks.get(t, [])]
        else:
            chunks = [c for cs in self.kb_chunks.values() for c in cs]
        return KBSearchOutput(
            query=inp.query, tags=list(inp.tags), mode=inp.mode, chunks=chunks[: inp.k]
        )

    def answer_strategy_question(self, inp):
        self._rec("answer_strategy_question", inp)
        return self.kb_answer.model_copy(update={"query": inp.query})

    def rank_players(self, inp: RankPlayersInput) -> RankPlayersOutput:
        """Как LiveTools.rank_players: окно туров считается только по RANK_HISTORY; без истории
        или вне GW1–GW4 — HistoryUnavailable (никакого отката к сезонным итогам)."""
        self._rec("rank_players", inp)
        window = resolve_window(
            inp.gw_from,
            inp.gw_to,
            inp.last_n_gws,
            upcoming_gw=inp.gw,
            last_finished_gw=RANK_LAST_FINISHED,
        )
        label = window_label(*window) if window else "season to date"
        if window is not None:
            if not self.history:
                raise HistoryUnavailable(
                    f"a gameweek window ({label}) needs player_gw_history, but the database is "
                    "unavailable — season totals are not a substitute for per-gameweek points"
                )
            missing = [g for g in range(window[0], window[1] + 1) if g not in RANK_ROUNDS]
            if missing:
                raise HistoryUnavailable(
                    f"a gameweek window ({label}) needs match history for "
                    f"{', '.join(f'GW{g}' for g in missing)}, but player_gw_history covers GW1–GW4"
                )
        min_minutes = (
            inp.min_minutes
            if inp.min_minutes is not None
            else (45 * (window[1] - window[0] + 1) if window else 180)
        )
        rows = rank_rows_from_history(
            min_minutes=min_minutes,
            position=inp.position,
            max_price=inp.max_price,
            window=window,
        )
        squad = set(inp.squad_ids)
        rows = [r.model_copy(update={"in_squad": r.id in squad}) for r in rows]
        ranked = rank_player_rows(rows, inp.metric)
        return RankPlayersOutput(
            gw=inp.gw,
            metric=inp.metric,
            metric_label=f"fake {inp.metric}",
            window_label=label,
            gw_from=window[0] if window else None,
            gw_to=window[1] if window else None,
            filters={
                "position": inp.position,
                "max_price": inp.max_price,
                "min_minutes": min_minutes,
                "window": label,
            },
            history_through_gw=4,
            candidates=len(rows),
            rows=ranked[: inp.limit],
        )

    def optimize_team(self, inp):
        self._rec("optimize_team", inp)
        starters = [
            LineupPlayer(
                id=pid,
                name=BS.player(pid).web_name,
                team="X",
                position="MID",
                price=9.0,
                xpts=XPTS[pid],
                sd=3.0,
                p_start=0.9,
                ownership=float(BS.player(pid).selected_by_percent or 0),
                fixture="vSUN (FSI 1)",
            )
            for pid in SQUAD_IDS
        ]
        options = [
            CaptainOption(
                id=s.id,
                name=s.name,
                team=s.team,
                xpts=s.xpts,
                captain_points=round(2 * s.xpts, 2),
                sd=s.sd,
                ownership=s.ownership,
                tag="safe" if s.ownership > 30 else "balanced",
                fixture=s.fixture,
                p_start=s.p_start,
            )
            for s in sorted(starters, key=lambda s: -s.xpts)[:3]
        ]
        return LineupOut(
            gw=inp.gw,
            formation="3-4-3",
            starters=starters,
            bench=[],
            captain="Haaland",
            vice="Guéhi",
            expected_points=56.14,
            captain_options=options,
            current_captain="Haaland",
            current_xi_points=56.14,
        )

    def recommend_transfers(self, inp):
        self._rec("recommend_transfers", inp)
        routes = self.routes
        if inp.allow_hit is False:  # после reject хиты запрещены: отдаём бесплатный маршрут
            routes = [
                r.model_copy(update={"hit_cost": 0, "hit_marginal_gain": None}) for r in self.routes
            ]
        rank = self.recommended_rank if routes else None
        return RoutesOut(
            gw=inp.gw,
            horizon=inp.horizon,
            strategy=inp.strategy,
            free_transfers=2,
            bank=0.1,
            routes=routes,
            recommended_rank=rank,
            recommendation=f"route {rank}" if rank else "hold",
            baseline_xi_points=56.14,
            constrained=bool(inp.sell or inp.buy),
        )

    def build_gameweek_plan(self, inp):
        self._rec("build_gameweek_plan", inp)
        if self.plan is not None:
            return self.plan
        return PlanOut(
            from_gw=inp.gw,
            horizon=inp.horizon,
            strategy=inp.strategy,
            moves_by_gw={
                "5": [
                    PlanMoveOut(
                        gw=5,
                        out="Palmer",
                        in_="Saka",
                        price_out=9.7,
                        price_in=9.5,
                        delta_xpts_horizon=3.4,
                    )
                ],
                "6": [],
            },
            hits_by_gw={"5": 0, "6": 0},
            ft_by_gw={"5": 2, "6": 2},
            bank_by_gw={"5": 0.3},
            xi_points_by_gw={"5": 57.6, "6": 55.0},
            captain_by_gw={"5": "Haaland"},
            expected_total=307.5,
            baseline_total=283.1,
            recommendation="transfers",
        )

    def simulate_scenario(self, inp):
        self._rec("simulate_scenario", inp)
        if self.scenario is not None:
            if inp.allow_hit is False and self.scenario.primary is not None:
                free = self.scenario.primary.model_copy(update={"hit_cost": 0})
                return self.scenario.model_copy(update={"primary": free, "free_alternative": None})
            return self.scenario
        return ScenarioOut(
            gw=inp.gw,
            horizon=inp.horizon,
            strategy=inp.strategy,
            feasible=True,
            free_transfers=2,
            bank=0.1,
            transfers_cap=2,
            primary=route(1, ["Palmer"], ["Saka"]),
            hold_xi_points=56.14,
            verdict="go",
        )
