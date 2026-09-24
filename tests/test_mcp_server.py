"""MCP-сервер fpl-intelligence без сети/БД/LLM: реестр инструментов, ресурсов и промпта; форма
ошибки разрешения имён; обрезка payload'ов; контракт ошибок; обёртка над фейковым LiveTools."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from mcp.server.mcpserver.exceptions import ToolError
from sqlalchemy.exc import OperationalError

from fplcopilot.agent.resolve import PlayerResolver
from fplcopilot.agent.tools import (
    RANKING_METRICS,
    TOOL_NAMES,
    FixtureBrief,
    GWPrediction,
    HistoryUnavailable,
    KBChunkOut,
    KBSearchOutput,
    PlanChipOut,
    PlanOut,
    PlayerPrediction,
    PlayerRef,
    RankedPlayerRow,
    RankPlayersOutput,
    RouteOut,
    ScenarioInfeasible,
    SquadUnavailable,
    window_label,
)
from fplcopilot.core.ext_stats import build_index
from fplcopilot.core.optimizer import PLANNABLE_CHIPS, ChipPlanError
from fplcopilot.core.understat import UnderstatPlayer, UnderstatSnapshot, UnderstatTeam
from fplcopilot.data.schemas import Bootstrap, PlayerGWHistory
from fplcopilot.mcp_server import runtime as rt
from fplcopilot.mcp_server import server as srv
from fplcopilot.mcp_server.server import RUNTIME, SERVER_NAME, _forced_sale_verdict, mcp

# Полный список инструментов сервера: 11 инструментов агента (TOOL_NAMES) + 3 справочных.
MCP_TOOLS: tuple[str, ...] = (
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
    "get_player_advanced_stats",
    "get_player_points_breakdown",
    "get_team_defensive_profile",
)

# ---------- фейковый bootstrap ----------


def _player(pid: int, first: str, second: str, web: str, team: int, pos: int, own: float) -> dict:
    return {
        "id": pid,
        "web_name": web,
        "first_name": first,
        "second_name": second,
        "team": team,
        "element_type": pos,
        "now_cost": 60,
        "status": "a",
        "selected_by_percent": str(own),
    }


def fake_bootstrap() -> Bootstrap:
    return Bootstrap.model_validate(
        {
            "events": [
                {"id": 4, "name": "Gameweek 4", "deadline_time": "2026-09-13T10:00:00Z"},
                {
                    "id": 5,
                    "name": "Gameweek 5",
                    "deadline_time": "2026-09-18T17:30:00Z",
                    "is_next": True,
                },
            ],
            "teams": [
                {"id": 1, "name": "Arsenal", "short_name": "ARS"},
                {"id": 2, "name": "Manchester City", "short_name": "MCI"},
                {"id": 3, "name": "Chelsea", "short_name": "CHE"},
                {"id": 4, "name": "West Ham", "short_name": "WHU"},
            ],
            "elements": [
                _player(411, "Erling", "Haaland", "Haaland", 2, 4, 72.7),
                _player(3, "Gabriel", "dos Santos Magalhães", "Gabriel", 1, 2, 40.0),
                _player(19, "Gabriel", "Martinelli Silva", "Martinelli", 1, 3, 5.0),
                _player(200, "Cole", "Palmer", "Palmer", 3, 3, 30.0),
                _player(201, "Alex", "Palmer", "A.Palmer", 4, 1, 10.0),  # < 5x -> не доминирует
            ],
        }
    )


class FakeTools:
    """Минимальный двойник LiveTools для обёрток MCP: bootstrap, next_gw, predict_player."""

    def __init__(self, bs: Bootstrap, *, fail: Exception | None = None) -> None:
        self._bs = bs
        self.fail = fail
        self.calls: list[Any] = []
        self.closed = False

    @property
    def bootstrap(self) -> Bootstrap:
        return self._bs

    def next_gw(self) -> int:
        return 5

    def now(self) -> datetime:
        return datetime(2026, 9, 17, 12, 0, tzinfo=UTC)

    def predict_player(self, inp: Any) -> PlayerPrediction:
        self.calls.append(inp)
        if self.fail is not None:
            raise self.fail
        p = self._bs.player(inp.player_id)
        by_gw = [
            GWPrediction(
                gw=g,
                xpts=7.1234 + i,
                sd=4.0789,
                p_start=0.98123,
                exp_minutes=84,
                components={"goals": 2.3456, "appearance": 1.9},
                fixtures=[
                    FixtureBrief(
                        opponent="SUN",
                        is_home=True,
                        fsi=1,
                        xg_for=2.34567,
                        xg_against=0.8,
                        clean_sheet_prob=0.44444,
                    )
                ],
                notes=[f"note {k}" for k in range(40)],  # длиннее MAX_LIST -> обрезка
            )
            for i, g in enumerate(range(inp.gw, inp.gw + inp.horizon))
        ]
        return PlayerPrediction(
            player=PlayerRef(
                id=p.id, name=p.web_name, team="MCI", position="FWD", price=15.5, ownership=72.7
            ),
            by_gw=by_gw,
            total_xpts=round(sum(g.xpts for g in by_gw), 2),
        )

    def search_strategy_kb(self, inp: Any) -> KBSearchOutput:
        self.calls.append(inp)
        if self.fail is not None:
            raise self.fail
        chunks = [
            KBChunkOut(
                chunk_id=1,
                doc_id=1,
                title="FPL Rules Copilot",
                url="https://www.premierleague.com/en/news/4661029",
                source="premierleague",
                tags=["rules", "hits"],
                text="x" * 500,  # длиннее MAX_STR -> обрезка
                score=0.98765,
            )
        ]
        return KBSearchOutput(query=inp.query, tags=list(inp.tags), chunks=chunks[: inp.k])

    def kb_stats(self) -> dict[str, Any]:
        return {"docs": 41, "chunks": 622, "tags": {"hits": {"docs": 2, "chunks": 43}}}

    def rank_players(self, inp: Any) -> RankPlayersOutput:
        self.calls.append(inp)
        if self.fail is not None:
            raise self.fail
        row = RankedPlayerRow(
            rank=1,
            id=411,
            name="Haaland",
            team="MCI",
            position="FWD",
            price=15.5,
            total_points=33,
            minutes=335,
            points_per_game=8.3,
            points_per_million=2.12903,
            form=8.2,
            gws_played=4,
            gws_in_history=4,
            gws_5plus=2,
            share_5plus_pct=50,
            gws_2plus=3,
            share_2plus_pct=75,
            std_points=6.905,
            min_points=1,
            max_points=17,
            points_by_gw={"GW1": 13, "GW2": 2, "GW3": 17, "GW4": 1},
            season_total_points=33 if inp.gw_from is not None else None,
        )
        return RankPlayersOutput(
            gw=inp.gw,
            metric=inp.metric,
            metric_label="x",
            window_label=window_label(inp.gw_from, inp.gw_to),
            gw_from=inp.gw_from,
            gw_to=inp.gw_to,
            filters={"position": inp.position, "max_price": inp.max_price},
            history_through_gw=4,
            candidates=1,
            rows=[row][: inp.limit],
        )

    def build_gameweek_plan(self, inp: Any) -> PlanOut:
        self.calls.append(inp)
        if self.fail is not None:
            raise self.fail
        gws = [str(g) for g in range(inp.gw, inp.gw + inp.horizon)]
        chips = [
            PlanChipOut(
                gw=c.gw,
                chip=c.chip,
                name={"bboost": "Bench Boost", "3xc": "Triple Captain"}[c.chip],
                points=4.56789,
                players=["Haaland"],
            )
            for c in inp.chips
        ]
        return PlanOut(
            from_gw=inp.gw,
            horizon=inp.horizon,
            strategy=inp.strategy,
            moves_by_gw={g: [] for g in gws},
            hits_by_gw={g: 0 for g in gws},
            ft_by_gw={g: 1 for g in gws},
            bank_by_gw={g: 0.5 for g in gws},
            xi_points_by_gw={g: 55.0 for g in gws},
            captain_by_gw={g: "Haaland" for g in gws},
            expected_total=275.0,
            baseline_total=270.0,
            recommendation="hold",
            allow_hits=inp.allow_hits,
            chips=chips,
        )

    def inputs(self, manager_id: int, gw: int, horizon: int, strategy: str) -> Any:
        chips = ["wildcard", "freehit", "3xc"]
        return SimpleNamespace(chips_by_gw={g: chips for g in range(gw, gw + horizon)})

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_runtime() -> Iterator[FakeTools]:
    saved_tools, saved_resolvers = RUNTIME._tools, dict(RUNTIME._resolvers)
    fake = FakeTools(fake_bootstrap())
    RUNTIME._tools = fake  # type: ignore[assignment]
    RUNTIME._resolvers.clear()
    try:
        yield fake
    finally:
        RUNTIME._tools = saved_tools
        RUNTIME._resolvers = saved_resolvers


def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = asyncio.run(mcp.call_tool(name, arguments))
    assert result.structured_content is not None
    return result.structured_content


# ---------- реестр ----------


def test_registry_has_fourteen_tools_with_descriptions_and_schemas():
    assert mcp.name == SERVER_NAME == "fpl-intelligence"
    tools = asyncio.run(mcp.list_tools())
    names = [t.name for t in tools]
    assert names == list(MCP_TOOLS) and len(names) == 14
    assert names[: len(TOOL_NAMES)] == list(TOOL_NAMES)  # инструменты агента — первыми
    for t in tools:
        assert t.description and len(t.description) > 80, t.name
        schema = t.input_schema
        assert schema["type"] == "object" and "properties" in schema, t.name
    by_name = {t.name: t for t in tools}
    assert by_name["predict_player"].input_schema["required"] == ["player"]
    assert by_name["compare_players"].input_schema["properties"]["players"]["type"] == "array"
    for name in ("optimize_team", "recommend_transfers", "build_gameweek_plan", "diagnose_squad"):
        assert by_name[name].input_schema["required"] == ["manager_id"], name
    strategy = by_name["optimize_team"].input_schema["properties"]["strategy"]
    assert strategy["enum"] == ["conservative", "balanced", "aggressive"]
    assert strategy["default"] == "balanced"
    plan_tool = by_name["build_gameweek_plan"]
    plan = plan_tool.input_schema["properties"]
    assert plan["allow_hits"] == {"type": "boolean", "default": True, "title": "Allow Hits"}
    assert plan["chips"]["default"] is None  # необязательный: без него план прежний
    array = next(s for s in plan["chips"]["anyOf"] if s.get("type") == "array")
    chip_def = plan_tool.input_schema["$defs"][array["items"]["$ref"].rsplit("/", 1)[-1]]
    assert chip_def["required"] == ["gw", "chip"]
    assert chip_def["properties"]["chip"]["enum"] == list(PLANNABLE_CHIPS) == ["bboost", "3xc"]
    assert "invalid_chip_plan" in plan_tool.description
    assert by_name["analyze_player_risk"].annotations.read_only_hint is False
    assert by_name["predict_player"].annotations.read_only_hint is True
    kb = by_name["search_strategy_kb"]
    assert kb.input_schema["required"] == ["query"]
    assert kb.input_schema["properties"]["k"]["default"] == 6
    assert kb.annotations.read_only_hint is True
    assert "never invent rules" in kb.description.lower() or "Never invent rules" in kb.description
    rank = by_name["rank_players"]
    assert rank.input_schema.get("required", []) == []  # все параметры со значениями по умолчанию
    assert rank.input_schema["properties"]["metric"]["default"] == "points_per_million"
    assert rank.input_schema["properties"]["metric"]["enum"] == list(RANKING_METRICS)
    assert rank.annotations.read_only_hint is True
    assert "not a forecast" in " ".join(rank.description.split())
    adv = by_name["get_player_advanced_stats"]
    assert adv.input_schema["required"] == ["player_id"]
    assert adv.annotations.read_only_hint is True
    assert "understat" in adv.description.lower()
    brk = by_name["get_player_points_breakdown"]
    assert brk.input_schema["required"] == ["player_id"]
    assert brk.annotations.read_only_hint is True
    assert "total_points" in brk.description.lower() or "category" in brk.description.lower()
    team = by_name["get_team_defensive_profile"]
    assert team.input_schema["required"] == ["team_id"]
    assert team.annotations.read_only_hint is True
    assert "xga" in team.description.lower() or "xGA" in team.description


def test_resources_templates_and_prompt_are_registered():
    resources = asyncio.run(mcp.list_resources())
    assert [str(r.uri) for r in resources] == ["fpl://gameweek/current", "fpl://kb/stats"]
    assert all(r.mime_type == "application/json" for r in resources)
    templates = asyncio.run(mcp.list_resource_templates())
    assert {t.uri_template for t in templates} == {
        "fpl://manager/{manager_id}/squad",
        "fpl://manager/{manager_id}/plan",
        "fpl://player/{player_id}/signal",
    }
    prompts = asyncio.run(mcp.list_prompts())
    assert [p.name for p in prompts] == ["pre_deadline_review"]
    args = {a.name: a.required for a in prompts[0].arguments}
    assert args == {"manager_id": True, "strategy": False}


def test_prompt_renders_the_skill_tool_sequence():
    out = asyncio.run(mcp.get_prompt("pre_deadline_review", {"manager_id": "895045"}))
    msg = out.messages[0]
    assert msg.role == "user"
    text = msg.content.text
    assert "manager 895045" in text and "strategy: balanced" in text
    order = [
        "get_gameweek_context",
        "diagnose_squad",
        "analyze_player_risk",
        "recommend_transfers",
        "optimize_team",
        "build_gameweek_plan",
    ]
    positions = [text.index(name) for name in order]
    assert positions == sorted(positions)
    assert "confirm" in text and "betting" in text
    assert "search_strategy_kb" in text and text.index("search_strategy_kb") > positions[-1]


def test_search_strategy_kb_tool_validates_input_and_dumps_chunks(fake_runtime: FakeTools):
    out = call(
        "search_strategy_kb", {"query": "when is a -4 hit worth it", "tags": ["hits"], "k": 2}
    )
    assert "error" not in out, out
    inp = fake_runtime.calls[0]
    assert inp.query == "when is a -4 hit worth it" and inp.tags == ["hits"] and inp.k == 2
    chunk = out["chunks"][0]
    assert chunk["source"] == "premierleague" and chunk["tags"] == ["rules", "hits"]
    assert chunk["score"] == 0.99 and len(chunk["text"]) == rt.MAX_STR  # округление и обрезка
    assert "note" not in out

    out = call("search_strategy_kb", {"query": "  ", "k": 3})
    assert out["error"] == "invalid_input" and "query" in out["hint"]
    out = call("search_strategy_kb", {"query": "chips", "k": 99})
    assert out["error"] == "invalid_input" and "k must be" in out["hint"]
    out = call("search_strategy_kb", {"query": "chips", "tags": ["bets"]})
    assert out["error"] == "invalid_input" and "unknown tags ['bets']" in out["hint"]
    assert len(fake_runtime.calls) == 1  # невалидный вход до инструмента не доходит

    fake_runtime.fail = RuntimeError("kb down")
    out = call("search_strategy_kb", {"query": "chips"})
    assert out["error"] == "internal" and out["tool"] == "search_strategy_kb"


def test_rank_players_tool_validates_input_and_dumps_rows(fake_runtime: FakeTools):
    out = call(
        "rank_players",
        {"metric": "consistency", "position": "FWD", "max_price": 16.0, "limit": 5},
    )
    assert "error" not in out, out
    inp = fake_runtime.calls[0]
    assert inp.gw == 5 and inp.metric == "consistency" and inp.position == "FWD"
    assert inp.max_price == 16.0 and inp.limit == 5 and inp.min_minutes is None
    assert out["candidates"] == 1 and out["history_through_gw"] == 4
    row = out["rows"][0]
    assert row["name"] == "Haaland" and row["points_per_million"] == 2.13  # округление
    assert row["std_points"] == 6.91 and row["points_by_gw"] == {
        "GW1": 13,
        "GW2": 2,
        "GW3": 17,
        "GW4": 1,
    }

    assert out["window_label"] == "season to date" and out["gw_from"] is None
    assert inp.gw_from is None and inp.gw_to is None and inp.last_n_gws is None

    # окно туров пробрасывается в инструмент как есть (разрешает его инструмент)
    out = call("rank_players", {"metric": "total_points", "gw_from": 4, "gw_to": 5})
    assert "error" not in out, out
    inp = fake_runtime.calls[1]
    assert (inp.gw_from, inp.gw_to, inp.last_n_gws) == (4, 5, None)
    assert out["window_label"] == "GW4–GW5" and (out["gw_from"], out["gw_to"]) == (4, 5)
    assert out["rows"][0]["season_total_points"] == 33
    out = call("rank_players", {"last_n_gws": 3})
    assert "error" not in out and fake_runtime.calls[2].last_n_gws == 3

    out = call("rank_players", {"limit": 0})
    assert out["error"] == "invalid_input" and "limit must be" in out["hint"]
    out = call("rank_players", {"max_price": 99.0})
    assert out["error"] == "invalid_input" and "max_price" in out["hint"]
    out = call("rank_players", {"gw_from": 0})
    assert out["error"] == "invalid_input" and "gw_from must be" in out["hint"]
    out = call("rank_players", {"last_n_gws": 39})
    assert out["error"] == "invalid_input" and "last_n_gws must be" in out["hint"]
    assert len(fake_runtime.calls) == 3  # невалидный вход до инструмента не доходит
    fake_runtime.fail = OperationalError("SELECT 1", {}, Exception("refused"))
    out = call("rank_players", {})
    assert out["error"] == "db_unavailable" and out["tool"] == "rank_players"
    # окно без истории — отдельный код ошибки с подсказкой, не «internal»
    fake_runtime.fail = HistoryUnavailable("a gameweek window (GW4–GW5) needs player_gw_history")
    out = call("rank_players", {"gw_from": 4, "gw_to": 5})
    assert out["error"] == "history_unavailable" and "GW4–GW5" in out["hint"]
    assert "Drop the gameweek window" in out["hint"]


def test_kb_stats_resource_reads_runtime_tools(fake_runtime: FakeTools):
    contents = asyncio.run(mcp.read_resource("fpl://kb/stats"))
    payload = json.loads(contents[0].content)
    assert payload["docs"] == 41 and payload["tags"]["hits"]["chunks"] == 43


# ---------- build_gameweek_plan: фишки ----------


def test_build_gameweek_plan_without_chips_is_unchanged(fake_runtime: FakeTools):
    out = call("build_gameweek_plan", {"manager_id": 895045})
    assert "error" not in out, out
    inp = fake_runtime.calls[0]
    assert (inp.manager_id, inp.gw, inp.horizon, inp.strategy) == (895045, 5, 5, "balanced")
    assert inp.allow_hits is True and inp.chips == [] and inp.use_wildcard is None
    assert out["chips"] == [] and out["recommendation"] == "hold"


def test_build_gameweek_plan_passes_typed_chips(fake_runtime: FakeTools):
    chips = [{"gw": 7, "chip": "bboost"}, {"gw": 9, "chip": "3xc"}]
    out = call("build_gameweek_plan", {"manager_id": 6856911, "horizon": 5, "chips": chips})
    assert "error" not in out, out
    inp = fake_runtime.calls[0]
    assert [(c.gw, c.chip) for c in inp.chips] == [(7, "bboost"), (9, "3xc")]
    assert [(c["gw"], c["chip"], c["name"]) for c in out["chips"]] == [
        (7, "bboost", "Bench Boost"),
        (9, "3xc", "Triple Captain"),
    ]
    assert out["chips"][0]["points"] == 4.57  # округление обёртки


def test_build_gameweek_plan_chip_error_is_structured(fake_runtime: FakeTools):
    fake_runtime.fail = ChipPlanError(
        "Bench Boost недоступен менеджеру в GW7: уже сыгран в этой половине сезона "
        "или окно фишки закрыто"
    )
    out = call(
        "build_gameweek_plan", {"manager_id": 895045, "chips": [{"gw": 7, "chip": "bboost"}]}
    )
    assert out["error"] == "invalid_chip_plan" and out["tool"] == "build_gameweek_plan"
    assert "Bench Boost недоступен менеджеру в GW7" in out["hint"]
    assert "Free Hit is not modelled" in out["hint"] and "Traceback" not in str(out)
    # только планируемые фишки (BB / TC) по каждому туру горизонта
    assert out["chips_available_by_gw"] == {str(g): ["3xc"] for g in range(5, 10)}


def test_build_gameweek_plan_schema_rejects_unplannable_chip(fake_runtime: FakeTools):
    for bad in ({"gw": 7, "chip": "freehit"}, {"gw": 0, "chip": "3xc"}):
        with pytest.raises(ToolError, match="chip|gw"):
            asyncio.run(mcp.call_tool("build_gameweek_plan", {"manager_id": 1, "chips": [bad]}))
    assert fake_runtime.calls == []  # до инструмента не дошло


# ---------- справочные инструменты (Understat, разбор очков) ----------


def _ext_index() -> Any:
    teams = [
        UnderstatTeam("88", "Manchester City", matches=5, xg=10.0, xga=4.0),
        UnderstatTeam("83", "Arsenal", matches=5, xg=9.0, xga=6.0),
        UnderstatTeam("80", "Chelsea", matches=5, xg=8.0, xga=9.0),
        UnderstatTeam("81", "West Ham", matches=5, xg=5.0, xga=12.0, pens_conceded=2),
    ]
    haaland = UnderstatPlayer(
        "1", "Erling Haaland", "Manchester City", 450, 5, 6, 1, 4.5, 0.6, 3.74, 20
    )
    snap = UnderstatSnapshot(season=2026, players=[haaland], teams=teams, source="cache")
    return build_index(fake_bootstrap(), snap)


def test_player_advanced_stats_tool_returns_fpl_and_understat(
    fake_runtime: FakeTools, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(srv, "load_ext_index", lambda bs: _ext_index())
    out = call("get_player_advanced_stats", {"player_id": 411})
    assert "error" not in out, out
    assert (out["web_name"], out["team"], out["team_id"], out["position"]) == (
        "Haaland",
        "MCI",
        2,
        "FWD",
    )
    assert out["matched"] is True and out["sources"] == ["fpl", "understat"]
    assert out["understat"]["xg90"] == 0.9 and out["understat"]["shots"] == 20
    assert "odds" not in json.dumps(out).lower()

    out = call("get_player_advanced_stats", {"player_id": 200})  # нет строки Understat
    assert out["matched"] is False and out["understat"] is None and out["note"]
    for bad in (0, 999):
        out = call("get_player_advanced_stats", {"player_id": bad})
        assert out["error"] == "invalid_input" and "player" in out["hint"]


def test_team_defensive_profile_tool_flags_low_xga(
    fake_runtime: FakeTools, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(srv, "load_ext_index", lambda bs: _ext_index())
    mci = call("get_team_defensive_profile", {"team_id": 2})
    assert "error" not in mci, mci
    assert mci["short_name"] == "MCI" and mci["understat"]["xga_per_game"] == 0.8
    assert mci["league_xga_median"] == 1.8 and mci["concedes_little"] is True
    whu = call("get_team_defensive_profile", {"team_id": 4})
    assert whu["concedes_little"] is False and whu["pens_conceded"] == 2
    out = call("get_team_defensive_profile", {"team_id": 99})
    assert out["error"] == "invalid_input" and "no team with id 99" in out["hint"]


def _gw_row(rnd: int, **kw: Any) -> PlayerGWHistory:
    return PlayerGWHistory(element=411, fixture=rnd, round=rnd, was_home=True, minutes=90, **kw)


def test_points_breakdown_tool_sums_categories_to_total(
    fake_runtime: FakeTools, monkeypatch: pytest.MonkeyPatch
):
    seen: dict[str, Any] = {}

    def fake_history(*, before_gw: int, player_ids: list[int]) -> list[PlayerGWHistory]:
        seen.update(before_gw=before_gw, player_ids=player_ids)
        return [
            _gw_row(1, total_points=2),
            _gw_row(2, total_points=13, goals_scored=2, bonus=3, expected_goals=0.8),
            _gw_row(3, total_points=2, expected_goals=0.4),
            _gw_row(
                4,
                total_points=10,
                goals_scored=1,
                assists=1,
                bonus=2,
                yellow_cards=1,
                expected_goals=0.3,
                expected_assists=0.3,
            ),
        ]

    monkeypatch.setattr("fplcopilot.core.history.load_history", fake_history)
    out = call("get_player_points_breakdown", {"player_id": 411})
    assert "error" not in out, out
    assert seen == {"before_gw": 5, "player_ids": [411]}  # только завершённые туры
    assert (out["last_n"], out["n_rounds"], out["total_points"]) == (3, 3, 25)
    cats = out["categories"]
    assert (cats["appearance"], cats["goals"], cats["assists"], cats["bonus"]) == (6, 12, 3, 5)
    assert cats["cards"] == -1 and cats["other"] == 0 and out["sums_to_total"] is True
    assert out["label"] == "lucky"  # 15 очков за голы+ассисты при моментах на 6.9

    monkeypatch.setattr("fplcopilot.core.history.load_history", lambda **kw: [])
    empty = call("get_player_points_breakdown", {"player_id": 411, "last_n": 5})
    assert empty["n_rounds"] == 0 and empty["categories"] == {} and empty["label"] is None
    out = call("get_player_points_breakdown", {"player_id": 411, "last_n": 9})
    assert out["error"] == "invalid_input" and "last_n" in out["hint"]


# ---------- разрешение имён ----------


def test_resolve_players_error_shapes_never_guess():
    resolver = PlayerResolver(fake_bootstrap())
    ids, notes, err = rt.resolve_players(resolver, ["Haaland"], field="player")
    assert (ids, err) == ([411], None)

    ids, _, err = rt.resolve_players(resolver, ["Gabriel"], field="player")
    assert ids == [] and err is not None
    assert err["error"] == "ambiguous" and err["field"] == "player" and err["mention"] == "Gabriel"
    assert {c["id"] for c in err["candidates"]} == {3, 19}
    assert {"id", "name", "full_name", "team", "position", "price", "ownership"} <= set(
        err["candidates"][0]
    )
    assert "ask the user" in err["hint"]

    _, _, err = rt.resolve_players(resolver, ["Haaland", "Nobody Here"], field="players")
    assert err is not None and err["error"] == "unknown_player" and err["mention"] == "Nobody Here"

    ids, notes, err = rt.resolve_players(resolver, ["411", 200], field="players")
    assert (ids, err) == ([411, 200], None) and notes == ["id 411 = Haaland", "id 200 = Palmer"]
    _, _, err = rt.resolve_players(resolver, ["999"], field="player")
    assert err is not None and err["error"] == "unknown_player"

    # «Palmer» неоднозначен без состава, но в составе менеджера ровно один -> он, с пометкой
    _, _, err = rt.resolve_players(resolver, ["Palmer"], field="sell")
    assert err is not None and err["error"] == "ambiguous"
    squad_resolver = PlayerResolver(fake_bootstrap(), squad_ids=[200])
    ids, notes, err = rt.resolve_players(squad_resolver, ["Palmer"], field="sell")
    assert (ids, err) == ([200], None) and "in your squad" in notes[0]


def test_tool_returns_ambiguous_error_payload_instead_of_guessing(fake_runtime: FakeTools):
    out = call("predict_player", {"player": "Gabriel", "horizon": 2})
    assert out["error"] == "ambiguous" and out["mention"] == "Gabriel"
    assert [c["name"] for c in out["candidates"]] == ["Gabriel", "Martinelli"]
    assert fake_runtime.calls == []  # до инструмента дело не дошло
    out = call("compare_players", {"players": ["Haaland", "Gabriel"]})
    assert out["error"] == "ambiguous" and out["field"] == "players"


# ---------- обёртка, округление, обрезка ----------


def test_tool_wrapper_dumps_rounded_and_capped_payload(fake_runtime: FakeTools):
    out = call("predict_player", {"player": "Haaland", "horizon": 2})
    assert "error" not in out
    assert fake_runtime.calls[0].player_id == 411 and fake_runtime.calls[0].gw == 5
    assert out["player"]["name"] == "Haaland"
    gw = out["by_gw"][0]
    assert gw["xpts"] == 7.12 and gw["sd"] == 4.08 and gw["p_start"] == 0.98
    assert gw["components"]["goals"] == 2.35
    assert gw["fixtures"][0]["xg_for"] == 2.35 and gw["fixtures"][0]["is_home"] is True
    assert len(gw["notes"]) == rt.MAX_LIST and gw["notes_omitted"] == 40 - rt.MAX_LIST
    assert out["total_xpts"] == round(7.1234 + 8.1234, 2)


def test_compact_caps_lists_rounds_floats_truncates_strings():
    payload = {
        "xs": list(range(30)),
        "f": 1.23456,
        "i": 7,
        "ok": True,
        "none": None,
        "s": "a" * 500,
        "nested": {"ys": [{"v": 0.005}] * 3, "zs": [[1.111] * 40]},
    }
    out = rt.compact(payload)
    assert out["xs"] == list(range(rt.MAX_LIST)) and out["xs_omitted"] == 5
    assert out["f"] == 1.23 and out["i"] == 7 and out["ok"] is True and out["none"] is None
    assert len(out["s"]) == rt.MAX_STR and out["s"].endswith("…")
    assert out["nested"]["ys"][0]["v"] == 0.01  # round half even -> 0.0 или 0.01: только формат
    assert len(out["nested"]["zs"][0]) == rt.MAX_LIST
    assert rt.compact([1.999] * 3, max_list=2) == [2.0, 2.0]


# ---------- контракт ошибок ----------


def test_error_payloads_are_structured_without_stack_traces(fake_runtime: FakeTools):
    fake_runtime.fail = SquadUnavailable("manager 1 not found in FPL API")
    out = call("predict_player", {"player": "Haaland"})
    assert out["error"] == "squad_unavailable" and "manager 1 not found" in out["hint"]
    assert out["tool"] == "predict_player" and isinstance(out["latency_ms"], int)
    assert "Traceback" not in str(out)

    fake_runtime.fail = RuntimeError("boom")
    out = call("predict_player", {"player": "Haaland"})
    assert out["error"] == "internal" and out["type"] == "RuntimeError" and out["hint"] == "boom"

    out = call("predict_player", {"player": "Haaland", "horizon": 99})
    assert out["error"] == "invalid_input" and "horizon" in out["hint"]
    out = call("compare_players", {"players": ["Haaland"]})
    assert out["error"] == "invalid_input"
    out = call("simulate_scenario", {"manager_id": 0})
    assert out["error"] == "invalid_input" and "manager_id" in out["hint"]


def test_error_payload_mapping():
    req = httpx.Request("GET", "https://fantasy.premierleague.com/api/entry/1/")
    resp = httpx.Response(404, request=req)
    e404 = rt.error_payload(httpx.HTTPStatusError("404", request=req, response=resp))
    assert e404["error"] == "fpl_api_error" and e404["status"] == 404
    assert rt.error_payload(httpx.ConnectError("no route"))["error"] == "fpl_api_unreachable"
    db = rt.error_payload(OperationalError("SELECT 1", {}, Exception("refused")))
    assert db["error"] == "db_unavailable" and "docker compose" in db["hint"]
    assert rt.error_payload(ScenarioInfeasible("needs 3 transfers"))["error"] == "infeasible"
    assert rt.error_payload(rt.ToolInputError("bad"))["error"] == "invalid_input"
    chip = rt.error_payload(ChipPlanError("Free Hit оптимизатор не моделирует"))
    assert chip["error"] == "invalid_chip_plan" and "chips_available_by_gw" not in chip
    rejected = rt.error_payload(rt.ChipPlanRejected("x", {"6": []}))
    assert rejected["error"] == "invalid_chip_plan" and rejected["chips_available_by_gw"] == {
        "6": []
    }
    assert rt.error_payload(KeyError("x"))["error"] == "internal"


def test_forced_sale_verdict_compares_horizon_gains():
    def route(gain: float) -> RouteOut:
        return RouteOut(
            rank=1,
            out=["Palmer"],
            **{"in": ["Saka"]},
            out_ids=[1],
            in_ids=[2],
            hit_cost=0,
            gain_next_gw=1.0,
            gain_horizon=gain,
            gain_discounted=gain,
            objective_gain=gain,
            hit_marginal_gain=None,
            new_bank=0.5,
            risk_note="",
            verdict="go",
            xi_points_after=60.0,
        )

    assert _forced_sale_verdict([route(3.0)], route(5.0), "Palmer").startswith("keep Palmer")
    assert _forced_sale_verdict([route(5.0)], route(3.0), "Palmer").startswith("sell Palmer")
    assert _forced_sale_verdict([route(3.0)], route(3.2), "Palmer").startswith("roughly a tie")
    assert _forced_sale_verdict([route(3.0)], None, "Palmer").startswith("sell Palmer")
    assert _forced_sale_verdict([], None, "Palmer").startswith("no feasible route")


def test_runtime_close_releases_tools_and_resolvers():
    runtime = rt.Runtime(factory=lambda: FakeTools(fake_bootstrap()))  # type: ignore[arg-type]
    fake = runtime.tools
    runtime.resolver([200])
    assert runtime._tools is fake and runtime._resolvers
    runtime.close()
    assert fake.closed and runtime._tools is None and not runtime._resolvers
