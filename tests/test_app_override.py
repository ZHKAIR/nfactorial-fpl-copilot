"""squad_override: компактный SquadOverride, прокси клиента, передача через агент в инструменты,
cached_only для analyze_player_risk. Без сети/LLM/БД."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from langgraph.checkpoint.memory import MemorySaver
from test_agent_fakes import BS, NOW, FakeExplain, FakeRouter, FakeTools, router_output

from fplcopilot.agent.graph import Agent, Deps, squad_tool
from fplcopilot.agent.resolve import PlayerResolver
from fplcopilot.agent.tools import (
    LiveTools,
    PlayerRiskInput,
    SquadOverride,
    _ClientWithSquad,
    as_override,
)
from fplcopilot.data.schemas import Pick, Squad, SquadPlayer

SQUAD_IDS = [301, 4, 388, 611, 499, 154, 12, 156, 18, 165, 411, 27, 331, 611, 499]


def make_squad(captain: int = 411, bank: float = 0.5) -> Squad:
    """15 picks из мини-bootstrap (дубликаты id допустимы для теста формата)."""
    players = []
    for slot, pid in enumerate(SQUAD_IDS, start=1):
        p = BS.player(pid)
        pick = Pick(
            element=pid,
            position=slot,
            multiplier=0 if slot > 11 else (2 if pid == captain else 1),
            is_captain=pid == captain and slot <= 11,
            is_vice_captain=slot == 2,
        )
        players.append(SquadPlayer(pick=pick, player=p, team=BS.team(p.team)))
    return Squad(manager_id=0, gw=5, players=players, bank=bank, team_value=100.0, free_transfers=1)


def test_squad_override_round_trip_and_fingerprint():
    squad = make_squad()
    ov = SquadOverride.from_squad(squad)
    assert ov.gw == 5 and ov.bank == 0.5 and ov.free_transfers == 1 and ov.source == "screenshot"
    assert [p.element for p in ov.picks] == SQUAD_IDS
    back = ov.to_squad(BS, manager_id=42)
    assert back.manager_id == 42 and back.gw == 5
    assert [p.player.id for p in back.players] == SQUAD_IDS
    assert back.captain is not None and back.captain.player.id == 411
    assert len(back.starting_xi) == 11 and len(back.bench) == 4
    # JSON round trip (состояние агента / чекпоинты)
    again = SquadOverride.model_validate(ov.model_dump(mode="json"))
    assert again.fingerprint() == ov.fingerprint()
    assert SquadOverride.from_squad(make_squad(bank=1.5)).fingerprint() != ov.fingerprint()
    assert as_override(None) is None and as_override(ov) is ov
    assert as_override(squad).fingerprint() == ov.fingerprint()


def test_client_proxy_returns_override_and_delegates_the_rest():
    class Client:
        def bootstrap(self):
            return "bs"

        def squad(self, manager_id, gw=None):
            raise AssertionError("real squad() must not be called")

    squad = make_squad()
    proxy = _ClientWithSquad(Client(), squad)  # type: ignore[arg-type]
    assert proxy.squad(895045, 4) is squad
    assert proxy.bootstrap() == "bs"


class OverrideAwareTools(FakeTools):
    """Фейк, принимающий squad_override у инструментов уровня состава (как LiveTools)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.overrides: dict[str, SquadOverride | None] = {}

    def get_gameweek_context(self, inp, *, squad_override=None):
        self.overrides["get_gameweek_context"] = squad_override
        if squad_override is not None and inp.manager_id is None:
            inp = inp.model_copy(update={"manager_id": 0})  # как LiveTools: состав без менеджера
        ctx = super().get_gameweek_context(inp)
        if squad_override is not None:
            ctx.squad_gw = squad_override.gw
            ctx.squad_note = "squad = uploaded screenshot"
        return ctx

    def diagnose_squad(self, inp, *, squad_override=None):
        self.overrides["diagnose_squad"] = squad_override
        return super().diagnose_squad(inp)

    def optimize_team(self, inp, *, squad_override=None):
        self.overrides["optimize_team"] = squad_override
        lu = super().optimize_team(inp)
        if squad_override is not None:
            lu = lu.model_copy(update={"expected_points": 52.44, "captain": "B.Fernandes"})
        return lu

    def recommend_transfers(self, inp, *, squad_override=None):
        self.overrides["recommend_transfers"] = squad_override
        return super().recommend_transfers(inp)


def make_agent(tools, router):
    deps = Deps(
        tools=tools,
        router_llm=router,
        explain_llm=FakeExplain(),
        resolver=lambda squad: PlayerResolver(BS, squad_ids=squad),
        clock=lambda: NOW,
        extra_known_names=["Arsenal", "Chelsea"],
    )
    return Agent(deps, MemorySaver())


def test_agent_passes_squad_override_to_squad_tools_and_keeps_it_in_state():
    tools = OverrideAwareTools()
    agent = make_agent(tools, FakeRouter(router_output("captain")))
    ov = SquadOverride.from_squad(make_squad())
    state = agent.run("Who should I captain?", manager_id=None, squad_override=ov)
    assert state["squad_override"] == ov.model_dump(mode="json")  # JSON в состоянии/чекпоинте
    for name in ("get_gameweek_context", "diagnose_squad", "optimize_team"):
        got = tools.overrides[name]
        assert isinstance(got, SquadOverride) and got.fingerprint() == ov.fingerprint(), name
    assert state["facts"]["best_xi"]["expected_points"] == 52.44
    assert state["facts"]["best_xi"]["captain"] == "B.Fernandes"
    assert state["squad_summary"]["squad_gw"] == 5
    assert any("uploaded screenshot" in c for c in state["caveats"])
    # без override — kwarg не передаётся вовсе (фейки со старой сигнатурой продолжают работать)
    plain = FakeTools()
    state2 = make_agent(plain, FakeRouter(router_output("captain"))).run("Captain?", manager_id=1)
    assert (
        state2["squad_override"] is None and state2["facts"]["best_xi"]["expected_points"] == 56.14
    )


def test_agent_accepts_full_squad_and_stream_kwarg():
    tools = OverrideAwareTools()
    agent = make_agent(tools, FakeRouter(router_output("captain")))
    events = list(
        agent.stream("Captain?", manager_id=1, squad_override=make_squad(), thread_id="t-ov")
    )
    assert events[0] == ("start", "thread t-ov")
    assert tools.overrides["optimize_team"] is not None
    assert agent.snapshot("t-ov")["squad_override"]["gw"] == 5


def test_squad_tool_wraps_only_when_override_present():
    calls = []

    def fn(inp, **kw):
        calls.append(kw)
        return inp

    assert squad_tool(fn, {"squad_override": None}) is fn
    wrapped = squad_tool(
        fn, {"squad_override": SquadOverride.from_squad(make_squad()).model_dump()}
    )
    wrapped("x")
    assert isinstance(calls[0]["squad_override"], SquadOverride)


class _Client:
    def bootstrap(self):
        return BS

    def close(self):
        pass


def test_live_tools_cached_only_never_calls_llm(monkeypatch):
    tools = LiveTools(client=_Client(), now=NOW)  # type: ignore[arg-type]
    as_of = NOW

    # 1) сигнала в БД нет -> unavailable без извлечения
    monkeypatch.setattr(tools, "_latest_signal", lambda pid, as_of: None)
    risk = tools.analyze_player_risk(PlayerRiskInput(player_id=154, as_of=as_of, cached_only=True))
    assert risk.origin == "unavailable" and "cached_only" in (risk.note or "")
    assert risk.player == "Palmer" and risk.fpl_status == "a"

    # 2) устаревший сигнал (30 ч > max_age 12) -> отдаётся как cached с пометкой stale
    stale = {
        "as_of": as_of - timedelta(hours=30),
        "availability": "fit",
        "start_probability": 0.9,
        "expected_minutes": 80,
        "rotation_risk": "low",
        "return_gw": None,
        "confidence": 0.8,
        "summary": "fit",
        "evidence": [],
        "fpl_status": "a",
        "fpl_chance_next": None,
        "abstained": False,
        "mode": "hybrid_rerank",
        "model": "gpt-4o-mini",
    }
    monkeypatch.setattr(tools, "_latest_signal", lambda pid, as_of: stale)
    risk = tools.analyze_player_risk(PlayerRiskInput(player_id=154, as_of=as_of, cached_only=True))
    assert risk.origin == "cached" and risk.age_h == 30.0 and "stale" in (risk.note or "")
    assert risk.availability == "fit" and risk.confidence == 0.8

    # 3) свежий сигнал -> без пометки
    fresh = {**stale, "as_of": as_of - timedelta(hours=1)}
    monkeypatch.setattr(tools, "_latest_signal", lambda pid, as_of: fresh)
    risk = tools.analyze_player_risk(PlayerRiskInput(player_id=154, as_of=as_of))
    assert risk.origin == "cached" and risk.note is None and risk.age_h == 1.0


def test_live_tools_override_squad_helpers():
    tools = LiveTools(client=_Client(), now=NOW)  # type: ignore[arg-type]
    squad = make_squad()
    assert tools.override_squad(1, None) is None
    assert tools.override_squad(1, squad) is squad
    rebuilt = tools.override_squad(7, SquadOverride.from_squad(squad))
    assert rebuilt.manager_id == 7 and [p.player.id for p in rebuilt.players] == SQUAD_IDS
    ov = tools.squad_override_from(squad, manager_id=None)
    assert ov.chips_used == [] and ov.gw == 5
    with pytest.raises(KeyError):  # чужой id в picks -> bootstrap его не знает
        SquadOverride(
            gw=5, picks=[Pick(element=999999, position=1, multiplier=1)], bank=0, team_value=0
        ).to_squad(BS)


def test_now_helper_is_utc():
    assert LiveTools(client=_Client(), now=NOW).now() == datetime(2026, 9, 17, 10, 0, tzinfo=UTC)  # type: ignore[arg-type]
