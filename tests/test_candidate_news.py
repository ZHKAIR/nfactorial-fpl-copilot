"""Узел candidate_news (после compute) на фейках: новости по кандидатам оптимизатора и их клубам,
бюджет LLM-извлечений, один пересчёт без недоступной покупки, клубные цитаты отдельно от
доказательств доступности игрока, блок NEWS промпта v3. Без сети / LLM / БД."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from test_agent_fakes import (
    BS,
    NOW,
    FakeExplain,
    FakeRouter,
    FakeTools,
    fit_risk,
    router_output,
)

from fplcopilot.agent import llm as agent_llm
from fplcopilot.agent.graph import Agent, Deps, route_targets
from fplcopilot.agent.llm import ExplainRequest, load_agent_prompt, render_explain_user
from fplcopilot.agent.resolve import PlayerResolver
from fplcopilot.agent.tools import (
    ClubAbsenceOut,
    EvidenceItem,
    PlayerRisk,
    RouteOut,
    TeamNewsItemOut,
    TeamNewsOut,
)
from fplcopilot.agent.validate import prune_undiscussed_sources

CLUB_QUOTE = "Arteta said he will rotate the squad for the cup tie."


def route_ids(rank: int, out: list[int], inn: list[int]) -> RouteOut:
    return RouteOut(
        rank=rank,
        out=[BS.player(p).web_name for p in out],
        in_=[BS.player(p).web_name for p in inn],
        out_ids=out,
        in_ids=inn,
        hit_cost=0,
        gain_next_gw=1.5,
        gain_horizon=4.03,
        gain_discounted=3.5,
        objective_gain=2.1,
        hit_marginal_gain=None,
        new_bank=0.3,
        risk_note="",
        verdict="go",
        xi_points_after=57.6,
    )


def risk_with_team(inp, *, availability="fit", conf=0.8, origin="cached", llm_calls=0):
    r = fit_risk(inp)
    p = BS.player(inp.player_id)
    return r.model_copy(
        update={
            "team_id": p.team,
            "availability": availability,
            "confidence": conf,
            "origin": "cached" if inp.cached_only else origin,
            "llm_calls": 0 if inp.cached_only else llm_calls,
            "age_h": 50.0 if inp.cached_only else 1.0,
        }
    )


class NewsTools(FakeTools):
    """FakeTools + дайджест клуба; маршрут: Palmer -> Saka, без Saka (exclude) -> Palmer -> Neto."""

    def __init__(self, *, risk=risk_with_team, **kw):
        super().__init__(risk=risk, **kw)

    def recommend_transfers(self, inp):
        buy = 156 if 12 in inp.exclude else 12
        self.routes = [route_ids(1, [154], [buy]), route_ids(2, [165], [611])]
        return super().recommend_transfers(inp)

    def team_news(self, inp):
        self._rec("team_news", inp)
        team = BS.team(inp.team_id)
        return TeamNewsOut(
            team_id=team.id,
            team=team.short_name,
            team_name=team.name,
            absences=[
                ClubAbsenceOut(
                    player_id=18,
                    player="Martinelli",
                    position="MID",
                    status="s",
                    status_label="suspended",
                    fpl_news="Suspended until 17 Oct",
                    return_date="2026-10-17",
                    return_gw=7,
                )
            ]
            if team.short_name == "ARS"
            else [],
            summary="Arteta plans rotation for the cup." if team.short_name == "ARS" else "",
            items=[
                TeamNewsItemOut(
                    kind="rotation",
                    claim="Arteta will rotate for the cup.",
                    source="sky_football",
                    url="https://sky.example/ars",
                    published_at="2026-09-16T09:00:00+00:00",
                    date="16.09",
                    quote=CLUB_QUOTE,
                )
            ]
            if team.short_name == "ARS"
            else [],
            origin="cached",
            digest_as_of=NOW.isoformat(),
            age_h=2.0,
        )


def make_agent(tools, router, explain, **kw) -> Agent:
    deps = Deps(
        tools=tools,
        router_llm=router,
        explain_llm=explain,
        resolver=lambda squad: PlayerResolver(BS, squad_ids=squad),
        clock=lambda: NOW,
        extra_known_names=["Arsenal", "Chelsea"],
        **kw,
    )
    return Agent(deps, MemorySaver())


def test_transfer_candidates_get_player_and_club_news_in_facts_and_evidence():
    tools = NewsTools()
    answer = (
        "**Verdict:** Palmer -> Saka (+4.03 over the horizon).\n\n**Why**\n"
        "- Saka trained fully [bbc, 16.09].\n- Club news: Arsenal rotate [sky_football, 16.09]."
        "\n\n**Sources**\n- bbc, 16.09, https://bbc.example/x\n"
        "- sky_football, 16.09, https://sky.example/ars"
    )
    explain = FakeExplain([answer])
    state = make_agent(tools, FakeRouter(router_output("transfer")), explain).run(
        "Best transfer this week?", manager_id=1
    )
    nodes = [t["node"] for t in state["tool_log"]]
    assert nodes.index("candidate_news") > nodes.index("compute")
    risk_calls = [c.player_id for c in tools.called("analyze_player_risk")]
    assert 12 in risk_calls and 611 in risk_calls  # покупки маршрутов 1 и 2
    assert tools.called("team_news")[0].team_id == 1  # клуб рекомендованной покупки
    cn = state["facts"]["candidate_news"]["players"]
    assert cn["Saka"]["role"] == "buy, route 1 (recommended)"
    assert cn["Saka"]["news_signal"]["availability"] == "fit"
    assert cn["Saka"]["xpts_by_gw"]["GW5"] == 6.2  # xPts модели рядом с новостью
    assert cn["Palmer"]["role"].startswith("sell")
    club = state["facts"]["team_news"]["ARS"]
    assert club["unavailable_or_doubtful_players"][0]["return_gw"] == 7
    assert club["context_items"][0]["kind"] == "rotation"
    # клубная цитата — в EVIDENCE с пометкой, но не в доказательствах доступности Saka
    club_ev = [e for e in state["evidence"] if e.get("scope") == "club"]
    assert club_ev and club_ev[0]["quote"] == CLUB_QUOTE and club_ev[0]["club"] == "Arsenal"
    saka = state["candidate_signals"]["12"]
    assert all(e["quote"] != CLUB_QUOTE for e in saka["evidence"])
    assert cn["Saka"]["news_signal"]["evidence_count"] == len(saka["evidence"]) == 1
    assert state["validation"]["passed"]
    assert "sky.example/ars" in state["answer"]  # клуб назван -> строка Sources остаётся
    assert not state["news_reoptimized"] and len(tools.called("recommend_transfers")) == 1


def test_unavailable_recommended_buy_triggers_one_reoptimization():
    def risk(inp):
        injured = inp.player_id == 12
        return risk_with_team(inp, availability="injured" if injured else "fit", conf=0.9)

    tools = NewsTools(risk=risk)
    state = make_agent(tools, FakeRouter(router_output("transfer")), FakeExplain()).run(
        "Best transfer this week?", manager_id=1
    )
    calls = tools.called("recommend_transfers")
    assert [c.exclude for c in calls] == [[], [12]]  # ровно один пересчёт без Saka
    assert state["news_reoptimized"] and state["news_exclude"] == [12]
    assert state["facts"]["transfers"]["routes"][0]["in"] == ["Neto"]
    assert any("Saka" in c and "re-ran without him" in c for c in state["caveats"])
    nodes = [t["node"] for t in state["tool_log"]]
    assert nodes.count("compute") == 2 and nodes.count("check_action") == 1


def test_forced_buy_is_not_excluded_only_flagged():
    def risk(inp):
        return risk_with_team(inp, availability="injured" if inp.player_id == 12 else "fit")

    tools = NewsTools(risk=risk)
    state = make_agent(
        tools, FakeRouter(router_output("transfer", ["Saka"], buy=["Saka"])), FakeExplain()
    ).run("Should I buy Saka?", manager_id=1)
    assert len(tools.called("recommend_transfers")) == 1 and not state["news_reoptimized"]


def test_budget_limits_extractions_and_falls_back_to_saved_signals():
    tools = NewsTools(risk=lambda inp: risk_with_team(inp, origin="extracted", llm_calls=1))
    state = make_agent(
        tools, FakeRouter(router_output("transfer")), FakeExplain(), news_max_llm_calls=1
    ).run("Best transfer this week?", manager_id=1)
    news_calls = [c for c in tools.called("analyze_player_risk") if c.player_id in (12, 154, 611)]
    assert news_calls[0].cached_only is False and all(c.cached_only for c in news_calls[1:])
    assert state["news_llm_used"] == 1
    assert all(c.cached_only for c in tools.called("team_news"))  # бюджет уже исчерпан
    assert any("not refreshed within this request's budget" in c for c in state["caveats"])


def test_player_status_gets_club_news_without_extra_player_signals():
    tools = NewsTools()
    state = make_agent(
        tools, FakeRouter(router_output("player_status", ["Saka"])), FakeExplain()
    ).run("Is Saka fit?")
    assert len(tools.called("analyze_player_risk")) == 1  # только ensure_signals
    assert "ARS" in state["facts"]["team_news"] and "candidate_news" not in state["facts"]


def test_disabled_or_irrelevant_intent_skips_the_node():
    tools = NewsTools()
    state = make_agent(
        tools, FakeRouter(router_output("transfer")), FakeExplain(), candidate_news=False
    ).run("Best transfer?", manager_id=1)
    assert not tools.called("team_news") and "candidate_news" not in state["facts"]
    tools = NewsTools()
    make_agent(tools, FakeRouter(router_output("captain")), FakeExplain()).run(
        "Captain?", manager_id=1
    )
    assert not tools.called("team_news")


def test_route_targets_order_and_roles():
    r1, r2 = route_ids(1, [154], [12]), route_ids(2, [165], [611])
    t = route_targets([r2, r1], 1, forced_buy={611}, n_routes=2)
    assert [(x["id"], x["role"], x["primary"]) for x in t] == [
        (12, "buy, route 1 (recommended)", True),
        (154, "sell, route 1 (recommended)", False),
        (611, "buy, route 2", False),
    ]
    assert t[2]["forced"] is True and t[1]["club_news"] is False


# ---------- промпт v3: блок NEWS, цепочка версий ----------


def _req(facts, evidence):
    return ExplainRequest(
        query="q", intent="transfer", strategy="balanced", as_of="now", facts=facts,
        evidence=evidence, caveats=[], language="en",
    )  # fmt: skip


def test_news_block_only_in_v3_with_ready_citations():
    facts = {
        "candidate_news": {
            "players": {
                "Saka": {"role": "buy, route 1", "news_signal": {"availability": "fit",
                         "confidence": 0.8, "signal_as_of": "2026-09-17 10:00Z"}},
                "Neto": {"role": "buy, route 2", "news_signal": {"availability": "n/a"}},
            }
        },
        "team_news": {
            "ARS": {"club": "Arsenal", "summary": "Rotation for the cup.",
                    "unavailable_or_doubtful_players": [
                        {"player": "Martinelli", "status_label": "suspended", "return_gw": 7}]}
        },
    }  # fmt: skip
    evidence = [
        EvidenceItem(
            player_id=12, player="Saka", source="bbc", url="u", published_at="", date="16.09",
            quote="q",
        ).model_dump(),
        {"scope": "club", "team": "ARS", "source": "sky_football", "date": "16.09"},
    ]  # fmt: skip
    v3 = render_explain_user(_req(facts, evidence), "v3")
    assert "NEWS (from FACTS.candidate_news" in v3
    assert "- Saka (buy, route 1): news fit" in v3 and "cite [bbc, 16.09]" in v3
    assert "Neto (buy, route 2): news n/a — no player-specific news" in v3
    assert "Martinelli suspended until GW7" in v3 and "cite [sky_football, 16.09]" in v3
    assert "NEWS (from" not in render_explain_user(_req(facts, evidence), "v2")


def test_v3_prompt_files_fall_back_through_v2_to_v1():
    assert load_agent_prompt("explain.system", "v3") != load_agent_prompt("explain.system", "v2")
    assert "NEWS FOR RECOMMENDED" in load_agent_prompt("explain.system", "v3")
    assert load_agent_prompt("grader.system", "v3") == load_agent_prompt("grader.system", "v1")
    assert load_agent_prompt("compare.system", "v3") == load_agent_prompt("compare.system", "v2")
    assert "weak spots" in load_agent_prompt("router.system", "v3")
    assert "weak spots" not in load_agent_prompt("router.system", "v2")
    assert agent_llm.news_block_enabled("v3") and not agent_llm.news_block_enabled("v2")


def test_prune_keeps_club_source_only_when_club_is_discussed():
    ev = [{"scope": "club", "club": "Arsenal", "team": "ARS", "url": "https://sky.example/ars"}]
    kept = (
        "**Why**\n- Arsenal rotate.\n\n**Sources**\n- sky_football, 16.09, https://sky.example/ars"
    )
    assert prune_undiscussed_sources(kept, ev)[1] == []
    gone = "**Why**\n- Chelsea.\n\n**Sources**\n- sky_football, 16.09, https://sky.example/ars"
    assert prune_undiscussed_sources(gone, ev)[1]


def test_player_risk_carries_team_id_default_none():
    assert PlayerRisk(player_id=1, player="x", fpl_status="a", fpl_chance=None).team_id is None
