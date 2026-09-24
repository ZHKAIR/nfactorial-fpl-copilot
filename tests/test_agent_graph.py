"""LangGraph-агент на фейках: ветвление, ограниченный цикл, HITL, валидация ответа.

Без сети, LLM и БД: инструменты/LLM/резолвер инжектируются через Deps, чекпоинтер — MemorySaver.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver
from test_agent_fakes import (
    BS,
    NOW,
    FakeExplain,
    FakeGrader,
    FakeRouter,
    FakeTools,
    ScenarioOut,
    fit_risk,
    low_conf_risk,
    route,
    router_output,
    unknown_risk,
)

from fplcopilot.agent.cli import compact_state, format_footer
from fplcopilot.agent.graph import RETRY_STRATEGIES, Agent, Deps, build_graph
from fplcopilot.agent.resolve import PlayerResolver


def make_agent(tools: FakeTools, router: FakeRouter, explain: FakeExplain, grader=None) -> Agent:
    deps = Deps(
        tools=tools,
        router_llm=router,
        explain_llm=explain,
        grader_llm=grader,
        resolver=lambda squad: PlayerResolver(BS, squad_ids=squad),
        clock=lambda: NOW,
        extra_known_names=["Arsenal", "Chelsea", "Brentford"],
    )
    return Agent(deps, MemorySaver())


def tool_names(state: dict, node: str | None = None) -> list[str]:
    return [t["tool"] for t in state["tool_log"] if node is None or t["node"] == node]


# ---------- ветвление ----------


def test_betting_goes_to_refuse_without_llm_explain():
    tools, router, explain = FakeTools(), FakeRouter(router_output("betting")), FakeExplain()
    agent = make_agent(tools, router, explain)
    state = agent.run("Best odds for Arsenal to win?", manager_id=1)
    assert state["intent"] == "betting"
    assert "betting advice" in state["answer"]
    assert explain.calls == []
    assert not tools.called("analyze_player_risk") and not tools.called("recommend_transfers")
    assert [t["node"] for t in state["tool_log"]][-1] == "refuse"
    assert state["llm_calls"][-1]["purpose"] == "route"  # единственный LLM-вызов — роутер
    assert not state["interrupted"]


def test_off_topic_refuses_in_one_line():
    agent = make_agent(FakeTools(), FakeRouter(router_output("off_topic")), FakeExplain())
    state = agent.run("Give me a recipe for plov", manager_id=1)
    assert state["intent"] == "off_topic"
    assert "Fantasy Premier League" in state["answer"] and "\n" not in state["answer"].strip()


def test_ambiguous_mention_branches_to_clarify_and_interrupts():
    tools = FakeTools()
    explain = FakeExplain()
    agent = make_agent(
        tools, FakeRouter(router_output("transfer", ["Gabriel"], sell=["Gabriel"])), explain
    )
    state = agent.run("Should I sell Gabriel?", manager_id=1)
    assert state["clarification"]["mentions"][0]["mention"] == "Gabriel"
    assert state["clarification"]["mentions"][0]["roles"] == ["sell"]
    names = {c["full_name"] for c in state["clarification"]["mentions"][0]["candidates"]}
    assert "Gabriel Martinelli Silva" in names and "Gabriel dos Santos Magalhães" in names
    assert "Which **Gabriel** do you mean?" in state["answer"]
    assert explain.calls == [] and not tools.called("recommend_transfers")
    assert [t["node"] for t in state["tool_log"]][-1] == "clarify"
    # v2: уточнение — HITL-прерывание перед resolve_clarification, а не терминальная ветка
    assert state["interrupted"] and state["next_nodes"] == ["resolve_clarification"]
    assert state["interrupt_kind"] == "clarification"
    pending = state["pending_clarification"]
    assert pending["mention"] == "Gabriel" and pending["roles"] == ["sell"]
    assert {c["player_id"] for c in pending["candidates"]} == {4, 18, 27, 331}
    assert next(c for c in pending["candidates"] if c["player_id"] == 4)["in_squad"] is True
    assert "PENDING CLARIFICATION" in format_footer(state)


def test_transfer_happy_path_calls_tools_and_validates():
    tools = FakeTools(routes=[route(1, ["Palmer"], ["Saka"])])
    explain = FakeExplain(
        [
            "**Verdict:** sell Palmer for Saka (+4.03 over the horizon).\n\n**Why**\n- Palmer 4.76 xPts."
        ]
    )
    agent = make_agent(
        tools, FakeRouter(router_output("transfer", ["Palmer"], sell=["Palmer"])), explain
    )
    state = agent.run("Should I sell Palmer?", manager_id=1)
    assert state["intent"] == "transfer"
    assert [p["id"] for p in state["players"]] == [154]  # Cole Palmer — тот, что в составе
    assert state["scenario"]["sell"] == [154]
    rt = tools.called("recommend_transfers")[0]
    assert rt.sell == [154] and rt.allow_hit is None
    assert "recommend_transfers" in tool_names(state, "compute")
    assert "predict_player" in tool_names(state, "compute")
    assert state["validation"]["passed"] and state["explain_attempts"] == 1
    assert state["facts"]["transfers"]["recommendation"] == "route 1"
    assert not state["interrupted"] and state["pending_action"] is None
    footer = format_footer(state)
    assert "intent: transfer" in footer and "recommend_transfers" in footer
    assert compact_state(state)["predictions"]["_omitted"] is True


def test_no_squad_answers_player_part_with_caveat():
    tools = FakeTools(squad=False)
    agent = make_agent(
        tools, FakeRouter(router_output("transfer", ["Palmer"], sell=["Palmer"])), FakeExplain()
    )
    state = agent.run("Should I sell Palmer?", manager_id=10835228)
    assert state["squad_summary"] is None
    assert not tools.called("recommend_transfers")
    assert tools.called("predict_player")  # уровень игрока отвечаем всё равно
    assert any("manager id" in c for c in state["caveats"])
    # без состава Palmer неоднозначен (Cole vs Alex) -> доминирование по владению с пометкой
    assert [p["id"] for p in state["players"]] == [154]
    assert any("assumed to be Cole Palmer" in c for c in state["caveats"])


# ---------- ограниченный цикл повторного поиска ----------


def test_retry_loop_is_bounded_and_uses_alternative_strategies():
    tools = FakeTools(risk=unknown_risk)  # João Pedro: статус d, сигнал unknown всегда
    agent = make_agent(
        tools, FakeRouter(router_output("player_status", ["João Pedro"])), FakeExplain()
    )
    state = agent.run("Is João Pedro fit for GW5?")
    assert state["retries"] == 2
    risk_calls = tools.called("analyze_player_risk")
    assert len(risk_calls) == 3  # первичный + 2 повтора
    assert [(c.force, c.mode, c.k) for c in risk_calls] == [
        (False, "hybrid_rerank", 8),
        (True, *RETRY_STRATEGIES[0]),
        (True, *RETRY_STRATEGIES[1]),
    ]
    retry_log = [t for t in state["tool_log"] if t["node"] == "rewrite_retry"]
    assert [t["args"]["strategy"] for t in retry_log] == ["dense k=8", "hybrid_rerank k=12"]
    grades = [t for t in state["tool_log"] if t["node"] == "grade_signals"]
    assert len(grades) == 3 and all("insufficient" in t["note"] for t in grades)
    assert state["grade"]["sufficient"] is False and state["grade"]["will_retry"] is False
    assert state["answer"]  # после исчерпания попыток граф продолжает и отвечает
    assert tools.invalidated >= 1  # свежие извлечения сбрасывают кэш прогнозов


def test_llm_grader_decides_for_low_confidence_signal_with_evidence():
    grader = FakeGrader(sufficient=False)
    tools = FakeTools(risk=low_conf_risk)
    agent = make_agent(
        tools, FakeRouter(router_output("player_status", ["João Pedro"])), FakeExplain(), grader
    )
    state = agent.run("Is João Pedro fit for GW5?")
    assert len(grader.calls) == 3 and grader.calls[0].player == "João Pedro"
    assert state["retries"] == 2
    assert any(c["purpose"] == "grade" for c in state["llm_calls"])


def test_sufficient_signal_skips_retry():
    tools = FakeTools(risk=fit_risk)
    agent = make_agent(
        tools, FakeRouter(router_output("player_status", ["João Pedro"])), FakeExplain()
    )
    state = agent.run("Is João Pedro fit for GW5?")
    assert state["retries"] == 0 and len(tools.called("analyze_player_risk")) == 1
    assert state["grade"]["sufficient"] is True
    assert state["evidence"][0]["source"] == "bbc"


# ---------- human-in-the-loop ----------


def hit_tools() -> FakeTools:
    return FakeTools(routes=[route(1, ["Palmer", "João Pedro"], ["Saka", "Guéhi"], hit=4)])


def test_hit_route_interrupts_before_confirm_action():
    tools = hit_tools()
    explain = FakeExplain()
    agent = make_agent(
        tools,
        FakeRouter(router_output("transfer", ["Palmer"], sell=["Palmer"], allow_hit=True)),
        explain,
    )
    state = agent.run("Should I sell Palmer even for a -4?", manager_id=1, thread_id="t-hit")
    assert state["interrupted"] and state["next_nodes"] == ["confirm_action"]
    assert state["pending_action"]["kind"] == "hit" and state["pending_action"]["cost"] == 4
    assert state["answer"] is None and explain.calls == []
    assert "PENDING CONFIRMATION" in format_footer(state)


def test_resume_reject_recomputes_without_hit():
    tools = hit_tools()
    explain = FakeExplain()
    agent = make_agent(
        tools,
        FakeRouter(router_output("transfer", ["Palmer"], sell=["Palmer"], allow_hit=True)),
        explain,
    )
    agent.run("Should I sell Palmer even for a -4?", manager_id=1, thread_id="t-reject")
    state = agent.resume("t-reject", "reject")
    calls = tools.called("recommend_transfers")
    assert [c.allow_hit for c in calls] == [True, False]  # второй прогон compute без хита
    assert state["user_decision"] == "reject" and state["scenario"]["allow_hit"] is False
    assert state["facts"]["transfers"]["routes"][0]["hit_cost"] == 0
    assert not state["interrupted"] and state["answer"]
    nodes = [t["node"] for t in state["tool_log"]]
    assert nodes.index("confirm_action") < nodes.index("compute", nodes.index("confirm_action"))
    assert any("rejected the hit" in c for c in state["caveats"])
    assert len(explain.calls) == 1


def test_resume_confirm_keeps_hit():
    tools = hit_tools()
    explain = FakeExplain()
    agent = make_agent(
        tools,
        FakeRouter(router_output("transfer", ["Palmer"], sell=["Palmer"], allow_hit=True)),
        explain,
    )
    agent.run("Should I sell Palmer even for a -4?", manager_id=1, thread_id="t-confirm")
    state = agent.resume("t-confirm", "confirm")
    assert len(tools.called("recommend_transfers")) == 1  # без пересчёта
    assert state["facts"]["transfers"]["routes"][0]["hit_cost"] == 4
    assert state["user_decision"] == "confirm" and state["answer"]
    assert explain.calls[0].facts["user_decision"]["decision"] == "confirm"
    assert any("confirmed the hit" in c for c in state["caveats"])


def test_resume_requires_waiting_thread():
    agent = make_agent(FakeTools(), FakeRouter(router_output("off_topic")), FakeExplain())
    agent.run("plov", thread_id="t-plain")
    with pytest.raises(ValueError):
        agent.resume("t-plain", "confirm")
    with pytest.raises(ValueError):
        agent.resume("t-unknown", "reject")


def test_what_if_wildcard_scenario_asks_confirmation():
    scenario = ScenarioOut(
        gw=5,
        horizon=5,
        strategy="balanced",
        feasible=True,
        free_transfers=2,
        bank=0.1,
        transfers_cap=0,
        wildcard={"expected_total": 321.9, "delta_vs_plan": 14.4, "moves": ["a -> b"]},
        verdict="transfers",
    )
    tools = FakeTools(scenario=scenario)
    agent = make_agent(
        tools, FakeRouter(router_output("what_if", use_wildcard=True)), FakeExplain()
    )
    state = agent.run("What if I wildcard now?", manager_id=1, thread_id="t-wc")
    assert state["pending_action"]["kind"] == "wildcard" and state["interrupted"]


# ---------- валидация ответа ----------


def test_validate_regenerates_once_on_fabricated_number():
    tools = FakeTools()
    explain = FakeExplain(
        [
            "**Verdict:** Palmer 4.76 xPts, Saka 6.31 xPts.",  # 6.31 нет в фактах
            "**Verdict:** Palmer 4.76 xPts, Saka 6.2 xPts.",
        ]
    )
    router = FakeRouter(router_output("compare_players", ["Palmer", "Saka"], needs_squad=False))
    state = make_agent(tools, router, explain).run("Palmer or Saka?")
    assert len(explain.calls) == 2
    assert "6.31" in (explain.calls[1].feedback or "")
    assert state["validation"]["passed"] and state["explain_attempts"] == 2
    assert state["validation"]["history"][0]["unknown_numbers"] == ["6.31"]
    assert "⚠" not in state["answer"]


def test_validate_regenerates_on_number_of_another_player():
    explain = FakeExplain(
        [
            "**Verdict:** keep Saka.\n\n**Why**\n- Palmer: 6.2 xPts next GW.",  # 6.2 — Saka
            "**Verdict:** keep Saka.\n\n**Why**\n- Palmer: 4.76 xPts next GW.",
        ]
    )
    router = FakeRouter(router_output("compare_players", ["Palmer", "Saka"], needs_squad=False))
    state = make_agent(FakeTools(), router, explain).run("Palmer or Saka?")
    assert len(explain.calls) == 2
    assert "belongs to Saka, not Palmer" in (explain.calls[1].feedback or "")
    first = state["validation"]["history"][0]
    assert first["unknown_numbers"] == []
    assert [(m["player"], m["number"]) for m in first["misattributed_numbers"]] == [
        ("Palmer", "6.2")
    ]
    assert state["validation"]["passed"] and state["explain_attempts"] == 2


def test_misattributed_number_gets_caveat_after_second_failure():
    wrong = "**Verdict:** keep Saka.\n\n**Why**\n- Palmer: 6.2 xPts next GW."
    explain = FakeExplain([wrong, wrong])
    router = FakeRouter(router_output("compare_players", ["Palmer", "Saka"], needs_squad=False))
    state = make_agent(FakeTools(), router, explain).run("Palmer or Saka?")
    assert state["validation"]["passed"] is False
    assert "could not be matched to the computed facts" in state["answer"]
    assert "Palmer 6.2" in state["answer"]


def test_validate_flags_unknown_name_and_adds_caveat_after_second_failure():
    explain = FakeExplain(["**Verdict:** bring in Salah instead of Palmer."])  # Salah не в фактах
    router = FakeRouter(router_output("compare_players", ["Palmer", "Saka"], needs_squad=False))
    state = make_agent(FakeTools(), router, explain).run("Palmer or Saka?")
    assert len(explain.calls) == 2  # ровно одна регенерация
    assert state["validation"]["passed"] is False
    assert state["validation"]["unknown_names"] == ["Salah"]
    # оговорка на языке вопроса, без английского технического «⚠ Validation»
    assert "could not be matched to the computed facts" in state["answer"]
    assert "⚠ Validation" not in state["answer"] and "Salah" in state["answer"]


# ---------- структура графа и стриминг ----------


def test_graph_structure_has_branch_loop_and_interrupt():
    deps = Deps(
        tools=FakeTools(),
        router_llm=FakeRouter(router_output("captain")),
        explain_llm=FakeExplain(),
        resolver=lambda squad: PlayerResolver(BS, squad_ids=squad),
    )
    graph = build_graph(deps, MemorySaver())
    mermaid = graph.get_graph().draw_mermaid()
    for node in (
        "router",
        "refuse",
        "clarify",
        "resolve_clarification",
        "ensure_signals",
        "grade_signals",
        "rewrite_retry",
        "compute",
        "check_action",
        "confirm_action",
        "explain",
        "validate_answer",
    ):
        assert node in mermaid
    assert "rewrite_retry --> grade_signals" in mermaid  # цикл
    assert (
        mermaid.count("__interrupt = before") == 2
    )  # HITL: confirm_action + resolve_clarification
    assert "clarify --> resolve_clarification" in mermaid
    assert "resolve_clarification -.-> ensure_signals" in mermaid


def test_stream_yields_node_statuses_and_captain_options():
    tools = FakeTools()
    agent = make_agent(tools, FakeRouter(router_output("captain")), FakeExplain())
    events = list(agent.stream("Who should I captain this week?", manager_id=1, thread_id="t-cap"))
    nodes = [n for n, _ in events]
    assert nodes[:3] == ["start", "load_context", "router"]
    assert "compute" in nodes and nodes[-1] == "validate_answer"
    assert "lang=en" in dict(events)["router"]
    state = agent.snapshot("t-cap")
    assert state["facts"]["captain_options"][0]["name"] == "Haaland"
    assert state["facts"]["captain_options"][0]["tag"] == "safe"
    assert tools.called("optimize_team")
