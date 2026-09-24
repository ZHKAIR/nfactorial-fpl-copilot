"""Агент v2 на фейках: стратегическая KB в графе (strategy_question, rules_context), уточнение
имени как HITL-прерывание с resume(player_id), язык ответа, промпты v2 и версия по умолчанию.

Без сети, LLM и БД: инструменты/LLM/резолвер инжектируются через Deps, чекпоинтер — MemorySaver.
"""

from __future__ import annotations

import json

import pytest
from langgraph.checkpoint.memory import MemorySaver
from test_agent_fakes import (
    BS,
    KB_ANSWER,
    NOW,
    RANK_HISTORY,
    RANK_ROUNDS,
    FakeExplain,
    FakeRouter,
    FakeTools,
    ScenarioOut,
    rank_rows_from_history,
    route,
    router_output,
    usage,
)

from fplcopilot.agent import llm as agent_llm
from fplcopilot.agent.cli import format_footer
from fplcopilot.agent.graph import (
    RULES_CONTEXT_K,
    Agent,
    Deps,
    clarification_text,
    ranking_params,
    rules_context_kind,
    rules_context_query,
    status_line,
)
from fplcopilot.agent.llm import (
    ExplainRequest,
    RankingDraft,
    RouterOutputV2,
    ScenarioDraft,
    detect_language,
    load_agent_prompt,
    normalize_language,
    render_explain_user,
)
from fplcopilot.agent.resolve import PlayerResolver
from fplcopilot.agent.state import INTENTS
from fplcopilot.agent.tools import (
    RANKING_METRICS,
    TOOL_NAMES,
    HistoryUnavailable,
    LiveTools,
    RankPlayersInput,
    StrategyAnswerOutput,
    consistency_stats,
    rank_player_rows,
    resolve_window,
    window_label,
)
from fplcopilot.config import settings


def make_agent(tools: FakeTools, router, explain: FakeExplain | None = None, **deps_kw) -> Agent:
    deps = Deps(
        tools=tools,
        router_llm=router,
        explain_llm=explain or FakeExplain(),
        resolver=lambda squad: PlayerResolver(BS, squad_ids=squad),
        clock=lambda: NOW,
        extra_known_names=["Arsenal", "Chelsea", "Brentford"],
        **deps_kw,
    )
    return Agent(deps, MemorySaver())


def tool_names(state: dict, node: str | None = None) -> list[str]:
    return [t["tool"] for t in state["tool_log"] if node is None or t["node"] == node]


# ---------- strategy_question -> стратегическая KB ----------


def test_strategy_question_calls_kb_and_keeps_citations():
    tools = FakeTools()
    explain = FakeExplain(
        [
            (
                "**Verdict:** play the Wildcard when several players are injured [1]; the first "
                "Wildcard expires at the GW19 deadline [2].\n\n**Sources**\n"
                "- [1] FPL Chips Strategy Guide (fplwatch, guide) — https://fplwatch.example/chips\n"
                "- [2] FPL Rules Copilot (premierleague, rules) — "
                "https://www.premierleague.com/en/news/4661029"
            )
        ]
    )
    agent = make_agent(tools, FakeRouter(router_output("strategy_question")), explain)
    state = agent.run("When should I play my wildcard?", manager_id=1)
    assert state["intent"] == "strategy_question"
    kb_calls = tools.called("answer_strategy_question")
    assert len(kb_calls) == 1 and kb_calls[0].query == "When should I play my wildcard?"
    assert "answer_strategy_question" in tool_names(state, "compute")
    assert not tools.called("recommend_transfers") and not tools.called("search_strategy_kb")
    sa = state["facts"]["strategy_answer"]
    assert sa["covered"] and [c["n"] for c in sa["citations"]] == [1, 2]
    assert sa["citations"][1]["official_rules"] is True
    assert "[1] FPL Chips Strategy Guide (fplwatch, guide)" in sa["sources_markdown"]
    assert "rules_digest" not in state["facts"]
    assert state["facts"]["headline"].startswith("Strategy KB answer with 2 cited source(s)")
    # объяснитель получил ответ KB и цитаты; его [1]/[2] прошли валидатор (номера есть в фактах)
    assert explain.calls[0].facts["strategy_answer"]["answer"] == KB_ANSWER.answer
    assert (
        state["validation"]["passed"] and state["validation"]["checked_refs"] == 4
    )  # текст + Sources
    assert "[1]" in state["answer"] and "[2]" in state["answer"]
    # единственный LLM-вызов инструмента учтён отдельно от роутера/объяснителя
    purposes = [c["purpose"] for c in state["llm_calls"]]
    assert purposes == ["route", "kb_answer", "explain"]
    kb_call = state["llm_calls"][1]
    assert kb_call["model"] == "fake-kb-model" and kb_call["prompt_version"] == "kb:v1"
    assert kb_call["cost_usd"] == 0.0002
    assert any("cited knowledge base" in c for c in state["caveats"])


def test_strategy_question_not_covered_adds_honest_caveat():
    not_covered = StrategyAnswerOutput(
        query="x",
        answer="Not covered by the strategy knowledge base.",
        covered=False,
        citations=[],
        retrieved=6,
        llm_calls=1,
        model="fake-kb-model",
    )
    tools = FakeTools(kb_answer=not_covered)
    explain = FakeExplain(["**Verdict:** the knowledge base does not cover this question."])
    state = make_agent(tools, FakeRouter(router_output("strategy_question")), explain).run(
        "Who wins UCL fantasy?"
    )
    assert state["facts"]["strategy_answer"]["covered"] is False
    assert "Not covered" in state["facts"]["headline"]
    assert any("does not cover" in c for c in state["caveats"])
    assert state["validation"]["passed"]


def test_strategy_question_falls_back_to_rules_digest_when_kb_fails():
    tools = FakeTools()

    def boom(inp):
        raise RuntimeError("kb down")

    tools.answer_strategy_question = boom  # type: ignore[method-assign]
    state = make_agent(tools, FakeRouter(router_output("strategy_question")), FakeExplain()).run(
        "How do free transfers work?"
    )
    assert "strategy_answer" not in state["facts"] and "rules_digest" in state["facts"]
    failed = [t for t in state["tool_log"] if t["tool"] == "answer_strategy_question"]
    assert failed and failed[0]["ok"] is False
    assert any("unavailable" in c for c in state["caveats"])


def test_validator_rejects_unknown_kb_reference_and_regenerates_once():
    tools = FakeTools()
    explain = FakeExplain(
        [
            "**Verdict:** wildcard when injured [1]; expires GW19 [7].",  # [7] нет в цитатах
            "**Verdict:** wildcard when injured [1]; expires GW19 [2].",
        ]
    )
    state = make_agent(tools, FakeRouter(router_output("strategy_question")), explain).run(
        "When should I play my wildcard?"
    )
    assert len(explain.calls) == 2
    assert "[7]" in (explain.calls[1].feedback or "")
    assert state["validation"]["history"][0]["unknown_refs"] == ["[7]"]
    assert state["validation"]["passed"] and state["explain_attempts"] == 2


# ---------- rules_context при хите / чипе ----------


def hit_router():
    return FakeRouter(router_output("transfer", ["Palmer"], sell=["Palmer"], allow_hit=True))


def test_hit_recommendation_adds_rules_context_from_kb():
    tools = FakeTools(routes=[route(1, ["Palmer", "João Pedro"], ["Saka", "Guéhi"], hit=4)])
    agent = make_agent(tools, hit_router(), FakeExplain())
    state = agent.run("Should I sell Palmer even for a -4?", manager_id=1, thread_id="t-rc")
    assert state["interrupted"] and state["pending_action"]["kind"] == "hit"
    kb = tools.called("search_strategy_kb")
    assert len(kb) == 1 and kb[0].tags == ["hits"] and kb[0].k == RULES_CONTEXT_K
    assert "points hit" in kb[0].query
    rc = state["facts"]["rules_context"]
    assert rc["about"] == "hit" and len(rc["excerpts"]) == 2
    assert rc["excerpts"][0]["source"] == "premierleague" and rc["excerpts"][0]["official_rules"]
    assert "deduct 4 points" in rc["excerpts"][0]["text"]
    assert rc["excerpts"][1]["source"] == "fplwatch" and not rc["excerpts"][1]["official_rules"]
    note = next(t for t in state["tool_log"] if t["tool"] == "search_strategy_kb")["note"]
    assert note.startswith("hit: 2 chunk(s)")
    # после confirm объяснитель видит rules_context и может цитировать его как [source]
    explain = FakeExplain(
        [
            "**Verdict:** hit confirmed.\n\n**Why**\n- every extra transfer costs 4 points [premierleague]."
        ]
    )
    agent.deps.explain_llm = explain
    final = agent.resume("t-rc", "confirm")
    assert explain.calls[0].facts["rules_context"]["about"] == "hit"
    assert final["validation"]["passed"] and "[premierleague]" in final["answer"]


def test_wildcard_scenario_adds_chips_context_and_plain_transfer_adds_none():
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
    state = make_agent(tools, FakeRouter(router_output("what_if", use_wildcard=True))).run(
        "What if I wildcard now?", manager_id=1
    )
    kb = tools.called("search_strategy_kb")
    assert len(kb) == 1 and kb[0].tags == ["chips"] and "wildcard" in kb[0].query
    assert state["facts"]["rules_context"]["about"] == "wildcard"
    assert state["facts"]["rules_context"]["excerpts"][0]["source"] == "fplwatch"

    plain = FakeTools()  # бесплатный маршрут, хит не спрашивали -> KB не трогаем
    state = make_agent(
        plain, FakeRouter(router_output("transfer", ["Palmer"], sell=["Palmer"]))
    ).run("Should I sell Palmer?", manager_id=1)
    assert not plain.called("search_strategy_kb") and "rules_context" not in state["facts"]

    off = FakeTools(routes=[route(1, ["Palmer"], ["Saka"], hit=4)])  # выключено в Deps
    state = make_agent(off, hit_router(), rules_context=False).run("sell Palmer -4", manager_id=1)
    assert not off.called("search_strategy_kb") and "rules_context" not in state["facts"]


def test_rules_context_kind_and_query_are_deterministic():
    assert rules_context_kind("transfer", {"transfers": {"recommendation": "hold"}}, {}, 5) is None
    facts = {
        "transfers": {
            "recommendation": "route 1",
            "routes": [{"hit_cost": 4, "out": ["A"], "in": ["B"]}],
        }
    }
    assert rules_context_kind("transfer", facts, {}, 5) == "hit"
    assert (
        rules_context_kind(
            "transfer", {"transfers": {"recommendation": "hold"}}, {"allow_hit": True}, 5
        )
        == "hit"
    )
    assert rules_context_kind("plan", {"plan": {"recommendation": "wildcard"}}, {}, 5) == "wildcard"
    assert (
        rules_context_kind(
            "plan",
            {
                "plan": {
                    "recommendation": "transfers",
                    "wildcard_alternative": {"expected_total": 1},
                }
            },
            {},
            5,
        )
        == "wildcard"
    )
    assert (
        rules_context_kind("captain", {}, {"allow_hit": True}, 5) == "hit"
    )  # интент фильтруется в compute
    q, tags = rules_context_query("hit", {"scenario": {"sell": [1, 2], "buy": [3, 4]}})
    assert "two transfers" in q and tags == ("hits",)
    q, tags = rules_context_query("hit", {"scenario": {}})
    assert q == "when is a -4 points hit worth it" and tags == ("hits",)
    many = {"issues": [{"severity": 3}, {"severity": 2}, {"severity": 2}]}
    assert "many injured" in rules_context_query("wildcard", many)[0]
    assert rules_context_query("wildcard", {"issues": []}) == (
        "when is the right time to play the wildcard",
        ("chips",),
    )


# ---------- уточнение имени: прерывание и resume с player_id ----------


def gabriel_router():
    return FakeRouter(router_output("transfer", ["Gabriel"], sell=["Gabriel"]))


def test_clarify_resume_with_player_id_continues_with_chosen_player():
    tools = FakeTools()
    explain = FakeExplain(["**Verdict:** sell Gabriel for Saka (+4.03 over the horizon)."])
    agent = make_agent(tools, gabriel_router(), explain)
    events = list(agent.stream("Should I sell Gabriel?", manager_id=1, thread_id="t-clar"))
    nodes = [n for n, _ in events]
    assert nodes[-2:] == ["clarify", "__interrupt__"]
    assert "waiting for the user's choice" in dict(events)["__interrupt__"]
    state = agent.snapshot("t-clar")
    assert state["interrupted"] and state["next_nodes"] == ["resolve_clarification"]
    assert explain.calls == [] and not tools.called("recommend_transfers")

    final = agent.resume("t-clar", player_id=4)  # Gabriel Magalhães (в составе)
    assert not final["interrupted"] and final["pending_clarification"] is None
    assert [p["id"] for p in final["players"]] == [4]
    assert final["players"][0]["full_name"] == "Gabriel dos Santos Magalhães"
    assert final["scenario"]["sell"] == [4]
    rt = tools.called("recommend_transfers")
    assert len(rt) == 1 and rt[0].sell == [4]
    assert tools.called("analyze_player_risk")[0].player_id == 4
    assert final["clarification_choice"] == {"decision": "choose", "player_id": 4}
    assert final["clarification_history"][0]["mention"] == "Gabriel"
    assert any("clarified by you as Gabriel dos Santos Magalhães" in c for c in final["caveats"])
    assert final["answer"].startswith("**Verdict:** sell Gabriel") and final["validation"]["passed"]
    nodes = [t["node"] for t in final["tool_log"]]
    assert nodes.index("resolve_clarification") < nodes.index("ensure_signals")
    assert "PENDING CLARIFICATION" not in format_footer(final)


def test_clarify_resume_validates_candidate_and_supports_cancel():
    tools = FakeTools()
    agent = make_agent(tools, gabriel_router(), FakeExplain())
    agent.run("Should I sell Gabriel?", manager_id=1, thread_id="t-bad")
    with pytest.raises(ValueError, match="candidates"):
        agent.resume("t-bad", player_id=411)  # Haaland — не кандидат
    with pytest.raises(ValueError, match="player choice"):
        agent.resume("t-bad", "confirm")  # решение для другого вида прерывания
    still = agent.snapshot("t-bad")
    assert still["interrupted"] and still["next_nodes"] == ["resolve_clarification"]

    cancelled = agent.resume("t-bad", "cancel")
    assert not cancelled["interrupted"] and cancelled["clarification_choice"] == {
        "decision": "cancel"
    }
    assert "Clarification cancelled" in cancelled["answer"]
    assert "Which **Gabriel** do you mean?" in cancelled["answer"]
    assert not tools.called("recommend_transfers") and not tools.called("analyze_player_risk")
    assert cancelled["tool_log"][-1]["node"] == "resolve_clarification"


def test_two_ambiguous_mentions_are_clarified_one_by_one():
    tools = FakeTools()
    router = FakeRouter(
        router_output("transfer", ["Gabriel", "Pedro"], sell=["Gabriel"], buy=["Pedro"])
    )
    agent = make_agent(tools, router, FakeExplain())
    state = agent.run("Should I sell Gabriel and buy Pedro?", manager_id=1, thread_id="t-two")
    assert state["pending_clarification"]["mention"] == "Gabriel"
    assert [m["mention"] for m in state["clarification"]["mentions"]] == ["Gabriel", "Pedro"]
    second = agent.resume("t-two", player_id=4)
    assert second["interrupted"] and second["pending_clarification"]["mention"] == "Pedro"
    assert second["pending_clarification"]["roles"] == ["buy"]
    # Neto (Pedro Lomba Neto), Pedro Porro и João Pedro (web name содержит «Pedro»)
    assert {c["player_id"] for c in second["pending_clarification"]["candidates"]} == {
        156,
        165,
        499,
    }
    assert "Which **Pedro** do you mean?" in second["answer"]
    final = agent.resume("t-two", player_id=499)  # Pedro Porro
    assert not final["interrupted"]
    assert sorted(p["id"] for p in final["players"]) == [4, 499]
    assert final["scenario"] == {**final["scenario"], "sell": [4], "buy": [499]}
    rt = tools.called("recommend_transfers")[0]
    assert rt.sell == [4] and rt.buy == [499]
    assert [h["mention"] for h in final["clarification_history"]] == ["Gabriel", "Pedro"]


def test_clarification_text_in_russian_and_status_lines():
    pending = {
        "mention": "Gabriel",
        "candidates": [
            {
                "player_id": 4,
                "name": "Gabriel",
                "full_name": "Gabriel dos Santos Magalhães",
                "team": "ARS",
                "position": "DEF",
                "price": 8.0,
                "ownership": 23.3,
                "status": "a",
                "in_squad": True,
            }
        ],
    }
    ru = clarification_text(pending, "ru")
    assert ru.startswith("Какого **Gabriel** вы имеете в виду?") and "в вашем составе" in ru
    en = clarification_text(pending, "en")
    # подсказка CLI (`--choose`) в тексте чата не нужна — CLI печатает команду resume сам
    assert "Which **Gabriel** do you mean?" in en and "--choose" not in en
    assert "choice of player" in status_line("__interrupt__", None, after="clarify")
    assert "confirm | reject" in status_line("__interrupt__", None, after="check_action")
    assert "ambiguous 'Gabriel': 1 candidates" in status_line(
        "clarify", {"pending_clarification": pending}
    )
    assert "cancelled" in status_line(
        "resolve_clarification", {"clarification_choice": {"decision": "cancel"}}
    )


# ---------- player_ranking: рейтинг всей лиги вместо стратегической KB ----------


def ranking_router_v2(**ranking) -> FakeRouter:
    out = RouterOutputV2(
        intent="player_ranking",
        player_mentions=[],
        horizon=None,
        scenario=ScenarioDraft(sell=[], buy=[], keep=[], allow_hit=None, use_wildcard=None),
        needs_squad=False,
        reason="fake",
        language="ru",
        ranking=RankingDraft(
            metric=ranking.get("metric"),
            position=ranking.get("position"),
            max_price=ranking.get("max_price"),
            min_price=ranking.get("min_price"),
            limit=ranking.get("limit"),
            gw_from=ranking.get("gw_from"),
            gw_to=ranking.get("gw_to"),
            last_n_gws=ranking.get("last_n_gws"),
        ),
    )
    return FakeRouter(out)


def test_player_ranking_intent_calls_rank_players_not_the_kb():
    tools = FakeTools()
    explain = FakeExplain(
        [
            (
                "**Вердикт:** лучший по очкам за £1m — Guéhi (£6.0, 25 pts, 4.17 pts/£m).\n\n"
                "| # | Player | Price | Points | Pts/£m |\n|---|---|---|---|---|\n"
                "| 1 | Guéhi (in your squad) | £6.0 | 25 | 4.17 |\n| 2 | Saka | £9.5 | 23 | 2.42 |"
            )
        ]
    )
    # роутер выбрал одну ось (consistency) — текст называет обе («дешёвый» + «стабильный»),
    # код поднимает метрику до составной budget_consistency (страховка, как HORIZON_HINT)
    agent = make_agent(tools, ranking_router_v2(metric="consistency"), explain)
    state = agent.run("самый дешевый но самый стабильный игрок по принесенным очкам?", manager_id=1)
    assert state["intent"] == "player_ranking" and "player_ranking" in INTENTS
    assert state["ranking"] == {
        "metric": "budget_consistency",
        "position": None,
        "max_price": None,
        "min_price": None,
        "limit": 8,
        "gw_from": None,
        "gw_to": None,
        "last_n_gws": None,
    }
    # инструмент рейтинга вызван, стратегическая KB — нет; состав передан для флага in_squad
    calls = tools.called("rank_players")
    assert len(calls) == 1 and calls[0].metric == "budget_consistency" and calls[0].gw == 5
    assert sorted(calls[0].squad_ids) == sorted([154, 165, 4, 411, 388])
    assert not tools.called("answer_strategy_question") and not tools.called("search_strategy_kb")
    assert not tools.called("predict_player") and not tools.called("recommend_transfers")
    pr = state["facts"]["player_ranking"]
    assert pr["metric"] == "budget_consistency" and pr["candidates_after_filters"] == 4
    assert [r["player"] for r in pr["rows"]] == ["Guéhi", "Saka", "Haaland", "Palmer"]
    top = pr["rows"][0]
    assert top["full_name"] == "Marc Guéhi"  # полное имя в фактах -> валидатор его знает
    assert top["price"] == 6.0 and top["total_points"] == 25 and top["points_per_million"] == 4.17
    assert top["gws_5plus"] == 4 and top["gws_played"] == 4 and top["in_your_squad"] is True
    assert top["gws_in_history"] == 4 and top["std_points"] == 1.09 and top["gws_dnp"] == 0
    assert "never re-rank" in pr["order"]
    assert "manager" not in state["facts"]  # рейтинг лиги — не про состав пользователя
    assert state["facts"]["headline"].startswith(
        "Top 4 of 4 players by fake budget_consistency, season to date "
        "[players with at least 180 minutes played]"
    )
    assert (
        "1. Guéhi (MCI, DEF, £6.0, 25 pts, 4.17 pts/£m, >=5 pts in 4 of 4 GWs)"
        in (state["facts"]["headline"])
    )
    # окно и смысл фильтров даны объяснителю словами: «не менее N минут», а не «менее»
    assert pr["window_label"] == "season to date" and pr["gw_from"] is None
    assert "AT LEAST this many minutes" in pr["filters_definition"]
    assert "season totals to date" in pr["points_definition"]
    assert "season_total_points" not in pr["rows"][0]  # только при окне туров
    assert explain.calls[0].intent == "player_ranking" and explain.calls[0].language == "ru"
    assert state["validation"]["passed"]  # 6.0 / 25 / 4.17 / 9.5 / 2.42 — все числа из фактов
    assert any(
        "season totals to date (past points, not a forecast)" in c and "at least 180 minutes" in c
        for c in state["caveats"]
    )
    note = next(t for t in state["tool_log"] if t["tool"] == "rank_players")["note"]
    assert note.startswith("budget_consistency, season to date: top 4 of 4 candidates")
    assert "ranking=metric=budget_consistency,limit=8" in status_line("router", state)
    assert state["llm_calls"] and [c["purpose"] for c in state["llm_calls"]] == ["route", "explain"]


def test_player_ranking_filters_from_router_v2_and_regex_fallback():
    tools = FakeTools()
    state = make_agent(
        tools, ranking_router_v2(metric="consistency", position="DEF", max_price=6.0, limit=3)
    ).run("какие защитники дешевле 6.0 играют стабильно?", manager_id=1)
    inp = tools.called("rank_players")[0]
    assert (inp.metric, inp.position, inp.max_price, inp.limit) == ("consistency", "DEF", 6.0, 3)
    rows = state["facts"]["player_ranking"]["rows"]
    assert [r["player"] for r in rows] == ["Guéhi"]  # единственный DEF <= £6.0 с минутами

    # роутер v1 (без поля ranking) / пустые поля: параметры достраивает regex-страховка
    tools = FakeTools()
    state = make_agent(tools, FakeRouter(router_output("player_ranking"))).run(
        "who is the best value midfielder under 10.0?"
    )
    inp = tools.called("rank_players")[0]
    assert (inp.metric, inp.position, inp.max_price) == ("points_per_million", "MID", 10.0)
    assert inp.squad_ids == []  # без менеджера — без флагов in_squad
    assert [r["player"] for r in state["facts"]["player_ranking"]["rows"]] == ["Saka", "Palmer"]
    assert state["language"] == "en" and state["horizon"] == 1


def test_ranking_params_regex_fallback_and_validation():
    assert ranking_params(None, "самый дешевый но самый стабильный игрок по очкам?") == {
        "metric": "budget_consistency",  # обе оси названы -> составная метрика
        "position": None,
        "max_price": None,
        "min_price": None,
        "limit": 8,
        "gw_from": None,  # окно туров не названо -> весь сезон
        "gw_to": None,
        "last_n_gws": None,
    }
    assert ranking_params(None, "cheapest players by points")["metric"] == "points_per_million"
    assert ranking_params(None, "most reliable defenders")["metric"] == "consistency"
    # LLM выбрал одну ось, а текст называет обе -> budget_consistency; явный total_points уважаем
    both = "cheap but reliable midfielders"
    assert ranking_params({"metric": "consistency"}, both)["metric"] == "budget_consistency"
    assert ranking_params({"metric": "points_per_million"}, both)["metric"] == "budget_consistency"
    assert ranking_params({"metric": "total_points"}, both)["metric"] == "total_points"
    p = ranking_params(None, "какие защитники дешевле 4.5 играют стабильно?")
    assert (p["metric"], p["position"], p["max_price"]) == ("consistency", "DEF", 4.5)
    p = ranking_params(None, "top 5 forwards by points")
    assert (p["metric"], p["position"], p["limit"]) == ("total_points", "FWD", 5)
    p = ranking_params(None, "кто в форме среди вратарей до 5.0?")
    assert (p["metric"], p["position"], p["max_price"]) == ("form", "GKP", 5.0)
    p = ranking_params(None, "5 лучших полузащитников дороже 8.0")
    assert (p["position"], p["min_price"], p["limit"]) == ("MID", 8.0, 5)
    # поля роутера главнее текста; мусор отбрасывается и достраивается по тексту
    draft = RankingDraft(metric="form", position="MID", max_price=99.0, min_price=None, limit=0)
    p = ranking_params(draft, "cheap defenders under 4.5, top 3")
    assert p == {
        "metric": "form",
        "position": "MID",
        "max_price": 4.5,  # 99.0 — не цена FPL -> из текста
        "min_price": None,
        "limit": 3,
        "gw_from": None,
        "gw_to": None,
        "last_n_gws": None,
    }
    assert ranking_params({"limit": 999}, "players")["limit"] == 25
    assert ranking_params({"min_price": 9.0, "max_price": 5.0}, "x")["min_price"] is None
    assert ranking_params({"position": "def", "metric": "xg"}, "x")["position"] == "DEF"
    assert ranking_params({"metric": "xg"}, "x")["metric"] == "points_per_million"  # дефолт


def _window(p: dict) -> tuple:
    return (p["gw_from"], p["gw_to"], p["last_n_gws"])


def test_ranking_params_gameweek_window_regex_ru_en():
    # явные номера туров: «gw4 и gw5» -> 4..5, один тур -> from == to
    p = ranking_params(None, "посмотри кто был лучшим в gw4 и gw5")
    assert _window(p) == (4, 5, None)
    assert p["metric"] == "total_points"  # «лучший» за окно без слов о цене/стабильности = очки
    p = ranking_params(None, "top 5 defenders in GW5")
    assert _window(p) == (5, 5, None) and (p["position"], p["limit"]) == ("DEF", 5)
    assert _window(ranking_params(None, "лучшие защитники в GW5")) == (5, 5, None)
    assert _window(ranking_params(None, "кто был лучшим в 5 туре")) == (5, 5, None)
    assert _window(ranking_params(None, "best players in gameweek 4")) == (4, 4, None)
    # диапазоны
    assert _window(ranking_params(None, "GW3–GW5 top scorers")) == (3, 5, None)
    assert _window(ranking_params(None, "с 3 по 5 тур лучшие защитники")) == (3, 5, None)
    # «последние N» остаются N — последний завершённый тур знает инструмент, не regex
    p = ranking_params(None, "top scorers in the last 3 gameweeks")
    assert _window(p) == (None, None, 3) and p["metric"] == "total_points"
    assert _window(ranking_params(None, "кто лучший за последние 3 тура")) == (None, None, 3)
    assert _window(ranking_params(None, "лучшие за 3 тура")) == (None, None, 3)
    assert _window(ranking_params(None, "в прошлом туре кто набрал больше всего очков")) == (
        None,
        None,
        1,
    )
    assert _window(ranking_params(None, "best in the last gameweek")) == (None, None, 1)
    # «3 тура» / «5 туров» — период, не номер тура; цены и top-N не путаются с турами
    assert _window(ranking_params(None, "cheap defenders under 4.5, top 3")) == (None, None, None)
    assert _window(ranking_params(None, "5 лучших полузащитников дороже 8.0")) == (
        None,
        None,
        None,
    )
    assert _window(ranking_params(None, "кто в форме среди вратарей до 5.0?")) == (None, None, None)
    # метрика из текста главнее дефолта окна; окно с ценой/позицией
    p = ranking_params(None, "самые стабильные защитники в GW5 дешевле 5.0")
    assert (p["metric"], p["position"], p["max_price"], _window(p)) == (
        "consistency",
        "DEF",
        5.0,
        (5, 5, None),
    )
    # поля роутера главнее текста; одно число -> один тур; перепутанные границы; мусор -> текст
    assert _window(ranking_params({"gw_from": 5, "gw_to": 4}, "x")) == (4, 5, None)
    assert _window(ranking_params({"gw_to": 3}, "x")) == (3, 3, None)
    assert _window(ranking_params({"last_n_gws": 2}, "x")) == (None, None, 2)
    assert _window(ranking_params({"gw_from": 2, "last_n_gws": 3}, "x")) == (2, 2, None)
    assert _window(ranking_params({"gw_from": 99}, "кто был лучшим в gw4 и gw5")) == (4, 5, None)
    assert _window(ranking_params({"last_n_gws": True}, "x")) == (None, None, None)
    draft = RankingDraft(
        metric=None,
        position=None,
        max_price=None,
        min_price=None,
        limit=None,
        gw_from=4,
        gw_to=5,
    )
    assert _window(ranking_params(draft, "кто был лучшим в gw4 и gw5")) == (4, 5, None)


def test_player_ranking_window_changes_order_and_labels(monkeypatch):
    # сезон по очкам: Haaland 33 > Guéhi 25 > Saka 23 > Palmer 19; в GW2: Palmer 12 > Saka 7 >
    # Guéhi 6 > Haaland 2 — окно считается только по истории, порядок другой
    tools = FakeTools()
    state = make_agent(tools, FakeRouter(router_output("player_ranking"))).run(
        "кто набрал больше всего очков в GW2?", manager_id=1
    )
    inp = tools.called("rank_players")[0]
    assert (inp.metric, inp.gw_from, inp.gw_to, inp.last_n_gws) == ("total_points", 2, 2, None)
    pr = state["facts"]["player_ranking"]
    assert [r["player"] for r in pr["rows"]] == ["Palmer", "Saka", "Guéhi", "Haaland"]
    top = pr["rows"][0]
    assert top["total_points"] == 12 and top["season_total_points"] == 19
    assert top["minutes"] == 90 and top["points_per_million"] == 1.24  # 12 / 9.7
    assert top["gws_played"] == 1 and top["gws_5plus"] == 1 and top["points_by_gw"] == {"GW2": 12}
    assert pr["window_label"] == "GW2" and (pr["gw_from"], pr["gw_to"]) == (2, 2)
    assert pr["filters"]["min_minutes"] == 45 and pr["filters"]["window"] == "GW2"
    assert "cover GW2 ONLY" in pr["points_definition"]
    assert state["facts"]["headline"].startswith(
        "Top 4 of 4 players by fake total_points, GW2 [players with at least 45 minutes played]: "
        "1. Palmer (CHE, MID, £9.7, 12 pts, 1.24 pts/£m, >=5 pts in 1 of 1 GWs)"
    )
    assert any(
        "points scored in GW2 only (past points, not a forecast)" in c for c in state["caveats"]
    )
    note = next(t for t in state["tool_log"] if t["tool"] == "rank_players")["note"]
    assert note.startswith("total_points, GW2: top 4 of 4 candidates")

    # «последние 2 тура» -> инструмент разрешает GW3–GW4 относительно последнего завершённого
    tools = FakeTools()
    state = make_agent(tools, FakeRouter(router_output("player_ranking"))).run(
        "top scorers in the last 2 gameweeks"
    )
    inp = tools.called("rank_players")[0]
    assert (inp.gw_from, inp.gw_to, inp.last_n_gws) == (None, None, 2)
    pr = state["facts"]["player_ranking"]
    assert pr["window_label"] == "GW3–GW4" and (pr["gw_from"], pr["gw_to"]) == (3, 4)
    assert [(r["player"], r["total_points"]) for r in pr["rows"]] == [
        ("Haaland", 18),
        ("Guéhi", 13),
        ("Saka", 11),
        ("Palmer", 5),
    ]
    assert pr["filters"]["min_minutes"] == 90  # 45 × 2 тура окна
    assert "GW3–GW4" in state["facts"]["headline"]

    # роутер v2 отдал окно полем — regex не нужен
    tools = FakeTools()
    state = make_agent(
        tools, ranking_router_v2(metric="total_points", gw_from=1, gw_to=2, limit=2)
    ).run("who was the best over the opening two rounds?")
    inp = tools.called("rank_players")[0]
    assert (inp.gw_from, inp.gw_to) == (1, 2)
    assert [(r["player"], r["total_points"]) for r in state["facts"]["player_ranking"]["rows"]] == [
        ("Haaland", 15),
        ("Palmer", 14),
    ]


def test_player_ranking_window_without_history_is_an_explicit_caveat_not_season_totals():
    tools = FakeTools(history=False)
    state = make_agent(tools, FakeRouter(router_output("player_ranking"))).run(
        "кто был лучшим в gw3 и gw4?", manager_id=1
    )
    assert "player_ranking" not in state["facts"]  # никакого отката к сезонным итогам
    failed = next(t for t in state["tool_log"] if t["tool"] == "rank_players")
    assert failed["ok"] is False and failed["note"].startswith("HistoryUnavailable:")
    assert "GW3–GW4" in failed["note"] and "not a substitute" in failed["note"]
    assert any(
        "ranking unavailable" in c.lower() and "needs player_gw_history" in c
        for c in state["caveats"]
    )
    assert state["facts"]["headline"] == "Player ranking unavailable — no numbers were computed"
    # без окна тот же фейк отвечает по сезонным итогам (история для окна не требуется)
    tools = FakeTools(history=False)
    state = make_agent(tools, FakeRouter(router_output("player_ranking"))).run(
        "cheapest consistent players?"
    )
    assert state["facts"]["player_ranking"]["window_label"] == "season to date"

    # окно на ещё не сыгранный тур (GW5 — предстоящий) -> понятная ошибка, а не пустая таблица
    tools = FakeTools()
    state = make_agent(tools, FakeRouter(router_output("player_ranking"))).run(
        "top 5 defenders in GW5"
    )
    failed = next(t for t in state["tool_log"] if t["tool"] == "rank_players")
    assert failed["ok"] is False and "GW5 has not been played yet" in failed["note"]
    assert any("GW5 has not been played yet" in c for c in state["caveats"])


def test_resolve_window_and_label_helpers():
    assert resolve_window(None, None, None, upcoming_gw=6, last_finished_gw=5) is None
    assert resolve_window(4, 5, None, upcoming_gw=6, last_finished_gw=5) == (4, 5)
    assert resolve_window(5, 4, None, upcoming_gw=6, last_finished_gw=5) == (4, 5)
    assert resolve_window(None, 5, None, upcoming_gw=6, last_finished_gw=5) == (5, 5)
    assert resolve_window(4, 5, 3, upcoming_gw=6, last_finished_gw=5) == (4, 5)  # явное главнее
    assert resolve_window(None, None, 3, upcoming_gw=6, last_finished_gw=5) == (3, 5)
    assert resolve_window(None, None, 10, upcoming_gw=6, last_finished_gw=5) == (1, 5)  # клип
    # текущий тур идёт (GW6 не завершён, предстоящий GW7): «последние 2» = GW5–GW6? нет — от
    # последнего завершённого (GW5): GW4–GW5
    assert resolve_window(None, None, 2, upcoming_gw=7, last_finished_gw=5) == (4, 5)
    with pytest.raises(ValueError, match="GW6 has not been played yet"):
        resolve_window(6, None, None, upcoming_gw=6, last_finished_gw=5)
    with pytest.raises(ValueError, match="outside GW1"):
        resolve_window(0, 2, None, upcoming_gw=6, last_finished_gw=5)
    with pytest.raises(ValueError, match="no gameweek has finished"):
        resolve_window(None, None, 2, upcoming_gw=1, last_finished_gw=None)
    with pytest.raises(ValueError, match="last_n_gws must be >= 1"):
        resolve_window(None, None, 0, upcoming_gw=6, last_finished_gw=5)
    assert window_label(4, 5) == "GW4–GW5" and window_label(5, 5) == "GW5"
    assert window_label(None, None) == "season to date"


class _BootstrapClient:
    def bootstrap(self):
        return BS

    def close(self):
        pass


def test_live_tools_rank_players_window_uses_history_only(monkeypatch):
    tools = LiveTools(client=_BootstrapClient(), now=NOW)  # type: ignore[arg-type]
    monkeypatch.setattr(tools, "_history_by_player", lambda gw: (RANK_HISTORY, True))
    out = tools.rank_players(RankPlayersInput(gw=5, metric="total_points", gw_from=2, gw_to=2))
    # bootstrap-итоги фейковых игроков нулевые — окно берёт очки/минуты только из истории
    assert out.window_label == "GW2" and (out.gw_from, out.gw_to) == (2, 2)
    assert [(r.name, r.total_points, r.minutes) for r in out.rows] == [
        ("Palmer", 12, 90),
        ("Saka", 7, 88),
        ("Guéhi", 6, 90),
        ("Haaland", 2, 90),
    ]  # Wan-Bissaka: 12 минут < 45 -> отфильтрован
    assert out.rows[0].season_total_points == 0 and out.rows[0].points_per_million == 1.24
    assert out.rows[0].gws_in_history == 1 and out.rows[0].points_by_gw == {"GW2": 12}
    assert out.filters["min_minutes"] == 45 and out.filters["gws_in_window"] == 1
    assert out.filters["window"] == "GW2" and out.metric_label == "total points"
    # «последние 2 тура» от последнего завершённого GW4 -> GW3–GW4, порог 90 минут
    out = tools.rank_players(RankPlayersInput(gw=5, metric="total_points", last_n_gws=2))
    assert out.window_label == "GW3–GW4" and out.filters["min_minutes"] == 90
    assert [r.name for r in out.rows] == ["Haaland", "Guéhi", "Saka", "Palmer"]
    # form при окне — с пометкой, что это 30-дневная форма FPL, а не окно
    out = tools.rank_players(RankPlayersInput(gw=5, metric="form", gw_from=3, gw_to=4))
    assert any("ignores the gameweek window" in n for n in out.notes)

    # история недоступна: окно -> HistoryUnavailable; без окна — прежняя деградация с пометкой
    monkeypatch.setattr(tools, "_history_by_player", lambda gw: ({}, False))
    with pytest.raises(HistoryUnavailable, match="database is unavailable"):
        tools.rank_players(RankPlayersInput(gw=5, gw_from=3, gw_to=4))
    out = tools.rank_players(RankPlayersInput(gw=5, min_minutes=0))
    assert out.history_available is False and out.window_label == "season to date"
    assert any("bootstrap totals only" in n for n in out.notes)
    # история есть, но не покрывает окно (синхронизация отстала) -> явная ошибка, не частичные суммы
    partial = {pid: {r: v for r, v in h.items() if r <= 3} for pid, h in RANK_HISTORY.items()}
    monkeypatch.setattr(tools, "_history_by_player", lambda gw: (partial, True))
    with pytest.raises(HistoryUnavailable, match="GW4, but player_gw_history covers GW1–GW3"):
        tools.rank_players(RankPlayersInput(gw=5, gw_from=3, gw_to=4))
    with pytest.raises(ValueError, match="GW5 has not been played yet"):
        tools.rank_players(RankPlayersInput(gw=5, gw_from=5, gw_to=5))


def test_load_context_falls_back_to_bootstrap_gameweek_when_squad_db_is_down():
    class DbDownTools(FakeTools):
        def get_gameweek_context(self, inp):
            if inp.manager_id is not None:
                self._rec("get_gameweek_context", inp)
                raise RuntimeError("OperationalError: connection refused (db down)")
            return super().get_gameweek_context(inp)

    tools = DbDownTools()
    state = make_agent(tools, FakeRouter(router_output("player_ranking"))).run(
        "cheapest consistent players?", manager_id=1
    )
    ctx_calls = [t for t in state["tool_log"] if t["tool"] == "get_gameweek_context"]
    assert [c["ok"] for c in ctx_calls] == [False, True]
    assert ctx_calls[1]["note"].startswith("fallback without squad")
    assert state["gw"] == 5 and state["current_gw"] == 4 and state["squad_summary"] is None
    assert "squad unavailable (RuntimeError: OperationalError" in state["squad_note"]
    # оговорка без сырого текста исключения (имя класса ошибки не должно попасть в ответ)
    assert any(c.startswith("Squad unavailable (database error)") for c in state["caveats"])
    assert not any("RuntimeError" in c for c in state["caveats"])
    # рейтинг посчитан для правильного тура (не GW1) и без флагов состава
    inp = tools.called("rank_players")[0]
    assert inp.gw == 5 and inp.squad_ids == []
    assert state["facts"]["gw"] == 5


def test_consistency_stats_and_rank_player_rows_on_synthetic_history():
    guehi = consistency_stats(RANK_HISTORY[388], RANK_ROUNDS)
    assert guehi["gws_played"] == 4 and guehi["gws_in_history"] == 4
    assert guehi["gws_5plus"] == 4 and guehi["share_5plus_pct"] == 100
    assert guehi["gws_2plus"] == 4 and guehi["std_points"] == 1.09  # mean 6.25, population std
    assert (guehi["min_points"], guehi["max_points"]) == (5, 8)
    assert guehi["points_by_gw"] == {"GW1": 6, "GW2": 6, "GW3": 8, "GW4": 5}
    haaland = consistency_stats(RANK_HISTORY[411], RANK_ROUNDS)
    assert haaland["gws_5plus"] == 2 and haaland["share_5plus_pct"] == 50
    assert haaland["gws_2plus"] == 3 and haaland["std_points"] == 6.91
    awb = consistency_stats(RANK_HISTORY[611], RANK_ROUNDS)  # только туры с минутами
    assert awb["gws_played"] == 1 and awb["points_by_gw"] == {"GW2": 1} and awb["min_points"] == 1
    empty = consistency_stats({}, RANK_ROUNDS)
    assert empty["gws_played"] == 0 and empty["std_points"] is None and empty["points_by_gw"] == {}

    rows = rank_rows_from_history()  # фильтр минут отсёк Wan-Bissaka (12 минут)
    assert sorted(r.name for r in rows) == ["Guéhi", "Haaland", "Palmer", "Saka"]
    by_ppm = rank_player_rows(rows, "points_per_million")
    assert [r.name for r in by_ppm] == ["Guéhi", "Saka", "Haaland", "Palmer"]
    assert [r.rank for r in by_ppm] == [1, 2, 3, 4]
    assert by_ppm[0].points_per_million == 4.17 and by_ppm[2].points_per_million == 2.13
    assert [r.name for r in rank_player_rows(rows, "total_points")] == [
        "Haaland",
        "Guéhi",
        "Saka",
        "Palmer",
    ]
    # стабильность: доля туров с >= 5, затем меньший разброс — Guéhi (4/4) > Saka (3/4) >
    # Haaland (2/4) > Palmer (1/4)
    assert [r.name for r in rank_player_rows(rows, "consistency")] == [
        "Guéhi",
        "Saka",
        "Haaland",
        "Palmer",
    ]
    assert rank_player_rows(rows, "form")[0].name == "Haaland"
    with pytest.raises(ValueError, match="unknown ranking metric"):
        rank_player_rows(rows, "xg")
    assert set(RANKING_METRICS) == {
        "points_per_million",
        "total_points",
        "consistency",
        "budget_consistency",
        "form",
    }

    # равная доля туров с >= 5: consistency решает меньший разброс, budget_consistency — цена;
    # пропущенный матч команды (0 минут) опускает игрока ниже при обеих метриках
    tie = {
        4: {1: (6, 90), 2: (6, 90), 3: (6, 90), 4: (6, 90)},  # Gabriel £8.0, std 0
        388: RANK_HISTORY[388],  # Guéhi £6.0, std 1.09
        411: {1: (6, 90), 2: (0, 0), 3: (6, 90), 4: (6, 90)},  # Haaland: 3/3, но 1 DNP
    }
    tie_rows = rank_rows_from_history(tie)
    assert {r.name: r.gws_dnp for r in tie_rows} == {"Gabriel": 0, "Guéhi": 0, "Haaland": 1}
    assert all(r.share_5plus_pct == 100 for r in tie_rows)
    assert [r.name for r in rank_player_rows(tie_rows, "consistency")] == [
        "Gabriel",
        "Guéhi",
        "Haaland",
    ]
    assert [r.name for r in rank_player_rows(tie_rows, "budget_consistency")] == [
        "Guéhi",
        "Gabriel",
        "Haaland",
    ]


def test_player_ranking_tool_failure_becomes_caveat_not_crash():
    tools = FakeTools()

    def boom(inp):
        raise RuntimeError("db down")

    tools.rank_players = boom  # type: ignore[method-assign]
    state = make_agent(tools, FakeRouter(router_output("player_ranking"))).run(
        "cheapest consistent players?", manager_id=1
    )
    assert "player_ranking" not in state["facts"]
    failed = [t for t in state["tool_log"] if t["tool"] == "rank_players"]
    assert failed and failed[0]["ok"] is False and "db down" in failed[0]["note"]
    assert any("ranking unavailable" in c.lower() for c in state["caveats"])
    assert state["facts"]["headline"] == "Player ranking unavailable — no numbers were computed"
    assert state["answer"]  # объяснитель всё равно отвечает (фейк), граф не падает


def test_router_prompt_v2_describes_player_ranking_and_narrows_strategy_question():
    router = load_agent_prompt("router.system", "v2")
    assert "player_ranking" in router and "ranking —" in router
    assert "самый дешёвый но стабильный игрок" in router
    assert "NOT for questions that need numbers about players" in router
    # окно туров: примеры RU/EN и запрет пересчитывать «последние N» в номера туров
    for example in (
        "кто был лучшим в gw4 и gw5",
        "top scorers in the last 3 gameweeks",
        "лучшие защитники в GW5",
    ):
        assert example in router
    assert "gw_from / gw_to" in router and "last_n_gws" in router
    assert 'Never turn "last N gameweeks" into gw_from / gw_to yourself' in router
    explain = load_agent_prompt("explain.system", "v2")
    assert "PLAYER RANKINGS (INTENT = player_ranking)" in explain
    assert (
        "| # | Player | Team | Position | Price | Points | Pts/£m | GWs ≥5 pts | Std | Form |"
        in (explain)
    )
    # объяснитель: окно и фильтры дословно из facts, «не менее N минут», а не «менее»
    assert "window_label" in explain and "never widen or narrow the window" in explain
    assert "AT LEAST that many minutes" in explain and 'never "less than"' in explain
    assert "ranking" in RouterOutputV2.model_fields and "ranking" not in (
        agent_llm.RouterOutput.model_fields
    )
    for field in ("gw_from", "gw_to", "last_n_gws"):
        assert field in RankingDraft.model_fields


# ---------- язык ответа ----------


def test_language_detected_from_cyrillic_query_reaches_explain_input():
    tools = FakeTools()
    explain = FakeExplain(["**Вердикт:** Palmer остаётся (4.76 xPts)."])
    agent = make_agent(
        tools, FakeRouter(router_output("transfer", ["Palmer"], sell=["Palmer"])), explain
    )
    state = agent.run("Стоит ли продавать Palmer?", manager_id=1)  # роутер v1 без поля language
    assert state["language"] == "ru"
    assert explain.calls[0].language == "ru"
    assert "LANGUAGE: ru (Russian)" in render_explain_user(explain.calls[0])
    assert state["validation"]["passed"]  # кириллица не считается «неизвестным именем»


def test_router_v2_language_field_wins_over_heuristic():
    out = RouterOutputV2(
        intent="captain",
        player_mentions=[],
        horizon=None,
        scenario=ScenarioDraft(sell=[], buy=[], keep=[], allow_hit=None, use_wildcard=None),
        needs_squad=True,
        reason="fake",
        language="de",
    )
    tools = FakeTools()
    explain = FakeExplain()
    state = make_agent(tools, FakeRouter(out), explain).run(
        "Wen soll ich als Kapitän aufstellen?", manager_id=1
    )
    assert state["language"] == "de" and explain.calls[0].language == "de"
    assert state["horizon"] == 1  # DEFAULT_HORIZON[captain]; horizon=null не сломал дефолт


def test_language_helpers():
    assert detect_language("Стоит ли продавать Palmer?") == "ru"
    assert detect_language("Should I sell Palmer?") == "en"
    assert detect_language("") == "en"
    assert normalize_language("RU", "x") == "ru" and normalize_language("ru-RU", "x") == "ru"
    assert normalize_language("english", "x") == "en"  # первые две буквы
    assert normalize_language(None, "Когда лучше сыграть Wildcard?") == "ru"
    assert normalize_language("?", "Should I") == "en"


def test_russian_horizon_hint_is_accepted():
    tools = FakeTools()
    router = FakeRouter(router_output("plan", horizon=5))
    state = make_agent(tools, router).run("Спланируй трансферы на 5 туров", manager_id=1)
    assert state["horizon"] == 5 and state["language"] == "ru"
    state = make_agent(FakeTools(), FakeRouter(router_output("transfer", horizon=1))).run(
        "Стоит ли продавать Palmer?", manager_id=1
    )
    assert state["horizon"] == 3  # горизонт без явного периода в вопросе не принимается


# ---------- промпты v2 ----------


def test_prompt_versions_v2_overrides_router_and_explain_only():
    # v3: новости кандидатов; v4: чат (история, новые интенты, гибкий формат)
    assert agent_llm.available_prompt_versions() == ["v1", "v2", "v3", "v4"]
    # v4 = чат (история, новые интенты) поверх v3 = v2 + NEWS (tests/test_candidate_news.py)
    assert settings.agent_prompt_version == "v4"
    v1, v2 = load_agent_prompt("explain.system", "v1"), load_agent_prompt("explain.system", "v2")
    assert v1 != v2
    for must in (
        "LANGUAGE",
        "rules_context",
        "[source]",
        "strategy_answer",
        "Δ next GW",
        "[FACTS]",
    ):
        assert must in v2, must
    r1, r2 = load_agent_prompt("router.system", "v1"), load_agent_prompt("router.system", "v2")
    assert r1 != r2 and "language" in r2 and "return null" in r2
    # grader и rules_digest в v2 не переопределены -> берутся из v1
    assert load_agent_prompt("grader.system", "v2") == load_agent_prompt("grader.system", "v1")
    assert load_agent_prompt("rules_digest", "v2") == load_agent_prompt("rules_digest", "v1")
    with pytest.raises(FileNotFoundError):
        load_agent_prompt("nope", "v2")
    assert (
        "language" in RouterOutputV2.model_fields
        and "language" not in agent_llm.RouterOutput.model_fields
    )


def test_explain_request_defaults_and_usage_version():
    req = ExplainRequest(
        query="q",
        intent="captain",
        strategy="balanced",
        as_of="now",
        facts={},
        evidence=[],
        caveats=[],
    )
    assert req.language == "en" and "LANGUAGE: en (English)" in render_explain_user(req)
    assert usage().prompt_version == settings.agent_prompt_version
    assert TOOL_NAMES[-2:] == ("search_strategy_kb", "rank_players") and len(TOOL_NAMES) == 11


def test_explain_user_message_v1_unchanged_and_v2_adds_blocks():
    facts = {
        "rules_context": {
            "about": "hit",
            "excerpts": [
                {
                    "source": "premierleague",
                    "title": "FPL Rules Copilot",
                    "url": "https://pl.example/rules",
                    "official_rules": True,
                    "text": "Each additional transfer will deduct 4 points.",
                }
            ],
        },
        "strategy_answer": {
            "covered": True,
            "answer": "Hits cost 4 points [1].",
            "sources_markdown": "- [1] FPL Rules Copilot (premierleague, rules) — https://pl.example/rules",
            "citations": [{"n": 1}],
        },
    }
    req = ExplainRequest(
        query="q",
        intent="transfer",
        strategy="balanced",
        as_of="now",
        facts=facts,
        evidence=[],
        caveats=["c"],
        language="ru",
    )
    v1 = render_explain_user(req, "v1")
    v2 = render_explain_user(req, "v2")
    assert "LANGUAGE" not in v1 and "RULES CONTEXT" not in v1 and "KB ANSWER" not in v1
    assert v1.startswith("QUESTION: q\nINTENT: transfer\nSTRATEGY PRESET: balanced")
    assert "LANGUAGE: ru (Russian)" in v2 and "translate the section titles too" in v2
    assert "RULES CONTEXT (strategy knowledge base, about: hit)" in v2
    assert "ends with the citation [premierleague]" in v2
    assert '1. cite as [premierleague] — FPL Rules Copilot (official rules): "Each additional' in v2
    assert "KB ANSWER (strategy knowledge base; covered=True)" in v2
    assert "SOURCES BLOCK (verbatim):\n- [1] FPL Rules Copilot" in v2
    assert (
        v2.index("EVIDENCE")
        < v2.index("RULES CONTEXT")
        < v2.index("KB ANSWER")
        < v2.index("CAVEATS")
    )
    # факты (JSON) одинаковы в обеих версиях — A/B сравнивает только промпт и подачу
    assert json.dumps(facts, ensure_ascii=False, indent=1) in v1
    assert json.dumps(facts, ensure_ascii=False, indent=1) in v2
