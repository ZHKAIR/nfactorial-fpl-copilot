"""Чат v4: история диалога, новые интенты, фишки в плане, служебные тексты, проверки валидатора.

Без сети, LLM и БД — на фейках tests/test_agent_fakes.py.
"""

from __future__ import annotations

from types import SimpleNamespace

from langgraph.checkpoint.memory import MemorySaver
from test_agent_fakes import BS, NOW, FakeExplain, FakeRouter, FakeTools, router_output

from fplcopilot.agent import chat
from fplcopilot.agent.graph import Agent, Deps, chat_chips, chat_target_gw
from fplcopilot.agent.llm import (
    ChipDraft,
    ExplainRequest,
    RouterOutputV4,
    RouterRequest,
    ScenarioDraftV3,
    render_explain_user,
    router_schema,
)
from fplcopilot.agent.resolve import PlayerResolver
from fplcopilot.agent.validate import captain_mismatch, gw_violations, validate_answer


def make_agent(tools: FakeTools, router: FakeRouter, explain: FakeExplain, version: str = "v4"):
    deps = Deps(
        tools=tools,
        router_llm=router,
        explain_llm=explain,
        resolver=lambda squad: PlayerResolver(BS, squad_ids=squad),
        clock=lambda: NOW,
        extra_known_names=["Arsenal", "Chelsea"],
        prompt_version=version,
    )
    return Agent(deps, MemorySaver())


def v4_output(intent: str, **kw) -> RouterOutputV4:
    # схема v3 не знает интентов v4: базу строим с любым допустимым интентом и заменяем
    base = router_output("transfer", kw.pop("mentions", None), horizon=kw.pop("horizon", None))
    scenario = ScenarioDraftV3(
        sell=kw.pop("sell", []),
        buy=[],
        keep=kw.pop("keep", []),
        allow_hit=kw.pop("allow_hit", None),
        use_wildcard=None,
    )
    return RouterOutputV4(
        **{**base.model_dump(), "intent": intent, "scenario": scenario.model_dump()},
        language=kw.pop("language", "ru"),
        standalone_query=kw.pop("standalone_query", ""),
        target_gw=kw.pop("target_gw", None),
        chips=kw.pop("chips", []),
        team_mentions=kw.pop("team_mentions", []),
    )


# ---------- история ----------


def test_history_from_messages_carries_turn_meta():
    state = {
        "intent": "plan",
        "query": "BB в GW7",
        "players": [{"id": 165, "name": "João Pedro"}],
        "gw": 6,
        "target_gw": 7,
        "chips_asked": [{"chip": "bboost", "gw": 7}],
        "scenario": {"keep": [165], "allow_hit": False},
        "answer": "План на GW6–GW7 …",
        "facts": {"headline": "Plan GW6–GW7"},
    }
    msgs = [{"role": "user", "content": "BB в GW7"}, {"role": "assistant", "state": state}]
    hist = chat.history_from_messages(msgs)
    assert hist[1]["meta"]["keep"] == ["João Pedro"] and hist[1]["meta"]["allow_hit"] is False
    text = chat.render_history(hist)
    assert "user: BB в GW7" in text and "chips=[{'chip': 'bboost', 'gw': 7}]" in text
    assert "computed verdict of that turn: Plan GW6–GW7" in text
    assert chat.render_history([]) == ""


def test_router_gets_history_and_follow_up_uses_standalone_query():
    router = FakeRouter(
        v4_output(
            "player_status",
            mentions=["João Pedro"],
            standalone_query="Сколько очков наберёт João Pedro в GW5?",
        )
    )
    agent = make_agent(FakeTools(), router, FakeExplain(["João Pedro 4.11 xPts."]))
    history = [
        {"role": "user", "content": "Is João Pedro fit?"},
        {
            "role": "assistant",
            "content": "João Pedro is doubtful.",
            "meta": {"intent": "player_status"},
        },
    ]
    state = agent.run("а он сколько наберёт?", manager_id=1, history=history)
    assert "Is João Pedro fit?" in router.calls[0].history
    assert state["standalone_query"].startswith("Сколько очков")
    assert [p["name"] for p in state["players"]] == ["João Pedro"]
    assert state["turn_meta"]["players"] == ["João Pedro"]


def test_router_v4_schema_only_from_v4():
    assert router_schema("v4") is RouterOutputV4
    assert router_schema("v3") is not RouterOutputV4


def test_explain_user_v4_includes_history_and_own_words():
    req = ExplainRequest(
        query="Состав на GW7?",
        original_query="а на gw7?",
        history="user: кого капитанить",
        intent="lineup",
        strategy="balanced",
        as_of="now",
        facts={},
        evidence=[],
        caveats=[],
        language="ru",
    )
    v4 = render_explain_user(req, "v4")
    assert "USER'S OWN WORDS: а на gw7?" in v4 and "CONVERSATION SO FAR" in v4
    assert "CONVERSATION SO FAR" not in render_explain_user(req, "v3")


# ---------- маршрутизация и граф ----------


def test_target_gw_and_chips_normalised():
    assert (
        chat_target_gw(7, 6) == 7
        and chat_target_gw(3, 6) is None
        and chat_target_gw(None, 6) is None
    )
    chips = chat_chips([ChipDraft(chip="bboost", gw=7), {"chip": "TC", "gw": 2}], 6)
    assert chips == [{"chip": "bboost", "gw": 7}, {"chip": "3xc", "gw": None}]


def test_lineup_with_bench_boost_becomes_plan_with_chip_no_hits_and_keep():
    tools = FakeTools()
    router = FakeRouter(
        v4_output(
            "lineup",
            mentions=["João Pedro"],
            keep=["João Pedro"],
            allow_hit=False,
            target_gw=7,
            chips=[ChipDraft(chip="bboost", gw=7)],
        )
    )
    state = make_agent(tools, router, FakeExplain(["План без хитов."])).run(
        "в gw7 буду делать bench boost, João Pedro не убирать, без хитов", manager_id=1
    )
    assert state["intent"] == "plan"
    plan_inp = next(inp for name, inp in tools.calls if name == "build_gameweek_plan")
    assert plan_inp.allow_hits is False and plan_inp.keep == [165]
    assert [(c.gw, c.chip) for c in plan_inp.chips] == [(7, "bboost")]
    assert plan_inp.horizon == 3  # GW5 (ближайший у фейка) .. GW7


def test_keep_of_player_not_in_squad_is_dropped_with_caveat():
    router = FakeRouter(v4_output("plan", mentions=["Saka"], keep=["Saka"]))
    state = make_agent(FakeTools(), router, FakeExplain(["ok"])).run(
        "не продавать Saka", manager_id=1
    )
    assert state["scenario"]["keep"] == []
    # оговорка на языке вопроса; игрок не из состава — не тема ответа и не «игрок состава»
    assert any("Saka нет в вашем составе" in c for c in state["caveats"])
    assert "Saka" not in [p["name"] for p in state["players"]]
    assert state["facts"]["not_in_your_squad"] == ["Saka"]


def test_sell_of_player_not_in_squad_is_dropped_too():
    router = FakeRouter(v4_output("transfer", mentions=["Saka"], sell=["Saka"], language="en"))
    state = make_agent(FakeTools(), router, FakeExplain(["ok"])).run("sell Saka?", manager_id=1)
    assert state["scenario"]["sell"] == []
    assert any("Saka is not in your squad — the 'sell Saka'" in c for c in state["caveats"])


def test_not_in_squad_claims_flagged_only_without_negation():
    from fplcopilot.agent.validate import not_in_squad_lines, strip_lines

    facts = {"not_in_your_squad": ["João Pedro"]}
    answer = "Состав на GW7:\n- João Pedro (CHE, FWD)\n\nJoão Pedro нет в вашем составе — условие не применено."
    names, lines = not_in_squad_lines(answer, facts)
    assert names == ["João Pedro"] and lines == [1]
    assert "(CHE, FWD)" not in strip_lines(answer, lines, names)
    prose = "Bench Boost в GW7 недоступен. Сохраним João Pedro в составе. План ниже."
    assert strip_lines(prose, [0], ["João Pedro"]) == "Bench Boost в GW7 недоступен. План ниже."
    assert not_in_squad_lines("João Pedro is not in your squad.", facts) == ([], [])
    assert not_in_squad_lines("Любой текст про Saka.", {}) == ([], [])


def test_gw_review_facts_and_headline():
    from fplcopilot.agent.tools import GWReviewOut, GWReviewRow

    rows = [
        GWReviewRow(
            name="Haaland",
            team="MCI",
            position="FWD",
            role="C",
            multiplier=2,
            points=0,
            counted_points=0,
            minutes=90,
            xpts_forecast=7.13,
            diff_vs_forecast=-7.13,
        ),
        GWReviewRow(
            name="Groß",
            team="BHA",
            position="MID",
            role="XI",
            multiplier=1,
            points=14,
            counted_points=14,
            minutes=90,
            xpts_forecast=3.94,
            diff_vs_forecast=10.06,
        ),
        GWReviewRow(
            name="Diop",
            team="IPS",
            position="DEF",
            role="bench",
            multiplier=0,
            points=4,
            counted_points=0,
            minutes=90,
            xpts_forecast=2.31,
            diff_vs_forecast=1.69,
        ),
    ]
    out = GWReviewOut(
        gw=5,
        finished=True,
        manager_points=82,
        average_points=48,
        points_on_bench=4,
        rows=rows,
        captain="Haaland",
        captain_points=0,
        best_starter="Groß",
        best_starter_points=14,
        best_bench="Diop",
        best_bench_points=4,
        forecast_available=True,
        history_available=True,
    )
    facts = {"gw": 6, "gw_review": chat.gw_review_facts(out)}
    assert facts["gw_review"]["biggest_shortfalls_vs_forecast"] == [
        "Haaland: 0 pts vs forecast 7.13 (-7.13)"
    ]
    head = chat.headline_v4("gw_review", facts)
    assert head.startswith("Review of GW5 (facts): you scored 82 pts vs average 48")
    empty = chat.gw_review_facts(GWReviewOut(gw=5, finished=True, notes=["no history"]))
    assert empty["available"] is False
    assert "nothing is invented" in chat.headline_v4("gw_review", {"gw_review": empty})


def test_general_fpl_marks_prices_as_not_predicted():
    agent = make_agent(
        FakeTools(), FakeRouter(v4_output("general_fpl")), FakeExplain(["Цены не прогнозируются."])
    )
    state = agent.run("кто подорожает сегодня ночью?", manager_id=1)
    assert state["facts"]["price_changes"]["price_change_prediction_available"] is False


def test_unknown_intent_is_general_fpl_in_v4_and_off_topic_before():
    out = v4_output("general_fpl")
    agent = make_agent(FakeTools(), FakeRouter(out), FakeExplain(["Дедлайн — в фактах."]))
    state = agent.run("когда дедлайн?", manager_id=1)
    assert state["intent"] == "general_fpl" and "gameweek" in state["facts"]
    assert state["facts"]["what_i_can_compute"]


def test_refusal_in_question_language():
    ru = make_agent(FakeTools(), FakeRouter(v4_output("off_topic")), FakeExplain()).run(
        "какая погода в Лондоне?", manager_id=1
    )
    assert ru["answer"].startswith("Я помогаю только с Fantasy Premier League")
    bet = make_agent(FakeTools(), FakeRouter(v4_output("betting")), FakeExplain()).run(
        "какой кэф на Арсенал?", manager_id=1
    )
    assert "Советов по ставкам я не даю" in bet["answer"]
    assert chat.refusal_text("betting", "en").startswith("I don't give betting advice")


def test_explain_fallback_is_localised_without_json():
    text = chat.explain_fallback_text("Best XI GW6: 57.6 xPts", ["squad = picks of GW5"], "ru")
    assert "Модель объяснения сейчас недоступна" in text and "```" not in text
    assert "Best XI GW6" in text


def test_resolve_teams_english_russian_and_declined():
    teams = [
        SimpleNamespace(id=1, short_name="ARS", name="Arsenal"),
        SimpleNamespace(id=15, short_name="MCI", name="Man City"),
        SimpleNamespace(id=19, short_name="TOT", name="Spurs"),
    ]
    ids, missing = chat.resolve_teams(["Arsenal", "Ман Сити", "Шпоры", "Барселона"], teams)
    assert ids == [1, 15, 19] and missing == ["Барселона"]
    ids, _ = chat.resolve_teams(["Арсенала"], teams)
    assert ids == [1]


# ---------- валидатор ----------


FACTS = {
    "question": "какой состав на gw7?",
    "intent": "lineup",
    "gw": 6,
    "best_xi": {"gw": 6, "captain": "Saka"},
}


def test_gw_relabel_is_flagged_but_honest_answer_passes():
    bad = "Рекомендуемый состав на GW7: 3-4-3 с капитаном Saka."
    assert gw_violations(bad, FACTS) == ([], ["GW7"])
    ok = "Посчитан состав на GW6 (GW7 модель пока не считает): капитан Saka."
    assert gw_violations(ok, FACTS) == ([], [])
    assert gw_violations("В GW9 играйте Saka.", FACTS) == (["GW9"], [])
    assert gw_violations("Состав на 6 тур.", FACTS) == ([], [])


def test_gw_ranges_and_by_gw_keys_count_as_computed():
    facts = {"gw": 6, "plan": {"gws": "GW6–GW8", "moves_by_gw": {"6": [], "7": [], "8": []}}}
    assert gw_violations("В GW7 продайте X; план GW6–GW8.", facts) == ([], [])


def test_invented_rating_and_captain_mismatch():
    res = validate_answer("Ваша команда — 6 из 10.", {"gw": 6, "squad_review": {}})
    assert not res.passed and "6 из 10" in res.unknown_numbers
    facts = {
        "intent": "captain",
        "gw": 6,
        "best_xi": {"captain": "Saka"},
        "captain_options": [{"name": "Saka"}, {"name": "Haaland"}],
    }
    assert captain_mismatch("Капитаном стоит выбрать Haaland.", facts) == "Saka"
    assert captain_mismatch("Капитан — Saka, вице — Haaland.", facts) is None
    assert (
        captain_mismatch("Haaland сыграет, капитан — Saka.", {**facts, "intent": "lineup"}) is None
    )


def test_router_request_history_field_defaults_empty():
    assert (
        RouterRequest(
            query="q", as_of="now", gw=6, deadline=None, has_manager=True, strategy="balanced"
        ).history
        == ""
    )


def test_bench_mismatch_flags_benching_a_starter():
    from fplcopilot.agent.validate import bench_mismatch

    facts = {
        "intent": "lineup",
        "best_xi": {
            "starters": ["Palmer (CHE, MID) 4.57 xPts", "Saka (ARS, MID) 7.17 xPts"],
            "bench": ["Slater (MID) 2.82", "O'Shea (DEF) 3.23"],
        },
    }
    assert bench_mismatch("На скамейку стоит посадить Palmer.", facts) == "Palmer"
    assert bench_mismatch("На скамейке: Slater и O'Shea, Palmer играет.", facts) is None
    assert bench_mismatch("Капитан — Saka.", facts) is None


def test_gw_count_is_not_a_gameweek_number():
    facts = {"question": "лучший на ближайшие 3 тура", "gw": 6, "plan": {"gws": "GW6–GW8"}}
    assert gw_violations("За 3 тура наберёт 9.75; на 5 туров — план.", facts) == ([], [])
    assert gw_violations("В 9 туре и в туре 10 — пауза.", facts) == (["GW9", "GW10"], [])
    assert gw_violations("Во 2-м туре он забил.", facts) == (["GW2"], [])


def test_strip_ratings_and_fix_name_typos():
    from fplcopilot.agent.validate import fix_name_typos, strip_ratings

    text = "Your team has issues. I would rate it around 5 out of 10. Sell Palmer."
    assert strip_ratings(text, ["5 out of 10"]) == "Your team has issues. Sell Palmer."
    fixed, typos = fix_name_typos("Капитан — Caka.", ["Caka"], ["Saka", "Haaland", "Leeds"])
    assert fixed == "Капитан — Saka." and typos == {"Caka": "Saka"}
    same, none = fix_name_typos("Капитан — Xyzq.", ["Xyzq"], ["Saka"])
    assert same == "Капитан — Xyzq." and none == {}


def test_captain_mismatch_ignores_vice_part():
    facts = {
        "intent": "captain",
        "best_xi": {"captain": "Saka"},
        "captain_options": [{"name": "Saka"}, {"name": "Haaland"}],
    }
    assert (
        captain_mismatch("Капитан на GW6 — Haaland (MCI), вице-капитан — Saka (ARS).", facts)
        == "Saka"
    )
    assert captain_mismatch("Капитан на GW6 — Saka (ARS), вице-капитан — Haaland.", facts) is None


def test_langsmith_quota_guard_disables_tracing_once(monkeypatch):
    import logging

    from fplcopilot.agent import tracing

    monkeypatch.setattr(tracing, "_state", {"errors": 0, "disabled_reason": None})
    monkeypatch.setattr(tracing, "TRACING_ENABLED", True)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    guard = tracing._LangSmithFailureGuard()

    def record(msg: str) -> logging.LogRecord:
        return logging.LogRecord("langsmith.client", logging.WARNING, "x", 1, msg, None, None)

    assert guard.filter(record("some other warning")) is True
    assert tracing.tracing_status() == (True, None)
    quota = "LangSmithRateLimitError: 429 ... Monthly unique traces usage limit exceeded"
    assert guard.filter(record(quota)) is False
    enabled, reason = tracing.tracing_status()
    assert enabled is False and "usage limit" in reason
    assert guard.filter(record(quota)) is False  # дальше — тишина
    import os

    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
