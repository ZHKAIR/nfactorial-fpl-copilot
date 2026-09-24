"""Детерминированный валидатор ответа (agent/validate.py) и помощники инструментов без БД."""

from __future__ import annotations

from fplcopilot.agent.tools import captain_tag
from fplcopilot.agent.validate import (
    prune_undiscussed_sources,
    strip_bad_citations,
    validate_answer,
)

FACTS = {
    "players": {
        "Palmer": {"team": "CHE", "price": 9.7, "xpts_by_gw": {"GW5": 4.76, "GW6": 4.4}},
        "Saka": {"team": "ARS", "price": 9.5, "xpts_by_gw": {"GW5": 6.2}},
    },
    "transfers": {"routes": [{"out": ["Palmer"], "in": ["Saka"], "gain_horizon": 4.03}]},
}
EVIDENCE = [
    {
        "player": "Palmer",
        "source": "bbc",
        "date": "16.09",
        "url": "https://bbc.example/a",
        "quote": "Palmer trained fully.",
    }
]


def test_numbers_within_rounding_pass_and_fabricated_fail():
    ok = validate_answer("Palmer 4.8 xPts vs Saka 6.20; gain +4.03 for £9.7.", FACTS, EVIDENCE)
    assert ok.passed, ok
    bad = validate_answer("Saka should score 6.31 xPts and Palmer 4.9.", FACTS, EVIDENCE)
    assert bad.unknown_numbers == ["6.31", "4.9"] and not bad.passed
    assert "6.31" in bad.feedback


def test_citations_urls_and_all_caps_are_ignored():
    answer = (
        "**Why**\n- Palmer trained fully [bbc, 16.09].\n\n**Sources**\n- bbc, 16.09, "
        "https://bbc.example/a\n\nFSI 3 for CHE at BRE in GW5."
    )
    res = validate_answer(answer, FACTS, EVIDENCE)
    assert res.passed, res


def test_unknown_player_name_is_flagged_but_known_and_common_words_pass():
    res = validate_answer(
        "Verdict: keep Palmer, ignore Salah. Arsenal look strong; Why not Saka?",
        FACTS,
        EVIDENCE,
        extra_names=["Arsenal"],
    )
    assert res.unknown_names == ["Salah"]
    assert "Salah" in res.feedback


def test_hyphenated_and_accented_names_from_facts_pass():
    facts = {"players": {"Gibbs-White": {"xpts": 6.43}, "Guéhi": {"xpts": 5.94}}}
    res = validate_answer("Gibbs-White (6.43) and Guehi (5.94) both start.", facts)
    assert res.passed, res


def test_captain_tags_by_ownership():
    assert captain_tag(72.6) == "safe"
    assert captain_tag(30.0) == "balanced"
    assert captain_tag(12.5) == "balanced"
    assert captain_tag(9.9) == "differential"


KB_FACTS = {
    "strategy_answer": {
        "answer": "Bank up to 5 [1]. Hits cost 4 points [2].",
        "covered": True,
        "citations": [
            {
                "n": 1,
                "title": "Rules",
                "url": "https://pl.example/rules",
                "source": "premierleague",
            },
            {"n": 2, "title": "Hits", "url": "https://fplwatch.example/hits", "source": "fplwatch"},
        ],
    }
}


def test_kb_references_must_exist_in_facts():
    ok = validate_answer(
        "Bank up to 5 [1]. Hits cost 4 points [2]; both [1][2] and [1, 2].", KB_FACTS
    )
    assert ok.passed and ok.checked_refs == 5 and ok.unknown_refs == []
    bad = validate_answer("Bank up to 5 [1]. Free Hit reverts [3]; see [2, 9].", KB_FACTS)
    assert not bad.passed
    assert bad.unknown_refs == ["[3]", "[2, 9]"] and bad.bad_citations == ["[3]", "[2, 9]"]
    assert "Knowledge-base references" in bad.feedback and "[3]" in bad.feedback
    assert "Placeholder citations" not in bad.feedback  # заглушек нет — только неизвестные [n]
    # без ответа KB в фактах любая [n] неизвестна
    none = validate_answer("Palmer 4.76 xPts [1].", FACTS, EVIDENCE)
    assert none.unknown_refs == ["[1]"] and not none.passed
    assert validate_answer("Palmer 4.76 xPts.", FACTS, EVIDENCE).checked_refs == 0


def test_kb_markers_must_survive_in_the_answer_text():
    only_sources = (
        "Bank up to 5 [premierleague].\n\n**Sources**\n- [1] Rules — https://pl.example/rules"
    )
    res = validate_answer(only_sources, KB_FACTS)
    assert res.missing_refs and not res.passed and "markers [n] are missing" in res.feedback
    kept = validate_answer(
        "Bank up to 5 [1].\n\n**Sources**\n- [1] Rules — https://pl.example/rules", KB_FACTS
    )
    assert kept.passed and not kept.missing_refs
    # непокрытый ответ KB маркеров не требует
    not_covered = {"strategy_answer": {"covered": False, "citations": []}}
    assert not validate_answer("Not covered.", not_covered).missing_refs
    assert not validate_answer("Palmer 4.76 xPts.", FACTS).missing_refs


def test_rules_context_source_citation_and_placeholders():
    facts = {**FACTS, "rules_context": {"excerpts": [{"source": "premierleague", "text": "x"}]}}
    res = validate_answer("Every extra transfer costs 4 points [premierleague].", facts, EVIDENCE)
    assert res.passed, res
    bad = validate_answer("Costs 4 points [source]. Gain +4.03 [FACTS].", facts, EVIDENCE)
    assert bad.bad_citations == ["[source]", "[FACTS]"] and bad.unknown_refs == []
    assert "Placeholder citations" in bad.feedback and "rules_context" in bad.feedback


def test_verdict_minus_four_requires_a_paid_transfer_in_facts():
    free = {
        "question": "Should I take a -4 this week?",
        "transfers": {"allow_hit": True, "routes": [{"hit_cost": 0, "gain_next_gw": 4.3}]},
    }
    bad = validate_answer("**Verdict:** You should take a -4 this week (+4.3 xPts).", free)
    assert bad.hit_mismatch and not bad.passed and "no paid transfer" in bad.feedback
    ok = validate_answer("**Verdict:** No hit needed — free transfers cover it (+4.3 xPts).", free)
    assert ok.passed and not ok.hit_mismatch
    # «-4» ниже вердикта не проверяется (там его может требовать rules_context)
    body = validate_answer(
        "**Verdict:** no hit.\n\n**Why**\n- extra transfers cost -4 points.", free
    )
    assert body.passed
    paid = {"transfers": {"routes": [{"hit_cost": 4, "gain_next_gw": 5.46}]}}
    assert validate_answer("**Verdict:** take the -4 (+5.46).", paid).passed
    plan = {"plan": {"hits_by_gw": {"5": 4}, "moves_by_gw": {"5": ["A -> B Δ+3.1 PAID -4"]}}}
    assert validate_answer("**Verdict:** the plan takes a -4 in GW5.", plan).passed
    decided = {
        "transfers": {"routes": [{"hit_cost": 0}]},
        "user_decision": {"on": {"action": {"cost": 4}}},
    }
    assert validate_answer("**Verdict:** you rejected the -4; free route instead.", decided).passed
    assert validate_answer(
        "**Вердикт:** брать -4 не нужно.", free
    ).hit_mismatch  # русская строка вердикта
    assert not validate_answer("**Verdict:** sd -4.1 only.", free).hit_mismatch  # «-4.1» — не хит


def test_prune_undiscussed_sources_keeps_discussed_rules_and_unknown_lines():
    evidence = [
        {"player": "Palmer", "source": "bbc", "date": "16.09", "url": "https://bbc.example/a"},
        {
            "player": "João Pedro",
            "source": "ffscout",
            "date": "16.09",
            "url": "https://ffs.example/jp",
        },
        {
            "player": "João Pedro",
            "source": "fpl_api",
            "date": "16.09",
            "url": "fpl://player/165/news/x",
        },
    ]
    facts = {
        "rules_context": {
            "excerpts": [{"url": "https://pl.example/rules", "source": "premierleague"}]
        }
    }
    answer = (
        "**Verdict:** keep Palmer.\n\n**Why**\n- Palmer trained fully [bbc, 16.09].\n"
        "- extra transfers cost 4 points [premierleague].\n\n**Sources**\n"
        "- bbc, 16.09, https://bbc.example/a\n"
        "- ffscout, 16.09, https://ffs.example/jp\n"
        "- fpl_api, 16.09, fpl://player/165/news/x\n"
        "- premierleague — Rules, https://pl.example/rules\n"
        "- sky, 15.09, https://sky.example/unknown\n\n**Caveats**\n- data as of now."
    )
    pruned, removed = prune_undiscussed_sources(answer, evidence, facts)
    assert removed == [
        "- ffscout, 16.09, https://ffs.example/jp",
        "- fpl_api, 16.09, fpl://player/165/news/x",
    ]
    assert "https://bbc.example/a" in pruned and "https://pl.example/rules" in pruned
    assert "https://sky.example/unknown" in pruned  # неизвестный URL не трогаем
    assert "ffs.example" not in pruned and pruned.endswith("**Caveats**\n- data as of now.")
    # игрок обсуждается (с диакритикой / без) -> строка остаётся; раздел без Sources не меняется
    ok, removed = prune_undiscussed_sources(
        answer.replace("keep Palmer.", "keep Palmer; Joao Pedro is doubtful."), evidence, facts
    )
    assert removed == [] and ok == answer.replace(
        "keep Palmer.", "keep Palmer; Joao Pedro is doubtful."
    )
    assert prune_undiscussed_sources("**Verdict:** x.", evidence) == ("**Verdict:** x.", [])
    # раздел опустел -> заголовок тоже уходит
    only_jp = "**Verdict:** keep Palmer.\n\n**Sources**\n- ffscout, 16.09, https://ffs.example/jp\n\n**Caveats**\n- c."
    pruned, removed = prune_undiscussed_sources(only_jp, evidence)
    assert len(removed) == 1 and "Sources" not in pruned and pruned.endswith("**Caveats**\n- c.")


def test_strip_bad_citations_removes_placeholders_and_unknown_refs():
    text = "Cap is 5 [1] and [7]; gain +4.03 [FACTS]."
    assert strip_bad_citations(text, ["[7]"]) == "Cap is 5 [1] and ; gain +4.03 ."
