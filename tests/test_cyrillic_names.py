"""Кириллические имена игроков: резолвер (agent/resolve.py + agent/translit.py), валидатор ответа
(agent/validate.py) и граф (повтор объяснения, HITL-уточнение) — без сети и БД.

Регрессионный набор — tests/fixtures/cyrillic_mentions.json на реальных именах FPL 2026/27
(tests/fixtures/bootstrap_names.json); до кириллического пути распознавалось 0 из 49 упоминаний.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver
from test_agent_fakes import (
    BS,
    NOW,
    SQUAD_IDS,
    FakeExplain,
    FakeRouter,
    FakeTools,
    router_output,
)

from fplcopilot.agent.graph import Agent, Deps
from fplcopilot.agent.llm import RouterOutputV2, RouterOutputV3, load_agent_prompt, router_schema
from fplcopilot.agent.resolve import PlayerResolver
from fplcopilot.agent.translit import (
    NameIndex,
    case_stems,
    cyr_to_latin,
    cyrillic_keys,
    fix_mixed_script,
    latin_keys,
    phonetic_key,
    similarity,
)
from fplcopilot.agent.validate import (
    cyrillic_player_names,
    fact_player_names,
    restore_latin_names,
    validate_answer,
)
from fplcopilot.data.schemas import Bootstrap, Player, Team

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def real() -> PlayerResolver:
    data = json.loads((FIXTURES / "bootstrap_names.json").read_text(encoding="utf-8"))
    return PlayerResolver(Bootstrap.model_validate(data))


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads((FIXTURES / "cyrillic_mentions.json").read_text(encoding="utf-8"))


# ---------- транслитерация и ключи ----------


def test_case_stems_restore_the_nominative():
    assert "сака" in case_stems("Саки") and "сака" in case_stems("Сакой")
    assert "палмер" in case_stems("Палмера") and "палмер" in case_stems("Палмером")
    assert "вирц" in case_stems("Вирцем") and "габриэл" in case_stems("Габриэля")
    assert "мбая" in case_stems("Мбаю")  # перед «я» основа может кончаться гласной
    assert case_stems("данные") == ["данные"]  # основа «данн-ы» не кончается согласной
    assert case_stems("или") == ["или"]  # короткие слова не склоняются («или» -> «иля» = Ilia)


def test_phonetic_keys_meet_between_scripts():
    assert phonetic_key(cyr_to_latin("Черки")) == phonetic_key("Cherki")
    assert phonetic_key(cyr_to_latin("Гросс")) in latin_keys("Groß")
    assert phonetic_key(cyr_to_latin("Семеньо")) in latin_keys("Semenyo")
    assert phonetic_key(cyr_to_latin("Холанд")) in latin_keys("Haaland")  # aa -> o
    assert phonetic_key(cyr_to_latin("Жоау")) in latin_keys("João")  # ão -> au
    assert phonetic_key(cyr_to_latin("Дейк")) in latin_keys("Dijk")  # ij -> ei
    assert cyrillic_keys("весь") == []  # ключ «ves» короче MIN_KEY


def test_latin_readings_of_diacritics_and_english_vowels():
    assert phonetic_key(cyr_to_latin("Шешко")) in latin_keys("Šeško")
    assert phonetic_key(cyr_to_latin("Хитон")) in latin_keys("Heaton")
    assert phonetic_key(cyr_to_latin("Майну")) in latin_keys("Mainoo")


def test_fix_mixed_script_uses_the_majority_alphabet():
    assert fix_mixed_script("О'Shea и Sаka") == "O'Shea и Saka"  # кириллические О и а
    assert fix_mixed_script("Cака") == "Сака"  # латинская C в русском слове
    assert fix_mixed_script("Saka-ой, xPts, Сака") == "Saka-ой, xPts, Сака"  # не двойники


def test_similarity_guards_short_keys_and_syllables():
    assert similarity("isak", "saka") == 0.0  # короткие — только точно
    assert similarity("troik", "tirik") == 0.0  # «тройка» ~ Tyrick: разное число слогов
    assert similarity("mbemo", "mbeumo") > 0.9


def test_name_index_prefers_exact_and_skips_stopwords():
    idx = NameIndex(["Saka", "Sakyi", "Touré", "Kamada", "Haaland"])
    assert [t for t, _ in idx.match("Сакой")] == ["Saka"]
    assert idx.match("туре") == [] and idx.match("команда") == []  # стоп-лист FPL-чата
    assert idx.match("Холанда")[0] == ("Haaland", 1.0)


# ---------- резолвер: мини-bootstrap из test_agent_fakes ----------


def test_cyrillic_mentions_on_fake_bootstrap():
    r = PlayerResolver(BS, squad_ids=SQUAD_IDS)
    saka = r.resolve("Саки")
    assert saka.status == "resolved" and saka.player["id"] == 12 and saka.mention == "Саки"
    palmer = r.resolve("Палмера")  # Cole — в составе
    assert (
        palmer.player["id"] == 154 and "'Palmer' resolved to the one in your squad" in palmer.note
    )
    assert r.resolve("Жоау Педро").player["id"] == 165
    assert r.resolve("Холанду").player["id"] == 411
    assert r.resolve("Гуэхи").player["id"] == 388
    gabriel = r.resolve("Габриэля")
    assert gabriel.status == "ambiguous" and gabriel.mention == "Габриэля"
    assert {c["id"] for c in gabriel.candidates} == {4, 18, 27, 331}
    assert r.resolve("Педро").status == "ambiguous"
    for word in ("состав", "капитан", "защитник", "тур", "взять", "мой"):
        assert r.resolve(word).status == "unknown", word


def test_salah_resolves_when_he_is_in_the_bootstrap():
    bs = Bootstrap(
        events=BS.events,
        teams=[*BS.teams, Team(id=7, name="Liverpool", short_name="LIV")],
        elements=[
            *BS.elements,
            Player(
                id=381,
                web_name="M.Salah",
                first_name="Mohamed",
                second_name="Salah",
                team=7,
                element_type=3,
                now_cost=145,
                selected_by_percent=40.0,
            ),
        ],
    )
    r = PlayerResolver(bs)
    for mention in ("Салах", "Салаха", "Мохамед Салах"):
        assert r.resolve(mention).player["id"] == 381, mention
    assert (
        PlayerResolver(BS).resolve("Салаха").status == "unknown"
    )  # нет в bootstrap — не подменяем


def test_latin_mentions_unchanged_and_matcher_fallback():
    r = PlayerResolver(BS)
    assert r.resolve("Saka").player["id"] == 12
    assert r.resolve("Gabriel").status == "ambiguous"
    assert r.resolve("Zzyzx").status == "unknown"


def test_resolve_all_keeps_cyrillic_mention_for_clarification():
    r = PlayerResolver(BS, squad_ids=SQUAD_IDS)
    players, ambiguous, unknown, _ = r.resolve_all(
        ["Саки", "Габриэля", "Салаха"], "кого взять вместо Саки и Габриэля? а Салаха?"
    )
    assert [p["id"] for p in players] == [12]
    assert [a.mention for a in ambiguous] == ["Габриэля"]
    assert [u.mention for u in unknown] == ["Салаха"]


# ---------- резолвер: регрессионный набор на реальных именах ----------


def test_golden_cyrillic_mentions_recognition(real, golden):
    missed = [
        (m, pid, (res := real.resolve(m)).status, (res.player or {}).get("id"))
        for m, pid in golden["resolved"]
        if not ((res := real.resolve(m)).status == "resolved" and res.player["id"] == pid)
    ]
    wrong_ambiguous = [
        m
        for m, ids in golden["ambiguous"]
        if not (
            (res := real.resolve(m)).status == "ambiguous"
            and {c["id"] for c in res.candidates} == set(ids)
        )
    ]
    recognized = len(golden["resolved"]) + len(golden["ambiguous"]) - len(missed)
    recognized -= len(wrong_ambiguous)
    total = len(golden["resolved"]) + len(golden["ambiguous"])
    assert recognized / total == 1.0, (missed, wrong_ambiguous)
    for m, pid in golden["homographs"]:
        assert real.resolve(m).player["id"] == pid


def test_absent_players_are_not_replaced_by_lookalikes(real, golden):
    for m in golden["absent"]:  # «Салах» ~ Salia (0.1%): нечёткое — только для популярных
        assert real.resolve(m).status == "unknown", m


def test_common_russian_words_are_not_players(real, golden):
    forms = {f for w in golden["negative"] for f in (w, w.lower(), w.capitalize())}
    hits = {f: real.resolve(f) for f in forms}
    false_positives = {
        f: r.player or r.candidates for f, r in hits.items() if r.status != "unknown"
    }
    assert false_positives == {}


# ---------- роутер v3: правило написания ----------


def test_router_v3_asks_for_latin_nominative_and_keeps_v2_schema():
    v3 = load_agent_prompt("router.system", "v3")
    assert '"Саки" / "Саку" -> "Saka"' in v3 and "never add a first name" in v3
    assert "code reads Cyrillic too" in v3
    assert "exactly as written, even in Latin script" in load_agent_prompt("router.system", "v2")
    assert router_schema("v3") is RouterOutputV3 and router_schema("v2") is RouterOutputV2
    desc = RouterOutputV3.model_json_schema()["properties"]["player_mentions"]["description"]
    assert "original Latin FPL spelling" in desc


# ---------- валидатор ответа ----------

FACTS = {
    "players": {"Saka": {"team": "ARS", "xpts": 6.2}, "Haaland": {"team": "MCI"}},
    "transfers": {"routes": [{"out": ["Saka"], "in": ["Groß"], "gain_horizon": 4.03}]},
    "player_ranking": {
        "rows": [
            {"player": "Groß", "full_name": "Pascal Groß"},
            {"player": "Mitchell", "full_name": "Tyrick Mitchell"},
            {"player": "Welbeck", "full_name": "Danny Welbeck"},
            {"player": "Hall", "full_name": "Lewis Hall"},
            {"player": "Mbeumo", "full_name": "Bryan Mbeumo"},
        ]
    },
}
CLUBS = ["Arsenal", "Chelsea", "Man City", "Hull", "Aston Villa", "Villa"]


def test_fact_player_names_reads_name_fields_and_player_keys():
    names = fact_player_names(FACTS)
    assert {"Saka", "Haaland", "Groß", "Pascal Groß", "Tyrick Mitchell"} <= set(names)
    assert "ARS" not in names and "transfers" not in names


def test_transliterated_names_fail_validation_with_feedback():
    answer = (
        "**Вердикт:** продать Саку и взять Паскаля Гросса (+4.03 xPts).\n"
        "- Вместо Саки — Гросс; Холанд — капитан, Мбемо в форме."
    )
    res = validate_answer(answer, FACTS, [], extra_names=CLUBS)
    assert not res.passed
    assert res.cyrillic_names == {
        "Саку": "Saka",
        "Паскаля": "Pascal",
        "Гросса": "Groß",
        "Саки": "Saka",
        "Гросс": "Groß",
        "Холанд": "Haaland",
        "Мбемо": "Mbeumo",
    }
    assert "'Саки' -> 'Saka'" in res.feedback and "Latin script" in res.feedback
    fixed = restore_latin_names(answer, res.cyrillic_names)
    assert "продать Saka и взять Pascal Groß" in fixed and "Вместо Saka — Groß" in fixed
    assert validate_answer(fixed, FACTS, [], extra_names=CLUBS).passed


def test_spellings_copied_from_the_question_are_caught():
    facts = {"players": {"Haaland": {}, "Ødegaard": {}, "Isak": {}}}
    found = cyrillic_player_names("Капитаним Холанна; Холанн, Эдегор и Исака — в форме.", facts)
    assert found == {
        "Холанна": "Haaland",
        "Холанн": "Haaland",
        "Эдегор": "Ødegaard",
        "Исака": "Isak",
    }


def test_common_words_vs_names_of_popular_players(real, golden):
    """185 игроков с владением >= 1% в «фактах» — ни одно обычное слово с заглавной не имя."""
    names = [
        n
        for p in real.bs.elements
        if (p.selected_by_percent or 0) >= 1.0
        for n in (p.web_name, p.full_name)
    ]
    words = sorted({w[:1].upper() + w[1:] for w in golden["negative"]})
    assert cyrillic_player_names(" ".join(words), {}, names) == {}


def test_ordinary_russian_text_with_capitals_passes():
    answer = (
        "**Вердикт:** продать Saka и взять Groß (+4.03 xPts).\n\n**Почему**\n"
        "- Тройка лучших: Groß, Mitchell, Welbeck. Форма у Mbeumo стабильная.\n"
        "- Сухой матч вероятен: Арсенал дома, Халл и Вилла слабее. Данные свежие.\n"
        "- Капитан — Haaland. Команда, Состав, Замена, Риск, Сила соперника, Матч, Цена.\n"
        "- Вода, Уже, Весь, Или, Мать, Будет, Итог, Туре, Море.\n\n**Оговорки**\n- Прогноз."
    )
    res = validate_answer(answer, FACTS, [], extra_names=CLUBS)
    assert res.cyrillic_names == {} and res.passed, res
    assert cyrillic_player_names("Сака сыграет.", {}, []) == {}  # имён в фактах нет — не с чем


# ---------- граф: повтор объяснения и HITL по кириллическому упоминанию ----------


def make_agent(tools: FakeTools, router: FakeRouter, explain: FakeExplain) -> Agent:
    deps = Deps(
        tools=tools,
        router_llm=router,
        explain_llm=explain,
        grader_llm=None,
        resolver=lambda squad: PlayerResolver(BS, squad_ids=squad),
        clock=lambda: NOW,
        extra_known_names=["Arsenal", "Chelsea"],
    )
    return Agent(deps, MemorySaver())


def test_cyrillic_name_in_answer_triggers_one_regeneration():
    explain = FakeExplain(
        [
            "**Вердикт:** Сака сильнее Палмера (6.2 xPts).",
            "**Вердикт:** Saka сильнее Palmer (6.2 xPts).",
        ]
    )
    router = FakeRouter(router_output("compare_players", ["Saka", "Palmer"], needs_squad=False))
    state = make_agent(FakeTools(), router, explain).run("сравни Саку и Палмера")
    assert len(explain.calls) == 2
    assert "'Сака' -> 'Saka'" in (explain.calls[1].feedback or "")
    assert state["validation"]["history"][0]["cyrillic_names"] == {
        "Сака": "Saka",
        "Палмера": "Palmer",
    }
    assert state["validation"]["passed"] and state["answer"].startswith("**Вердикт:** Saka")


def test_cyrillic_names_restored_after_regenerations_run_out():
    explain = FakeExplain(["**Вердикт:** Сака сильнее Палмера (6.2 xPts)."])
    router = FakeRouter(router_output("compare_players", ["Saka", "Palmer"], needs_squad=False))
    state = make_agent(FakeTools(), router, explain).run("сравни Саку и Палмера")
    assert len(explain.calls) == 2 and not state["validation"]["passed"]
    assert state["answer"].startswith("**Вердикт:** Saka сильнее Palmer")
    assert "Сака" not in state["answer"] and "⚠" not in state["answer"]


def test_homoglyph_in_latin_name_is_fixed_without_regeneration():
    explain = FakeExplain(["**Вердикт:** Sаka сильнее Palmer (6.2 xPts)."])  # кириллическая «а»
    router = FakeRouter(router_output("compare_players", ["Saka", "Palmer"], needs_squad=False))
    state = make_agent(FakeTools(), router, explain).run("сравни Saka и Palmer")
    assert len(explain.calls) == 1 and state["validation"]["passed"]
    assert state["answer"] == "**Вердикт:** Saka сильнее Palmer (6.2 xPts)."


def test_cyrillic_ambiguous_mention_goes_to_clarification():
    tools = FakeTools()
    router = FakeRouter(router_output("transfer", ["Габриэля"], sell=["Габриэля"]))
    agent = make_agent(tools, router, FakeExplain())
    state = agent.run("стоит ли продать Габриэля", manager_id=1, thread_id="t-cyr")
    pending = state["pending_clarification"]
    assert state["interrupted"] and pending["mention"] == "Габриэля"
    assert pending["roles"] == ["sell"]
    assert {c["player_id"] for c in pending["candidates"]} == {4, 18, 27, 331}
    final = agent.resume("t-cyr", player_id=4)
    assert final["scenario"]["sell"] == [4] and not final["interrupted"]
