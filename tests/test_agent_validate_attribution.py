"""Валидатор: число одного игрока, приписанное другому (misattributed_numbers)."""

from __future__ import annotations

import pytest

from fplcopilot.agent.validate import misattributed_numbers, validate_answer

FACTS = {
    "intent": "captain",
    "gw": 6,
    "question": "кого поставить капитаном?",
    "manager": {
        "bank": 0.1,
        "squad": [
            "Saka (ARS, MID, £9.5)",
            "Palmer (CHE, MID, £9.7)",
            "Sels (NFO, GKP, £5.0)",
            "N.Williams (NFO, DEF, £5.0)",
            "Haaland (MCI, FWD, £15.6, C)",
        ],
    },
    "best_xi": {
        "gw": 6,
        "expected_points": 57.63,
        "captain": "Saka",
        "vice": "Haaland",
        "starters": [
            "Saka (ARS, MID) 7.17 xPts, p_start 0.94",
            "Palmer (CHE, MID) 4.57 xPts, p_start 0.74",
            "Haaland (MCI, FWD) 5.53 xPts, p_start 0.85",
        ],
    },
    "captain_options": [
        {"name": "Saka", "xpts": 7.17, "captain_points": 14.34, "ownership_pct": 13.5},
        {"name": "Haaland", "xpts": 5.53, "captain_points": 11.06, "ownership_pct": 73.7},
    ],
    "plan": {
        "moves_by_gw": {"6": ["Palmer (£9.7) -> Groß (£5.8) Δ+1.66"]},
        "bank_by_gw": {"6": 3.0},
        "expected_total": 126.22,
    },
    "transfers": {"routes": [{"out": ["Palmer"], "in": ["Groß"], "gain_next_gw": 2.0}]},
    "headline": "Best XI GW6: 57.63 xPts; captain Saka (7.17 xPts, x2 = 14.34); vice Haaland",
}


def _mis(answer: str) -> list[tuple[str, str]]:
    return [(m["player"], m["number"]) for m in misattributed_numbers(answer, FACTS)]


def test_price_of_another_player_given_as_xpts_is_flagged_and_explained():
    answer = "- **Palmer**: 9.5 xPts на следующий тур, сомнителен."
    assert _mis(answer) == [("Palmer", "9.5")]
    res = validate_answer(answer, FACTS)
    assert not res.passed
    assert res.unknown_numbers == []  # число в фактах есть — ловит только атрибуция
    assert res.misattributed_numbers == [{"player": "Palmer", "number": "9.5", "owners": ["Saka"]}]
    assert "belongs to Saka, not Palmer" in res.feedback
    assert res.as_dict()["misattributed_numbers"]


def test_table_row_with_another_players_xpts_is_flagged():
    answer = (
        "| Игрок | Цена | xPts |\n|---|---|---|\n| Palmer | £9.7 | 7.17 |\n| Saka | £9.5 | 7.17 |\n"
    )
    assert _mis(answer) == [("Palmer", "7.17")]


def test_prose_sentence_is_the_unit():
    answer = "Palmer приносит 5.53 xPts. Haaland даёт 5.53 xPts и остаётся вице."
    assert _mis(answer) == [("Palmer", "5.53")]


@pytest.mark.parametrize(
    "answer",
    [
        # капитан ×2: удвоенные очки — число самого капитана
        "Капитан — Saka: 7.17 xPts, с повязкой 14.34 очка.",
        "- Haaland (C) даст 11.06 очка, владение 73.7%.",
        # суммы плана, банк, выигрыш маршрута — не числа одного игрока
        "- Продажа Palmer: план даёт 126.22 xPts за два тура, в банке останется £3.0.",
        "- Palmer уходит: маршрут даёт +2.0 xPts в следующем туре.",
        "Без Palmer лучший состав набирает 57.63 xPts, банк £0.1.",
        # одно число у двух игроков
        "- Sels: £5.0.\n- N.Williams: £5.0.",
        # сравнение двух игроков в одной строке / предложении не проверяется
        "Palmer 5.53 xPts против Saka 7.17 xPts.",
        "- Saka (7.17 xPts) лучше, чем Palmer (£9.7).",
        # свои числа, округление до показанных знаков
        "- Palmer: 4.6 xPts, цена £9.7.",
        "- Groß (£5.8) заменяет Palmer, +1.66 xPts.",
    ],
)
def test_conservative_negatives(answer):
    assert _mis(answer) == []


def test_tables_skip_multi_player_rows_and_accept_own_values():
    answer = (
        "| Трансфер | Δ xPts |\n"
        "|---|---|\n"
        "| Palmer → Groß | +1.66 |\n"
        "\n"
        "| Игрок | xPts | Цена |\n"
        "|---|---|---|\n"
        "| Palmer | 4.57 | £9.7 |\n"
        "| Haaland | 5.53 | £15.6 |\n"
    )
    assert _mis(answer) == []


def test_multi_player_fact_string_parentheses_belong_to_the_name_before_them():
    # «Palmer (£9.7) -> Groß (£5.8) Δ+1.66»: £5.8 — цена Groß, Δ — общий выигрыш хода
    assert _mis("- Palmer стоит £5.8.") == [("Palmer", "5.8")]
    assert _mis("- Продажа Palmer даст +1.66 xPts.") == []
    assert _mis("- Groß стоит £5.8 и даёт +1.66 xPts.") == []


def test_number_absent_from_facts_is_left_to_unknown_numbers():
    res = validate_answer("- Palmer: 8.88 xPts.", FACTS)
    assert res.misattributed_numbers == []
    assert res.unknown_numbers == ["8.88"]


def test_unit_with_unknown_capitalised_name_is_skipped():
    # «Enzo» — не игрок из фактов: во фрагменте может быть второй человек
    assert _mis("- Palmer и Enzo: 7.17 xPts.") == []


def test_numbers_without_metric_marker_are_not_checked():
    assert _mis("- Palmer: 13.5 минут на поле в среднем, 9.5 ударов.") == []


def test_extra_names_count_as_players_and_clubs_do_not():
    facts = {
        "players": {"Saka": {"price": 9.5, "xpts": 7.17}, "Rice": {"price": 6.5, "xpts": 4.2}},
    }
    # «Arsenal» — клуб из extra_names графа, игроком не считается: фрагмент с одним Saka
    found = misattributed_numbers("- Saka (Arsenal): 4.2 xPts.", facts, extra_names=["Arsenal"])
    assert [(m["player"], m["number"], m["owners"]) for m in found] == [("Saka", "4.2", ["Rice"])]
    # игрок-омоним обычного слова из именного поля остаётся игроком: два игрока — не проверяем
    assert misattributed_numbers("- Saka и Rice: 4.2 xPts.", facts) == []
