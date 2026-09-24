"""Страница «План»: фраза-итог, выбор чипов, эффект чипов, две серии прогноза по турам и
склонение «очко / очка / очков» — чистые функции app/plan_view.py, app/briefing.py, format."""

from __future__ import annotations

from fplcopilot.agent.tools import PlanChipOut, PlanMoveOut, PlanOut
from fplcopilot.app import briefing, plan_view
from fplcopilot.app import format as fmt


def plan(**kw) -> PlanOut:
    base = {
        "from_gw": 6,
        "horizon": 5,
        "strategy": "balanced",
        "moves_by_gw": {
            "6": [
                PlanMoveOut(
                    gw=6,
                    out="Slater",
                    **{"in": "Groß"},
                    price_out=4.5,
                    price_in=5.8,
                    delta_xpts_horizon=10.8,
                ),
                PlanMoveOut(
                    gw=6,
                    out="Sels",
                    **{"in": "Raya"},
                    price_out=5.0,
                    price_in=5.5,
                    delta_xpts_horizon=6.0,
                    paid=True,
                ),
            ],
            "7": [],
        },
        "hits_by_gw": {"6": 4, "7": 0},
        "ft_by_gw": {"6": 1, "7": 1},
        "bank_by_gw": {"6": 1.3, "7": 1.3},
        "xi_points_by_gw": {"6": 63.2, "7": 69.8},
        "captain_by_gw": {"6": "Saka", "7": "Haaland"},
        "expected_total": 325.77,
        "baseline_total": 300.68,
        "recommendation": "transfers",
        "baseline_xi_points_by_gw": {"6": 59.4, "7": 62.0},
    }
    base.update(kw)
    return PlanOut(**base)


def test_points_text_declension():
    assert fmt.points_text(1) == "1 очко"
    assert fmt.points_text(2) == "2 очка"
    assert fmt.points_text(5) == "5 очков"
    assert fmt.points_text(21) == "21 очко"
    assert fmt.points_text(12) == "12 очков"
    assert fmt.points_text(2.5) == "2.5 очка"
    assert fmt.points_text(7.2, signed=True) == "+7.2 очка"
    assert fmt.points_text(-4, signed=True) == "−4 очка"
    assert fmt.points_text(0, signed=True) == "0 очков"


def test_headline_transfers_with_hit_and_gain():
    assert plan_view.headline(plan()) == (
        "В GW6 сделайте трансфер Slater → Groß и ещё 1 ход (платных: 1, −4 очка). "
        "План на 5 туров даст 325.8 очка — на 25.1 очка больше, чем если ничего не менять."
    )


def test_headline_hold_and_equal():
    p = plan(
        moves_by_gw={"6": []}, recommendation="hold", expected_total=300.7, baseline_total=300.7
    )
    assert plan_view.headline(p) == (
        "В GW6 трансфер не нужен — бесплатный переносится на следующий тур. План на 5 туров "
        "даст 300.7 очка — столько же, сколько если ничего не менять."
    )


def test_headline_wildcard_recommendation():
    p = plan(
        recommendation="wildcard",
        wildcard={"gw": 6, "expected_total": 340.2, "delta_vs_plan": 14.4, "squad": []},
    )
    assert plan_view.headline(p) == (
        "Выгоднее сыграть Wildcard в GW6: 340.2 очка за 5 туров — на 14.4 очка больше "
        "обычного плана."
    )


def test_answer_numbers_tones():
    nums = plan_view.answer_numbers(plan())
    assert nums == [
        ("325.8", "очков по плану", "accent"),
        ("300.7", "если ничего не менять", "plain"),
        ("+25.1", "разница", "good"),
    ]
    worse = plan_view.answer_numbers(plan(expected_total=290.0))
    assert worse[2] == ("-10.7", "разница", "bad")


def test_chip_selection_one_chip_per_gw_and_order():
    chips, notes = plan_view.chip_selection({"bboost": 7, "3xc": 9, "freehit": None})
    assert chips == ((7, "bboost"), (9, "3xc"))
    assert notes == []
    chips, notes = plan_view.chip_selection({"3xc": 7, "bboost": 7})
    assert chips == ((7, "bboost"),)  # Bench Boost — первый по порядку
    assert notes == [
        (
            "В один тур можно сыграть только один чип: Triple Captain в GW7 не учтён — там уже "
            "Bench Boost."
        )
    ]
    assert plan_view.chips_text(chips) == "Bench Boost в GW7"
    assert plan_view.chips_text({"9": "3xc", "7": "bboost"}) == (
        "Bench Boost в GW7, Triple Captain в GW9"
    )


def test_plannable_chips_order_and_filter():
    # план моделирует только Bench Boost и Triple Captain (Wildcard — альтернатива, FH — нет)
    assert plan_view.plannable_chips(["wildcard", "freehit", "3xc", "bboost"]) == ["bboost", "3xc"]
    assert plan_view.plannable_chips([]) == []


def chip(gw, name="bboost", points=8.3):
    return PlanChipOut(gw=gw, chip=name, name=plan_view.chip_label(name), points=points)


def test_chip_effect_compares_with_same_plan_without_chips():
    with_chip = plan(expected_total=334.1, chips=[chip(7)])
    assert plan_view.chip_effect(with_chip, plan()) == (
        "С Bench Boost в GW7 план даёт на 8.3 очка больше, чем без чипов."
    )
    assert plan_view.chip_effect(plan(), plan()) is None  # чипов нет
    same = plan(chips=[chip(7, "3xc")])
    assert plan_view.chip_effect(same, plan()) == (
        "С чипами (Triple Captain в GW7) план даёт столько же очков, сколько без них."
    )


def test_chips_map_for_gw_badges():
    assert plan_view.chips_map(plan(chips=[chip(7), chip(9, "3xc")])) == {"7": "bboost", "9": "3xc"}
    assert plan_view.chips_map(plan()) == {}


def test_plan_vs_hold_two_series_net_of_hits():
    assert briefing.plan_vs_hold(plan()) == [("GW6", 59.4, 59.2), ("GW7", 62.0, 69.8)]
    no_base = plan(baseline_xi_points_by_gw={})
    assert briefing.plan_vs_hold(no_base)[0] == ("GW6", None, 59.2)
