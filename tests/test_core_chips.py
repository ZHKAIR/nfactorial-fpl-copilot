"""Фишки в многотуровом плане (Bench Boost, Triple Captain) на игрушечных данных: цель тура,
смена решений, вклад фишки, правила и валидация. Без БД и сети, каждый solve < 1 с."""

from __future__ import annotations

import pytest
from test_core_optimizer import GWS, as_pool, base_squad, cand, ids

from fplcopilot.core.candidates import chip_available, chips_available
from fplcopilot.core.optimizer import (
    ChipPlanError,
    Model,
    ModelSpec,
    best_starters,
    best_xi,
    plan_transfers,
    validate_chip_plan,
)
from fplcopilot.core.strategy import Strategy, get_strategy
from fplcopilot.data.schemas import Bootstrap

BOTH = ["bboost", "3xc"]


def bench_squad():
    """Сильные 11 (3-5-2, все ≥ 5.0) и мёртвая скамейка (GK 1.0, полевые по 0.5)."""
    return [
        cand(1, 1, 1, 5.0, 5.0),
        cand(2, 1, 2, 4.0, 1.0),
        cand(3, 2, 3, 6.0, 5.0),
        cand(4, 2, 4, 6.0, 5.0),
        cand(5, 2, 5, 6.0, 5.0),
        cand(6, 2, 6, 4.0, 0.5),
        cand(7, 2, 7, 4.0, 0.5),
        cand(8, 3, 8, 8.0, 6.0),
        cand(9, 3, 9, 8.0, 6.0),
        cand(10, 3, 10, 8.0, 6.0),
        cand(11, 3, 11, 8.0, 6.0),
        cand(12, 3, 12, 8.0, 6.0),
        cand(13, 4, 13, 9.0, 7.0),
        cand(14, 4, 14, 9.0, 7.0),
        cand(15, 4, 15, 4.5, 0.5),
    ]


def bench_market():
    """Два запасных, которые хороши только в GW7 и в старт не проходят (4.0 < 5.0, 3.5 < 7.0)."""
    return [
        cand(40, 2, 40, 4.0, {5: 0.5, 6: 0.5, 7: 4.0}),
        cand(41, 4, 41, 4.5, {5: 0.5, 6: 0.5, 7: 3.5}),
    ]


def squads(plan) -> dict[int, set[int]]:
    return {gw: set(lu.starters) | set(lu.bench_order) for gw, lu in plan.lineups_by_gw.items()}


# ---------- Bench Boost ----------


def test_bench_boost_counts_all_15_and_changes_bench_transfers():
    squad = bench_squad()
    pool = as_pool(squad, bench_market())
    plain = plan_transfers(ids(squad), 0.0, 1, BOTH, 5, 3, "balanced", pool=pool)
    # без фишки запасные на 4.0/3.5 в одном туре не окупают FT (скамейка весит 0.21 / 0.06)
    assert plain.recommendation == "hold" and not any(plain.moves_by_gw.values())
    assert plain.chips_by_gw == {} and all(lu.chip is None for lu in plain.lineups_by_gw.values())

    bb = plan_transfers(ids(squad), 0.0, 1, BOTH, 5, 3, "balanced", pool=pool, chips={7: "bboost"})
    assert bb.chips_by_gw == {7: "bboost"}
    bought = {m.in_ for ms in bb.moves_by_gw.values() for m in ms}
    assert bought == {40, 41}  # FT накоплены и потрачены на скамейку под матчи GW7
    assert {40, 41} <= squads(bb)[7]
    assert all(m.gw <= 7 and not m.paid for ms in bb.moves_by_gw.values() for m in ms)
    lu7 = bb.lineups_by_gw[7]
    xp = {p: pool[p].xpts(7) for p in squads(bb)[7]}
    assert lu7.chip == "bboost"
    assert lu7.chip_points == pytest.approx(sum(xp[p] for p in lu7.bench_order))
    assert lu7.expected_points == pytest.approx(sum(xp.values()) + xp[lu7.captain])  # все 15
    assert bb.lineups_by_gw[5].chip is None and bb.lineups_by_gw[6].chip is None
    # baseline — те же фишки без трансферов: разница с планом = ценность трансферов при BB
    baseline_bench_7 = 1.0 + 0.5 * 3
    assert bb.baseline_total == pytest.approx(plain.baseline_total + baseline_bench_7)
    assert bb.expected_total > bb.baseline_total
    assert bb.expected_total - plain.expected_total == pytest.approx(lu7.chip_points)


def test_bench_boost_objective_replaces_bench_weights_with_full_points():
    """Один тур без трансферов: цель тура с BB = Σ xPts всех 15 + капитан − λ·Σ дисперсии 15."""
    squad = bench_squad()
    pool = as_pool(squad)
    S = get_strategy("balanced")

    def solve(chips):
        spec = ModelSpec(
            gws=[5],
            pool=pool,
            initial_squad=ids(squad),
            bank=0.0,
            free_transfers=1,
            strategy=S,
            max_transfers={5: 0},
            chips=chips,
        )
        return Model(spec).solve()[0]

    plain, bb = solve({}), solve({5: "bboost"})
    total = sum(c.xpts(5) for c in squad)
    risk = sum(c.variance(5) for c in squad)
    ft_itb = S.ft_value * 1 + S.itb_value * 0.0  # FT 1 -> 2
    assert bb.objective == pytest.approx(total + 7.0 - S.variance_penalty * risk + ft_itb)
    assert bb.lineups[5].objective == pytest.approx(total + 7.0 - S.variance_penalty * risk)
    assert plain.lineups[5].chip_points == 0.0 and bb.lineups[5].chip_points == pytest.approx(2.5)
    assert bb.lineups[5].expected_points == pytest.approx(plain.lineups[5].expected_points + 2.5)
    # старт при BB — лучшие 11 (как без фишки), а не произвольные 11 из равноценных для цели
    assert set(bb.lineups[5].starters) == set(plain.lineups[5].starters)


@pytest.mark.parametrize("seed", range(6))
def test_best_starters_matches_milp_best_xi(seed):
    import random

    rng = random.Random(seed)
    squad = [
        cand(c.player_id, c.position, c.team_id, c.price, round(rng.uniform(0, 9), 2))
        for c in base_squad()
    ]
    pure = Strategy(
        name="pure", variance_penalty=0.0, hit_threshold=0.0, ownership_weight=0.0,
        wc_margin=0.0, bench_weights={},
    )  # fmt: skip
    lu = best_xi(squad, 5, pure)
    pool = as_pool(squad)
    greedy = best_starters(pool, 5, ids(squad))
    assert len(greedy) == 11
    assert sum(pool[p].xpts(5) for p in greedy) == pytest.approx(
        lu.expected_points - pool[lu.captain].xpts(5)
    )


# ---------- Triple Captain ----------


def test_triple_captain_triples_captain_only_in_its_gameweek():
    squad = base_squad()
    pool = as_pool(squad, [cand(60, 4, 60, 5.0, 1.1)])
    plain = plan_transfers(ids(squad), 0.0, 1, BOTH, 5, 3, "balanced", pool=pool)
    tc = plan_transfers(ids(squad), 0.0, 1, BOTH, 5, 3, "balanced", pool=pool, chips={6: "3xc"})
    lu6 = tc.lineups_by_gw[6]
    assert lu6.chip == "3xc" and lu6.captain == 13
    assert lu6.chip_points == pytest.approx(pool[13].xpts(6))
    assert lu6.expected_points == pytest.approx(
        sum(pool[p].xpts(6) for p in lu6.starters) + 2 * pool[13].xpts(6)
    )
    for gw in (5, 7):
        assert tc.lineups_by_gw[gw].chip is None and tc.lineups_by_gw[gw].chip_points == 0.0
        assert tc.lineups_by_gw[gw].expected_points == plain.lineups_by_gw[gw].expected_points
    assert tc.expected_total - plain.expected_total == pytest.approx(pool[13].xpts(6))
    assert tc.baseline_total - plain.baseline_total == pytest.approx(pool[13].xpts(6))


def test_triple_captain_makes_a_captaincy_upgrade_worth_the_free_transfer():
    """Z = капитан 13 (7.5) + 0.65 только в GW6, та же цена. Без TC апгрейд +2·0.65·0.84 = 1.09
    не окупает FT (1.5·0.84 = 1.26); с TC +3·0.65·0.84 = 1.64 — окупает: Z берут в GW6."""
    squad = base_squad()
    z = cand(70, 4, 70, 15.0, {5: 7.5, 6: 8.15, 7: 7.5}, var=18.0)
    pool = as_pool(squad, [z])
    plain = plan_transfers(ids(squad), 0.0, 1, BOTH, 5, 3, "balanced", pool=pool)
    assert not any(plain.moves_by_gw.values())
    tc = plan_transfers(ids(squad), 0.0, 1, BOTH, 5, 3, "balanced", pool=pool, chips={6: "3xc"})
    moves = [m for ms in tc.moves_by_gw.values() for m in ms]
    assert [(m.gw, m.out, m.in_) for m in moves] == [(6, 13, 70)]
    assert tc.lineups_by_gw[6].captain == 70
    assert tc.lineups_by_gw[6].chip_points == pytest.approx(8.15)


# ---------- без фишек, Wildcard ----------


def test_no_chips_plan_is_identical_to_plan_without_chip_arguments():
    squad = base_squad()
    pool = as_pool(squad, [cand(61, 4, 61, 5.0, 4.8), cand(62, 2, 62, 4.0, 4.2)])
    skip = {"runtime_s", "created_at"}
    old = plan_transfers(ids(squad), 0.0, 1, ["wildcard"], 5, 3, "balanced", pool=pool)
    new = plan_transfers(
        ids(squad),
        0.0,
        1,
        ["wildcard", *BOTH],
        5,
        3,
        "balanced",
        pool=pool,
        chips={},
        chips_available_by_gw={5: BOTH, 6: BOTH, 7: BOTH},
    )
    assert old.model_dump(exclude=skip) == new.model_dump(exclude=skip)
    assert new.chips_by_gw == {} and new.notes == []
    assert new.wildcard_alternative is not None


def test_chip_in_first_gw_blocks_wildcard_alternative_later_chip_keeps_it():
    squad = base_squad()
    pool = as_pool(squad, [cand(61, 4, 61, 5.0, 4.8)])
    chips_now = ["wildcard", *BOTH]
    first = plan_transfers(
        ids(squad), 0.0, 1, chips_now, 5, 3, "balanced", pool=pool, chips={5: "bboost"}
    )
    assert first.wildcard_alternative is None
    assert first.notes and "Wildcard в GW5" in first.notes[0] and "Bench Boost" in first.notes[0]
    later = plan_transfers(
        ids(squad), 0.0, 1, chips_now, 5, 3, "balanced", pool=pool, chips={6: "bboost"}
    )
    wc = later.wildcard_alternative
    assert wc is not None and later.notes == []
    assert wc.lineups_by_gw[6].chip == "bboost" and wc.lineups_by_gw[5].chip is None


def test_model_spec_rejects_wildcard_and_chip_in_same_gw():
    squad = base_squad()
    spec = ModelSpec(
        gws=GWS,
        pool=as_pool(squad),
        initial_squad=ids(squad),
        bank=0.0,
        free_transfers=1,
        strategy=get_strategy("balanced"),
        max_transfers={w: None for w in GWS},
        wildcard_gw=5,
        chips={5: "3xc"},
    )
    with pytest.raises(ChipPlanError, match="Wildcard и Triple Captain"):
        Model(spec)


# ---------- валидация ----------


AVAIL = {5: BOTH, 6: BOTH, 7: BOTH}


@pytest.mark.parametrize(
    ("chips", "available", "match"),
    [
        ([(6, "bboost"), (6, "3xc")], AVAIL, "две фишки в одном туре"),
        ([(6, "bboost"), (6, "bboost")], AVAIL, "две фишки в одном туре"),
        ({9: "bboost"}, AVAIL, r"вне горизонта плана GW5–GW7"),
        ({4: "3xc"}, AVAIL, "вне горизонта"),
        ({6: "bboost"}, {5: BOTH, 6: ["3xc"], 7: BOTH}, "Bench Boost недоступен менеджеру в GW6"),
        ({5: "3xc"}, {5: [], 6: [], 7: []}, "Triple Captain недоступен"),
        ({6: "freehit"}, AVAIL, "Free Hit оптимизатор не моделирует"),
        ({6: "wildcard"}, AVAIL, "Wildcard не задаётся условием"),
        ({6: "manager"}, AVAIL, "неизвестная фишка"),
        ([(5, "bboost"), (7, "bboost")], AVAIL, "дважды в одной половине сезона"),
    ],
)
def test_validate_chip_plan_rejects_invalid_conditions(chips, available, match):
    with pytest.raises(ChipPlanError, match=match):
        validate_chip_plan(chips, GWS, available)


def test_validate_chip_plan_accepts_one_chip_per_half_and_sorts():
    gws = [18, 19, 20, 21]
    avail = {g: BOTH for g in gws}
    assert validate_chip_plan([(21, "bboost"), (18, "bboost")], gws, avail) == {
        18: "bboost",
        21: "bboost",
    }
    assert validate_chip_plan({19: "3xc", 18: "bboost"}, gws, avail) == {18: "bboost", 19: "3xc"}
    assert validate_chip_plan((), gws, avail) == {}


def test_plan_transfers_raises_instead_of_ignoring_invalid_chip():
    squad = base_squad()
    pool = as_pool(squad)
    with pytest.raises(ChipPlanError, match="недоступен"):
        plan_transfers(ids(squad), 0.0, 1, [], 5, 3, "balanced", pool=pool, chips={6: "bboost"})
    with pytest.raises(ChipPlanError, match="вне горизонта"):
        plan_transfers(ids(squad), 0.0, 1, BOTH, 5, 3, "balanced", pool=pool, chips={8: "3xc"})
    # доступность по туру важнее плоского списка на from_gw
    with pytest.raises(ChipPlanError, match="GW7"):
        plan_transfers(
            ids(squad),
            0.0,
            1,
            BOTH,
            5,
            3,
            "balanced",
            pool=pool,
            chips={7: "3xc"},
            chips_available_by_gw={5: BOTH, 6: BOTH, 7: ["bboost"]},
        )


# ---------- снимки: diff и хэш ----------


def test_plan_diff_and_inputs_hash_track_chips():
    from test_core_plan import _move, _plan

    from fplcopilot.core.plan import diff_plans, inputs_hash

    prev = _plan([_move(5, 1, 101)])
    new = prev.model_copy(update={"chips_by_gw": {7: "bboost"}})
    (change,) = diff_plans(prev, new)
    assert change.kind == "chips" and change.detail == "— -> GW7 Bench Boost"
    assert diff_plans(new, new) == []
    base = inputs_hash([1, 2, 3], 1.0, 1, "balanced", 3, 5)
    assert inputs_hash([1, 2, 3], 1.0, 1, "balanced", 3, 5, chips={}) == base  # без фишек прежний
    assert inputs_hash([1, 2, 3], 1.0, 1, "balanced", 3, 5, chips={7: "bboost"}) != base


# ---------- доступность по окнам bootstrap ----------


def test_chip_availability_by_half_season_window():
    bs = Bootstrap.model_validate(
        {
            "events": [],
            "teams": [],
            "elements": [],
            "chips": [
                {"name": "wildcard", "start_event": 2, "stop_event": 19},
                {"name": "wildcard", "start_event": 20, "stop_event": 38},
                {"name": "freehit", "start_event": 2, "stop_event": 19},
                {"name": "bboost", "start_event": 1, "stop_event": 19},
                {"name": "3xc", "start_event": 1, "stop_event": 19},
                {"name": "freehit", "start_event": 20, "stop_event": 38},
                {"name": "bboost", "start_event": 20, "stop_event": 38},
                {"name": "3xc", "start_event": 20, "stop_event": 38},
            ],
        }
    )
    assert chips_available(bs, [], 6) == ["wildcard", "freehit", "bboost", "3xc"]
    used = ["bboost", "3xc", "wildcard", "freehit"]  # 895045: весь первый набор сыгран
    assert chips_available(bs, used, 6) == []
    assert chips_available(bs, used, 20) == ["wildcard", "freehit", "bboost", "3xc"]
    assert chip_available(bs, ["bboost"], 1, "3xc") and not chip_available(
        bs, ["bboost"], 1, "bboost"
    )
    assert not chip_available(bs, ["bboost", "bboost"], 25, "bboost")
    assert chips_available(bs, [], 1) == ["bboost", "3xc"]  # WC/FH — с GW2
