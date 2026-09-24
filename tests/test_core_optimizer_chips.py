"""Прогноз по турам «без трансферов» в многотуровом плане (TransferPlan.baseline_points_by_gw) —
для парных столбиков на странице «План». Фишки Bench Boost / Triple Captain покрыты
tests/test_core_chips.py; здесь — что baseline по турам согласован с baseline_total и
считается с теми же фишками."""

import pytest
from test_core_optimizer import as_pool, base_squad, ids

from fplcopilot.core.optimizer import plan_transfers


def test_baseline_points_by_gw_sum_to_baseline_total():
    squad = base_squad()
    pool = as_pool(squad)
    plan = plan_transfers(ids(squad), 0.0, 1, [], 5, 3, "balanced", pool=pool)
    assert set(plan.baseline_points_by_gw) == {5, 6, 7}
    assert sum(plan.baseline_points_by_gw.values()) == pytest.approx(plan.baseline_total, abs=1e-2)


def test_baseline_points_by_gw_include_planned_chip():
    squad = base_squad()
    pool = as_pool(squad)
    plain = plan_transfers(ids(squad), 0.0, 1, [], 5, 3, "balanced", pool=pool)
    bb = plan_transfers(
        ids(squad), 0.0, 1, ["bboost"], 5, 3, "balanced", pool=pool, chips={6: "bboost"}
    )
    bench = sum(pool[p].xpts(6) for p in plain.lineups_by_gw[6].bench_order)
    assert bb.baseline_points_by_gw[6] == pytest.approx(
        plain.baseline_points_by_gw[6] + bench, abs=1e-2
    )
    assert bb.baseline_points_by_gw[5] == pytest.approx(plain.baseline_points_by_gw[5])
