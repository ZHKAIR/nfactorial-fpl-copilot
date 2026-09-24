"""Структурный diff планов и хэш входов (без БД) + один db-тест round-trip plan_snapshots."""

import pytest
from sqlalchemy import text

from fplcopilot.core.candidates import Candidate
from fplcopilot.core.optimizer import LineupResult, PlannedMove, TransferPlan
from fplcopilot.core.plan import diff_plans, inputs_hash, latest_plan, save_plan
from fplcopilot.db import ping, session_scope


def _lineup(gw: int) -> LineupResult:
    return LineupResult(
        gw=gw,
        formation="3-4-3",
        starters=list(range(1, 12)),
        bench_order=[12, 13, 14, 15],
        captain=1,
        vice=2,
        expected_points=50.0,
        objective=50.0,
    )


def _move(gw: int, out: int, inn: int) -> PlannedMove:
    return PlannedMove(gw=gw, out=out, in_=inn, out_name=f"P{out}", in_name=f"P{inn}")


def _plan(moves: list[PlannedMove], recommendation: str = "transfers") -> TransferPlan:
    by_gw: dict[int, list[PlannedMove]] = {gw: [] for gw in (5, 6, 7)}
    for m in moves:
        by_gw[m.gw].append(m)
    return TransferPlan(
        manager_id=1,
        from_gw=5,
        horizon=3,
        strategy="balanced",
        moves_by_gw=by_gw,
        hits_by_gw={5: 0, 6: 0, 7: 0},
        ft_by_gw={5: 1, 6: 1, 7: 1, 8: 2},
        bank_by_gw={5: 0.0, 6: 0.0, 7: 0.0},
        lineups_by_gw={gw: _lineup(gw) for gw in (5, 6, 7)},
        target_squad=list(range(1, 16)),
        expected_total=150.0,
        baseline_total=140.0,
        objective=120.0,
        baseline_objective=110.0,
        recommendation=recommendation,  # type: ignore[arg-type]
        solver="HiGHS",
        runtime_s=0.1,
    )


def test_diff_plans_detects_every_structural_change():
    prev = _plan([_move(5, 1, 101), _move(6, 2, 102), _move(6, 3, 103), _move(7, 4, 104)])
    new = _plan(
        [_move(5, 1, 111), _move(7, 2, 102), _move(6, 5, 105), _move(7, 4, 104)],
        recommendation="hold",
    )
    changes = diff_plans(prev, new)
    kinds = {(c.kind, tuple(c.player_ids)): c for c in changes}
    assert ("changed_target", (1, 101, 111)) in kinds  # тот же out, другой in
    assert kinds[("changed_target", (1, 101, 111))].gw == 5
    assert ("moved_gw", (2, 102)) in kinds and kinds[("moved_gw", (2, 102))].gw == 7
    assert ("added", (5, 105)) in kinds
    assert ("removed", (3, 103)) in kinds
    assert ("recommendation", ()) in kinds
    assert kinds[("recommendation", ())].detail == "transfers -> hold"
    # неизменённый ход 4 -> 104 в GW7 не попадает в diff
    assert not any(4 in c.player_ids for c in changes)
    assert len(changes) == 5


def test_diff_plans_identical_is_empty():
    plan = _plan([_move(5, 1, 101)])
    assert diff_plans(plan, plan) == []


def test_plan_json_roundtrip_keeps_aliases():
    plan = _plan([_move(5, 1, 101)])
    data = plan.model_dump(by_alias=True, mode="json")
    assert data["moves_by_gw"]["5"][0]["in"] == 101
    restored = TransferPlan.model_validate(data)
    assert restored.moves_by_gw[5][0].in_ == 101


@pytest.mark.db
def test_plan_snapshot_roundtrip_in_db():
    if not ping():
        pytest.skip("Postgres недоступен (docker compose up -d db)")
    with session_scope() as s:
        if s.execute(text("SELECT to_regclass('plan_snapshots')")).scalar() is None:
            pytest.skip("plan_snapshots не создана: uv run python scripts/migrate.py")
    manager_id = 999_999_001  # тестовый id, чтобы не мешать реальным снимкам
    plan = _plan([_move(5, 1, 101)], recommendation="hold")
    plan = plan.model_copy(update={"manager_id": manager_id})
    try:
        pid = save_plan(plan, inputs_hash="test")
        assert pid > 0
        loaded = latest_plan(manager_id, 5)
        assert loaded is not None and loaded.recommendation == "hold"
        assert loaded.moves_by_gw[5][0].in_ == 101
        assert latest_plan(manager_id, 5, "aggressive") is None
        assert diff_plans(plan, loaded) == []
    finally:
        with session_scope() as s:
            s.execute(text("DELETE FROM plan_snapshots WHERE manager_id = :m"), {"m": manager_id})


def test_inputs_hash_is_stable_and_sensitive():
    pool = {
        1: Candidate(
            player_id=1, name="A", position=3, team_id=1, price=5.0, xpts_by_gw={5: 3.004}
        ),
    }
    h1 = inputs_hash([1, 2, 3], 1.0, 1, "balanced", 3, 5, pool)
    h2 = inputs_hash([3, 2, 1], 1.0, 1, "balanced", 3, 5, pool)
    assert h1 == h2  # порядок состава не важен
    pool_noise = {
        1: Candidate(
            player_id=1, name="A", position=3, team_id=1, price=5.0, xpts_by_gw={5: 3.001}
        ),
    }
    assert inputs_hash([1, 2, 3], 1.0, 1, "balanced", 3, 5, pool_noise) == h1  # шум < 0.01
    assert inputs_hash([1, 2, 3], 1.5, 1, "balanced", 3, 5, pool) != h1
    assert inputs_hash([1, 2, 3], 1.0, 1, "aggressive", 3, 5, pool) != h1
