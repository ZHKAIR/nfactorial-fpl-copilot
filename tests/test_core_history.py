"""Leakage guard, командная статистика по матчам и (db) идемпотентность upsert истории."""

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import text

from fplcopilot.core.history import rows_before, team_match_stats, upsert_history
from fplcopilot.data.schemas import Fixture, PlayerGWHistory

DEADLINE_GW3 = datetime(2026, 9, 4, 17, 30, tzinfo=UTC)


def _row(**kw) -> PlayerGWHistory:
    base = {"element": 1, "fixture": 1, "round": 1, "was_home": True}
    base.update(kw)
    return PlayerGWHistory.model_validate(base)


def test_rows_before_excludes_current_and_future_rounds():
    rows = [
        _row(round=1, kickoff_time="2026-08-21T19:00:00Z"),
        _row(round=2, kickoff_time="2026-08-29T14:00:00Z"),
        _row(round=3, kickoff_time="2026-09-05T14:00:00Z"),
        _row(round=4, kickoff_time="2026-09-12T14:00:00Z"),
    ]
    kept = rows_before(rows, 3)
    assert [r.round for r in kept] == [1, 2]
    assert rows_before(rows, 1) == []
    assert [r.round for r in rows_before(rows, 99)] == [1, 2, 3, 4]


def test_rows_before_excludes_postponed_match_played_after_deadline():
    postponed = _row(
        round=1, kickoff_time="2026-09-10T19:00:00Z"
    )  # раунд 1, сыгран после дедлайна GW3
    normal = _row(round=2, kickoff_time="2026-08-29T14:00:00Z")
    naive = _row(round=2, kickoff_time="2026-08-30T14:00:00")  # без tz -> считаем UTC
    kept = rows_before([postponed, normal, naive], 3, DEADLINE_GW3)
    assert kept == [normal, naive]
    assert rows_before([postponed], 3) == [postponed]  # без дедлайна фильтруем только по раунду


def test_team_match_stats_sums_player_xg_per_side():
    fixtures = [Fixture.model_validate({"id": 10, "team_h": 1, "team_a": 2, "event": 1})]
    rows = [
        _row(
            element=1,
            fixture=10,
            round=1,
            was_home=True,
            expected_goals=0.8,
            team_h_score=2,
            team_a_score=1,
        ),
        _row(
            element=2,
            fixture=10,
            round=1,
            was_home=True,
            expected_goals=0.7,
            team_h_score=2,
            team_a_score=1,
        ),
        _row(
            element=3,
            fixture=10,
            round=1,
            was_home=False,
            expected_goals=0.4,
            team_h_score=2,
            team_a_score=1,
        ),
        _row(
            element=4, fixture=11, round=2, was_home=False, expected_goals=9.0
        ),  # раунд 2 -> вне окна
    ]
    stats = team_match_stats(1, rows=rows, fixtures=fixtures)
    assert set(stats) == {(10, 1), (10, 2)}
    home, away = stats[(10, 1)], stats[(10, 2)]
    assert (
        home.xg_for == 1.5
        and home.xg_against == 0.4
        and home.goals_for == 2
        and home.goals_against == 1
    )
    assert (
        away.xg_for == 0.4 and away.xg_against == 1.5 and away.opponent == 1 and not away.was_home
    )


@pytest.mark.db
def test_upsert_history_is_idempotent_and_updates_in_place():
    from fplcopilot.db import ping, session_scope

    if not ping():
        pytest.skip("Postgres недоступен")
    pid = 900_000 + uuid.uuid4().int % 90_000
    row = _row(element=pid, fixture=9_999, round=1, minutes=90, bonus=1, total_points=5)
    with session_scope() as s:
        if s.execute(text("SELECT to_regclass('player_gw_history')")).scalar() is None:
            pytest.skip("таблицы не созданы: uv run python scripts/migrate.py")
        try:
            assert upsert_history(s, pid, [row]) == 1
            assert upsert_history(s, pid, [row]) == 1  # повтор — та же строка
            updated = row.model_copy(update={"bonus": 3, "total_points": 7})
            upsert_history(s, pid, [updated])
            n, bonus, pts = s.execute(
                text(
                    "SELECT count(*), max(bonus), max(total_points) FROM player_gw_history "
                    "WHERE player_id = :p"
                ),
                {"p": pid},
            ).one()
            assert (n, bonus, pts) == (1, 3, 7)
        finally:
            s.execute(text("DELETE FROM player_gw_history WHERE player_id = :p"), {"p": pid})
            s.commit()
