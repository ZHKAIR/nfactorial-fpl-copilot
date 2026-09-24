"""Константы скоринга FPL 2026/27 и разбор фактических очков по компонентам."""

from fplcopilot.core import scoring
from fplcopilot.data import Position
from fplcopilot.data.schemas import PlayerGWHistory


def _row(**kw) -> PlayerGWHistory:
    base = {"element": 1, "fixture": 1, "round": 1, "was_home": True}
    base.update(kw)
    return PlayerGWHistory.model_validate(base)


def test_goal_points_per_position():
    assert scoring.goal_points(Position.GKP) == 10
    assert scoring.goal_points(Position.DEF) == 6
    assert scoring.goal_points(Position.MID) == 5
    assert scoring.goal_points(Position.FWD) == 4


def test_clean_sheet_points_per_position():
    assert scoring.clean_sheet_points(Position.GKP) == 4
    assert scoring.clean_sheet_points(Position.DEF) == 4
    assert scoring.clean_sheet_points(Position.MID) == 1
    assert scoring.clean_sheet_points(Position.FWD) == 0


def test_defcon_thresholds_and_counts():
    assert scoring.defcon_threshold(Position.GKP) is None
    assert scoring.defcon_threshold(Position.DEF) == 10
    assert scoring.defcon_threshold(Position.MID) == 12
    assert scoring.defcon_threshold(Position.FWD) == 12
    row = _row(clearances_blocks_interceptions=5, tackles=3, recoveries=4)
    assert scoring.cbit(row) == 8
    assert scoring.cbirt(row) == 12
    assert scoring.defcon_count(row, Position.DEF) == 8  # CBIT
    assert scoring.defcon_count(row, Position.MID) == 12  # CBIRT
    assert scoring.concedes_penalty(Position.DEF) and not scoring.concedes_penalty(Position.MID)


def test_points_breakdown_matches_total_points_for_defender():
    # 90 мин, гол, сухарь, 2 пропущенных (нет — сухарь), 10 CBIT, 2 бонуса, жёлтая
    row = _row(
        minutes=90,
        goals_scored=1,
        assists=1,
        clean_sheets=1,
        goals_conceded=0,
        clearances_blocks_interceptions=7,
        tackles=3,
        bonus=2,
        yellow_cards=1,
        total_points=2 + 6 + 3 + 4 + 2 + 2 - 1,
    )
    parts = scoring.points_breakdown(row, Position.DEF)
    assert parts["appearance"] == 2
    assert parts["goals"] == 6
    assert parts["assists"] == 3
    assert parts["clean_sheet"] == 4
    assert parts["defcon"] == 2
    assert parts["bonus"] == 2
    assert parts["cards"] == -1
    assert sum(parts.values()) == row.total_points


def test_points_breakdown_goalkeeper_saves_and_conceded():
    row = _row(minutes=90, saves=7, goals_conceded=3, clean_sheets=0, total_points=2 + 2 - 1)
    parts = scoring.points_breakdown(row, Position.GKP)
    assert parts["saves"] == 2  # 7 // 3
    assert parts["goals_conceded"] == -1  # 3 // 2
    assert parts["clean_sheet"] == 0
    assert parts["defcon"] == 0
    assert sum(parts.values()) == row.total_points


def test_points_breakdown_sub_under_60_no_clean_sheet():
    row = _row(minutes=30, clean_sheets=1, total_points=1)
    parts = scoring.points_breakdown(row, Position.DEF)
    assert parts["appearance"] == 1
    assert parts["clean_sheet"] == 0
