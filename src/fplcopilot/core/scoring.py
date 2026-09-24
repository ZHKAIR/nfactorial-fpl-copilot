"""Правила начисления очков FPL 2026/27 — единственное место с константами скоринга.

Источник: официальные правила FPL (Rules -> Scoring). DefCon введён в 2025/26:
DEF — 2 очка за >= 10 CBIT (clearances + blocks + interceptions + tackles) за матч,
MID/FWD — 2 очка за >= 12 CBIRT (CBIT + recoveries); максимум 2 очка за матч.
"""

from __future__ import annotations

from fplcopilot.data import Position
from fplcopilot.data.schemas import PlayerGWHistory

PTS_APPEARANCE = 1  # любое время на поле
PTS_APPEARANCE_60 = 1  # дополнительно за >= 60 минут
PTS_GOAL: dict[Position, int] = {
    Position.GKP: 10,
    Position.DEF: 6,
    Position.MID: 5,
    Position.FWD: 4,
}
PTS_ASSIST = 3
PTS_CLEAN_SHEET: dict[Position, int] = {
    Position.GKP: 4,
    Position.DEF: 4,
    Position.MID: 1,
    Position.FWD: 0,
}
CLEAN_SHEET_MIN_MINUTES = 60
SAVES_PER_POINT = 3  # GKP: 1 очко за каждые 3 сейва
PTS_PENALTY_SAVE = 5
GOALS_CONCEDED_PER_POINT = 2  # GKP/DEF: -1 за каждые 2 пропущенных
PTS_YELLOW = -1
PTS_RED = -3
PTS_OWN_GOAL = -2
PTS_PENALTY_MISS = -2
BONUS_MAX = 3

DEFCON_POINTS = 2
DEFCON_THRESHOLD: dict[Position, int | None] = {
    Position.GKP: None,  # вратари DefCon не получают
    Position.DEF: 10,  # CBIT
    Position.MID: 12,  # CBIRT
    Position.FWD: 12,  # CBIRT
}


def setpiece_xpts(
    *,
    penalty_order: int | None,
    dfk_order: int | None,
    penalty_weight: float,
    dfk_weight: float,
) -> float:
    """Аддитивный бонус за тур: только первый пенальтист / первый штатник прямых штрафных."""
    bonus = 0.0
    if penalty_order == 1:
        bonus += penalty_weight
    if dfk_order == 1:
        bonus += dfk_weight
    return bonus


def goal_points(pos: Position) -> int:
    return PTS_GOAL[pos]


def clean_sheet_points(pos: Position) -> int:
    return PTS_CLEAN_SHEET[pos]


def concedes_penalty(pos: Position) -> bool:
    """Штраф за пропущенные голы получают только GKP и DEF."""
    return pos in (Position.GKP, Position.DEF)


def defcon_threshold(pos: Position) -> int | None:
    return DEFCON_THRESHOLD[pos]


def defcon_uses_recoveries(pos: Position) -> bool:
    return pos in (Position.MID, Position.FWD)


def cbit(row: PlayerGWHistory) -> int:
    return row.clearances_blocks_interceptions + row.tackles


def cbirt(row: PlayerGWHistory) -> int:
    return cbit(row) + row.recoveries


def defcon_count(row: PlayerGWHistory, pos: Position) -> int:
    """Оборонительные действия, которые считаются для DefCon в позиции pos.

    Совпадает с полем defensive_contribution в FPL API (проверено на всех строках GW1–4).
    """
    return cbirt(row) if defcon_uses_recoveries(pos) else cbit(row)


def points_breakdown(row: PlayerGWHistory, pos: Position) -> dict[str, int]:
    """Фактические очки матча по правилам FPL 2026/27.

    Автоголы, незабитые и отбитые пенальти входят, если поля заполнены (в БД их может не быть —
    тогда 0). Сумма окна сходится с total_points через остаток `other` в points_form.
    В xPts эти редкие события по-прежнему не моделируются.
    """
    played = row.minutes > 0
    sixty = row.minutes >= CLEAN_SHEET_MIN_MINUTES
    thr = defcon_threshold(pos)
    own_goals = int(getattr(row, "own_goals", 0) or 0)
    pen_miss = int(getattr(row, "penalties_missed", 0) or 0)
    pen_save = int(getattr(row, "penalties_saved", 0) or 0)
    return {
        "appearance": (PTS_APPEARANCE if played else 0) + (PTS_APPEARANCE_60 if sixty else 0),
        "goals": row.goals_scored * PTS_GOAL[pos],
        "assists": row.assists * PTS_ASSIST,
        "clean_sheet": row.clean_sheets * PTS_CLEAN_SHEET[pos] if sixty else 0,
        "goals_conceded": (
            -(row.goals_conceded // GOALS_CONCEDED_PER_POINT) if concedes_penalty(pos) else 0
        ),
        "saves": row.saves // SAVES_PER_POINT if pos == Position.GKP else 0,
        "defcon": DEFCON_POINTS if thr is not None and defcon_count(row, pos) >= thr else 0,
        "bonus": row.bonus,
        "cards": row.yellow_cards * PTS_YELLOW + row.red_cards * PTS_RED,
        "own_goals": own_goals * PTS_OWN_GOAL,
        "penalties_missed": pen_miss * PTS_PENALTY_MISS,
        "penalties_saved": pen_save * PTS_PENALTY_SAVE if pos == Position.GKP else 0,
    }
