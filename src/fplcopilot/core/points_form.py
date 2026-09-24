"""Раскладка очков за последние N туров и метка удачи / недобора / повторяемости.

Считаем из player_gw_history по правилам FPL (core/scoring.py). Live explain
`/api/event/{gw}/live/` не дергаем: история уже в БД (~3200 строк), сеть на рендер
не нужна. Редкие события без колонки в БД (автогол, незабитый пенальти) уходят в
`other`, чтобы сумма категорий = total_points. В xPts это не подмешивается.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from fplcopilot.config import settings
from fplcopilot.core import scoring
from fplcopilot.data.schemas import Player, PlayerGWHistory, Position

LABEL_LUCKY = "lucky"
LABEL_UNLUCKY = "unlucky"
LABEL_REPEATABLE = "repeatable"

LABEL_RU: dict[str, str] = {
    LABEL_LUCKY: "везение",
    LABEL_UNLUCKY: "не повезло",
    LABEL_REPEATABLE: "повторяемые",
}

# порядок как в карточке: плюсы, потом минусы
CATEGORY_ORDER: tuple[str, ...] = (
    "appearance",
    "goals",
    "assists",
    "clean_sheet",
    "saves",
    "bonus",
    "defcon",
    "goals_conceded",
    "cards",
    "own_goals",
    "penalties_missed",
    "penalties_saved",
    "other",
)

CATEGORY_RU: dict[str, str] = {
    "appearance": "выход",
    "goals": "голы",
    "assists": "ассисты",
    "clean_sheet": "сухие",
    "saves": "сейвы",
    "bonus": "бонус",
    "defcon": "защита",
    "goals_conceded": "пропущенные",
    "cards": "карточки",
    "own_goals": "автоголы",
    "penalties_missed": "пенальти мимо",
    "penalties_saved": "сейв пенальти",
    "other": "прочее",
}

# только стабильные источники; выход на поле сам по себе метку не даёт
REPEATABLE_KEYS: tuple[str, ...] = ("clean_sheet", "saves", "defcon")


@dataclass(frozen=True)
class PointsWindow:
    """Окно последних туров: категории, моменты, метка. Не вход модели xPts."""

    player_id: int
    last_n: int
    n_rounds: int
    total_points: int
    categories: dict[str, int]
    xg: float
    xa: float
    ga_points: int
    moments_points: float
    luck_delta: float
    label: str | None

    @property
    def label_ru(self) -> str | None:
        return LABEL_RU.get(self.label) if self.label else None

    @property
    def sums_to_total(self) -> bool:
        return sum(self.categories.values()) == self.total_points

    def as_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "last_n": self.last_n,
            "n_rounds": self.n_rounds,
            "total_points": self.total_points,
            "categories": dict(self.categories),
            "xg": round(self.xg, 3),
            "xa": round(self.xa, 3),
            "ga_points": self.ga_points,
            "moments_points": round(self.moments_points, 2),
            "luck_delta": round(self.luck_delta, 2),
            "label": self.label,
            "label_ru": self.label_ru,
            "sums_to_total": self.sums_to_total,
        }


def _last_n_rounds(rows: Sequence[PlayerGWHistory], n: int) -> list[PlayerGWHistory]:
    if not rows or n <= 0:
        return []
    rounds = sorted({int(r.round) for r in rows})
    keep = set(rounds[-n:])
    return [r for r in rows if int(r.round) in keep]


def _position(player: Player | None, position: Position | None) -> Position | None:
    if position is not None:
        return position
    if player is not None:
        return player.position
    return None


def _load_history(player_id: int) -> list[PlayerGWHistory]:
    from fplcopilot.core.history import load_history

    return load_history(player_ids=[int(player_id)])


def form_label(
    *,
    total_points: int,
    ga_points: int,
    moments_points: float,
    categories: dict[str, int],
    luck_over: float | None = None,
    luck_under: float | None = None,
    repeatable_share: float | None = None,
    repeatable_min_points: int | None = None,
    position: Position | None = None,
) -> str | None:
    """Метка по порогам: удача, недобор, повторяемые. Иначе None — молчим.

    Везение / «не повезло» по голам+ассистам — не для вратарей: у них только
    повторяемые источники (сухие, сейвы, защита).
    """
    over = getattr(settings, "why_luck_over", 5.0) if luck_over is None else luck_over
    under = getattr(settings, "why_luck_under", -4.0) if luck_under is None else luck_under
    share_cut = (
        getattr(settings, "why_repeatable_share", 0.50)
        if repeatable_share is None
        else repeatable_share
    )
    min_pts = (
        getattr(settings, "why_repeatable_min_points", 8)
        if repeatable_min_points is None
        else repeatable_min_points
    )
    delta = ga_points - moments_points
    allow_luck = position is None or position != Position.GKP
    if allow_luck:
        if delta >= over:
            return LABEL_LUCKY
        if delta <= under:
            return LABEL_UNLUCKY
    if total_points >= min_pts:
        repeatable = sum(int(categories.get(k, 0) or 0) for k in REPEATABLE_KEYS)
        if repeatable / max(total_points, 1) >= share_cut:
            return LABEL_REPEATABLE
    return None


def points_breakdown(
    player_id: int,
    last_n: int = 3,
    *,
    history: Sequence[PlayerGWHistory] | None = None,
    player: Player | None = None,
    position: Position | None = None,
) -> PointsWindow | None:
    """Раскладка очков игрока за последние last_n туров. Нет строк — None."""
    pos = _position(player, position)
    if pos is None:
        return None
    rows = list(history) if history is not None else None
    if rows is None:
        try:
            rows = _load_history(player_id)
        except Exception:  # noqa: BLE001 — БД/сеть не роняют карточку
            return None
    mine = [r for r in rows if int(r.element) == int(player_id)]
    window = _last_n_rounds(mine, last_n)
    if not window:
        return None
    totals = {k: 0 for k in CATEGORY_ORDER if k != "other"}
    xg = 0.0
    xa = 0.0
    total_points = 0
    for row in window:
        parts = scoring.points_breakdown(row, pos)
        for k, v in parts.items():
            totals[k] = totals.get(k, 0) + int(v)
        xg += float(row.expected_goals or 0.0)
        xa += float(row.expected_assists or 0.0)
        total_points += int(row.total_points)
    known = sum(totals.values())
    totals["other"] = total_points - known
    ordered = {k: int(totals.get(k, 0)) for k in CATEGORY_ORDER}
    ga_points = int(ordered.get("goals", 0)) + int(ordered.get("assists", 0))
    moments_points = xg * scoring.goal_points(pos) + xa * scoring.PTS_ASSIST
    luck_delta = ga_points - moments_points
    label = form_label(
        total_points=total_points,
        ga_points=ga_points,
        moments_points=moments_points,
        categories=ordered,
        position=pos,
    )
    return PointsWindow(
        player_id=int(player_id),
        last_n=last_n,
        n_rounds=len({int(r.round) for r in window}),
        total_points=total_points,
        categories=ordered,
        xg=xg,
        xa=xa,
        ga_points=ga_points,
        moments_points=moments_points,
        luck_delta=luck_delta,
        label=label,
    )


def compact_line(categories: dict[str, int]) -> str:
    """Короткая строка ненулевых категорий: «выход 6 · голы 10 · бонус 2»."""
    bits: list[str] = []
    for key in CATEGORY_ORDER:
        val = int(categories.get(key, 0) or 0)
        if val == 0:
            continue
        name = CATEGORY_RU.get(key, key)
        bits.append(f"{name} {val:+d}" if val < 0 else f"{name} {val}")
    return " · ".join(bits)
