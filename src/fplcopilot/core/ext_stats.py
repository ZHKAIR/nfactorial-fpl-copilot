"""Внешняя стата поверх FPL: стандарты из bootstrap + Understat xG/xA/shots/xGA.

Основной сигнал — компонентная модель v0. Здесь только маленький аддитивный бонус
(пенальтист / штатник) и сжатый blend per-90 (`XPTS_UNDERSTAT_BLEND`, сдвиг ≤ ±15 %).
Фолы и стандарты соперника Understat стабильно не отдаёт — FBref в этой итерации не скрапим.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from fplcopilot.config import settings
from fplcopilot.core import scoring
from fplcopilot.core.odds import canonical_team, team_index
from fplcopilot.core.understat import (
    UnderstatClient,
    UnderstatPlayer,
    UnderstatSnapshot,
    UnderstatTeam,
    match_players,
)
from fplcopilot.data.schemas import Bootstrap, Player, Position, Team

log = logging.getLogger(__name__)

FOULS_NEXT_SOURCE = "FBref (не скрапим в этой итерации: нестабильная вёрстка, нет публичного JSON)"


@dataclass(frozen=True)
class PlayerExt:
    """Сводка для UI / MCP: FPL-стандарты + Understat, если имя сопоставилось."""

    player_id: int
    web_name: str
    penalties_order: int | None
    direct_freekicks_order: int | None
    corners_order: int | None
    saves: int
    fpl_xg: float | None
    fpl_xa: float | None
    fpl_minutes: int
    fpl_xg90: float | None
    fpl_xa90: float | None
    fpl_saves90: float | None
    understat: UnderstatPlayer | None
    setpiece_bonus: float
    understat_blend: float
    sources: tuple[str, ...]

    @property
    def matched(self) -> bool:
        return self.understat is not None


@dataclass
class ExtIndex:
    snapshot: UnderstatSnapshot
    by_player: dict[int, UnderstatPlayer] = field(default_factory=dict)
    by_team: dict[int, UnderstatTeam] = field(default_factory=dict)
    unmatched: list[UnderstatPlayer] = field(default_factory=list)
    matched_n: int = 0
    fpl_n: int = 0

    def player_row(self, player_id: int) -> UnderstatPlayer | None:
        return self.by_player.get(player_id)

    def team_row(self, team_id: int) -> UnderstatTeam | None:
        return self.by_team.get(team_id)


_INDEX: ExtIndex | None = None


def setpiece_bonus_xpts(
    player: Player,
    *,
    penalty_weight: float | None = None,
    freekick_weight: float | None = None,
) -> float:
    """Аддитивный бонус за тур: только order=1. Веса — `XPTS_SETPIECE_WEIGHT` / `XPTS_FREEKICK_WEIGHT`."""
    return scoring.setpiece_xpts(
        penalty_order=player.penalties_order,
        dfk_order=player.direct_freekicks_order,
        penalty_weight=settings.xpts_setpiece_weight if penalty_weight is None else penalty_weight,
        dfk_weight=settings.xpts_freekick_weight if freekick_weight is None else freekick_weight,
    )


def setpiece_phrase(player: Player, team: Team | str) -> str | None:
    """Одна фраза для UI: «Calafiori бьёт пенальти Arsenal». None — не штатник."""
    club = team.name if isinstance(team, Team) else team
    name = player.web_name
    if player.penalties_order == 1:
        return f"{name} бьёт пенальти {club}"
    if player.direct_freekicks_order == 1:
        return f"{name} бьёт штрафные {club}"
    if player.corners_and_indirect_freekicks_order == 1:
        return f"{name} подаёт угловые {club}"
    return None


def setpiece_label(player: Player) -> str:
    """Короткая метка: «пенальтист №1», «штатник №2», иначе «—»."""
    parts: list[str] = []
    if player.penalties_order:
        parts.append(f"пенальтист №{player.penalties_order}")
    if player.direct_freekicks_order:
        parts.append(f"штатник №{player.direct_freekicks_order}")
    if player.corners_and_indirect_freekicks_order and not parts:
        parts.append(f"угловые №{player.corners_and_indirect_freekicks_order}")
    return ", ".join(parts) if parts else "—"


def _per90(total: float | None, minutes: int) -> float | None:
    if total is None or minutes <= 0:
        return None
    return float(total) * 90.0 / minutes


def blend_rate(ours: float, theirs: float | None, *, weight: float, cap: float) -> float:
    """`weight`·Understat + `(1-weight)`·наш; сдвиг не больше ±cap от нашего per-90."""
    if theirs is None or weight <= 0:
        return ours
    raw = (1.0 - weight) * ours + weight * theirs
    lo, hi = ours * (1.0 - cap), ours * (1.0 + cap)
    if lo > hi:
        lo, hi = hi, lo
    return min(hi, max(lo, raw))


def blend_rates_with_understat(rates: Any, row: UnderstatPlayer | None) -> Any:
    """Сжимает только xG90/xA90; остальные ставки не трогаем. `rates` — `xpts.Rates`."""
    if row is None:
        return rates
    w = settings.xpts_understat_blend
    cap = settings.xpts_understat_shift_cap
    if w <= 0:
        return rates
    return type(rates)(
        xg=blend_rate(rates.xg, row.xg90, weight=w, cap=cap),
        xa=blend_rate(rates.xa, row.xa90, weight=w, cap=cap),
        saves=rates.saves,
        bps=rates.bps,
        cbit=rates.cbit,
        cbirt=rates.cbirt,
        yellow=rates.yellow,
    )


def player_ext(player: Player, index: ExtIndex | None = None) -> PlayerExt:
    row = index.player_row(player.id) if index is not None else None
    sources = ["fpl"]
    if row is not None:
        sources.append("understat")
    mins = player.minutes
    return PlayerExt(
        player_id=player.id,
        web_name=player.web_name,
        penalties_order=player.penalties_order,
        direct_freekicks_order=player.direct_freekicks_order,
        corners_order=player.corners_and_indirect_freekicks_order,
        saves=player.saves,
        fpl_xg=player.expected_goals,
        fpl_xa=player.expected_assists,
        fpl_minutes=mins,
        fpl_xg90=_per90(player.expected_goals, mins),
        fpl_xa90=_per90(player.expected_assists, mins),
        fpl_saves90=_per90(float(player.saves), mins) if player.position == Position.GKP else None,
        understat=row,
        setpiece_bonus=setpiece_bonus_xpts(player),
        understat_blend=settings.xpts_understat_blend,
        sources=tuple(sources),
    )


def player_advanced_stats(
    player: Player, bs: Bootstrap, *, index: ExtIndex | None = None
) -> dict[str, Any]:
    """Числа + источники для MCP `get_player_advanced_stats`. Без коэффициентов."""
    ext = player_ext(player, index)
    u = ext.understat
    team = bs.team(player.team)
    return {
        "player_id": player.id,
        "web_name": player.web_name,
        "full_name": player.full_name,
        "team": team.short_name,
        "position": player.position.short,
        "fpl": {
            "expected_goals": player.expected_goals,
            "expected_assists": player.expected_assists,
            "expected_goals_conceded": player.expected_goals_conceded,
            "saves": player.saves,
            "clean_sheets": player.clean_sheets,
            "minutes": player.minutes,
            "xg90": ext.fpl_xg90,
            "xa90": ext.fpl_xa90,
            "saves90": ext.fpl_saves90,
            "penalties_order": player.penalties_order,
            "direct_freekicks_order": player.direct_freekicks_order,
            "corners_and_indirect_freekicks_order": player.corners_and_indirect_freekicks_order,
        },
        "understat": (
            None
            if u is None
            else {
                "player_name": u.player_name,
                "team_title": u.team_title,
                "minutes": u.minutes,
                "xg": u.xg,
                "xa": u.xa,
                "npxg": u.npxg,
                "shots": u.shots,
                "xg90": u.xg90,
                "xa90": u.xa90,
                "npxg90": u.npxg90,
                "shots90": u.shots90,
                "season": index.snapshot.season if index else settings.understat_season,
            }
        ),
        "setpiece_label": setpiece_label(player),
        "setpiece_phrase": setpiece_phrase(player, team),
        "setpiece_bonus": ext.setpiece_bonus,
        "understat_blend": ext.understat_blend,
        "matched": ext.matched,
        "sources": list(ext.sources),
        "note": (
            index.snapshot.note
            if index and index.snapshot.note
            else (
                None
                if ext.matched
                else "игрок не сопоставлен с Understat — показываем только поля FPL"
            )
        ),
    }


def team_defensive_profile(
    team: Team, bs: Bootstrap, *, index: ExtIndex | None = None
) -> dict[str, Any]:
    """xGA команды как «мало пропускает». Фолы не берём — нет стабильного источника."""
    row = index.team_row(team.id) if index is not None else None
    league_xga: list[float] = []
    if index is not None:
        league_xga = [
            t.xga_per_game for t in index.by_team.values() if t.xga_per_game is not None
        ]
    median = sorted(league_xga)[len(league_xga) // 2] if league_xga else None
    xga_pg = row.xga_per_game if row else None
    concedes_little = (
        xga_pg is not None and median is not None and xga_pg < median
    )
    gk_cs = sum(
        p.clean_sheets
        for p in bs.elements
        if p.team == team.id and p.position == Position.GKP
    )
    sources = ["fpl"]
    if row is not None:
        sources.append("understat")
    return {
        "team_id": team.id,
        "name": team.name,
        "short_name": team.short_name,
        "understat": (
            None
            if row is None
            else {
                "title": row.title,
                "matches": row.matches,
                "xg": row.xg,
                "xga": row.xga,
                "xg_per_game": row.xg_per_game,
                "xga_per_game": row.xga_per_game,
                "season": index.snapshot.season if index else settings.understat_season,
            }
        ),
        "fpl_gk_clean_sheets": gk_cs,
        "league_xga_median": median,
        "concedes_little": concedes_little,
        "pens_won": int(getattr(row, "pens_won", 0) or 0) if row is not None else 0,
        "pens_conceded": int(getattr(row, "pens_conceded", 0) or 0) if row is not None else 0,
        "pens_matches": int(getattr(row, "matches", 0) or 0) if row is not None else 0,
        "sources": sources,
        "note": (
            "«Мало пропускает» — командный xGA Understat ниже медианы лиги. "
            f"Фолы / стандарты соперника не берём: {FOULS_NEXT_SOURCE}."
        ),
    }


def build_index(
    bs: Bootstrap,
    snapshot: UnderstatSnapshot,
) -> ExtIndex:
    matched, unmatched = match_players(bs.elements, snapshot.players, bs.teams)
    t_index = team_index(bs.teams)
    by_team: dict[int, UnderstatTeam] = {}
    for t in snapshot.teams:
        tid = t_index.get(canonical_team(t.title))
        if tid is None:
            log.warning("understat: клуб %r не сопоставлен с FPL", t.title)
            continue
        by_team[tid] = t
    return ExtIndex(
        snapshot=snapshot,
        by_player=matched,
        by_team=by_team,
        unmatched=unmatched,
        matched_n=len(matched),
        fpl_n=len(bs.elements),
    )


def load_ext_index(
    bs: Bootstrap,
    *,
    client: UnderstatClient | None = None,
    snapshot: UnderstatSnapshot | None = None,
    force: bool = False,
) -> ExtIndex:
    """Индекс FPL↔Understat. Сеть — только если нет снимка; никогда не бросает."""
    global _INDEX
    if snapshot is None:
        if (
            not force
            and _INDEX is not None
            and _INDEX.fpl_n == len(bs.elements)
            and _INDEX.snapshot.season == settings.understat_season
        ):
            return _INDEX
        client = client or UnderstatClient()
        snapshot = client.snapshot(force=force)
    index = build_index(bs, snapshot)
    _INDEX = index
    log.info(
        "understat: сопоставлено %d/%d игроков FPL, %d клубов, source=%s",
        index.matched_n,
        index.fpl_n,
        len(index.by_team),
        snapshot.source,
    )
    return index


def reset_ext_index() -> None:
    global _INDEX
    _INDEX = None


def peek_ext_index() -> ExtIndex | None:
    return _INDEX
