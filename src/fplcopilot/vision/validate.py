"""Правила FPL для состава со скриншота — детерминированно, каждое нарушение -> SquadIssue.

Blocking (состав нельзя передавать дальше): нераспознанный / неоднозначный / дублирующийся игрок,
размер не 15, распределение по позициям не 2/5/5/3, больше 3 игроков одного клуба.
Warnings (оптимизатор исправит сам, пользователю просто показываем): старт не 11, невалидная
схема, капитан/вице не ровно один или на скамейке, вратарь скамейки не первый, скамейка не
распознана (экран трансферов), расхождения подсказок со скриншота (позиция/цена/клуб) с bootstrap,
сумма цен на экране далеко от bootstrap (устаревший скриншот) — информационно.
"""

from __future__ import annotations

from collections import Counter

from fplcopilot.vision.resolve import PRICE_TOLERANCE, team_matches_hint
from fplcopilot.vision.schemas import (
    FORMATION_LIMITS,
    MAX_PER_CLUB,
    POSITION_ORDER,
    SQUAD_BY_POSITION,
    SQUAD_SIZE,
    XI_SIZE,
    ResolvedPlayer,
    SquadIssue,
)

TEAM_VALUE_TOLERANCE = 1.0  # £m: суммарное расхождение цен экран vs bootstrap, выше — информируем


def _blocking(code: str, message: str) -> SquadIssue:
    return SquadIssue(code=code, message=message, blocking=True)


def _warning(code: str, message: str) -> SquadIssue:
    return SquadIssue(code=code, message=message, blocking=False)


def _names(players: list[ResolvedPlayer]) -> str:
    return ", ".join(p.label for p in players)


def validate_squad(players: list[ResolvedPlayer], *, bank: float | None = None) -> list[SquadIssue]:
    """Все нарушения для списка карточек после резолюции. Порядок: blocking-правила, затем warnings."""
    issues: list[SquadIssue] = []

    # --- распознавание: каждый нераспознанный/неоднозначный — отдельное blocking ---
    for p in players:
        if p.match_method == "ambiguous":
            issues.append(
                _blocking(
                    "ambiguous",
                    f"ambiguous: {p.name_as_shown} — candidates [{', '.join(p.candidates)}]",
                )
            )
        elif not p.resolved:
            issues.append(
                _blocking("unresolved", f"unresolved: '{p.name_as_shown}' not found in FPL players")
            )

    # --- дубликаты ---
    by_id: dict[int, list[ResolvedPlayer]] = {}
    for p in players:
        if p.resolved:
            by_id.setdefault(p.player_id, []).append(p)  # type: ignore[arg-type]
    for dupes in by_id.values():
        if len(dupes) > 1:
            issues.append(_blocking("duplicate", f"duplicate: {dupes[0].label} ×{len(dupes)}"))

    # --- размер состава (по карточкам: нераспознанная карточка уже учтена выше) ---
    if len(players) != SQUAD_SIZE:
        issues.append(_blocking("squad_size", f"squad size {len(players)} ≠ {SQUAD_SIZE}"))

    # --- по позициям: 2 GKP / 5 DEF / 5 MID / 3 FWD (позиция bootstrap, иначе — ряд на экране) ---
    pos_counts = Counter(p.position for p in players if p.position)
    for pos in POSITION_ORDER:
        need = SQUAD_BY_POSITION[pos]
        if pos_counts[pos] != need:
            issues.append(_blocking("position_count", f"{pos}: {pos_counts[pos]} ≠ {need}"))

    # --- не больше 3 из одного клуба ---
    club_counts = Counter(p.team_short for p in players if p.resolved and p.team_short)
    for club, n in sorted(club_counts.items()):
        if n > MAX_PER_CLUB:
            issues.append(_blocking("club_limit", f"{club}: {n} players > {MAX_PER_CLUB}"))

    # --- старт / скамейка ---
    has_bench = any(p.is_bench for p in players)
    starters = [p for p in players if not p.is_bench]
    bench = sorted(
        (p for p in players if p.is_bench),
        key=lambda p: (p.bench_order is None, p.bench_order or 0),
    )
    if not has_bench:
        issues.append(
            _warning("no_bench", "no bench detected — lineup unknown, a default XI will be chosen")
        )
    else:
        if len(starters) != XI_SIZE:
            issues.append(
                _warning("xi_count", f"starting XI has {len(starters)} players ≠ {XI_SIZE}")
            )
        xi_counts = Counter(p.position for p in starters if p.position)
        broken = [
            f"{pos} {xi_counts[pos]} not in {lo}..{hi}"
            for pos, (lo, hi) in FORMATION_LIMITS.items()
            if not lo <= xi_counts[pos] <= hi
        ]
        if broken:
            shape = "-".join(str(xi_counts[pos]) for pos in POSITION_ORDER[1:])
            issues.append(_warning("formation", f"formation {shape} invalid: {'; '.join(broken)}"))
        if bench and bench[0].position != "GKP":
            issues.append(
                _warning("bench_gk", f"bench: goalkeeper must be first, got {bench[0].label}")
            )

    # --- капитан и вице: ровно по одному, оба в старте, не один и тот же ---
    captains = [p for p in players if p.is_captain]
    vices = [p for p in players if p.is_vice_captain]
    if len(captains) != 1:
        msg = (
            "no captain badge detected"
            if not captains
            else f"{len(captains)} captains: {_names(captains)}"
        )
        issues.append(_warning("captain", msg))
    if len(vices) != 1:
        msg = (
            "no vice-captain badge detected"
            if not vices
            else f"{len(vices)} vice-captains: {_names(vices)}"
        )
        issues.append(_warning("vice", msg))
    if has_bench:
        for p in captains:
            if p.is_bench:
                issues.append(_warning("captain", f"captain {p.label} is on the bench"))
        for p in vices:
            if p.is_bench:
                issues.append(_warning("vice", f"vice-captain {p.label} is on the bench"))
    if (
        captains
        and vices
        and any(
            c is v or (c.resolved and c.player_id == v.player_id) for c in captains for v in vices
        )
    ):
        issues.append(_warning("captain", "captain and vice-captain are the same player"))

    # --- подсказки со скриншота против bootstrap (игрок найден, но что-то не сходится) ---
    for p in players:
        if not p.resolved:
            continue
        if p.position_as_shown and p.position and p.position_as_shown != p.position:
            issues.append(
                _warning(
                    "position_mismatch",
                    f"{p.label}: shown in {p.position_as_shown} row, FPL position {p.position}",
                )
            )
        if (
            p.price is not None
            and p.price_as_shown is not None
            and abs(p.price - p.price_as_shown) > PRICE_TOLERANCE + 1e-9
        ):
            issues.append(
                _warning(
                    "price_mismatch",
                    f"{p.label}: shown £{p.price_as_shown:.1f}m, FPL now £{p.price:.1f}m",
                )
            )
        if (
            p.club_hint
            and p.team_short
            and not team_matches_hint(p.team_short, p.team_name or "", p.club_hint)
        ):
            issues.append(
                _warning(
                    "club_mismatch",
                    f"{p.label}: club hint '{p.club_hint}' but FPL club is {p.team_short}",
                )
            )

    # --- сумма цен: информационно (устаревший скриншот / цены изменились) ---
    resolved = [p for p in players if p.resolved and p.price is not None]
    if resolved and all(p.price_as_shown is not None for p in resolved):
        shown_total = sum(p.price_as_shown or 0.0 for p in resolved)
        now_total = sum(p.price or 0.0 for p in resolved)
        if abs(shown_total - now_total) > TEAM_VALUE_TOLERANCE:
            bank_note = f" + bank £{bank:.1f}m" if bank is not None else ""
            issues.append(
                _warning(
                    "team_value",
                    f"prices as shown sum to £{shown_total:.1f}m{bank_note}, "
                    f"FPL prices now £{now_total:.1f}m — screenshot may be stale",
                )
            )
    return issues


def is_valid(issues: list[SquadIssue]) -> bool:
    return not any(i.blocking for i in issues)
