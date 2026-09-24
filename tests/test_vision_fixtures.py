"""Мини-bootstrap для тестов vision (без сети и без .cache): реальные имена/клубы/цены на 17.09.2026,
включая ловушки резолюции: «Gabriel» ×4 (web_name у одного, first_name у трёх), «Fernandes» ×2,
«Palmer» ×2, «João Pedro» (web_name) vs Costinha (first_name «João Pedro»), «Silva» как фамилия у
четырёх, «Timber» ×2, «Pedro» как имя у двух."""

from __future__ import annotations

from datetime import UTC, datetime

from fplcopilot.data.schemas import Bootstrap, Event, Player, Team
from fplcopilot.vision.schemas import RawPlayer, ResolvedPlayer

AS_OF = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)

TEAMS = {
    1: ("Arsenal", "ARS"), 2: ("Aston Villa", "AVL"), 3: ("Bournemouth", "BOU"),
    5: ("Brighton", "BHA"), 6: ("Chelsea", "CHE"), 8: ("Crystal Palace", "CRY"),
    12: ("Ipswich", "IPS"), 13: ("Leeds", "LEE"), 14: ("Liverpool", "LIV"), 15: ("Man City", "MCI"),
    16: ("Man Utd", "MUN"), 18: ("Nott'm Forest", "NFO"), 19: ("Spurs", "TOT"),
}  # fmt: skip

# (id, web_name, first_name, second_name, team, element_type, now_cost)
PLAYERS = [
    (1, "Raya", "David", "Raya Martín", 1, 1, 60),
    (467, "Sels", "Matz", "Sels", 18, 1, 50),
    (301, "Palmer", "Alex", "Palmer", 12, 1, 40),
    (4, "Gabriel", "Gabriel", "dos Santos Magalhães", 1, 2, 80),
    (331, "Gudmundsson", "Gabriel", "Gudmundsson", 13, 2, 45),
    (356, "Virgil", "Virgil", "van Dijk", 14, 2, 65),
    (391, "Gvardiol", "Joško", "Gvardiol", 15, 2, 57),
    (498, "Senesi", "Marcos", "Senesi Barón", 19, 2, 58),
    (5, "J.Timber", "Jurriën", "Timber", 1, 2, 65),
    (566, "Silva", "António João", "Pereira de Albuquerque Tavares da Silva", 3, 2, 50),
    (470, "Morato", "Felipe", "Rodrigues da Silva", 18, 2, 50),
    (119, "Costinha", "João Pedro", "Loureiro da Costa", 5, 2, 45),
    (499, "Pedro Porro", "Pedro", "Porro Sauceda", 19, 2, 55),
    (18, "Martinelli", "Gabriel", "Martinelli Silva", 1, 3, 63),
    (644, "Timber", "Quinten", "Timber", 8, 3, 50),
    (426, "B.Fernandes", "Bruno", "Borges Fernandes", 16, 3, 120),
    (525, "Fernandes", "Mateus", "Fernandes", 19, 3, 58),
    (397, "Semenyo", "Antoine", "Semenyo", 15, 3, 84),
    (427, "Mbeumo", "Bryan", "Mbeumo", 16, 3, 79),
    (14, "Eze", "Eberechi", "Eze", 1, 3, 63),
    (40, "Rogers", "Morgan", "Rogers", 6, 3, 77),
    (12, "Saka", "Bukayo", "Saka", 1, 3, 95),
    (154, "Palmer", "Cole", "Palmer", 6, 3, 97),
    (156, "Neto", "Pedro", "Lomba Neto", 6, 3, 65),
    (27, "G.Jesus", "Gabriel", "Fernando de Jesus", 1, 4, 59),
    (165, "João Pedro", "João Pedro", "Junqueira de Jesus", 6, 4, 78),
    (411, "Haaland", "Erling", "Haaland", 15, 4, 155),
    (55, "Watkins", "Ollie", "Watkins", 2, 4, 78),
    (490, "Wood", "Chris", "Wood", 18, 4, 58),
]


def make_bootstrap() -> Bootstrap:
    teams = [Team(id=i, name=n, short_name=s) for i, (n, s) in TEAMS.items()]
    players = [
        Player(
            id=pid, web_name=web, first_name=first, second_name=second,
            team=team, element_type=et, now_cost=cost,
        )
        for pid, web, first, second, team, et, cost in PLAYERS
    ]  # fmt: skip
    events = [
        Event(id=4, name="Gameweek 4", deadline_time=datetime(2026, 9, 13, 10, 0, tzinfo=UTC), finished=True, is_current=True),
        Event(id=5, name="Gameweek 5", deadline_time=datetime(2026, 9, 20, 12, 0, tzinfo=UTC), is_next=True),
        Event(id=6, name="Gameweek 6", deadline_time=datetime(2026, 9, 27, 12, 0, tzinfo=UTC)),
    ]  # fmt: skip
    return Bootstrap(events=events, teams=teams, elements=players)


def raw(
    name: str,
    position: str | None = None,
    *,
    club: str | None = None,
    price: float | None = None,
    captain: bool = False,
    vice: bool = False,
    bench: int | None = None,
    flagged: bool = False,
) -> RawPlayer:
    return RawPlayer(
        name_as_shown=name,
        position=position,  # type: ignore[arg-type]
        club_hint=club,
        price_as_shown=price,
        is_captain=captain,
        is_vice_captain=vice,
        is_bench=bench is not None,
        bench_order=bench,
        is_flagged=flagged,
    )


def valid_raw_players() -> list[RawPlayer]:
    """15 карточек эталонного валидного скриншота (samples/pick_team_valid.png), 4-4-2."""
    return [
        raw("Raya", "GKP", price=6.0),
        raw("Gabriel", "DEF", price=8.0),
        raw("Virgil", "DEF", price=6.5),
        raw("Gvardiol", "DEF", price=5.7, flagged=True),
        raw("Senesi", "DEF", price=5.8),
        raw("B.Fernandes", "MID", price=12.0, captain=True),
        raw("Semenyo", "MID", price=8.4),
        raw("Mbeumo", "MID", price=7.9),
        raw("Eze", "MID", price=6.3),
        raw("João Pedro", "FWD", price=7.8, vice=True),
        raw("Watkins", "FWD", price=7.8),
        raw("Sels", "GKP", price=5.0, bench=1),
        raw("Gudmundsson", "DEF", price=4.5, bench=2),
        raw("Rogers", "MID", price=7.7, bench=3),
        raw("Wood", "FWD", price=5.8, bench=4),
    ]


def rp(
    name: str,
    position: str | None,
    *,
    pid: int | None = 1000,
    team: str | None = "XXX",
    price: float | None = 5.0,
    shown_price: float | None = None,
    shown_position: str | None = None,
    club_hint: str | None = None,
    captain: bool = False,
    vice: bool = False,
    bench: int | None = None,
    method: str | None = None,
    candidates: list[str] | None = None,
) -> ResolvedPlayer:
    """Быстрый конструктор ResolvedPlayer для тестов validate/to_squad (без резолюции)."""
    resolved = pid is not None and method not in ("ambiguous", "unresolved")
    return ResolvedPlayer(
        name_as_shown=name,
        player_id=pid if resolved else None,
        web_name=name if resolved else None,
        position=position,  # type: ignore[arg-type]
        position_as_shown=shown_position or position,  # type: ignore[arg-type]
        team_short=team if resolved else None,
        team_name=team if resolved else None,
        price=price if resolved else None,
        price_as_shown=shown_price if shown_price is not None else price,
        club_hint=club_hint,
        match_confidence=1.0 if resolved else 0.0,
        match_method=method or ("exact" if resolved else "unresolved"),  # type: ignore[arg-type]
        candidates=candidates or [],
        is_captain=captain,
        is_vice_captain=vice,
        is_bench=bench is not None,
        bench_order=bench,
    )


def valid_resolved_players() -> list[ResolvedPlayer]:
    """Те же 15, уже «резолвнутые» с id из мини-bootstrap."""
    return [
        rp("Raya", "GKP", pid=1, team="ARS", price=6.0),
        rp("Gabriel", "DEF", pid=4, team="ARS", price=8.0),
        rp("Virgil", "DEF", pid=356, team="LIV", price=6.5),
        rp("Gvardiol", "DEF", pid=391, team="MCI", price=5.7),
        rp("Senesi", "DEF", pid=498, team="TOT", price=5.8),
        rp("B.Fernandes", "MID", pid=426, team="MUN", price=12.0, captain=True),
        rp("Semenyo", "MID", pid=397, team="MCI", price=8.4),
        rp("Mbeumo", "MID", pid=427, team="MUN", price=7.9),
        rp("Eze", "MID", pid=14, team="ARS", price=6.3),
        rp("João Pedro", "FWD", pid=165, team="CHE", price=7.8, vice=True),
        rp("Watkins", "FWD", pid=55, team="AVL", price=7.8),
        rp("Sels", "GKP", pid=467, team="NFO", price=5.0, bench=1),
        rp("Gudmundsson", "DEF", pid=331, team="LEE", price=4.5, bench=2),
        rp("Rogers", "MID", pid=40, team="CHE", price=7.7, bench=3),
        rp("Wood", "FWD", pid=490, team="NFO", price=5.8, bench=4),
    ]


def test_fixture_bootstrap_is_consistent():
    bs = make_bootstrap()
    assert len(bs.elements) == len(PLAYERS)
    assert bs.next_event is not None and bs.next_event.id == 5
    assert bs.player(165).web_name == "João Pedro"
    assert bs.team(bs.player(426).team).short_name == "MUN"
