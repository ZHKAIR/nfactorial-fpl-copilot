"""ParsedSquad -> Squad: тот же объект, что у FPLClient.squad(), для оптимизатора и агента."""

from __future__ import annotations

import pytest
from test_vision_fixtures import make_bootstrap, rp, valid_resolved_players

from fplcopilot.data.schemas import Position
from fplcopilot.vision.schemas import ParsedSquad
from fplcopilot.vision.to_squad import choose_default_lineup, to_squad
from fplcopilot.vision.validate import is_valid, validate_squad


def make_parsed(players, *, bank=0.5, free_transfers=1, gw=5) -> ParsedSquad:
    issues = validate_squad(players, bank=bank)
    return ParsedSquad(
        players=players,
        bank=bank,
        free_transfers=free_transfers,
        gw=gw,
        captain_id=next((p.player_id for p in players if p.is_captain), None),
        vice_id=next((p.player_id for p in players if p.is_vice_captain), None),
        starting_ids=[p.player_id for p in players if p.resolved and not p.is_bench],
        bench_order=[p.player_id for p in players if p.is_bench],
        issues=issues,
        is_valid=is_valid(issues),
    )


def test_to_squad_valid_matches_fpl_client_shape():
    bs = make_bootstrap()
    squad = to_squad(make_parsed(valid_resolved_players()), bs, manager_id=42)

    assert squad.manager_id == 42 and squad.gw == 5
    assert len(squad.players) == 15
    assert [p.pick.position for p in squad.players] == list(range(1, 16))
    assert len(squad.starting_xi) == 11 and len(squad.bench) == 4
    # старт в порядке FPL: GKP, DEF, MID, FWD
    xi_types = [p.player.position for p in squad.starting_xi]
    assert xi_types == sorted(xi_types)
    assert xi_types[0] is Position.GKP
    # скамейка: вратарь на 12-й позиции, дальше в порядке bench_order
    assert [p.player.web_name for p in squad.bench] == ["Sels", "Gudmundsson", "Rogers", "Wood"]
    assert squad.bench[0].pick.position == 12
    # капитан/вице и multiplier
    assert squad.captain is not None and squad.captain.player.web_name == "B.Fernandes"
    assert squad.captain.pick.multiplier == 2
    vice = [p for p in squad.players if p.pick.is_vice_captain]
    assert (
        len(vice) == 1 and vice[0].player.web_name == "João Pedro" and vice[0].pick.multiplier == 1
    )
    assert all(p.pick.multiplier == 0 for p in squad.bench)
    assert all(p.pick.element_type is p.player.position for p in squad.players)
    # деньги и трансферы
    assert squad.bank == 0.5
    assert squad.team_value == pytest.approx(105.2 + 0.5)
    assert squad.free_transfers == 1
    assert squad.active_chip is None and squad.chips_used == []
    # роли SquadPlayer как у FPLClient
    roles = {p.player.web_name: p.role for p in squad.players}
    assert roles["B.Fernandes"] == "C" and roles["João Pedro"] == "VC" and roles["Sels"] == "B1"


def test_to_squad_uses_shown_prices_for_team_value_when_present():
    bs = make_bootstrap()
    players = [
        p.model_copy(update={"price_as_shown": (p.price or 0) + 0.1})
        for p in valid_resolved_players()
    ]
    squad = to_squad(make_parsed(players, bank=0.0), bs)
    assert squad.team_value == pytest.approx(105.2 + 1.5)


def test_to_squad_rejects_blocking_issues_unless_allowed():
    bs = make_bootstrap()
    players = [p for p in valid_resolved_players() if p.name_as_shown != "Eze"]
    parsed = make_parsed(players)
    assert not parsed.is_valid
    with pytest.raises(ValueError, match="blocking issues"):
        to_squad(parsed, bs)
    squad = to_squad(parsed, bs, allow_invalid=True)
    assert len(squad.players) == 14


def test_default_lineup_when_bench_unknown():
    bs = make_bootstrap()
    players = [
        p.model_copy(update={"is_bench": False, "bench_order": None})
        for p in valid_resolved_players()
    ]
    parsed = make_parsed(players)
    assert parsed.is_valid and any(i.code == "no_bench" for i in parsed.issues)

    starters, bench = choose_default_lineup(players, bs)
    assert len(starters) == 11 and len(bench) == 4
    assert bench[0].position == "GKP" and bench[0].name_as_shown == "Sels"  # более дешёвый вратарь
    from collections import Counter

    xi = Counter(p.position for p in starters)
    assert xi["GKP"] == 1 and 3 <= xi["DEF"] <= 5 and 2 <= xi["MID"] <= 5 and 1 <= xi["FWD"] <= 3
    # на скамейку ушли самые дешёвые полевые: Gudmundsson 4.5, Gvardiol 5.7, Wood 5.8
    assert {p.name_as_shown for p in bench[1:]} == {"Gudmundsson", "Gvardiol", "Wood"}

    squad = to_squad(parsed, bs)
    assert len(squad.starting_xi) == 11 and squad.bench[0].player.web_name == "Sels"
    assert squad.captain is not None and squad.captain.player.web_name == "B.Fernandes"


def test_captain_on_bench_is_dropped_and_two_captains_take_first():
    bs = make_bootstrap()
    players = valid_resolved_players()
    players[10] = rp(
        "Watkins", "FWD", pid=55, team="AVL", price=7.8, captain=True
    )  # второй капитан
    squad = to_squad(make_parsed(players), bs)
    assert squad.captain is not None and squad.captain.player.web_name == "B.Fernandes"
    assert sum(p.pick.is_captain for p in squad.players) == 1

    players = valid_resolved_players()
    players[5] = rp("B.Fernandes", "MID", pid=426, team="MUN", price=12.0)
    players[11] = rp("Sels", "GKP", pid=467, team="NFO", price=5.0, bench=1, captain=True)
    squad = to_squad(make_parsed(players), bs)
    assert squad.captain is None
    assert all(p.pick.multiplier in (0, 1) for p in squad.players)


def test_gw_fallbacks():
    bs = make_bootstrap()
    parsed = make_parsed(valid_resolved_players(), gw=None)
    assert to_squad(parsed, bs).gw == 5  # bootstrap.next_event
    assert to_squad(parsed, bs, gw=7).gw == 7
