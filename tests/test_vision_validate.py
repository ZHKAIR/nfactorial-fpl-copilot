"""Правила FPL для состава со скриншота: какие нарушения blocking, какие — предупреждения."""

from __future__ import annotations

from test_vision_fixtures import rp, valid_resolved_players

from fplcopilot.vision.validate import is_valid, validate_squad


def codes(issues, *, blocking: bool | None = None) -> list[str]:
    return [i.code for i in issues if blocking is None or i.blocking is blocking]


def test_valid_squad_has_no_issues():
    issues = validate_squad(valid_resolved_players(), bank=0.5)
    assert issues == []
    assert is_valid(issues)


# ---------- blocking ----------


def test_missing_player_blocks_size_and_position_count():
    players = [p for p in valid_resolved_players() if p.name_as_shown != "Eze"]  # 14, MID 4
    issues = validate_squad(players)
    assert "squad_size" in codes(issues, blocking=True)
    assert any(i.code == "position_count" and i.message == "MID: 4 ≠ 5" for i in issues)
    assert not is_valid(issues)


def test_duplicate_player_blocks():
    players = valid_resolved_players()
    players[10] = rp("Haaland", "FWD", pid=426, team="MUN", price=15.5)  # id капитана ещё раз
    issues = validate_squad(players)
    assert "duplicate" in codes(issues, blocking=True)
    assert not is_valid(issues)


def test_unresolved_and_ambiguous_block_with_readable_messages():
    players = valid_resolved_players()
    players[1] = rp(
        "Gabriel",
        "DEF",
        method="ambiguous",
        candidates=["Gabriel (ARS DEF £8.0m)", "Gudmundsson (LEE DEF £4.5m)"],
    )
    players[2] = rp("Zzyzx", "DEF", method="unresolved")
    issues = validate_squad(players)
    blocking = {i.code: i.message for i in issues if i.blocking}
    assert (
        blocking["ambiguous"]
        == "ambiguous: Gabriel — candidates [Gabriel (ARS DEF £8.0m), Gudmundsson (LEE DEF £4.5m)]"
    )
    assert blocking["unresolved"] == "unresolved: 'Zzyzx' not found in FPL players"
    # позиция берётся из ряда на экране, поэтому 2/5/5/3 всё ещё сходится — лишних нарушений нет
    assert "position_count" not in codes(issues)
    assert "squad_size" not in codes(issues)


def test_four_players_from_one_club_block():
    players = valid_resolved_players()
    players[2] = rp(
        "J.Timber", "DEF", pid=5, team="ARS", price=6.5
    )  # ARS: Raya, Gabriel, Eze, J.Timber
    issues = validate_squad(players)
    assert any(
        i.code == "club_limit" and i.message == "ARS: 4 players > 3" and i.blocking for i in issues
    )


def test_wrong_position_split_blocks():
    players = valid_resolved_players()
    players[10] = rp(
        "Saka", "MID", pid=12, team="ARS", price=9.5
    )  # вместо Watkins: MID 6, FWD 2, ARS 4
    issues = validate_squad(players)
    msgs = [i.message for i in issues if i.code == "position_count"]
    assert "MID: 6 ≠ 5" in msgs and "FWD: 2 ≠ 3" in msgs


# ---------- warnings (не блокируют) ----------


def test_two_captains_is_a_warning_only():
    players = valid_resolved_players()
    players[10] = rp("Watkins", "FWD", pid=55, team="AVL", price=7.8, captain=True)
    issues = validate_squad(players)
    assert is_valid(issues)
    assert any(
        i.code == "captain" and i.message == "2 captains: B.Fernandes, Watkins" for i in issues
    )


def test_missing_captain_and_vice_warn():
    players = valid_resolved_players()
    players[5] = rp("B.Fernandes", "MID", pid=426, team="MUN", price=12.0)
    players[9] = rp("João Pedro", "FWD", pid=165, team="CHE", price=7.8)
    issues = validate_squad(players)
    assert is_valid(issues)
    assert {"captain", "vice"} <= set(codes(issues, blocking=False))


def test_captain_on_bench_and_same_as_vice_warn():
    players = valid_resolved_players()
    players[5] = rp("B.Fernandes", "MID", pid=426, team="MUN", price=12.0, vice=True)  # C -> V
    players[9] = rp("João Pedro", "FWD", pid=165, team="CHE", price=7.8)
    players[11] = rp("Sels", "GKP", pid=467, team="NFO", price=5.0, bench=1, captain=True)
    issues = validate_squad(players)
    assert is_valid(issues)
    assert any("captain Sels is on the bench" in i.message for i in issues)

    players = valid_resolved_players()
    players[5] = rp("B.Fernandes", "MID", pid=426, team="MUN", price=12.0, captain=True, vice=True)
    players[9] = rp("João Pedro", "FWD", pid=165, team="CHE", price=7.8)
    issues = validate_squad(players)
    assert any("same player" in i.message for i in issues)


def test_invalid_formation_and_xi_count_warn():
    players = valid_resolved_players()
    # Watkins на скамейку, Gudmundsson в старт: 5 DEF, 4 MID, 1 FWD — валидно; ещё Wood в старт -> 12
    players[10] = rp("Watkins", "FWD", pid=55, team="AVL", price=7.8, bench=2)
    players[12] = rp("Gudmundsson", "DEF", pid=331, team="LEE", price=4.5)
    players[14] = rp("Wood", "FWD", pid=490, team="NFO", price=5.8)
    issues = validate_squad(players)
    assert is_valid(issues)
    assert any(i.code == "xi_count" and "12" in i.message for i in issues)

    players = valid_resolved_players()
    players[9] = rp("João Pedro", "FWD", pid=165, team="CHE", price=7.8, vice=True, bench=5)
    players[10] = rp("Watkins", "FWD", pid=55, team="AVL", price=7.8, bench=6)
    players[12] = rp("Gudmundsson", "DEF", pid=331, team="LEE", price=4.5)
    players[13] = rp("Rogers", "MID", pid=40, team="CHE", price=7.7)
    issues = validate_squad(players)  # 1-5-5-0
    assert is_valid(issues)
    assert any(i.code == "formation" and "FWD 0 not in 1..3" in i.message for i in issues)


def test_bench_goalkeeper_must_be_first():
    players = valid_resolved_players()
    players[11] = rp("Sels", "GKP", pid=467, team="NFO", price=5.0, bench=2)
    players[12] = rp("Gudmundsson", "DEF", pid=331, team="LEE", price=4.5, bench=1)
    issues = validate_squad(players)
    assert is_valid(issues)
    assert any(i.code == "bench_gk" and "Gudmundsson" in i.message for i in issues)


def test_no_bench_detected_is_a_warning():
    players = [
        p.model_copy(update={"is_bench": False, "bench_order": None})
        for p in valid_resolved_players()
    ]
    issues = validate_squad(players)
    assert is_valid(issues)
    assert codes(issues) == ["no_bench"]


def test_hint_mismatches_warn_but_do_not_block():
    players = valid_resolved_players()
    players[3] = rp(
        "Gvardiol",
        "DEF",
        pid=391,
        team="MCI",
        price=5.7,
        shown_price=6.5,
        shown_position="MID",
        club_hint="LIV",
    )
    issues = validate_squad(players)
    assert is_valid(issues)
    got = set(codes(issues, blocking=False))
    assert {"position_mismatch", "price_mismatch", "club_mismatch"} <= got
    assert any("shown £6.5m, FPL now £5.7m" in i.message for i in issues)


def test_stale_prices_produce_team_value_note():
    players = [
        p.model_copy(update={"price_as_shown": (p.price or 0) - 0.2})
        for p in valid_resolved_players()
    ]
    issues = validate_squad(players, bank=0.5)  # 15 × 0.2 = 3.0 > 1.0, но каждая в пределах ±0.3
    assert is_valid(issues)
    assert codes(issues) == ["team_value"]
    assert "bank £0.5m" in issues[0].message
