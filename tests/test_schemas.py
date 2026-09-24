from datetime import UTC, datetime

from fplcopilot.data.fpl_client import estimate_free_transfers
from fplcopilot.data.schemas import ChipPlay, EntryHistoryRow, ManagerHistory, Player, Position


def make_player(**overrides):
    base = {
        "id": 1,
        "web_name": "Saka",
        "team": 1,
        "element_type": 3,
        "now_cost": 104,
        "ep_next": "6.4",  # строка, как отдаёт API
        "expected_goals": "0.45",
        "selected_by_percent": "41.2",
        "unknown_future_field": {"x": 1},  # должно игнорироваться
    }
    base.update(overrides)
    return Player.model_validate(base)


def test_player_parses_string_numbers_and_ignores_unknown_fields():
    p = make_player()
    assert p.price == 10.4
    assert p.position is Position.MID
    assert p.ep_next == 6.4
    assert p.expected_goals == 0.45
    assert p.selected_by_percent == 41.2
    assert p.is_available


def test_player_tolerates_missing_optional_fields():
    p = make_player(chance_of_playing_next_round=None, news="", penalties_order=None)
    assert p.chance_of_playing_next_round is None
    assert p.penalties_order is None


def _history(rows: list[tuple[int, int]], chips: dict[int, str] | None = None) -> ManagerHistory:
    """rows: [(gw, transfers_made)], chips: {gw: chip_name}."""
    return ManagerHistory(
        current=[EntryHistoryRow(event=gw, event_transfers=t) for gw, t in rows],
        chips=[
            ChipPlay(name=name, time=datetime(2026, 8, 1, tzinfo=UTC), event=gw)
            for gw, name in (chips or {}).items()
        ],
    )


def test_free_transfers_start_with_one_after_gw1():
    assert estimate_free_transfers(_history([(1, 0)])) == 1


def test_free_transfers_accumulate_when_unused():
    # GW1 -> 1 FT; GW2 не использован -> 2; GW3 не использован -> 3
    assert estimate_free_transfers(_history([(1, 0), (2, 0), (3, 0)])) == 3


def test_free_transfers_capped_at_five():
    rows = [(gw, 0) for gw in range(1, 12)]
    assert estimate_free_transfers(_history(rows)) == 5


def test_free_transfers_consumed_by_transfers_and_hits_do_not_go_negative():
    # GW1 -> 1 FT; GW2: сделал 3 трансфера (1 FT + 2 хита) -> к GW3 снова 1
    assert estimate_free_transfers(_history([(1, 0), (2, 3)])) == 1


def test_wildcard_keeps_banked_transfers():
    # GW1 -> 1; GW2 не использован -> 2; GW3 Wildcard с 11 трансферами -> банк сохраняется: 3
    assert estimate_free_transfers(_history([(1, 0), (2, 0), (3, 11)], {3: "wildcard"})) == 3


def test_upto_gw_limits_history():
    hist = _history([(1, 0), (2, 0), (3, 0), (4, 0)])
    assert estimate_free_transfers(hist, upto_gw=2) == 2
