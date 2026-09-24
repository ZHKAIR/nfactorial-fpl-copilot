"""Живые проверки против API FPL. Запуск: uv run pytest -m network."""

import pytest

from fplcopilot.data import FPLClient, Position

pytestmark = pytest.mark.network

OVERALL_LEAGUE_ID = 314


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    return FPLClient(cache_dir=tmp_path_factory.mktemp("cache"), cache_ttl=600)


def test_bootstrap_has_full_squad_universe(client):
    bs = client.bootstrap()
    assert len(bs.teams) == 20
    assert len(bs.elements) > 500
    assert {p.position for p in bs.elements} == set(Position)
    assert bs.next_event is not None or bs.current_event is not None


def test_chips_come_in_two_sets(client):
    chips = client.bootstrap().chips
    names = sorted(c.name for c in chips)
    assert names == sorted(
        ["wildcard", "wildcard", "freehit", "freehit", "bboost", "bboost", "3xc", "3xc"]
    )


def test_squad_of_public_manager_is_valid(client):
    leader = client.league_standings(OVERALL_LEAGUE_ID).results[0].entry
    squad = client.squad(leader)
    assert len(squad.players) == 15
    assert len(squad.starting_xi) == 11
    assert squad.captain is not None
    assert all(n <= 3 for n in squad.count_by_team().values())
    assert 1 <= (squad.free_transfers or 0) <= 5


def test_element_summary_has_defcon_fields(client):
    bs = client.bootstrap()
    # берём игрока с максимумом минут — у него точно есть история
    p = max(bs.elements, key=lambda x: x.minutes)
    s = client.element_summary(p.id)
    assert s.history, "история по турам пуста"
    row = s.history[-1]
    assert row.minutes >= 0
    assert row.expected_goals is not None
    assert row.defensive_contribution >= 0


def test_cache_is_used_on_second_call(client, tmp_path):
    c = FPLClient(cache_dir=tmp_path, cache_ttl=600)
    c.fixtures(event=1)
    files = list((tmp_path / "fpl").glob("*.json"))
    assert len(files) == 1
    c.fixtures(event=1)  # второй вызов — из кэша, файлов не прибавилось
    assert len(list((tmp_path / "fpl").glob("*.json"))) == 1
