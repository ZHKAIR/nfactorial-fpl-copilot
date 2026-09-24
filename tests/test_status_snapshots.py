"""Логика «снимок только при изменении относительно последнего» (без БД)."""

from datetime import UTC, datetime

from fplcopilot.data.schemas import Player
from fplcopilot.rag.ingest import snapshot_needed, state_of

T1 = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)
T2 = datetime(2026, 9, 15, 19, 30, tzinfo=UTC)


def _player(status="a", chance=None, news="", news_added=None) -> Player:
    return Player.model_validate(
        {
            "id": 1,
            "web_name": "Saka",
            "team": 1,
            "element_type": 3,
            "now_cost": 100,
            "status": status,
            "chance_of_playing_next_round": chance,
            "news": news,
            "news_added": news_added,
        }
    )


def test_healthy_player_without_history_is_not_recorded():
    assert snapshot_needed(_player(), last=None) is False


def test_first_injury_is_recorded_and_repeat_is_not():
    injured = _player("i", 0, "Knee injury - Unknown return date", T1)
    assert snapshot_needed(injured, last=None) is True
    assert snapshot_needed(injured, last=state_of(injured)) is False


def test_unavailable_without_news_is_recorded():
    assert snapshot_needed(_player("u"), last=None) is True


def test_chance_change_with_same_news_is_recorded():
    before = _player("d", 50, "Muscular injury - 50% chance of playing", T1)
    after = before.model_copy(update={"chance_of_playing_next_round": 75})
    assert snapshot_needed(after, last=state_of(before)) is True


def test_recovery_is_recorded_once():
    injured = _player("i", 0, "Knee injury", T1)
    recovered = _player()  # status=a, news='', news_added=None
    assert snapshot_needed(recovered, last=state_of(injured)) is True
    assert snapshot_needed(recovered, last=state_of(recovered)) is False


def test_reinjury_after_recovery_is_recorded_even_with_same_text():
    recovered = _player()
    reinjured = _player("i", 0, "Knee injury", T2)
    assert snapshot_needed(reinjured, last=state_of(recovered)) is True
    # и снова выздоровел — состояние повторяет первое выздоровление, но это новое изменение
    assert snapshot_needed(recovered, last=state_of(reinjured)) is True
