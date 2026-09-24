"""Срок действия новостного сигнала по содержанию (core/signals.signal_active + истечение по туру
в core/minutes.estimate_minutes). Без БД."""

from datetime import UTC, datetime, timedelta

from fplcopilot.core.minutes import estimate_minutes
from fplcopilot.core.signals import SignalLite, signal_active
from fplcopilot.data import Position

AS_OF = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)


def sig(availability: str, *, days_ago: float, return_gw: int | None = None, conf=0.9):
    return SignalLite(
        player_id=398,
        as_of=AS_OF - timedelta(days=days_ago),
        availability=availability,
        start_probability=0.0 if availability != "doubtful" else 0.5,
        expected_minutes=0,
        confidence=conf,
        return_gw=return_gw,
    )


def test_old_suspension_with_future_return_gw_stays_active():
    old = sig("suspended", days_ago=12, return_gw=7)
    assert signal_active(old, AS_OF)  # 12 дней > недели, но горизонт явный
    # и продолжает гасить минуты до GW7 (ближайший тур GW6)
    mins = estimate_minutes([], position=Position.MID, status="a", signal=old, gw=6, status_gw=6)
    assert mins.p_start < 0.2 and mins.signal_weight > 0


def test_same_suspension_after_return_gw_no_longer_applies():
    old = sig("suspended", days_ago=12, return_gw=7)
    mins = estimate_minutes([], position=Position.MID, status="a", signal=old, gw=7, status_gw=7)
    assert mins.signal_weight == 0 and any("expired" in n for n in mins.notes)
    base = estimate_minutes([], position=Position.MID, status="a", gw=7, status_gw=7)
    assert mins.p_start == base.p_start


def test_old_soft_signal_without_horizon_expires_after_a_week():
    assert not signal_active(sig("doubtful", days_ago=8), AS_OF)
    assert signal_active(sig("doubtful", days_ago=3), AS_OF)
    assert not signal_active(sig("injured", days_ago=8), AS_OF)  # без return_gw — мягкий срок
    assert not signal_active(sig("suspended", days_ago=90, return_gw=12), AS_OF)  # верхняя граница


def test_fpl_status_a_after_the_signal_wins():
    old = sig("suspended", days_ago=12, return_gw=7)
    later = AS_OF - timedelta(days=2)
    assert not signal_active(old, AS_OF, fpl_status="a", fpl_changed_at=later)
    # статус «a» старше сигнала не отменяет его (новость новее FPL)
    earlier = AS_OF - timedelta(days=20)
    assert signal_active(old, AS_OF, fpl_status="a", fpl_changed_at=earlier)
    # fit-сигнал статусом «a» не отменяется
    assert signal_active(sig("fit", days_ago=1), AS_OF, fpl_status="a", fpl_changed_at=later)
