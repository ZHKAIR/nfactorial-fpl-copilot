"""Модель минут: статусы FPL, новостной сигнал, приоры."""

import math
from datetime import UTC, datetime

from fplcopilot.core.history import SeasonTotals
from fplcopilot.core.minutes import availability_factor, estimate_minutes, start_prior_from_past
from fplcopilot.core.signals import SignalLite
from fplcopilot.data import Position
from fplcopilot.data.schemas import PlayerGWHistory

AS_OF = datetime(2026, 9, 17, 8, 0, tzinfo=UTC)


def _rows(spec: list[tuple[int, int]]) -> list[PlayerGWHistory]:
    """spec: [(starts, minutes), ...] по времени."""
    return [
        PlayerGWHistory.model_validate(
            {
                "element": 1,
                "fixture": i + 1,
                "round": i + 1,
                "was_home": True,
                "starts": s,
                "minutes": m,
            }
        )
        for i, (s, m) in enumerate(spec)
    ]


def _signal(
    availability: str,
    p: float,
    conf: float,
    *,
    return_gw: int | None = None,
    rotation_risk: str = "unknown",
) -> SignalLite:
    return SignalLite(
        1,
        AS_OF,
        availability,
        p,
        int(p * 80),
        conf,
        rotation_risk=rotation_risk,
        return_gw=return_gw,
    )


REGULAR = _rows([(1, 90), (1, 90), (1, 88), (1, 90)])
BENCH = _rows([(0, 0), (0, 12), (0, 0), (0, 20)])


def test_availability_factor_by_status():
    assert availability_factor("a", None) == 1.0
    assert availability_factor("a", 75) == 0.75
    assert availability_factor("d", 75) == 0.75
    assert availability_factor("d", None) == 0.5
    for status in "isun":
        assert availability_factor(status, 100) == 0.0


def test_regular_starter_gets_high_p_start_and_full_minutes():
    est = estimate_minutes(REGULAR, position=Position.DEF)
    assert 0.85 <= est.p_start <= 0.95  # сжатие к приору 0.4 с K=1 не даёт ровно 1
    assert est.exp_minutes > 78
    assert est.p60 <= est.p_appear <= 1.0
    assert est.p_appear >= est.p_start


def test_bench_player_low_p_start_but_some_sub_minutes():
    est = estimate_minutes(BENCH, position=Position.MID)
    assert est.p_start < 0.15
    assert 0 < est.exp_minutes < 20
    assert est.p60 < 0.1


def test_doubtful_status_scales_and_injured_zeroes():
    fit = estimate_minutes(REGULAR, position=Position.MID)
    doubtful = estimate_minutes(REGULAR, position=Position.MID, status="d", chance=75)
    injured = estimate_minutes(REGULAR, position=Position.MID, status="i", chance=0)
    assert math.isclose(doubtful.p_start, fit.p_start * 0.75, abs_tol=1e-3)
    assert doubtful.exp_minutes < fit.exp_minutes
    assert injured.p_start == 0.0 and injured.exp_minutes == 0.0 and injured.p_appear == 0.0
    assert injured.availability_factor == 0.0


def test_signal_blend_uses_confidence_as_weight():
    fit = estimate_minutes(REGULAR, position=Position.FWD)
    sig = _signal("doubtful", 0.4, 0.5)
    blended = estimate_minutes(REGULAR, position=Position.FWD, signal=sig)
    assert math.isclose(blended.p_start, 0.5 * 0.4 + 0.5 * fit.p_start, abs_tol=1e-3)
    assert blended.signal_weight == 0.5
    hard = estimate_minutes(REGULAR, position=Position.FWD, signal=_signal("injured", 0.0, 1.0))
    assert hard.p_start == 0.0 and hard.p_appear == 0.0 and hard.exp_minutes == 0.0
    unknown = estimate_minutes(REGULAR, position=Position.FWD, signal=_signal("unknown", 0.5, 0.0))
    assert unknown.p_start == fit.p_start  # unknown-сигнал игнорируется


NAILED_PAST = SeasonTotals(season_name="2025/26", minutes=3200, starts=36)


def test_fit_signal_never_lowers_nailed_starter():
    """Haaland: 4/4 стартов + 36/38 в прошлом сезоне -> p_start 0.99; «fit 0.9» не режет его."""
    prior = estimate_minutes(REGULAR, position=Position.FWD, past=NAILED_PAST)
    assert prior.p_start > 0.95
    fit = estimate_minutes(
        REGULAR, position=Position.FWD, past=NAILED_PAST, signal=_signal("fit", 0.9, 0.8)
    )
    assert fit.p_start == prior.p_start
    assert fit.signal_weight == 0.0
    assert any("confirms prior" in n for n in fit.notes)
    # fit с явным предупреждением о ротации (rotation_risk=high) — смешиваем как раньше
    rotation = estimate_minutes(
        REGULAR,
        position=Position.FWD,
        past=NAILED_PAST,
        signal=_signal("fit", 0.6, 0.8, rotation_risk="high"),
    )
    assert math.isclose(rotation.p_start, 0.8 * 0.6 + 0.2 * prior.p_start, abs_tol=1e-3)
    # fit 0.6 без предупреждения о ротации — не режем
    soft = estimate_minutes(
        REGULAR, position=Position.FWD, past=NAILED_PAST, signal=_signal("fit", 0.6, 0.8)
    )
    assert soft.p_start == prior.p_start
    # doubtful/injured по-прежнему снижают
    doubt = estimate_minutes(
        REGULAR, position=Position.FWD, past=NAILED_PAST, signal=_signal("doubtful", 0.4, 0.7)
    )
    assert doubt.p_start < prior.p_start
    # fit поднимает игрока, которого FPL пометил d (новости разрешают сомнение вверх)
    doubtful_fpl = estimate_minutes(
        REGULAR, position=Position.FWD, past=NAILED_PAST, status="d", chance=50
    )
    lifted = estimate_minutes(
        REGULAR,
        position=Position.FWD,
        past=NAILED_PAST,
        status="d",
        chance=50,
        signal=_signal("fit", 0.9, 0.8),
    )
    assert lifted.p_start > doubtful_fpl.p_start


def test_return_gw_drives_horizon_availability():
    """Травма с return_gw=7 (статус i относится к GW5): 0 до возвращения, ×0.8 в GW7, приор в GW8."""
    sig = _signal("injured", 0.0, 1.0, return_gw=7)
    prior = estimate_minutes(REGULAR, position=Position.MID, past=NAILED_PAST)

    def est(gw: int, status: str = "i", chance: int | None = 0):
        return estimate_minutes(
            REGULAR,
            position=Position.MID,
            past=NAILED_PAST,
            status=status,
            chance=chance,
            signal=sig,
            gw=gw,
            status_gw=5,
        )

    assert est(5).p_start == 0.0 and est(5).exp_minutes == 0.0
    assert est(6).p_start == 0.0 and est(6).availability_factor == 0.0
    back = est(7)
    assert back.availability_factor == 0.8
    assert math.isclose(back.p_start, 0.8 * prior.p_start, abs_tol=1e-3)
    assert back.signal_weight == 0.0  # сигнал «injured» после возвращения не смешивается
    assert est(8).p_start == prior.p_start and est(8).availability_factor == 1.0
    # FPL ещё не обновил статус (a), но новости знают о травме -> 0 до return_gw
    assert est(6, status="a", chance=None).p_start == 0.0
    # FPL d 75% до возвращения — оставляем chance-based множитель
    assert est(6, status="d", chance=75).availability_factor == 0.75
    # без gw/status_gw (ближайший тур) — обычная логика: статус i обнуляет
    plain = estimate_minutes(REGULAR, position=Position.MID, status="i", chance=0, signal=sig)
    assert plain.p_start == 0.0


def test_previous_season_prior_drives_players_without_matches():
    past = SeasonTotals(season_name="2025/26", minutes=3000, starts=34)
    assert math.isclose(start_prior_from_past(past), 34 / 38)
    assert start_prior_from_past(None) is None
    old = SeasonTotals(season_name="2021/22", minutes=1710, starts=0)
    assert math.isclose(start_prior_from_past(old), 0.5)
    est = estimate_minutes([], position=Position.MID, past=past)
    assert math.isclose(est.p_start, 34 / 38, abs_tol=1e-4)  # округление до 4 знаков
    est_none = estimate_minutes([], position=Position.MID)
    assert est_none.p_start == 0.4
    # четыре матча на скамейке перевешивают прошлый сезон, но не до нуля
    est_bench = estimate_minutes(BENCH, position=Position.MID, past=past)
    assert 0.1 < est_bench.p_start < 0.25
