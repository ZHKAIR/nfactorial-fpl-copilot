"""Пресеты стратегии: только числа, монотонность между пресетами, дефолты сообщества."""

import dataclasses
import math

import pytest

from fplcopilot.core.strategy import (
    DEFAULT_BENCH_WEIGHTS,
    HIT_COST,
    MAX_FREE_TRANSFERS,
    PRESETS,
    Strategy,
    get_strategy,
)


def test_presets_exist_and_are_frozen():
    assert set(PRESETS) == {"conservative", "balanced", "aggressive"}
    s = get_strategy("balanced")
    assert isinstance(s, Strategy) and s.name == "balanced"
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.variance_penalty = 1.0  # type: ignore[misc]
    assert get_strategy(s) is s
    with pytest.raises(ValueError):
        get_strategy("yolo")


def test_presets_are_ordered_by_risk_appetite():
    c, b, a = (get_strategy(n) for n in ("conservative", "balanced", "aggressive"))
    assert c.variance_penalty > b.variance_penalty > a.variance_penalty
    assert c.hit_threshold > b.hit_threshold > a.hit_threshold
    assert c.ownership_weight > 0 == b.ownership_weight > a.ownership_weight
    assert c.wc_margin > b.wc_margin > a.wc_margin


def test_community_defaults_shared_by_all_presets():
    for s in PRESETS.values():
        assert s.ft_value == 1.5
        assert s.itb_value == 0.08
        assert s.decay_base == 0.84
        assert (
            dict(s.bench_weights)
            == DEFAULT_BENCH_WEIGHTS
            == {12: 0.03, 13: 0.21, 14: 0.06, 15: 0.002}
        )
    assert HIT_COST == 4 and MAX_FREE_TRANSFERS == 5


def test_decay_and_bench_weight_helpers():
    s = get_strategy("balanced")
    assert s.decay(5, 5) == 1.0
    assert math.isclose(s.decay(7, 5), 0.84**2)
    assert s.decay(4, 5) == 1.0  # прошлые туры не «усиливаются»
    assert s.bench_weight(13) == 0.21 and s.bench_weight(99) == 0.0
    tweaked = s.with_overrides(variance_penalty=0.5)
    assert tweaked.variance_penalty == 0.5 and tweaked.name == "balanced"
    assert s.variance_penalty != 0.5
