"""Пуассон/NegBin, снятие маржи, подбор xG по рынку, OddsProvider без fallback."""

import math

import pytest

from fplcopilot.config import settings
from fplcopilot.core import odds, stats
from fplcopilot.core.fixtures import get_provider
from fplcopilot.data.schemas import Fixture


def test_poisson_pmf_sums_to_one_and_sf_edges():
    lam = 2.3
    assert math.isclose(sum(stats.poisson_pmf(k, lam) for k in range(60)), 1.0, abs_tol=1e-9)
    assert stats.poisson_sf(0, lam) == 1.0
    assert math.isclose(stats.poisson_sf(1, lam), 1 - math.exp(-lam), abs_tol=1e-9)
    assert stats.poisson_sf(3, 0.0) == 0.0


def test_clean_sheet_probability_is_exp_minus_lambda():
    assert odds.clean_sheet_prob(0.0) == 1.0
    assert math.isclose(odds.clean_sheet_prob(1.0), math.exp(-1.0))
    assert odds.clean_sheet_prob(0.5) > odds.clean_sheet_prob(1.5)


def test_defcon_probability_poisson_values():
    # DEF: порог 10 CBIT; при среднем 8 — ~28%, при 12 — ~79%
    assert math.isclose(stats.poisson_sf(10, 8.0), 0.2834, abs_tol=1e-3)
    assert math.isclose(stats.poisson_sf(10, 12.0), 0.7576, abs_tol=1e-3)
    assert stats.poisson_sf(12, 6.0) < 0.03  # MID с 6 CBIRT почти никогда


def test_negbin_matches_poisson_at_dispersion_one_and_has_fatter_tail():
    for k, mean in ((10, 8.0), (12, 6.0), (3, 2.9)):
        assert math.isclose(stats.negbin_sf(k, mean, 1.0), stats.poisson_sf(k, mean), abs_tol=1e-9)
    # ниже порога сверхдисперсия повышает шанс, выше порога — понижает
    assert stats.negbin_sf(10, 7.0, 1.3) > stats.poisson_sf(10, 7.0)
    assert stats.negbin_sf(10, 13.0, 1.3) < stats.poisson_sf(10, 13.0)
    assert stats.negbin_sf(0, 5.0, 1.3) == 1.0
    assert stats.negbin_sf(3, 0.0, 1.3) == 0.0


def test_expected_floor_div_below_naive():
    e2 = stats.expected_floor_div(1.45, 2)
    assert math.isclose(e2, 0.4888, abs_tol=1e-3)
    assert e2 < 1.45 / 2
    assert stats.expected_floor_div(3.0, 3) < 1.0
    assert stats.expected_floor_div(0.0, 3) == 0.0


def test_shrink_and_ranks():
    assert stats.shrink(1.0, 0.0, n=0, k=5) == 0.0
    assert math.isclose(stats.shrink(1.0, 0.0, n=5, k=5), 0.5)
    assert math.isclose(stats.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
    assert math.isclose(stats.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
    assert math.isnan(stats.spearman([1, 1, 1], [1, 2, 3]))
    assert stats.percentile_rank([1, 2, 3, 4, 5], 3) == 0.5
    assert stats.percentile_rank([1, 2, 3, 4, 5], 9) == 1.0
    assert stats.percentile_rank([], 3) == 0.5


def test_remove_margin_sums_to_one_and_preserves_order():
    decimal = [1.9, 3.6, 4.2]  # сумма implied ≈ 1.04
    probs = odds.remove_margin(decimal)
    assert math.isclose(sum(probs), 1.0, abs_tol=1e-12)
    assert probs[0] > probs[1] > probs[2]
    assert all(0 < p < 1 for p in probs)
    assert odds.remove_margin([]) == []
    with pytest.raises(ValueError):
        odds.remove_margin([1.0, 2.0])


@pytest.mark.parametrize("lam_home,lam_away", [(1.8, 1.1), (1.2, 1.2), (2.6, 0.7), (0.9, 2.1)])
def test_poisson_grid_recovers_known_lambdas(lam_home, lam_away):
    ph, pd, pa = odds.poisson_match_probs(lam_home, lam_away)
    assert math.isclose(ph + pd + pa, 1.0, abs_tol=1e-4)
    p_over = odds.prob_total_over(lam_home, lam_away, 2.5)
    est_h, est_a = odds.xg_from_1x2_and_total(ph, pd, pa, 2.5, p_over)
    assert math.isclose(est_h, lam_home, abs_tol=0.03)
    assert math.isclose(est_a, lam_away, abs_tol=0.03)
    est_h2, est_a2 = odds.xg_from_1x2_and_total(ph, pd, pa, None, None)  # только 1X2
    assert math.isclose(est_h2, lam_home, abs_tol=0.05)
    assert math.isclose(est_a2, lam_away, abs_tol=0.05)


def test_xg_fit_rejects_unnormalised_probabilities():
    with pytest.raises(ValueError):
        odds.xg_from_1x2_and_total(0.6, 0.3, 0.3)


def test_odds_provider_without_teams_has_no_fallback(tmp_path, monkeypatch):
    """Без списка команд fallback не собрать: непокрытая фикстура -> KeyError (как раньше пустой
    snapshot). Полный гибрид с fallback — tests/test_core_odds_provider.py."""
    monkeypatch.setattr(settings, "odds_api_key", None)
    monkeypatch.setattr(settings, "cache_dir", tmp_path)  # без кэша и без ключа -> пустой снимок
    provider = get_provider("odds")
    assert provider.name == "odds"
    fx = Fixture.model_validate({"id": 1, "team_h": 1, "team_a": 2, "event": 5})
    with pytest.raises(KeyError, match="нет fallback"):
        provider.expected_goals(fx)
    with pytest.raises(ValueError):
        get_provider("bookies")
