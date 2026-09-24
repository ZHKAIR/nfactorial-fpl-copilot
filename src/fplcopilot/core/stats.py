"""Маленькая статистика без scipy: Пуассон, сжатие к приору, ранговая корреляция, ошибки."""

from __future__ import annotations

import math
from collections.abc import Sequence

POISSON_MAX_K = 40  # хвост Пуассона дальше пренебрежимо мал для λ <= ~12


def poisson_pmf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def poisson_cdf(k: int, lam: float) -> float:
    """P(X <= k)."""
    if k < 0:
        return 0.0
    return min(1.0, sum(poisson_pmf(i, lam) for i in range(k + 1)))


def poisson_sf(k: int, lam: float) -> float:
    """P(X >= k)."""
    if k <= 0:
        return 1.0
    return max(0.0, 1.0 - poisson_cdf(k - 1, lam))


def expected_floor_div(lam: float, d: int, max_k: int = POISSON_MAX_K) -> float:
    """E[floor(X / d)] для X ~ Poisson(lam): очки за сейвы (d=3) и штраф за пропущенные (d=2).

    Заметно меньше наивного lam/d: при lam=1.45, d=2 получаем 0.49, а не 0.73.
    """
    if lam <= 0:
        return 0.0
    return sum((k // d) * poisson_pmf(k, lam) for k in range(d, max_k + 1))


def expected_floor_half(lam: float, max_k: int = POISSON_MAX_K) -> float:
    return expected_floor_div(lam, 2, max_k)


def negbin_sf(k: int, mean: float, dispersion: float = 1.0, max_k: int = 200) -> float:
    """P(X >= k) для отрицательного биномиального с Var = dispersion · mean (dispersion >= 1).

    dispersion == 1 — обычный Пуассон. Используется для счётчиков оборонительных действий
    (CBIT/CBIRT), которые сверхдисперсны относительно Пуассона (зависят от хода матча).
    """
    if k <= 0:
        return 1.0
    if mean <= 0:
        return 0.0
    if dispersion <= 1.0 + 1e-9:
        return poisson_sf(k, mean)
    r = mean / (dispersion - 1.0)  # size; p = r / (r + mean)
    log_p = math.log(r / (r + mean))
    log_q = math.log(mean / (r + mean))
    cdf = 0.0
    for i in range(min(k, max_k)):
        cdf += math.exp(
            math.lgamma(i + r) - math.lgamma(r) - math.lgamma(i + 1) + r * log_p + i * log_q
        )
    return max(0.0, 1.0 - cdf)


def shrink(value: float, prior: float, n: float, k: float) -> float:
    """Сжатие к приору с весом наблюдений n/(n+k)."""
    if n <= 0:
        return prior
    return (n * value + k * prior) / (n + k)


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _ranks(values: Sequence[float]) -> list[float]:
    """Средние ранги при ничьих."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for idx in order[i : j + 1]:
            ranks[idx] = avg
        i = j + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return float("nan")
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    return sxy / math.sqrt(sxx * syy)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(_ranks(x), _ranks(y))


def mae(pred: Sequence[float], actual: Sequence[float]) -> float:
    return sum(abs(p - a) for p, a in zip(pred, actual, strict=True)) / max(1, len(pred))


def rmse(pred: Sequence[float], actual: Sequence[float]) -> float:
    return math.sqrt(
        sum((p - a) ** 2 for p, a in zip(pred, actual, strict=True)) / max(1, len(pred))
    )


def percentile_rank(sorted_values: Sequence[float], value: float) -> float:
    """Доля значений строго меньше value (+ половина равных) в отсортированном списке."""
    if not sorted_values:
        return 0.5
    lo = 0
    hi = len(sorted_values)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_values[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    left = lo
    hi = len(sorted_values)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_values[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    right = lo
    return (left + 0.5 * (right - left)) / len(sorted_values)
