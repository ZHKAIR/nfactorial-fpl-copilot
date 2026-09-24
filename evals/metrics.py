"""Pure metric functions (no DB, no network) — unit-tested on toy data.

Retrieval metrics are computed at ARTICLE level: chunk ids change on every reindex, article
ids and URLs do not. `dedupe_by_article` collapses a ranked chunk list to the ranked list of
distinct article ids (first occurrence wins), which is then scored against the golden set.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

# A/B #4 (цитаты про форму в доказательствах доступности): словарь общий с сигналом v4, который
# им же переносит «форменные» цитаты из evidence, — поэтому доля по словарю дополняется ручной
# разметкой (docs/EVALS.md §4.6).
from fplcopilot.rag.quote_kind import quote_kind  # noqa: F401


def dedupe_by_article(chunks: Iterable[Any]) -> list[int]:
    """Ranked chunks -> ranked distinct article ids (order of first occurrence)."""
    seen: set[int] = set()
    out: list[int] = []
    for c in chunks:
        aid = c.article_id if hasattr(c, "article_id") else int(c)
        if aid not in seen:
            seen.add(aid)
            out.append(aid)
    return out


def precision_at_k(ranked: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """|top-k ∩ relevant| / k. Empty relevant set -> 0.0 (every hit is a false positive)."""
    if k <= 0:
        raise ValueError("k must be positive")
    rel = set(relevant)
    top = list(ranked[:k])
    return sum(1 for a in top if a in rel) / k


def recall_at_k(ranked: Sequence[int], relevant: Iterable[int], k: int) -> float | None:
    """|top-k ∩ relevant| / |relevant|; None when there is nothing to recall."""
    rel = set(relevant)
    if not rel:
        return None
    top = set(ranked[:k])
    return len(top & rel) / len(rel)


def hit_at_k(ranked: Sequence[int], relevant: Iterable[int], k: int) -> float | None:
    """1.0 if any relevant article is in the top-k; None when relevant is empty."""
    rel = set(relevant)
    if not rel:
        return None
    return 1.0 if any(a in rel for a in ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[int], relevant: Iterable[int]) -> float | None:
    """1/rank of the first relevant article (ranks start at 1); 0.0 if none; None if no relevant."""
    rel = set(relevant)
    if not rel:
        return None
    for i, a in enumerate(ranked, start=1):
        if a in rel:
            return 1.0 / i
    return 0.0


def mean(values: Iterable[float | None]) -> float | None:
    """Mean over non-None values; None if nothing to average."""
    xs = [float(v) for v in values if v is not None]
    return sum(xs) / len(xs) if xs else None


def percentile(values: Iterable[float], p: float) -> float | None:
    """Nearest-rank percentile (p in 0..100) on the sorted values; None for empty input."""
    xs = sorted(float(v) for v in values)
    if not xs:
        return None
    if not 0 <= p <= 100:
        raise ValueError("p must be within 0..100")
    rank = max(1, math.ceil(p / 100 * len(xs)))
    return xs[rank - 1]


def within_range(x: float | None, lo: float | None, hi: float | None) -> bool | None:
    """lo <= x <= hi; None when the range is unspecified (metric not applicable)."""
    if lo is None or hi is None or x is None:
        return None
    return lo <= x <= hi


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float | None:
    """Unweighted mean F1 over the classes present in y_true ∪ y_pred (sklearn 'macro')."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return None
    classes = sorted(set(y_true) | set(y_pred))
    f1s = []
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == c and p != c)
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return sum(f1s) / len(f1s)


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float | None:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return None
    return sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p) / len(y_true)


def confusion(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, dict[str, int]]:
    """{true_label: {pred_label: count}} — compact enough to store in the results JSON."""
    out: dict[str, dict[str, int]] = {}
    for t, p in zip(y_true, y_pred, strict=True):
        out.setdefault(t, {})
        out[t][p] = out[t].get(p, 0) + 1
    return out
