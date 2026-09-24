"""Metric functions of evals/metrics.py on toy data, plus the aggregators on synthetic rows."""

from types import SimpleNamespace

import pytest

from evals.metrics import (
    accuracy,
    confusion,
    dedupe_by_article,
    hit_at_k,
    macro_f1,
    mean,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    within_range,
)
from evals.run_rag import aggregate_retrieval, aggregate_signals, parse_modes, render_table


def chunk(cid: int, aid: int):
    return SimpleNamespace(chunk_id=cid, article_id=aid)


def test_dedupe_by_article_keeps_first_occurrence_order():
    chunks = [chunk(1, 10), chunk(2, 20), chunk(3, 10), chunk(4, 30), chunk(5, 20)]
    assert dedupe_by_article(chunks) == [10, 20, 30]
    assert dedupe_by_article([10, 10, 11]) == [10, 11]  # plain ints work too
    assert dedupe_by_article([]) == []


def test_precision_recall_hit_mrr_on_toy_ranking():
    ranked = [10, 20, 30, 40, 50]
    relevant = {20, 50, 60}
    assert precision_at_k(ranked, relevant, 5) == pytest.approx(2 / 5)
    assert precision_at_k(ranked, relevant, 2) == pytest.approx(1 / 2)
    assert recall_at_k(ranked, relevant, 5) == pytest.approx(2 / 3)
    assert recall_at_k(ranked, relevant, 1) == 0.0
    assert hit_at_k(ranked, relevant, 1) == 0.0 and hit_at_k(ranked, relevant, 2) == 1.0
    assert reciprocal_rank(ranked, relevant) == pytest.approx(1 / 2)
    assert reciprocal_rank([1, 2, 3], {9}) == 0.0
    assert reciprocal_rank([20], relevant) == 1.0


def test_precision_is_scored_at_article_level_after_dedupe():
    # three chunks of the same relevant article must count once, not three times
    chunks = [chunk(1, 10), chunk(2, 10), chunk(3, 10), chunk(4, 99)]
    arts = dedupe_by_article(chunks)
    assert precision_at_k(arts, {10}, 4) == pytest.approx(1 / 4)  # k stays 4 (slots), 1 article hit
    assert recall_at_k(arts, {10}, 4) == 1.0


def test_metrics_with_no_relevant_articles_are_not_applicable():
    assert precision_at_k([1, 2], set(), 2) == 0.0  # every hit is a false positive
    assert recall_at_k([1, 2], set(), 2) is None
    assert hit_at_k([1, 2], set(), 2) is None
    assert reciprocal_rank([1, 2], set()) is None
    with pytest.raises(ValueError):
        precision_at_k([1], {1}, 0)


def test_macro_f1_and_accuracy():
    y_true = ["fit", "fit", "injured", "unknown", "doubtful"]
    y_pred = ["fit", "injured", "injured", "unknown", "fit"]
    # fit: tp1 fp1 fn1 -> 0.5; injured: tp1 fp1 fn0 -> 2/3; unknown: 1; doubtful: tp0 fp0 fn1 -> 0
    assert macro_f1(y_true, y_pred) == pytest.approx((0.5 + 2 / 3 + 1.0 + 0.0) / 4)
    assert accuracy(y_true, y_pred) == pytest.approx(3 / 5)
    assert macro_f1(["a"], ["a"]) == 1.0
    assert macro_f1([], []) is None and accuracy([], []) is None
    assert confusion(y_true, y_pred) == {
        "fit": {"fit": 1, "injured": 1},
        "injured": {"injured": 1},
        "unknown": {"unknown": 1},
        "doubtful": {"fit": 1},
    }
    with pytest.raises(ValueError):
        macro_f1(["a"], ["a", "b"])


def test_mean_percentile_within_range():
    assert mean([1, None, 3]) == 2.0
    assert mean([]) is None and mean([None]) is None
    assert mean([True, False, None]) == 0.5  # booleans -> rates
    assert percentile([5, 1, 3], 50) == 3
    assert percentile([5, 1, 3], 95) == 5
    assert percentile([5, 1, 3], 0) == 1
    assert percentile([], 50) is None
    with pytest.raises(ValueError):
        percentile([1], 101)
    assert within_range(0.7, 0.6, 0.75) is True
    assert within_range(0.8, 0.6, 0.75) is False
    assert within_range(0.8, None, None) is None
    assert within_range(None, 0, 1) is None


def _ret_row(mode, has_rel, p, r, rr, top1, lat, age_hours=1):
    return {
        "mode": mode,
        "has_relevant": has_rel,
        "as_of": "2026-09-17T09:00:00+00:00",
        "precision_at_k": p,
        "recall_at_k": r,
        "hit_at_k": None if r is None else float(r > 0),
        "reciprocal_rank": rr,
        "top1_score": top1,
        "top1_published_at": f"2026-09-17T{9 - age_hours:02d}:00:00+00:00",
        "latency_ms": lat,
        "timings": {"total_ms": lat, "embed_ms": lat / 2},
    }


def test_aggregate_retrieval_excludes_no_relevant_queries_from_recall_and_splits_top1():
    rows = [
        _ret_row("dense", True, 0.5, 1.0, 1.0, 0.9, 100),
        _ret_row("dense", True, 0.25, 0.5, 0.5, 0.7, 300),
        _ret_row("dense", False, 0.0, None, None, 0.4, 200, age_hours=3),
    ]
    agg = aggregate_retrieval(rows, ["dense"])["dense"]
    assert agg["n_queries"] == 3 and agg["n_with_relevant"] == 2 and agg["n_no_relevant"] == 1
    assert agg["precision_at_k"] == pytest.approx(0.375)  # over queries with relevant only
    assert agg["recall_at_k"] == pytest.approx(0.75)
    assert agg["mrr"] == pytest.approx(0.75)
    assert agg["hit_at_k"] == 1.0
    assert agg["top1_score_relevant_queries"] == pytest.approx(0.8)
    assert agg["top1_score_no_relevant_queries"] == pytest.approx(0.4)
    assert agg["latency_p50_ms"] == 200 and agg["latency_p95_ms"] == 300
    assert agg["mean_top1_age_days"] == pytest.approx((1 + 1 + 3) / 3 / 24)
    assert agg["stage_ms_mean"]["embed_ms"] == pytest.approx(100)


def _sig_row(mode, exp, pred, *, alt=(), tags=(), ev=1, conf=0.8, faith=None, fixes=0, ret=None):
    accepted = {exp, *alt}
    return {
        "mode": mode,
        "error": None,
        "tags": list(tags),
        "expected_availability": exp,
        "availability_pred": pred,
        "availability_exact": pred == exp,
        "availability_lenient": pred in accepted,
        "start_probability_in_range": True,
        "expected_minutes_in_range": None,
        "evidence_present_ok": (ev > 0) if "no_coverage" not in tags else None,
        "false_evidence": (ev > 0) if "no_coverage" in tags else None,
        "return_gw_exact": ret,
        "return_gw_within_1": ret,
        "confidence": conf,
        "validation_fixes": fixes,
        "evidence_n": ev,
        "faithfulness": faith,
        "faithfulness_verdict": None if faith is None else ("supported" if faith == 1 else "other"),
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "judge_prompt_tokens": 500 if faith is not None else 0,
        "judge_completion_tokens": 20 if faith is not None else 0,
        "latency_ms": 2000.0,
        "retrieval_ms": 300.0,
        "llm_ms": 1500.0,
    }


def test_aggregate_signals_rates_and_costs():
    rows = [
        _sig_row("dense", "injured", "injured", tags=("injured",), faith=1.0, ret=True),
        _sig_row("dense", "fit", "doubtful", alt=("doubtful",), tags=("conflict",), faith=0.5),
        _sig_row("dense", "unknown", "unknown", tags=("no_coverage",), ev=0, conf=0.0),
        _sig_row("dense", "unknown", "fit", tags=("no_coverage",), ev=1, conf=0.6, fixes=2),
        {"mode": "dense", "error": "boom", "tags": ["x"]},
    ]
    agg = aggregate_signals(rows, ["dense"], judge_model="gpt-4o-mini")["dense"]
    assert agg["n_examples"] == 5 and agg["n_errors"] == 1
    assert agg["availability_accuracy"] == pytest.approx(2 / 4)
    assert agg["availability_accuracy_lenient"] == pytest.approx(3 / 4)
    assert agg["evidence_present_rate"] == 1.0 and agg["evidence_present_n_applicable"] == 2
    assert agg["false_evidence_rate"] == pytest.approx(0.5)
    assert agg["false_evidence_n_applicable"] == 2
    assert agg["return_gw_exact_rate"] == 1.0 and agg["return_gw_n_applicable"] == 1
    assert agg["mean_confidence_expected_unknown"] == pytest.approx(0.3)
    assert agg["mean_confidence_expected_known"] == pytest.approx(0.8)
    assert agg["mean_validation_fixes"] == pytest.approx(0.5)
    assert agg["faithfulness_mean"] == pytest.approx(0.75) and agg["faithfulness_n_judged"] == 2
    assert agg["prompt_tokens"] == 4000 and agg["completion_tokens"] == 400
    assert agg["cost_usd_signals"] == pytest.approx((4000 * 0.15 + 400 * 0.60) / 1e6)
    assert agg["cost_usd_judge"] == pytest.approx((1000 * 0.15 + 40 * 0.60) / 1e6)
    assert agg["by_tag"]["no_coverage"]["n"] == 2
    assert agg["by_tag"]["no_coverage"]["availability_accuracy"] == pytest.approx(0.5)
    assert agg["confusion"]["unknown"] == {"unknown": 1, "fit": 1}


def test_render_table_and_parse_modes():
    table = render_table({"recall_at_k": {"dense": 0.5, "hybrid": None}}, ["dense", "hybrid"])
    lines = table.splitlines()
    assert lines[0].split() == ["metric", "dense", "hybrid"]
    assert lines[2].split() == ["recall_at_k", "0.500", "-"]
    assert parse_modes("dense, hybrid_rerank") == ["dense", "hybrid_rerank"]
    with pytest.raises(Exception, match="unknown mode"):
        parse_modes("dense,sparse")
