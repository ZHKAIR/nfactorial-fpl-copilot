"""Metric helpers of evals/run_hparams.py and evals/run_models.py on toy data (no network, no DB)."""

import pytest

from evals.run_hparams import (
    EXPLAIN_PRODUCTION,
    BudgetExceeded,
    Spend,
    aggregate_cost,
    aggregate_explain,
    agreement,
    cell_label,
    cost_usd,
    default_extraction_arms,
    determinism_report,
    explain_cells,
    format_checks,
    price_for,
    projected_cost,
    salvage_truncated_answer,
    truncation_rate,
    verified_numbers,
)
from evals.run_models import aggregate_router, cost_projections

PRICES = {"mini": (0.15, 0.60), "big": (2.0, 8.0)}


# ---------- cost ----------


def test_price_lookup_resolves_dated_snapshots_to_the_longest_family_prefix():
    assert price_for("gpt-4o-mini") == (0.15, 0.60)
    assert price_for("gpt-4o-mini-2024-07-18") == (0.15, 0.60)  # not gpt-4o's (2.5, 10)
    assert price_for("gpt-4.1-nano-2025-04-14") == (0.10, 0.40)
    assert price_for("gpt-5.4-mini") == (0.75, 4.50)
    assert price_for("llama-local") is None


def test_cost_usd_is_list_price_per_million_and_none_for_unknown_models():
    assert cost_usd("mini", 1_000_000, 0, PRICES) == pytest.approx(0.15)
    assert cost_usd("mini", 2000, 100, PRICES) == pytest.approx((2000 * 0.15 + 100 * 0.60) / 1e6)
    assert cost_usd("nope", 2000, 100, PRICES) is None


def test_aggregate_cost_sums_per_model_and_refuses_partial_totals():
    rows = [
        {"model": "mini", "prompt_tokens": 1000, "completion_tokens": 100},
        {"model": "mini", "prompt_tokens": 3000, "completion_tokens": 300},
        {"model": "big", "prompt_tokens": 1000, "completion_tokens": 100},
        {"model": "mini", "prompt_tokens": 0, "completion_tokens": 0},  # abstention: no call
        {"error": "boom"},  # no model at all
    ]
    agg = aggregate_cost(rows, prices=PRICES)
    assert agg["calls"] == 3
    assert agg["prompt_tokens"] == 5000 and agg["completion_tokens"] == 500
    assert agg["by_model"]["mini"]["calls"] == 2
    assert agg["by_model"]["mini"]["cost_usd"] == pytest.approx((4000 * 0.15 + 400 * 0.60) / 1e6)
    assert agg["by_model"]["big"]["cost_usd"] == pytest.approx((1000 * 2.0 + 100 * 8.0) / 1e6)
    assert agg["cost_usd"] == pytest.approx(
        (4000 * 0.15 + 400 * 0.60 + 1000 * 2.0 + 100 * 8.0) / 1e6
    )
    # one unpriced model poisons the total instead of silently understating it
    rows.append({"model": "mystery", "prompt_tokens": 10, "completion_tokens": 1})
    agg = aggregate_cost(rows, prices=PRICES)
    assert agg["by_model"]["mystery"]["cost_usd"] is None
    assert agg["cost_usd"] is None
    assert agg["by_model"]["mini"]["cost_usd"] is not None  # per-model sums survive


def test_projected_cost_scales_the_mean_call_cost():
    per_call = cost_usd("mini", 3000, 150, PRICES)
    assert projected_cost(
        "mini", mean_prompt_tokens=3000.4, mean_completion_tokens=149.6, calls=659, prices=PRICES
    ) == pytest.approx(per_call * 659)
    assert (
        projected_cost(
            "nope", mean_prompt_tokens=1, mean_completion_tokens=1, calls=1, prices=PRICES
        )
        is None
    )


def test_spend_tracks_running_total_and_enforces_the_budget():
    spend = Spend(budget_usd=0.001)
    spend.add("gpt-4o-mini", 1000, 100)  # 0.00021
    spend.check()
    assert spend.total_usd == pytest.approx(0.00021)
    assert spend.add("unknown-model", 10, 10) == 0.0 and spend.unknown_price_calls == 1
    spend.add("gpt-4o-mini", 10_000, 0)  # +0.0015 -> over budget
    with pytest.raises(BudgetExceeded):
        spend.check()
    d = spend.as_dict()
    assert d["budget_usd"] == 0.001 and set(d["by_model"]) == {"gpt-4o-mini"}


# ---------- truncation / determinism ----------


def test_truncation_rate_counts_length_finishes_only():
    assert truncation_rate(["stop", "length", "stop", "length"]) == 0.5
    assert truncation_rate(["stop", None, "stop"]) == 0.0  # None = not applicable, skipped
    assert truncation_rate([]) is None and truncation_rate([None]) is None


def _row(eid, avail, *, p=0.8, llm=1, summary="s", ev=(1,), ret=(1, 2), gw=None, error=None):
    return {
        "example_id": eid,
        "availability_pred": avail,
        "start_probability": p,
        "return_gw_pred": gw,
        "summary": summary,
        "evidence": [{"chunk_id": c} for c in ev],
        "retrieved_chunk_ids": list(ret),
        "llm_calls": llm,
        "error": error,
    }


def test_agreement_is_per_id_and_ignores_rows_missing_on_either_side():
    a = [_row("x", "fit"), _row("y", "injured"), _row("z", "fit"), _row("err", "fit", error="boom")]
    b = [_row("x", "fit"), _row("y", "doubtful"), _row("w", "fit"), _row("err", "fit")]
    res = agreement(a, b, "availability_pred")
    assert res == {"n": 2, "identical": 1, "rate": 0.5, "differing": ["y"]}
    assert agreement(a, b, "availability_pred", only_ids=["x"])["rate"] == 1.0
    assert agreement([], b, "availability_pred")["rate"] is None


def test_determinism_report_separates_llm_rows_from_deterministic_abstentions():
    run1 = [
        _row("a", "fit", summary="same"),
        _row("b", "doubtful", p=0.5, summary="one"),
        _row("k", "unknown", llm=0, ev=(), p=0.5),  # abstention: no LLM call
    ]
    run2 = [
        _row("a", "fit", summary="same"),
        _row("b", "doubtful", p=0.6, summary="two"),  # same label, different number and text
        _row("k", "unknown", llm=0, ev=(), p=0.5),
    ]
    rep = determinism_report(run1, run2)
    assert rep["n_llm_rows_both"] == 2
    assert rep["availability_pred"]["all_rows"]["rate"] == 1.0
    assert rep["availability_pred"]["llm_rows"]["rate"] == 1.0
    assert rep["start_probability"]["all_rows"] == {
        "n": 3,
        "identical": 2,
        "rate": pytest.approx(2 / 3),
        "differing": ["b"],
    }
    assert (
        rep["start_probability"]["llm_rows"]["rate"] == 0.5
    )  # the abstention no longer flatters it
    assert rep["summary"]["llm_rows"]["identical"] == 1
    assert rep["evidence_chunk_ids"]["all_rows"]["rate"] == 1.0
    assert rep["fully_identical"]["all_rows"] == {
        "n": 3,
        "identical": 2,
        "rate": pytest.approx(2 / 3),
    }
    assert rep["fully_identical"]["llm_rows"] == {"n": 2, "identical": 1, "rate": 0.5}
    assert rep["retrieval_identical"]["rate"] == 1.0


# ---------- explain helpers ----------


def test_salvage_truncated_answer_decodes_the_json_prefix():
    raw = '{"answer_markdown":"**Verdict:** hold.\\n\\n**Why**\\n- xPts 6.2 \\"quoted'
    assert salvage_truncated_answer(raw) == '**Verdict:** hold.\n\n**Why**\n- xPts 6.2 "quoted'
    # cut in the middle of an escape sequence, and a complete document
    assert salvage_truncated_answer('{"answer_markdown":"line\\') == "line"
    assert salvage_truncated_answer('{"answer_markdown":"done"}') == "done"
    # not a structured payload -> returned as is
    assert salvage_truncated_answer("plain text") == "plain text"
    assert salvage_truncated_answer("") == ""


def test_format_checks_follow_the_explain_prompt_sections():
    good = "**Verdict:** go.\n\n| Option | xPts |\n|---|---|\n\n**Why**\n- a\n\n**Sources**\n- x\n\n**Caveats**\n- data as of 2026-09-17"
    fc = format_checks(good, has_evidence=True)
    assert fc["complete"] and fc["has_table"] and fc["mentions_as_of"] and fc["sources_consistent"]
    # a Sources section without evidence is an invented citation list
    assert format_checks(good, has_evidence=False)["sources_consistent"] is False
    cut = "**Verdict:** go.\n\n**Why**\n- a"
    fc = format_checks(cut, has_evidence=False)
    assert fc["has_verdict"] and fc["has_why"] and not fc["has_caveats"] and not fc["complete"]
    assert fc["sources_consistent"] is True  # no evidence, no Sources: consistent


def test_verified_numbers_matches_facts_at_the_shown_rounding():
    facts = {"players": [{"p_start": 0.64, "xpts": [6.2, 5.13], "expected_minutes": 30}], "gw": 5}
    answer = "p(start) 0.64, xPts 6.2 and 5.1 (not 5.2), minutes 30, GW5, invented 7.77, 75%"
    assert verified_numbers(answer, facts) == ["0.64", "6.2", "5.1", "30"]
    # numbers inside evidence strings count too; "GW5" is not a standalone number
    assert verified_numbers("75% chance", {}, [{"quote": "75% chance of playing"}]) == ["75"]


def _explain_row(
    cell,
    *,
    finish="stop",
    passed=True,
    viol=0,
    tokens=400,
    words=150,
    judge=1.0,
    cost=0.001,
    lat=3000,
):
    return {
        "cell": cell,
        "model": "gpt-4o-mini",
        "error": None,
        "finish_reason": finish,
        "truncated": finish == "length",
        "refusal": None,
        "validation_passed": passed,
        "violations": viol,
        "unknown_numbers": ["9.99"] * viol,
        "unknown_names": [],
        "bad_citations": [],
        "completion_tokens": tokens,
        "answer_words": words,
        "format": {"complete": finish == "stop", "sources_consistent": True},
        "judge_score": None if finish == "length" else judge,
        "judge_verdict": None
        if finish == "length"
        else ("supported" if judge == 1.0 else "partially_supported"),
        "judge_unsupported_claims": [] if judge == 1.0 else ["x"],
        "judge_model": "gpt-4o-mini",
        "judge_prompt_tokens": 0 if finish == "length" else 5000,
        "judge_completion_tokens": 0 if finish == "length" else 50,
        "prompt_tokens": 4000,
        "latency_ms": lat,
    }


def test_aggregate_explain_reports_truncation_and_validator_over_completed_answers():
    rows = [
        _explain_row(
            "T0/max300", finish="length", passed=False, viol=1, tokens=300, words=100, lat=2000
        ),
        _explain_row("T0/max300", finish="stop", tokens=280, words=110, judge=0.5, lat=2500),
        _explain_row("T0/max1000", tokens=520, judge=1.0, lat=4000),
        _explain_row("T0/max1000", passed=False, viol=2, tokens=480, judge=0.5, lat=3000),
        {"cell": "T0/max1000", "model": "gpt-4o-mini", "error": "APIError", "latency_ms": 10},
    ]
    agg = aggregate_explain(rows, "cell")
    assert list(agg) == ["T0/max300", "T0/max1000"]
    short = agg["T0/max300"]
    assert short["truncation_rate"] == 0.5 and short["truncated_n"] == 1
    assert short["validator_pass_rate_all"] == 0.5
    assert short["validator_pass_rate_completed"] == 1.0  # the truncated row is excluded
    assert short["mean_completion_tokens"] == 290 and short["mean_answer_words_completed"] == 110
    assert short["judge_mean"] == 0.5 and short["judge_n"] == 1
    assert short["judge_verdicts"] == {"supported": 0, "partially_supported": 1, "unsupported": 0}
    assert short["latency_p50_ms"] == 2000 and short["latency_p95_ms"] == 2500
    assert short["cost_usd"] == pytest.approx(2 * (4000 * 0.15 + 290 * 0.60) / 1e6, rel=1e-6)
    long = agg["T0/max1000"]
    assert long["n"] == 3 and long["n_errors"] == 1
    assert long["truncation_rate"] == 0.0
    assert long["validator_pass_rate_completed"] == 0.5 and long["mean_violations_completed"] == 1.0
    assert long["unknown_numbers_total"] == 2
    assert long["judge_mean"] == 0.75 and long["judge_unsupported_claims_mean"] == 0.5
    assert long["format_complete_rate_completed"] == 1.0
    assert long["cost_usd_judge"] == pytest.approx((10_000 * 0.15 + 100 * 0.60) / 1e6)


def test_grid_helpers():
    cells = explain_cells((0.0, 0.7), (300, 600))
    assert cells == [(0.0, 300), (0.0, 600), (0.7, 300), (0.7, 600), EXPLAIN_PRODUCTION]
    assert explain_cells((0.2,), (1400,)) == [(0.2, 1400)]  # production cell not duplicated
    assert cell_label(0.2, 1400) == "T0.2/max1400"
    arms = default_extraction_arms("m", (0.0, 0.3), (0.0,), ((0.3, 0.5),))
    assert [a.label for a in arms] == ["T0-r1", "T0-r2", "T0.3", "T0.3-p0.5"]
    assert arms[1].run == 2 and arms[-1].top_p == 0.5 and arms[-1].temperature == 0.3


# ---------- run_models helpers ----------


def _router_row(model, query, intent, expected, *, err=None):
    return {
        "model": model,
        "query": query,
        "expected_intent": expected,
        "intent": intent,
        "intent_ok": intent == expected,
        "prompt_tokens": 1400,
        "completion_tokens": 70,
        "latency_ms": 900,
        "error": err,
    }


def test_aggregate_router_accuracy_and_agreement_with_baseline():
    rows = [
        _router_row("gpt-4o-mini", "q1", "transfer", "transfer"),
        _router_row("gpt-4o-mini", "q2", "captain", "captain"),
        _router_row("gpt-4o-mini", "q3", "what_if", "transfer"),  # baseline is wrong here
        _router_row("gpt-4.1", "q1", "transfer", "transfer"),
        _router_row("gpt-4.1", "q2", "captain", "captain"),
        _router_row("gpt-4.1", "q3", "transfer", "transfer"),
    ]
    agg = aggregate_router(rows, ["gpt-4o-mini", "gpt-4.1"])
    assert agg["gpt-4o-mini"]["intent_accuracy"] == pytest.approx(2 / 3)
    assert agg["gpt-4o-mini"]["intent_agreement_with_baseline"] == 1.0
    assert agg["gpt-4.1"]["intent_accuracy"] == 1.0
    assert agg["gpt-4.1"]["intent_agreement_with_baseline"] == pytest.approx(2 / 3)
    assert agg["gpt-4.1"]["disagreements_with_baseline"] == ["q3"]
    assert agg["gpt-4o-mini"]["misses"] == [
        {"query": "q3", "intent": "what_if", "expected": "transfer"}
    ]
    assert agg["gpt-4.1"]["cost_usd_per_call"] == pytest.approx((1400 * 2.0 + 70 * 8.0) / 1e6)
    assert agg["gpt-4.1"]["latency_p50_ms"] == 900


def test_cost_projections_scale_measured_tokens_to_batch_and_query():
    extraction = {
        "gpt-4o-mini": {
            "n_examples": 27,
            "llm_calls": 23,
            "mean_prompt_tokens_per_call": 3000.0,
            "mean_completion_tokens_per_call": 150.0,
        }
    }
    router = {"gpt-4o-mini": {"cost_usd_per_call": 0.0002}}
    explain = {"gpt-4o-mini": {"cost_usd_per_call": 0.001}}
    proj = cost_projections(
        ["gpt-4o-mini", "gpt-4.1"],
        extraction_agg=extraction,
        router_agg=router,
        explain_agg=explain,
        n_players=659,
    )
    per_call = (3000 * 0.15 + 150 * 0.60) / 1e6
    p = proj["gpt-4o-mini"]
    assert p["extraction_cost_per_call"] == pytest.approx(per_call)
    assert p["batch_all_players_call_llm"] == pytest.approx(per_call * 659)
    assert p["golden_llm_call_share"] == pytest.approx(23 / 27)
    assert p["batch_with_golden_abstention_share"] == pytest.approx(per_call * 659 * 23 / 27)
    assert p["query_router_plus_explain"] == pytest.approx(0.0012)
    assert p["query_with_one_fresh_extraction"] == pytest.approx(0.0012 + per_call)
    # a model without measurements gets None, not a fabricated zero
    assert proj["gpt-4.1"]["extraction_cost_per_call"] is None
    assert proj["gpt-4.1"]["query_router_plus_explain"] is None
