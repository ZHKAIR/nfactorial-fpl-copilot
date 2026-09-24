"""Hyperparameter evals: extraction temperature / top_p / determinism, explain temperature × max_tokens.

    # extraction (27 golden signals, hybrid_rerank v2, prompt v2, gpt-4o-mini):
    #   T ∈ {0.0, 0.3, 0.7}; T = 0 and T = 0.7 repeated once (determinism); top_p 0.5 at T = 0.7
    uv run python -m evals.run_hparams --suite extraction
    # explain (6 fixed agent queries, facts computed ONCE, then only the explain call varies):
    #   T ∈ {0.0, 0.2, 0.7} × max_tokens ∈ {300, 600, 1000} + the production cell (0.2, 1400)
    uv run python -m evals.run_hparams --suite explain --manager 895045
    uv run python -m evals.run_hparams --suite all --budget-usd 1.0

Design notes
- Extraction arms share ONE retriever (same ChunkStore, same query-embedding cache) so every arm
  sees the same chunks for the same player; `retrieval_agreement_vs_first_arm` records whether the
  live corpus moved between arms. Latency is therefore reported for the LLM stage (`llm_ms`), not
  for the full pipeline.
- Determinism = per-row agreement between two runs with identical settings (availability,
  start_probability, return_gw, summary, cited chunk ids), over all rows and over the rows that
  actually called the LLM (abstentions are deterministic by construction).
- Explain: the graph runs once per query with a capturing `Deps.explain_llm`; the captured
  `ExplainRequest` (facts JSON, evidence, caveats) is replayed with the production prompt through
  the OpenAI client for every (temperature, max_tokens) cell. Truncation = finish_reason "length"
  (the SDK raises LengthFinishReasonError; in production the explain node would then fall back to a
  JSON dump of the facts). The validator is the same one the graph runs (agent/validate.py).
- Faithfulness of an explain answer: LLM-as-judge (gpt-4o-mini, T = 0, structured output,
  evals/prompts/explain_faithfulness_judge.md) over (FACTS, EVIDENCE, ANSWER).
- Costs are list-price estimates from token counts (PRICES below), never a bill. `Spend` keeps a
  running total and stops the run at --budget-usd.
- Results: evals/results/<UTC ts>_hparams_extraction.json and <UTC ts>_hparams_explain.json with
  rows + aggregates + git commit + corpus stats + config (+ the captured explain requests, so
  evals.run_models can compare models on identical facts).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from openai import LengthFinishReasonError
from pydantic import BaseModel, Field

from evals import PROMPTS_DIR, RESULTS_DIR
from evals.judge import (
    JUDGE_MODEL,
    JUDGE_PROMPT_VERSION,
    JudgeVerdict,
    format_fpl_status,
    judge_faithfulness,
    verdict_score,
)
from evals.metrics import mean, percentile
from evals.run_rag import (
    _fmt,
    _fpl_news,
    _signal_row,
    aggregate_signals,
    build_retrievers,
    corpus_stats,
    git_commit,
    render_table,
)
from evals.schemas import SignalExample, load_signals
from fplcopilot.agent.llm import PRICES_PER_1M as AGENT_PRICES
from fplcopilot.agent.llm import PROMPT_VERSION as AGENT_PROMPT_VERSION
from fplcopilot.agent.llm import (
    ExplainOutput,
    ExplainRequest,
    load_agent_prompt,
    render_explain_user,
)
from fplcopilot.agent.validate import allowed_numbers, validate_answer
from fplcopilot.config import settings
from fplcopilot.data import Bootstrap, FPLClient
from fplcopilot.rag.extract import GWCalendar, extract_signal
from fplcopilot.rag.llm import get_openai_client
from fplcopilot.rag.retrieve import Mode, Retriever, retrieval_config

log = logging.getLogger("evals.hparams")

# ---------- prices (USD per 1M tokens: input, output) ----------
# agent/llm.py holds the table the product uses for its own cost footers; the gpt-5.4 family was
# added here from the OpenAI model pages (developers.openai.com/api/docs/models/*, 17 Sep 2026).
PRICES: dict[str, tuple[float, float]] = {
    **AGENT_PRICES,
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
}

EXTRACTION_MODE: Mode = "hybrid_rerank"
EXTRACTION_K = 8
JUDGE_BUDGET = 20
DEFAULT_TEMPERATURES = (0.0, 0.3, 0.7)
DEFAULT_REPEAT_TEMPERATURES = (0.0, 0.7)  # run twice -> determinism / sampling noise
DEFAULT_TOP_P_ARMS = ((0.7, 0.5),)  # (temperature, top_p) extra arms; top_p 1.0 = the T arm itself
EXPLAIN_TEMPERATURES = (0.0, 0.2, 0.7)
EXPLAIN_MAX_TOKENS = (300, 600, 1000)
EXPLAIN_PRODUCTION = (0.2, 1400)  # agent/llm.py explain_llm defaults
EXPLAIN_JUDGE_PROMPT_VERSION = "v1"
DEFAULT_MANAGER = 895045
# Six fixed queries: demo scenarios 1–4 (status, transfer, captain, plan) + compare + lineup, so six
# different intents and six different fact shapes go through the same explain prompt.
EXPLAIN_QUERIES: tuple[tuple[str, str], ...] = (
    ("player_status", "Is João Pedro fit for GW5?"),
    ("transfer", "Should I sell Palmer?"),
    ("captain", "Who should I captain this week?"),
    ("plan", "Plan my transfers for the next 5 gameweeks"),
    ("compare_players", "Compare Haaland and Salah for the next 3 gameweeks"),
    ("lineup", "Who should I start this week?"),
)


# ---------- pure helpers (unit-tested in tests/test_evals_hparams.py) ----------


def price_for(
    model: str, prices: dict[str, tuple[float, float]] = PRICES
) -> tuple[float, float] | None:
    """(input, output) USD per 1M tokens; dated snapshots resolve to their family (longest prefix)."""
    if model in prices:
        return prices[model]
    best = None
    for name, price in prices.items():
        if model.startswith(name) and (best is None or len(name) > len(best[0])):
            best = (name, price)
    return best[1] if best else None


def cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    prices: dict[str, tuple[float, float]] = PRICES,
) -> float | None:
    """List-price estimate; None for a model without a known price (never silently 0)."""
    price = price_for(model, prices)
    if price is None:
        return None
    p_in, p_out = price
    return (prompt_tokens * p_in + completion_tokens * p_out) / 1_000_000


def aggregate_cost(
    rows: Iterable[dict[str, Any]],
    *,
    model_key: str = "model",
    prompt_key: str = "prompt_tokens",
    completion_key: str = "completion_tokens",
    prices: dict[str, tuple[float, float]] = PRICES,
) -> dict[str, Any]:
    """Token totals and cost per model over rows; rows without a model or tokens are skipped.
    `cost_usd` is None if ANY model has no price (a partial sum would understate the bill)."""
    by_model: dict[str, dict[str, Any]] = {}
    for r in rows:
        model = r.get(model_key)
        if not model:
            continue
        p = int(r.get(prompt_key) or 0)
        c = int(r.get(completion_key) or 0)
        if p == 0 and c == 0:
            continue
        slot = by_model.setdefault(
            model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
        )
        slot["calls"] += 1
        slot["prompt_tokens"] += p
        slot["completion_tokens"] += c
        cost = cost_usd(model, p, c, prices)
        slot["cost_usd"] = (
            None if (cost is None or slot["cost_usd"] is None) else slot["cost_usd"] + cost
        )
    total_cost: float | None = 0.0
    for slot in by_model.values():
        total_cost = (
            None
            if (slot["cost_usd"] is None or total_cost is None)
            else total_cost + slot["cost_usd"]
        )
    return {
        "calls": sum(s["calls"] for s in by_model.values()),
        "prompt_tokens": sum(s["prompt_tokens"] for s in by_model.values()),
        "completion_tokens": sum(s["completion_tokens"] for s in by_model.values()),
        "cost_usd": total_cost,
        "by_model": by_model,
    }


def projected_cost(
    model: str,
    *,
    mean_prompt_tokens: float,
    mean_completion_tokens: float,
    calls: float,
    prices: dict[str, tuple[float, float]] = PRICES,
) -> float | None:
    """Cost of `calls` LLM calls with the measured mean token usage (batch refresh / per query)."""
    per_call = cost_usd(model, round(mean_prompt_tokens), round(mean_completion_tokens), prices)
    return None if per_call is None else per_call * calls


def truncation_rate(finish_reasons: Iterable[str | None]) -> float | None:
    """Fraction of calls that stopped on the token limit (finish_reason == "length")."""
    xs = [fr for fr in finish_reasons if fr is not None]
    if not xs:
        return None
    return sum(1 for fr in xs if fr == "length") / len(xs)


def agreement(
    rows_a: Sequence[dict[str, Any]],
    rows_b: Sequence[dict[str, Any]],
    key: str,
    *,
    id_key: str = "example_id",
    only_ids: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Per-id agreement of `key` between two runs: {n, identical, rate, differing: [ids]}.
    Only ids present in both runs count; `only_ids` restricts further (e.g. rows that called the LLM)."""
    a = {r[id_key]: r for r in rows_a if id_key in r and not r.get("error")}
    b = {r[id_key]: r for r in rows_b if id_key in r and not r.get("error")}
    ids = sorted(set(a) & set(b), key=str)
    if only_ids is not None:
        keep = set(only_ids)
        ids = [i for i in ids if i in keep]
    differing = [i for i in ids if a[i].get(key) != b[i].get(key)]
    return {
        "n": len(ids),
        "identical": len(ids) - len(differing),
        "rate": (len(ids) - len(differing)) / len(ids) if ids else None,
        "differing": differing,
    }


def salvage_truncated_answer(raw: str) -> str:
    """Best-effort text of a structured-output answer cut by max_tokens:
    '{"answer_markdown":"**Verdict:** ...' -> '**Verdict:** ...' (JSON escapes decoded)."""
    if not raw:
        return ""
    m = re.match(r'\s*\{\s*"answer_markdown"\s*:\s*"', raw)
    if not m:
        return raw
    body = raw[m.end() :]
    if body.endswith('"}'):
        body = body[:-2]
    elif body.endswith('"'):
        body = body[:-1]
    body = body.removesuffix("\\")  # cut in the middle of an escape sequence
    try:
        return json.loads(f'"{body}"')
    except json.JSONDecodeError:
        return body.replace("\\n", "\n").replace('\\"', '"')


_SECTION = {
    "has_verdict": re.compile(r"\*\*Verdict:?\*\*", re.IGNORECASE),
    "has_why": re.compile(r"\*\*Why\*\*", re.IGNORECASE),
    "has_sources": re.compile(r"\*\*Sources\*\*", re.IGNORECASE),
    "has_caveats": re.compile(r"\*\*Caveats\*\*", re.IGNORECASE),
}


def format_checks(answer: str, *, has_evidence: bool) -> dict[str, bool]:
    """Deterministic format compliance against the explain prompt (sections, table, 'as of')."""
    out = {name: bool(rx.search(answer or "")) for name, rx in _SECTION.items()}
    out["has_table"] = any(line.lstrip().startswith("|") for line in (answer or "").splitlines())
    out["mentions_as_of"] = "as of" in (answer or "").lower()
    # Sources section only when there is evidence to list, and always when there is
    out["sources_consistent"] = out["has_sources"] == has_evidence
    out["complete"] = out["has_verdict"] and out["has_why"] and out["has_caveats"]
    return out


def word_count(text: str) -> int:
    return len((text or "").split())


def parse_floats(value: str) -> list[float]:
    return [float(v) for v in value.split(",") if v.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(v) for v in value.split(",") if v.strip()]


class Spend:
    """Running cost total with a hard budget (raises BudgetExceeded)."""

    def __init__(self, budget_usd: float | None = None) -> None:
        self.budget_usd = budget_usd
        self.total_usd = 0.0
        self.unknown_price_calls = 0
        self.by_model: dict[str, float] = {}

    def add(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        cost = cost_usd(model, prompt_tokens, completion_tokens)
        if cost is None:
            self.unknown_price_calls += 1
            return 0.0
        self.total_usd += cost
        self.by_model[model] = self.by_model.get(model, 0.0) + cost
        return cost

    def add_cost(self, cost: float | None, *, model: str = "other") -> None:
        if cost:
            self.total_usd += cost
            self.by_model[model] = self.by_model.get(model, 0.0) + cost

    def check(self) -> None:
        if self.budget_usd is not None and self.total_usd > self.budget_usd:
            raise BudgetExceeded(f"spend ${self.total_usd:.3f} > budget ${self.budget_usd:.2f}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_usd": round(self.total_usd, 6),
            "budget_usd": self.budget_usd,
            "by_model": {k: round(v, 6) for k, v in self.by_model.items()},
            "unknown_price_calls": self.unknown_price_calls,
        }


class BudgetExceeded(RuntimeError):
    pass


# ---------- extraction suite ----------


@dataclass(frozen=True)
class ExtractionArm:
    label: str
    model: str
    temperature: float
    top_p: float | None = None
    run: int = 1  # repeat index for determinism pairs

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "run": self.run,
        }


def default_extraction_arms(
    model: str,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
    repeat: Sequence[float] = DEFAULT_REPEAT_TEMPERATURES,
    top_p_arms: Sequence[tuple[float, float]] = DEFAULT_TOP_P_ARMS,
) -> list[ExtractionArm]:
    arms: list[ExtractionArm] = []
    for t in temperatures:
        runs = 2 if t in repeat else 1
        for run in range(1, runs + 1):
            suffix = f"-r{run}" if runs > 1 else ""
            arms.append(ExtractionArm(f"T{t:g}{suffix}", model, t, None, run))
    for t, top_p in top_p_arms:
        arms.append(ExtractionArm(f"T{t:g}-p{top_p:g}", model, t, top_p, 1))
    return arms


def run_extraction_arm(
    examples: Sequence[SignalExample],
    arm: ExtractionArm,
    *,
    retriever: Retriever,
    bs: Bootstrap,
    calendar: GWCalendar,
    spend: Spend,
    judge: bool = True,
    judge_budget: int = JUDGE_BUDGET,
    judge_model: str = JUDGE_MODEL,
    mode: Mode = EXTRACTION_MODE,
    k: int = EXTRACTION_K,
) -> list[dict[str, Any]]:
    """27 extractions with the arm's model / temperature / top_p; rows in evals.run_rag format plus
    the arm fields, the LLM finish_reason and a price-table cost."""
    rows: list[dict[str, Any]] = []
    judged = 0
    for ex in examples:
        timings: dict[str, Any] = {}
        try:
            sig = extract_signal(
                ex.player_id,
                ex.as_of,
                mode=mode,
                k=k,
                retriever=retriever,
                bs=bs,
                prompt_version="v2",
                abstain=True,
                calendar=calendar,
                save=False,
                timings=timings,
                model=arm.model,
                temperature=arm.temperature,
                top_p=arm.top_p,
            )
        except Exception as exc:  # one bad call must not kill the arm
            log.exception("[%s] %s failed", arm.label, ex.id)
            rows.append(
                {
                    "example_id": ex.id,
                    "mode": arm.label,
                    "player_id": ex.player_id,
                    "tags": ex.tags,
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                    **arm.as_dict(),
                }
            )
            continue
        row = _signal_row(ex, sig, arm.label, timings)
        row.update(arm.as_dict())
        row["model"] = arm.model
        row["finish_reason"] = timings.get("finish_reason")
        row["evidence_chunk_ids"] = [e["chunk_id"] for e in row["evidence"]]
        row["cost_usd"] = cost_usd(arm.model, row["prompt_tokens"], row["completion_tokens"])
        spend.add(arm.model, row["prompt_tokens"], row["completion_tokens"])
        if judge and sig.evidence and judged < judge_budget:
            res = judge_faithfulness(
                sig.summary,
                sig.evidence,
                fpl_status=format_fpl_status(
                    sig.fpl_status, sig.fpl_chance_next, _fpl_news(ex, sig, bs)
                ),
                model=judge_model,
            )
            judged += 1
            spend.add(judge_model, res.prompt_tokens, res.completion_tokens)
            row.update(
                {
                    "faithfulness": res.score,
                    "faithfulness_verdict": res.verdict,
                    "faithfulness_reason": res.reason,
                    "judge_prompt_tokens": res.prompt_tokens,
                    "judge_completion_tokens": res.completion_tokens,
                }
            )
        rows.append(row)
        log.info(
            "[%s] %s: %s (exp %s) p=%.2f fixes=%d ev=%d%s faith=%s llm=%.0f ms  | spend $%.3f",
            arm.label,
            ex.id,
            sig.availability,
            ex.expected.availability,
            sig.start_probability,
            sig.validation_fixes,
            len(sig.evidence),
            " ABSTAINED" if sig.abstained else "",
            _fmt(row["faithfulness"]),
            timings.get("llm_ms", 0),
            spend.total_usd,
        )
        spend.check()
    return rows


DETERMINISM_KEYS = (
    "availability_pred",
    "start_probability",
    "return_gw_pred",
    "summary",
    "evidence_chunk_ids",
)


def llm_called_ids(rows: Sequence[dict[str, Any]]) -> list[Any]:
    return [
        r["example_id"] for r in rows if not r.get("error") and int(r.get("llm_calls") or 0) > 0
    ]


def determinism_report(
    rows_a: Sequence[dict[str, Any]], rows_b: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Agreement between two runs with identical settings, per output field, all rows and LLM rows."""
    for r in (*rows_a, *rows_b):
        if not r.get("error") and "evidence_chunk_ids" not in r:
            r["evidence_chunk_ids"] = [e["chunk_id"] for e in r.get("evidence") or []]
    llm_ids = set(llm_called_ids(rows_a)) & set(llm_called_ids(rows_b))
    out: dict[str, Any] = {
        "n_llm_rows_both": len(llm_ids),
        "retrieval_identical": agreement(rows_a, rows_b, "retrieved_chunk_ids"),
    }
    for key in DETERMINISM_KEYS:
        out[key] = {
            "all_rows": agreement(rows_a, rows_b, key),
            "llm_rows": agreement(rows_a, rows_b, key, only_ids=llm_ids),
        }
    # "fully identical" = every compared field equal
    ids = sorted(
        {r["example_id"] for r in rows_a if not r.get("error")}
        & {r["example_id"] for r in rows_b if not r.get("error")}
    )
    a = {r["example_id"]: r for r in rows_a if not r.get("error")}
    b = {r["example_id"]: r for r in rows_b if not r.get("error")}
    same = [i for i in ids if all(a[i].get(k) == b[i].get(k) for k in DETERMINISM_KEYS)]
    out["fully_identical"] = {
        "all_rows": {
            "n": len(ids),
            "identical": len(same),
            "rate": len(same) / len(ids) if ids else None,
        },
        "llm_rows": {
            "n": len(llm_ids),
            "identical": sum(1 for i in same if i in llm_ids),
            "rate": (sum(1 for i in same if i in llm_ids) / len(llm_ids)) if llm_ids else None,
        },
    }
    return out


def aggregate_extraction_arm(
    rows: Sequence[dict[str, Any]],
    arm: ExtractionArm,
    *,
    judge_model: str = JUDGE_MODEL,
    reference_rows: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """evals.run_rag aggregates + price-table cost, LLM-stage latency, finish reasons, retrieval
    agreement with the first arm (did the live corpus move between arms?)."""
    agg = aggregate_signals(rows, [arm.label], judge_model=judge_model)[arm.label]
    ok = [r for r in rows if not r.get("error")]
    llm_rows = [r for r in ok if int(r.get("llm_calls") or 0) > 0]
    costs = aggregate_cost(llm_rows)
    judge_cost = cost_usd(
        judge_model,
        sum(int(r.get("judge_prompt_tokens") or 0) for r in ok),
        sum(int(r.get("judge_completion_tokens") or 0) for r in ok),
    )
    llm_ms = [r["llm_ms"] for r in llm_rows if r.get("llm_ms") is not None]
    agg.update(
        {
            **arm.as_dict(),
            "cost_usd_signals": costs["cost_usd"],
            "cost_usd_judge": judge_cost,
            "cost_usd_total": (costs["cost_usd"] or 0.0) + (judge_cost or 0.0),
            "mean_prompt_tokens_per_call": mean(r["prompt_tokens"] for r in llm_rows),
            "mean_completion_tokens_per_call": mean(r["completion_tokens"] for r in llm_rows),
            "cost_usd_per_llm_call": (costs["cost_usd"] / len(llm_rows))
            if llm_rows and costs["cost_usd"] is not None
            else None,
            "llm_ms_p50": percentile(llm_ms, 50),
            "llm_ms_p95": percentile(llm_ms, 95),
            "llm_ms_mean": mean(llm_ms),
            "finish_reasons": {
                str(fr): sum(1 for r in llm_rows if r.get("finish_reason") == fr)
                for fr in sorted({r.get("finish_reason") for r in llm_rows}, key=str)
            },
            "strict_misses": [
                {
                    "id": r["example_id"],
                    "pred": r["availability_pred"],
                    "expected": r["expected_availability"],
                }
                for r in ok
                if not r.get("availability_exact")
            ],
        }
    )
    if reference_rows is not None:
        agg["retrieval_agreement_vs_first_arm"] = agreement(
            reference_rows, rows, "retrieved_chunk_ids"
        )
    return agg


EXTRACTION_TABLE = (
    "availability_accuracy",
    "availability_accuracy_lenient",
    "availability_macro_f1",
    "start_probability_in_range_rate",
    "return_gw_exact_rate",
    "evidence_present_rate",
    "false_evidence_rate",
    "mean_validation_fixes",
    "abstained_count",
    "llm_calls",
    "faithfulness_mean",
    "faithfulness_n_judged",
    "mean_prompt_tokens_per_call",
    "mean_completion_tokens_per_call",
    "cost_usd_signals",
    "cost_usd_judge",
    "llm_ms_p50",
    "llm_ms_p95",
)


def run_extraction_suite(
    *,
    arms: Sequence[ExtractionArm],
    spend: Spend,
    judge: bool = True,
    judge_budget: int = JUDGE_BUDGET,
    ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    examples = load_signals()
    if ids:
        wanted = set(ids)
        examples = [e for e in examples if e.id in wanted]
    if limit:
        examples = examples[:limit]
    as_of = max(e.as_of for e in examples)
    r_config = retrieval_config("v2", candidates=settings.rag_candidates)
    retriever = build_retrievers([EXTRACTION_MODE], r_config)[EXTRACTION_MODE]
    client = FPLClient()
    bs = client.bootstrap()
    calendar = GWCalendar.from_fpl(bs, client.fixtures())
    corpus_before = corpus_stats(as_of)
    print(
        f"[extraction] {len(examples)} players, as_of={as_of:%Y-%m-%dT%H:%MZ}, mode={EXTRACTION_MODE} "
        f"k={EXTRACTION_K}, retrieval v2, prompt v2, arms={[a.label for a in arms]}\n"
        f"corpus: {corpus_before['articles_at_as_of']} articles / {corpus_before['chunks_at_as_of']} "
        f"chunks visible at as_of"
    )
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    aggregates: dict[str, dict[str, Any]] = {}
    first_rows: list[dict[str, Any]] | None = None
    for arm in arms:
        started = time.perf_counter()
        rows = run_extraction_arm(
            examples,
            arm,
            retriever=retriever,
            bs=bs,
            calendar=calendar,
            spend=spend,
            judge=judge,
            judge_budget=judge_budget,
        )
        rows_by_arm[arm.label] = rows
        aggregates[arm.label] = aggregate_extraction_arm(rows, arm, reference_rows=first_rows)
        aggregates[arm.label]["wall_s"] = round(time.perf_counter() - started, 1)
        if first_rows is None:
            first_rows = rows
        a = aggregates[arm.label]
        print(
            f"  {arm.label:<10} acc {_fmt(a['availability_accuracy'])} lenient "
            f"{_fmt(a['availability_accuracy_lenient'])} fixes {_fmt(a['mean_validation_fixes'])} "
            f"faith {_fmt(a['faithfulness_mean'])} cost ${a['cost_usd_total']:.4f} "
            f"llm p50 {_fmt(a['llm_ms_p50'])} ms | running spend ${spend.total_usd:.3f}"
        )
    # determinism pairs: same model / temperature / top_p, run 1 vs run 2
    determinism: dict[str, Any] = {}
    by_setting: dict[tuple[str, float, float | None], list[ExtractionArm]] = {}
    for arm in arms:
        by_setting.setdefault((arm.model, arm.temperature, arm.top_p), []).append(arm)
    for (model, t, top_p), group in by_setting.items():
        if len(group) >= 2:
            a, b = group[0], group[1]
            determinism[f"{a.label}~{b.label}"] = {
                "model": model,
                "temperature": t,
                "top_p": top_p,
                **determinism_report(rows_by_arm[a.label], rows_by_arm[b.label]),
            }
    corpus_after = corpus_stats(as_of)
    all_rows = [r for rows in rows_by_arm.values() for r in rows]
    return {
        "suite": "hparams_extraction",
        "as_of": as_of.isoformat(),
        "config": {
            "mode": EXTRACTION_MODE,
            "k": EXTRACTION_K,
            "prompt_version": "v2",
            "retrieval_version": r_config.version,
            "retrieval": {
                "per_article_cap": r_config.per_article_cap,
                "date_aware_rerank": r_config.date_aware_rerank,
                "candidates": r_config.candidates,
            },
            "abstain": True,
            "embedding_model": settings.rag_embedding_model,
            "reranker": retriever.reranker.name,
            "judge_model": JUDGE_MODEL,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "judge_budget_per_arm": judge_budget if judge else 0,
            "max_completion_tokens": 800,
            "prices_per_1m_usd": {k: list(v) for k, v in PRICES.items()},
            "shared_retriever": True,
        },
        "arms": [a.as_dict() for a in arms],
        "corpus_before": corpus_before,
        "corpus_after": corpus_after,
        "golden_n": len(examples),
        "aggregates": aggregates,
        "determinism": determinism,
        "rows": all_rows,
    }


# ---------- explain suite ----------


class ExplainJudgeVerdict(BaseModel):
    verdict: Literal["supported", "partially_supported", "unsupported"]
    unsupported_claims: list[str] = Field(
        description="up to 5 short verbatim fragments of the answer that FACTS/EVIDENCE do not support"
    )
    reason: str = Field(description="one sentence, <= 200 characters")


@lru_cache(maxsize=1)
def load_explain_judge_prompt() -> str:
    return (PROMPTS_DIR / "explain_faithfulness_judge.md").read_text(encoding="utf-8").strip()


_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")


def verified_numbers(answer: str, facts: Any, evidence: Any = None) -> list[str]:
    """Numeric tokens of the answer that code can find in FACTS/EVIDENCE at the shown rounding
    (the validator's rule, extended to integers). Handed to the judge as pre-verified so it does
    not have to locate 0.64 inside a 3k-token JSON — its job is names, fixtures, news, verdicts."""
    numbers, forms = allowed_numbers(facts, evidence or [])
    # integers written inside strings ("75% chance of playing", "GW5 deadline") are numbers too
    forms = forms | set(_NUMBER.findall(json.dumps([facts, evidence or []], default=str)))
    out: list[str] = []
    for tok in dict.fromkeys(_NUMBER.findall(answer or "")):
        if tok in forms:
            out.append(tok)
            continue
        decimals = len(tok.split(".")[1]) if "." in tok else 0
        x = float(tok)
        if any(abs(round(abs(n), decimals) - x) < 1e-9 for n in numbers):
            out.append(tok)
    return out


def build_explain_judge_messages(
    answer: str, facts: dict[str, Any], evidence: Sequence[dict[str, Any]]
) -> list[dict[str, str]]:
    lines = []
    for i, ev in enumerate(evidence, start=1):
        quote = " ".join(str(ev.get("quote") or "").split())
        lines.append(f'{i}. [{ev.get("source")} {ev.get("date")}] ({ev.get("player")}) "{quote}"')
    verified = verified_numbers(answer, facts, evidence)
    user = (
        "FACTS (JSON):\n"
        + json.dumps(facts, ensure_ascii=False, indent=1, default=str)
        + "\n\nEVIDENCE:\n"
        + ("\n".join(lines) if lines else "(none)")
        + "\n\nNUMBERS IN THE ANSWER ALREADY VERIFIED AGAINST FACTS BY CODE (treat as supported):\n"
        + (", ".join(verified) if verified else "(none)")
        + "\n\nANSWER:\n"
        + (answer or "").strip()
    )
    return [
        {"role": "system", "content": load_explain_judge_prompt()},
        {"role": "user", "content": user},
    ]


def judge_explain(
    answer: str,
    facts: dict[str, Any],
    evidence: Sequence[dict[str, Any]],
    *,
    model: str = JUDGE_MODEL,
    client: Any | None = None,
) -> dict[str, Any]:
    client = client or get_openai_client()
    completion = client.chat.completions.parse(
        model=model,
        messages=build_explain_judge_messages(answer, facts, evidence),
        response_format=ExplainJudgeVerdict,
        temperature=0,
        max_completion_tokens=400,
    )
    msg = completion.choices[0].message
    usage = completion.usage
    parsed: ExplainJudgeVerdict | None = msg.parsed
    return {
        "judge_score": verdict_score(JudgeVerdict(verdict=parsed.verdict, reason=parsed.reason))
        if parsed
        else None,
        "judge_verdict": parsed.verdict if parsed else None,
        "judge_unsupported_claims": parsed.unsupported_claims if parsed else [],
        "judge_reason": parsed.reason.strip() if parsed else "judge returned no structured verdict",
        "judge_prompt_tokens": usage.prompt_tokens if usage else 0,
        "judge_completion_tokens": usage.completion_tokens if usage else 0,
        "judge_model": model,
    }


@dataclass
class CapturedExplain:
    """One agent run: the first (feedback-free) ExplainRequest and what the validator needs."""

    intent_expected: str
    query: str
    thread_id: str
    request: dict[str, Any]  # ExplainRequest.model_dump()
    validator_names: list[str]
    intent: str | None
    graph_answer: str | None
    graph_validation: dict[str, Any] | None
    graph_llm_calls: list[dict[str, Any]] = field(default_factory=list)
    graph_cost_usd: float = 0.0
    wall_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent_expected": self.intent_expected,
            "query": self.query,
            "thread_id": self.thread_id,
            "request": self.request,
            "validator_names": self.validator_names,
            "intent": self.intent,
            "graph_answer": self.graph_answer,
            "graph_validation": self.graph_validation,
            "graph_llm_calls": self.graph_llm_calls,
            "graph_cost_usd": self.graph_cost_usd,
            "wall_s": self.wall_s,
        }


def capture_explain_requests(
    queries: Sequence[tuple[str, str]],
    *,
    manager_id: int,
    strategy: str,
    spend: Spend,
) -> list[CapturedExplain]:
    """Run the live agent once per query with a capturing explain_llm. Facts are computed by the
    tools exactly as in production (this is the normal agent path, so stale signals may be
    re-extracted and saved, as in scripts/agent_demo.py); the graph's own explain call (T = 0.2,
    max_tokens 1400) still happens, so the graph state is a real production sample."""
    from fplcopilot.agent.graph import Agent, live_deps

    deps = live_deps(grader=True)
    real_explain = deps.explain_llm
    captured: list[ExplainRequest] = []

    def capturing(req: ExplainRequest):
        captured.append(req)
        return real_explain(req)

    deps.explain_llm = capturing
    agent = Agent(deps)
    out: list[CapturedExplain] = []
    try:
        for intent_expected, query in queries:
            captured.clear()
            started = time.perf_counter()
            state = agent.run(query, manager_id=manager_id, strategy=strategy)
            if state.get("interrupted") and state.get("pending_action"):
                log.info("%s: interrupted on %s -> resume confirm", query, state["pending_action"])
                state = agent.resume(state["thread_id"], "confirm")
            wall = time.perf_counter() - started
            for c in state.get("llm_calls") or []:
                spend.add(
                    str(c.get("model")),
                    int(c.get("prompt_tokens") or 0),
                    int(c.get("completion_tokens") or 0),
                )
            first = next((r for r in captured if not r.feedback), captured[0] if captured else None)
            if first is None:
                log.warning(
                    "%s: graph produced no explain call (intent=%s)", query, state.get("intent")
                )
                continue
            names = [p["name"] for p in state.get("players") or []] + [
                p.get("full_name", "") for p in state.get("players") or []
            ]
            summary = state.get("squad_summary") or {}
            names += [p["name"] for p in summary.get("squad") or []]
            names += deps.extra_known_names
            validation = state.get("validation") or {}
            out.append(
                CapturedExplain(
                    intent_expected=intent_expected,
                    query=query,
                    thread_id=state["thread_id"],
                    request=first.model_dump(mode="json"),
                    validator_names=names,
                    intent=state.get("intent"),
                    graph_answer=state.get("answer"),
                    graph_validation=(validation.get("history") or [None])[0],
                    graph_llm_calls=list(state.get("llm_calls") or []),
                    graph_cost_usd=float(state.get("cost_estimate") or 0.0),
                    wall_s=round(wall, 1),
                )
            )
            print(
                f"  captured {intent_expected:<16} intent={state.get('intent')} facts keys="
                f"{sorted((first.facts or {}).keys())} evidence={len(first.evidence)} "
                f"({wall:.1f} s, ${state.get('cost_estimate', 0):.4f}) | spend ${spend.total_usd:.3f}"
            )
            spend.check()
    finally:
        agent.close()
    return out


def load_captured(path: Path) -> list[CapturedExplain]:
    """Captured ExplainRequests of an earlier explain run: replay the grid on identical facts."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        CapturedExplain(
            intent_expected=c["intent_expected"],
            query=c["query"],
            thread_id=c["thread_id"],
            request=c["request"],
            validator_names=c["validator_names"],
            intent=c.get("intent"),
            graph_answer=c.get("graph_answer"),
            graph_validation=c.get("graph_validation"),
            graph_llm_calls=c.get("graph_llm_calls") or [],
            graph_cost_usd=float(c.get("graph_cost_usd") or 0.0),
            wall_s=float(c.get("wall_s") or 0.0),
        )
        for c in data["captured"]
    ]


def replay_explain(
    req: ExplainRequest,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    client: Any | None = None,
) -> dict[str, Any]:
    """The explain call exactly as agent/llm.explain_llm makes it (same prompt file of the active
    AGENT_PROMPT_VERSION, same user rendering, same schema), with temperature / max_tokens / model
    as parameters and the finish_reason exposed. Truncated answers are salvaged from the raw JSON
    for length/validator."""
    client = client or get_openai_client()
    messages = [
        {"role": "system", "content": load_agent_prompt("explain.system")},
        {"role": "user", "content": render_explain_user(req)},
    ]
    started = time.perf_counter()
    finish: str | None = None
    answer = ""
    refusal: str | None = None
    error: str | None = None
    completion: Any = None
    try:
        completion = client.chat.completions.parse(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            response_format=ExplainOutput,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        choice = completion.choices[0]
        finish = choice.finish_reason
        if choice.message.parsed is not None:
            answer = choice.message.parsed.answer_markdown
        else:
            refusal = choice.message.refusal or "no structured answer"
    except LengthFinishReasonError as exc:
        completion = exc.completion
        choice = completion.choices[0]
        finish = "length"
        answer = salvage_truncated_answer(choice.message.content or "")
    except Exception as exc:  # content filter, API error: record, do not kill the grid
        log.warning(
            "explain replay failed (%s, T=%s, max=%s)",
            model,
            temperature,
            max_tokens,
            exc_info=True,
        )
        error = f"{type(exc).__name__}: {exc}"[:300]
        completion = getattr(exc, "completion", None)
    latency_ms = (time.perf_counter() - started) * 1000
    usage = getattr(completion, "usage", None)
    p_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
    c_tok = int(getattr(usage, "completion_tokens", 0) or 0)
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "finish_reason": finish,
        "truncated": finish == "length",
        "refusal": refusal,
        "error": error,
        "answer": answer,
        "answer_chars": len(answer),
        "answer_words": word_count(answer),
        "prompt_tokens": p_tok,
        "completion_tokens": c_tok,
        "cost_usd": cost_usd(model, p_tok, c_tok),
        "latency_ms": round(latency_ms),
    }


def score_explain_answer(answer: str, cap: CapturedExplain) -> dict[str, Any]:
    """Deterministic checks the graph would run: validator (names / numbers / placeholder citations)
    and format compliance."""
    req = cap.request
    v = validate_answer(
        answer, req.get("facts") or {}, req.get("evidence") or [], extra_names=cap.validator_names
    )
    fmt = format_checks(answer, has_evidence=bool(req.get("evidence")))
    return {
        "validation_passed": v.passed,
        "unknown_names": v.unknown_names,
        "unknown_numbers": v.unknown_numbers,
        "bad_citations": v.bad_citations,
        "violations": len(v.unknown_names) + len(v.unknown_numbers) + len(v.bad_citations),
        "checked_names": v.checked_names,
        "checked_numbers": v.checked_numbers,
        "format": fmt,
    }


def explain_cells(
    temperatures: Sequence[float] = EXPLAIN_TEMPERATURES,
    max_tokens: Sequence[int] = EXPLAIN_MAX_TOKENS,
    production: tuple[float, int] | None = EXPLAIN_PRODUCTION,
) -> list[tuple[float, int]]:
    cells = [(t, m) for t in temperatures for m in max_tokens]
    if production is not None and production not in cells:
        cells.append(production)
    return cells


def cell_label(temperature: float, max_tokens: int) -> str:
    return f"T{temperature:g}/max{max_tokens}"


def run_explain_grid(
    captured: Sequence[CapturedExplain],
    *,
    model: str,
    cells: Sequence[tuple[float, int]],
    spend: Spend,
    judge: bool = True,
    judge_model: str = JUDGE_MODEL,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    client = get_openai_client()
    for cap in captured:
        req = ExplainRequest.model_validate(cap.request)
        for t, m in cells:
            res = replay_explain(req, model=model, temperature=t, max_tokens=m, client=client)
            spend.add(model, res["prompt_tokens"], res["completion_tokens"])
            row: dict[str, Any] = {
                "query": cap.query,
                "intent": cap.intent,
                "intent_expected": cap.intent_expected,
                "cell": cell_label(t, m),
                **res,
            }
            if res["error"] is None:
                row.update(score_explain_answer(res["answer"], cap))
            if judge and res["error"] is None and not res["truncated"] and res["answer"]:
                j = judge_explain(
                    res["answer"], req.facts, req.evidence, model=judge_model, client=client
                )
                spend.add(judge_model, j["judge_prompt_tokens"], j["judge_completion_tokens"])
                row.update(j)
            rows.append(row)
            log.info(
                "[%s] %-16s %s: finish=%s tokens=%d words=%d valid=%s viol=%s judge=%s (%d ms) | spend $%.3f",
                model,
                cap.intent_expected,
                row["cell"],
                res["finish_reason"],
                res["completion_tokens"],
                res["answer_words"],
                row.get("validation_passed"),
                row.get("violations"),
                row.get("judge_verdict"),
                res["latency_ms"],
                spend.total_usd,
            )
            spend.check()
    return rows


def aggregate_explain(
    rows: Sequence[dict[str, Any]], group_key: str = "cell"
) -> dict[str, dict[str, Any]]:
    """Per group (cell / temperature / max_tokens / model): truncation, validator, length, judge, cost, latency."""
    out: dict[str, dict[str, Any]] = {}
    groups = sorted({str(r[group_key]) for r in rows}, key=_group_sort_key)
    for g in groups:
        gr = [r for r in rows if str(r[group_key]) == g]
        ok = [r for r in gr if r.get("error") is None]
        done = [r for r in ok if not r.get("truncated")]
        judged = [r for r in done if r.get("judge_score") is not None]
        lat = [r["latency_ms"] for r in ok]
        costs = aggregate_cost(ok)
        judge_cost = cost_usd(
            str(next((r.get("judge_model") for r in judged), JUDGE_MODEL)),
            sum(int(r.get("judge_prompt_tokens") or 0) for r in judged),
            sum(int(r.get("judge_completion_tokens") or 0) for r in judged),
        )
        out[g] = {
            "n": len(gr),
            "n_errors": len(gr) - len(ok),
            "truncation_rate": truncation_rate(r.get("finish_reason") for r in ok),
            "truncated_n": sum(1 for r in ok if r.get("truncated")),
            "refusals": sum(1 for r in ok if r.get("refusal")),
            "validator_pass_rate_all": mean(r.get("validation_passed") for r in ok),
            "validator_pass_rate_completed": mean(r.get("validation_passed") for r in done),
            "mean_violations_all": mean(r.get("violations") for r in ok),
            "mean_violations_completed": mean(r.get("violations") for r in done),
            "unknown_numbers_total": sum(len(r.get("unknown_numbers") or []) for r in ok),
            "unknown_names_total": sum(len(r.get("unknown_names") or []) for r in ok),
            "bad_citations_total": sum(len(r.get("bad_citations") or []) for r in ok),
            "mean_completion_tokens": mean(r.get("completion_tokens") for r in ok),
            "mean_completion_tokens_completed": mean(r.get("completion_tokens") for r in done),
            "mean_answer_words": mean(r.get("answer_words") for r in ok),
            "mean_answer_words_completed": mean(r.get("answer_words") for r in done),
            "format_complete_rate": mean((r.get("format") or {}).get("complete") for r in ok),
            "format_complete_rate_completed": mean(
                (r.get("format") or {}).get("complete") for r in done
            ),
            "sources_consistent_rate_completed": mean(
                (r.get("format") or {}).get("sources_consistent") for r in done
            ),
            "judge_mean": mean(r.get("judge_score") for r in judged),
            "judge_n": len(judged),
            "judge_verdicts": {
                v: sum(1 for r in judged if r.get("judge_verdict") == v)
                for v in ("supported", "partially_supported", "unsupported")
            },
            "judge_unsupported_claims_mean": mean(
                len(r.get("judge_unsupported_claims") or []) for r in judged
            ),
            "prompt_tokens": costs["prompt_tokens"],
            "completion_tokens": costs["completion_tokens"],
            "cost_usd": costs["cost_usd"],
            "cost_usd_per_call": (costs["cost_usd"] / len(ok))
            if ok and costs["cost_usd"] is not None
            else None,
            "cost_usd_judge": judge_cost,
            "latency_p50_ms": percentile(lat, 50),
            "latency_p95_ms": percentile(lat, 95),
            "latency_mean_ms": mean(lat),
        }
    return out


def _group_sort_key(g: str) -> tuple:
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", g)]
    return (tuple(nums), g)


EXPLAIN_TABLE = (
    "truncation_rate",
    "validator_pass_rate_all",
    "validator_pass_rate_completed",
    "mean_violations_completed",
    "unknown_numbers_total",
    "unknown_names_total",
    "bad_citations_total",
    "mean_completion_tokens",
    "mean_answer_words_completed",
    "format_complete_rate_completed",
    "judge_mean",
    "judge_n",
    "cost_usd",
    "latency_p50_ms",
)


def run_explain_suite(
    *,
    manager_id: int,
    strategy: str,
    model: str,
    cells: Sequence[tuple[float, int]],
    spend: Spend,
    judge: bool = True,
    queries: Sequence[tuple[str, str]] = EXPLAIN_QUERIES,
    captured: Sequence[CapturedExplain] | None = None,
    captured_from: Path | None = None,
) -> dict[str, Any]:
    if captured is None:
        print(
            f"[explain] capturing facts once for {len(queries)} queries "
            f"(manager {manager_id}, {strategy})"
        )
        captured = capture_explain_requests(
            queries, manager_id=manager_id, strategy=strategy, spend=spend
        )
    else:
        print(f"[explain] reusing {len(captured)} captured requests from {captured_from}")
    print(f"[explain] replaying {len(cells)} cells × {len(captured)} queries with {model}")
    rows = run_explain_grid(captured, model=model, cells=cells, spend=spend, judge=judge)
    for r in rows:
        r["temperature_label"] = f"T{r['temperature']:g}"
        r["max_tokens_label"] = f"max{r['max_tokens']}"
    return {
        "suite": "hparams_explain",
        "config": {
            "model": model,
            "manager_id": manager_id,
            "strategy": strategy,
            "cells": [cell_label(t, m) for t, m in cells],
            "production_cell": cell_label(*EXPLAIN_PRODUCTION),
            # the active agent prompt version (AGENT_PROMPT_VERSION); the 17 Sep 2026 runs used v1
            "explain_prompt_version": AGENT_PROMPT_VERSION,
            "captured_from": str(captured_from) if captured_from else None,
            "judge_model": JUDGE_MODEL,
            "judge_prompt_version": EXPLAIN_JUDGE_PROMPT_VERSION,
            "judge_prompt": "evals/prompts/explain_faithfulness_judge.md",
            "prices_per_1m_usd": {k: list(v) for k, v in PRICES.items()},
        },
        "captured": [c.as_dict() for c in captured],
        "aggregates": {
            "by_cell": aggregate_explain(rows, "cell"),
            "by_temperature": aggregate_explain(rows, "temperature_label"),
            "by_max_tokens": aggregate_explain(rows, "max_tokens_label"),
        },
        "rows": rows,
    }


def rejudge_explain_results(path: Path, *, judge_model: str, spend: Spend) -> dict[str, Any]:
    """Re-score every completed answer of an explain results file with another judge model
    (same prompt, same FACTS/EVIDENCE) and recompute the aggregates. The previous verdicts are kept
    as judge_verdict_prev so the two judges can be compared row by row."""
    data = json.loads(path.read_text(encoding="utf-8"))
    by_query = {c["query"]: c for c in data["captured"]}
    client = get_openai_client()
    prev_model = data["config"].get("judge_model")
    for r in data["rows"]:
        r["judge_verdict_prev"] = r.get("judge_verdict")
        r["judge_score_prev"] = r.get("judge_score")
        r["judge_unsupported_claims_prev"] = r.get("judge_unsupported_claims")
        if r.get("error") is not None or r.get("truncated") or not r.get("answer"):
            continue
        req = by_query[r["query"]]["request"]
        j = judge_explain(
            r["answer"],
            req.get("facts") or {},
            req.get("evidence") or [],
            model=judge_model,
            client=client,
        )
        spend.add(judge_model, j["judge_prompt_tokens"], j["judge_completion_tokens"])
        r.update(j)
        log.info(
            "[rejudge/%s] %-16s %s: %s (prev %s) | spend $%.3f",
            judge_model,
            r["intent"],
            r["cell"],
            r["judge_verdict"],
            r["judge_verdict_prev"],
            spend.total_usd,
        )
        spend.check()
    judged = [r for r in data["rows"] if r.get("judge_verdict") and r.get("judge_verdict_prev")]
    data["config"]["judge_model"] = judge_model
    data["config"]["judge_model_prev"] = prev_model
    data["config"]["rejudged_from"] = str(path)
    data["aggregates"] = {
        "by_cell": aggregate_explain(data["rows"], "cell"),
        "by_temperature": aggregate_explain(data["rows"], "temperature_label"),
        "by_max_tokens": aggregate_explain(data["rows"], "max_tokens_label"),
    }
    data["judge_comparison"] = {
        "n": len(judged),
        "verdict_agreement": (
            sum(1 for r in judged if r["judge_verdict"] == r["judge_verdict_prev"]) / len(judged)
            if judged
            else None
        ),
        "mean_score_prev": mean(r["judge_score_prev"] for r in judged),
        "mean_score_new": mean(r["judge_score"] for r in judged),
        "verdicts_prev": {
            v: sum(1 for r in judged if r["judge_verdict_prev"] == v)
            for v in ("supported", "partially_supported", "unsupported")
        },
        "verdicts_new": {
            v: sum(1 for r in judged if r["judge_verdict"] == v)
            for v in ("supported", "partially_supported", "unsupported")
        },
        "mean_unsupported_claims_prev": mean(
            len(r.get("judge_unsupported_claims_prev") or []) for r in judged
        ),
        "mean_unsupported_claims_new": mean(
            len(r.get("judge_unsupported_claims") or []) for r in judged
        ),
    }
    return data


# ---------- persistence / CLI ----------


def save(payload: dict[str, Any], out_dir: Path, name: str, stamp: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stamp}_{name}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    return path


def table_for(
    aggregates: dict[str, dict[str, Any]], keys: Sequence[str], cols: Sequence[str]
) -> str:
    return render_table({k: {c: aggregates[c].get(k) for c in cols} for k in keys}, cols)


def print_determinism(det: dict[str, Any]) -> None:
    for pair, d in det.items():
        print(f"  determinism {pair} (T={d['temperature']:g}, top_p={d['top_p']}):")
        for key in (
            "availability_pred",
            "start_probability",
            "summary",
            "evidence_chunk_ids",
            "fully_identical",
        ):
            a, b = d[key]["all_rows"], d[key]["llm_rows"]
            print(
                f"    {key:<20} all {a['identical']}/{a['n']} ({_fmt(a['rate'])})  "
                f"llm rows {b['identical']}/{b['n']} ({_fmt(b['rate'])})"
            )
        ri = d["retrieval_identical"]
        print(f"    retrieval identical  {ri['identical']}/{ri['n']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="FPL Copilot: hyperparameter evals (T / top_p / max_tokens)"
    )
    ap.add_argument("--suite", choices=("extraction", "explain", "all"), default="all")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    ap.add_argument(
        "--budget-usd", type=float, default=1.5, help="stop when the estimated spend exceeds this"
    )
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument(
        "--judge-budget", type=int, default=JUDGE_BUDGET, help="extraction judge calls per arm"
    )
    # extraction
    ap.add_argument(
        "--model", default=settings.rag_llm_model, help="extraction model for the T/top_p arms"
    )
    ap.add_argument("--temperatures", type=parse_floats, default=list(DEFAULT_TEMPERATURES))
    ap.add_argument(
        "--repeat",
        type=parse_floats,
        default=list(DEFAULT_REPEAT_TEMPERATURES),
        help="temperatures run twice",
    )
    ap.add_argument(
        "--top-p", default="0.7:0.5", help="extra arms as T:top_p[,T:top_p]; '' for none"
    )
    ap.add_argument("--ids", help="comma-separated golden ids (debug)")
    ap.add_argument("--limit", type=int)
    # explain
    ap.add_argument("--manager", type=int, default=DEFAULT_MANAGER)
    ap.add_argument("--strategy", default="balanced")
    ap.add_argument("--explain-model", default=settings.agent_explain_model)
    ap.add_argument("--explain-temperatures", type=parse_floats, default=list(EXPLAIN_TEMPERATURES))
    ap.add_argument("--explain-max-tokens", type=parse_ints, default=list(EXPLAIN_MAX_TOKENS))
    ap.add_argument(
        "--no-production-cell", action="store_true", help="skip the (0.2, 1400) reference cell"
    )
    ap.add_argument(
        "--explain-requests",
        type=Path,
        help="reuse the captured ExplainRequests of an earlier *_hparams_explain.json (same facts)",
    )
    ap.add_argument(
        "--rejudge",
        type=Path,
        help="re-score the completed answers of an *_hparams_explain.json with --judge-model "
        "(no new explain calls); writes <ts>_hparams_explain_judge-<model>.json",
    )
    ap.add_argument("--judge-model", default=JUDGE_MODEL, help="judge model for --rejudge")
    ap.add_argument("--tag", default="")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("httpx", "httpx2", "httpcore", "openai", "langsmith", "fplcopilot", "evals"):
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("evals.hparams").setLevel(logging.INFO)

    stamp = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    commit = git_commit()
    spend = Spend(args.budget_usd)
    base = {"created_at": datetime.now(UTC).isoformat(), "git_commit": commit}
    tag = f"_{args.tag.strip('_')}" if args.tag else ""
    written: list[Path] = []
    print(f"commit={(commit or '?')[:8]} budget=${args.budget_usd:.2f}")
    try:
        if args.rejudge:
            payload = rejudge_explain_results(
                args.rejudge, judge_model=args.judge_model, spend=spend
            )
            payload["rejudged_at"] = datetime.now(UTC).isoformat()
            payload["rejudge_spend"] = spend.as_dict()
            written.append(
                save(payload, args.out, f"hparams_explain_judge-{args.judge_model}{tag}", stamp)
            )
            agg = payload["aggregates"]
            print(f"\n== explain re-judged with {args.judge_model} ==")
            print(table_for(agg["by_cell"], EXPLAIN_TABLE, list(agg["by_cell"])))
            print("\njudge comparison:", json.dumps(payload["judge_comparison"], indent=1))
            return 0
        if args.suite in ("extraction", "all"):
            top_p_arms = [
                (float(t), float(p))
                for t, p in (x.split(":") for x in args.top_p.split(",") if x.strip())
            ]
            arms = default_extraction_arms(args.model, args.temperatures, args.repeat, top_p_arms)
            payload = run_extraction_suite(
                arms=arms,
                spend=spend,
                judge=not args.no_judge,
                judge_budget=args.judge_budget,
                ids=args.ids.split(",") if args.ids else None,
                limit=args.limit,
            )
            payload.update(base)
            payload["spend"] = spend.as_dict()
            written.append(save(payload, args.out, f"hparams_extraction{tag}", stamp))
            print(f"\n== extraction: {args.model}, {payload['golden_n']} players ==")
            print(table_for(payload["aggregates"], EXTRACTION_TABLE, [a.label for a in arms]))
            print_determinism(payload["determinism"])
        if args.suite in ("explain", "all"):
            cells = explain_cells(
                args.explain_temperatures,
                args.explain_max_tokens,
                None if args.no_production_cell else EXPLAIN_PRODUCTION,
            )
            payload = run_explain_suite(
                manager_id=args.manager,
                strategy=args.strategy,
                model=args.explain_model,
                cells=cells,
                spend=spend,
                judge=not args.no_judge,
                captured=load_captured(args.explain_requests) if args.explain_requests else None,
                captured_from=args.explain_requests,
            )
            payload.update(base)
            payload["spend"] = spend.as_dict()
            written.append(save(payload, args.out, f"hparams_explain{tag}", stamp))
            agg = payload["aggregates"]
            print(f"\n== explain: {args.explain_model}, {len(payload['captured'])} queries ==")
            print(table_for(agg["by_cell"], EXPLAIN_TABLE, list(agg["by_cell"])))
            print("\nby temperature (pooled over max_tokens):")
            print(table_for(agg["by_temperature"], EXPLAIN_TABLE, list(agg["by_temperature"])))
            print("\nby max_tokens (pooled over temperature):")
            print(table_for(agg["by_max_tokens"], EXPLAIN_TABLE, list(agg["by_max_tokens"])))
    except BudgetExceeded as exc:
        print(f"\nSTOPPED: {exc}", file=sys.stderr)
        return 2
    finally:
        print(f"\nestimated spend: ${spend.total_usd:.4f} {spend.by_model}")
        if written:
            print("results: " + ", ".join(str(p) for p in written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
