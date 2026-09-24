"""Model-choice evals: the same prompts and golden data, several OpenAI models.

    uv run python -m evals.run_models                                  # extraction + router, default models
    uv run python -m evals.run_models --explain-requests evals/results/<ts>_hparams_explain.json
    uv run python -m evals.run_models --models gpt-4o-mini,gpt-4.1 --skip-router --budget-usd 0.5

Three roles, one file (evals/results/<UTC ts>_models.json):
- extraction — 27 golden signals, hybrid_rerank v2, prompt v2, T = 0, one arm per model (metrics of
  evals.run_rag: availability accuracy strict/lenient, ranges, evidence, return GW, validator fixes,
  faithfulness judge, tokens, cost, LLM latency);
- router — the 8 demo queries (scripts/agent_demo.py) through agent.llm.router_llm per model:
  intent accuracy against the expected intent, agreement with the baseline model, latency, cost;
- explain — the ExplainRequests captured by evals.run_hparams (facts computed once, identical for
  every model) replayed at the production cell (T = 0.2, max_tokens 1400): validator violations,
  truncation, length, faithfulness judge, latency, cost.
Plus cost projections from the measured token usage: one batch refresh (every player in the
bootstrap) and one typical user query (router + explain) per model. Prices: run_hparams.PRICES
(list prices, an estimate — never a bill).

Candidate set (17 Sep 2026, `client.models.list()` on this key): gpt-4o-mini (shipped),
gpt-4.1-nano (cheapest), gpt-4.1-mini, gpt-5.4-mini (newest small model; accepts temperature 0 and
used 0 reasoning tokens in a probe), gpt-4.1 (strongest non-reasoning model, T = 0 supported).
Reasoning models (o-series, gpt-5/5.x full) are excluded: no temperature / top_p control — the
very knobs this eval justifies — and reasoning tokens make cost and latency prompt-dependent.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals import RESULTS_DIR
from evals.judge import JUDGE_MODEL, JUDGE_PROMPT_VERSION
from evals.metrics import mean, percentile
from evals.run_hparams import (
    EXPLAIN_PRODUCTION,
    EXPLAIN_TABLE,
    EXTRACTION_K,
    EXTRACTION_MODE,
    EXTRACTION_TABLE,
    JUDGE_BUDGET,
    PRICES,
    BudgetExceeded,
    ExtractionArm,
    Spend,
    aggregate_cost,
    aggregate_explain,
    aggregate_extraction_arm,
    agreement,
    cost_usd,
    load_captured,
    projected_cost,
    run_explain_grid,
    run_extraction_arm,
    save,
    table_for,
)
from evals.run_rag import _fmt, build_retrievers, corpus_stats, git_commit
from evals.schemas import load_signals
from fplcopilot.agent.llm import PROMPT_VERSION as AGENT_PROMPT_VERSION
from fplcopilot.agent.llm import RouterRequest, router_llm
from fplcopilot.config import settings
from fplcopilot.data import FPLClient
from fplcopilot.rag.extract import GWCalendar, next_event_as_of
from fplcopilot.rag.retrieve import retrieval_config

log = logging.getLogger("evals.models")

DEFAULT_MODELS = ("gpt-4o-mini", "gpt-4.1-nano", "gpt-4.1-mini", "gpt-5.4-mini", "gpt-4.1")
BASELINE_MODEL = "gpt-4o-mini"
# scripts/agent_demo.py scenarios with the intent the router must produce (name resolution and
# the clarify / refuse branches happen in code after the router).
ROUTER_QUERIES: tuple[tuple[str, str], ...] = (
    ("Is João Pedro fit for GW5?", "player_status"),
    ("Should I sell Palmer?", "transfer"),
    ("Who should I captain this week?", "captain"),
    ("Plan my transfers for the next 5 gameweeks", "plan"),
    ("What if I take a -4 to bring in Saka and Guéhi?", "what_if"),
    ("Best odds for Arsenal to win?", "betting"),
    ("Should I sell Gabriel?", "transfer"),
    ("Give me a recipe for plov", "off_topic"),
)
ROUTER_TABLE = (
    "intent_accuracy",
    "intent_agreement_with_baseline",
    "mean_prompt_tokens",
    "mean_completion_tokens",
    "cost_usd",
    "cost_usd_per_call",
    "latency_p50_ms",
    "latency_p95_ms",
)


# ---------- router ----------


def run_router(
    models: Sequence[str], *, spend: Spend, has_manager: bool = True, strategy: str = "balanced"
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    bs = FPLClient().bootstrap()
    now = datetime.now(UTC)
    nxt = next_event_as_of(bs, now)
    rows: list[dict[str, Any]] = []
    for model in models:
        for query, expected in ROUTER_QUERIES:
            req = RouterRequest(
                query=query,
                as_of=f"{now:%Y-%m-%d %H:%M}Z",
                gw=nxt.id if nxt else None,
                deadline=f"{nxt.deadline_time:%Y-%m-%d %H:%M}Z" if nxt else None,
                has_manager=has_manager,
                strategy=strategy,
            )
            started = time.perf_counter()
            try:
                out, usage = router_llm(req, model=model)
                row: dict[str, Any] = {
                    "model": model,
                    "query": query,
                    "expected_intent": expected,
                    "intent": out.intent,
                    "intent_ok": out.intent == expected,
                    "player_mentions": out.player_mentions,
                    "horizon": out.horizon,
                    "needs_squad": out.needs_squad,
                    "scenario": out.scenario.model_dump(),
                    "reason": out.reason,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "latency_ms": usage.latency_ms,
                    "error": None,
                }
            except Exception as exc:
                log.warning("router %s failed on %r", model, query, exc_info=True)
                row = {
                    "model": model,
                    "query": query,
                    "expected_intent": expected,
                    "intent": None,
                    "intent_ok": False,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                }
            row["cost_usd"] = cost_usd(model, row["prompt_tokens"], row["completion_tokens"])
            spend.add(model, row["prompt_tokens"], row["completion_tokens"])
            rows.append(row)
            log.info(
                "[router/%s] %-45s -> %-16s (exp %s) %s %d ms | spend $%.3f",
                model,
                query[:45],
                row["intent"],
                expected,
                "ok" if row["intent_ok"] else "MISS",
                row["latency_ms"],
                spend.total_usd,
            )
            spend.check()
    return rows, aggregate_router(rows, models)


def aggregate_router(
    rows: Sequence[dict[str, Any]], models: Sequence[str]
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    base = [r for r in rows if r["model"] == BASELINE_MODEL]
    for model in models:
        mr = [r for r in rows if r["model"] == model]
        ok = [r for r in mr if not r.get("error")]
        costs = aggregate_cost(ok)
        lat = [r["latency_ms"] for r in ok]
        agree = agreement(base, mr, "intent", id_key="query") if base else None
        out[model] = {
            "n": len(mr),
            "n_errors": len(mr) - len(ok),
            "intent_accuracy": mean(r["intent_ok"] for r in mr),
            "misses": [
                {"query": r["query"], "intent": r.get("intent"), "expected": r["expected_intent"]}
                for r in mr
                if not r["intent_ok"]
            ],
            "intent_agreement_with_baseline": agree["rate"] if agree else None,
            "disagreements_with_baseline": agree["differing"] if agree else [],
            "mean_prompt_tokens": mean(r["prompt_tokens"] for r in ok),
            "mean_completion_tokens": mean(r["completion_tokens"] for r in ok),
            "cost_usd": costs["cost_usd"],
            "cost_usd_per_call": (costs["cost_usd"] / len(ok))
            if ok and costs["cost_usd"] is not None
            else None,
            "latency_p50_ms": percentile(lat, 50),
            "latency_p95_ms": percentile(lat, 95),
        }
    return out


# ---------- projections ----------


def cost_projections(
    models: Sequence[str],
    *,
    extraction_agg: dict[str, dict[str, Any]],
    router_agg: dict[str, dict[str, Any]] | None,
    explain_agg: dict[str, dict[str, Any]] | None,
    n_players: int,
) -> dict[str, dict[str, Any]]:
    """Per model: one batch refresh (all players; and with the golden-set abstention share) and one
    typical user query (1 router + 1 explain call), from the measured mean token usage."""
    out: dict[str, dict[str, Any]] = {}
    for model in models:
        ex = extraction_agg.get(model) or {}
        p, c = ex.get("mean_prompt_tokens_per_call"), ex.get("mean_completion_tokens_per_call")
        llm_share = (ex.get("llm_calls") or 0) / ex["n_examples"] if ex.get("n_examples") else None
        proj: dict[str, Any] = {
            "extraction_cost_per_call": cost_usd(model, round(p), round(c))
            if p is not None
            else None,
            "batch_all_players_call_llm": (
                projected_cost(
                    model, mean_prompt_tokens=p, mean_completion_tokens=c, calls=n_players
                )
                if p is not None
                else None
            ),
            "batch_with_golden_abstention_share": (
                projected_cost(
                    model,
                    mean_prompt_tokens=p,
                    mean_completion_tokens=c,
                    calls=n_players * llm_share,
                )
                if p is not None and llm_share is not None
                else None
            ),
            "golden_llm_call_share": llm_share,
            "n_players": n_players,
        }
        r = (router_agg or {}).get(model) or {}
        e = (explain_agg or {}).get(model) or {}
        proj["router_cost_per_call"] = r.get("cost_usd_per_call")
        proj["explain_cost_per_call"] = e.get("cost_usd_per_call")
        proj["query_router_plus_explain"] = (
            (r["cost_usd_per_call"] or 0.0) + (e["cost_usd_per_call"] or 0.0)
            if r.get("cost_usd_per_call") is not None and e.get("cost_usd_per_call") is not None
            else None
        )
        proj["query_with_one_fresh_extraction"] = (
            proj["query_router_plus_explain"] + proj["extraction_cost_per_call"]
            if proj["query_router_plus_explain"] is not None
            and proj["extraction_cost_per_call"] is not None
            else None
        )
        out[model] = proj
    return out


PROJECTION_TABLE = (
    "extraction_cost_per_call",
    "batch_all_players_call_llm",
    "batch_with_golden_abstention_share",
    "router_cost_per_call",
    "explain_cost_per_call",
    "query_router_plus_explain",
    "query_with_one_fresh_extraction",
)


# ---------- CLI ----------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="FPL Copilot: model-choice evals (extraction / router / explain)"
    )
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    ap.add_argument("--budget-usd", type=float, default=1.5)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--judge-budget", type=int, default=JUDGE_BUDGET)
    ap.add_argument("--skip-extraction", action="store_true")
    ap.add_argument("--skip-router", action="store_true")
    ap.add_argument(
        "--explain-requests",
        type=Path,
        help="evals/results/<ts>_hparams_explain.json with captured ExplainRequests; omit to skip explain",
    )
    ap.add_argument("--ids", help="comma-separated golden ids (debug)")
    ap.add_argument("--limit", type=int)
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
    for name in ("evals.models", "evals.hparams"):
        logging.getLogger(name).setLevel(logging.INFO)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if cost_usd(m, 1, 1) is None]
    if unknown:
        print(
            f"WARNING: no list price for {unknown}; their cost will be reported as None",
            file=sys.stderr,
        )
    stamp = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    commit = git_commit()
    spend = Spend(args.budget_usd)
    payload: dict[str, Any] = {
        "suite": "models",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "models": models,
        "baseline_model": BASELINE_MODEL,
        "config": {
            "extraction": {
                "mode": EXTRACTION_MODE,
                "k": EXTRACTION_K,
                "prompt_version": "v2",
                "retrieval_version": "v2",
                "temperature": 0.0,
                "judge_model": JUDGE_MODEL,
                "judge_prompt_version": JUDGE_PROMPT_VERSION,
                "judge_budget_per_model": 0 if args.no_judge else args.judge_budget,
            },
            # active agent prompt version (AGENT_PROMPT_VERSION); the 17 Sep 2026 run used v1
            "router": {
                "queries": [q for q, _ in ROUTER_QUERIES],
                "prompt_version": AGENT_PROMPT_VERSION,
            },
            "explain": {
                "cell": f"T{EXPLAIN_PRODUCTION[0]:g}/max{EXPLAIN_PRODUCTION[1]}",
                "prompt_version": AGENT_PROMPT_VERSION,
                "requests_file": str(args.explain_requests) if args.explain_requests else None,
            },
            "prices_per_1m_usd": {k: list(v) for k, v in PRICES.items()},
        },
    }
    print(f"commit={(commit or '?')[:8]} models={models} budget=${args.budget_usd:.2f}")
    client = FPLClient()
    bs = client.bootstrap()
    n_players = len(bs.elements)
    extraction_agg: dict[str, dict[str, Any]] = {}
    router_agg: dict[str, dict[str, Any]] | None = None
    explain_agg: dict[str, dict[str, Any]] | None = None
    written: Path | None = None
    try:
        if not args.skip_extraction:
            examples = load_signals()
            if args.ids:
                wanted = set(args.ids.split(","))
                examples = [e for e in examples if e.id in wanted]
            if args.limit:
                examples = examples[: args.limit]
            as_of = max(e.as_of for e in examples)
            r_config = retrieval_config("v2", candidates=settings.rag_candidates)
            retriever = build_retrievers([EXTRACTION_MODE], r_config)[EXTRACTION_MODE]
            calendar = GWCalendar.from_fpl(bs, client.fixtures())
            payload["as_of"] = as_of.isoformat()
            payload["corpus_before"] = corpus_stats(as_of)
            print(
                f"[extraction] {len(examples)} players, as_of={as_of:%Y-%m-%dT%H:%MZ}; corpus "
                f"{payload['corpus_before']['articles_at_as_of']} articles / "
                f"{payload['corpus_before']['chunks_at_as_of']} chunks at as_of"
            )
            rows_all: list[dict[str, Any]] = []
            first_rows: list[dict[str, Any]] | None = None
            for model in models:
                arm = ExtractionArm(label=model, model=model, temperature=0.0)
                started = time.perf_counter()
                rows = run_extraction_arm(
                    examples,
                    arm,
                    retriever=retriever,
                    bs=bs,
                    calendar=calendar,
                    spend=spend,
                    judge=not args.no_judge,
                    judge_budget=args.judge_budget,
                )
                extraction_agg[model] = aggregate_extraction_arm(
                    rows, arm, reference_rows=first_rows
                )
                extraction_agg[model]["wall_s"] = round(time.perf_counter() - started, 1)
                if first_rows is None:
                    first_rows = rows
                rows_all.extend(rows)
                a = extraction_agg[model]
                print(
                    f"  {model:<14} acc {_fmt(a['availability_accuracy'])} lenient "
                    f"{_fmt(a['availability_accuracy_lenient'])} fixes {_fmt(a['mean_validation_fixes'])} "
                    f"faith {_fmt(a['faithfulness_mean'])} cost ${a['cost_usd_total']:.4f} "
                    f"llm p50 {_fmt(a['llm_ms_p50'])} ms | running spend ${spend.total_usd:.3f}"
                )
            payload["corpus_after"] = corpus_stats(as_of)
            payload["extraction"] = {"aggregates": extraction_agg, "rows": rows_all}
            # per-row agreement of the availability label with the baseline model
            base_rows = [r for r in rows_all if r.get("model") == BASELINE_MODEL]
            if base_rows:
                for model in models:
                    mr = [r for r in rows_all if r.get("model") == model]
                    extraction_agg[model]["availability_agreement_with_baseline"] = agreement(
                        base_rows, mr, "availability_pred"
                    )
        if not args.skip_router:
            print("[router] 8 demo queries per model")
            router_rows, router_agg = run_router(models, spend=spend)
            payload["router"] = {"aggregates": router_agg, "rows": router_rows}
        if args.explain_requests:
            captured = load_captured(args.explain_requests)
            print(
                f"[explain] {len(captured)} captured requests × {len(models)} models at production cell"
            )
            explain_rows: list[dict[str, Any]] = []
            for model in models:
                explain_rows.extend(
                    run_explain_grid(
                        captured,
                        model=model,
                        cells=[EXPLAIN_PRODUCTION],
                        spend=spend,
                        judge=not args.no_judge,
                    )
                )
            explain_agg = aggregate_explain(explain_rows, "model")
            payload["explain"] = {"aggregates": explain_agg, "rows": explain_rows}
        payload["projections"] = cost_projections(
            models,
            extraction_agg=extraction_agg,
            router_agg=router_agg,
            explain_agg=explain_agg,
            n_players=n_players,
        )
    except BudgetExceeded as exc:
        print(f"\nSTOPPED: {exc}", file=sys.stderr)
        payload["stopped"] = str(exc)
    finally:
        payload["spend"] = spend.as_dict()
        tag = f"_{args.tag.strip('_')}" if args.tag else ""
        written = save(payload, args.out, f"models{tag}", stamp)
        if extraction_agg:
            print("\n== extraction per model (T = 0) ==")
            print(
                table_for(
                    extraction_agg, EXTRACTION_TABLE, [m for m in models if m in extraction_agg]
                )
            )
            for m in models:
                if m in extraction_agg:
                    print(f"  {m}: strict misses {extraction_agg[m]['strict_misses']}")
        if router_agg:
            print("\n== router (8 demo queries) ==")
            print(table_for(router_agg, ROUTER_TABLE, models))
            for m in models:
                if router_agg[m]["misses"]:
                    print(f"  {m}: misses {router_agg[m]['misses']}")
        if explain_agg:
            print("\n== explain per model (production cell) ==")
            print(table_for(explain_agg, EXPLAIN_TABLE, [m for m in models if m in explain_agg]))
        if payload.get("projections"):
            print(f"\n== cost projections (USD, {n_players} players in bootstrap) ==")
            print(table_for(payload["projections"], PROJECTION_TABLE, models))
        print(f"\nestimated spend: ${spend.total_usd:.4f} {spend.by_model}")
        print(f"results: {written}")
    return 0 if not payload.get("stopped") else 2


if __name__ == "__main__":
    sys.exit(main())
