"""RAG evals runner: golden dataset -> per-mode retrieval / signal metrics -> results JSON.

    uv run python -m evals.run_rag --suite all --modes dense,hybrid,hybrid_rerank --k 8 --out evals/results/
    uv run python -m evals.run_rag --suite retrieval --modes dense,hybrid_rerank
    uv run python -m evals.run_rag --suite signals --no-judge --ids sig_caicedo_injured_return
    # A/B #2 arms (docs/EVALS.md): prompt v1|v2, retrieval v1|v2, pre-LLM abstention on|off
    uv run python -m evals.run_rag --suite signals --modes dense,hybrid_rerank --prompt v1 --retrieval v1 --abstain off
    uv run python -m evals.run_rag --suite signals --modes dense,hybrid_rerank --prompt v2 --retrieval v2 --abstain on
    # corpus: default = operational replay (fetched / observed by as_of); A/B #2 corpus; live corpus
    uv run python -m evals.run_rag --suite signals --corpus-cutoff 2026-09-17T10:14:17Z
    uv run python -m evals.run_rag --suite signals --corpus-cutoff none

Design notes
- One Retriever per mode (shared ChunkStore + reranker): the query-embedding cache would
  otherwise make the second mode look faster than the first for the same query.
- Retrieval is scored at ARTICLE level (chunks deduped by article_id) against golden labels.
- Signals are extracted with save=False (player_signals is never written).
- Corpus: published_at <= as_of alone is not a frozen snapshot — the ingest loop back-fills and
  some feeds back-date pubDate (FFScout "FPL notes" 17 Sep: pubDate 01:30Z, public ~17:00Z), so
  --corpus-cutoff also requires news_articles.fetched_at / player_status_snapshots.snapshot_at <= cutoff.
- Faithfulness = LLM-as-judge (gpt-4o-mini, T=0, structured output), capped per mode.
- Results: evals/results/<UTC ts>_<suite>_p<prompt>-r<retrieval>-a<on|off>.json with rows +
  aggregates + git commit + corpus stats + the full arm configuration.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import text

from evals import EVALS_DIR, RESULTS_DIR
from evals.judge import JUDGE_MODEL, JUDGE_PROMPT_VERSION, format_fpl_status, judge_faithfulness
from evals.metrics import (
    accuracy,
    confusion,
    dedupe_by_article,
    hit_at_k,
    macro_f1,
    mean,
    percentile,
    precision_at_k,
    quote_kind,
    recall_at_k,
    reciprocal_rank,
    within_range,
)
from evals.schemas import RetrievalExample, SignalExample, load_retrieval, load_signals
from fplcopilot.config import settings
from fplcopilot.data import Bootstrap, FPLClient
from fplcopilot.db import session_scope
from fplcopilot.rag.extract import PROMPT_VERSION, GWCalendar, PlayerSignal, extract_signal
from fplcopilot.rag.retrieve import (
    MODES,
    RETRIEVAL_VERSIONS,
    ChunkStore,
    Mode,
    RetrievalConfig,
    Retriever,
    make_reranker,
    retrieval_config,
)

log = logging.getLogger("evals")

DEFAULT_MODES: tuple[Mode, ...] = ("dense", "hybrid", "hybrid_rerank")
JUDGE_BUDGET_PER_MODE = 25
WARMUP_QUERY = "premier league team news warm-up query"
# List prices (USD per 1M tokens) used ONLY for the cost estimate stored in results.
PRICE_PER_1M: dict[str, dict[str, float]] = {"gpt-4o-mini": {"prompt": 0.15, "completion": 0.60}}
TOP1_STAGE: dict[str, str] = {
    "dense": "dense_score",
    "bm25": "bm25_score",
    "hybrid": "decayed_score",
    "hybrid_rerank": "rerank_score",
}


# ---------- helpers ----------


def parse_modes(value: str) -> list[Mode]:
    modes = [m.strip() for m in value.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown mode(s) {bad}; choose from {MODES}")
    return modes  # type: ignore[return-value]


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=EVALS_DIR.parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def estimate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float | None:
    price = PRICE_PER_1M.get(model)
    if price is None:
        return None
    return (prompt_tokens * price["prompt"] + completion_tokens * price["completion"]) / 1_000_000


CorpusCutoff = datetime | Literal["as_of"] | None


def parse_corpus_cutoff(value: str) -> CorpusCutoff:
    """--corpus-cutoff: ISO timestamp (UTC if naive), 'as_of' (per example) or 'none'."""
    v = value.strip()
    if v.lower() in ("", "none", "off"):
        return None
    if v.lower() in ("as_of", "asof"):
        return "as_of"
    try:
        ts = datetime.fromisoformat(v)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad --corpus-cutoff {value!r}: {exc}") from exc
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


def cutoff_for(cutoff: CorpusCutoff, as_of: datetime) -> datetime | None:
    return as_of if cutoff == "as_of" else cutoff  # type: ignore[return-value]


def cutoff_tag(cutoff: CorpusCutoff) -> str:
    if cutoff is None:
        return ""
    if cutoff == "as_of":
        return "cutasof"
    return f"cut{cutoff.astimezone(UTC):%m%dT%H%M}"  # type: ignore[union-attr]


def corpus_stats(as_of: datetime, cutoff: datetime | None = None) -> dict[str, Any]:
    """Article/chunk counts (total and visible at as_of), max published_at, per-source counts,
    status snapshots. cutoff — also require fetched_at / snapshot_at <= cutoff (--corpus-cutoff)."""
    art = "published_at <= :as_of" + (" AND fetched_at <= :cutoff" if cutoff else "")
    chunk = "c.published_at <= :as_of" + (" AND a.fetched_at <= :cutoff" if cutoff else "")
    p = {"as_of": as_of, "cutoff": cutoff}
    with session_scope() as s:
        n_art, max_pub = s.execute(
            text("SELECT count(*), max(published_at) FROM news_articles")
        ).one()
        n_art_asof = s.execute(
            text(f"SELECT count(*) FROM news_articles WHERE {art}"), p
        ).scalar_one()
        n_chunks, n_emb = s.execute(
            text("SELECT count(*), count(embedding) FROM news_chunks")
        ).one()
        n_chunks_asof = s.execute(
            text(
                "SELECT count(*) FROM news_chunks c JOIN news_articles a ON a.id = c.article_id "
                f"WHERE {chunk}"
            ),
            p,
        ).scalar_one()
        by_source = dict(
            s.execute(
                text(
                    f"SELECT source, count(*) FROM news_articles WHERE {art} "
                    "GROUP BY source ORDER BY 2 DESC"
                ),
                p,
            ).all()
        )
        n_snap, n_snap_asof = s.execute(
            text(
                "SELECT count(*), count(*) FILTER (WHERE snapshot_at <= :as_of) "
                "FROM player_status_snapshots"
            ),
            p,
        ).one()
    return {
        "articles": int(n_art),
        "articles_at_as_of": int(n_art_asof),
        "chunks": int(n_chunks),
        "chunks_embedded": int(n_emb),
        "chunks_at_as_of": int(n_chunks_asof),
        "max_published_at": max_pub.astimezone(UTC).isoformat() if max_pub else None,
        "articles_by_source_at_as_of": {k: int(v) for k, v in by_source.items()},
        "status_snapshots": int(n_snap),
        "status_snapshots_observed_by_as_of": int(n_snap_asof),
        "corpus_cutoff": cutoff.astimezone(UTC).isoformat() if cutoff else None,
    }


def build_retrievers(
    modes: Sequence[Mode], config: RetrievalConfig | None = None
) -> dict[str, Retriever]:
    store = ChunkStore()
    reranker = make_reranker()
    return {m: Retriever(store=store, reranker=reranker, config=config) for m in modes}


def arm_tag(prompt_version: str, retrieval_version: str, abstain: bool) -> str:
    """Suffix for results files: which A/B #2 arm produced them."""
    return f"p{prompt_version}-r{retrieval_version}-a{'on' if abstain else 'off'}"


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.0f}"
    return str(v)


def render_table(metrics: dict[str, dict[str, Any]], modes: Sequence[str]) -> str:
    """metrics: {metric_name: {mode: value}} -> aligned text table (rows=metrics, cols=modes)."""
    name_w = max([len("metric"), *(len(k) for k in metrics)])
    col_w = max([14, *(len(m) for m in modes)])
    head = f"{'metric':<{name_w}}  " + "  ".join(f"{m:>{col_w}}" for m in modes)
    lines = [head, "-" * len(head)]
    for name, per_mode in metrics.items():
        lines.append(
            f"{name:<{name_w}}  " + "  ".join(f"{_fmt(per_mode.get(m)):>{col_w}}" for m in modes)
        )
    return "\n".join(lines)


# ---------- retrieval suite ----------


def run_retrieval(
    examples: Sequence[RetrievalExample],
    *,
    modes: Sequence[Mode],
    k: int,
    retrievers: dict[str, Retriever],
    warmup: bool = True,
    corpus_cutoff: CorpusCutoff = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for mode in modes:
        retriever = retrievers[mode]
        if warmup and examples:
            # model load / BM25 index build / DB connect happen here, not inside the timed loop;
            # a synthetic query so no golden example gets a cached embedding
            as_of0 = examples[0].as_of
            retriever.search(
                WARMUP_QUERY,
                as_of=as_of0,
                k=k,
                mode=mode,
                corpus_cutoff=cutoff_for(corpus_cutoff, as_of0),
            )
        for ex in examples:
            started = time.perf_counter()
            chunks = retriever.search(
                ex.query,
                player_ids=[ex.player_id] if ex.player_id else None,
                team_ids=[ex.team_id] if ex.team_id else None,
                as_of=ex.as_of,
                k=k,
                mode=mode,
                corpus_cutoff=cutoff_for(corpus_cutoff, ex.as_of),
            )
            latency_ms = (time.perf_counter() - started) * 1000
            arts = dedupe_by_article(chunks)
            rel = ex.relevant_article_ids
            top = chunks[0] if chunks else None
            stage = TOP1_STAGE[mode]
            rows.append(
                {
                    "example_id": ex.id,
                    "mode": mode,
                    "query": ex.query,
                    "player_id": ex.player_id,
                    "team_id": ex.team_id,
                    "as_of": ex.as_of.isoformat(),
                    "has_relevant": ex.has_relevant,
                    "n_relevant": len(rel),
                    "retrieved_article_ids": arts,
                    "retrieved_chunk_ids": [c.chunk_id for c in chunks],
                    "retrieved": [
                        {
                            "article_id": c.article_id,
                            "chunk_id": c.chunk_id,
                            "source": c.source,
                            "published_at": c.published_at.astimezone(UTC).isoformat(),
                            "relevant": c.article_id in rel,
                            "score": c.final_score,
                        }
                        for c in chunks
                    ],
                    "precision_at_k": precision_at_k(arts, rel, k),
                    "recall_at_k": recall_at_k(arts, rel, k),
                    "hit_at_k": hit_at_k(arts, rel, k),
                    "reciprocal_rank": reciprocal_rank(arts, rel),
                    "top1_stage": stage,
                    "top1_score": getattr(top, stage) if top is not None else None,
                    # ordering score (= rerank × date multiplier under retrieval v2); the raw
                    # rerank score above stays comparable with A/B #1
                    "top1_final_score": top.final_score if top is not None else None,
                    "max_stage_score": max(
                        (getattr(c, stage) or 0.0 for c in retriever.last_candidates),
                        default=None,
                    ),
                    "n_distinct_articles": len(arts),
                    "top1_article_id": top.article_id if top is not None else None,
                    "top1_published_at": (
                        top.published_at.astimezone(UTC).isoformat() if top is not None else None
                    ),
                    "latency_ms": latency_ms,
                    "timings": dict(retriever.last_timings),
                }
            )
            log.info(
                "[retrieval/%s] %s: P@%d=%.2f R@%d=%s RR=%s (%.0f ms)",
                mode,
                ex.id,
                k,
                rows[-1]["precision_at_k"],
                k,
                _fmt(rows[-1]["recall_at_k"]),
                _fmt(rows[-1]["reciprocal_rank"]),
                latency_ms,
            )
    return rows, aggregate_retrieval(rows, modes)


def _age_days(row: dict[str, Any]) -> float | None:
    """Age of the top-1 chunk relative to the query's as_of (freshness proxy)."""
    if not row.get("top1_published_at"):
        return None
    as_of = datetime.fromisoformat(row["as_of"])
    published = datetime.fromisoformat(row["top1_published_at"])
    return (as_of - published).total_seconds() / 86_400


def aggregate_retrieval(
    rows: Sequence[dict[str, Any]], modes: Sequence[str]
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for mode in modes:
        mr = [r for r in rows if r["mode"] == mode]
        with_rel = [r for r in mr if r["has_relevant"]]
        no_rel = [r for r in mr if not r["has_relevant"]]
        lat = [r["latency_ms"] for r in mr]
        out[mode] = {
            "n_queries": len(mr),
            "n_with_relevant": len(with_rel),
            "n_no_relevant": len(no_rel),
            "precision_at_k": mean(r["precision_at_k"] for r in with_rel),
            "recall_at_k": mean(r["recall_at_k"] for r in with_rel),
            "hit_at_k": mean(r["hit_at_k"] for r in with_rel),
            "mrr": mean(r["reciprocal_rank"] for r in with_rel),
            "top1_score_relevant_queries": mean(r["top1_score"] for r in with_rel),
            "top1_score_no_relevant_queries": mean(r["top1_score"] for r in no_rel),
            "max_score_relevant_queries": mean(r.get("max_stage_score") for r in with_rel),
            "max_score_no_relevant_queries": mean(r.get("max_stage_score") for r in no_rel),
            "mean_distinct_articles": mean(r.get("n_distinct_articles") for r in mr),
            "mean_top1_age_days": mean(_age_days(r) for r in mr),
            "latency_p50_ms": percentile(lat, 50),
            "latency_p95_ms": percentile(lat, 95),
            "latency_mean_ms": mean(lat),
            "stage_ms_mean": {
                key: mean(r["timings"].get(key) for r in mr)
                for key in ("embed_ms", "dense_ms", "bm25_ms", "fuse_ms", "rerank_ms", "total_ms")
                if any(key in r["timings"] for r in mr)
            },
        }
    return out


# ---------- signals suite ----------


def _signal_row(ex: SignalExample, sig: PlayerSignal, mode: str, timings: dict[str, float]) -> dict:
    exp = ex.expected
    no_cov = "no_coverage" in ex.tags
    row: dict[str, Any] = {
        "example_id": ex.id,
        "mode": mode,
        "player_id": ex.player_id,
        "player_name": sig.player_name,
        "tags": ex.tags,
        "expected_availability": exp.availability,
        "accepted_availability": sorted(exp.accepted),
        "availability_pred": sig.availability,
        "availability_exact": sig.availability == exp.availability,
        "availability_lenient": sig.availability in exp.accepted,
        "start_probability": sig.start_probability,
        "start_probability_in_range": within_range(
            sig.start_probability, exp.start_probability_min, exp.start_probability_max
        ),
        "expected_minutes": sig.expected_minutes,
        "expected_minutes_in_range": within_range(
            sig.expected_minutes, exp.expected_minutes_min, exp.expected_minutes_max
        ),
        "return_gw_expected": exp.return_gw,
        "return_gw_pred": sig.return_gw,
        "return_gw_exact": (sig.return_gw == exp.return_gw) if exp.return_gw is not None else None,
        "return_gw_within_1": (
            (sig.return_gw is not None and abs(sig.return_gw - exp.return_gw) <= 1)
            if exp.return_gw is not None
            else None
        ),
        "must_have_evidence": ex.must_have_evidence,
        "evidence_n": len(sig.evidence),
        "evidence_present_ok": (len(sig.evidence) > 0) if ex.must_have_evidence else None,
        "false_evidence": (len(sig.evidence) > 0) if no_cov else None,
        "confidence": sig.confidence,
        "rotation_risk": sig.rotation_risk,
        "validation_fixes": sig.validation_fixes,
        "fpl_status": sig.fpl_status,
        "fpl_chance_next": sig.fpl_chance_next,
        "summary": sig.summary,
        "evidence": [
            {
                "chunk_id": e.chunk_id,
                "source": e.source,
                "url": e.url,
                "published_at": e.published_at.astimezone(UTC).isoformat(),
                "quote": e.quote,
                "quote_kind": quote_kind(e.quote),  # словарь A/B #4 (evals/metrics.py)
            }
            for e in sig.evidence
        ],
        "form_notes": [
            {
                "kind": n.kind,
                "text": n.text,
                "chunk_id": n.chunk_id,
                "source": n.source,
                "published_at": n.published_at.astimezone(UTC).isoformat(),
                "quote": n.quote,
            }
            for n in sig.form_notes
        ],
        "form_quotes_moved": int(timings.get("form_quotes_moved", 0)),
        "form_note_fixes": int(timings.get("form_note_fixes", 0)),
        "confidence_capped": int(timings.get("confidence_capped", 0)),
        "retrieved_chunk_ids": sig.retrieved_chunk_ids,
        "abstained": sig.abstained,
        "abstain_reason": timings.get("abstain_reason"),
        "abstain_top_score": timings.get("abstain_top_score"),
        "llm_calls": int(timings.get("llm_calls", 1)),
        "return_gw_source": timings.get("return_gw_source"),
        "summary_fixes": list(timings.get("summary_fixes") or []),  # rag/summary_check
        "prompt_tokens": int(timings.get("prompt_tokens", 0)),
        "completion_tokens": int(timings.get("completion_tokens", 0)),
        "cost_usd": estimate_cost(
            int(timings.get("prompt_tokens", 0)),
            int(timings.get("completion_tokens", 0)),
            sig.model,
        ),
        "latency_ms": timings.get("total_ms"),
        "retrieval_ms": timings.get("retrieve_total_ms"),
        "llm_ms": timings.get("llm_ms"),
        "model": sig.model,
        "prompt_version": sig.prompt_version,
        "faithfulness": None,
        "faithfulness_verdict": None,
        "faithfulness_reason": None,
        "judge_prompt_tokens": 0,
        "judge_completion_tokens": 0,
        "error": None,
    }
    return row


def run_signals(
    examples: Sequence[SignalExample],
    *,
    modes: Sequence[Mode],
    k: int,
    retrievers: dict[str, Retriever],
    bs: Bootstrap,
    judge: bool = True,
    judge_budget: int = JUDGE_BUDGET_PER_MODE,
    judge_model: str = JUDGE_MODEL,
    prompt_version: str = PROMPT_VERSION,
    abstain: bool = True,
    calendar: GWCalendar | None = None,
    corpus_cutoff: CorpusCutoff = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for mode in modes:
        judged = 0
        for ex in examples:
            cutoff = cutoff_for(corpus_cutoff, ex.as_of)
            player = bs.player(ex.player_id)
            if ex.player_name.lower() not in {player.web_name.lower(), player.full_name.lower()}:
                raise ValueError(
                    f"{ex.id}: player_id {ex.player_id} is {player.web_name!r}, "
                    f"golden says {ex.player_name!r} — id drift?"
                )
            timings: dict[str, Any] = {}
            try:
                sig = extract_signal(
                    ex.player_id,
                    ex.as_of,
                    mode=mode,
                    k=k,
                    retriever=retrievers[mode],
                    bs=bs,
                    prompt_version=prompt_version,
                    abstain=abstain,
                    calendar=calendar,
                    save=False,
                    timings=timings,
                    corpus_cutoff=cutoff,
                )
            except Exception as exc:  # record and continue: one bad call must not kill the run
                log.exception("[signals/%s] %s failed", mode, ex.id)
                rows.append(
                    {
                        "example_id": ex.id,
                        "mode": mode,
                        "player_id": ex.player_id,
                        "tags": ex.tags,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            row = _signal_row(ex, sig, mode, timings)
            if judge and sig.evidence and judged < judge_budget:
                res = judge_faithfulness(
                    sig.summary,
                    sig.evidence,
                    fpl_status=format_fpl_status(
                        sig.fpl_status, sig.fpl_chance_next, _fpl_news(ex, sig, bs, cutoff)
                    ),
                    model=judge_model,
                )
                judged += 1
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
                "[signals/%s] %s: %s (exp %s) p=%.2f conf=%.2f ev=%d fixes=%d ret=%s/%s%s "
                "faith=%s (%.0f ms)",
                mode,
                ex.id,
                sig.availability,
                ex.expected.availability,
                sig.start_probability,
                sig.confidence,
                len(sig.evidence),
                sig.validation_fixes,
                _fmt(sig.return_gw),
                _fmt(ex.expected.return_gw),
                " ABSTAINED" if sig.abstained else "",
                _fmt(row["faithfulness"]),
                timings.get("total_ms", 0),
            )
    return rows, aggregate_signals(rows, modes, judge_model=judge_model)


def return_gw_breakdown(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Over rows with a labelled return GW: exact / within ±1 / wrong / missing, and the source
    of the predicted GW (fpl_news date, llm_date, llm_gw)."""
    applicable = [r for r in rows if r.get("return_gw_expected") is not None]
    exact = [r for r in applicable if r["return_gw_exact"]]
    within = [r for r in applicable if r["return_gw_within_1"] and not r["return_gw_exact"]]
    missing = [r for r in applicable if r["return_gw_pred"] is None]
    wrong = [
        r for r in applicable if r["return_gw_pred"] is not None and not r["return_gw_within_1"]
    ]
    return {
        "n": len(applicable),
        "exact": len(exact),
        "within_1_not_exact": len(within),
        "wrong": len(wrong),
        "missing": len(missing),
        "by_source": dict(
            Counter(r.get("return_gw_source") or "none" for r in applicable if r["return_gw_pred"])
        ),
        "false_positive_return_gw": sum(
            1
            for r in rows
            if r.get("return_gw_expected") is None
            and r.get("return_gw_pred") is not None
            and not r.get("error")
        ),
    }


def _fpl_news(
    ex: SignalExample, sig: PlayerSignal, bs: Bootstrap, cutoff: datetime | None = None
) -> str:
    """Official FPL news text at as_of (what the extractor saw), for the judge's status block."""
    from fplcopilot.rag.extract import fpl_prior

    try:
        return fpl_prior(bs.player(ex.player_id), ex.as_of, known_before=cutoff).news
    except Exception:  # judge context only; never fail the run for it
        log.warning("fpl_prior failed for %s", ex.id, exc_info=True)
        return ""


def aggregate_signals(
    rows: Sequence[dict[str, Any]], modes: Sequence[str], *, judge_model: str = JUDGE_MODEL
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for mode in modes:
        mr = [r for r in rows if r["mode"] == mode]
        ok = [r for r in mr if not r.get("error")]
        y_true = [r["expected_availability"] for r in ok]
        y_pred = [r["availability_pred"] for r in ok]
        exp_unknown = [r for r in ok if r["expected_availability"] == "unknown"]
        exp_known = [r for r in ok if r["expected_availability"] != "unknown"]
        # A/B #4: fit-строки со статусом FPL a без двусторонних новостей (Haaland, Gabriel,
        # Timber, Raya) — где confidence легко «завысить» цитатами про форму
        fit_a = [
            r
            for r in ok
            if r["expected_availability"] == "fit"
            and r.get("fpl_status") == "a"
            and "conflict" not in r["tags"]
        ]
        kinds = Counter(e.get("quote_kind") for r in ok for e in r.get("evidence") or [])
        n_quotes = sum(kinds.values())
        form_only_rows = [
            r
            for r in ok
            if r.get("evidence")
            and all(e.get("quote_kind") == "form" for e in r["evidence"])
            and float(r["confidence"]) >= 0.5
        ]
        pred_unknown = [r for r in ok if r["availability_pred"] == "unknown"]
        pred_known = [r for r in ok if r["availability_pred"] != "unknown"]
        judged = [r for r in ok if r["faithfulness"] is not None]
        prompt_tok = sum(r["prompt_tokens"] for r in ok)
        compl_tok = sum(r["completion_tokens"] for r in ok)
        j_prompt = sum(r["judge_prompt_tokens"] for r in ok)
        j_compl = sum(r["judge_completion_tokens"] for r in ok)
        lat = [r["latency_ms"] for r in ok if r["latency_ms"] is not None]
        tags = sorted({t for r in ok for t in r["tags"]})
        out[mode] = {
            "n_examples": len(mr),
            "n_errors": len(mr) - len(ok),
            "availability_accuracy": accuracy(y_true, y_pred),
            "availability_accuracy_lenient": mean(r["availability_lenient"] for r in ok),
            "availability_macro_f1": macro_f1(y_true, y_pred),
            "confusion": confusion(y_true, y_pred),
            "start_probability_in_range_rate": mean(r["start_probability_in_range"] for r in ok),
            "expected_minutes_in_range_rate": mean(r["expected_minutes_in_range"] for r in ok),
            "expected_minutes_n_applicable": sum(
                1 for r in ok if r["expected_minutes_in_range"] is not None
            ),
            "evidence_present_rate": mean(r["evidence_present_ok"] for r in ok),
            "evidence_present_n_applicable": sum(
                1 for r in ok if r["evidence_present_ok"] is not None
            ),
            "false_evidence_rate": mean(r["false_evidence"] for r in ok),
            "false_evidence_n_applicable": sum(1 for r in ok if r["false_evidence"] is not None),
            "return_gw_exact_rate": mean(r["return_gw_exact"] for r in ok),
            "return_gw_within_1_rate": mean(r["return_gw_within_1"] for r in ok),
            "return_gw_n_applicable": sum(1 for r in ok if r["return_gw_exact"] is not None),
            "return_gw_breakdown": return_gw_breakdown(ok),
            "abstained_count": sum(1 for r in ok if r.get("abstained")),
            "abstained_ids": [r["example_id"] for r in ok if r.get("abstained")],
            "abstained_on_no_coverage": sum(
                1 for r in ok if r.get("abstained") and "no_coverage" in r["tags"]
            ),
            "abstained_false": sum(
                1 for r in ok if r.get("abstained") and "no_coverage" not in r["tags"]
            ),
            "llm_calls": sum(int(r.get("llm_calls", 1)) for r in ok),
            "mean_confidence_expected_unknown": mean(r["confidence"] for r in exp_unknown),
            "mean_confidence_expected_known": mean(r["confidence"] for r in exp_known),
            "mean_confidence_pred_unknown": mean(r["confidence"] for r in pred_unknown),
            "mean_confidence_pred_known": mean(r["confidence"] for r in pred_known),
            "mean_validation_fixes": mean(r["validation_fixes"] for r in ok),
            "mean_evidence_n": mean(r["evidence_n"] for r in ok),
            "evidence_quotes": n_quotes,
            "evidence_quote_kinds": dict(kinds),
            "evidence_form_share": (kinds.get("form", 0) / n_quotes) if n_quotes else None,
            "evidence_non_availability_share": (
                (n_quotes - kinds.get("availability", 0)) / n_quotes if n_quotes else None
            ),
            "rows_confident_on_form_only": [r["example_id"] for r in form_only_rows],
            "mean_confidence_fit_status_a": mean(float(r["confidence"]) for r in fit_a),
            "confidence_fit_status_a": {r["example_id"]: r["confidence"] for r in fit_a},
            "form_notes_total": sum(len(r.get("form_notes") or []) for r in ok),
            "form_notes_rows": sum(1 for r in ok if r.get("form_notes")),
            "form_notes_by_kind": dict(
                Counter(n["kind"] for r in ok for n in r.get("form_notes") or [])
            ),
            "form_quotes_moved": sum(int(r.get("form_quotes_moved") or 0) for r in ok),
            "form_note_fixes": sum(int(r.get("form_note_fixes") or 0) for r in ok),
            "confidence_capped": sum(int(r.get("confidence_capped") or 0) for r in ok),
            "faithfulness_mean": mean(r["faithfulness"] for r in judged),
            "faithfulness_n_judged": len(judged),
            "faithfulness_verdicts": dict(Counter(r["faithfulness_verdict"] for r in judged)),
            "prompt_tokens": prompt_tok,
            "completion_tokens": compl_tok,
            "cost_usd_signals": estimate_cost(prompt_tok, compl_tok, settings.rag_llm_model),
            "judge_prompt_tokens": j_prompt,
            "judge_completion_tokens": j_compl,
            "cost_usd_judge": estimate_cost(j_prompt, j_compl, judge_model),
            "cost_usd_total": (
                (estimate_cost(prompt_tok, compl_tok, settings.rag_llm_model) or 0.0)
                + (estimate_cost(j_prompt, j_compl, judge_model) or 0.0)
            ),
            "latency_p50_ms": percentile(lat, 50),
            "latency_p95_ms": percentile(lat, 95),
            "retrieval_ms_mean": mean(r["retrieval_ms"] for r in ok),
            "llm_ms_mean": mean(r["llm_ms"] for r in ok),
            "by_tag": {
                t: {
                    "n": sum(1 for r in ok if t in r["tags"]),
                    "availability_accuracy": mean(
                        r["availability_exact"] for r in ok if t in r["tags"]
                    ),
                    "availability_accuracy_lenient": mean(
                        r["availability_lenient"] for r in ok if t in r["tags"]
                    ),
                    "evidence_n_mean": mean(r["evidence_n"] for r in ok if t in r["tags"]),
                }
                for t in tags
            },
        }
    return out


# ---------- persistence / reporting ----------


def golden_summary(signals: Sequence[SignalExample], retrieval: Sequence[RetrievalExample]) -> dict:
    return {
        "signals": {
            "n": len(signals),
            "by_expected_availability": dict(
                Counter(s.expected.availability for s in signals).most_common()
            ),
            "by_tag": dict(Counter(t for s in signals for t in s.tags).most_common()),
            "must_have_evidence": sum(1 for s in signals if s.must_have_evidence),
        },
        "retrieval": {
            "n": len(retrieval),
            "with_relevant": sum(1 for r in retrieval if r.has_relevant),
            "no_relevant": sum(1 for r in retrieval if not r.has_relevant),
            "player_filtered": sum(1 for r in retrieval if r.player_id),
            "team_filtered": sum(1 for r in retrieval if r.team_id),
            "relevant_articles_total": sum(len(r.relevant_article_ids) for r in retrieval),
        },
    }


def save_results(payload: dict[str, Any], out_dir: Path, suite: str, stamp: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stamp}_{suite}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    return path


RETRIEVAL_TABLE = (
    "precision_at_k",
    "recall_at_k",
    "hit_at_k",
    "mrr",
    "top1_score_relevant_queries",
    "top1_score_no_relevant_queries",
    "mean_top1_age_days",
    "latency_p50_ms",
    "latency_p95_ms",
)
SIGNALS_TABLE = (
    "availability_accuracy",
    "availability_accuracy_lenient",
    "availability_macro_f1",
    "start_probability_in_range_rate",
    "expected_minutes_in_range_rate",
    "evidence_present_rate",
    "false_evidence_rate",
    "return_gw_exact_rate",
    "return_gw_within_1_rate",
    "mean_confidence_expected_unknown",
    "mean_confidence_expected_known",
    "mean_validation_fixes",
    "evidence_quotes",
    "evidence_form_share",
    "evidence_non_availability_share",
    "mean_confidence_fit_status_a",
    "form_notes_total",
    "form_notes_rows",
    "abstained_count",
    "abstained_on_no_coverage",
    "abstained_false",
    "llm_calls",
    "faithfulness_mean",
    "faithfulness_n_judged",
    "prompt_tokens",
    "completion_tokens",
    "cost_usd_signals",
    "cost_usd_judge",
    "latency_p50_ms",
    "latency_p95_ms",
)


def table_for(
    aggregates: dict[str, dict[str, Any]], keys: Sequence[str], modes: Sequence[str]
) -> str:
    return render_table({key: {m: aggregates[m].get(key) for m in modes} for key in keys}, modes)


# ---------- CLI ----------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FPL Copilot: RAG evals on the golden dataset")
    ap.add_argument("--suite", choices=("signals", "retrieval", "all"), default="all")
    ap.add_argument("--modes", type=parse_modes, default=list(DEFAULT_MODES))
    ap.add_argument("-k", "--k", type=int, default=8)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    ap.add_argument("--no-judge", action="store_true", help="skip the faithfulness judge")
    ap.add_argument("--judge-budget", type=int, default=JUDGE_BUDGET_PER_MODE)
    ap.add_argument("--ids", help="comma-separated example ids (debug subset)")
    ap.add_argument("--limit", type=int, help="first N examples of each suite (debug)")
    ap.add_argument(
        "--prompt",
        choices=("v1", "v2", "v3", "v4"),
        default=PROMPT_VERSION,
        help="signal-extraction prompt version (v1 = A/B #1 pipeline; default from settings)",
    )
    ap.add_argument(
        "--retrieval",
        choices=tuple(RETRIEVAL_VERSIONS),
        default="v2",
        help="retrieval knobs: v1 = no article cap, date-blind rerank; v2 = cap 2 + date-aware",
    )
    ap.add_argument(
        "--abstain",
        choices=("on", "off"),
        default="on",
        help="pre-LLM abstention for players without player-specific candidates",
    )
    ap.add_argument(
        "--corpus-cutoff",
        type=parse_corpus_cutoff,
        default="as_of",
        help="only articles fetched and FPL status snapshots taken by this time (ISO UTC, e.g. "
        "2026-09-17T10:14:17Z = the A/B #2 corpus). Default 'as_of' = operational replay: what "
        "the system had collected at each example's as_of (reproducible, leak-free); 'none' = "
        "everything published by as_of in today's corpus (research reconstruction, drifts)",
    )
    ap.add_argument("--tag", default="", help="extra suffix for the results file name")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("httpx", "httpx2", "openai", "fplcopilot.rag.extract"):
        logging.getLogger(name).setLevel(logging.WARNING)

    modes: list[Mode] = args.modes
    prompt_version: str = args.prompt
    abstain = args.abstain == "on"
    r_config = retrieval_config(args.retrieval, candidates=settings.rag_candidates)
    corpus_cutoff: CorpusCutoff = args.corpus_cutoff
    tag = arm_tag(prompt_version, r_config.version, abstain)
    if corpus_cutoff is not None:
        tag += f"-{cutoff_tag(corpus_cutoff)}"
    if args.tag:
        tag += f"-{args.tag.strip('-')}"
    stamp = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    commit = git_commit()
    signals = load_signals()
    retrieval = load_retrieval()
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",")}
        signals = [s for s in signals if s.id in wanted]
        retrieval = [r for r in retrieval if r.id in wanted]
    if args.limit:
        signals, retrieval = signals[: args.limit], retrieval[: args.limit]
    as_of_values = {s.as_of for s in signals} | {r.as_of for r in retrieval}
    as_of = max(as_of_values) if as_of_values else datetime.now(UTC)

    retrievers = build_retrievers(modes, r_config)
    reranker_name = next(iter(retrievers.values())).reranker.name
    base = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "k": args.k,
        "modes": modes,
        "as_of": as_of.isoformat(),
        "arm": tag,
        "config": {
            "llm_model": settings.rag_llm_model,
            "prompt_version": prompt_version,
            "retrieval_version": r_config.version,
            "retrieval": {
                "per_article_cap": r_config.per_article_cap,
                "date_aware_rerank": r_config.date_aware_rerank,
                "rerank_decay_floor": r_config.rerank_decay_floor,
                "candidates": r_config.candidates,
            },
            "abstain": abstain,
            "abstain_score": settings.rag_abstain_score,
            "abstain_dense_score": settings.rag_abstain_dense_score,
            "summary_check": settings.rag_summary_check,
            "corpus_cutoff": (
                corpus_cutoff.isoformat() if isinstance(corpus_cutoff, datetime) else corpus_cutoff
            ),
            "embedding_model": settings.rag_embedding_model,
            "reranker": reranker_name,
            "half_life_days": settings.rag_half_life_days,
            "candidates": r_config.candidates,
            "judge_model": JUDGE_MODEL,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "judge_budget_per_mode": 0 if args.no_judge else args.judge_budget,
            "price_per_1m_usd": PRICE_PER_1M,
        },
        "corpus": corpus_stats(as_of, cutoff_for(corpus_cutoff, as_of)),
        "golden": golden_summary(signals, retrieval),
    }
    print(
        f"as_of={as_of:%Y-%m-%dT%H:%MZ} modes={','.join(modes)} k={args.k} arm={tag} "
        f"reranker={reranker_name} commit={(commit or '?')[:8]}"
    )
    print(
        f"corpus: {base['corpus']['articles_at_as_of']} articles / "
        f"{base['corpus']['chunks_at_as_of']} chunks visible at as_of "
        f"(max published_at {base['corpus']['max_published_at']})\n"
    )

    written: list[Path] = []
    if args.suite in ("retrieval", "all") and retrieval:
        rows, agg = run_retrieval(
            retrieval, modes=modes, k=args.k, retrievers=retrievers, corpus_cutoff=corpus_cutoff
        )
        path = save_results(
            {**base, "suite": "retrieval", "aggregates": agg, "rows": rows},
            args.out,
            f"retrieval_{tag}",
            stamp,
        )
        written.append(path)
        print(
            f"\n== retrieval ({len(retrieval)} queries, article-level, k={args.k}, "
            f"retrieval {r_config.version}) =="
        )
        print(table_for(agg, RETRIEVAL_TABLE, modes))
    if args.suite in ("signals", "all") and signals:
        client = FPLClient()
        bs = client.bootstrap()
        calendar = GWCalendar.from_fpl(bs, client.fixtures()) if prompt_version != "v1" else None
        rows, agg = run_signals(
            signals,
            modes=modes,
            k=args.k,
            retrievers=retrievers,
            bs=bs,
            judge=not args.no_judge,
            judge_budget=args.judge_budget,
            prompt_version=prompt_version,
            abstain=abstain,
            calendar=calendar,
            corpus_cutoff=corpus_cutoff,
        )
        path = save_results(
            {**base, "suite": "signals", "aggregates": agg, "rows": rows},
            args.out,
            f"signals_{tag}",
            stamp,
        )
        written.append(path)
        print(f"\n== signals ({len(signals)} players, arm {tag}) ==")
        print(table_for(agg, SIGNALS_TABLE, modes))
        for mode in modes:
            print(f"  return_gw breakdown [{mode}]: {agg[mode]['return_gw_breakdown']}")
        errors = sum(a["n_errors"] for a in agg.values())
        if errors:
            print(f"\nWARNING: {errors} extraction error(s) — see rows[].error")
    print("\nresults: " + ", ".join(_display_path(p) for p in written))
    return 0


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(EVALS_DIR.parent))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    sys.exit(main())
