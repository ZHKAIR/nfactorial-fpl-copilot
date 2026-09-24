"""Strategy-KB evals: golden questions -> retrieval hit@k + cited answers -> results JSON.

    uv run python -m evals.run_kb                                   # hybrid_rerank, answers on
    uv run python -m evals.run_kb --modes dense,hybrid,hybrid_rerank  # retrieval hit@k per mode (no extra LLM calls)
    uv run python -m evals.run_kb --no-answers --ids kb_ft_bank
    uv run python -m evals.run_kb --k 6 --answer-mode hybrid_rerank --out evals/results/

Per question (evals/golden/strategy.jsonl):
- retrieval: hit@k = some retrieved chunk comes from one of `expected_source_domains` (None if empty);
- answer (answer_strategy_question, gpt-4o-mini T=0): keyword recall over `expected_keywords`
  (each entry may list alternatives with `|`), citation validity (raw citations from the model vs
  the ones that survived the deterministic check; quotes re-verified here as substrings of the
  retrieved chunk text), `must_cite`, "not covered" behaviour vs `expect_not_covered` (null = either
  is fine), `forbidden_patterns` (regexes that must NOT appear, e.g. an invented points number),
  tokens / cost / latency.
Results: evals/results/<UTC ts>_kb.json with rows, aggregates, corpus stats, git commit, config.
Read-only with respect to product tables (search + LLM only).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from evals import GOLDEN_DIR, RESULTS_DIR
from evals.metrics import mean, percentile
from evals.run_rag import _fmt, git_commit, render_table
from fplcopilot.config import settings
from fplcopilot.rag.kb.answer import (
    DEFAULT_MODEL,
    NOT_COVERED,
    PROMPT_VERSION,
    answer_strategy_question,
)
from fplcopilot.rag.kb.ingest import corpus_stats
from fplcopilot.rag.kb.retrieve import KB_MODES, KBMode, KBRetriever, KBStore
from fplcopilot.rag.retrieve import make_reranker

log = logging.getLogger("evals.kb")

WARMUP_QUERY = "fantasy premier league strategy warm-up query"
DEFAULT_K = 6


# ---------- golden ----------


class StrategyExample(BaseModel):
    id: str = Field(pattern=r"^kb_[a-z0-9_]+$")
    question: str = Field(min_length=8)
    filter_tags: list[str] | None = None
    expected_keywords: list[str] = Field(default_factory=list)
    expected_source_domains: list[str] = Field(default_factory=list)
    must_cite: bool = False
    expect_not_covered: bool | None = None  # None = either behaviour is acceptable
    forbidden_patterns: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=40)


def load_strategy(path: Path | None = None) -> list[StrategyExample]:
    path = path or GOLDEN_DIR / "strategy.jsonl"
    rows: list[StrategyExample] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                try:
                    rows.append(StrategyExample.model_validate(json.loads(line)))
                except ValueError as exc:
                    raise ValueError(f"{path.name}:{line_no}: {exc}") from exc
    ids = [r.id for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path.name}: duplicate ids")
    return rows


# ---------- pure metrics ----------


def domain_of(url: str) -> str:
    host = urlsplit(url).netloc.lower()
    return host.removeprefix("www.")


def domain_hit(urls: Sequence[str], domains: Sequence[str]) -> float | None:
    """1.0 if any retrieved URL belongs to one of the expected domains; None when none expected."""
    if not domains:
        return None
    wanted = {d.lower().removeprefix("www.") for d in domains}
    return 1.0 if any(domain_of(u) in wanted for u in urls) else 0.0


def keyword_recall(answer: str, keywords: Sequence[str]) -> float | None:
    """Share of keyword groups present in the answer (case-insensitive; `a|b` = alternatives)."""
    if not keywords:
        return None
    low = answer.lower()
    hits = sum(
        1 for group in keywords if any(alt.strip().lower() in low for alt in group.split("|"))
    )
    return hits / len(keywords)


def forbidden_hits(answer: str, patterns: Sequence[str]) -> list[str]:
    return [p for p in patterns if re.search(p, answer, flags=re.IGNORECASE)]


def quotes_verbatim(
    citations: Sequence[dict[str, Any]], retrieved: Sequence[dict[str, Any]]
) -> bool:
    """Independent re-check: every citation quote is a substring of its chunk text (whitespace-normalised)."""
    by_id = {r["chunk_id"]: " ".join(r["text"].split()).lower() for r in retrieved}
    for c in citations:
        text = by_id.get(c["chunk_id"])
        if text is None or " ".join(c["quote"].split()).lower() not in text:
            return False
    return True


def parse_modes(value: str) -> list[KBMode]:
    modes = [m.strip() for m in value.split(",") if m.strip()]
    bad = [m for m in modes if m not in KB_MODES]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown mode(s) {bad}; choose from {KB_MODES}")
    return modes  # type: ignore[return-value]


# ---------- runs ----------


def run_retrieval(
    examples: Sequence[StrategyExample],
    *,
    modes: Sequence[KBMode],
    k: int,
    retrievers: dict[str, KBRetriever],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for mode in modes:
        retriever = retrievers[mode]
        retriever.search(WARMUP_QUERY, k=k, mode=mode)  # model load / BM25 build outside timing
        for ex in examples:
            started = time.perf_counter()
            chunks = retriever.search(ex.question, tags=ex.filter_tags, k=k, mode=mode)
            latency = (time.perf_counter() - started) * 1000
            rows.append(
                {
                    "example_id": ex.id,
                    "mode": mode,
                    "hit_at_k": domain_hit([c.url for c in chunks], ex.expected_source_domains),
                    "n_distinct_docs": len({c.article_id for c in chunks}),
                    "top1_score": chunks[0].final_score if chunks else None,
                    "top1_source": chunks[0].source if chunks else None,
                    "retrieved": [
                        {
                            "doc_id": c.article_id,
                            "chunk_id": c.chunk_id,
                            "source": c.source,
                            "domain": domain_of(c.url),
                            "title": c.title,
                            "tags": list(c.tags),
                            "score": c.final_score,
                        }
                        for c in chunks
                    ],
                    "latency_ms": latency,
                    "timings": dict(retriever.last_timings),
                }
            )
            log.info(
                "[retrieval/%s] %s: hit@%d=%s top1=%s (%.0f ms)",
                mode,
                ex.id,
                k,
                _fmt(rows[-1]["hit_at_k"]),
                rows[-1]["top1_source"],
                latency,
            )
    agg: dict[str, dict[str, Any]] = {}
    for mode in modes:
        mr = [r for r in rows if r["mode"] == mode]
        lat = [r["latency_ms"] for r in mr]
        agg[mode] = {
            "n_queries": len(mr),
            "n_with_expected_domain": sum(1 for r in mr if r["hit_at_k"] is not None),
            "hit_at_k": mean(r["hit_at_k"] for r in mr),
            "mean_distinct_docs": mean(r["n_distinct_docs"] for r in mr),
            "top1_score_mean": mean(r["top1_score"] for r in mr),
            "latency_p50_ms": percentile(lat, 50),
            "latency_p95_ms": percentile(lat, 95),
            "stage_ms_mean": {
                key: mean(r["timings"].get(key) for r in mr)
                for key in ("embed_ms", "dense_ms", "bm25_ms", "fuse_ms", "rerank_ms", "total_ms")
                if any(key in r["timings"] for r in mr)
            },
        }
    return rows, agg


def run_answers(
    examples: Sequence[StrategyExample],
    *,
    k: int,
    mode: KBMode,
    retriever: KBRetriever,
    model: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ex in examples:
        try:
            res = answer_strategy_question(
                ex.question, k=k, tags=ex.filter_tags, mode=mode, retriever=retriever, model=model
            )
        except Exception as exc:  # one failing call must not kill the run
            log.exception("[answers] %s failed", ex.id)
            rows.append({"example_id": ex.id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        raw_n = len(res["raw_citations"])
        valid_n = len(res["citations"])
        forbidden = forbidden_hits(res["answer"], ex.forbidden_patterns)
        row = {
            "example_id": ex.id,
            "question": ex.question,
            "answer": res["answer"],
            "covered": res["covered"],
            "expect_not_covered": ex.expect_not_covered,
            "not_covered_ok": (
                None
                if ex.expect_not_covered is None
                else (res["covered"] is not ex.expect_not_covered)
                and (res["answer"] == NOT_COVERED) is ex.expect_not_covered
            ),
            "keyword_recall": keyword_recall(res["answer"], ex.expected_keywords),
            "keywords_missing": [
                g
                for g in ex.expected_keywords
                if not any(a.strip().lower() in res["answer"].lower() for a in g.split("|"))
            ],
            "must_cite": ex.must_cite,
            "must_cite_ok": (valid_n > 0) if ex.must_cite else None,
            "citations_raw": raw_n,
            "citations_valid": valid_n,
            "citation_validity": (valid_n / raw_n) if raw_n else None,
            "refs_removed": res["validation"]["removed_refs"],
            "all_refs_resolved": not res["validation"]["removed_refs"],
            "quotes_verbatim": quotes_verbatim(res["citations"], res["retrieved"]),
            "forced_not_covered": res["validation"]["forced_not_covered"],
            "validation_fixes": res["validation"]["fixes"],
            "forbidden_hits": forbidden,
            "forbidden_ok": (not forbidden) if ex.forbidden_patterns else None,
            "cites_rules_source": any("rules" in c["tags"] for c in res["citations"]),
            "citations": res["citations"],
            "retrieval_hit_at_k": domain_hit(
                [r["url"] for r in res["retrieved"]], ex.expected_source_domains
            ),
            "prompt_tokens": res["usage"]["prompt_tokens"],
            "completion_tokens": res["usage"]["completion_tokens"],
            "cost_usd": res["cost_usd"],
            "latency_ms": res["timings"]["total_ms"],
            "retrieve_ms": res["timings"]["retrieve_ms"],
            "llm_ms": res["timings"]["llm_ms"],
            "model": res["model"],
            "prompt_version": res["prompt_version"],
            "error": None,
        }
        rows.append(row)
        log.info(
            "[answers] %s: covered=%s kw=%s cites=%d/%d refs_ok=%s forbidden=%d (%.0f ms, $%.5f)",
            ex.id,
            row["covered"],
            _fmt(row["keyword_recall"]),
            valid_n,
            raw_n,
            row["all_refs_resolved"],
            len(forbidden),
            row["latency_ms"],
            row["cost_usd"] or 0.0,
        )
    ok = [r for r in rows if not r.get("error")]
    lat = [r["latency_ms"] for r in ok]
    prompt_tok = sum(r["prompt_tokens"] for r in ok)
    compl_tok = sum(r["completion_tokens"] for r in ok)
    agg = {
        "n_examples": len(rows),
        "n_errors": len(rows) - len(ok),
        "keyword_recall": mean(r["keyword_recall"] for r in ok),
        "keyword_recall_n_applicable": sum(1 for r in ok if r["keyword_recall"] is not None),
        "must_cite_ok_rate": mean(r["must_cite_ok"] for r in ok),
        "citation_validity": mean(r["citation_validity"] for r in ok),
        "citations_raw_total": sum(r["citations_raw"] for r in ok),
        "citations_valid_total": sum(r["citations_valid"] for r in ok),
        "all_refs_resolved_rate": mean(r["all_refs_resolved"] for r in ok),
        "quotes_verbatim_rate": mean(r["quotes_verbatim"] for r in ok),
        "not_covered_accuracy": mean(r["not_covered_ok"] for r in ok),
        "not_covered_n_applicable": sum(1 for r in ok if r["not_covered_ok"] is not None),
        "covered_count": sum(1 for r in ok if r["covered"]),
        "forced_not_covered_count": sum(1 for r in ok if r["forced_not_covered"]),
        "forbidden_ok_rate": mean(r["forbidden_ok"] for r in ok),
        "cites_rules_source_count": sum(1 for r in ok if r["cites_rules_source"]),
        "mean_validation_fixes": mean(r["validation_fixes"] for r in ok),
        "retrieval_hit_at_k": mean(r["retrieval_hit_at_k"] for r in ok),
        "prompt_tokens": prompt_tok,
        "completion_tokens": compl_tok,
        "cost_usd_total": sum(r["cost_usd"] or 0.0 for r in ok),
        "latency_p50_ms": percentile(lat, 50),
        "latency_p95_ms": percentile(lat, 95),
        "retrieve_ms_mean": mean(r["retrieve_ms"] for r in ok),
        "llm_ms_mean": mean(r["llm_ms"] for r in ok),
    }
    return rows, agg


RETRIEVAL_TABLE = (
    "hit_at_k",
    "mean_distinct_docs",
    "top1_score_mean",
    "latency_p50_ms",
    "latency_p95_ms",
)
ANSWERS_TABLE = (
    "keyword_recall",
    "must_cite_ok_rate",
    "citation_validity",
    "citations_raw_total",
    "citations_valid_total",
    "all_refs_resolved_rate",
    "quotes_verbatim_rate",
    "not_covered_accuracy",
    "covered_count",
    "forced_not_covered_count",
    "forbidden_ok_rate",
    "cites_rules_source_count",
    "retrieval_hit_at_k",
    "mean_validation_fixes",
    "prompt_tokens",
    "completion_tokens",
    "cost_usd_total",
    "latency_p50_ms",
    "latency_p95_ms",
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FPL Copilot: strategy-KB evals")
    ap.add_argument("--modes", type=parse_modes, default=["hybrid_rerank"])
    ap.add_argument("--answer-mode", default="hybrid_rerank", choices=KB_MODES)
    ap.add_argument("-k", "--k", type=int, default=DEFAULT_K)
    ap.add_argument("--no-answers", action="store_true", help="retrieval only (no LLM calls)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ids", help="comma-separated example ids")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("httpx", "httpx2", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)

    examples = load_strategy()
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",")}
        examples = [e for e in examples if e.id in wanted]
    modes: list[KBMode] = list(args.modes)
    if args.answer_mode not in modes:
        modes.append(args.answer_mode)

    store, reranker = KBStore(), make_reranker()
    retrievers = {m: KBRetriever(store=store, reranker=reranker) for m in modes}
    stamp = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    commit = git_commit()
    corpus = corpus_stats()
    print(
        f"corpus: {corpus['docs']} docs / {corpus['chunks']} chunks ({corpus['tokens']} tokens); "
        f"modes={','.join(modes)} k={args.k} answer_mode={args.answer_mode} model={args.model} "
        f"commit={(commit or '?')[:8]}\n"
    )

    ret_rows, ret_agg = run_retrieval(examples, modes=modes, k=args.k, retrievers=retrievers)
    print(f"== retrieval hit@{args.k} by expected source domain ({len(examples)} questions) ==")
    print(
        render_table(
            {key: {m: ret_agg[m].get(key) for m in modes} for key in RETRIEVAL_TABLE}, modes
        )
    )

    ans_rows: list[dict[str, Any]] = []
    ans_agg: dict[str, Any] = {}
    if not args.no_answers:
        ans_rows, ans_agg = run_answers(
            examples,
            k=args.k,
            mode=args.answer_mode,
            retriever=retrievers[args.answer_mode],
            model=args.model,
        )
        print(f"\n== answers ({args.answer_mode}, {args.model}, prompt {PROMPT_VERSION}) ==")
        print(render_table({key: {"value": ans_agg.get(key)} for key in ANSWERS_TABLE}, ["value"]))
        for r in ans_rows:
            if r.get("error"):
                print(f"  ERROR {r['example_id']}: {r['error']}")

    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "suite": "kb",
        "k": args.k,
        "modes": modes,
        "answer_mode": args.answer_mode,
        "config": {
            "llm_model": args.model,
            "prompt_version": PROMPT_VERSION,
            "embedding_model": settings.rag_embedding_model,
            "reranker": reranker.name,
            "per_doc_cap": retrievers[args.answer_mode].per_doc_cap,
            "candidates": retrievers[args.answer_mode].candidates,
            "time_decay": None,
        },
        "corpus": corpus,
        "golden": {"n": len(examples), "ids": [e.id for e in examples]},
        "retrieval": {"aggregates": ret_agg, "rows": ret_rows},
        "answers": {"aggregates": ans_agg, "rows": ans_rows},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{stamp}_kb.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    print(f"\nresults: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
