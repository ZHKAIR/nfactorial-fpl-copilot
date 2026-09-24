"""CLI стратегической базы знаний.

uv run python -m fplcopilot.rag.kb --sources                      # реестр: что включено, теги
uv run python -m fplcopilot.rag.kb --probe [--offline]            # проверить источники (без БД)
uv run python -m fplcopilot.rag.kb --ingest [--limit N] [--offline] [--dry-run]
uv run python -m fplcopilot.rag.kb --stats
uv run python -m fplcopilot.rag.kb --query "when should I use my wildcard?" [--tags chips] [-k 6] [--mode hybrid_rerank]
uv run python -m fplcopilot.rag.kb --answer "how many free transfers can I bank?" [--tags transfers] [-k 6]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys


def _parse_tags(value: str | None) -> list[str] | None:
    if not value:
        return None
    tags = [t.strip() for t in value.split(",") if t.strip()]
    return tags or None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FPL Copilot: стратегическая база знаний (RAG #2)")
    ap.add_argument("--sources", action="store_true", help="показать реестр источников")
    ap.add_argument("--probe", action="store_true", help="загрузить и проверить источники")
    ap.add_argument("--ingest", action="store_true", help="загрузить, чанковать, эмбеддить в БД")
    ap.add_argument("--limit", type=int, default=None, help="максимум источников за запуск")
    ap.add_argument("--offline", action="store_true", help="только из кэша .cache/kb/http")
    ap.add_argument("--dry-run", action="store_true", help="ingest без эмбеддингов и записи в БД")
    ap.add_argument(
        "--prune", action="store_true", help="ingest: удалить документы, выключенные в реестре"
    )
    ap.add_argument("--stats", action="store_true", help="статистика kb_docs / kb_chunks")
    ap.add_argument("--query", help="поиск по чанкам KB")
    ap.add_argument("--answer", help="цитируемый ответ LLM на стратегический вопрос")
    ap.add_argument("--tags", help="фильтр тегов через запятую (chips,transfers)")
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--mode", default="hybrid_rerank", help="dense | bm25 | hybrid | hybrid_rerank")
    ap.add_argument("--json", action="store_true", help="ответ --answer в JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("httpx", "httpx2", "openai", "trafilatura"):
        logging.getLogger(name).setLevel(logging.WARNING)

    did_something = False
    if args.sources:
        from fplcopilot.rag.kb.registry import load_registry, summarize

        sources = load_registry()
        s = summarize(sources)
        print(f"источников: {s.total}, включено: {s.enabled}, по типу загрузки: {s.by_fetch}")
        print("по тегам:", dict(sorted(s.by_tag.items(), key=lambda kv: -kv[1])))
        for src in sources:
            flag = "on " if src.enabled else "off"
            print(f"  [{flag}] {src.name:<32} {src.fetch:<11} {','.join(src.tags):<32} {src.url}")
            if src.note:
                print(f"        note: {src.note}")
        did_something = True

    if args.probe:
        from fplcopilot.rag.kb.fetch import KBFetcher
        from fplcopilot.rag.kb.ingest import format_probe_table, probe
        from fplcopilot.rag.kb.registry import enabled_sources

        sources = enabled_sources()
        if args.limit:
            sources = sources[: args.limit]
        fetcher = KBFetcher(offline=args.offline)
        results = probe(sources, fetcher=fetcher)
        print(format_probe_table(results))
        print(f"network={fetcher.network_calls} cache_hits={fetcher.cache_hits}")
        did_something = True

    if args.ingest:
        from fplcopilot.rag.kb.fetch import KBFetcher
        from fplcopilot.rag.kb.ingest import ingest

        st = ingest(
            limit=args.limit,
            fetcher=KBFetcher(offline=args.offline),
            dry_run=args.dry_run,
            prune=args.prune,
        )
        print(f"OK: {st}")
        for r in st.results:
            if r.action in ("failed", "skipped"):
                print(f"  {r.action:<8} {r.name}: {r.error}")
        for url in st.stale_docs:
            print(f"  stale    {url}{' (pruned)' if st.pruned else ' — --prune удалит'}")
        did_something = True

    if args.stats:
        from fplcopilot.rag.kb.ingest import build_stats

        print(build_stats())
        did_something = True

    if args.query:
        from fplcopilot.rag.kb.retrieve import KBRetriever, format_results

        retriever = KBRetriever()
        chunks = retriever.search(args.query, tags=_parse_tags(args.tags), k=args.k, mode=args.mode)
        print(
            f"query: {args.query}\ntags: {_parse_tags(args.tags) or '-'}  mode: {args.mode}  "
            f"reranker: {retriever.reranker.name}\n"
        )
        print(format_results(chunks))
        print("\n" + "  ".join(f"{k}={v:.0f}" for k, v in retriever.last_timings.items()))
        did_something = True

    if args.answer:
        from fplcopilot.rag.kb.answer import answer_strategy_question, format_answer

        result = answer_strategy_question(args.answer, k=args.k, tags=_parse_tags(args.tags))
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=1, default=str))
        else:
            print(format_answer(result))
        did_something = True

    if not did_something:
        ap.print_help()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
