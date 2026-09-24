"""CLI поиска по чанкам — для отладки retrieval и A/B режимов.

    uv run python -m fplcopilot.rag.query "Is João Pedro fit for GW5?" --player "João Pedro"
    uv run python -m fplcopilot.rag.query "Chelsea team news" --team CHE --mode dense -k 5
    uv run python -m fplcopilot.rag.query "..." --player Haaland --as-of 2026-09-10T12:00Z

Без --as-of берётся «сейчас» (только в CLI; в коде as_of обязателен).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime

from fplcopilot.data import Bootstrap, FPLClient, Team
from fplcopilot.rag.retrieve import (
    MODES,
    RetrievedChunk,
    Retriever,
    expand_player_query,
    expand_team_query,
    find_player,
)


def parse_as_of(value: str | None) -> datetime:
    """ISO-8601 ('2026-09-17T12:00Z', '2026-09-17 12:00', с офсетом) -> aware UTC; пусто -> сейчас."""
    if not value:
        return datetime.now(UTC)
    dt = datetime.fromisoformat(value)  # Python 3.11+ понимает суффикс Z
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def find_team(bs: Bootstrap, query: str) -> Team:
    q = query.lower()
    hits = [t for t in bs.teams if q in (t.name.lower(), t.short_name.lower())] or [
        t for t in bs.teams if q in t.name.lower()
    ]
    if len(hits) != 1:
        raise LookupError(f"клуб не найден или неоднозначен: {query!r}")
    return hits[0]


def _fmt(score: float | None, width: int = 6, digits: int = 3) -> str:
    return f"{score:{width}.{digits}f}" if score is not None else " " * width


def format_table(chunks: list[RetrievedChunk]) -> str:
    head = (
        f"{'#':>2} {'source':<18} {'published':<16} {'dense':>6} {'bm25':>6} {'rrf':>6} "
        f"{'decay':>6} {'rerank':>6}  text"
    )
    lines = [head, "-" * len(head)]
    for i, c in enumerate(chunks, start=1):
        snippet = " ".join(c.text.split())[:120]
        lines.append(
            f"{i:>2} {c.source:<18} {c.published_at:%Y-%m-%d %H:%M} {_fmt(c.dense_score)} "
            f"{_fmt(c.bm25_score, digits=2)} {_fmt(c.rrf_score, digits=4)} "
            f"{_fmt(c.decayed_score, digits=4)} {_fmt(c.rerank_score)}  {snippet}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FPL Copilot: поиск по новостным чанкам")
    ap.add_argument("query", help="вопрос/запрос")
    ap.add_argument("--player", help="имя игрока: фильтр по сущности + расширение запроса")
    ap.add_argument("--team", help="клуб (название или код): фильтр + расширение")
    ap.add_argument("--as-of", dest="as_of", help="ISO-время, документы новее отбрасываются")
    ap.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--no-expand", action="store_true", help="не добавлять расширение запроса")
    ap.add_argument("--timings", action="store_true", help="напечатать латентность стадий")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    for name in ("httpx", "httpx2"):
        logging.getLogger(name).setLevel(logging.WARNING)

    bs = FPLClient().bootstrap()
    query = args.query
    player_ids: list[int] | None = None
    team_ids: list[int] | None = None
    try:
        if args.player:
            p = find_player(bs, args.player)
            player_ids = [p.id]
            if not args.no_expand:
                query = f"{query} {expand_player_query(p, bs.team(p.team))}"
        if args.team:
            t = find_team(bs, args.team)
            team_ids = [t.id]
            if not args.no_expand:
                query = f"{query} {expand_team_query(t)}"
    except LookupError as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 2

    as_of = parse_as_of(args.as_of)
    retriever = Retriever()
    chunks = retriever.search(
        query, player_ids=player_ids, team_ids=team_ids, as_of=as_of, k=args.k, mode=args.mode
    )
    print(
        f"query: {query}\nas_of: {as_of:%Y-%m-%d %H:%M}Z  mode: {args.mode}  reranker: {retriever.reranker.name}\n"
    )
    print(format_table(chunks))
    if args.timings:
        print("\n" + "  ".join(f"{k}={v:.0f}" for k, v in retriever.last_timings.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
