"""Индекс для RAG: news_articles -> news_chunks с эмбеддингами OpenAI (pgvector).

Запуск:
    uv run python -m fplcopilot.rag.index --rebuild-missing [--limit N]   # только статьи без чанков
    uv run python -m fplcopilot.rag.index --stats
    uv run python -m fplcopilot.rag.index --reset --yes                    # удалить все чанки

Инкрементально: обрабатываются только статьи, у которых ещё нет ни одного чанка; статья
пишется целиком в одной транзакции (иначе «полстатьи» выглядела бы как проиндексированная).
Эмбеддинги — батчами <= 100 текстов (llm.embed_texts). Тот же вызов делает ingest --index
после каждого цикла сбора.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from fplcopilot.config import settings
from fplcopilot.db import session_scope
from fplcopilot.rag.chunking import chunk_article
from fplcopilot.rag.llm import embed_texts

log = logging.getLogger("fplcopilot.index")

EMBEDDING_PRICE_PER_MTOK = 0.02  # USD, text-embedding-3-small (для оценки в отчёте)
ARTICLES_PER_BATCH = 40  # ~100-200 чанков за транзакцию

_MISSING = text(
    """
    SELECT a.id, a.source, a.title, a.summary, a.content, a.published_at, a.players, a.teams,
           coalesce(a.raw->>'fulltext', '') = 'true' AS fulltext
    FROM news_articles a
    WHERE NOT EXISTS (SELECT 1 FROM news_chunks c WHERE c.article_id = a.id)
    ORDER BY a.id
    LIMIT :n
    """
)
_INSERT_CHUNK = text(
    """
    INSERT INTO news_chunks
        (article_id, chunk_index, text, n_tokens, embedding, players, teams, source, published_at)
    VALUES
        (:article_id, :chunk_index, :text, :n_tokens, CAST(:embedding AS vector),
         :players, :teams, :source, :published_at)
    ON CONFLICT (article_id, chunk_index) DO NOTHING
    """
)


@dataclass
class IndexStats:
    articles: int = 0
    chunks: int = 0
    tokens: int = 0
    elapsed_s: float = 0.0

    @property
    def cost_usd(self) -> float:
        return self.tokens / 1_000_000 * EMBEDDING_PRICE_PER_MTOK

    def __str__(self) -> str:
        return (
            f"articles={self.articles} chunks={self.chunks} tokens={self.tokens} "
            f"(~${self.cost_usd:.4f}) in {self.elapsed_s:.1f}s"
        )


def vector_literal(vec: list[float]) -> str:
    """Список float -> литерал pgvector '[0.1,0.2,...]' (без регистрации типа в psycopg)."""
    return json.dumps(vec, separators=(",", ":"))


def index_articles(session: Session, rows: list[Any]) -> IndexStats:
    """Чанкует и эмбеддит переданные статьи, пишет чанки; commit — на вызывающей стороне."""
    st = IndexStats()
    pending: list[dict[str, Any]] = []
    for row in rows:
        chunks = chunk_article(
            title=row.title,
            summary=row.summary,
            content=row.content,
            fulltext=bool(row.fulltext),
            max_tokens=settings.rag_chunk_tokens,
        )
        for ch in chunks:
            pending.append(
                {
                    "article_id": row.id,
                    "chunk_index": ch.index,
                    "text": ch.text,
                    "n_tokens": ch.n_tokens,
                    "players": list(row.players or []),
                    "teams": list(row.teams or []),
                    "source": row.source,
                    "published_at": row.published_at,
                }
            )
            st.tokens += ch.n_tokens
        st.articles += 1
    if not pending:
        return st
    vectors = embed_texts([p["text"] for p in pending])
    for p, vec in zip(pending, vectors, strict=True):
        p["embedding"] = vector_literal(vec)
    session.execute(_INSERT_CHUNK, pending)
    st.chunks = len(pending)
    return st


def rebuild_missing(*, limit: int | None = None) -> IndexStats:
    """Индексирует все статьи без чанков (или первые limit по id). Возвращает суммарную статистику."""
    started = time.monotonic()
    total = IndexStats()
    remaining = limit
    with session_scope() as session:
        while remaining is None or remaining > 0:
            n = ARTICLES_PER_BATCH if remaining is None else min(ARTICLES_PER_BATCH, remaining)
            rows = session.execute(_MISSING, {"n": n}).all()
            if not rows:
                break
            st = index_articles(session, rows)
            session.commit()
            total.articles += st.articles
            total.chunks += st.chunks
            total.tokens += st.tokens
            if remaining is not None:
                remaining -= len(rows)
            log.info("indexed batch: %s (total %d articles)", st, total.articles)
    total.elapsed_s = time.monotonic() - started
    log.info("rebuild-missing done: %s", total)
    return total


def reset() -> int:
    with session_scope() as s:
        n = s.execute(text("DELETE FROM news_chunks")).rowcount
    log.info("news_chunks cleared: %d row(s)", n)
    return n


def build_stats() -> str:
    lines: list[str] = []
    with session_scope() as s:
        n_art, n_indexed = s.execute(
            text(
                "SELECT count(*), count(*) FILTER (WHERE EXISTS "
                "(SELECT 1 FROM news_chunks c WHERE c.article_id = a.id)) FROM news_articles a"
            )
        ).one()
        n_chunks, n_emb, tokens, avg_tok, max_tok = s.execute(
            text(
                "SELECT count(*), count(embedding), coalesce(sum(n_tokens), 0), "
                "coalesce(avg(n_tokens), 0)::int, coalesce(max(n_tokens), 0) FROM news_chunks"
            )
        ).one()
        lines.append(
            f"Статей: {n_art}, проиндексировано: {n_indexed}, без чанков: {n_art - n_indexed}"
        )
        lines.append(
            f"Чанков: {n_chunks}, с эмбеддингом: {n_emb}, токенов: {tokens} "
            f"(avg {avg_tok}, max {max_tok}; ~${tokens / 1e6 * EMBEDDING_PRICE_PER_MTOK:.4f})"
        )
        lines.append("\nПо источникам (статей / чанков / чанков на статью / avg токенов):")
        rows = s.execute(
            text(
                "SELECT source, count(DISTINCT article_id), count(*), "
                "round(count(*)::numeric / count(DISTINCT article_id), 2), avg(n_tokens)::int "
                "FROM news_chunks GROUP BY source ORDER BY 3 DESC"
            )
        ).all()
        for source, arts, chunks, per, avg in rows:
            lines.append(f"  {source:<18}{arts:>6}{chunks:>7}{per:>7}{avg:>6}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FPL Copilot: индекс чанков и эмбеддингов")
    ap.add_argument(
        "--rebuild-missing", action="store_true", help="проиндексировать статьи без чанков"
    )
    ap.add_argument("--limit", type=int, default=None, help="максимум статей за запуск")
    ap.add_argument("--stats", action="store_true", help="статистика индекса")
    ap.add_argument("--reset", action="store_true", help="удалить все чанки (нужен --yes)")
    ap.add_argument("--yes", action="store_true", help="подтверждение для --reset")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.reset:
        if not args.yes:
            print("--reset удалит все чанки и эмбеддинги; добавьте --yes для подтверждения")
            return 2
        print(f"удалено чанков: {reset()}")
    if args.rebuild_missing:
        st = rebuild_missing(limit=args.limit)
        print(f"OK: {st}")
    if args.stats or not (args.rebuild_missing or args.reset):
        print(build_stats())
    return 0


if __name__ == "__main__":
    sys.exit(main())
