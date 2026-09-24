"""Поиск по стратегической KB: dense (pgvector) + BM25 -> RRF -> cross-encoder -> cap на документ.

Переиспользует rag/retrieve.py: rrf_fuse, tokenize_bm25, протокол Reranker / make_reranker
(flashrank), apply_article_cap (article_id = doc_id). Отличия от новостного Retriever:

- НЕТ time-decay и date-aware rerank: корпус вечнозелёный (правила, стратегия). Дата публикации
  гайда про Bench Boost 2024 года ничего не говорит о его полезности; свежесть важна только для
  правил сезона — за неё отвечает реестр (тег rules + официальные страницы 2026/27).
- Фильтр по тегам вместо сущностей: `tags && :tags` в SQL и по чанкам для BM25 (строгий фильтр —
  тег задаёт намерение пользователя, добора «из остального» нет).
- per_doc_cap=2 по умолчанию: один длинный гайд не должен занимать весь top-k.

KBChunk наследует RetrievedChunk (article_id = doc_id, published_at = дата публикации или
fetched_at, players/teams пустые), поэтому reranker и cap работают без изменений.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal

from rank_bm25 import BM25Okapi
from sqlalchemy import text

from fplcopilot.db import session_scope
from fplcopilot.rag.index import vector_literal
from fplcopilot.rag.retrieve import (
    Reranker,
    RetrievedChunk,
    apply_article_cap,
    make_reranker,
    rrf_fuse,
    tokenize_bm25,
)

log = logging.getLogger(__name__)

KBMode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]
KB_MODES: tuple[KBMode, ...] = ("dense", "bm25", "hybrid", "hybrid_rerank")
DEFAULT_K = 6
DEFAULT_CANDIDATES = 30
DEFAULT_PER_DOC_CAP = 2


@dataclass
class KBChunk(RetrievedChunk):
    """Чанк KB: article_id = kb_docs.id; tags — теги документа (rules = официальные правила)."""

    tags: list[str] = field(default_factory=list)

    @property
    def doc_id(self) -> int:
        return self.article_id

    @property
    def is_rules(self) -> bool:
        return "rules" in self.tags


# ---------- доступ к БД ----------

_SELECT = (
    "c.id, c.doc_id, c.text, c.source, d.url, d.title, coalesce(d.published_at, d.fetched_at), "
    "c.tags"
)
_DENSE_SQL = (
    f"SELECT {_SELECT}, 1 - (c.embedding <=> CAST(:q AS vector)) AS score "
    "FROM kb_chunks c JOIN kb_docs d ON d.id = c.doc_id "
    "WHERE c.embedding IS NOT NULL{tags} "
    "ORDER BY c.embedding <=> CAST(:q AS vector) LIMIT :n"
)
_TAGS_CLAUSE = " AND c.tags && CAST(:tags AS text[])"
_CORPUS_SQL = text(
    f"SELECT {_SELECT} FROM kb_chunks c JOIN kb_docs d ON d.id = c.doc_id ORDER BY c.id"
)
_VERSION_SQL = text("SELECT coalesce(max(id), 0), count(*) FROM kb_chunks")


def _row_to_chunk(row: Any) -> KBChunk:
    return KBChunk(
        chunk_id=int(row[0]),
        article_id=int(row[1]),
        text=row[2],
        source=row[3],
        url=row[4],
        title=row[5],
        published_at=row[6],
        tags=list(row[7] or []),
    )


@dataclass
class KBCorpusDoc:
    chunk: KBChunk
    tokens: list[str]


class KBStore:
    """SQL-доступ к kb_chunks; корпус для BM25 кэшируется в памяти по версии (max id, count)."""

    def __init__(self) -> None:
        self._corpus: list[KBCorpusDoc] = []
        self._version: tuple[int, int] | None = None

    def dense(self, qvec: list[float], *, tags: Sequence[str] | None, n: int) -> list[KBChunk]:
        clause = _TAGS_CLAUSE if tags else ""
        params: dict[str, Any] = {"q": vector_literal(qvec), "n": n}
        if tags:
            params["tags"] = list(tags)
        with session_scope() as s:
            s.execute(text("SET LOCAL hnsw.iterative_scan = 'strict_order'"))
            rows = s.execute(text(_DENSE_SQL.format(tags=clause)), params).all()
        out = []
        for row in rows:
            c = _row_to_chunk(row)
            c.dense_score = float(row[8])
            out.append(c)
        return out

    def version(self) -> tuple[int, int]:
        with session_scope() as s:
            max_id, n = s.execute(_VERSION_SQL).one()
        return int(max_id), int(n)

    def corpus(self) -> tuple[tuple[int, int], list[KBCorpusDoc]]:
        version = self.version()
        if version != self._version:
            started = time.perf_counter()
            with session_scope() as s:
                rows = s.execute(_CORPUS_SQL).all()
            self._corpus = [
                KBCorpusDoc(chunk=(c := _row_to_chunk(r)), tokens=tokenize_bm25(c.text))
                for r in rows
            ]
            self._version = version
            log.debug(
                "kb bm25 corpus: %d chunks in %.0fms",
                len(rows),
                (time.perf_counter() - started) * 1000,
            )
        return version, self._corpus


# ---------- retriever ----------


def matches_tags(chunk: KBChunk, tags: Sequence[str] | None) -> bool:
    return not tags or bool(set(chunk.tags) & set(tags))


class KBRetriever:
    def __init__(
        self,
        *,
        store: KBStore | None = None,
        embed: Callable[[str], list[float]] | None = None,
        reranker: Reranker | None = None,
        candidates: int = DEFAULT_CANDIDATES,
        per_doc_cap: int | None = DEFAULT_PER_DOC_CAP,
    ) -> None:
        self.store = store or KBStore()
        self._embed = embed
        self.reranker = reranker or make_reranker()
        self.candidates = candidates
        self.per_doc_cap = per_doc_cap
        self.last_timings: dict[str, float] = {}
        self.last_candidates: list[KBChunk] = []  # все кандидаты после ранжирования, до cap/k
        self._qcache: dict[str, list[float]] = {}
        self._bm25_cache: dict[
            tuple[tuple[int, int], tuple[str, ...]], tuple[BM25Okapi | None, list[KBCorpusDoc]]
        ] = {}

    def search(
        self,
        query: str,
        *,
        tags: Sequence[str] | None = None,
        k: int = DEFAULT_K,
        mode: KBMode = "hybrid_rerank",
        per_doc_cap: int | None = None,
    ) -> list[KBChunk]:
        if mode not in KB_MODES:
            raise ValueError(f"mode должен быть одним из {KB_MODES}, получен {mode!r}")
        if not query or not query.strip():
            raise ValueError("пустой запрос")
        tags = [t for t in (tags or []) if t] or None
        # per_doc_cap: None -> дефолт retriever'а; 0 -> без ограничения (apply_article_cap)
        cap = self.per_doc_cap if per_doc_cap is None else per_doc_cap
        timings: dict[str, float] = {}
        started = time.perf_counter()
        n = max(self.candidates, k)

        dense_hits: list[KBChunk] = []
        bm25_hits: list[KBChunk] = []
        if mode != "bm25":
            t = time.perf_counter()
            qvec = self._embed_query(query)
            timings["embed_ms"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            dense_hits = self.store.dense(qvec, tags=tags, n=n)
            timings["dense_ms"] = (time.perf_counter() - t) * 1000
        if mode != "dense":
            t = time.perf_counter()
            bm25_hits = self._bm25(query, tags, n)
            timings["bm25_ms"] = (time.perf_counter() - t) * 1000

        # защита в глубину: фильтр тегов применяется и в Python
        dense_hits = [c for c in dense_hits if matches_tags(c, tags)]
        bm25_hits = [c for c in bm25_hits if matches_tags(c, tags)]

        if mode == "dense":
            ranked: list[KBChunk] = dense_hits
        elif mode == "bm25":
            ranked = bm25_hits
        else:
            t = time.perf_counter()
            ranked = self._fuse(dense_hits, bm25_hits)[:n]
            timings["fuse_ms"] = (time.perf_counter() - t) * 1000
            if mode == "hybrid_rerank":
                t = time.perf_counter()
                ranked = list(self.reranker.rerank(query, ranked))  # type: ignore[arg-type]
                timings["rerank_ms"] = (time.perf_counter() - t) * 1000

        self.last_candidates = list(ranked)
        result = apply_article_cap(ranked, k, cap)
        timings["total_ms"] = (time.perf_counter() - started) * 1000
        self.last_timings = timings
        return result  # type: ignore[return-value]

    # -- стадии --

    def _embed_query(self, query: str) -> list[float]:
        if query not in self._qcache:
            if self._embed is None:
                from fplcopilot.rag.llm import embed_query

                self._embed = embed_query
            if len(self._qcache) >= 256:
                self._qcache.clear()
            self._qcache[query] = self._embed(query)
        return self._qcache[query]

    def _bm25_index(self, tags: Sequence[str] | None) -> tuple[BM25Okapi | None, list[KBCorpusDoc]]:
        version, docs = self.store.corpus()
        key = (version, tuple(sorted(tags or ())))
        if key not in self._bm25_cache:
            eligible = [d for d in docs if matches_tags(d.chunk, tags)]
            if len(self._bm25_cache) >= 8:
                self._bm25_cache.pop(next(iter(self._bm25_cache)))
            index = BM25Okapi([d.tokens for d in eligible]) if eligible else None
            self._bm25_cache[key] = (index, eligible)
        return self._bm25_cache[key]

    def _bm25(self, query: str, tags: Sequence[str] | None, n: int) -> list[KBChunk]:
        index, eligible = self._bm25_index(tags)
        if index is None:
            return []
        scores = index.get_scores(tokenize_bm25(query))
        order = sorted((i for i in range(len(eligible)) if scores[i] > 0), key=lambda i: -scores[i])
        return [replace(eligible[i].chunk, bm25_score=float(scores[i])) for i in order[:n]]

    @staticmethod
    def _fuse(dense_hits: list[KBChunk], bm25_hits: list[KBChunk]) -> list[KBChunk]:
        """RRF без time-decay: rrf_score и есть итоговая оценка hybrid."""
        by_id: dict[int, KBChunk] = {}
        for c in dense_hits:
            by_id[c.chunk_id] = replace(c)
        for c in bm25_hits:
            if c.chunk_id in by_id:
                by_id[c.chunk_id].bm25_score = c.bm25_score
            else:
                by_id[c.chunk_id] = replace(c)
        rrf = rrf_fuse([[c.chunk_id for c in dense_hits], [c.chunk_id for c in bm25_hits]])
        for cid, c in by_id.items():
            c.rrf_score = rrf[cid]
        return sorted(by_id.values(), key=lambda c: -(c.rrf_score or 0.0))


# ---------- вывод ----------


def _fmt(score: float | None, width: int = 6, digits: int = 3) -> str:
    return f"{score:{width}.{digits}f}" if score is not None else " " * width


def format_results(chunks: Sequence[KBChunk]) -> str:
    head = f"{'#':>2} {'source':<16} {'doc':>4} {'dense':>6} {'bm25':>6} {'rrf':>6} {'rerank':>6}  title / text"
    lines = [head, "-" * len(head)]
    for i, c in enumerate(chunks, start=1):
        snippet = " ".join(c.text.split("\n", 1)[-1].split())[:110]
        lines.append(
            f"{i:>2} {c.source:<16} {c.article_id:>4} {_fmt(c.dense_score)} "
            f"{_fmt(c.bm25_score, digits=2)} {_fmt(c.rrf_score, digits=4)} {_fmt(c.rerank_score)}  "
            f"{c.title[:60]} [{','.join(c.tags)}]\n{'':>42}{snippet}"
        )
    return "\n".join(lines)


def chunk_to_dict(c: KBChunk) -> dict[str, Any]:
    return {
        "chunk_id": c.chunk_id,
        "doc_id": c.article_id,
        "title": c.title,
        "url": c.url,
        "source": c.source,
        "tags": list(c.tags),
        "published_at": c.published_at.isoformat()
        if isinstance(c.published_at, datetime)
        else None,
        "dense_score": c.dense_score,
        "bm25_score": c.bm25_score,
        "rrf_score": c.rrf_score,
        "rerank_score": c.rerank_score,
        "score": c.final_score,
        "text": c.text,
    }
