"""Ингест KB: реестр -> загрузка (кэш) -> kb_docs -> чанки с заголовками -> эмбеддинги -> kb_chunks.

    uv run python -m fplcopilot.rag.kb --probe            # проверить источники (без БД и эмбеддингов)
    uv run python -m fplcopilot.rag.kb --ingest [--limit N] [--offline]
    uv run python -m fplcopilot.rag.kb --stats

Идемпотентность: документ пишется целиком в одной транзакции; при повторном запуске документ
с тем же content_hash (и уже имеющимися чанками) пропускается без эмбеддинга; изменившийся —
перезаписывается вместе с чанками (DELETE + INSERT). Эмбеддинги — rag.llm.embed_texts
(батчи <= 100), литерал вектора — rag.index.vector_literal.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from fplcopilot.config import settings
from fplcopilot.db import session_scope
from fplcopilot.rag.index import EMBEDDING_PRICE_PER_MTOK, vector_literal
from fplcopilot.rag.ingest import content_hash, normalize_url
from fplcopilot.rag.kb.chunking import chunk_document, clean_title
from fplcopilot.rag.kb.fetch import Fetched, KBFetcher
from fplcopilot.rag.kb.registry import KBSource, enabled_sources, load_registry
from fplcopilot.rag.llm import embed_texts

log = logging.getLogger("fplcopilot.kb.ingest")

REDDIT_BLOCK_MARKERS = ("login", "HTTP 403", "HTTP 429")


@dataclass
class SourceResult:
    name: str
    source: str
    url: str
    fetch: str
    tags: list[str]
    action: str = "pending"  # ok | new | updated | retagged | unchanged | failed | skipped
    status: int | None = None
    chars: int = 0
    method: str = "none"
    title: str = ""
    error: str | None = None
    from_cache: bool = False
    chunks: int = 0
    tokens: int = 0
    doc_id: int | None = None
    published_at: str | None = None

    @classmethod
    def from_fetched(cls, src: KBSource, f: Fetched) -> SourceResult:
        return cls(
            name=src.name,
            source=src.source,
            url=src.canonical_url,
            fetch=src.fetch,
            tags=list(src.tags),
            action="ok" if f.ok else "failed",
            status=f.status,
            chars=f.chars,
            method=f.method,
            title=f.title,
            error=f.error,
            from_cache=f.from_cache,
            published_at=f.published_at.isoformat() if f.published_at else None,
        )


@dataclass
class IngestStats:
    results: list[SourceResult] = field(default_factory=list)
    docs_new: int = 0
    docs_updated: int = 0
    docs_retagged: int = 0  # текст тот же, изменились теги/издатель в реестре — без эмбеддинга
    docs_unchanged: int = 0
    docs_failed: int = 0
    docs_skipped: int = 0
    chunks: int = 0
    tokens: int = 0
    elapsed_s: float = 0.0
    network_calls: int = 0
    cache_hits: int = 0
    stale_docs: list[str] = field(default_factory=list)  # в БД, но нет среди включённых источников
    pruned: int = 0

    @property
    def cost_usd(self) -> float:
        return self.tokens / 1_000_000 * EMBEDDING_PRICE_PER_MTOK

    def __str__(self) -> str:
        return (
            f"new={self.docs_new} updated={self.docs_updated} retagged={self.docs_retagged} "
            f"unchanged={self.docs_unchanged} failed={self.docs_failed} skipped={self.docs_skipped} "
            f"chunks={self.chunks} "
            f"tokens={self.tokens} (~${self.cost_usd:.4f}) network={self.network_calls} "
            f"cache={self.cache_hits} stale={len(self.stale_docs)} pruned={self.pruned} "
            f"in {self.elapsed_s:.1f}s"
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cost_usd"] = self.cost_usd
        return d


# ---------- SQL ----------

_SELECT_DOC = text(
    "SELECT d.id, d.content_hash, (SELECT count(*) FROM kb_chunks c WHERE c.doc_id = d.id), "
    "d.tags, d.source FROM kb_docs d WHERE d.url = :url"
)
_RETAG_DOC = text(
    "UPDATE kb_docs SET tags = CAST(:tags AS text[]), source = :source WHERE id = :id"
)
_RETAG_CHUNKS = text(
    "UPDATE kb_chunks SET tags = CAST(:tags AS text[]), source = :source WHERE doc_id = :id"
)
_INSERT_DOC = text(
    """
    INSERT INTO kb_docs (source, url, title, content, tags, published_at, fetched_at, content_hash)
    VALUES (:source, :url, :title, :content, CAST(:tags AS text[]), :published_at, now(),
            :content_hash)
    RETURNING id
    """
)
_UPDATE_DOC = text(
    """
    UPDATE kb_docs
    SET source = :source, title = :title, content = :content, tags = CAST(:tags AS text[]),
        published_at = :published_at, fetched_at = now(), content_hash = :content_hash
    WHERE id = :id
    """
)
_DELETE_CHUNKS = text("DELETE FROM kb_chunks WHERE doc_id = :id")
_ALL_DOC_URLS = text("SELECT id, url, source FROM kb_docs ORDER BY id")
_DELETE_DOC = text("DELETE FROM kb_docs WHERE id = :id")  # чанки — ON DELETE CASCADE
_INSERT_CHUNK = text(
    """
    INSERT INTO kb_chunks (doc_id, chunk_index, text, n_tokens, embedding, tags, source)
    VALUES (:doc_id, :chunk_index, :text, :n_tokens, CAST(:embedding AS vector),
            CAST(:tags AS text[]), :source)
    ON CONFLICT (doc_id, chunk_index) DO NOTHING
    """
)


# ---------- проверка источников (без БД) ----------


class _RedditGate:
    """После первого отказа Reddit (login-redirect / 403 / 429) остальные reddit-источники
    пропускаем — иначе тратим лимит запросов на заведомо одинаковый ответ."""

    def __init__(self) -> None:
        self.blocked_reason: str | None = None

    def allow(self, src: KBSource) -> bool:
        return not (src.fetch == "reddit_json" and self.blocked_reason)

    def observe(self, src: KBSource, f: Fetched) -> None:
        if (
            src.fetch == "reddit_json"
            and f.error
            and any(m in f.error for m in REDDIT_BLOCK_MARKERS)
        ):
            self.blocked_reason = f.error


def probe(
    sources: Sequence[KBSource] | None = None, *, fetcher: KBFetcher | None = None
) -> list[SourceResult]:
    """Загрузить каждый включённый источник (через кэш) и вернуть статус/размер текста."""
    sources = list(sources) if sources is not None else enabled_sources()
    fetcher = fetcher or KBFetcher()
    gate = _RedditGate()
    out: list[SourceResult] = []
    for src in sources:
        if not gate.allow(src):
            res = SourceResult(src.name, src.source, src.canonical_url, src.fetch, list(src.tags))
            res.action, res.error = "skipped", f"skipped: {gate.blocked_reason}"
            out.append(res)
            continue
        f = fetcher.fetch(src)
        gate.observe(src, f)
        out.append(SourceResult.from_fetched(src, f))
        log.info(
            "%-32s %-7s %6d chars  %s%s",
            src.name,
            f.status if f.status is not None else "-",
            f.chars,
            "OK" if f.ok else "FAIL",
            f" ({f.error})" if f.error else "",
        )
    return out


def format_probe_table(results: Sequence[SourceResult]) -> str:
    head = f"{'name':<34} {'src':<16} {'st':>4} {'chars':>6} {'method':<12} result"
    lines = [head, "-" * len(head)]
    for r in results:
        st = "-" if r.status is None else str(r.status)
        result = r.action.upper() if r.action in ("ok", "skipped") else "FAIL"
        if r.error:
            result += f": {r.error}"
        elif r.title:
            result += f": {r.title[:70]}"
        lines.append(f"{r.name:<34} {r.source:<16} {st:>4} {r.chars:>6} {r.method:<12} {result}")
    ok = sum(1 for r in results if r.action == "ok")
    lines.append(f"\nok={ok} failed={sum(1 for r in results if r.action == 'failed')} "
                 f"skipped={sum(1 for r in results if r.action == 'skipped')} of {len(results)}")  # fmt: skip
    return "\n".join(lines)


# ---------- ингест в БД ----------


@dataclass(frozen=True)
class ExistingDoc:
    id: int
    content_hash: str
    n_chunks: int
    tags: tuple[str, ...]
    source: str


def _existing(session: Session, url: str) -> ExistingDoc | None:
    row = session.execute(_SELECT_DOC, {"url": url}).first()
    if row is None:
        return None
    return ExistingDoc(int(row[0]), str(row[1]), int(row[2]), tuple(row[3] or ()), str(row[4]))


def ingest_document(
    session: Session, src: KBSource, f: Fetched, *, dry_run: bool = False
) -> SourceResult:
    """Один документ: сравнить hash -> при необходимости (пере)записать doc и чанки. Commit снаружи."""
    res = SourceResult.from_fetched(src, f)
    if not f.ok:
        return res
    url = src.canonical_url
    title = clean_title(f.title) or src.title_hint
    digest = content_hash(title, f.text)
    existing = _existing(session, url)
    if existing and existing.content_hash == digest and existing.n_chunks > 0:
        res.doc_id = existing.id
        if existing.tags == src.tags and existing.source == src.source:
            res.action = "unchanged"
            return res
        # текст тот же — эмбеддинги не трогаем, только теги/издатель из реестра
        res.action = "retagged"
        if not dry_run:
            params = {"tags": list(src.tags), "source": src.source, "id": existing.id}
            session.execute(_RETAG_DOC, params)
            session.execute(_RETAG_CHUNKS, params)
        return res

    chunks = chunk_document(title=title, text=f.text, max_tokens=settings.rag_chunk_tokens)
    res.chunks = len(chunks)
    res.tokens = sum(c.n_tokens for c in chunks)
    if dry_run:
        res.action = "updated" if existing else "new"
        return res

    vectors = embed_texts([c.text for c in chunks])
    params = {
        "source": src.source,
        "url": url,
        "title": title,
        "content": f.text,
        "tags": list(src.tags),
        "published_at": f.published_at,
        "content_hash": digest,
    }
    if existing:
        session.execute(_UPDATE_DOC, params | {"id": existing.id})
        session.execute(_DELETE_CHUNKS, {"id": existing.id})
        doc_id, res.action = existing.id, "updated"
    else:
        doc_id = int(session.execute(_INSERT_DOC, params).scalar_one())
        res.action = "new"
    session.execute(
        _INSERT_CHUNK,
        [
            {
                "doc_id": doc_id,
                "chunk_index": c.index,
                "text": c.text,
                "n_tokens": c.n_tokens,
                "embedding": vector_literal(v),
                "tags": list(src.tags),
                "source": src.source,
            }
            for c, v in zip(chunks, vectors, strict=True)
        ],
    )
    res.doc_id = doc_id
    return res


def _derived_reddit_sources(hub: KBSource, f: Fetched, known_urls: set[str]) -> list[KBSource]:
    """Хаб reddit: ссылки на другие посты сабреддита -> производные источники (теги хаба)."""
    out: list[KBSource] = []
    for url in f.linked_urls:
        if len(out) >= hub.follow_links:
            break
        key = normalize_url(url)
        if key in known_urls:
            continue
        known_urls.add(key)
        post_id = key.rstrip("/").rsplit("/", 1)[-1]
        out.append(
            KBSource(
                name=f"{hub.name}__{post_id}",
                source=hub.source,
                url=url,
                title_hint=f"{hub.title_hint} — linked post {post_id}",
                tags=hub.tags,
                fetch="reddit_json",
                note=f"followed from {hub.name}",
            )
        )
    return out


def stale_docs(session: Session, known_urls: set[str]) -> list[tuple[int, str]]:
    """Документы kb_docs, которых нет среди включённых источников (выключены/удалены из реестра)."""
    return [
        (int(row[0]), str(row[1]))
        for row in session.execute(_ALL_DOC_URLS).all()
        if str(row[1]) not in known_urls
    ]


def ingest(
    *,
    limit: int | None = None,
    sources: Iterable[KBSource] | None = None,
    fetcher: KBFetcher | None = None,
    dry_run: bool = False,
    prune: bool = False,
) -> IngestStats:
    """Полный прогон по включённым источникам реестра (первые limit по порядку файла).

    prune=True удаляет из БД документы, которых больше нет среди включённых источников
    (только при полном прогоне без limit — иначе «лишними» окажутся ещё не дошедшие до очереди).
    """
    started = time.monotonic()
    fetcher = fetcher or KBFetcher()
    all_enabled = list(sources) if sources is not None else enabled_sources(load_registry())
    queue = list(all_enabled)
    if limit is not None:
        queue = queue[:limit]
    known_urls = {s.canonical_url for s in queue}
    gate = _RedditGate()
    stats = IngestStats()

    with session_scope() as session:
        while queue:
            src = queue.pop(0)
            if not gate.allow(src):
                res = SourceResult(
                    src.name, src.source, src.canonical_url, src.fetch, list(src.tags)
                )
                res.action, res.error = "skipped", f"skipped: {gate.blocked_reason}"
                stats.results.append(res)
                stats.docs_skipped += 1
                continue
            f = fetcher.fetch(src)
            gate.observe(src, f)
            try:
                res = ingest_document(session, src, f, dry_run=dry_run)
                session.commit()
            except Exception as exc:  # один источник не должен валить прогон
                session.rollback()
                log.exception("%s: ingest failed", src.name)
                res = SourceResult.from_fetched(src, f)
                res.action, res.error = "failed", f"{type(exc).__name__}: {exc}"
            stats.results.append(res)
            if res.action == "new":
                stats.docs_new += 1
            elif res.action == "updated":
                stats.docs_updated += 1
            elif res.action == "retagged":
                stats.docs_retagged += 1
            elif res.action == "unchanged":
                stats.docs_unchanged += 1
            else:
                stats.docs_failed += 1
            stats.chunks += res.chunks
            stats.tokens += res.tokens
            log.info(
                "%-32s %-9s chunks=%-3d tokens=%-5d %s",
                src.name,
                res.action,
                res.chunks,
                res.tokens,
                res.error or res.title[:60],
            )
            if f.ok and src.follow_links:
                derived = _derived_reddit_sources(src, f, known_urls)
                queue[:0] = derived
                log.info("%s: following %d linked post(s)", src.name, len(derived))

        if limit is None:
            stale = stale_docs(session, known_urls)
            stats.stale_docs = [url for _, url in stale]
            for doc_id, url in stale:
                if prune and not dry_run:
                    session.execute(_DELETE_DOC, {"id": doc_id})
                    stats.pruned += 1
                    log.info("pruned stale doc %d: %s", doc_id, url)
                else:
                    log.warning(
                        "stale doc %d not in enabled registry: %s (--prune removes)", doc_id, url
                    )
            session.commit()

    stats.network_calls = fetcher.network_calls
    stats.cache_hits = fetcher.cache_hits
    stats.elapsed_s = time.monotonic() - started
    log.info("kb ingest done: %s", stats)
    return stats


# ---------- статистика ----------


def build_stats() -> str:
    lines: list[str] = []
    with session_scope() as s:
        n_docs, n_src = s.execute(
            text("SELECT count(*), count(DISTINCT source) FROM kb_docs")
        ).one()
        n_chunks, n_emb, tokens, avg_tok, max_tok = s.execute(
            text(
                "SELECT count(*), count(embedding), coalesce(sum(n_tokens), 0), "
                "coalesce(avg(n_tokens), 0)::int, coalesce(max(n_tokens), 0) FROM kb_chunks"
            )
        ).one()
        lines.append(f"Документов: {n_docs} (источников: {n_src})")
        lines.append(
            f"Чанков: {n_chunks}, с эмбеддингом: {n_emb}, токенов: {tokens} "
            f"(avg {avg_tok}, max {max_tok}; ~${tokens / 1e6 * EMBEDDING_PRICE_PER_MTOK:.4f})"
        )
        lines.append("\nПо источникам (документов / чанков / чанков на документ / avg токенов):")
        for source, docs, chunks, per, avg in s.execute(
            text(
                "SELECT source, count(DISTINCT doc_id), count(*), "
                "round(count(*)::numeric / count(DISTINCT doc_id), 2), avg(n_tokens)::int "
                "FROM kb_chunks GROUP BY source ORDER BY 3 DESC"
            )
        ).all():
            lines.append(f"  {source:<18}{docs:>6}{chunks:>7}{per:>7}{avg:>6}")
        lines.append("\nПо тегам (документов / чанков):")
        for tag, docs, chunks in s.execute(
            text(
                "SELECT t, count(DISTINCT doc_id), count(*) FROM kb_chunks, unnest(tags) AS t "
                "GROUP BY t ORDER BY 3 DESC"
            )
        ).all():
            lines.append(f"  {tag:<18}{docs:>6}{chunks:>7}")
    return "\n".join(lines)


def corpus_stats() -> dict[str, Any]:
    """Для evals: размеры корпуса и разбиение по источникам."""
    with session_scope() as s:
        n_docs = int(s.execute(text("SELECT count(*) FROM kb_docs")).scalar_one())
        n_chunks, n_emb, tokens = s.execute(
            text("SELECT count(*), count(embedding), coalesce(sum(n_tokens), 0) FROM kb_chunks")
        ).one()
        by_source = dict(
            s.execute(
                text("SELECT source, count(*) FROM kb_docs GROUP BY source ORDER BY 2 DESC")
            ).all()
        )
        max_fetched = s.execute(text("SELECT max(fetched_at) FROM kb_docs")).scalar_one()
    return {
        "docs": n_docs,
        "chunks": int(n_chunks),
        "chunks_embedded": int(n_emb),
        "tokens": int(tokens),
        "docs_by_source": {k: int(v) for k, v in by_source.items()},
        "max_fetched_at": max_fetched.isoformat() if max_fetched else None,
    }
