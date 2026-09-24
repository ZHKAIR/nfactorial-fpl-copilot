"""Гибридный поиск по чанкам: dense (pgvector) + BM25 -> RRF -> time-decay -> reranker.

Режимы (для A/B): dense | bm25 | hybrid (RRF + time-decay) | hybrid_rerank (+ cross-encoder).

HARD RULE (утечка из будущего): любой поиск фильтрует published_at <= as_of — в SQL для
dense-кандидатов и при отборе кандидатов BM25; перед слиянием фильтр применяется ещё раз
в Python. as_of обязателен (дефолт «сейчас» есть только в CLI).

Сущности: если переданы player_ids/team_ids, сначала берём кандидатов, где
players && player_ids (или teams && team_ids), и лишь при нехватке добираем остальными.

Retrieval v2 (RetrievalConfig, A/B #2): per_article_cap — не больше N чанков одной статьи в
финальном top-k (добор из остатка); date_aware_rerank — итоговая оценка hybrid_rerank =
rerank_score × (floor + (1-floor)·time_decay), чтобы августовские заголовки не обгоняли свежие;
candidates — N кандидатов до reranker. v1 = cap None + date_aware False (старое поведение).
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol

from rank_bm25 import BM25Okapi
from sqlalchemy import text

from fplcopilot.config import settings
from fplcopilot.data.schemas import Bootstrap, Player, Team
from fplcopilot.db import session_scope
from fplcopilot.rag.entity_matcher import strip_accents
from fplcopilot.rag.index import vector_literal

if TYPE_CHECKING:
    from flashrank import Ranker

log = logging.getLogger(__name__)

Mode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]
MODES: tuple[Mode, ...] = ("dense", "bm25", "hybrid", "hybrid_rerank")
RRF_K = 60
INJURY_TERMS = "injury fitness training doubt return"
TEAM_TERMS = "team news injuries press conference squad"

_BM25_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for", "with", "by", "from",
        "as", "is", "are", "was", "were", "be", "been", "has", "have", "had", "he", "his", "him",
        "she", "her", "it", "its", "they", "their", "this", "that", "these", "those", "but", "not",
        "no", "will", "would", "could", "after", "before", "over", "under", "into", "than", "then",
        "there", "here", "who", "what", "which", "when", "where", "why", "how",
    }
)  # fmt: skip


# ---------- модель результата ----------


@dataclass
class RetrievedChunk:
    chunk_id: int
    article_id: int
    text: str
    source: str
    url: str
    title: str
    published_at: datetime
    players: list[int] = field(default_factory=list)
    teams: list[int] = field(default_factory=list)
    # оценки по стадиям (None — стадия не участвовала); нужны для A/B и отладки
    dense_score: float | None = None  # косинусная близость (1 - distance)
    bm25_score: float | None = None
    rrf_score: float | None = None
    decayed_score: float | None = None  # rrf * time_decay
    rerank_score: float | None = None  # «сырая» оценка cross-encoder (для порога abstention)
    date_aware_score: float | None = None  # rerank_score × date-multiplier (retrieval v2)

    @property
    def final_score(self) -> float | None:
        for s in (
            self.date_aware_score,
            self.rerank_score,
            self.decayed_score,
            self.rrf_score,
            self.dense_score,
        ):
            if s is not None:
                return s
        return self.bm25_score


# ---------- конфигурация поиска (v1/v2 для A/B) ----------


@dataclass(frozen=True)
class RetrievalConfig:
    """Ручки retrieval v2. Дефолты = v2; RETRIEVAL_V1 воспроизводит поведение A/B #1."""

    per_article_cap: int | None = 2  # макс. чанков одной статьи в top-k; None — без ограничения
    date_aware_rerank: bool = True  # hybrid_rerank: сортировать по rerank × date-multiplier
    candidates: int = 30  # N кандидатов (на каждый из dense/bm25) до слияния и reranker
    rerank_decay_floor: float = 0.5  # нижняя граница множителя: старое релевантное не хоронится

    @property
    def version(self) -> str:
        if self.per_article_cap is None and not self.date_aware_rerank:
            return "v1"
        if self.per_article_cap == 2 and self.date_aware_rerank and self.rerank_decay_floor == 0.5:
            return "v2"
        return "custom"


RETRIEVAL_V1 = RetrievalConfig(per_article_cap=None, date_aware_rerank=False)
RETRIEVAL_V2 = RetrievalConfig()
RETRIEVAL_VERSIONS: dict[str, RetrievalConfig] = {"v1": RETRIEVAL_V1, "v2": RETRIEVAL_V2}


def retrieval_config(version: str, *, candidates: int | None = None) -> RetrievalConfig:
    try:
        cfg = RETRIEVAL_VERSIONS[version]
    except KeyError as exc:
        raise ValueError(
            f"retrieval version должен быть одним из {sorted(RETRIEVAL_VERSIONS)}"
        ) from exc
    return replace(cfg, candidates=candidates) if candidates else cfg


def default_retrieval_config() -> RetrievalConfig:
    """Из settings: RAG_PER_ARTICLE_CAP / RAG_DATE_AWARE_RERANK / RAG_CANDIDATES."""
    return RetrievalConfig(
        per_article_cap=settings.rag_per_article_cap,
        date_aware_rerank=settings.rag_date_aware_rerank,
        candidates=settings.rag_candidates,
    )


# ---------- чистая математика (unit-тесты) ----------


def rrf_fuse(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum_i 1 / (k + rank_i(d)), ранги с 1."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def time_decay(published_at: datetime, as_of: datetime, half_life_days: float) -> float:
    """exp(-ln2 * age_days / half_life): 1.0 для свежего, 0.5 через half_life дней."""
    if half_life_days <= 0:
        return 1.0
    age_days = max(0.0, (as_of - published_at).total_seconds() / 86_400)
    return math.exp(-math.log(2) * age_days / half_life_days)


def date_multiplier(
    published_at: datetime, as_of: datetime, half_life_days: float, floor: float
) -> float:
    """floor + (1-floor)·decay: 1.0 для свежего, floor для очень старого (не хоронит официальное)."""
    floor = min(1.0, max(0.0, floor))
    return floor + (1.0 - floor) * time_decay(published_at, as_of, half_life_days)


def apply_date_aware_rerank(
    chunks: Sequence[RetrievedChunk], as_of: datetime, half_life_days: float, floor: float
) -> list[RetrievedChunk]:
    """date_aware_score = rerank_score × date_multiplier; сортировка по нему (rerank_score не трогаем)."""
    out = []
    for c in chunks:
        base = c.rerank_score if c.rerank_score is not None else (c.final_score or 0.0)
        out.append(
            replace(
                c,
                date_aware_score=base
                * date_multiplier(c.published_at, as_of, half_life_days, floor),
            )
        )
    return sorted(out, key=lambda c: -(c.date_aware_score or 0.0))


def apply_article_cap(
    chunks: Sequence[RetrievedChunk], k: int, cap: int | None
) -> list[RetrievedChunk]:
    """Не больше cap чанков одной статьи в top-k; нехватку добираем пропущенными по порядку."""
    if cap is None or cap <= 0:
        return list(chunks[:k])
    taken: list[RetrievedChunk] = []
    skipped: list[RetrievedChunk] = []
    per_article: Counter[int] = Counter()
    for c in chunks:
        if len(taken) >= k:
            break
        if per_article[c.article_id] < cap:
            taken.append(c)
            per_article[c.article_id] += 1
        else:
            skipped.append(c)
    if len(taken) < k:
        taken.extend(skipped[: k - len(taken)])
    return taken


def tokenize_bm25(text: str) -> list[str]:
    """Нижний регистр, без диакритики и стоп-слов: «João» и «Joao» — один токен."""
    return [t for t in _BM25_TOKEN.findall(strip_accents(text).lower()) if t not in _STOPWORDS]


def matches_entities(
    chunk: RetrievedChunk, player_ids: Sequence[int] | None, team_ids: Sequence[int] | None
) -> bool:
    if player_ids and set(chunk.players) & set(player_ids):
        return True
    return bool(team_ids and set(chunk.teams) & set(team_ids))


# ---------- расширение запроса (детерминированное; хук для LLM-переписывания ниже) ----------


def expand_player_query(player: Player, team: Team | None = None) -> str:
    parts = [player.full_name]
    if player.web_name.lower() != player.full_name.lower():
        parts.append(player.web_name)
    if team is not None:
        parts.append(team.name)
    parts.append(INJURY_TERMS)
    return " ".join(parts)


def expand_team_query(team: Team) -> str:
    return f"{team.name} {team.short_name} {TEAM_TERMS}"


def find_player(bs: Bootstrap, query: str) -> Player:
    """Игрок по имени без учёта регистра и диакритики; неоднозначность — LookupError со списком."""
    q = strip_accents(query).lower().strip()
    if not q:
        raise LookupError("пустое имя игрока")

    def norm(s: str) -> str:
        return strip_accents(s).lower()

    exact = [p for p in bs.elements if q in (norm(p.web_name), norm(p.full_name))]
    if len(exact) == 1:
        return exact[0]
    partial = exact or [p for p in bs.elements if q in norm(p.web_name) or q in norm(p.full_name)]
    if len(partial) == 1:
        return partial[0]
    if not partial:
        raise LookupError(f"игрок не найден: {query!r}")
    names = ", ".join(
        f"{p.web_name} ({bs.team(p.team).short_name}, id={p.id})" for p in partial[:8]
    )
    raise LookupError(f"неоднозначно {query!r}: {names}")


# ---------- reranker ----------


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]: ...


class NoopReranker:
    """Порядок не меняет; rerank_score = decayed/final score, чтобы поля были заполнены."""

    name = "none"

    def rerank(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        return [replace(c, rerank_score=c.final_score) for c in chunks]


class FlashrankReranker:
    """Cross-encoder ONNX (flashrank), без torch; модель качается при первом вызове в .cache."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.rag_reranker_model
        self.name = f"flashrank:{self.model_name}"
        self._ranker: Ranker | None = None

    def _load(self) -> Ranker:
        if self._ranker is None:
            from flashrank import Ranker

            cache_dir = settings.cache_dir / "flashrank"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._ranker = Ranker(
                model_name=self.model_name, cache_dir=str(cache_dir), log_level="WARNING"
            )
        return self._ranker

    def rerank(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        from flashrank import RerankRequest

        by_id = {c.chunk_id: c for c in chunks}
        passages = [{"id": c.chunk_id, "text": c.text} for c in chunks]
        ranked = self._load().rerank(RerankRequest(query=query, passages=passages))
        out = [replace(by_id[p["id"]], rerank_score=float(p["score"])) for p in ranked]
        return sorted(out, key=lambda c: -(c.rerank_score or 0.0))


def make_reranker(kind: str | None = None, model_name: str | None = None) -> Reranker:
    kind = kind or settings.rag_reranker
    if kind == "none":
        return NoopReranker()
    try:
        import flashrank  # noqa: F401 — проверяем, что зависимость установилась (onnxruntime)
    except ImportError as exc:
        log.warning("flashrank недоступен (%s) — reranker отключён, порядок = RRF*decay", exc)
        return NoopReranker()
    return FlashrankReranker(model_name)


# ---------- доступ к БД ----------

_SELECT_COLUMNS = (
    "c.id, c.article_id, c.text, c.source, a.url, a.title, c.published_at, c.players, c.teams"
)
_DENSE_SQL = (
    f"SELECT {_SELECT_COLUMNS}, 1 - (c.embedding <=> CAST(:q AS vector)) AS score "
    "FROM news_chunks c JOIN news_articles a ON a.id = c.article_id "
    "WHERE c.embedding IS NOT NULL AND c.published_at <= :as_of{entity}{cutoff} "
    "ORDER BY c.embedding <=> CAST(:q AS vector) LIMIT :n"
)
_ENTITY_CLAUSE = " AND (c.players && CAST(:players AS int[]) OR c.teams && CAST(:teams AS int[]))"
_CUTOFF_CLAUSE = " AND a.fetched_at <= :fetched_before"
_CORPUS_SQL = text(
    f"SELECT {_SELECT_COLUMNS}, a.fetched_at "
    "FROM news_chunks c JOIN news_articles a ON a.id = c.article_id ORDER BY c.id"
)
_VERSION_SQL = text("SELECT coalesce(max(id), 0), count(*) FROM news_chunks")


def _row_to_chunk(row: Any) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=row[0],
        article_id=row[1],
        text=row[2],
        source=row[3],
        url=row[4],
        title=row[5],
        published_at=row[6],
        players=list(row[7] or []),
        teams=list(row[8] or []),
    )


@dataclass
class CorpusDoc:
    chunk: RetrievedChunk
    tokens: list[str]
    fetched_at: datetime | None = None  # news_articles.fetched_at: время сбора (corpus_cutoff)


class ChunkStore:
    """SQL-доступ к news_chunks. Корпус для BM25 кэшируется в памяти по версии (max id, count)."""

    def __init__(self) -> None:
        self._corpus: list[CorpusDoc] = []
        self._version: tuple[int, int] | None = None

    def dense(
        self,
        qvec: list[float],
        *,
        as_of: datetime,
        player_ids: Sequence[int] | None,
        team_ids: Sequence[int] | None,
        n: int,
        fetched_before: datetime | None = None,
    ) -> list[RetrievedChunk]:
        entity = _ENTITY_CLAUSE if (player_ids or team_ids) else ""
        cutoff = _CUTOFF_CLAUSE if fetched_before is not None else ""
        params: dict[str, Any] = {"q": vector_literal(qvec), "as_of": as_of, "n": n}
        if entity:
            params |= {"players": list(player_ids or []), "teams": list(team_ids or [])}
        if cutoff:
            params["fetched_before"] = fetched_before
        with session_scope() as s:
            # iterative scan: с фильтром HNSW иначе может вернуть меньше LIMIT строк
            s.execute(text("SET LOCAL hnsw.iterative_scan = 'strict_order'"))
            rows = s.execute(text(_DENSE_SQL.format(entity=entity, cutoff=cutoff)), params).all()
        out = []
        for row in rows:
            c = _row_to_chunk(row)
            c.dense_score = float(row[9])
            out.append(c)
        return out

    def version(self) -> tuple[int, int]:
        with session_scope() as s:
            max_id, n = s.execute(_VERSION_SQL).one()
        return int(max_id), int(n)

    def corpus(self) -> tuple[tuple[int, int], list[CorpusDoc]]:
        """(версия, все чанки с токенами для BM25); перечитывается только при изменении версии."""
        version = self.version()
        if version != self._version:
            started = time.perf_counter()
            with session_scope() as s:
                rows = s.execute(_CORPUS_SQL).all()
            self._corpus = [
                CorpusDoc(
                    chunk=(c := _row_to_chunk(r)), tokens=tokenize_bm25(c.text), fetched_at=r[9]
                )
                for r in rows
            ]
            self._version = version
            log.debug(
                "bm25 corpus loaded: %d chunks in %.0fms",
                len(rows),
                (time.perf_counter() - started) * 1000,
            )
        return version, self._corpus


# ---------- retriever ----------


class Retriever:
    def __init__(
        self,
        *,
        store: ChunkStore | None = None,
        embed: Callable[[str], list[float]] | None = None,
        reranker: Reranker | None = None,
        half_life_days: float | None = None,
        candidates: int | None = None,
        query_rewriter: Callable[[str], str] | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.store = store or ChunkStore()
        self._embed = embed
        self.reranker = reranker or make_reranker()
        self.half_life_days = (
            settings.rag_half_life_days if half_life_days is None else half_life_days
        )
        cfg = config or default_retrieval_config()
        self.config = replace(cfg, candidates=candidates) if candidates else cfg
        self.query_rewriter = query_rewriter  # хук: LLM-переписывание запроса (позже, для A/B)
        self.last_timings: dict[str, float] = {}
        # все кандидаты последнего поиска после ранжирования, до cap/обрезки до k
        # (нужны abstention: есть ли официальный fpl_api-чанк, каков максимальный rerank_score)
        self.last_candidates: list[RetrievedChunk] = []
        self._qcache: dict[str, list[float]] = {}
        self._bm25_cache: dict[
            tuple[tuple[int, int], datetime, datetime | None],
            tuple[BM25Okapi | None, list[CorpusDoc]],
        ] = {}

    @property
    def candidates(self) -> int:
        return self.config.candidates

    # -- public --

    def search(
        self,
        query: str,
        *,
        player_ids: Sequence[int] | None = None,
        team_ids: Sequence[int] | None = None,
        as_of: datetime,
        k: int = 8,
        mode: Mode = "hybrid_rerank",
        config: RetrievalConfig | None = None,
        corpus_cutoff: datetime | None = None,
    ) -> list[RetrievedChunk]:
        """corpus_cutoff (только evals): видны лишь статьи, собранные до этого момента
        (news_articles.fetched_at) — воспроизводит корпус прошлого прогона / операционный replay.
        Не заменяет HARD RULE published_at <= as_of, а добавляется к нему."""
        if as_of.tzinfo is None:
            raise ValueError("as_of должен быть timezone-aware (UTC)")
        if corpus_cutoff is not None and corpus_cutoff.tzinfo is None:
            raise ValueError("corpus_cutoff должен быть timezone-aware (UTC)")
        if mode not in MODES:
            raise ValueError(f"mode должен быть одним из {MODES}, получен {mode!r}")
        as_of = as_of.astimezone(UTC)
        if self.query_rewriter is not None:
            query = self.query_rewriter(query)
        cfg = config or self.config

        timings: dict[str, float] = {}
        started = time.perf_counter()
        n = max(cfg.candidates, k)

        dense_hits: list[RetrievedChunk] = []
        bm25_hits: list[RetrievedChunk] = []
        if mode != "bm25":
            t = time.perf_counter()
            qvec = self._embed_query(query)
            timings["embed_ms"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            dense_hits = self._dense(qvec, as_of, player_ids, team_ids, n, corpus_cutoff)
            timings["dense_ms"] = (time.perf_counter() - t) * 1000
        if mode != "dense":
            t = time.perf_counter()
            bm25_hits = self._bm25(query, as_of, player_ids, team_ids, n, corpus_cutoff)
            timings["bm25_ms"] = (time.perf_counter() - t) * 1000

        # защита в глубину: ничего опубликованного после as_of дальше не проходит
        dense_hits = [c for c in dense_hits if c.published_at <= as_of]
        bm25_hits = [c for c in bm25_hits if c.published_at <= as_of]

        if mode == "dense":
            ranked = dense_hits
        elif mode == "bm25":
            ranked = bm25_hits
        else:
            t = time.perf_counter()
            ranked = self._fuse(dense_hits, bm25_hits, as_of)[:n]
            timings["fuse_ms"] = (time.perf_counter() - t) * 1000
            if mode == "hybrid_rerank":
                t = time.perf_counter()
                ranked = self.reranker.rerank(query, ranked)
                if cfg.date_aware_rerank:
                    ranked = apply_date_aware_rerank(
                        ranked, as_of, self.half_life_days, cfg.rerank_decay_floor
                    )
                timings["rerank_ms"] = (time.perf_counter() - t) * 1000

        self.last_candidates = list(ranked)
        result = apply_article_cap(ranked, k, cfg.per_article_cap)

        timings["total_ms"] = (time.perf_counter() - started) * 1000
        self.last_timings = timings
        return result

    # -- stages --

    def _embed_query(self, query: str) -> list[float]:
        if query not in self._qcache:
            if self._embed is None:
                from fplcopilot.rag.llm import embed_query

                self._embed = embed_query
            if len(self._qcache) >= 256:
                self._qcache.clear()
            self._qcache[query] = self._embed(query)
        return self._qcache[query]

    def _dense(
        self,
        qvec: list[float],
        as_of: datetime,
        player_ids: Sequence[int] | None,
        team_ids: Sequence[int] | None,
        n: int,
        corpus_cutoff: datetime | None = None,
    ) -> list[RetrievedChunk]:
        cutoff: dict[str, Any] = {"fetched_before": corpus_cutoff} if corpus_cutoff else {}
        hits: list[RetrievedChunk] = []
        if player_ids or team_ids:
            hits = self.store.dense(
                qvec, as_of=as_of, player_ids=player_ids, team_ids=team_ids, n=n, **cutoff
            )
        if len(hits) < n:
            seen = {c.chunk_id for c in hits}
            rest = self.store.dense(
                qvec, as_of=as_of, player_ids=None, team_ids=None, n=n + len(seen), **cutoff
            )
            hits += [c for c in rest if c.chunk_id not in seen][: n - len(hits)]
        return hits

    def _bm25_index(
        self, as_of: datetime, corpus_cutoff: datetime | None = None
    ) -> tuple[BM25Okapi | None, list[CorpusDoc]]:
        version, docs = self.store.corpus()
        key = (version, as_of, corpus_cutoff)
        if key not in self._bm25_cache:
            eligible = [d for d in docs if d.chunk.published_at <= as_of]  # HARD RULE для BM25
            if corpus_cutoff is not None:
                eligible = [
                    d for d in eligible if d.fetched_at is None or d.fetched_at <= corpus_cutoff
                ]
            if len(self._bm25_cache) >= 8:
                self._bm25_cache.pop(next(iter(self._bm25_cache)))
            index = BM25Okapi([d.tokens for d in eligible]) if eligible else None
            self._bm25_cache[key] = (index, eligible)
        return self._bm25_cache[key]

    def _bm25(
        self,
        query: str,
        as_of: datetime,
        player_ids: Sequence[int] | None,
        team_ids: Sequence[int] | None,
        n: int,
        corpus_cutoff: datetime | None = None,
    ) -> list[RetrievedChunk]:
        index, eligible = self._bm25_index(as_of, corpus_cutoff)
        if index is None:
            return []
        scores = index.get_scores(tokenize_bm25(query))
        order = sorted((i for i in range(len(eligible)) if scores[i] > 0), key=lambda i: -scores[i])
        if player_ids or team_ids:
            first = [i for i in order if matches_entities(eligible[i].chunk, player_ids, team_ids)]
            rest = [
                i for i in order if not matches_entities(eligible[i].chunk, player_ids, team_ids)
            ]
            order = first + rest
        out = []
        for i in order[:n]:
            c = replace(eligible[i].chunk, bm25_score=float(scores[i]))
            out.append(c)
        return out

    def _fuse(
        self, dense_hits: list[RetrievedChunk], bm25_hits: list[RetrievedChunk], as_of: datetime
    ) -> list[RetrievedChunk]:
        by_id: dict[int, RetrievedChunk] = {}
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
            c.decayed_score = c.rrf_score * time_decay(c.published_at, as_of, self.half_life_days)
        return sorted(by_id.values(), key=lambda c: -(c.decayed_score or 0.0))
