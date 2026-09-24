"""Retrieval v2 (rag/retrieve.py): RetrievalConfig, per-article cap, date-aware rerank, last_candidates.
Игрушечные данные, без БД и сети."""

from datetime import UTC, datetime, timedelta

import pytest

from fplcopilot.rag.retrieve import (
    RETRIEVAL_V1,
    RETRIEVAL_V2,
    CorpusDoc,
    NoopReranker,
    RetrievalConfig,
    RetrievedChunk,
    Retriever,
    apply_article_cap,
    apply_date_aware_rerank,
    date_multiplier,
    matches_entities,
    retrieval_config,
    time_decay,
    tokenize_bm25,
)

AS_OF = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)


def chunk(cid: int, article: int, text: str = "x", *, days_ago: float = 0, rerank=None, players=()):
    return RetrievedChunk(
        chunk_id=cid,
        article_id=article,
        text=text,
        source="bbc",
        url=f"https://x/{article}",
        title=text.split("\n")[0],
        published_at=AS_OF - timedelta(days=days_ago),
        players=list(players),
        rerank_score=rerank,
    )


# ---------- конфигурация ----------


def test_retrieval_config_versions():
    assert RETRIEVAL_V1.version == "v1" and RETRIEVAL_V1.per_article_cap is None
    assert not RETRIEVAL_V1.date_aware_rerank
    assert RETRIEVAL_V2.version == "v2" and RETRIEVAL_V2.per_article_cap == 2
    assert RETRIEVAL_V2.date_aware_rerank and RETRIEVAL_V2.candidates == 30
    assert RetrievalConfig(per_article_cap=3).version == "custom"
    assert retrieval_config("v1", candidates=15) == RetrievalConfig(
        per_article_cap=None, date_aware_rerank=False, candidates=15
    )
    assert retrieval_config("v2").version == "v2"
    with pytest.raises(ValueError, match="retrieval version"):
        retrieval_config("v9")


# ---------- per-article cap ----------


def test_article_cap_limits_chunks_per_article_and_fills_from_remainder():
    ranked = [
        chunk(1, 100), chunk(2, 100), chunk(3, 100), chunk(4, 100),  # 4 чанка одной статьи
        chunk(5, 200), chunk(6, 300), chunk(7, 200), chunk(8, 200), chunk(9, 400),
    ]  # fmt: skip
    top = apply_article_cap(ranked, k=6, cap=2)
    assert [c.chunk_id for c in top] == [1, 2, 5, 6, 7, 9]  # 3, 4, 8 пропущены; порядок сохранён
    assert max(sum(1 for c in top if c.article_id == a) for a in (100, 200)) == 2
    # без cap — просто top-k
    assert [c.chunk_id for c in apply_article_cap(ranked, k=3, cap=None)] == [1, 2, 3]
    # нехватка после cap -> добор пропущенными по порядку
    top = apply_article_cap(ranked[:4] + [chunk(5, 200)], k=4, cap=1)
    assert [c.chunk_id for c in top] == [1, 5, 2, 3]
    # k больше числа чанков — вернуть всё
    assert len(apply_article_cap(ranked, k=50, cap=2)) == len(ranked)
    assert apply_article_cap([], k=5, cap=2) == []


def test_article_cap_applied_in_retriever_dense_and_hybrid_rerank():
    docs = [
        chunk(1, 100, "Saka injury\nSaka trained.", players=[12]),
        chunk(2, 100, "Saka injury\nSaka trained again.", players=[12]),
        chunk(3, 100, "Saka injury\nSaka trained a third time.", players=[12]),
        chunk(4, 200, "Saka doubt\nSaka is a doubt.", players=[12]),
        chunk(5, 300, "Arsenal squad\nArteta on the squad.", players=[]),
    ]
    store = _FakeStore(docs)
    r = Retriever(
        store=store,  # type: ignore[arg-type]
        embed=lambda q: tokenize_bm25(q),
        reranker=NoopReranker(),
        half_life_days=7,
        config=RetrievalConfig(per_article_cap=1, date_aware_rerank=False, candidates=10),
    )
    for mode in ("dense", "hybrid_rerank"):
        res = r.search("Saka injury trained", player_ids=[12], as_of=AS_OF, k=3, mode=mode)
        assert len({c.article_id for c in res}) == 3, mode  # 3 разных статьи
        assert len(r.last_candidates) >= 4  # кандидаты до cap сохранены
    # v1: без cap — статья 100 занимает несколько мест
    v1 = r.search(
        "Saka injury trained", player_ids=[12], as_of=AS_OF, k=3, mode="dense", config=RETRIEVAL_V1
    )
    assert sum(1 for c in v1 if c.article_id == 100) >= 2


# ---------- date-aware rerank ----------


def test_date_multiplier_has_floor_and_decays():
    assert date_multiplier(AS_OF, AS_OF, 7, 0.5) == 1.0
    assert date_multiplier(AS_OF - timedelta(days=7), AS_OF, 7, 0.5) == pytest.approx(0.75)
    assert date_multiplier(AS_OF - timedelta(days=700), AS_OF, 7, 0.5) == pytest.approx(
        0.5, abs=1e-6
    )
    assert date_multiplier(AS_OF - timedelta(days=700), AS_OF, 7, 0.0) == pytest.approx(
        time_decay(AS_OF - timedelta(days=700), AS_OF, 7)
    )


def test_date_aware_rerank_prefers_fresh_item_over_slightly_better_old_headline():
    august = chunk(1, 1, "August presser\nold", days_ago=21, rerank=0.95)
    fresh = chunk(2, 2, "Fresh update\nnew", days_ago=1, rerank=0.80)
    fresh_weak = chunk(3, 3, "Fresh noise\nmeh", days_ago=0, rerank=0.30)
    out = apply_date_aware_rerank([august, fresh, fresh_weak], AS_OF, 7, 0.5)
    assert [c.chunk_id for c in out] == [2, 1, 3]  # свежий релевантный обгоняет августовский
    by_id = {c.chunk_id: c for c in out}
    assert by_id[1].rerank_score == 0.95  # сырая оценка не тронута
    assert by_id[1].date_aware_score == pytest.approx(
        0.95 * date_multiplier(august.published_at, AS_OF, 7, 0.5)
    )
    assert by_id[1].date_aware_score < by_id[2].date_aware_score
    # но сильно более релевантное старое не хоронится под слабым свежим (пол 0.5)
    assert by_id[1].date_aware_score > by_id[3].date_aware_score
    assert all(c.final_score == c.date_aware_score for c in out)


def test_retriever_applies_date_aware_only_in_hybrid_rerank_and_only_when_enabled():
    docs = [
        chunk(
            1,
            1,
            "Saka injury update\nSaka doubt for the weekend (August).",
            days_ago=25,
            players=[12],
        ),
        chunk(
            2,
            2,
            "Saka injury update\nSaka doubt for the weekend (fresh).",
            days_ago=1,
            players=[12],
        ),
    ]

    class ReverseFreshReranker:  # ставит августовский чуть выше свежего
        name = "toy"

        def rerank(self, query, chunks):
            out = []
            for c in chunks:
                c.rerank_score = 0.95 if c.chunk_id == 1 else 0.90
                out.append(c)
            return sorted(out, key=lambda c: -c.rerank_score)

    store = _FakeStore(docs)
    r = Retriever(
        store=store,  # type: ignore[arg-type]
        embed=lambda q: tokenize_bm25(q),
        reranker=ReverseFreshReranker(),
        half_life_days=7,
        config=RETRIEVAL_V2,
    )
    v2 = r.search("Saka injury doubt", player_ids=[12], as_of=AS_OF, k=2, mode="hybrid_rerank")
    assert [c.chunk_id for c in v2] == [2, 1]  # date-aware: свежий первый
    assert v2[0].date_aware_score is not None and v2[0].rerank_score == 0.90
    v1 = r.search(
        "Saka injury doubt",
        player_ids=[12],
        as_of=AS_OF,
        k=2,
        mode="hybrid_rerank",
        config=RETRIEVAL_V1,
    )
    assert [c.chunk_id for c in v1] == [1, 2]  # v1: чистый порядок cross-encoder
    assert all(c.date_aware_score is None for c in v1)
    dense = r.search("Saka injury doubt", player_ids=[12], as_of=AS_OF, k=2, mode="dense")
    assert all(c.date_aware_score is None and c.rerank_score is None for c in dense)


# ---------- FakeStore (как в test_retrieve.py) ----------


class _FakeStore:
    def __init__(self, docs):
        self.docs = docs

    def dense(self, qvec, *, as_of, player_ids, team_ids, n):
        out = []
        for d in self.docs:
            if d.published_at > as_of:
                continue
            if (player_ids or team_ids) and not matches_entities(d, player_ids, team_ids):
                continue
            c = RetrievedChunk(**d.__dict__)
            c.dense_score = len(set(qvec) & set(tokenize_bm25(d.text))) / 10
            out.append(c)
        return sorted(out, key=lambda c: -c.dense_score)[:n]

    def corpus(self):
        return (len(self.docs), len(self.docs)), [
            CorpusDoc(chunk=d, tokens=tokenize_bm25(d.text)) for d in self.docs
        ]
