"""KBRetriever без БД/сети: cap на документ, строгий фильтр тегов в dense и bm25, RRF без
time-decay, режимы, протокол reranker (FakeKBStore имитирует SQL-слой)."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from fplcopilot.rag.kb.retrieve import (
    KB_MODES,
    KBChunk,
    KBCorpusDoc,
    KBRetriever,
    chunk_to_dict,
    format_results,
    matches_tags,
)
from fplcopilot.rag.retrieve import NoopReranker, apply_article_cap, tokenize_bm25

NOW = datetime(2026, 9, 17, tzinfo=UTC)


def kb_chunk(cid: int, doc: int, text: str, *, tags=("chips",), years_old: int = 0) -> KBChunk:
    return KBChunk(
        chunk_id=cid,
        article_id=doc,
        text=text,
        source=f"src{doc}",
        url=f"https://example.com/{doc}",
        title=f"Doc {doc}",
        published_at=NOW - timedelta(days=365 * years_old),
        tags=list(tags),
    )


CORPUS = [
    # документ 1: четыре чанка про wildcard, все близки к запросу (cap должен оставить два)
    kb_chunk(1, 1, "Guide\nWildcard\nWhen should I use my wildcard? Use the wildcard for a squad overhaul."),
    kb_chunk(2, 1, "Guide\nWildcard\nWhen should I use my wildcard? Before the wildcard expires in gameweek 19."),
    kb_chunk(3, 1, "Guide\nWildcard\nWhen should I use my wildcard? Never use the wildcard on a whim."),
    kb_chunk(4, 1, "Guide\nWildcard\nWhen should I use my wildcard? Use it around the international break."),
    # документ 2: правила, старый текст — decay отсутствует
    kb_chunk(5, 2, "Rules\nChips\nThe first Wildcard expires at the gameweek 19 deadline.", tags=("rules", "chips"), years_old=6),
    # документ 3 и 5: цены, другой тег (несколько чанков, чтобы IDF BM25 не вырождался в 0)
    kb_chunk(6, 3, "Prices\nSelling\nYou keep half the profit when selling a player.", tags=("prices",)),
    kb_chunk(7, 3, "Prices\nTiming\nPrice changes happen overnight based on transfers.", tags=("prices",)),
    kb_chunk(9, 5, "Team value\nGrowth\nTeam value grows slowly across the season for active managers.", tags=("prices",)),
    kb_chunk(10, 5, "Team value\nPre-season\nPre-season rises reward early registration and patience.", tags=("prices",)),
    # документ 4: капитанство
    kb_chunk(8, 4, "Captaincy\nEO\nEffective ownership matters for the captain pick and rank.", tags=("captaincy", "rank")),
]  # fmt: skip


class FakeKBStore:
    """SQL-слой понарошку: фильтр тегов, косинус = доля общих токенов."""

    def __init__(self, docs: list[KBChunk], *, honour_tags: bool = True) -> None:
        self.docs = docs
        self.honour_tags = honour_tags
        self.dense_calls: list[dict] = []

    def dense(self, qvec, *, tags, n):
        self.dense_calls.append({"tags": tags, "n": n})
        out = []
        for d in self.docs:
            if self.honour_tags and not matches_tags(d, tags):
                continue
            c = replace(d)
            c.dense_score = len(set(qvec) & set(tokenize_bm25(d.text))) / 10
            out.append(c)
        return sorted(out, key=lambda c: -c.dense_score)[:n]

    def corpus(self):
        return (len(self.docs), len(self.docs)), [
            KBCorpusDoc(chunk=d, tokens=tokenize_bm25(d.text)) for d in self.docs
        ]


def make(store: FakeKBStore, **kw) -> KBRetriever:
    return KBRetriever(
        store=store,  # type: ignore[arg-type]
        embed=lambda q: tokenize_bm25(q),
        reranker=NoopReranker(),
        candidates=10,
        **kw,
    )


def test_per_doc_cap_limits_one_document_and_fills_from_others():
    r = make(FakeKBStore(CORPUS))
    res = r.search("when should I use my wildcard", k=4, mode="hybrid_rerank")
    ids = [c.chunk_id for c in res]
    assert len(res) == 4
    assert sum(1 for c in res if c.article_id == 1) == 2  # не больше двух чанков документа 1
    assert 5 in ids  # правило про wildcard из документа 2 добралось в top-k
    # без cap (per_doc_cap=0) документ 1 занял бы весь top-k
    uncapped = r.search("when should I use my wildcard", k=4, mode="hybrid_rerank", per_doc_cap=0)
    assert sum(1 for c in uncapped if c.article_id == 1) == 4
    assert len(r.last_candidates) >= len(res)  # кандидаты до cap сохранены


def test_apply_article_cap_works_on_kb_chunks_directly():
    capped = apply_article_cap(CORPUS[:5], k=3, cap=1)
    assert [c.article_id for c in capped] == [1, 2, 1]  # добор пропущенными по порядку


@pytest.mark.parametrize("mode", KB_MODES)
def test_tag_filter_is_strict_in_every_mode(mode):
    store = FakeKBStore(CORPUS)
    r = make(store)
    res = r.search("wildcard price selling profit", tags=["prices"], k=5, mode=mode)
    assert res, mode
    assert all("prices" in c.tags for c in res), mode
    if mode != "bm25":
        assert store.dense_calls[-1]["tags"] == ["prices"]


def test_tag_filter_is_enforced_in_python_even_if_store_ignores_it():
    r = make(FakeKBStore(CORPUS, honour_tags=False))
    res = r.search("wildcard price selling profit", tags=["prices"], k=5, mode="hybrid")
    assert res and all("prices" in c.tags for c in res)


def test_hybrid_has_no_time_decay_old_rules_doc_keeps_its_rrf_score():
    r = make(FakeKBStore(CORPUS))
    res = r.search("first wildcard expires gameweek 19 deadline", k=5, mode="hybrid")
    old = next(c for c in res if c.chunk_id == 5)  # 6 лет «назад»
    assert old.rrf_score is not None and old.decayed_score is None and old.date_aware_score is None
    assert old.final_score == old.rrf_score
    assert res[0].chunk_id in (5, 2)  # старый официальный чанк конкурирует на равных


def test_modes_fill_expected_scores_and_timings():
    r = make(FakeKBStore(CORPUS))
    dense = r.search("effective ownership captain", k=2, mode="dense")
    assert dense[0].chunk_id == 8 and dense[0].rrf_score is None and dense[0].bm25_score is None
    assert "dense_ms" in r.last_timings and "bm25_ms" not in r.last_timings
    bm25 = r.search("effective ownership captain", k=2, mode="bm25")
    assert bm25[0].chunk_id == 8 and bm25[0].dense_score is None
    assert "embed_ms" not in r.last_timings
    hybrid = r.search("effective ownership captain", k=2, mode="hybrid")
    assert hybrid[0].rrf_score is not None
    with pytest.raises(ValueError, match="mode"):
        r.search("x", mode="fancy")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="пустой"):
        r.search("   ")


def test_custom_reranker_only_in_hybrid_rerank():
    class Reverse:
        name = "reverse"

        def rerank(self, query, chunks):
            out = list(reversed(chunks))
            for i, c in enumerate(out):
                c.rerank_score = float(len(out) - i)
            return out

    store = FakeKBStore(CORPUS)
    plain = make(store).search("wildcard", k=3, mode="hybrid", per_doc_cap=0)
    rer = KBRetriever(
        store=store,  # type: ignore[arg-type]
        embed=lambda q: tokenize_bm25(q),
        reranker=Reverse(),
        candidates=10,
    ).search("wildcard", k=3, mode="hybrid_rerank", per_doc_cap=0)
    assert [c.chunk_id for c in rer] != [c.chunk_id for c in plain]
    assert all(c.rerank_score is not None for c in rer)


def test_kb_chunk_helpers_and_formatting():
    c = CORPUS[4]
    assert c.doc_id == 2 and c.is_rules
    d = chunk_to_dict(replace(c, rerank_score=0.9))
    assert d["doc_id"] == 2 and d["score"] == 0.9 and d["tags"] == ["rules", "chips"]
    assert d["published_at"].startswith("2020-")
    table = format_results([replace(c, dense_score=0.5, rrf_score=0.03)])
    assert "Doc 2 [rules,chips]" in table and "0.500" in table
