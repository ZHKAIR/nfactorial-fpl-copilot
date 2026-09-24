"""Воспроизводимость прошлого as_of (docs/EVALS.md §4.5): FPL-prior по времени наблюдения
снимка, срез корпуса по времени сбора (corpus_cutoff), CLI-флаг evals. Без БД и сети."""

from datetime import UTC, datetime, timedelta

import pytest

from evals.run_rag import cutoff_for, cutoff_tag, parse_corpus_cutoff
from fplcopilot.rag.extract import FPLPrior, StatusSnapshot, prior_from_snapshots
from fplcopilot.rag.retrieve import (
    CorpusDoc,
    NoopReranker,
    RetrievedChunk,
    Retriever,
    matches_entities,
    tokenize_bm25,
)

AS_OF = datetime(2026, 9, 17, 9, 0, tzinfo=UTC)
FIRST_SEEN = datetime(2026, 9, 17, 7, 34, tzinfo=UTC)  # первый снимок ингеста


def snap(status, chance, news, added, seen):
    return StatusSnapshot(status, chance, news, added, seen)


# ---------- FPL-prior: время события = snapshot_at ----------

CAICEDO = [
    snap("i", 0, "Calf injury - Expected back 18 Sep", datetime(2026, 8, 30, 16, tzinfo=UTC), FIRST_SEEN),
    # FPL сменил текст и chance, news_added оставил 30.08; снимок сделан после as_of
    snap("d", 50, "Calf injury - 50% chance of playing", datetime(2026, 8, 30, 16, tzinfo=UTC),
         datetime(2026, 9, 17, 13, 53, tzinfo=UTC)),
]  # fmt: skip
DOKU = [
    snap("i", 0, "Calf injury - Expected back 20 Sep", datetime(2026, 8, 18, 17, tzinfo=UTC), FIRST_SEEN),
    snap("a", 100, "", datetime(2026, 8, 18, 17, tzinfo=UTC), datetime(2026, 9, 22, 15, 8, tzinfo=UTC)),
]  # fmt: skip


def test_prior_ignores_snapshot_taken_after_as_of_with_old_news_added():
    p = prior_from_snapshots(CAICEDO, AS_OF)
    assert (p.status, p.chance_next, p.news) == ("i", 0, "Calf injury - Expected back 18 Sep")


def test_prior_status_cleared_later_does_not_leak_into_past():
    assert prior_from_snapshots(DOKU, AS_OF).status == "i"
    assert prior_from_snapshots(DOKU, datetime(2026, 9, 23, tzinfo=UTC)).status == "a"


def test_prior_takes_latest_snapshot_seen_by_as_of():
    later = datetime(2026, 9, 18, tzinfo=UTC)
    p = prior_from_snapshots(CAICEDO, later)
    assert (p.status, p.chance_next) == ("d", 50)


def test_prior_before_first_observation_uses_first_snapshot_if_news_older():
    p = prior_from_snapshots(CAICEDO, datetime(2026, 9, 10, tzinfo=UTC))
    assert (p.status, p.news) == ("i", "Calf injury - Expected back 18 Sep")
    # новость опубликована позже as_of -> ничего не знаем, «доступен без новостей»
    assert prior_from_snapshots(CAICEDO, datetime(2026, 8, 20, tzinfo=UTC)) == FPLPrior(
        "a", None, "", None
    )
    assert prior_from_snapshots([], AS_OF) == FPLPrior("a", None, "", None)


def test_prior_known_before_hides_snapshots_collected_later():
    assert prior_from_snapshots(CAICEDO, AS_OF, known_before=FIRST_SEEN - timedelta(hours=1)) == (
        FPLPrior("a", None, "", None)
    )
    p = prior_from_snapshots(CAICEDO, datetime(2026, 9, 18, tzinfo=UTC), known_before=AS_OF)
    assert (p.status, p.chance_next) == ("i", 0)


# ---------- corpus_cutoff: статьи, собранные позже, не видны ----------


def _chunk(cid, text, *, hours_ago, players=(7,)):
    return RetrievedChunk(
        chunk_id=cid,
        article_id=cid,
        text=text,
        source="ffscout",
        url=f"https://x/{cid}",
        title=text,
        published_at=AS_OF - timedelta(hours=hours_ago),
        players=list(players),
    )


class _Store:
    """published_at <= as_of у обоих; чанк 2 собран через 8 ч после as_of (backfill)."""

    def __init__(self):
        self.docs = [
            (_chunk(1, "Saka injury update training", hours_ago=30), AS_OF - timedelta(hours=29)),
            (_chunk(2, "Saka injury knock doubt", hours_ago=7), AS_OF + timedelta(hours=8)),
            (_chunk(3, "Everton press conference", hours_ago=5, players=()), AS_OF),
            (_chunk(4, "Villa price changes", hours_ago=5, players=()), AS_OF),
            (_chunk(5, "Leeds transfer window verdict", hours_ago=5, players=()), AS_OF),
            (_chunk(6, "Chelsea clean sheet odds", hours_ago=5, players=()), AS_OF),
        ]
        self.dense_kwargs: list[dict] = []

    def dense(self, qvec, *, as_of, player_ids, team_ids, n, **kw):
        self.dense_kwargs.append(kw)
        cutoff = kw.get("fetched_before")
        out = []
        for c, fetched in self.docs:
            if c.published_at > as_of or (cutoff is not None and fetched > cutoff):
                continue
            if (player_ids or team_ids) and not matches_entities(c, player_ids, team_ids):
                continue
            hit = RetrievedChunk(**c.__dict__)
            hit.dense_score = 0.5
            out.append(hit)
        return out[:n]

    def corpus(self):
        return (6, 6), [
            CorpusDoc(chunk=c, tokens=tokenize_bm25(c.text), fetched_at=f) for c, f in self.docs
        ]


def _retriever(store):
    return Retriever(store=store, embed=lambda q: [0.0], reranker=NoopReranker(), half_life_days=7)


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid", "hybrid_rerank"])
def test_corpus_cutoff_hides_late_fetched_articles(mode):
    store = _Store()
    r = _retriever(store)
    everything = r.search("Saka injury", player_ids=[7], as_of=AS_OF, k=8, mode=mode)
    assert {1, 2} <= {c.chunk_id for c in everything}
    replay = r.search(
        "Saka injury", player_ids=[7], as_of=AS_OF, k=8, mode=mode, corpus_cutoff=AS_OF
    )
    ids = {c.chunk_id for c in replay}
    assert 1 in ids and 2 not in ids


def test_no_cutoff_keeps_production_store_call_unchanged():
    store = _Store()
    _retriever(store).search("Saka injury", player_ids=[7], as_of=AS_OF, k=8, mode="dense")
    assert store.dense_kwargs and all(kw == {} for kw in store.dense_kwargs)


def test_corpus_cutoff_must_be_timezone_aware():
    with pytest.raises(ValueError, match="corpus_cutoff"):
        naive = datetime.fromisoformat("2026-09-17T00:00:00")
        _retriever(_Store()).search("x", as_of=AS_OF, corpus_cutoff=naive)


# ---------- CLI evals ----------


def test_parse_corpus_cutoff_values():
    assert parse_corpus_cutoff("none") is None
    assert parse_corpus_cutoff("as_of") == "as_of"
    ts = parse_corpus_cutoff("2026-09-17T10:14:17Z")
    assert ts == datetime(2026, 9, 17, 10, 14, 17, tzinfo=UTC)
    assert parse_corpus_cutoff("2026-09-17T10:14") == datetime(2026, 9, 17, 10, 14, tzinfo=UTC)
    assert cutoff_for("as_of", AS_OF) == AS_OF and cutoff_for(ts, AS_OF) == ts
    assert cutoff_for(None, AS_OF) is None
    assert cutoff_tag(ts) == "cut0917T1014" and cutoff_tag("as_of") == "cutasof"
    assert cutoff_tag(None) == ""
