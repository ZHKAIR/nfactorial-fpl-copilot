"""Retriever без БД и сети: RRF, time-decay, фильтр as_of в dense и bm25, режимы, reranker-протокол.

ChunkStore подменяется FakeStore: он ведёт себя как SQL (фильтрует published_at <= as_of и сущности)
и запоминает, с каким as_of его вызывали, — так проверяем, что Retriever передаёт as_of в оба пути.
Второй тест «ленивого» store, который игнорирует as_of, проверяет защиту в глубину в Python.
"""

import math
from datetime import UTC, datetime, timedelta

import pytest

from fplcopilot.rag.retrieve import (
    CorpusDoc,
    NoopReranker,
    RetrievedChunk,
    Retriever,
    expand_player_query,
    find_player,
    matches_entities,
    rrf_fuse,
    time_decay,
    tokenize_bm25,
)

AS_OF = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)


def chunk(cid: int, text: str, *, days_ago: float = 0, players=(), source="bbc") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        article_id=cid,
        text=text,
        source=source,
        url=f"https://x/{cid}",
        title=text.split("\n")[0],
        published_at=AS_OF - timedelta(days=days_ago),
        players=list(players),
    )


CORPUS = [
    chunk(
        1,
        "Saka injury update\nBukayo Saka trained fully and is fit for GW5.",
        days_ago=1,
        players=[12],
    ),
    chunk(
        2, "Saka ruled out\nSaka has a knee injury and is out for weeks.", days_ago=20, players=[12]
    ),
    chunk(3, "Haaland scores\nErling Haaland scored twice.", days_ago=2, players=[7]),
    chunk(
        4, "FUTURE Saka\nSaka injury: out for the season.", days_ago=-1, players=[12]
    ),  # после as_of
    chunk(5, "Arsenal news\nArteta praised the squad depth.", days_ago=3, players=[]),
    # старые нерелевантные документы: без них корпус слишком мал и IDF BM25 вырождается в 0
    chunk(6, "Palace fixtures\nCrystal Palace face a busy schedule.", days_ago=15, players=[]),
    chunk(7, "Referee appointments\nOfficials confirmed for the weekend.", days_ago=18, players=[]),
]


class FakeStore:
    """Имитирует SQL-слой: фильтр по as_of и сущностям, косинус = число общих слов."""

    def __init__(self, docs: list[RetrievedChunk], *, honour_as_of: bool = True) -> None:
        self.docs = docs
        self.honour_as_of = honour_as_of
        self.dense_calls: list[dict] = []

    def dense(self, qvec, *, as_of, player_ids, team_ids, n):
        self.dense_calls.append({"as_of": as_of, "player_ids": player_ids, "n": n})
        out = []
        for d in self.docs:
            if self.honour_as_of and d.published_at > as_of:
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


def make_retriever(store: FakeStore, **kw) -> Retriever:
    return Retriever(
        store=store,  # type: ignore[arg-type]
        embed=lambda q: tokenize_bm25(q),  # «эмбеддинг» = токены, косинус считает FakeStore
        reranker=NoopReranker(),
        half_life_days=7,
        candidates=10,
        **kw,
    )


# ---------- математика ----------


def test_rrf_fusion_math():
    scores = rrf_fuse([[1, 2, 3], [2, 1]], k=60)
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[2] == pytest.approx(1 / 62 + 1 / 61)
    assert scores[3] == pytest.approx(1 / 63)
    assert scores[1] == pytest.approx(scores[2])  # симметрично
    assert rrf_fuse([[], []]) == {}


def test_time_decay_math():
    assert time_decay(AS_OF, AS_OF, 7) == 1.0
    assert time_decay(AS_OF - timedelta(days=7), AS_OF, 7) == pytest.approx(0.5)
    assert time_decay(AS_OF - timedelta(days=14), AS_OF, 7) == pytest.approx(0.25)
    assert time_decay(AS_OF - timedelta(days=3.5), AS_OF, 7) == pytest.approx(math.sqrt(0.5))
    assert time_decay(AS_OF + timedelta(days=1), AS_OF, 7) == 1.0  # будущее не «усиливаем»
    assert time_decay(AS_OF - timedelta(days=100), AS_OF, 0) == 1.0  # half_life<=0 -> выключено


def test_tokenize_bm25_strips_accents_and_stopwords():
    assert tokenize_bm25("João Pedro is a doubt for the Chelsea game") == [
        "joao", "pedro", "doubt", "chelsea", "game",
    ]  # fmt: skip


# ---------- as_of: оба пути ----------


def test_as_of_is_passed_to_dense_and_applied_in_bm25():
    store = FakeStore(CORPUS)
    r = make_retriever(store)
    res = r.search("Saka injury", player_ids=[12], as_of=AS_OF, k=5, mode="hybrid_rerank")
    ids = [c.chunk_id for c in res]
    assert 4 not in ids  # будущий документ не прошёл ни dense, ни bm25
    assert set(ids) >= {1, 2}
    assert all(call["as_of"] == AS_OF for call in store.dense_calls)
    assert all(c.published_at <= AS_OF for c in res)


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid", "hybrid_rerank"])
def test_future_chunks_are_dropped_even_if_store_ignores_as_of(mode):
    store = FakeStore(CORPUS, honour_as_of=False)
    r = make_retriever(store)
    res = r.search("Saka injury out season", player_ids=[12], as_of=AS_OF, k=5, mode=mode)
    assert res, mode
    assert all(c.published_at <= AS_OF for c in res), mode
    assert 4 not in {c.chunk_id for c in res}


def test_bm25_index_respects_as_of_when_building_eligible_set():
    r = make_retriever(FakeStore(CORPUS))
    earlier = AS_OF - timedelta(days=10)  # до этого опубликованы только чанки 2, 6, 7
    res = r.search("Saka injury", player_ids=[12], as_of=earlier, k=5, mode="bm25")
    assert [c.chunk_id for c in res] == [2]  # чанк 1 (1 день назад) в индекс не попал


def test_naive_as_of_is_rejected():
    r = make_retriever(FakeStore(CORPUS))
    with pytest.raises(ValueError, match="timezone-aware"):
        r.search("x", as_of=datetime(2026, 9, 17, 12, 0), k=3)  # noqa: DTZ001


# ---------- слияние, decay, сущности, режимы ----------


def test_hybrid_scores_are_filled_and_recent_doc_wins_after_decay():
    r = make_retriever(FakeStore(CORPUS))
    res = r.search("Saka injury", player_ids=[12], as_of=AS_OF, k=6, mode="hybrid")
    top = res[0]
    assert top.chunk_id == 1  # свежий (1 день) обгоняет 20-дневный при близком RRF
    assert top.rrf_score is not None and top.decayed_score is not None
    assert top.decayed_score == pytest.approx(
        top.rrf_score * time_decay(top.published_at, AS_OF, 7)
    )
    old = next(c for c in res if c.chunk_id == 2)
    assert old.bm25_score is not None and old.dense_score is not None  # в обоих списках
    assert old.decayed_score < old.rrf_score * 0.2  # 20 дней при half-life 7 -> ~0.14
    # из-за decay 20-дневный документ об игроке уходит ниже свежих нерелевантных
    assert [c.chunk_id for c in res].index(2) > [c.chunk_id for c in res].index(3)


def test_entity_prefilter_first_then_fill_from_rest():
    store = FakeStore(CORPUS)
    r = make_retriever(store)
    res = r.search("news squad injury", player_ids=[7], as_of=AS_OF, k=4, mode="dense")
    assert res[0].chunk_id == 3  # единственный чанк Холанда идёт первым
    assert len(res) > 1  # добор из остального корпуса
    assert store.dense_calls[0]["player_ids"] == [7] and store.dense_calls[1]["player_ids"] is None


def test_dense_mode_has_no_fusion_scores_and_timings_recorded():
    r = make_retriever(FakeStore(CORPUS))
    res = r.search("Saka injury", player_ids=[12], as_of=AS_OF, k=2, mode="dense")
    assert all(c.rrf_score is None and c.bm25_score is None for c in res)
    assert "dense_ms" in r.last_timings and "bm25_ms" not in r.last_timings
    r.search("Saka injury", as_of=AS_OF, k=2, mode="bm25")
    assert "embed_ms" not in r.last_timings and "bm25_ms" in r.last_timings


def test_custom_reranker_protocol_is_used_in_hybrid_rerank_only():
    class ReverseReranker:
        name = "reverse"

        def rerank(self, query, chunks):
            out = list(reversed(chunks))
            for i, c in enumerate(out):
                c.rerank_score = float(len(out) - i)
            return out

    store = FakeStore(CORPUS)
    plain = make_retriever(store).search(
        "Saka injury", player_ids=[12], as_of=AS_OF, k=3, mode="hybrid"
    )
    rer = Retriever(
        store=store,  # type: ignore[arg-type]
        embed=lambda q: tokenize_bm25(q),
        reranker=ReverseReranker(),
        half_life_days=7,
        candidates=10,
    )
    reranked = rer.search("Saka injury", player_ids=[12], as_of=AS_OF, k=3, mode="hybrid_rerank")
    assert [c.chunk_id for c in reranked] != [c.chunk_id for c in plain]
    assert all(c.rerank_score is not None for c in reranked)
    assert rer.last_timings["rerank_ms"] >= 0


def test_query_rewriter_hook():
    store = FakeStore(CORPUS)
    seen = []

    def rewriter(q: str) -> str:
        seen.append(q)
        return "Haaland"

    r = make_retriever(store, query_rewriter=rewriter)
    res = r.search("who is the best striker", as_of=AS_OF, k=1, mode="bm25")
    assert seen == ["who is the best striker"]
    assert res[0].chunk_id == 3


# ---------- расширение запроса и поиск игрока ----------


def _bs():
    from fplcopilot.data.schemas import Bootstrap

    return Bootstrap.model_validate(
        {
            "events": [],
            "teams": [{"id": 1, "name": "Arsenal", "short_name": "ARS"}, {"id": 6, "name": "Chelsea", "short_name": "CHE"}],
            "elements": [
                {"id": 12, "web_name": "Saka", "first_name": "Bukayo", "second_name": "Saka", "team": 1, "element_type": 3, "now_cost": 100},
                {"id": 165, "web_name": "João Pedro", "first_name": "João Pedro", "second_name": "Junqueira de Jesus", "team": 6, "element_type": 4, "now_cost": 78},
                {"id": 20, "web_name": "Silva", "first_name": "Bernardo", "second_name": "Silva", "team": 6, "element_type": 3, "now_cost": 60},
                {"id": 21, "web_name": "A.Silva", "first_name": "António", "second_name": "Silva", "team": 1, "element_type": 2, "now_cost": 45},
            ],
        }
    )  # fmt: skip


def test_expand_player_query_is_deterministic():
    bs = _bs()
    q = expand_player_query(bs.player(165), bs.team(6))
    assert (
        q == "João Pedro Junqueira de Jesus João Pedro Chelsea injury fitness training doubt return"
    )
    assert (
        expand_player_query(bs.player(12), bs.team(1))
        == "Bukayo Saka Saka Arsenal injury fitness training doubt return"
    )


def test_find_player_is_accent_insensitive_and_reports_ambiguity():
    bs = _bs()
    assert find_player(bs, "joao pedro").id == 165
    assert find_player(bs, "SAKA").id == 12
    assert find_player(bs, "Bernardo").id == 20  # уникальная подстрока полного имени
    assert find_player(bs, "Silva").id == 20  # точное web_name побеждает подстроку в "A.Silva"
    with pytest.raises(LookupError, match="неоднозначно"):
        find_player(bs, "silv")  # подстрока у двух игроков, точного совпадения нет
    with pytest.raises(LookupError, match="не найден"):
        find_player(bs, "Mbappe")
