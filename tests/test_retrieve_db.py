"""Интеграция с Postgres/pgvector: документ из будущего не извлекается ни одним режимом.

Нужен `docker compose up -d db` + migrate; без БД тесты скипаются. OpenAI не нужен:
эмбеддинги подменяются фиксированным вектором, reranker — NoopReranker.
"""

import json
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from fplcopilot.config import settings
from fplcopilot.db import ping, session_scope
from fplcopilot.rag.extract import Evidence, PlayerSignal, save_signal
from fplcopilot.rag.retrieve import MODES, ChunkStore, NoopReranker, Retriever

pytestmark = pytest.mark.db

NOW = datetime.now(UTC).replace(microsecond=0)
MARKER = f"zzq{uuid.uuid4().hex[:10]}"  # уникальный токен, чтобы BM25 находил только наши чанки


def unit_vector(i: int) -> list[float]:
    v = [0.0] * settings.rag_embedding_dims
    v[i] = 1.0
    return v


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    if not ping():
        pytest.skip(
            "Postgres недоступен (docker compose up -d db && uv run python scripts/migrate.py)"
        )
    with session_scope() as s:
        if s.execute(text("SELECT to_regclass('news_chunks')")).scalar() is None:
            pytest.skip("таблица news_chunks не создана: uv run python scripts/migrate.py")


@pytest.fixture(scope="module")
def articles():
    """Две статьи с одинаковым «эмбеддингом»: одна в прошлом, одна в будущем относительно NOW."""
    rows = {
        "past": (NOW - timedelta(days=1), f"{MARKER} past article: Saka trained fully"),
        "future": (NOW + timedelta(days=1), f"{MARKER} future article: Saka ruled out"),
    }
    ids: dict[str, int] = {}
    with session_scope() as s:
        for key, (published, title) in rows.items():
            art_id = s.execute(
                text(
                    "INSERT INTO news_articles (source, url, title, content, published_at, players) "
                    "VALUES ('test', :url, :title, :title, :published, ARRAY[12]) RETURNING id"
                ),
                {"url": f"test://{MARKER}/{key}", "title": title, "published": published},
            ).scalar_one()
            s.execute(
                text(
                    "INSERT INTO news_chunks (article_id, chunk_index, text, n_tokens, embedding, "
                    "players, source, published_at) VALUES (:a, 0, :t, 5, CAST(:e AS vector), "
                    "ARRAY[12], 'test', :p)"
                ),
                {"a": art_id, "t": title, "e": json.dumps(unit_vector(0)), "p": published},
            )
            ids[key] = art_id
        s.commit()
    yield ids
    with session_scope() as s:  # каскад удалит чанки
        s.execute(text("DELETE FROM news_articles WHERE url LIKE :u"), {"u": f"test://{MARKER}/%"})
        s.commit()


def retriever() -> Retriever:
    return Retriever(
        store=ChunkStore(),
        embed=lambda _q: unit_vector(0),
        reranker=NoopReranker(),
        candidates=50,
    )


@pytest.mark.parametrize("mode", MODES)
def test_future_article_is_not_retrieved_before_its_publication(articles, mode):
    res = retriever().search(f"{MARKER} Saka", player_ids=[12], as_of=NOW, k=50, mode=mode)
    got = {c.article_id for c in res}
    assert articles["past"] in got, mode  # позитивный контроль: прошлое находится
    assert articles["future"] not in got, mode
    assert all(c.published_at <= NOW for c in res)


def test_same_article_is_retrieved_once_as_of_moves_past_publication(articles):
    later = NOW + timedelta(days=2)
    res = retriever().search(f"{MARKER} Saka", player_ids=[12], as_of=later, k=50, mode="hybrid")
    got = {c.article_id for c in res}
    assert {articles["past"], articles["future"]} <= got


def test_dense_sql_filters_as_of_without_entity_filter(articles):
    store = ChunkStore()
    hits = store.dense(unit_vector(0), as_of=NOW, player_ids=None, team_ids=None, n=5)
    assert hits and hits[0].article_id == articles["past"]  # косинус = 1.0 с нашим вектором
    assert hits[0].dense_score == pytest.approx(1.0)
    assert articles["future"] not in {c.article_id for c in hits}


def test_player_signal_roundtrip(articles):
    sig = PlayerSignal(
        player_id=999_999,
        player_name="Test Player",
        as_of=NOW,
        availability="doubtful",
        start_probability=0.6,
        expected_minutes=60,
        rotation_risk="medium",
        return_gw=None,
        confidence=0.7,
        summary="Test signal.",
        evidence=[
            Evidence(
                chunk_id=1,
                source="test",
                url="test://x",
                published_at=NOW,
                quote="Saka trained fully",
            )
        ],
        fpl_status="d",
        fpl_chance_next=75,
        model="test-model",
        prompt_version="v1",
        retrieved_chunk_ids=[1, 2, 3],
        mode="hybrid_rerank",
        validation_fixes=1,
    )
    row_id = save_signal(sig)
    with session_scope() as s:
        try:
            row = s.execute(
                text(
                    "SELECT availability, start_probability, evidence, retrieved_chunk_ids, as_of, "
                    "validation_fixes FROM player_signals WHERE id = :id"
                ),
                {"id": row_id},
            ).one()
            assert row[0] == "doubtful"
            assert row[1] == pytest.approx(0.6)
            assert row[2][0]["quote"] == "Saka trained fully"
            assert row[3] == [1, 2, 3]
            assert row[4] == NOW
            assert row[5] == 1
        finally:
            s.execute(text("DELETE FROM player_signals WHERE id = :id"), {"id": row_id})
            s.commit()
