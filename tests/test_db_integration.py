"""Интеграция с Postgres: идемпотентность вставок. Нужен `docker compose up -d db` + migrate.

Запуск: uv run pytest -m db. Без доступной БД тесты аккуратно скипаются.
"""

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import text

from fplcopilot.data.schemas import Player
from fplcopilot.db import ping, session_scope
from fplcopilot.rag.ingest import (
    Article,
    article_exists,
    insert_article,
    last_snapshots,
    record_status,
)

pytestmark = pytest.mark.db


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    if not ping():
        pytest.skip(
            "Postgres недоступен (docker compose up -d db && uv run python scripts/migrate.py)"
        )
    with session_scope() as s:
        has_tables = s.execute(text("SELECT to_regclass('news_articles')")).scalar()
    if has_tables is None:
        pytest.skip("таблицы не созданы: uv run python scripts/migrate.py")


def test_article_insert_is_idempotent():
    url = f"test://article/{uuid.uuid4()}"
    art = Article(
        source="test",
        url=url,
        title="Saka fit for GW5",
        summary="short",
        content="Bukayo Saka trained fully.",
        published_at=datetime(2026, 9, 16, 12, 0, tzinfo=UTC),
        players=[12],
        teams=[1],
        raw={"link": url, "fulltext": False},
    )
    with session_scope() as s:
        try:
            assert not article_exists(s, url)
            assert insert_article(s, art) is True
            assert insert_article(s, art) is False  # повтор — no-op
            assert article_exists(s, url)
            n, players, published = s.execute(
                text(
                    "SELECT count(*), max(players), max(published_at) FROM news_articles WHERE url=:u"
                ),
                {"u": url},
            ).one()
            assert n == 1
            assert players == [12]
            assert published == art.published_at  # timestamptz вернулся aware и без сдвига
        finally:
            s.execute(text("DELETE FROM news_articles WHERE url = :u"), {"u": url})
            s.commit()


def test_status_transitions_recorded_only_on_change():
    pid = 900_000 + uuid.uuid4().int % 90_000  # заведомо несуществующий игрок
    injured = Player.model_validate(
        {
            "id": pid,
            "web_name": "Test",
            "team": 1,
            "element_type": 2,
            "now_cost": 40,
            "status": "d",
            "chance_of_playing_next_round": 75,
            "news": "Knock - 75% chance of playing",
            "news_added": "2026-09-15T19:30:09.494382Z",  # микросекунды должны пережить round-trip
        }
    )
    recovered = injured.model_copy(
        update={"status": "a", "chance_of_playing_next_round": None, "news": "", "news_added": None}
    )

    def run(p: Player) -> bool:
        return record_status(s, p, 5, last_snapshots(s).get(pid))

    with session_scope() as s:
        try:
            assert run(injured) is True
            assert run(injured) is False  # без изменений — ничего
            assert run(recovered) is True  # выздоровление фиксируется
            assert run(recovered) is False
            assert run(injured) is True  # повторная травма с тем же текстом — новое изменение
            rows = s.execute(
                text(
                    "SELECT status, news FROM player_status_snapshots "
                    "WHERE player_id=:p ORDER BY id"
                ),
                {"p": pid},
            ).all()
            assert [r[0] for r in rows] == ["d", "a", "d"]
            assert rows[1][1] == ""
        finally:
            s.execute(text("DELETE FROM player_status_snapshots WHERE player_id = :p"), {"p": pid})
            s.commit()
