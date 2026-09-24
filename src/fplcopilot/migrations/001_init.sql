-- 001_init.sql — новостной корпус и снимки статусов игроков из FPL API.
-- Файл идемпотентен: применяется scripts/migrate.py при каждом запуске.

CREATE EXTENSION IF NOT EXISTS vector;

-- Статьи/новости из всех источников (RSS, Google News, FPL API).
-- players/teams — FPL id, найденные детерминированным матчером (rag/entity_matcher.py).
CREATE TABLE IF NOT EXISTS news_articles (
    id            bigserial   PRIMARY KEY,
    source        text        NOT NULL,
    url           text        NOT NULL UNIQUE,
    title         text        NOT NULL,
    summary       text,
    content       text,
    published_at  timestamptz NOT NULL,
    fetched_at    timestamptz NOT NULL DEFAULT now(),
    content_hash  text,
    players       int[]       NOT NULL DEFAULT '{}',
    teams         int[]       NOT NULL DEFAULT '{}',
    raw           jsonb
);

CREATE INDEX IF NOT EXISTS news_articles_published_at_idx ON news_articles (published_at DESC);
CREATE INDEX IF NOT EXISTS news_articles_source_idx       ON news_articles (source);
CREATE INDEX IF NOT EXISTS news_articles_content_hash_idx ON news_articles (content_hash);
CREATE INDEX IF NOT EXISTS news_articles_players_gin      ON news_articles USING gin (players);
CREATE INDEX IF NOT EXISTS news_articles_teams_gin        ON news_articles USING gin (teams);

-- История статусов игрока из FPL API: строка на каждое изменение
-- (status, chance_next, news, news_added), включая возврат в строй. Дедуп — в коде
-- (сравнение с последним снимком), см. 002_status_transitions.sql.
CREATE TABLE IF NOT EXISTS player_status_snapshots (
    id           bigserial   PRIMARY KEY,
    player_id    int         NOT NULL,
    gw           int,
    status       text        NOT NULL,
    chance_next  int,
    news         text        NOT NULL DEFAULT '',
    news_added   timestamptz,
    snapshot_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS player_status_snapshots_player_idx
    ON player_status_snapshots (player_id, snapshot_at DESC);
CREATE INDEX IF NOT EXISTS player_status_snapshots_news_added_idx
    ON player_status_snapshots (news_added DESC);

-- Чанки с эмбеддингами — в 003_news_chunks.sql, сигналы игроков — в 004_player_signals.sql.
