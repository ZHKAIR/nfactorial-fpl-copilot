-- 009_team_news.sql — командный дайджест новостей клуба (rag/team_news.py, docs/rag.md «Новости клуба»).
-- Одна строка на вызов extract_team_news: отсутствующие / возвращающиеся игроки, слова тренера о
-- ротации и составе, форма / контекст команды — каждое утверждение с verbatim-цитатой, проверенной
-- кодом. Это контекст клуба, а НЕ доказательство доступности конкретного игрока: в player_signals
-- эти цитаты не попадают.
-- Файл идемпотентен (scripts/migrate.py применяет все миграции при каждом запуске).

CREATE TABLE IF NOT EXISTS team_news_digests (
    id                  bigserial    PRIMARY KEY,
    team_id             int          NOT NULL,
    as_of               timestamptz  NOT NULL,
    summary             text         NOT NULL,
    items               jsonb        NOT NULL DEFAULT '[]',
    model               text         NOT NULL,
    prompt_version      text         NOT NULL,
    mode                text         NOT NULL,
    retrieved_chunk_ids int[]        NOT NULL DEFAULT '{}',
    validation_fixes    int          NOT NULL DEFAULT 0,
    abstained           boolean      NOT NULL DEFAULT false,
    created_at          timestamptz  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS team_news_digests_team_idx ON team_news_digests (team_id, as_of DESC);
