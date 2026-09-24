-- 004_player_signals.sql — структурированные сигналы о доступности игрока (rag/extract.py).
-- Одна строка на вызов extract_signal: что решил LLM после retrieval + детерминированной
-- пост-валидации. Храним всё, что нужно для evals/A-B: модель, версию промпта, режим поиска,
-- id использованных чанков, число исправлений валидатора.
-- TODO(step 5+): team_signals (командный уровень: ротация, пресс-конференции) — пока не нужны.

CREATE TABLE IF NOT EXISTS player_signals (
    id                  bigserial    PRIMARY KEY,
    player_id           int          NOT NULL,
    as_of               timestamptz  NOT NULL,
    availability        text         NOT NULL,
    start_probability   real         NOT NULL,
    expected_minutes    int          NOT NULL,
    rotation_risk       text         NOT NULL,
    return_gw           int,
    confidence          real         NOT NULL,
    summary             text         NOT NULL,
    evidence            jsonb        NOT NULL DEFAULT '[]',
    fpl_status          text         NOT NULL,
    fpl_chance_next     int,
    model               text         NOT NULL,
    prompt_version      text         NOT NULL,
    mode                text         NOT NULL,
    retrieved_chunk_ids int[]        NOT NULL DEFAULT '{}',
    validation_fixes    int          NOT NULL DEFAULT 0,
    created_at          timestamptz  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS player_signals_player_idx ON player_signals (player_id, as_of DESC);
CREATE INDEX IF NOT EXISTS player_signals_created_idx ON player_signals (created_at DESC);
