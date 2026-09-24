-- 006_plans.sql — снимки планов трансферов (fplcopilot.core.plan).
-- Файл идемпотентен: применяется scripts/migrate.py при каждом запуске.

-- Один снимок = один вызов plan_transfers: весь TransferPlan как jsonb (ходы по турам, лучшие 11,
-- целевой состав, альтернатива с Wildcard, рекомендация, солвер, время). inputs_hash — sha1 от
-- входов (состав, банк, FT, стратегия, горизонт, xPts пула): если хэш совпал, план пересчитывать
-- не нужно; если планы отличаются при разных хэшах — diff_plans объясняет, что поменялось.
CREATE TABLE IF NOT EXISTS plan_snapshots (
    id          bigserial   PRIMARY KEY,
    manager_id  int         NOT NULL,
    gw          int         NOT NULL,
    strategy    text        NOT NULL,
    horizon     int         NOT NULL,
    plan        jsonb       NOT NULL,
    inputs_hash text        NOT NULL DEFAULT '',
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS plan_snapshots_manager_idx
    ON plan_snapshots (manager_id, gw, created_at DESC);
