-- 005_core.sql — данные для компонентной модели xPts (fplcopilot.core).
-- Файл идемпотентен: применяется scripts/migrate.py при каждом запуске.

-- Построчная история игрока за матч (element-summary.history), зеркало схемы PlayerGWHistory.
-- Источник истины для per-90 ставок, модели минут и командных xG/xGC (team_match_stats).
-- Строки перезаписываются при повторной синхронизации: бонусы и BPS уточняются после тура.
-- Ключ включает season: id фикстур (1..380) повторяются из сезона в сезон.
CREATE TABLE IF NOT EXISTS player_gw_history (
    id                              bigserial   PRIMARY KEY,
    player_id                       int         NOT NULL,
    season                          text        NOT NULL DEFAULT '2026/27',
    fixture                         int         NOT NULL,
    opponent_team                   int,
    round                           int         NOT NULL,
    kickoff_time                    timestamptz,
    was_home                        boolean     NOT NULL,
    minutes                         int         NOT NULL DEFAULT 0,
    starts                          int         NOT NULL DEFAULT 0,
    total_points                    int         NOT NULL DEFAULT 0,
    goals_scored                    int         NOT NULL DEFAULT 0,
    assists                         int         NOT NULL DEFAULT 0,
    clean_sheets                    int         NOT NULL DEFAULT 0,
    goals_conceded                  int         NOT NULL DEFAULT 0,
    saves                           int         NOT NULL DEFAULT 0,
    bonus                           int         NOT NULL DEFAULT 0,
    bps                             int         NOT NULL DEFAULT 0,
    expected_goals                  real,
    expected_assists                real,
    expected_goals_conceded         real,
    tackles                         int         NOT NULL DEFAULT 0,
    recoveries                      int         NOT NULL DEFAULT 0,
    clearances_blocks_interceptions int         NOT NULL DEFAULT 0,
    defensive_contribution          int         NOT NULL DEFAULT 0,
    yellow_cards                    int         NOT NULL DEFAULT 0,
    red_cards                       int         NOT NULL DEFAULT 0,
    value                           int,
    selected                        int,
    team_h_score                    int,
    team_a_score                    int,
    synced_at                       timestamptz NOT NULL DEFAULT now(),
    UNIQUE (season, player_id, fixture)
);

CREATE INDEX IF NOT EXISTS player_gw_history_round_idx  ON player_gw_history (season, round);
CREATE INDEX IF NOT EXISTS player_gw_history_player_idx ON player_gw_history (player_id, round);

-- Итоги прошлых сезонов игрока (element-summary.history_past): приор для per-90 ставок.
-- Статистика DefCon (tackles/recoveries/CBI) заполнена только с сезона 2025/26.
CREATE TABLE IF NOT EXISTS player_season_history (
    id                              bigserial   PRIMARY KEY,
    player_id                       int         NOT NULL,
    season_name                     text        NOT NULL,
    element_code                    int,
    start_cost                      int,
    end_cost                        int,
    total_points                    int         NOT NULL DEFAULT 0,
    minutes                         int         NOT NULL DEFAULT 0,
    starts                          int         NOT NULL DEFAULT 0,
    goals_scored                    int         NOT NULL DEFAULT 0,
    assists                         int         NOT NULL DEFAULT 0,
    clean_sheets                    int         NOT NULL DEFAULT 0,
    goals_conceded                  int         NOT NULL DEFAULT 0,
    saves                           int         NOT NULL DEFAULT 0,
    bonus                           int         NOT NULL DEFAULT 0,
    bps                             int         NOT NULL DEFAULT 0,
    expected_goals                  real,
    expected_assists                real,
    expected_goals_conceded         real,
    tackles                         int         NOT NULL DEFAULT 0,
    recoveries                      int         NOT NULL DEFAULT 0,
    clearances_blocks_interceptions int         NOT NULL DEFAULT 0,
    defensive_contribution          int         NOT NULL DEFAULT 0,
    yellow_cards                    int         NOT NULL DEFAULT 0,
    red_cards                       int         NOT NULL DEFAULT 0,
    synced_at                       timestamptz NOT NULL DEFAULT now(),
    UNIQUE (player_id, season_name)
);

-- Прогнозы xPts: одна строка на (игрок, тур, версия модели, момент расчёта).
-- ep_next_snapshot — ep_next из bootstrap на момент прогноза (бейзлайн FPL для сравнения
-- после тура, scripts/gw_review.py). components — разбивка по компонентам (jsonb).
CREATE TABLE IF NOT EXISTS xpts_predictions (
    id               bigserial   PRIMARY KEY,
    player_id        int         NOT NULL,
    gw               int         NOT NULL,
    as_of            timestamptz NOT NULL,
    model_version    text        NOT NULL,
    xpts             real        NOT NULL,
    p_start          real,
    exp_minutes      real,
    components       jsonb       NOT NULL DEFAULT '{}',
    variance         real,
    ep_next_snapshot real,
    created_at       timestamptz NOT NULL DEFAULT now(),
    UNIQUE (player_id, gw, model_version, as_of)
);

CREATE INDEX IF NOT EXISTS xpts_predictions_gw_idx ON xpts_predictions (gw, model_version, as_of DESC);
