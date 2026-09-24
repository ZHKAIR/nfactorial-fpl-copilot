-- 002_status_transitions.sql — снимки статусов как история переходов.
-- Раньше дедуп был UNIQUE(player_id, news, news_added) и снимок писался только при непустом news.
-- Теперь снимок пишется при ЛЮБОМ изменении (status, chance_next, news, news_added) относительно
-- последнего снимка игрока, включая возврат в строй (status='a', news=''). Состояние может
-- законно повторяться (травма -> здоров -> травма -> здоров), поэтому UNIQUE снимаем;
-- дедуп — сравнение с последним снимком в rag/ingest.py (snapshot_needed).

ALTER TABLE player_status_snapshots
    DROP CONSTRAINT IF EXISTS player_status_snapshots_player_id_news_news_added_key;

-- Быстрый поиск последнего снимка игрока (DISTINCT ON ... ORDER BY player_id, snapshot_at DESC, id DESC).
CREATE INDEX IF NOT EXISTS player_status_snapshots_last_idx
    ON player_status_snapshots (player_id, snapshot_at DESC, id DESC);
