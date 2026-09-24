-- 010_signal_form_notes.sql — промпт сигнала v4 (rag/extract.py, docs/rag.md, docs/EVALS.md A/B #4).
-- form_notes: форма / роль / позиция / стандарты игрока — контекст для пользователя, НЕ
-- доказательство доступности: [{kind: form|role|position|set_pieces, text, quote, chunk_id,
-- source, url, published_at}], цитаты verbatim и про этого игрока (проверяет код).
-- Модель минут / xPts эту колонку не читает. Сигналы v1–v3 получают [].
-- Файл идемпотентен (scripts/migrate.py применяет все миграции при каждом запуске).

ALTER TABLE player_signals ADD COLUMN IF NOT EXISTS form_notes jsonb NOT NULL DEFAULT '[]'::jsonb;
