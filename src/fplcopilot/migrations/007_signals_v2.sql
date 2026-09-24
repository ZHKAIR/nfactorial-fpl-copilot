-- 007_signals_v2.sql — RAG v2 (rag/extract.py, docs/rag.md): pre-LLM abstention.
-- abstained = true: сигнал выдан без вызова LLM (нет официального fpl_api-чанка игрока и
-- ни один кандидат не упоминает игрока / максимальный rerank-score ниже RAG_ABSTAIN_SCORE);
-- availability=unknown, confidence 0, evidence пустой, стоимость 0.
-- Файл идемпотентен (scripts/migrate.py применяет все миграции при каждом запуске).

ALTER TABLE player_signals ADD COLUMN IF NOT EXISTS abstained boolean NOT NULL DEFAULT false;
