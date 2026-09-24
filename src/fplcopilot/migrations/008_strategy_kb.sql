-- 008_strategy_kb.sql — стратегическая база знаний (RAG #2, rag/kb, docs/strategy_kb.md).
-- kb_docs: один документ на URL реестра sources_kb.yaml (гайд, статья правил, пост, внутренний
-- дайджест); kb_chunks: чанки с заголовком раздела и эмбеддингом (OpenAI text-embedding-3-small,
-- 1536 = settings.rag_embedding_dims). tags/source дублируются в чанки, чтобы фильтровать без JOIN
-- (tags && '{chips}'). Вечнозелёный корпус: published_at может быть NULL и НЕ участвует в ранжировании.
-- Файл идемпотентен (scripts/migrate.py применяет все миграции при каждом запуске).

CREATE TABLE IF NOT EXISTS kb_docs (
    id            bigserial    PRIMARY KEY,
    source        text         NOT NULL,             -- издатель: premierleague, livefpl, reddit, internal…
    url           text         NOT NULL UNIQUE,      -- канонический URL (normalize_url) или путь файла
    title         text         NOT NULL,
    content       text         NOT NULL,             -- извлечённый markdown-текст целиком
    tags          text[]       NOT NULL DEFAULT '{}',
    published_at  timestamptz,                       -- дата публикации, если известна
    fetched_at    timestamptz  NOT NULL DEFAULT now(),
    content_hash  text         NOT NULL,             -- sha256 нормализованного title+content
    created_at    timestamptz  NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS kb_chunks (
    id            bigserial    PRIMARY KEY,
    doc_id        bigint       NOT NULL REFERENCES kb_docs(id) ON DELETE CASCADE,
    chunk_index   int          NOT NULL,
    text          text         NOT NULL,             -- "title\n<heading path>\n<body>"
    n_tokens      int          NOT NULL,
    embedding     vector(1536),
    tags          text[]       NOT NULL DEFAULT '{}',
    source        text         NOT NULL,
    created_at    timestamptz  NOT NULL DEFAULT now(),
    UNIQUE (doc_id, chunk_index)
);

-- HNSW по косинусу: ORDER BY embedding <=> query (m/ef_construction — дефолты pgvector 16/64).
CREATE INDEX IF NOT EXISTS kb_chunks_embedding_hnsw ON kb_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS kb_chunks_tags_gin       ON kb_chunks USING gin (tags);
CREATE INDEX IF NOT EXISTS kb_chunks_doc_idx        ON kb_chunks (doc_id);
CREATE INDEX IF NOT EXISTS kb_docs_tags_gin         ON kb_docs USING gin (tags);
CREATE INDEX IF NOT EXISTS kb_docs_source_idx       ON kb_docs (source);
