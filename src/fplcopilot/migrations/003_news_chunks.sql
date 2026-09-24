-- 003_news_chunks.sql — чанки статей с эмбеддингами (pgvector) для RAG.
-- Родитель — news_articles (parent), ребёнок — абзац или склейка коротких абзацев (child).
-- players/teams/source/published_at дублируются из статьи, чтобы фильтровать без JOIN
-- (в т.ч. обязательный фильтр published_at <= as_of против утечки из будущего).
-- Размерность 1536 = OpenAI text-embedding-3-small (settings.rag_embedding_dims).

CREATE TABLE IF NOT EXISTS news_chunks (
    id            bigserial    PRIMARY KEY,
    article_id    bigint       NOT NULL REFERENCES news_articles(id) ON DELETE CASCADE,
    chunk_index   int          NOT NULL,
    text          text         NOT NULL,
    n_tokens      int          NOT NULL,
    embedding     vector(1536),
    players       int[]        NOT NULL DEFAULT '{}',
    teams         int[]        NOT NULL DEFAULT '{}',
    source        text         NOT NULL,
    published_at  timestamptz  NOT NULL,
    created_at    timestamptz  NOT NULL DEFAULT now(),
    UNIQUE (article_id, chunk_index)
);

-- HNSW по косинусу: ORDER BY embedding <=> query. m/ef_construction — дефолты pgvector (16/64).
CREATE INDEX IF NOT EXISTS news_chunks_embedding_hnsw
    ON news_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS news_chunks_published_at_idx ON news_chunks (published_at DESC);
CREATE INDEX IF NOT EXISTS news_chunks_players_gin      ON news_chunks USING gin (players);
CREATE INDEX IF NOT EXISTS news_chunks_teams_gin        ON news_chunks USING gin (teams);
CREATE INDEX IF NOT EXISTS news_chunks_article_idx      ON news_chunks (article_id);
