-- migrate:up
CREATE TABLE documents (
    id text PRIMARY KEY,
    source text NOT NULL CHECK (source IN ('rails', 'postgresql')),
    version text NOT NULL,
    title text NOT NULL,
    source_url text NOT NULL,
    content_hash char(64) NOT NULL,
    ingestion_fingerprint char(64) NOT NULL,
    chunk_count integer NOT NULL CHECK (chunk_count >= 0),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
    id text PRIMARY KEY,
    document_id text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    section_path text[] NOT NULL,
    source_url text NOT NULL,
    text text NOT NULL,
    token_count integer NOT NULL CHECK (token_count > 0),
    embedding vector(384) NOT NULL,
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english'::regconfig, text)) STORED,
    UNIQUE (document_id, ordinal)
);

CREATE INDEX chunks_document_id_idx ON chunks (document_id);
CREATE INDEX chunks_search_vector_idx ON chunks USING gin (search_vector);
CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE ingestion_attempts (
    id bigserial PRIMARY KEY,
    document_id text NOT NULL,
    status text NOT NULL CHECK (status IN ('completed', 'skipped', 'failed')),
    detail text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX ingestion_attempts_document_id_created_at_idx
    ON ingestion_attempts (document_id, created_at DESC);

-- migrate:down
DROP TABLE ingestion_attempts;
DROP TABLE chunks;
DROP TABLE documents;
