-- migrate:up
CREATE TABLE llm_cache (
    key char(64) PRIMARY KEY,
    answer text NOT NULL,
    citations jsonb NOT NULL,
    model text NOT NULL,
    prompt_version text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- migrate:down
DROP TABLE llm_cache;
