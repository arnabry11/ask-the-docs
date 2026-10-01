-- migrate:up
CREATE EXTENSION IF NOT EXISTS vector;


-- migrate:down
-- Keep the extension installed: other schemas may use it.
