# Ask the Docs

A Python and FastAPI service for answering questions about the Rails Guides and PostgreSQL documentation with cited sources. The project will measure retrieval quality, control paid model use, and show the engineering behind a production style RAG system.

The service foundation, corpus preparation, local ingestion, hybrid retrieval, local reranking, and a [retrieval evaluation dataset and first experiment](evals/README.md) are in place. Answer generation will arrive in separate reviewable PRs. Ingestion, retrieval, and the local evaluation make no paid model calls.

## Planned request flow

1. Retrieve from PostgreSQL full text search and pgvector using local embeddings.
2. Fuse results, rerank locally, and refuse questions below a calibrated confidence threshold.
3. Return a cached answer when available; otherwise enforce a daily budget before one OpenRouter generation call.
4. Return cited sources and record latency, token use, and estimated cost.

## Development workflow

Features are reviewed as small, focused PR stacks. Each PR describes **What**, **Why**, and **How**. Later milestones are tracked in GitHub issues and started after review of the current stack.

## Quick start

With Docker running:

```sh
cp .env.example .env
docker compose up --build
```

The dbmate container applies migrations before the app starts. `GET http://127.0.0.1:8000/health` checks its database connection. PostgreSQL is available on port 5432 for local development; Redis runs inside the Compose network and will support the later ingestion worker. The example credentials are for local development only.

## Run the API locally

Install [uv](https://docs.astral.sh/uv/), then run:

```sh
uv sync
cp .env.example .env
docker compose up -d db redis
dbmate migrate
uv run uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/health` for the database readiness response and `/docs` for FastAPI's interactive API docs. Run the checks with:

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy app evals
uv run pytest
```

The real database integration test is opt-in locally: `RUN_INTEGRATION_TESTS=1 uv run pytest -m integration`. CI runs it against PostgreSQL with pgvector. A 503 health response means the database cannot be reached. Install [dbmate](https://github.com/amacneil/dbmate) for host-based migration commands (`brew install dbmate` on macOS); the Docker path supplies it automatically.

## Download the source corpus

```sh
uv run python -m scripts.download_corpus
```

The command fetches the 24 selected pages in [`corpus/sources.json`](corpus/sources.json) into ignored `data/raw/` and writes `data/raw/manifest.json` with URLs, versions, byte counts, and SHA-256 hashes. Repeating it reuses matching local files. Use `--refresh` to check the publisher again. A failed download does not replace the previous manifest. No model or paid API is involved.

The Rails Guides pages are sourced from [Rails 8.1.4 at commit `c3466ea`](https://github.com/rails/rails/tree/c3466ea00d7121798e3aa3144ffdf7174b81d8cb/guides/source) and link readers to the published [Rails 8.1 Guides](https://guides.rubyonrails.org/v8.1/). Rails is [MIT licensed](https://github.com/rails/rails/blob/v8.1.4/MIT-LICENSE). The PostgreSQL pages link to the official [PostgreSQL 16 documentation](https://www.postgresql.org/docs/16/) and remain on the publisher's major-version URL, which can receive patch updates; the local manifest records the exact downloaded bytes. PostgreSQL documentation is covered by the [PostgreSQL License](https://www.postgresql.org/about/licence/) (copyright © 1996–2026 The PostgreSQL Global Development Group and © 1994 The Regents of the University of California). Keep the publishers' attribution and license notices with any redistributed corpus copy. This repository contains links and tooling, not downloaded documentation.

## Parse and chunk the corpus

After downloading, run:

```sh
uv run python -m scripts.prepare_corpus --max-tokens 400 --overlap-tokens 40
```

This verifies each file against the download manifest, then writes deterministic JSON Lines to ignored `data/chunks.jsonl`. Each row carries a stable chunk ID, document hash, title, section path, source URL, text, and local token count. The parser keeps Rails heading structure and Markdown code blocks; it removes PostgreSQL page navigation while retaining heading anchors, links, and code examples. Chunks stay within the configured token cap except when a single code block is larger; that block remains intact. The overlap copies trailing prose into the next chunk. Repeating the command on unchanged inputs produces identical bytes. This step uses a local tokenizer and makes no model calls.

## Ingest into PostgreSQL

The worker fetches each selected page, parses and chunks it, embeds each chunk locally with [FastEmbed's 384-dimensional BGE small model](https://qdrant.github.io/fastembed/examples/Supported_Models/), then stores it in PostgreSQL with pgvector and full-text search indexes. The embedding text includes the page title and heading path. The first job downloads the model into ignored `models/` (a Docker volume in Compose); later jobs reuse it. No paid LLM API is called.

With Compose running, enqueue one page or the full catalog:

```sh
docker compose exec app python -m scripts.enqueue_corpus --document-id rails:getting_started
docker compose exec app python -m scripts.enqueue_corpus
docker compose logs -f worker
```

For a host-based worker, run `uv run rq worker ingestion --url redis://localhost:6379/0 --with-scheduler --worker-class rq.worker.SpawnWorker` after setting `DATABASE_URL` and starting Redis. SpawnWorker is compatible with macOS and the local embedding runtime. Jobs retry up to three times with short delays. Each attempt records `completed`, `skipped`, or `failed` with its document ID in `ingestion_attempts`. To inspect failures in Compose:

```sh
docker compose exec db psql -U ask -d ask_the_docs -c "SELECT document_id, status, detail, created_at FROM ingestion_attempts WHERE status = 'failed' ORDER BY created_at DESC LIMIT 20"
```

Re-ingesting a page checks a fingerprint of its source bytes, source metadata, model name, chunk settings, and pipeline version. An unchanged page skips embedding and database replacement. A changed page replaces its chunks in one transaction. The page and chunk metadata remain available for the retrieval feature.

`POST /ingest` can enqueue one `{"document_id": "rails:getting_started"}` or all pages with `{}`. It is disabled until `INGEST_ADMIN_KEY` is set to a private random value in `.env`; send that value in the `X-Admin-Key` header. The CLI above works without the HTTP admin key. The database host port can be changed with `POSTGRES_PORT` if 5432 is already in use.

## Inspect hybrid retrieval

After a worker has ingested at least one page, query its stored chunks:

```sh
curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  -d '{"question":"How do PostgreSQL indexes help queries?","top_k":5}' \
  http://127.0.0.1:8000/query
```

`POST /query` embeds the question locally with the same model as ingestion and retrieves up to 30 vector candidates and 30 PostgreSQL full-text candidates. Keyword search uses `websearch_to_tsquery`, so quoted phrases and terms such as `OR` work as web-style search syntax. Reciprocal rank fusion (RRF, default `k=60`) combines the two ordered lists. The top 20 fused candidates are scored by [FastEmbed's local MiniLM cross-encoder](https://qdrant.github.io/fastembed/examples/Supported_Models/), using the question, title, section path, and passage text. The endpoint returns the best `top_k` unique chunks with their document ID, title, section path, source URL, and text. `top_k` defaults to 5 and is limited to 1–20; questions are limited to 500 characters. The cross-encoder downloads into ignored `models/` on the first query, then reuses that cache. An empty corpus returns an empty `results` list, and a question with no keyword matches can still return vector results.

Each result includes `rerank_score`, `rrf_score`, one-based `vector_rank` and `keyword_rank`, `cosine_distance`, and `fts_rank`. A missing rank or score is `null` when a chunk came from only one search. Higher cross-encoder scores determine the final order; these raw scores are not probabilities or calibrated confidence values. The candidate limits, fusion constant, model, and rerank limit can be changed with `TOP_K_VECTOR`, `TOP_K_FTS`, `RRF_K`, `RERANK_MODEL`, and `RERANK_TOP_N` in `.env`.

The response also includes `gated`, `gate_reason`, `gate_threshold`, and `refusal`. No sources produce a canned refusal. The score threshold is unset by default, so retrieved passages are not refused on an uncalibrated score. A finite `GATE_THRESHOLD` can be set for experiments; the [first retrieval experiment](evals/README.md#first-run-2026-10-01) establishes a baseline, and a later PR will calibrate the threshold. Changing `RERANK_MODEL` requires recalibrating that threshold. A gated response still includes its closest sources. The endpoint returns sources for inspection and does not generate an answer or call an LLM.

For Ruby developers: `pyproject.toml` plus `uv.lock` serve the role of a Gemfile and lockfile. `app/main.py` assembles the FastAPI application; `app/api` contains thin HTTP routes. Later domain services and external API clients will stay outside routes, like service objects and client/resource classes in Ruby.

## License

The service code is MIT licensed. The upstream documentation keeps its own license and attribution.
