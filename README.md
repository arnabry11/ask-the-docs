# Ask the Docs

A Python and FastAPI service for answering questions about the Rails Guides and PostgreSQL documentation with cited sources. The project will measure retrieval quality, control paid model use, and show the engineering behind a production style RAG system.

The first milestone is the service foundation. Search, ingestion, answer generation, and evaluation will arrive in separate reviewable PRs. No paid model calls are made by the foundation.

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
uv run mypy app
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

For Ruby developers: `pyproject.toml` plus `uv.lock` serve the role of a Gemfile and lockfile. `app/main.py` assembles the FastAPI application; `app/api` contains thin HTTP routes. Later domain services and external API clients will stay outside routes, like service objects and client/resource classes in Ruby.

## License

The service code is MIT licensed. The upstream documentation keeps its own license and attribution.
