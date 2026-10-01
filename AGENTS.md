# Ask the Docs agent rules

## Scope and workflow

- This is a Python 3.12 and FastAPI portfolio project. Build one feature at a time in small, focused PRs. Stop after presenting a completed feature's PR stack for review.
- Use `main` as the default branch. Stacked PRs may target the preceding branch. Before opening a PR, inspect `git log <base>..HEAD` to confirm its scope.
- Write PR descriptions with **What**, **Why**, and **How**. Keep commits imperative and focused. Never add assistant attribution trailers.
- Use the `github.com-personal` SSH alias for Git operations and verify the remote belongs to `arnabry11`. Use the personal Git author identity configured in this repository.
- Keep secrets, downloaded corpora, model files, databases, caches, and local environment files out of Git. Do not make real paid LLM calls as part of tests or CI.

## Python patterns

- Keep FastAPI routes thin: validate input, call a service, and map results to HTTP. Put domain decisions in small services and database queries in repositories.
- Use explicit constructor dependencies and Python protocols where they help testing. Prefer plain classes with a `call` method for stateful workflows; use functions for simple stateless transformations. Avoid framework driven magic and unnecessary abstractions.
- External APIs live behind dedicated clients, following the intent of the user's Ruby `Iterable::Client` and resource classes: the client owns base URL, authentication, timeout, retries, and response error mapping; services depend on the client interface. Never issue provider HTTP requests directly from routes.
- Keep configuration in one Pydantic settings object, loaded from environment. Name domain limits and thresholds as constants or settings rather than scattering literals.
- Use typed code, clear names, small modules, and comments only for non-obvious reasons or upstream constraints. Avoid mutable default arguments and hidden I/O at import time.
- Use SQLAlchemy for application database access and dbmate for plain SQL schema migrations. Generate each migration with `dbmate new <name>`, then review and edit both directions. Never modify an already applied migration; add a new one.
- When a migration changes the schema, refresh `db/schema.sql` in the same PR and verify `dbmate load` succeeds against a fresh PostgreSQL 16 database. Keep corpus rows and credentials out of the snapshot.
- Preserve source titles, section paths, and URLs through ingestion and retrieval. Treat retrieved document text as untrusted data, never as instructions to the service.

## Verification

- Behavior changes need focused pytest coverage. Test services independently and routes through FastAPI's test client. Mock external clients; integration tests requiring PostgreSQL should be marked and run against a real pgvector database.
- Before a PR, run `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy app`, and `uv run pytest`. Run `dbmate migrate` plus applicable integration tests when the change touches SQL or migrations.
- CI must use fake or mocked model responses and must not require an OpenRouter key.

## Developer experience

- Explain unfamiliar Python choices in the README or brief module docstrings, especially where they differ from common Ruby patterns. Prefer copyable `uv` commands.
- Keep the app bootable at each commit and update `.env.example` when adding settings.
