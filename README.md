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

The source corpus is fetched from its original publishers at development time. It is not committed to this repository. [Ruby on Rails is MIT licensed](https://github.com/rails/rails/blob/main/MIT-LICENSE); [PostgreSQL's license](https://www.postgresql.org/about/licence/) permits use of its documentation with attribution. Any downloader must retain source URLs and required notices.

## License

The service code is MIT licensed. The upstream documentation keeps its own license and attribution.
