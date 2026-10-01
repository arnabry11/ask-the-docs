# Retrieval evaluation dataset

`dataset.jsonl` contains 50 answerable and 10 unanswerable questions for the 24 pages in `corpus/sources.json`. The answerable set has 22 how-to, 21 factual, 4 misleading-keyword, and 3 multi-document questions. Every selected source page appears as a gold document at least once.

Questions and key facts were written against the locally downloaded Rails 8.1.4 and PostgreSQL 16 pages, then checked against the deterministic prepared chunks. No LLM generated or judged the labels. The negative questions ask about related but unsupported topics; they must be reviewed again if the source catalog expands.

Each line has a stable `id`, a natural-language `question`, `gold_doc_ids`, two or three short `key_facts` for answerable questions, an `answerable` flag, and a `category`. `gold_doc_ids` use the source catalog's document IDs. Multi-document questions list both required pages. Unanswerable rows have empty gold IDs and key facts. The labels are document-level: another page may also contain useful context, so these results are a baseline rather than an exhaustive relevance judgment.

Validate the committed schema and catalog references without downloading anything:

```sh
uv run python -m evals.dataset
```

To recheck that every key fact appears in its gold source pages, first run `uv run python -m scripts.download_corpus` and `uv run python -m scripts.prepare_corpus`, then run:

```sh
uv run python -m evals.dataset --check-evidence data/chunks.jsonl
```

The prepared chunks and downloaded pages remain ignored local files.

## Run the retrieval comparison

Use a dedicated PostgreSQL database containing exactly these 24 pages. With Docker available, a fresh isolated evaluation database can be started like this:

```sh
COMPOSE_PROJECT_NAME=ask-the-docs-eval POSTGRES_PORT=55432 docker compose up -d db
DATABASE_URL='postgresql://ask:ask@localhost:55432/ask_the_docs?sslmode=disable' dbmate --no-dump-schema migrate
DATABASE_URL='postgresql://ask:ask@localhost:55432/ask_the_docs?sslmode=disable' uv run python -m evals.seed_corpus
DATABASE_URL='postgresql://ask:ask@localhost:55432/ask_the_docs?sslmode=disable' uv run python -m evals.run_retrieval_evals
```

`seed_corpus` verifies the downloaded manifest and files, then ingests them directly with the same parser, chunker, embedder, and repository as the worker. It does not need Redis or source HTTP fetches; the embedding and reranking models may need a one-time download if they are not cached. The evaluator refuses a database with missing or extra documents, records a corpus fingerprint and model/settings metadata, and writes `evals/results/retrieval.json`. It compares vector-only, keyword-only, RRF hybrid, and hybrid plus local reranking on the same question set. No paid model is called.

Hit rate@k means at least one gold document appears among the first *k chunks*. Recall@k is the fraction of gold documents seen there; MRR@k is the reciprocal rank of the first gold-document chunk. Multiple chunks from one document occupy multiple ranks but give no extra recall. Unanswerable questions are excluded from retrieval accuracy and reported separately as gate outcomes. With the default unset threshold, the gate should usually refuse only questions returning no sources; calibration comes in the next feature.
