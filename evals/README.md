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

The prepared chunks and downloaded pages remain ignored local files. The next PR adds retrieval metrics and repeatable experiments over this dataset.
