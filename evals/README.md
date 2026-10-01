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

## First run: 2026-10-01

The committed [JSON report](results/retrieval.json) used 24 pages, 785 chunks, and all 60 questions. Its corpus fingerprint is `5626b48eef5578ead0ab3a47e2a7c4d100c239c5bf17970ad463a350e215df29`; the report also records the dataset hash, model names, settings, per-question rankings, and raw best rerank scores. The table below uses only the 50 answerable questions. A hit means a curated gold *document* appeared in the first *k chunks*; it does not measure answer correctness.

| Retrieval mode | Hit@1 | Hit@5 | Hit@10 | Recall@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vector only | 40/50 | 47/50 | 48/50 | 0.92 | 0.855 |
| Keyword only | 8/50 | 8/50 | 8/50 | 0.16 | 0.160 |
| RRF hybrid | 40/50 | 47/50 | 48/50 | 0.92 | 0.852 |
| Hybrid + rerank | 43/50 | 48/50 | 49/50 | 0.94 | 0.903 |

The reranker moved one additional question's gold page into the top five and three into rank one compared with hybrid search. This small, curated set is useful for finding regressions, not for claiming general performance. The keyword query returned no matches for 40 of 50 answerable questions. The current `websearch_to_tsquery` receives the whole natural-language question; its conjunction of content terms is often too restrictive for a single chunk. Hybrid retrieval therefore behaved much like vector-only retrieval here. Query formulation is a concrete follow-up to test.

The two reranked top-five misses were Q015 and Q025. Q015's gold label points to the Rails validation guide, while the top results came from other Rails pages; the labels are primary reference pages rather than a complete list of relevant pages. For Q025, the gold PostgreSQL indexes overview reached rank eight, behind related index pages. Reviewing those passages and adding relevance judgments would make the evaluation stronger.

With the threshold unset, the gate refused 0 of 10 unanswerable questions and incorrectly refused 0 of 50 answerable questions. These are expected baseline counts, not evidence that the system answers unsupported questions safely. Do not treat raw reranker scores as probabilities.

## Gate calibration study

Reproduce the [calibration report](results/gate_calibration.json) from the committed retrieval scores without a database or model call:

```sh
uv run python -m evals.calibrate_gate
```

The script verifies the dataset hash, question IDs, labels, and finite reranker scores. It sweeps every distinct score boundary using the same rule as the runtime gate: a score **strictly below** the threshold is refused. It selects the threshold that refuses the most unanswerable questions while incorrectly refusing at most 2% of answerable questions (one of 50 here). Ties favor fewer incorrect refusals and a lower threshold. A one-decimal threshold is used when rounding leaves every decision unchanged.

| Threshold | Unanswerable refused | Answerable incorrectly refused | Balanced accuracy |
| --- | ---: | ---: | ---: |
| Unset baseline | 0/10 | 0/50 | 0.50 |
| 0.0 | 5/10 | 1/50 | 0.74 |
| **1.5, selected** | **7/10** | **1/50** | **0.84** |
| 2.04, highest balanced accuracy | 8/10 | 3/50 | 0.87 |

At 1.5, Q025 is the single incorrectly refused answerable question. Its gold indexes overview reached rank eight in the retrieval baseline, so the low score may reflect a context retrieval failure. Q054, Q055, and Q058 are unsupported questions that still pass the gate. A high score means the retrieved text looks relevant to the reranker; it does not prove the answer is in the selected passages.

The same 60 questions selected and evaluated this threshold, and only 10 are unanswerable. These counts are calibration-set results, not held-out performance estimates. The score scale also belongs to the recorded reranker and retrieval settings; changing the model, corpus, or candidate strategy requires another evaluation. This gate reduces unnecessary generation calls but cannot replace the answer prompt's instruction to refuse unsupported requests.
