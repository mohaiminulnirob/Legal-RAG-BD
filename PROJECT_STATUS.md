# Project Status

This is the living implementation record for the Bangladesh Legal RAG project.
Update it whenever code, dependencies, data artifacts, tests, or project structure change.

Last updated: 2026-09-10

## Current phase

**Phase 6 — Retrieval evaluation: complete**

Next: **Phase 7 — Cross-encoder reranker**

## Completed work

| Date | Area | Result | Verification |
| --- | --- | --- | --- |
| 2026-09-08 | Project setup | Created the source, data, test, and Chroma storage directory structure. | Directory structure inspected. |
| 2026-09-08 | Python environment | Created `.venv` and installed the initial preprocessing and BM25 packages. | Package versions frozen in `requirements.txt`. |
| 2026-09-08 | Ingestion | Added `src/ingestion/preprocess.py`, a schema-aware section-level normalizer. | CLI and in-memory normalization smoke test passed. |
| 2026-09-09 | Dataset inspection | Confirmed the raw contextualized dataset contains 1,484 acts and 35,633 sections. | Read-only schema inspection completed. |
| 2026-09-09 | Preprocessing | Generated `data/processed/legal_sections.json` from the raw dataset. | 35,633 records with unique `chunk_id` values validated. |
| 2026-09-09 | BM25 retrieval | Added a persistent BM25 index builder and command-line search interface. | Unit tests passed; targeted Penal Code section 302 query returned section 302 first. |
| 2026-09-09 | BM25 indexing | Built `data/processed/bm25_index.pkl` from the non-empty processed legal sections. | 35,630 sections indexed; saved index inspected. |
| 2026-09-09 | Dense retrieval | Added Chroma-backed dense retrieval, a semantic-search CLI, and deterministic retrieval tests. | Dense retrieval test passes using a local test embedder. |
| 2026-09-09 | Embedding model | Downloaded and verified the local `BAAI/bge-small-en-v1.5` model cache. | Model loads successfully and produces 384-dimensional normalized vectors. |
| 2026-09-09 | Dense indexing | Started the full CPU batch embedding job for non-empty legal sections. | Persistent Chroma collection is being populated in `chroma_db/`. |
| 2026-09-10 | Dense indexing | Completed the production Chroma build. | 35,630 non-empty legal sections persist in the collection. |
| 2026-09-10 | Hybrid retrieval | Added BM25+dense Reciprocal Rank Fusion (RRF), an analysis-aware CLI, and unit tests. | Four hybrid tests and end-to-end retrieval test pass. |
| 2026-09-10 | Retrieval benchmark | Added a manually curated benchmark of 36 statute-grounded queries across criminal, contract, evidence, and procedure law. | All query IDs and relevant chunk IDs validated against the processed dataset. |
| 2026-09-10 | Retrieval evaluation | Added a reproducible evaluator for BM25, dense, and hybrid RRF. | Per-query and aggregate JSON results exported; all metric tests pass. |

## Current project artifacts

| Artifact | Purpose | Status |
| --- | --- | --- |
| `data/raw/Contextualized_Bangladesh_Legal_Acts.json` | Original contextualized legal-acts dataset. | Present; unchanged. |
| `data/raw/acts/` | 1,484 individual raw act JSON files. | Present; unchanged. |
| `data/processed/legal_sections.json` | Section-level retrieval documents with preserved legal metadata and provenance. | Generated; 35,633 records. |
| `src/ingestion/preprocess.py` | Converts the source dataset into retrieval units. | Implemented and validated. |
| `src/retrieval/bm25.py` | Builds, persists, loads, and queries the lexical BM25 index. | Implemented and validated. |
| `src/retrieval/dense.py` | Builds and queries semantic legal retrieval with Chroma. | Implemented, unit-tested, and fully indexed. |
| `src/retrieval/hybrid.py` | Fuses BM25 and dense candidates with Reciprocal Rank Fusion. | Implemented and validated. |
| `chroma_db/` | Persistent Chroma database for dense legal vectors. | Complete; 35,630 sections indexed. |
| `data/model_cache/` | Project-local cache of the BGE embedding model. | Downloaded; ignored by Git. |
| `data/processed/bm25_index.pkl` | Persistent BM25 index and citation metadata. | Generated; 35,630 non-empty sections indexed. |
| `tests/test_bm25.py` | BM25 tokenizer, persistence, and ranking tests. | Passing. |
| `tests/test_dense.py` | Dense-index persistence and semantic ranking tests. | Passing. |
| `tests/test_hybrid.py` | RRF fusion, duplicate merging, source preservation, and limit tests. | Passing. |
| `data/benchmark/retrieval_queries.json` | Versioned, manually mapped legal retrieval benchmark. | 36 validated queries. |
| `src/evaluation/retrieval_eval.py` | Runs retrieval experiments and exports Recall@5, Recall@10, and MRR. | Implemented and validated. |
| `data/benchmark/results/` | Reproducible per-query and aggregate evaluation artifacts. | Generated. |
| `tests/test_retrieval_eval.py` | Metric calculation and result-export tests. | Passing. |
| `requirements.txt` | Pinned installed Python dependencies. | Present. |
| `.env.example` | Placeholder for future provider configuration. | Present. |

## Processed-record design

Each processed record contains:

- A unique `chunk_id` based on source act and section position.
- Act metadata: title, number, year, language, repeal status, and source URL.
- Legal content: section text, section identifier, section position, and footnotes.
- Provenance/context: source file, government context, and legal-system context.

The source act number cannot serve as a unique identifier because it may be reused. The processed `chunk_id` therefore uses the source file plus the section's position within that act.

## Verification summary

- `src` compiles successfully.
- Preprocessor command executes successfully against the real dataset.
- Processed record count: **35,633**.
- `chunk_id` uniqueness: **validated**.
- Raw dataset remains readable after preprocessing.
- BM25 unit tests pass (tokenization, index persistence, and relevance ranking).
- The targeted query `Section 302 punishment for murder` returns Penal Code section 302 as the top result.
- Dense retrieval unit test passes using a deterministic local embedding model.
- The production BGE model loads successfully and returns 384-dimensional vectors.
- Chroma contains all **35,630** non-empty legal sections.
- Hybrid RRF tests pass: common documents are boosted, duplicates merge, source-only results remain, and limits are enforced.
- The end-to-end hybrid CLI returns fused legal evidence with retriever ranks and scores.
- Evaluation benchmark contains 36 manually assigned, source-validated relevant section IDs.
- All nine retrieval and evaluation tests pass.

## Retrieval evaluation results

| Retriever | Recall@5 | Recall@10 | MRR |
| --- | ---: | ---: | ---: |
| BM25 | 0.694444 | 0.833333 | 0.541545 |
| Dense | 0.750000 | 0.888889 | 0.592626 |
| Hybrid RRF | **0.833333** | **0.916667** | **0.672718** |

These results are from the initial 36-query benchmark. They support using hybrid RRF as the evidence-retrieval baseline, but the benchmark should be expanded and independently reviewed before treating the figures as final thesis results.

## Deferred work

- Reranking, LLM baseline, and reasoning-aware modules.

## Next implementation task

Implement a cross-encoder reranker after hybrid retrieval, including:

1. Rerank the hybrid top-20 candidates with a cross-encoder.
2. Return the top 5–10 citation-preserving evidence sections.
3. Add reranker unit tests and a command-line interface.
4. Re-run this benchmark to measure the reranker's effect.

## Update rule

Whenever implementation work is performed, add a dated entry to **Completed work** and update the affected artifact, verification, deferred-work, and next-task sections as needed.