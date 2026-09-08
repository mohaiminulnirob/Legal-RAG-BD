# Project Status

This is the living implementation record for the Bangladesh Legal RAG project.
Update it whenever code, dependencies, data artifacts, tests, or project structure change.

Last updated: 2026-09-09

## Current phase

**Phase 3 — BM25 retrieval: complete**

Next: **Phase 4 — Dense retrieval**

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

## Current project artifacts

| Artifact | Purpose | Status |
| --- | --- | --- |
| `data/raw/Contextualized_Bangladesh_Legal_Acts.json` | Original contextualized legal-acts dataset. | Present; unchanged. |
| `data/raw/acts/` | 1,484 individual raw act JSON files. | Present; unchanged. |
| `data/processed/legal_sections.json` | Section-level retrieval documents with preserved legal metadata and provenance. | Generated; 35,633 records. |
| `src/ingestion/preprocess.py` | Converts the source dataset into retrieval units. | Implemented and validated. |
| `src/retrieval/bm25.py` | Builds, persists, loads, and queries the lexical BM25 index. | Implemented and validated. |
| `data/processed/bm25_index.pkl` | Persistent BM25 index and citation metadata. | Generated; 35,630 non-empty sections indexed. |
| `tests/test_bm25.py` | BM25 tokenizer, persistence, and ranking tests. | Passing. |
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

## Deferred work

- Dense retrieval and vector storage dependencies (`sentence-transformers`, `chromadb`).
- Hybrid retrieval, reranking, LLM baseline, and reasoning-aware modules.

## Next implementation task

Implement dense retrieval using sentence-transformer embeddings and a persistent Chroma collection, including:

1. Install and pin `sentence-transformers` and `chromadb`.
2. Generate embeddings for the 35,630 non-empty legal sections.
3. Persist the vectors and citation metadata in Chroma.
4. Add dense-retrieval tests and a command-line query interface.

## Update rule

Whenever implementation work is performed, add a dated entry to **Completed work** and update the affected artifact, verification, deferred-work, and next-task sections as needed.
