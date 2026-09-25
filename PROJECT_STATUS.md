# Project Status

This is the living implementation record for the Bangladesh Legal RAG project.
Update it whenever code, dependencies, data artifacts, tests, or project structure change.

Last updated: 2026-09-25

## Current phase

**Phase 12 - Deterministic clarification and next-action policy: complete**

Phase 8 conventional baseline remains available as the comparison system.

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
| 2026-09-10 | Cross-encoder reranking | Added a hybrid-candidate reranker using `cross-encoder/ms-marco-MiniLM-L-6-v2`. | Three deterministic reranker tests pass; production model loads from project-local cache. |
| 2026-09-10 | Reranker evaluation | Started an apples-to-apples four-retriever run on the 36-query benchmark. | CPU background worker is scoring the hybrid top-20 candidates per query. |
| 2026-09-16 | Reranker evaluation | Completed the four-retriever comparison. | Reranker underperformed Hybrid RRF, so it remains an experimental branch. |
| 2026-09-16 | Baseline RAG | Added a conventional Hybrid-RRF-to-LLM legal-information baseline using the OpenAI Responses API. | Context, abstention, citation traceability, API-key handling, and orchestration tests pass. |
| 2026-09-19 | Baseline RAG provider | Switched the fixed baseline LLM provider to Groq using its OpenAI-compatible Responses API. | Groq key/base-URL configuration tests pass; retrieval code unchanged. |
| 2026-09-24 | Phase 9 query/case analysis | Added deterministic query kind/complexity signals, versioned structured output, and opt-in Groq span extraction; baseline behavior remains unchanged. | Four Phase 9 unit tests pass; seven representative offline CLI cases inspected. |
| 2026-09-24 | Phase 9 classification refinement | Distinguished procedural and out-of-scope queries; multi-issue classification now requires multiple questions or joined legal issue types. | Phase 9 unit tests and all seven documented CLI scenarios pass on Python 3.13. |
| 2026-09-24 | Phase 10 reasoning planner | Added deterministic, versioned reasoning plans for offence, punishment, exception/defence, procedural, provision lookup, and general legal-rule retrieval. | Planner tests pass for legal query shapes, dependencies, and out-of-scope queries. |
| 2026-09-24 | Phase 10 step-level retrieval | Added independent Hybrid RRF retrieval per plan step with full evidence provenance and a JSON CLI; Phase 8 baseline path unchanged. | Ten Phase 10 tests pass; full CLI retrieved five records per step with `top_k=5`, `candidate_k=20` using the local model cache in offline mode. |
| 2026-09-24 | Phase 11 evidence sufficiency | Added deterministic per-step `SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`, and `UNCERTAIN` assessments, matched/missing requirement types, and auditable reason codes. | 24 combined Phase 9?11 tests pass; malformed, ambiguous, conflicting, empty, partial, and high-score unsupported cases covered. |
| 2026-09-24 | Versioned Penal Code label correction | Fixed parsing for unnumbered Exception 1 after section 300 and the leading-dot `.301.` label; preserved raw records, positions, and chunk IDs. Rebuilt v2 processed JSON/BM25 artifacts and cloned the dense baseline into a separate v2 database, updating only citation metadata because all three corrected records have unchanged embedding text. | Positions 341-344 map to 300, 300 (Exception 1), 301, 302; v2 indexes contain 35,630 non-empty records. |
| 2026-09-25 | Phase 12 next-action policy | Added a pure deterministic policy mapping per-step Phase 11 assessments and bounded execution state to continue, retrieve more, clarify, acknowledge uncertainty, or stop. Clarification requires explicitly identified missing user facts. | Ten policy tests and 35 combined Phase 9-12 tests pass. |

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
| `src/retrieval/reranker.py` | Reranks hybrid candidates with a cross-encoder while preserving citations. | Implemented and tested; benchmark comparison complete. |
| `src/rag/context.py` | Builds structured, bounded evidence blocks from retrieved sections. | Implemented and tested. |
| `src/rag/prompt.py` | Defines the evidence-only legal-information prompt. | Implemented. |
| `src/rag/llm.py` | Fixed-model Groq Responses API adapter using the OpenAI client. | Implemented; uses `openai/gpt-oss-120b` by default. |
| `src/rag/baseline.py` | Conventional Hybrid-RRF baseline answer generation with traceable citations. | Implemented and tested. |
| `src/analysis/schemas.py` | Versioned query-analysis types and exact-source span validation. | Implemented and verified. |
| `src/analysis/rules.py` | Auditable deterministic query-shape signals and initial type/complexity classification. | Implemented and verified. |
| `src/analysis/query_analyzer.py` | Offline-first analysis API and CLI with optional validated LLM assistance. | Implemented and verified. |
| `tests/test_query_analyzer.py` | Phase 9 category, schema, span, and LLM fallback coverage. | Four tests pass. |
| `src/reasoning/schemas.py` | Versioned reasoning plan/step schemas, dependencies, status, and attached provenance-rich evidence. | Implemented and verified. |
| `src/reasoning/planner.py` | Deterministic plan generation from Phase 9 analysis and explicit issue cues. | Implemented and verified. |
| `src/reasoning/step_retriever.py` | Separate Hybrid RRF call per step with preserved candidate settings and evidence records. | Implemented and verified. |
| `src/reasoning/cli.py` | JSON plan/retrieval CLI; supports plan-only inspection. | Implemented and verified. |
| `src/reasoning/evidence_rules.py` | Deterministic requirement rules by reasoning-step type. | Implemented and verified. |
| `src/reasoning/sufficiency.py` | Independent evidence-attached step assessments with explicit uncertainty and reason codes. | Implemented and verified; no retrieval or LLM calls. |
| `src/reasoning/next_action.py` | Versioned deterministic next-action decision and validated retry/clarification state. | Implemented and verified; pure policy with no retrieval or LLM calls. |
| `tests/test_next_action.py` | Supported, partial, unsupported, uncertain, budget, clarification, stop, validation, and serialization policy coverage. | Ten tests pass. |
| `data/processed/legal_sections_v2.json` | Corrected section labels with stable source positions and chunk IDs. | Generated; 35,633 valid records. |
| `data/processed/bm25_index_v2.pkl` | BM25 index built from corrected processed records. | Generated; 35,630 non-empty records indexed. |
| `chroma_db_v2/` | Versioned dense index cloned from the baseline with corrected citation metadata. | Complete; 35,630 records. Embeddings are unchanged because retrieval text is unchanged. |
| `tests/test_evidence_sufficiency.py`, `tests/test_preprocess.py` | Phase 11 sufficiency and section-label regression coverage. | Passing. |
| `tests/test_reasoning_planner.py` | Phase 10 query-shape planning and dependency tests. | Six tests pass. |
| `tests/test_step_retrieval.py` | Per-step search, candidate-setting, status, and provenance tests. | Four tests pass. |
| `chroma_db/` | Persistent Chroma database for dense legal vectors. | Complete; 35,630 sections indexed. |
| `data/model_cache/` | Project-local cache of the BGE embedding model. | Downloaded; ignored by Git. |
| `data/processed/bm25_index.pkl` | Persistent BM25 index and citation metadata. | Generated; 35,630 non-empty sections indexed. |
| `tests/test_bm25.py` | BM25 tokenizer, persistence, and ranking tests. | Passing. |
| `tests/test_dense.py` | Dense-index persistence and semantic ranking tests. | Passing. |
| `tests/test_hybrid.py` | RRF fusion, duplicate merging, source preservation, and limit tests. | Passing. |
| `tests/test_reranker.py` | Reranker scoring, ranking, metadata, limits, and tie behavior. | Passing. |
| `tests/test_baseline_rag.py` | Context, citations, abstention, configuration, and baseline orchestration tests. | Passing. |
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

- 2026-09-20 retrieval ablation completed: 48 configurations on the unchanged 36 queries plus one excluded diagnostic. Added `src/evaluation/retrieval_ablation.py`, three passing tests, and a README command. Results, raw ranks and run manifest are in `data/benchmark/ablations/2026-09-20/`; see `REPORT.md` for all single-factor comparisons and limitations. Production defaults remain unchanged.
- Ablation control reproduced Hybrid RRF Recall@5 0.833333, Recall@10 0.916667, MRR@10 0.672718. Pool 100 alone yielded 0.861111 / 0.972222 / 0.680556. Best grid MRR@10 was 0.692008 (pool 100, RRF 60, weights 0.8/1.2); highest Recall@5 was 0.888889. Treat this benchmark as development data after tuning; validate on independent queries before promoting a setting.

- 2026-09-20 diagnostic: for `What punishment applies for murder?`, the exact Hybrid CLI and baseline (`top_k=5`, `candidate_k=20`) both returned chunk suffixes 0351, 0125, 0040, 0342, 0118. A temporary print immediately after baseline retrieval records the ranks and scores. No retrieval logic or indexes were changed.
- Penal Code section 302 (`act_act-print-11_section_0344`) ranks 7 with the baseline candidate pool: dense rank 2, absent from BM25 top 20, RRF 0.01612903. The earlier reported hybrid rank 3 used `candidate_k=100`, so it did not describe the baseline. The live baseline completed and reported insufficient evidence.
- Provenance review of raw `act-print-11.json`: array item 342 contains an unnumbered “Exception 1” continuance immediately after item 341, whose text begins with section 300. `preprocess.section_identifier()` falls back to the array position when no leading number matches, so it labels that exception “342”; this is a raw-record boundary/numbering gap combined with a position-as-section fallback. Item 343 begins `.301.`, which also fails the current leading-number pattern and is liable to become label “343”. No source data or processed index was changed. Review/repair label mapping before relying on these citations.
- Phase 10 CLI smoke retrieval for “What punishment applies for murder?” produced independent offence and punishment searches (five evidence records each); the per-step result lists and full provenance remain separate. No answer or evidence-sufficiency decision is produced.
- Combined Phase 9/10 suite: 14 tests pass (`tests.test_query_analyzer`, `tests.test_reasoning_planner`, `tests.test_step_retrieval`).

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
- Cross-encoder unit tests pass; each reranked result retains chunk ID, statute metadata, source URL, RRF score, and `reranker_score`.
- Baseline RAG tests pass and the OpenAI-compatible Python SDK is pinned in `requirements.txt`.
- Baseline API requests use `GROQ_API_KEY` and `https://api.groq.com/openai/v1` without affecting retrieval.
- The baseline uses Hybrid RRF by default; reranking is explicitly opt-in for experimentation.
- Evaluation benchmark contains 36 manually assigned, source-validated relevant section IDs.
- All nine retrieval and evaluation tests pass.

## Retrieval evaluation results

| Retriever | Recall@5 | Recall@10 | MRR |
| --- | ---: | ---: | ---: |
| BM25 | 0.694444 | 0.833333 | 0.541545 |
| Dense | 0.750000 | 0.888889 | 0.592626 |
| Hybrid RRF | **0.833333** | **0.916667** | **0.672718** |
| Cross-encoder reranker | 0.750000 | 0.833333 | 0.609369 |

These results are from the initial 36-query benchmark. Hybrid RRF is the strongest configuration and is therefore the baseline RAG retriever. The reranker remains available for experiments but is not part of the default baseline. The benchmark should be expanded and independently reviewed before treating the figures as final thesis results.

## Deferred work

- None for Phases 9-12. Adaptive execution, clarification wording, and final answer integration are planned for later phases.

## Next implementation task

Phase 13: define uncertainty acknowledgement and response wording using Phase 12 decisions, without adding a full retrieve/clarify/answer loop yet.

## Update rule

Whenever implementation work is performed, add a dated entry to **Completed work** and update the affected artifact, verification, deferred-work, and next-task sections as needed.
