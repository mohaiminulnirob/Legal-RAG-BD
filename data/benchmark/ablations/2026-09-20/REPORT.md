# Retrieval ablation — 2026-09-20

Completed 48 configurations on the unchanged 36-query development benchmark.
No production retrieval, indexes, prompt, or Phase 8 settings were changed.
The natural-language murder diagnostic is a separate 37th query excluded from averages.

## One-factor comparisons

Each row varies the named setting from the control: pool 20, RRF constant 60,
BM25/dense weights 1/1. Final evaluation cutoffs are 5 and 10.

| Change | Recall@5 | Recall@10 | MRR@10 | Diagnostic section 302 rank |
| --- | ---: | ---: | ---: | ---: |
| Control | 0.833333 | 0.916667 | 0.672718 | 7 |
| Pool 50 | 0.833333 | 0.972222 | 0.679938 | 3 |
| Pool 100 | 0.861111 | 0.972222 | 0.680556 | 3 |
| RRF constant 20 | 0.805556 | 0.916667 | 0.671792 | 7 |
| RRF constant 40 | 0.833333 | 0.916667 | 0.672718 | 7 |
| RRF constant 100 | 0.833333 | 0.916667 | 0.672718 | 7 |
| Weights 0.8/1.2 | 0.833333 | 0.916667 | 0.683135 | 5 |
| Weights 0.6/1.4 | 0.861111 | 0.916667 | 0.679167 | 5 |
| Weights 0.5/1.5 | 0.861111 | 0.916667 | 0.680093 | 5 |

## Combined configurations

| Pool | RRF constant | BM25/dense weights | Recall@5 | Recall@10 | MRR@10 |
| ---: | ---: | --- | ---: | ---: | ---: |
| 100 | 60 | 0.8/1.2 | 0.861111 | 0.972222 | **0.692008** |
| 100 | 100 | 0.6/1.4 | **0.888889** | 0.972222 | 0.687654 |

The first row has the highest MRR@10 across the grid. The second has the highest
MRR@10 among configurations tied for highest Recall@5. It places section 302 at
rank 2 in the diagnostic. The latter improves top-five hits from 30 to 32 of 36;
this small difference is not evidence of statistical significance or generalization.

## Interpretation and next experiment

Pool 100 is a simple promising candidate: it improves the measured metrics without
changing fusion weights. Adjusting only the RRF constant did not resolve the murder
diagnostic. Dense weighting improved that diagnostic but did not uniformly improve
all benchmark metrics. BM25 should remain part of the comparison.

Freeze these results and validate pool 100 and the two combined candidates on a
new independently reviewed query set before selecting a production default.
Include multi-section, exception, paraphrase, and case-based queries. Existing
section-label errors and incomplete relevance judgments should be audited separately.
Query expansion and domain-specific reranking remain separate future experiments;
do not inject answer-derived section numbers into benchmark queries.

## Reproduction and metric definitions

Run `python -m src.evaluation.retrieval_ablation --output-dir data/benchmark/ablations/new-run`.
Output directories must be new. `manifest.json` records completion, settings,
timestamps, query/index/source hashes and the embedding model. The run uses the
existing local Chroma collection and model cache. Exact future reruns also require
preserving those artifacts and the environment; their full contents are not hashed.

`candidates_*.json` stores the actual source rankings and scores for every query.
Dense retrieval is invoked independently with top-k 20, 50, and 100 to account for
possible approximate-search differences. Candidate lists are reused across fusion
settings at the same pool size. `per_query_results.json` includes top-ten IDs,
component ranks/scores, metrics and first relevant rank across the fused pool.
`aggregate_results.json` contains all 48 configurations.

Recall@k is the fraction of labeled relevant chunks retrieved; Hit@k means at least
one labeled chunk was retrieved. These coincide for this single-label benchmark.
MRR@10 assigns zero when no relevant result appears within the top ten, matching
the old evaluator's effective cutoff. These are development-set tuning results,
not held-out thesis estimates. No legal answer generation or paid LLM calls occur.
