# Bangladesh Legal RAG

A research prototype for evidence-aware, case-based legal retrieval in the Bangladesh legal domain.

Implementation progress is tracked in [PROJECT_STATUS.md](PROJECT_STATUS.md).

## Phase 1 setup

Use Python 3.11 or newer.

1. Create and activate the virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install the initial packages:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

3. Put the supplied legal dataset archive or JSON file in `data/raw/` without modifying it.

4. Inspect and preprocess it:

   ```powershell
   python -m src.ingestion.preprocess --input data/raw/Contextualized_Bangladesh_Legal_Acts.json
   ```

The preprocessing command creates `data/processed/legal_sections.json` and prints the act/section counts. It deliberately requires the real input structure rather than guessing it.

## BM25 retrieval

Build the persistent lexical retrieval index:

```powershell
python -m src.retrieval.bm25 build
```

Query it with a case or legal question:

```powershell
python -m src.retrieval.bm25 search --query "What punishment applies for murder?" --top-k 5
```

## Dense retrieval

Build semantic embeddings and persist them in Chroma (the model downloads on first use):

```powershell
python -m src.retrieval.dense build --reset
```

Search using a natural-language scenario:

```powershell
python -m src.retrieval.dense search --query "A person intentionally killed another person" --top-k 5
```

## Hybrid RRF retrieval

Fuse the top 20 lexical and semantic candidates with Reciprocal Rank Fusion (RRF):

```powershell
python -m src.retrieval.hybrid search --query "A person intentionally killed another person" --top-k 10
```

The results retain the RRF score and the rank/score supplied by each retriever, making the evidence selection auditable during later evaluation.

## Retrieval evaluation

The manually curated benchmark in `data/benchmark/retrieval_queries.json` maps each query to statute section IDs verified from the source records. Run the comparison experiment with:

```powershell
python -m src.evaluation.retrieval_eval
```

It writes per-query ranks and aggregate Recall@5, Recall@10, and MRR for BM25, dense retrieval, hybrid RRF, and (after Phase 7) cross-encoder reranking to `data/benchmark/results/`.

## Phase 9 query/case analysis

Analyze query type, complexity, and deterministic surface signals without an
LLM:

```powershell
python -m src.analysis.query_analyzer --query "A person was injured during an argument. What offence and punishment may apply?"
python -m unittest tests.test_query_analyzer -v
```

Optional Groq assistance can extract issue and fact spans that must match exact
text in the user's query. Its JSON output and spans are validated. It is not
used for legal conclusions, evidence sufficiency, or retrieval control:

Set `GROQ_API_KEY` in `.env` before using `--llm-assisted`.

```powershell
python -m src.analysis.query_analyzer --query "A person was injured during an argument. What offence and punishment may apply?" --llm-assisted
```

The output is a versioned JSON `QueryAnalysis` record. Deterministic analysis
works offline, and invalid LLM output is ignored with a warning. This phase
does not alter the conventional baseline, preserving it for later ablations.

## Phase 10 reasoning plan and step-level retrieval

Create a deterministic plan, then retrieve Hybrid RRF evidence separately for
each step. Defaults preserve the baseline retrieval settings (`top_k=5`,
`candidate_k=20`). The command returns a structured plan with each step's
retrieval query, dependencies, and full evidence records; it does not generate
a legal answer or assess evidence sufficiency:

```powershell
# Optional when the embedding model is already cached and network access is unavailable:
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -m src.reasoning.cli --query "What punishment applies when a person intentionally kills another?"
python -m unittest tests.test_reasoning_planner tests.test_step_retrieval -v
```

Use `--plan-only` to inspect the plan without loading the embedding model or
querying the indexes. The standard command requires the local dense model and
both the BM25 and Chroma indexes.

## Phase 11 evidence sufficiency

Assess each Phase 10 step independently against only its attached retrieved
records. Deterministic rules report `SUPPORTED`, `PARTIALLY_SUPPORTED`,
`UNSUPPORTED`, or `UNCERTAIN`, with matched/missing requirement types and
reason codes. Retrieval scores do not establish legal support. Assessment does
not generate a legal answer or retrieve additional evidence:

```python
from src.reasoning.sufficiency import assess_plan

assessments = assess_plan(retrieved_plan)
```

The corrected Penal Code labels are built into separate versioned artifacts:
`legal_sections_v2.json`, `bm25_index_v2.pkl`, and `chroma_db_v2/`. To run
Phase 10 against these indexes without replacing the original baseline:

```powershell
python -m src.reasoning.cli --query "What punishment applies for murder?" --bm25-index data/processed/bm25_index_v2.pkl --dense-database chroma_db_v2
```

The raw acts and original processed/index artifacts remain unchanged.

## Phase 12 deterministic next-action policy

Choose one bounded action from a Phase 11 assessment and explicit execution
state. The policy is a pure function: it does not retrieve, ask the user, or
call an LLM. Missing statute evidence can trigger a bounded retrieval retry;
clarification is selected only when the caller explicitly supplies missing
user facts. Uncertain evidence is acknowledged unless a clarifiable user fact
is supplied. Retry counters in `NextActionState` count attempts already used;
the returned decision reports the attempt number for a newly selected action.

```python
from src.reasoning.next_action import NextActionState, decide_next_action
from src.reasoning.sufficiency import assess_step

assessment = assess_step(reasoning_step)
decision = decide_next_action(
    assessment,
    NextActionState(
        max_retrieval_attempts=2,
        missing_user_facts=("whether the act was intentional",),
    ),
)
print(decision.to_dict())
```

Possible actions are `CONTINUE`, `RETRIEVE_MORE`, `CLARIFY`,
`ACKNOWLEDGE_UNCERTAINTY`, and `STOP`. This policy selects whether to clarify;
it does not phrase a question or run the adaptive retrieval/answer loop.
Run its tests with:

```powershell
python -m unittest tests.test_next_action -v
```

## Cross-encoder reranking

Rerank the hybrid top-20 candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2` (downloaded to the project-local model cache on first use):

```powershell
python -m src.retrieval.reranker search --query "A person intentionally killed another person" --top-k 5
```

Each result retains its chunk ID, statute metadata, source URL, RRF score, and cross-encoder relevance score.

## Retrieval ablation experiments

Run the existing 36-query benchmark with candidate pools 20/50/100, RRF constants
20/40/60/100, and BM25/dense weights 1/1, 0.8/1.2, 0.6/1.4, 0.5/1.5:

```powershell
python -m src.evaluation.retrieval_ablation --output-dir data/benchmark/ablations/my-new-run
python -m unittest tests.test_retrieval_ablation -v
```

Use a new output directory for every run. The runner preserves the production
retrievers and baseline settings, records input/source hashes and raw candidate
ranks, and writes per-query and aggregate results for all 48 combinations.
`manifest.json` is marked `complete` only after the results are saved.
The separate murder-query diagnostic is excluded from benchmark averages.
Metrics are true Recall@5/10, Hit@5/10, and MRR@10; older evaluation fields called
recall are hit rates (equivalent for this benchmark's single-label queries).
These are development-set tuning results, requiring held-out evaluation before
making general improvement claims. No query expansion or reranker is applied.

## Baseline RAG usage

Copy `.env.example` to `.env` and set `GROQ_API_KEY`. The fixed baseline model is `openai/gpt-oss-120b` through Groq's OpenAI-compatible Responses API. Then run the conventional Hybrid-RRF baseline:

```powershell
python -m src.rag.baseline --query "What punishment applies for murder?" --top-k 5
```

The default path uses Hybrid RRF directly. `--use-reranker` is available only for comparison experiments because it performed worse than Hybrid RRF on the initial 36-query benchmark.
