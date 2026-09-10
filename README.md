# Bangladesh Legal RAG

A research prototype for evidence-aware, case-based legal retrieval in the Bangladesh legal domain.

Implementation progress is tracked in [PROJECT_STATUS.md](PROJECT_STATUS.md).

## Phase 1 setup

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

## Cross-encoder reranking

Rerank the hybrid top-20 candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2` (downloaded to the project-local model cache on first use):

```powershell
python -m src.retrieval.reranker search --query "A person intentionally killed another person" --top-k 5
```

Each result retains its chunk ID, statute metadata, source URL, RRF score, and cross-encoder relevance score.
