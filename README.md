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
