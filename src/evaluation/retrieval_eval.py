"""Reproducibly evaluate BM25, dense, and hybrid legal retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from src.retrieval import bm25, dense, hybrid


DEFAULT_QUERIES = Path("data/benchmark/retrieval_queries.json")
DEFAULT_RESULTS = Path("data/benchmark/results")
RETRIEVER_NAMES = ("bm25", "dense", "hybrid")


def first_relevant_rank(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> int | None:
    """Return the one-based rank of the first manually relevant evidence record."""
    relevant = set(relevant_ids)
    return next((rank for rank, chunk_id in enumerate(retrieved_ids, start=1) if chunk_id in relevant), None)


def metric_record(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> dict[str, float | int | None]:
    """Compute the per-query metrics used by the retrieval experiment."""
    rank = first_relevant_rank(retrieved_ids, relevant_ids)
    return {
        "first_relevant_rank": rank,
        "recall_at_5": int(rank is not None and rank <= 5),
        "recall_at_10": int(rank is not None and rank <= 10),
        "reciprocal_rank": round(1 / rank, 8) if rank else 0.0,
    }


def aggregate(records: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    """Average query metrics for one retriever."""
    total = len(records)
    if not total:
        raise ValueError("Cannot aggregate an empty evaluation set.")
    return {
        "query_count": total,
        "recall_at_5": round(sum(record["recall_at_5"] for record in records) / total, 6),
        "recall_at_10": round(sum(record["recall_at_10"] for record in records) / total, 6),
        "mrr": round(sum(record["reciprocal_rank"] for record in records) / total, 6),
    }


def load_queries(query_path: Path = DEFAULT_QUERIES) -> list[dict[str, Any]]:
    """Load and validate the manually curated benchmark."""
    with query_path.open("r", encoding="utf-8") as source:
        queries = json.load(source)
    if not isinstance(queries, list) or not queries:
        raise ValueError("Benchmark must be a non-empty JSON list.")
    ids = [query.get("query_id") for query in queries]
    if len(ids) != len(set(ids)) or not all(isinstance(query.get("query"), str) and query.get("relevant_chunk_ids") for query in queries):
        raise ValueError("Every benchmark query needs a unique query_id, text, and relevant_chunk_ids.")
    return queries


def evaluate(
    queries: Sequence[dict[str, Any]],
    retrievers: dict[str, Callable[[str, int], Sequence[dict[str, Any]]]],
    *,
    top_k: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float | int]]]:
    """Evaluate named retrievers and return deterministic per-query and aggregate data."""
    per_query: list[dict[str, Any]] = []
    metric_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in retrievers}
    for query in queries:
        evaluation: dict[str, Any] = {
            "query_id": query["query_id"],
            "category": query.get("category"),
            "query": query["query"],
            "relevant_chunk_ids": query["relevant_chunk_ids"],
            "retrievers": {},
        }
        for name, retrieve in retrievers.items():
            results = retrieve(query["query"], top_k)
            retrieved_ids = [str(result["chunk_id"]) for result in results]
            metrics = metric_record(retrieved_ids, query["relevant_chunk_ids"])
            evaluation["retrievers"][name] = {**metrics, "retrieved_chunk_ids": retrieved_ids}
            metric_rows[name].append(metrics)
        per_query.append(evaluation)
    return per_query, {name: aggregate(rows) for name, rows in metric_rows.items()}


def write_results(per_query: Sequence[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    """Write stable JSON artifacts that can be versioned or compared across runs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "per_query_results.json").write_text(json.dumps(per_query, indent=2), encoding="utf-8")
    (output_dir / "aggregate_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BM25, dense, and hybrid legal retrieval.")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=20)
    args = parser.parse_args()

    queries = load_queries(args.queries)
    model = dense.create_embedding_model()
    retrievers = {
        "bm25": lambda query, limit: bm25.search(query, top_k=limit),
        "dense": lambda query, limit: dense.search(query, model, top_k=limit),
        "hybrid": lambda query, limit: hybrid.search(query, top_k=limit, candidate_k=args.candidate_k, model=model),
    }
    per_query, summary = evaluate(queries, retrievers, top_k=args.top_k)
    write_results(per_query, summary, args.output_dir)
    print(json.dumps(summary, indent=2))
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()