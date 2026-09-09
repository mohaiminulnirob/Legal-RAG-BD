"""Fuse BM25 and dense legal retrieval using Reciprocal Rank Fusion (RRF).

Example:
    python -m src.retrieval.hybrid search \
        --query "A person intentionally killed another person" --top-k 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from src.retrieval import bm25, dense


DEFAULT_CANDIDATE_K = 20
DEFAULT_RRF_K = 60


def rrf_fuse(
    bm25_results: Sequence[dict[str, Any]],
    dense_results: Sequence[dict[str, Any]],
    *,
    top_k: int = 10,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[dict[str, Any]]:
    """Merge ranked result lists by ``chunk_id`` and score them with RRF.

    Ranks are one-based. The original retriever scores and ranks are retained for
    later evaluation and for transparent legal-evidence inspection.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative.")

    merged: dict[str, dict[str, Any]] = {}

    def add_results(results: Sequence[dict[str, Any]], source: str, score_key: str) -> None:
        for rank, result in enumerate(results, start=1):
            chunk_id = result.get("chunk_id")
            if not chunk_id:
                raise ValueError(f"{source} result at rank {rank} has no chunk_id.")

            entry = merged.setdefault(str(chunk_id), {"chunk_id": str(chunk_id), "rrf_score": 0.0})
            # Prefer fields already present, but let the exact BM25 section content
            # replace dense's embedding text when both sources return the document.
            for key, value in result.items():
                if value is not None and (key not in entry or source == "bm25"):
                    entry[key] = value
            entry[f"{source}_rank"] = rank
            entry[f"{source}_score"] = result.get(score_key)
            entry["rrf_score"] += 1 / (rrf_k + rank)

    add_results(bm25_results, "bm25", "score")
    add_results(dense_results, "dense", "similarity")

    ranked = sorted(
        merged.values(),
        key=lambda result: (
            -float(result["rrf_score"]),
            min(result.get("bm25_rank", float("inf")), result.get("dense_rank", float("inf"))),
            result["chunk_id"],
        ),
    )
    for result in ranked:
        result["rrf_score"] = round(float(result["rrf_score"]), 8)
    return ranked[:top_k]


def search(
    query: str,
    *,
    top_k: int = 10,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    rrf_k: int = DEFAULT_RRF_K,
    bm25_index: Path = bm25.DEFAULT_INDEX,
    dense_database: Path = dense.DEFAULT_DATABASE,
    model: dense.TextEmbedder | None = None,
) -> list[dict[str, Any]]:
    """Retrieve and fuse lexical and semantic legal evidence for a query."""
    if candidate_k < 1:
        raise ValueError("candidate_k must be at least 1.")
    embedding_model = model or dense.create_embedding_model()
    bm25_results = bm25.search(query, bm25_index, candidate_k)
    dense_results = dense.search(query, embedding_model, database_path=dense_database, top_k=candidate_k)
    return rrf_fuse(bm25_results, dense_results, top_k=top_k, rrf_k=rrf_k)


def format_rank(rank: int | None) -> str:
    """Display a retriever rank or an em dash for an absent result."""
    return str(rank) if rank is not None else "—"


def print_results(query: str, results: Sequence[dict[str, Any]]) -> None:
    """Render fused legal evidence and its component ranks for analysis."""
    print(f'Query: "{query}"')
    for rank, result in enumerate(results, start=1):
        print(
            f"\n{rank}. {result.get('act_title', 'Unknown Act')} — section "
            f"{result.get('section_id', '?')} (RRF: {result['rrf_score']:.6f})"
        )
        print(
            f"   BM25 rank: {format_rank(result.get('bm25_rank'))}"
            f" | score: {result.get('bm25_score', '—')}"
            f" | Dense rank: {format_rank(result.get('dense_rank'))}"
            f" | similarity: {result.get('dense_score', '—')}"
        )
        print(f"   ID: {result['chunk_id']}")
        print(f"   {str(result.get('section_content', '')).strip()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse BM25 and dense legal retrieval with RRF.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    search_parser = subcommands.add_parser("search", help="Retrieve fused legal evidence.")
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--top-k", type=int, default=10)
    search_parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    search_parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    search_parser.add_argument("--bm25-index", type=Path, default=bm25.DEFAULT_INDEX)
    search_parser.add_argument("--dense-database", type=Path, default=dense.DEFAULT_DATABASE)
    search_parser.add_argument("--model", default=dense.MODEL_NAME)
    args = parser.parse_args()

    model = dense.create_embedding_model(args.model)
    results = search(
        args.query,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
        bm25_index=args.bm25_index,
        dense_database=args.dense_database,
        model=model,
    )
    print_results(args.query, results)


if __name__ == "__main__":
    main()
