"""Cross-encoder reranking for hybrid legal-retrieval candidates.

Example:
    python -m src.retrieval.reranker search \
        --query "A person intentionally killed another person" --top-k 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Protocol, Sequence

from src.retrieval import hybrid


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MODEL_CACHE = Path("data/model_cache")
DEFAULT_CANDIDATE_K = 20


class PairScorer(Protocol):
    """Minimal interface provided by a sentence-transformers CrossEncoder."""

    def predict(self, sentences: Sequence[tuple[str, str]], **kwargs: Any) -> Any: ...


def create_reranker(model_name: str = MODEL_NAME) -> PairScorer:
    """Load the local cross-encoder, downloading it to the project cache if needed."""
    from sentence_transformers import CrossEncoder

    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    return CrossEncoder(model_name, cache_folder=str(MODEL_CACHE))


def rerank_candidates(
    query: str,
    candidates: Sequence[dict[str, Any]],
    model: PairScorer,
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Score query/section pairs and return the highest-ranked cited evidence."""
    if not query.strip():
        raise ValueError("The query must not be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if len({candidate.get("chunk_id") for candidate in candidates}) != len(candidates):
        raise ValueError("Candidates must have unique chunk_id values before reranking.")

    pairs = [(query, str(candidate.get("section_content", ""))) for candidate in candidates]
    raw_scores = model.predict(pairs, show_progress_bar=False) if pairs else []
    scores = raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)
    scored = [{**candidate, "reranker_score": round(float(score), 8)} for candidate, score in zip(candidates, scores)]
    # Stable sort preserves original RRF order for exact score ties.
    return sorted(scored, key=lambda candidate: -float(candidate["reranker_score"]))[:top_k]


def search(
    query: str,
    *,
    top_k: int = 5,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    model: PairScorer | None = None,
    **hybrid_options: Any,
) -> list[dict[str, Any]]:
    """Retrieve hybrid candidates then rerank them with a cross-encoder."""
    if candidate_k < 1:
        raise ValueError("candidate_k must be at least 1.")
    candidates = hybrid.search(query, top_k=candidate_k, candidate_k=candidate_k, **hybrid_options)
    return rerank_candidates(query, candidates, model or create_reranker(), top_k=top_k)


def print_results(query: str, results: Sequence[dict[str, Any]]) -> None:
    """Print reranked legal evidence with provenance and both retrieval scores."""
    print(f'Query: "{query}"')
    for rank, result in enumerate(results, start=1):
        print(
            f"\n{rank}. {result.get('act_title', 'Unknown Act')} — section {result.get('section_id', '?')}"
            f" (reranker: {result['reranker_score']:.6f}; RRF: {result.get('rrf_score', 0):.6f})"
        )
        print(f"   ID: {result['chunk_id']}")
        print(f"   {str(result.get('section_content', '')).strip()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank hybrid legal evidence with a cross-encoder.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    search_parser = subcommands.add_parser("search", help="Retrieve and rerank legal evidence.")
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    search_parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    print_results(args.query, search(args.query, top_k=args.top_k, candidate_k=args.candidate_k, model=create_reranker(args.model)))


if __name__ == "__main__":
    main()
