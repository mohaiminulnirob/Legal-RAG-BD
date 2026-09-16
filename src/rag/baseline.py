"""Conventional Hybrid-RRF baseline RAG with explicit evidence citations."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from src.rag.context import build_context, citation_label
from src.rag.llm import DEFAULT_MODEL, LLMClient, OpenAIResponsesLLM
from src.rag.prompt import SYSTEM_PROMPT, build_user_prompt
from src.retrieval import hybrid, reranker


@dataclass(frozen=True)
class BaselineResponse:
    """A generated answer plus the exact legal evidence supplied to the LLM."""

    answer: str
    citations: list[dict[str, str]]
    evidence: list[dict[str, Any]]
    model: str
    insufficient_evidence: bool


def citation_records(evidence: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """Create a compact, machine-traceable citation list."""
    return [
        {
            "chunk_id": str(item.get("chunk_id", "unknown")),
            "citation": citation_label(item),
            "source_url": str(item.get("source_url") or ""),
        }
        for item in evidence
    ]


def answer_query(
    query: str,
    llm: LLMClient,
    *,
    top_k: int = 5,
    candidate_k: int = 20,
    use_reranker: bool = False,
    retrieval_model: Any | None = None,
    reranker_model: Any | None = None,
) -> BaselineResponse:
    """Retrieve Hybrid RRF evidence, then produce a conventional grounded answer."""
    if not query.strip():
        raise ValueError("The query must not be empty.")
    if top_k < 1 or candidate_k < top_k:
        raise ValueError("candidate_k must be greater than or equal to a positive top_k.")

    evidence = hybrid.search(query, top_k=candidate_k if use_reranker else top_k, candidate_k=candidate_k, model=retrieval_model)
    if use_reranker:
        evidence = reranker.rerank_candidates(query, evidence, reranker_model or reranker.create_reranker(), top_k=top_k)

    if not evidence:
        return BaselineResponse(
            answer="The retrieved legal evidence is insufficient to answer this question reliably.",
            citations=[],
            evidence=[],
            model=getattr(llm, "model", DEFAULT_MODEL),
            insufficient_evidence=True,
        )

    context = build_context(evidence)
    answer = llm.generate(SYSTEM_PROMPT, build_user_prompt(query, context))
    return BaselineResponse(
        answer=answer,
        citations=citation_records(evidence),
        evidence=list(evidence),
        model=getattr(llm, "model", DEFAULT_MODEL),
        insufficient_evidence=False,
    )


def print_response(response: BaselineResponse) -> None:
    """Print answer content followed by traceable evidence references."""
    print(response.answer)
    print("\nEvidence supplied to the model:")
    for citation in response.citations:
        source = f" — {citation['source_url']}" if citation["source_url"] else ""
        print(f"- {citation['citation']} [{citation['chunk_id']}]{source}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Hybrid-RRF conventional legal RAG baseline.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--model", default=os.getenv("BASELINE_RAG_MODEL", DEFAULT_MODEL))
    parser.add_argument("--use-reranker", action="store_true", help="Experimental: rerank hybrid candidates before generation.")
    args = parser.parse_args()
    response = answer_query(
        args.query,
        OpenAIResponsesLLM(args.model),
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        use_reranker=args.use_reranker,
    )
    print_response(response)


if __name__ == "__main__":
    main()
