"""CLI for deterministic reasoning planning and step-level retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analysis.query_analyzer import analyze_query
from src.reasoning.planner import build_reasoning_plan
from src.reasoning.step_retriever import retrieve_plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a deterministic legal reasoning plan and retrieve evidence per step."
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--model", help="Optional local embedding model override.")
    parser.add_argument("--bm25-index", type=Path, default=Path("data/processed/bm25_index.pkl"))
    parser.add_argument("--dense-database", type=Path, default=Path("chroma_db"))
    parser.add_argument("--plan-only", action="store_true", help="Print the plan without loading retrieval models or indexes.")
    args = parser.parse_args()

    analysis = analyze_query(args.query)
    plan = build_reasoning_plan(analysis)
    if not args.plan_only and plan.steps:
        from src.retrieval import dense

        model = dense.create_embedding_model(args.model or dense.MODEL_NAME)
        plan = retrieve_plan(
            plan,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            retrieval_model=model,
            bm25_index=args.bm25_index,
            dense_database=args.dense_database,
        )
    print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
