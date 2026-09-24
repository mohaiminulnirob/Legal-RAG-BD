"""Run an independent Hybrid RRF retrieval for each reasoning-plan step."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Sequence

from src.reasoning.schemas import ReasoningPlan, StepStatus


SearchFunction = Callable[..., Sequence[dict[str, Any]]]


def retrieve_plan(
    plan: ReasoningPlan,
    *,
    top_k: int = 5,
    candidate_k: int = 20,
    search_fn: SearchFunction | None = None,
    retrieval_model: Any | None = None,
    bm25_index: Path | None = None,
    dense_database: Path | None = None,
) -> ReasoningPlan:
    """Retrieve candidates separately and attach full records to their steps.

    This records retrieval completion only. It does not assess relevance,
    sufficiency, legal support, or whether any conclusion follows.
    """
    if top_k < 1 or candidate_k < top_k:
        raise ValueError("candidate_k must be greater than or equal to a positive top_k.")
    if not plan.steps:
        return plan
    if search_fn is None:
        from src.retrieval import dense
        from src.retrieval.hybrid import search as hybrid_search

        search_options: dict[str, Any] = {}
        if bm25_index is not None:
            search_options["bm25_index"] = bm25_index
        if dense_database is not None:
            search_options["dense_database"] = dense_database
        search = partial(hybrid_search, **search_options)
        if retrieval_model is None:
            retrieval_model = dense.create_embedding_model()
    else:
        search = search_fn
    retrieved_steps = []
    for step in plan.steps:
        evidence = search(
            step.retrieval_query,
            top_k=top_k,
            candidate_k=candidate_k,
            model=retrieval_model,
        )
        retrieved_steps.append(
            replace(
                step,
                status=StepStatus.RETRIEVAL_COMPLETE,
                evidence=tuple(dict(item) for item in evidence),
            )
        )
    return replace(plan, steps=tuple(retrieved_steps))
