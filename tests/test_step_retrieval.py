"""Tests for per-step retrieval and provenance retention."""

from __future__ import annotations

import unittest

from src.analysis.query_analyzer import analyze_query
from src.reasoning.planner import build_reasoning_plan
from src.reasoning.schemas import StepStatus
from src.reasoning.step_retriever import retrieve_plan


class StepRetrievalTests(unittest.TestCase):
    def test_each_step_gets_an_independent_hybrid_request_and_its_evidence(self) -> None:
        plan = build_reasoning_plan(analyze_query("What punishment applies when a person intentionally kills another?"))
        calls: list[dict[str, object]] = []
        model = object()

        def fake_search(query: str, **kwargs):
            calls.append({"query": query, **kwargs})
            return [{
                "chunk_id": f"chunk-{len(calls)}",
                "act_title": "Example Act",
                "section_id": "123",
                "rrf_score": 0.02,
                "bm25_rank": 2,
                "dense_rank": 3,
                "source_url": "https://example.invalid/source",
            }]

        result = retrieve_plan(plan, search_fn=fake_search, retrieval_model=model)
        self.assertEqual(len(calls), len(plan.steps))
        self.assertEqual([call["query"] for call in calls], [step.retrieval_query for step in plan.steps])
        self.assertTrue(all(call["top_k"] == 5 and call["candidate_k"] == 20 for call in calls))
        self.assertTrue(all(call["model"] is model for call in calls))
        for index, step in enumerate(result.steps, start=1):
            self.assertEqual(step.status, StepStatus.RETRIEVAL_COMPLETE)
            self.assertEqual(step.evidence[0]["chunk_id"], f"chunk-{index}")
            self.assertEqual(step.evidence[0]["source_url"], "https://example.invalid/source")
            self.assertEqual(step.evidence[0]["rrf_score"], 0.02)

    def test_empty_retrieval_is_recorded_without_a_sufficiency_label(self) -> None:
        plan = build_reasoning_plan(analyze_query("What does section 302 say?"))
        result = retrieve_plan(plan, search_fn=lambda query, **kwargs: [])
        self.assertEqual(result.steps[0].status, StepStatus.RETRIEVAL_COMPLETE)
        self.assertEqual(result.steps[0].evidence, ())

    def test_candidate_settings_are_validated(self) -> None:
        plan = build_reasoning_plan(analyze_query("What does section 302 say?"))
        with self.assertRaises(ValueError):
            retrieve_plan(plan, top_k=5, candidate_k=4, search_fn=lambda query, **kwargs: [])

    def test_empty_plan_does_not_load_retrieval_dependencies(self) -> None:
        plan = build_reasoning_plan(analyze_query("How can I learn Python?"))
        self.assertIs(retrieve_plan(plan), plan)


if __name__ == "__main__":
    unittest.main()
