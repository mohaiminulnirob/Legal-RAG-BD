"""Tests for deterministic Phase 10 plan construction."""

from __future__ import annotations

import unittest

from src.analysis.query_analyzer import analyze_query
from src.reasoning.planner import build_reasoning_plan
from src.reasoning.schemas import IssueType, StepStatus


def plan_for(query: str):
    return build_reasoning_plan(analyze_query(query))


class ReasoningPlannerTests(unittest.TestCase):
    def test_offence_and_punishment_are_separate_ordered_steps(self) -> None:
        plan = plan_for("What punishment applies when a person intentionally kills another?")
        self.assertEqual([step.issue_type for step in plan.steps], [IssueType.OFFENCE, IssueType.PUNISHMENT])
        self.assertEqual(plan.steps[1].depends_on, ("step_1",))
        self.assertEqual(plan.steps[0].status, StepStatus.PENDING)
        self.assertIn("intentionally kills", plan.steps[0].retrieval_query)
        self.assertNotIn("302", plan.steps[1].retrieval_query)

    def test_explicit_defence_adds_a_dependent_exception_step(self) -> None:
        plan = plan_for("A person caused a death in self-defence. Could an exception apply?")
        types = [step.issue_type for step in plan.steps]
        self.assertEqual(types, [IssueType.OFFENCE, IssueType.EXCEPTION])
        self.assertEqual(plan.steps[1].depends_on, ("step_1",))

    def test_procedural_question_creates_only_a_procedural_step(self) -> None:
        plan = plan_for("Can this evidence be admitted in court?")
        self.assertEqual([step.issue_type for step in plan.steps], [IssueType.PROCEDURAL])

    def test_section_lookup_and_incomplete_case_get_retrieval_plans(self) -> None:
        section_plan = plan_for("What does section 302 say?")
        case_plan = plan_for("A person took another person's property. Is this theft?")
        self.assertEqual([step.issue_type for step in section_plan.steps], [IssueType.PROVISION])
        self.assertEqual([step.issue_type for step in case_plan.steps], [IssueType.OFFENCE])
        self.assertIn("offence_elements", case_plan.steps[0].required_evidence_types)

    def test_out_of_scope_query_has_no_steps(self) -> None:
        plan = plan_for("How can I learn Python?")
        self.assertEqual(plan.steps, ())
        self.assertEqual(plan.schema_version, "1.0")

    def test_plan_serialization_has_no_answer_or_sufficiency_decision(self) -> None:
        plan = plan_for("What punishment applies for murder?")
        result = plan.to_dict()
        self.assertEqual(result["schema_version"], "1.0")
        self.assertNotIn("answer", result)
        self.assertNotIn("sufficiency", result)
        self.assertTrue(all(step["retrieval_query"] for step in result["steps"]))


if __name__ == "__main__":
    unittest.main()
