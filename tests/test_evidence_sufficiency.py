"""Deterministic evidence-rule and per-step sufficiency tests."""

from __future__ import annotations

import unittest

from src.reasoning.schemas import IssueType, ReasoningPlan, ReasoningStep
from src.reasoning.sufficiency import (
    ReasonCode,
    SufficiencyStatus,
    assess_plan,
    assess_step,
)


def evidence(chunk_id: str, content: str, **extra):
    return {
        "chunk_id": chunk_id,
        "act_title": "Penal Code",
        "section_id": "302",
        "section_content": content,
        "source_file": "act-print-11.json",
        "source_url": "https://example.test/act-print-11",
        **extra,
    }


def step(issue: IssueType, requirements: tuple[str, ...], records=(), query="murder"):
    return ReasoningStep(
        step_id="step_1",
        issue_type=issue,
        objective="Retrieve the relevant legal provision",
        retrieval_query=f"statutory evidence relevant to: {query}",
        required_evidence_types=requirements,
        evidence=tuple(records),
    )


class EvidenceSufficiencyTests(unittest.TestCase):
    def test_supported_requires_direct_required_type_match(self) -> None:
        current = step(
            IssueType.PUNISHMENT,
            ("punishment_provision",),
            [evidence("penal-302", "Whoever commits murder shall be punished with death or imprisonment for life.", bm25_rank=1, dense_rank=2, rrf_score=0.9)],
        )
        result = assess_step(current)
        self.assertEqual(result.status, SufficiencyStatus.SUPPORTED)
        self.assertEqual(result.relevant_evidence_ids, ("penal-302",))
        self.assertIn(ReasonCode.REQUIRED_TYPE_MATCH, result.reason_codes)
        self.assertIn(ReasonCode.MULTIPLE_RETRIEVERS_AGREE, result.reason_codes)
        self.assertFalse(result.requires_clarification)

    def test_partial_status_records_each_unmatched_requirement(self) -> None:
        current = step(
            IssueType.OFFENCE,
            ("offence_definition", "offence_elements"),
            [evidence("penal-definition", "Culpable homicide is murder.")],
        )
        result = assess_step(current)
        self.assertEqual(result.status, SufficiencyStatus.PARTIALLY_SUPPORTED)
        self.assertEqual(result.missing_requirements, ("offence_elements",))
        self.assertTrue(result.requires_clarification)

    def test_unsupported_is_not_inferred_from_a_high_retrieval_score(self) -> None:
        current = step(
            IssueType.PUNISHMENT,
            ("punishment_provision",),
            [evidence("unrelated", "This provision describes court procedure.", rrf_score=999.0)],
        )
        result = assess_step(current)
        self.assertEqual(result.status, SufficiencyStatus.UNSUPPORTED)
        self.assertIn(ReasonCode.LOW_RETRIEVAL_SUPPORT, result.reason_codes)
        self.assertNotIn(ReasonCode.REQUIRED_TYPE_MATCH, result.reason_codes)

    def test_ambiguous_punishment_relationship_is_uncertain(self) -> None:
        current = step(
            IssueType.PUNISHMENT,
            ("punishment_provision",),
            [evidence("attempt", "If the act caused death the person would be guilty of murder, and shall be punished with imprisonment.")],
        )
        result = assess_step(current)
        self.assertEqual(result.status, SufficiencyStatus.UNCERTAIN)
        self.assertIn(ReasonCode.AMBIGUOUS_REQUIREMENT_MATCH, result.reason_codes)
        self.assertEqual(result.missing_requirements, ("punishment_provision",))

    def test_empty_and_malformed_evidence_are_uncertain(self) -> None:
        requirement = ("punishment_provision",)
        empty_result = assess_step(step(IssueType.PUNISHMENT, requirement))
        malformed_result = assess_step(
            step(IssueType.PUNISHMENT, requirement, [{"chunk_id": "missing-content"}])
        )
        self.assertEqual(empty_result.status, SufficiencyStatus.UNCERTAIN)
        self.assertIn(ReasonCode.NO_EVIDENCE, empty_result.reason_codes)
        self.assertTrue(empty_result.requires_clarification)
        self.assertEqual(malformed_result.status, SufficiencyStatus.UNCERTAIN)
        self.assertIn(ReasonCode.MALFORMED_EVIDENCE, malformed_result.reason_codes)

    def test_conflicting_records_with_same_id_are_uncertain(self) -> None:
        current = step(
            IssueType.PUNISHMENT,
            ("punishment_provision",),
            [evidence("duplicate", "Unrelated civil procedure."), evidence("duplicate", "Unrelated contract rule.")],
        )
        result = assess_step(current)
        self.assertEqual(result.status, SufficiencyStatus.UNCERTAIN)
        self.assertIn(ReasonCode.CONFLICTING_EVIDENCE_RECORDS, result.reason_codes)

    def test_plan_assesses_each_step_without_flattening_evidence(self) -> None:
        first = step(
            IssueType.PUNISHMENT,
            ("punishment_provision",),
            [evidence("penal-302", "Whoever commits murder shall be punished with death.")],
        )
        second = ReasoningStep(
            step_id="step_2",
            issue_type=IssueType.EXCEPTION,
            objective="Retrieve exceptions",
            retrieval_query="statutory exceptions relevant to murder",
            required_evidence_types=("exception_or_defence_provision",),
            depends_on=("step_1",),
            evidence=(evidence("exception-1", "Exception 1. Culpable homicide is not murder if grave and sudden provocation applies."),),
        )
        plan = ReasoningPlan("murder question", "multi_issue", (first, second))
        results = assess_plan(plan)
        self.assertEqual([result.step_id for result in results], ["step_1", "step_2"])
        self.assertEqual([result.status for result in results], [SufficiencyStatus.SUPPORTED, SufficiencyStatus.SUPPORTED])
        self.assertEqual(results[0].evidence_count, 1)
        self.assertEqual(results[1].relevant_evidence_ids, ("exception-1",))
        self.assertNotIn("answer", results[0].to_dict())


if __name__ == "__main__":
    unittest.main()
