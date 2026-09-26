import unittest
from dataclasses import replace

from src.reasoning.next_action import ActionReason, NextAction, NextActionDecision
from src.reasoning.schemas import IssueType, ReasoningStep
from src.reasoning.sufficiency import EvidenceAssessment, SufficiencyStatus
from src.reasoning.uncertainty import build_uncertainty_acknowledgement


def make_case(action=NextAction.ACKNOWLEDGE_UNCERTAINTY):
    step = ReasoningStep(
        step_id="step-1",
        issue_type=IssueType.OFFENCE,
        objective="identify the applicable offence",
        retrieval_query="offence elements",
        required_evidence_types=("offence_definition", "offence_elements"),
        evidence=(
            {
                "chunk_id": "act_1_section_1",
                "section_content": "Whoever intentionally causes death commits an offence.",
                "act_title": "Penal Code",
                "section_id": "300",
                "source_file": "act-1.json",
                "source_url": "https://example.test/300",
            },
        ),
    )
    assessment = EvidenceAssessment(
        step_id="step-1",
        status=SufficiencyStatus.PARTIALLY_SUPPORTED,
        evidence_count=1,
        relevant_evidence_ids=("act_1_section_1",),
        missing_requirements=("offence_elements",),
        support_signals=(),
        reason_codes=(),
        requires_clarification=True,
    )
    decision = NextActionDecision(
        step_id="step-1",
        sufficiency=assessment.status,
        action=action,
        reason_code=ActionReason.EVIDENCE_INSUFFICIENT,
        retrieval_attempt=1,
        clarification_attempt=1,
    )
    return step, assessment, decision


class UncertaintyTests(unittest.TestCase):
    def test_acknowledgement_separates_support_missing_and_limit(self):
        step, assessment, decision = make_case()
        result = build_uncertainty_acknowledgement(
            decision,
            assessment,
            step,
            unresolved_user_facts=("whether the act was intentional",),
        )
        self.assertEqual(len(result.supported_evidence), 1)
        self.assertEqual(result.missing_requirements, ("offence_elements",))
        self.assertEqual(result.unresolved_user_facts, ("whether the act was intentional",))
        rendered = result.render()
        self.assertIn("Available evidence:", rendered)
        self.assertIn("Missing or unresolved:", rendered)
        self.assertIn("cannot be established", rendered)

    def test_no_relevant_record_is_reported_as_none(self):
        step, assessment, decision = make_case()
        assessment = EvidenceAssessment(
            step_id=assessment.step_id,
            status=SufficiencyStatus.UNCERTAIN,
            evidence_count=0,
            relevant_evidence_ids=(),
            missing_requirements=("offence_definition",),
            support_signals=(),
            reason_codes=(),
            requires_clarification=True,
        )
        decision = replace(decision, sufficiency=SufficiencyStatus.UNCERTAIN)
        result = build_uncertainty_acknowledgement(decision, assessment, step)
        self.assertEqual(result.supported_evidence, ())
        self.assertIn("no verified supporting passage", result.render())

    def test_wrong_action_or_step_is_rejected(self):
        step, assessment, decision = make_case(NextAction.CONTINUE)
        with self.assertRaises(ValueError):
            build_uncertainty_acknowledgement(decision, assessment, step)

    def test_excerpt_and_record_counts_are_bounded(self):
        step, assessment, decision = make_case()
        result = build_uncertainty_acknowledgement(
            decision,
            assessment,
            step,
            max_excerpt_chars=12,
        )
        self.assertLessEqual(len(result.supported_evidence[0].excerpt), 12)

    def test_zero_evidence_limit_produces_no_excerpts(self):
        step, assessment, decision = make_case()
        result = build_uncertainty_acknowledgement(
            decision,
            assessment,
            step,
            max_evidence_records=0,
        )
        self.assertEqual(result.supported_evidence, ())


if __name__ == "__main__":
    unittest.main()
