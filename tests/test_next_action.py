import unittest

from src.reasoning.next_action import (
    ActionReason,
    NextAction,
    NextActionState,
    decide_next_action,
)
from src.reasoning.sufficiency import EvidenceAssessment, ReasonCode as SufficiencyReasonCode, SufficiencyStatus


def assessment(status: SufficiencyStatus, missing=()) -> EvidenceAssessment:
    return EvidenceAssessment(
        step_id="step-1",
        status=status,
        evidence_count=0,
        relevant_evidence_ids=(),
        missing_requirements=tuple(missing),
        support_signals=(),
        reason_codes=(),
        requires_clarification=bool(missing),
    )


class NextActionTests(unittest.TestCase):
    def test_supported_continues(self):
        result = decide_next_action(assessment(SufficiencyStatus.SUPPORTED))
        self.assertEqual(result.action, NextAction.CONTINUE)

    def test_partial_retrieves_more_with_attempt_number(self):
        result = decide_next_action(assessment(SufficiencyStatus.PARTIALLY_SUPPORTED, ("punishment_provision",)))
        self.assertEqual(result.action, NextAction.RETRIEVE_MORE)
        self.assertEqual(result.retrieval_attempt, 1)

    def test_explicit_missing_user_fact_clarifies_before_retrieval(self):
        state = NextActionState(missing_user_facts=("whether the act was intentional",))
        result = decide_next_action(assessment(SufficiencyStatus.PARTIALLY_SUPPORTED), state)
        self.assertEqual(result.action, NextAction.CLARIFY)
        self.assertEqual(result.clarification_targets, ("whether the act was intentional",))
        self.assertEqual(result.clarification_attempt, 1)

    def test_unsupported_gets_bounded_retry(self):
        state = NextActionState(retrieval_attempts=1, max_retrieval_attempts=2)
        result = decide_next_action(assessment(SufficiencyStatus.UNSUPPORTED), state)
        self.assertEqual(result.action, NextAction.RETRIEVE_MORE)
        self.assertEqual(result.retrieval_attempt, 2)

    def test_uncertain_is_acknowledged_without_retry(self):
        result = decide_next_action(assessment(SufficiencyStatus.UNCERTAIN))
        self.assertEqual(result.action, NextAction.ACKNOWLEDGE_UNCERTAINTY)

    def test_empty_retrieval_is_retryable_within_budget(self):
        empty = EvidenceAssessment(
            step_id="step-1",
            status=SufficiencyStatus.UNCERTAIN,
            evidence_count=0,
            relevant_evidence_ids=(),
            missing_requirements=("offence_definition",),
            support_signals=(),
            reason_codes=(SufficiencyReasonCode.NO_EVIDENCE,),
            requires_clarification=True,
        )
        result = decide_next_action(empty)
        self.assertEqual(result.action, NextAction.RETRIEVE_MORE)

    def test_uncertain_with_explicit_critical_fact_missing_clarifies(self):
        state = NextActionState(missing_user_facts=("whether the person acted intentionally",))
        result = decide_next_action(assessment(SufficiencyStatus.UNCERTAIN), state)
        self.assertEqual(result.action, NextAction.CLARIFY)

    def test_exhausted_clarification_budget_acknowledges_uncertainty(self):
        state = NextActionState(
            clarification_attempts=1,
            max_clarification_attempts=1,
            missing_user_facts=("intent",),
        )
        result = decide_next_action(assessment(SufficiencyStatus.UNSUPPORTED), state)
        self.assertEqual(result.action, NextAction.ACKNOWLEDGE_UNCERTAINTY)
        self.assertEqual(result.reason_code, ActionReason.CLARIFICATION_LIMIT_REACHED)

    def test_exhausted_retrieval_budget_acknowledges_uncertainty(self):
        state = NextActionState(retrieval_attempts=1, max_retrieval_attempts=1)
        result = decide_next_action(assessment(SufficiencyStatus.PARTIALLY_SUPPORTED), state)
        self.assertEqual(result.action, NextAction.ACKNOWLEDGE_UNCERTAINTY)
        self.assertEqual(result.reason_code, ActionReason.RETRIEVAL_LIMIT_REACHED)

    def test_explicit_stop_takes_precedence(self):
        result = decide_next_action(
            assessment(SufficiencyStatus.SUPPORTED), NextActionState(stop_requested=True)
        )
        self.assertEqual(result.action, NextAction.STOP)

    def test_state_rejects_invalid_attempt_limits(self):
        with self.assertRaises(ValueError):
            NextActionState(retrieval_attempts=2, max_retrieval_attempts=1)

    def test_decision_serialization_is_versioned_and_auditable(self):
        result = decide_next_action(assessment(SufficiencyStatus.UNSUPPORTED))
        encoded = result.to_dict()
        self.assertEqual(encoded["schema_version"], "1.0")
        self.assertEqual(encoded["reason_code"], "retrieval_retry_available")
        self.assertEqual(encoded["action"], "retrieve_more")


if __name__ == "__main__":
    unittest.main()
