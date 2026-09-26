import unittest

from src.reasoning.clarification import build_clarification_requests
from src.reasoning.next_action import NextAction, NextActionDecision
from src.reasoning.sufficiency import SufficiencyStatus


def decision(action=NextAction.CLARIFY, targets=("whether the killing was intentional",)):
    return NextActionDecision(
        step_id="step-1",
        sufficiency=SufficiencyStatus.PARTIALLY_SUPPORTED,
        action=action,
        reason_code="user_fact_missing",
        retrieval_attempt=0,
        clarification_attempt=1,
        clarification_targets=targets,
    )


class ClarificationTests(unittest.TestCase):
    def test_target_becomes_neutral_question(self):
        requests = build_clarification_requests(decision())
        self.assertEqual(len(requests), 1)
        self.assertEqual(
            requests[0].question,
            "Could you clarify whether the killing was intentional?",
        )

    def test_multiple_questions_are_bounded(self):
        targets = ("intent", "circumstances", "time", "location")
        requests = build_clarification_requests(decision(targets=targets), max_questions=2)
        self.assertEqual(len(requests), 2)

    def test_non_clarify_action_is_rejected(self):
        with self.assertRaises(ValueError):
            build_clarification_requests(decision(NextAction.CONTINUE))

    def test_missing_target_is_rejected(self):
        with self.assertRaises(ValueError):
            build_clarification_requests(decision(targets=()))


if __name__ == "__main__":
    unittest.main()
