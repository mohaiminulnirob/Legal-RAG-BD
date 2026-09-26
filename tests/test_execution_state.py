import json
import unittest

from src.reasoning.execution_state import ExecutionState
from src.reasoning.clarification import build_clarification_requests
from src.reasoning.next_action import NextAction, NextActionDecision, decide_next_action
from src.reasoning.schemas import IssueType, ReasoningPlan, ReasoningStep
from src.reasoning.sufficiency import EvidenceAssessment, SufficiencyStatus


def supported_assessment(step_id, status=SufficiencyStatus.SUPPORTED):
    return EvidenceAssessment(
        step_id=step_id,
        status=status,
        evidence_count=1,
        relevant_evidence_ids=("chunk-1",) if status == SufficiencyStatus.SUPPORTED else (),
        missing_requirements=() if status == SufficiencyStatus.SUPPORTED else ("offence_definition",),
        support_signals=(),
        reason_codes=(),
        requires_clarification=False,
    )


def make_plan(count=2):
    query = "What offence and punishment apply?"
    steps = tuple(
        ReasoningStep(
            step_id=f"step_{i}",
            issue_type=IssueType.OFFENCE,
            objective=f"Objective {i}",
            retrieval_query=f"Evidence {i}",
            required_evidence_types=("offence_definition",),
        )
        for i in range(1, count + 1)
    )
    return ReasoningPlan(query=query, analysis_kind="general_question", steps=steps)


def continue_decision(step_id):
    return NextActionDecision(
        step_id=step_id,
        sufficiency=SufficiencyStatus.SUPPORTED,
        action=NextAction.CONTINUE,
        reason_code="evidence_supported",
        retrieval_attempt=0,
        clarification_attempt=0,
    )


class ExecutionStateTests(unittest.TestCase):
    def test_tracks_query_plan_and_active_step(self):
        plan = make_plan()
        state = ExecutionState.start(plan)
        self.assertEqual(state.query, plan.query)
        self.assertEqual(state.current_step_id, plan.steps[0].step_id)

    def test_attempt_counts_and_pending_facts_feed_phase12_state(self):
        plan = make_plan()
        state = ExecutionState.start(plan)
        first_assessment = supported_assessment(state.current_step_id, SufficiencyStatus.UNSUPPORTED)
        state = state.record_assessment(first_assessment)
        state = state.record_action(decide_next_action(first_assessment, state.next_action_state()))
        state = state.record_retrieval_attempt()
        state = state.with_pending_user_facts(("whether the act was intentional",))
        second_assessment = supported_assessment(state.current_step_id, SufficiencyStatus.PARTIALLY_SUPPORTED)
        state = state.record_assessment(second_assessment)
        state = state.record_action(decide_next_action(second_assessment, state.next_action_state()))
        request = build_clarification_requests(state.selected_actions[-1])[0]
        state = state.record_clarification_request(request)
        next_state = state.next_action_state()
        self.assertEqual(next_state.retrieval_attempts, 1)
        self.assertEqual(next_state.clarification_attempts, 1)
        self.assertEqual(next_state.missing_user_facts, ("whether the act was intentional",))

    def test_user_fact_resolves_matching_pending_target(self):
        plan = make_plan(1)
        state = ExecutionState.start(plan).with_pending_user_facts(("intent",))
        state = state.record_user_fact("intent", "The act was intentional")
        self.assertEqual(state.user_facts[0].value, "The act was intentional")
        self.assertEqual(state.next_action_state().missing_user_facts, ())

    def test_multiple_clarification_questions_count_as_one_attempt(self):
        plan = make_plan(1)
        state = ExecutionState.start(plan).with_pending_user_facts(("intent", "circumstances"))
        missing = supported_assessment(state.current_step_id, SufficiencyStatus.PARTIALLY_SUPPORTED)
        state = state.record_assessment(missing)
        decision = decide_next_action(missing, state.next_action_state())
        state = state.record_action(decision)
        requests = build_clarification_requests(decision)
        state = state.record_clarification_requests(requests)
        self.assertEqual(state.next_action_state().clarification_attempts, 1)

    def test_assessment_action_and_advance_are_recorded(self):
        plan = make_plan()
        state = ExecutionState.start(plan)
        first = state.current_step_id
        state = state.record_assessment(supported_assessment(first))
        state = state.record_action(continue_decision(first))
        state = state.advance_after_continue()
        self.assertEqual(state.current_step_id, plan.steps[1].step_id)
        self.assertEqual(len(state.assessments), 1)
        self.assertEqual(len(state.selected_actions), 1)

    def test_attempt_limit_is_enforced(self):
        plan = make_plan(1)
        state = ExecutionState.start(plan, max_retrieval_attempts=1)
        missing = supported_assessment(state.current_step_id, SufficiencyStatus.UNSUPPORTED)
        state = state.record_assessment(missing)
        state = state.record_action(decide_next_action(missing, state.next_action_state()))
        state = state.record_retrieval_attempt()
        with self.assertRaises(ValueError):
            state.record_retrieval_attempt()

    def test_state_is_json_serializable_and_versioned(self):
        plan = make_plan(1)
        state = ExecutionState.start(plan)
        encoded = state.to_dict()
        self.assertEqual(encoded["schema_version"], "1.0")
        self.assertEqual(json.loads(json.dumps(encoded))["query"], plan.query)


if __name__ == "__main__":
    unittest.main()
