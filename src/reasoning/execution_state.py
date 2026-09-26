"""Immutable bounded execution state for adaptive reasoning orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from src.reasoning.clarification import ClarificationRequest
from src.reasoning.next_action import NextAction, NextActionDecision, NextActionState
from src.reasoning.schemas import ReasoningPlan, ReasoningStep
from src.reasoning.sufficiency import EvidenceAssessment

MAX_RETRIEVAL_ATTEMPTS = 10
MAX_CLARIFICATION_ATTEMPTS = 5
MAX_PENDING_FACTS = 10


@dataclass(frozen=True)
class UserFact:
    step_id: str
    fact: str
    value: str
    source: str = "user"

    def to_dict(self) -> dict[str, str]:
        return {"step_id": self.step_id, "fact": self.fact, "value": self.value, "source": self.source}


@dataclass(frozen=True)
class ExecutionState:
    query: str
    plan: ReasoningPlan
    current_step_id: str | None
    max_retrieval_attempts: int = 2
    max_clarification_attempts: int = 1
    retrieval_attempts: tuple[tuple[str, int], ...] = ()
    clarification_attempts: tuple[tuple[str, int], ...] = ()
    pending_user_facts: tuple[tuple[str, tuple[str, ...]], ...] = ()
    user_facts: tuple[UserFact, ...] = ()
    assessments: tuple[EvidenceAssessment, ...] = ()
    selected_actions: tuple[NextActionDecision, ...] = ()
    stopped: bool = False
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("Execution state must retain the non-empty original query.")
        step_ids = {step.step_id for step in self.plan.steps}
        if self.current_step_id is not None and self.current_step_id not in step_ids:
            raise ValueError("current_step_id must identify a step in the plan or be None.")
        if self.current_step_id is None and self.plan.steps and not self.stopped:
            raise ValueError("An active plan with steps must have a current step.")
        if not 0 <= self.max_retrieval_attempts <= MAX_RETRIEVAL_ATTEMPTS:
            raise ValueError("Retrieval attempt limit must be between 0 and 10.")
        if not 0 <= self.max_clarification_attempts <= MAX_CLARIFICATION_ATTEMPTS:
            raise ValueError("Clarification attempt limit must be between 0 and 5.")
        for counter_name, counter_values in (
            ("retrieval", self.retrieval_attempts),
            ("clarification", self.clarification_attempts),
        ):
            seen: set[str] = set()
            for step_id, count in counter_values:
                if step_id not in step_ids or step_id in seen or count < 0:
                    raise ValueError(f"Invalid {counter_name} attempt counter.")
                if count > (self.max_retrieval_attempts if counter_name == "retrieval" else self.max_clarification_attempts):
                    raise ValueError(f"{counter_name} attempts exceed the configured limit.")
                seen.add(step_id)
        if any(step_id not in step_ids for step_id, _ in self.pending_user_facts):
            raise ValueError("Pending user facts must belong to a plan step.")
        if len({step_id for step_id, _ in self.pending_user_facts}) != len(self.pending_user_facts):
            raise ValueError("Pending user facts may appear only once per step.")
        if any(any(not isinstance(fact, str) or not fact.strip() for fact in facts) for _, facts in self.pending_user_facts):
            raise ValueError("Pending user facts must be non-empty strings.")
        if any(len(facts) > MAX_PENDING_FACTS or any(len(fact) > 240 for fact in facts) for _, facts in self.pending_user_facts):
            raise ValueError("Pending user facts are limited to 10 targets of at most 240 characters each.")
        if any(fact.step_id not in step_ids for fact in self.user_facts):
            raise ValueError("Recorded user facts must belong to a plan step.")
        if any(not fact.fact.strip() or not fact.value.strip() or fact.source != "user" for fact in self.user_facts):
            raise ValueError("Recorded facts must contain a value and identify the user as their source.")
        if any(item.step_id not in step_ids for item in self.assessments + self.selected_actions):
            raise ValueError("Assessment and action history must belong to plan steps.")

    @classmethod
    def start(
        cls,
        plan: ReasoningPlan,
        *,
        query: str | None = None,
        max_retrieval_attempts: int = 2,
        max_clarification_attempts: int = 1,
    ) -> "ExecutionState":
        current = plan.steps[0].step_id if plan.steps else None
        return cls(
            query=query if query is not None else plan.query,
            plan=plan,
            current_step_id=current,
            max_retrieval_attempts=max_retrieval_attempts,
            max_clarification_attempts=max_clarification_attempts,
        )

    def _require_current_step(self, step_id: str) -> None:
        if self.stopped or self.current_step_id != step_id:
            raise ValueError("Operation must target the active, non-stopped reasoning step.")

    def next_action_state(self) -> NextActionState:
        if self.current_step_id is None:
            raise ValueError("No current step is available for a next-action decision.")
        pending = dict(self.pending_user_facts).get(self.current_step_id, ())
        return NextActionState(
            retrieval_attempts=dict(self.retrieval_attempts).get(self.current_step_id, 0),
            max_retrieval_attempts=self.max_retrieval_attempts,
            clarification_attempts=dict(self.clarification_attempts).get(self.current_step_id, 0),
            max_clarification_attempts=self.max_clarification_attempts,
            missing_user_facts=pending,
            stop_requested=self.stopped,
        )

    def retrieval_attempts_for(self, step_id: str) -> int:
        return dict(self.retrieval_attempts).get(step_id, 0)

    def clarification_attempts_for(self, step_id: str) -> int:
        return dict(self.clarification_attempts).get(step_id, 0)

    def with_pending_user_facts(
        self,
        facts: tuple[str, ...],
        *,
        step_id: str | None = None,
    ) -> "ExecutionState":
        target_step_id = step_id or self.current_step_id
        if target_step_id is None or target_step_id not in {step.step_id for step in self.plan.steps}:
            raise ValueError("Pending facts must target a step in the plan.")
        if any(not isinstance(fact, str) or not fact.strip() for fact in facts):
            raise ValueError("Pending facts must be non-empty strings.")
        if len(facts) > MAX_PENDING_FACTS or any(len(fact) > 240 for fact in facts):
            raise ValueError("Pending facts are limited to 10 targets of at most 240 characters each.")
        pending = dict(self.pending_user_facts)
        pending[target_step_id] = tuple(dict.fromkeys(fact.strip() for fact in facts))
        return replace(self, pending_user_facts=tuple((key, value) for key, value in pending.items()))

    def update_current_step(self, updated_step: ReasoningStep) -> "ExecutionState":
        self._require_current_step(updated_step.step_id)
        updated_plan = replace(
            self.plan,
            steps=tuple(updated_step if step.step_id == updated_step.step_id else step for step in self.plan.steps),
        )
        return replace(self, plan=updated_plan)

    def record_assessment(self, assessment: EvidenceAssessment) -> "ExecutionState":
        self._require_current_step(assessment.step_id)
        return replace(self, assessments=(*self.assessments, assessment))

    def record_action(self, decision: NextActionDecision) -> "ExecutionState":
        self._require_current_step(decision.step_id)
        if not self.assessments or self.assessments[-1].step_id != decision.step_id:
            raise ValueError("An action requires a recorded assessment for the current step.")
        if self.assessments[-1].status != decision.sufficiency:
            raise ValueError("Action sufficiency must match the current assessment.")
        updated = replace(self, selected_actions=(*self.selected_actions, decision))
        if decision.action == NextAction.STOP:
            updated = replace(updated, stopped=True)
        return updated

    def record_retrieval_attempt(self) -> "ExecutionState":
        if self.current_step_id is None or self.stopped:
            raise ValueError("No active step is available for retrieval.")
        counts = dict(self.retrieval_attempts)
        used = counts.get(self.current_step_id, 0)
        if used >= self.max_retrieval_attempts:
            raise ValueError("Retrieval attempt limit reached.")
        if (
            not self.selected_actions
            or self.selected_actions[-1].step_id != self.current_step_id
            or self.selected_actions[-1].action != NextAction.RETRIEVE_MORE
            or self.selected_actions[-1].retrieval_attempt != used + 1
        ):
            raise ValueError("A retrieval attempt requires an unconsumed RETRIEVE_MORE action.")
        counts[self.current_step_id] = used + 1
        return replace(self, retrieval_attempts=tuple(counts.items()))

    def record_clarification_request(self, request: ClarificationRequest) -> "ExecutionState":
        return self.record_clarification_requests((request,))

    def record_clarification_requests(
        self,
        requests: tuple[ClarificationRequest, ...],
    ) -> "ExecutionState":
        if not requests:
            raise ValueError("At least one clarification request is required.")
        step_id = requests[0].step_id
        self._require_current_step(step_id)
        counts = dict(self.clarification_attempts)
        used = counts.get(step_id, 0)
        if used >= self.max_clarification_attempts:
            raise ValueError("Clarification attempt limit reached.")
        if (
            not self.selected_actions
            or self.selected_actions[-1].step_id != step_id
            or self.selected_actions[-1].action != NextAction.CLARIFY
            or self.selected_actions[-1].clarification_attempt != used + 1
            or any(request.step_id != step_id or request.attempt != used + 1 for request in requests)
            or not set(request.target for request in requests).issubset(
                set(self.selected_actions[-1].clarification_targets)
            )
        ):
            raise ValueError("A clarification request requires an unconsumed CLARIFY action.")
        counts[step_id] = used + 1
        return replace(self, clarification_attempts=tuple(counts.items()))

    def record_user_fact(self, fact: str, value: str) -> "ExecutionState":
        if self.current_step_id is None or self.stopped:
            raise ValueError("No active step is available for user facts.")
        if not fact.strip() or not value.strip():
            raise ValueError("User fact and value must be non-empty.")
        if len(fact) > 240 or len(value) > 2_000:
            raise ValueError("User facts are limited to 240 characters and values to 2,000 characters.")
        if fact.strip().casefold() not in {
            pending.casefold() for pending in dict(self.pending_user_facts).get(self.current_step_id, ())
        }:
            raise ValueError("The user fact must match an explicitly pending clarification target.")
        recorded = UserFact(self.current_step_id, fact.strip(), value.strip())
        pending = dict(self.pending_user_facts)
        remaining = tuple(item for item in pending.get(self.current_step_id, ()) if item.casefold() != fact.strip().casefold())
        pending[self.current_step_id] = remaining
        return replace(
            self,
            user_facts=(*self.user_facts, recorded),
            pending_user_facts=tuple(pending.items()),
        )

    def advance_after_continue(self) -> "ExecutionState":
        if self.current_step_id is None or self.stopped:
            raise ValueError("No active step can be advanced.")
        if not self.selected_actions or self.selected_actions[-1].step_id != self.current_step_id or self.selected_actions[-1].action != NextAction.CONTINUE:
            raise ValueError("The current step can advance only after a CONTINUE action.")
        ids = [step.step_id for step in self.plan.steps]
        next_index = ids.index(self.current_step_id) + 1
        if next_index == len(ids):
            return replace(self, current_step_id=None, stopped=True)
        return replace(self, current_step_id=ids[next_index])

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "plan": self.plan.to_dict(),
            "current_step_id": self.current_step_id,
            "max_retrieval_attempts": self.max_retrieval_attempts,
            "max_clarification_attempts": self.max_clarification_attempts,
            "retrieval_attempts": dict(self.retrieval_attempts),
            "clarification_attempts": dict(self.clarification_attempts),
            "pending_user_facts": {step_id: list(facts) for step_id, facts in self.pending_user_facts},
            "user_facts": [item.to_dict() for item in self.user_facts],
            "assessments": [item.to_dict() for item in self.assessments],
            "selected_actions": [item.to_dict() for item in self.selected_actions],
            "stopped": self.stopped,
            "schema_version": self.schema_version,
        }
