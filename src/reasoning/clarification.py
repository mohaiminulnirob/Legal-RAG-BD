"""Deterministic construction of targeted clarification requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.reasoning.next_action import NextAction, NextActionDecision


@dataclass(frozen=True)
class ClarificationRequest:
    step_id: str
    target: str
    question: str
    attempt: int
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "target": self.target,
            "question": self.question,
            "attempt": self.attempt,
            "schema_version": self.schema_version,
        }


def build_clarification_requests(
    decision: NextActionDecision,
    *,
    max_questions: int = 3,
) -> tuple[ClarificationRequest, ...]:
    """Turn Phase 12's explicit clarification targets into short questions.

    This function does not infer which facts are missing. Targets must have
    been identified and approved upstream; no LLM or legal reasoning is used.
    """
    if decision.action != NextAction.CLARIFY:
        raise ValueError("Clarification requests can only be built for a CLARIFY decision.")
    if max_questions < 1 or max_questions > 10:
        raise ValueError("max_questions must be between 1 and 10.")
    if decision.clarification_attempt < 1:
        raise ValueError("A clarification decision must identify a positive attempt number.")
    if any(not isinstance(target, str) for target in decision.clarification_targets):
        raise ValueError("Clarification targets must be strings.")
    targets = tuple(
        dict.fromkeys(" ".join(target.split()) for target in decision.clarification_targets if target.strip())
    )
    if not targets:
        raise ValueError("A CLARIFY decision must include at least one clarification target.")
    if any(len(target) > 240 for target in targets):
        raise ValueError("Clarification targets must be at most 240 characters.")

    requests = []
    for target in targets[:max_questions]:
        question = f"Could you clarify {target.rstrip(' ?.!')}?"
        requests.append(
            ClarificationRequest(
                step_id=decision.step_id,
                target=target,
                question=question,
                attempt=decision.clarification_attempt,
            )
        )
    return tuple(requests)
