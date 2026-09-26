"""Deterministic next-action policy for assessed reasoning steps.

This module is intentionally a pure policy layer: it does not call retrieval,
ask the user, or invoke an LLM. Callers provide execution limits and any
user-specific facts they have already identified as missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.reasoning.sufficiency import EvidenceAssessment, ReasonCode as SufficiencyReasonCode, SufficiencyStatus


class NextAction(StrEnum):
    CONTINUE = "continue"
    RETRIEVE_MORE = "retrieve_more"
    CLARIFY = "clarify"
    ACKNOWLEDGE_UNCERTAINTY = "acknowledge_uncertainty"
    STOP = "stop"


class ActionReason(StrEnum):
    EVIDENCE_SUPPORTED = "evidence_supported"
    USER_FACT_MISSING = "user_fact_missing"
    RETRIEVAL_RETRY_AVAILABLE = "retrieval_retry_available"
    RETRIEVAL_LIMIT_REACHED = "retrieval_limit_reached"
    CLARIFICATION_LIMIT_REACHED = "clarification_limit_reached"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"
    EXECUTION_STOP_REQUESTED = "execution_stop_requested"


@dataclass(frozen=True)
class NextActionState:
    """Execution state; counters count attempts already used."""

    retrieval_attempts: int = 0
    max_retrieval_attempts: int = 1
    clarification_attempts: int = 0
    max_clarification_attempts: int = 1
    missing_user_facts: tuple[str, ...] = ()
    stop_requested: bool = False

    def __post_init__(self) -> None:
        for name, used, limit in (
            ("retrieval", self.retrieval_attempts, self.max_retrieval_attempts),
            ("clarification", self.clarification_attempts, self.max_clarification_attempts),
        ):
            if limit < 0 or used < 0 or used > limit:
                raise ValueError(f"{name} attempts must satisfy 0 <= used <= limit.")
        if any(not isinstance(fact, str) or not fact.strip() for fact in self.missing_user_facts):
            raise ValueError("Missing user facts must be non-empty strings.")


@dataclass(frozen=True)
class NextActionDecision:
    step_id: str
    sufficiency: SufficiencyStatus
    action: NextAction
    reason_code: ActionReason
    retrieval_attempt: int
    clarification_attempt: int
    clarification_targets: tuple[str, ...] = ()
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "sufficiency": self.sufficiency.value,
            "action": self.action.value,
            "reason_code": self.reason_code.value,
            "retrieval_attempt": self.retrieval_attempt,
            "clarification_attempt": self.clarification_attempt,
            "clarification_targets": list(self.clarification_targets),
            "schema_version": self.schema_version,
        }


def decide_next_action(
    assessment: EvidenceAssessment,
    state: NextActionState = NextActionState(),
) -> NextActionDecision:
    """Choose a bounded next action from one sufficiency assessment and state.

    Explicit missing user facts take precedence over retrieval because another
    statute search cannot supply facts about the user's situation. In their
    absence, insufficient but well-formed evidence may get a bounded retry.
    Ambiguous or malformed evidence is acknowledged as uncertain.
    """
    action: NextAction
    reason: ActionReason
    retrieval_attempt = state.retrieval_attempts
    clarification_attempt = state.clarification_attempts
    targets: tuple[str, ...] = ()

    if state.stop_requested:
        action, reason = NextAction.STOP, ActionReason.EXECUTION_STOP_REQUESTED
    elif assessment.status == SufficiencyStatus.SUPPORTED:
        action, reason = NextAction.CONTINUE, ActionReason.EVIDENCE_SUPPORTED
    elif assessment.status == SufficiencyStatus.UNCERTAIN:
        if state.missing_user_facts and state.clarification_attempts < state.max_clarification_attempts:
            action, reason = NextAction.CLARIFY, ActionReason.USER_FACT_MISSING
            clarification_attempt += 1
            targets = state.missing_user_facts
        elif state.missing_user_facts and state.clarification_attempts >= state.max_clarification_attempts:
            action, reason = NextAction.ACKNOWLEDGE_UNCERTAINTY, ActionReason.CLARIFICATION_LIMIT_REACHED
        elif (
            SufficiencyReasonCode.NO_EVIDENCE in assessment.reason_codes
            and state.retrieval_attempts < state.max_retrieval_attempts
        ):
            action, reason = NextAction.RETRIEVE_MORE, ActionReason.RETRIEVAL_RETRY_AVAILABLE
            retrieval_attempt += 1
        else:
            action, reason = NextAction.ACKNOWLEDGE_UNCERTAINTY, ActionReason.EVIDENCE_INSUFFICIENT
    elif state.missing_user_facts:
        if state.clarification_attempts < state.max_clarification_attempts:
            action, reason = NextAction.CLARIFY, ActionReason.USER_FACT_MISSING
            clarification_attempt += 1
            targets = state.missing_user_facts
        else:
            action, reason = NextAction.ACKNOWLEDGE_UNCERTAINTY, ActionReason.CLARIFICATION_LIMIT_REACHED
    elif state.retrieval_attempts < state.max_retrieval_attempts:
        action, reason = NextAction.RETRIEVE_MORE, ActionReason.RETRIEVAL_RETRY_AVAILABLE
        retrieval_attempt += 1
    elif state.clarification_attempts >= state.max_clarification_attempts:
        action, reason = NextAction.ACKNOWLEDGE_UNCERTAINTY, ActionReason.CLARIFICATION_LIMIT_REACHED
    else:
        action, reason = NextAction.ACKNOWLEDGE_UNCERTAINTY, ActionReason.RETRIEVAL_LIMIT_REACHED

    return NextActionDecision(
        step_id=assessment.step_id,
        sufficiency=assessment.status,
        action=action,
        reason_code=reason,
        retrieval_attempt=retrieval_attempt,
        clarification_attempt=clarification_attempt,
        clarification_targets=targets,
    )
