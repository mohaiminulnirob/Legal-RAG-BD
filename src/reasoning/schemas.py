"""Versioned structures for reasoning plans and per-step retrieval evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class IssueType(StrEnum):
    OFFENCE = "offence"
    PUNISHMENT = "punishment"
    EXCEPTION = "exception_or_defence"
    PROCEDURAL = "procedural_rule"
    PROVISION = "statutory_provision"
    LEGAL_RULE = "legal_rule"


class StepStatus(StrEnum):
    PENDING = "pending"
    RETRIEVAL_COMPLETE = "retrieval_complete"


@dataclass(frozen=True)
class ReasoningStep:
    step_id: str
    issue_type: IssueType
    objective: str
    retrieval_query: str
    required_evidence_types: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    status: StepStatus = StepStatus.PENDING
    evidence: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.step_id.strip() or not self.objective.strip() or not self.retrieval_query.strip():
            raise ValueError("Each reasoning step needs an ID, objective, and retrieval query.")
        if not self.required_evidence_types:
            raise ValueError("Each reasoning step must name required evidence types.")
        if self.step_id in self.depends_on:
            raise ValueError("A reasoning step cannot depend on itself.")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["issue_type"] = self.issue_type.value
        result["status"] = self.status.value
        result["required_evidence_types"] = list(self.required_evidence_types)
        result["depends_on"] = list(self.depends_on)
        result["evidence"] = [dict(item) for item in self.evidence]
        return result


@dataclass(frozen=True)
class ReasoningPlan:
    query: str
    analysis_kind: str
    steps: tuple[ReasoningStep, ...]
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        ids = [step.step_id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("Reasoning step IDs must be unique.")
        seen: set[str] = set()
        for step in self.steps:
            if any(dependency not in seen for dependency in step.depends_on):
                raise ValueError("Dependencies must reference earlier steps in the plan.")
            seen.add(step.step_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "analysis_kind": self.analysis_kind,
            "steps": [step.to_dict() for step in self.steps],
            "schema_version": self.schema_version,
        }
