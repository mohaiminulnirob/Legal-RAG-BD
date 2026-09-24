"""Rule-based evidence sufficiency assessment for retrieved reasoning steps."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.reasoning.evidence_rules import match_requirement, requirements_for
from src.reasoning.schemas import ReasoningPlan, ReasoningStep


class SufficiencyStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    UNCERTAIN = "uncertain"


class ReasonCode(StrEnum):
    HAS_EVIDENCE = "has_evidence"
    NO_EVIDENCE = "no_evidence"
    REQUIRED_TYPE_MATCH = "required_type_match"
    MISSING_REQUIRED_EVIDENCE = "missing_required_evidence"
    MULTIPLE_RETRIEVERS_AGREE = "multiple_retrievers_agree"
    REQUIRED_TERM_COVERAGE = "required_term_coverage"
    LOW_RETRIEVAL_SUPPORT = "low_retrieval_support"
    AMBIGUOUS_REQUIREMENT_MATCH = "ambiguous_requirement_match"
    CONFLICTING_EVIDENCE_RECORDS = "conflicting_evidence_records"
    MALFORMED_EVIDENCE = "malformed_evidence"
    UNKNOWN_REQUIREMENT = "unknown_requirement"


@dataclass(frozen=True)
class EvidenceAssessment:
    step_id: str
    status: SufficiencyStatus
    evidence_count: int
    relevant_evidence_ids: tuple[str, ...]
    missing_requirements: tuple[str, ...]
    support_signals: tuple[str, ...]
    reason_codes: tuple[ReasonCode, ...]
    requires_clarification: bool
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "status": self.status.value,
            "evidence_count": self.evidence_count,
            "relevant_evidence_ids": list(self.relevant_evidence_ids),
            "missing_requirements": list(self.missing_requirements),
            "support_signals": list(self.support_signals),
            "reason_codes": [code.value for code in self.reason_codes],
            "requires_clarification": self.requires_clarification,
            "schema_version": self.schema_version,
        }


_STOP_WORDS = {
    "what", "when", "where", "which", "that", "this", "with", "from", "into",
    "does", "applies", "apply", "person", "another", "under", "shall", "legal",
    "provisions", "provision", "relevant", "evidence", "elements", "offences",
}


def _validate_evidence(step: ReasoningStep) -> tuple[tuple[dict[str, Any], ...], bool, bool]:
    raw = step.evidence
    if not isinstance(raw, (tuple, list)):
        return (), True, False
    valid: list[dict[str, Any]] = []
    malformed = False
    seen: dict[str, str] = {}
    conflicting = False
    for item in raw:
        if not isinstance(item, dict):
            malformed = True
            continue
        chunk_id = item.get("chunk_id")
        content = item.get("section_content")
        has_source = bool(item.get("source_file") or item.get("source_url"))
        if (
            not isinstance(chunk_id, str)
            or not chunk_id.strip()
            or not isinstance(content, str)
            or not item.get("act_title")
            or item.get("section_id") in (None, "")
            or not has_source
        ):
            malformed = True
            continue
        prior = seen.get(chunk_id)
        if prior is not None and prior != content:
            conflicting = True
        seen[chunk_id] = content
        valid.append(item)
    return tuple(valid), malformed, conflicting


def _query_term_signals(step: ReasoningStep, records: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    terms = {
        token.casefold()
        for token in re.findall(r"[A-Za-z]{4,}", step.retrieval_query)
        if token.casefold() not in _STOP_WORDS
    }
    if not terms or not records:
        return ()
    evidence_text = " ".join(" ".join(str(value or "") for value in record.values()) for record in records).casefold()
    covered = sorted(term for term in terms if re.search(rf"\b{re.escape(term)}\b", evidence_text))
    if not covered:
        return ()
    return (f"query_terms_covered:{len(covered)}/{len(terms)}",)


def assess_step(step: ReasoningStep) -> EvidenceAssessment:
    """Assess attached evidence only; this function never invokes retrieval or an LLM."""
    records, malformed, conflicting = _validate_evidence(step)
    reasons: list[ReasonCode] = []
    signals: list[str] = []
    relevant: set[str] = set()
    ambiguous = False
    unknown_requirement = False
    try:
        requirements = requirements_for(step)
    except ValueError:
        requirements = tuple(step.required_evidence_types)
        unknown_requirement = True
        reasons.append(ReasonCode.UNKNOWN_REQUIREMENT)

    if records:
        reasons.append(ReasonCode.HAS_EVIDENCE)
    else:
        reasons.append(ReasonCode.NO_EVIDENCE)
    if malformed:
        reasons.append(ReasonCode.MALFORMED_EVIDENCE)
    if conflicting:
        reasons.append(ReasonCode.CONFLICTING_EVIDENCE_RECORDS)

    missing: list[str] = []
    for requirement in requirements:
        try:
            match = match_requirement(requirement, step, records)
        except ValueError:
            if ReasonCode.UNKNOWN_REQUIREMENT not in reasons:
                reasons.append(ReasonCode.UNKNOWN_REQUIREMENT)
            unknown_requirement = True
            missing.append(requirement)
            continue
        if match.matched_evidence_ids:
            relevant.update(match.matched_evidence_ids)
            signals.append(f"required_type_match:{requirement}")
            if ReasonCode.REQUIRED_TYPE_MATCH not in reasons:
                reasons.append(ReasonCode.REQUIRED_TYPE_MATCH)
        else:
            missing.append(requirement)
        if match.ambiguous_evidence_ids:
            ambiguous = True
            signals.append(f"ambiguous_type_cue:{requirement}")

    if missing:
        reasons.append(ReasonCode.MISSING_REQUIRED_EVIDENCE)
    if ambiguous:
        reasons.append(ReasonCode.AMBIGUOUS_REQUIREMENT_MATCH)
    for signal in _query_term_signals(step, records):
        signals.append(signal)
        reasons.append(ReasonCode.REQUIRED_TERM_COVERAGE)

    agreeing_ids = {
        record["chunk_id"]
        for record in records
        if record["chunk_id"] in relevant
        and record.get("bm25_rank") is not None
        and record.get("dense_rank") is not None
    }
    if agreeing_ids:
        reasons.append(ReasonCode.MULTIPLE_RETRIEVERS_AGREE)
        signals.extend(f"bm25_dense_agree:{chunk_id}" for chunk_id in sorted(agreeing_ids))

    if records and not relevant and not ambiguous:
        reasons.append(ReasonCode.LOW_RETRIEVAL_SUPPORT)

    if malformed or conflicting or ambiguous or unknown_requirement or not records:
        status = SufficiencyStatus.UNCERTAIN
    elif not missing:
        status = SufficiencyStatus.SUPPORTED
    elif relevant:
        status = SufficiencyStatus.PARTIALLY_SUPPORTED
    else:
        status = SufficiencyStatus.UNSUPPORTED

    # This flag records missing step requirements for a later clarification
    # controller; it does not decide which missing facts to ask the user about.
    return EvidenceAssessment(
        step_id=step.step_id,
        status=status,
        evidence_count=len(step.evidence) if isinstance(step.evidence, (tuple, list)) else 0,
        relevant_evidence_ids=tuple(sorted(relevant)),
        missing_requirements=tuple(missing),
        support_signals=tuple(dict.fromkeys(signals)),
        reason_codes=tuple(dict.fromkeys(reasons)),
        requires_clarification=bool(missing),
    )


def assess_plan(plan: ReasoningPlan) -> tuple[EvidenceAssessment, ...]:
    """Assess each step independently using evidence already attached to the plan."""
    return tuple(assess_step(step) for step in plan.steps)
