"""Bounded, evidence-traceable records for uncertainty acknowledgement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.reasoning.next_action import NextAction, NextActionDecision
from src.reasoning.schemas import ReasoningStep
from src.reasoning.sufficiency import EvidenceAssessment


def _bounded_text(value: Any, maximum: int) -> str:
    compact = " ".join(str(value or "").split())
    if len(compact) <= maximum:
        return compact
    return compact[: maximum - 1].rstrip() + "…"


@dataclass(frozen=True)
class SupportedEvidenceExcerpt:
    chunk_id: str
    act_title: str
    section_id: str
    excerpt: str
    source_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "act_title": self.act_title,
            "section_id": self.section_id,
            "excerpt": self.excerpt,
            "source_url": self.source_url,
        }


@dataclass(frozen=True)
class UncertaintyAcknowledgement:
    step_id: str
    supported_evidence: tuple[SupportedEvidenceExcerpt, ...]
    missing_requirements: tuple[str, ...]
    unresolved_user_facts: tuple[str, ...]
    cannot_conclude: str
    reason_code: str
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "supported_evidence": [item.to_dict() for item in self.supported_evidence],
            "missing_requirements": list(self.missing_requirements),
            "unresolved_user_facts": list(self.unresolved_user_facts),
            "cannot_conclude": self.cannot_conclude,
            "reason_code": self.reason_code,
            "schema_version": self.schema_version,
        }

    def render(self) -> str:
        """Render a bounded factual summary without drawing a legal conclusion."""
        parts: list[str] = []
        if self.supported_evidence:
            entries = []
            for item in self.supported_evidence:
                cite = f"{item.act_title}, section {item.section_id}"
                entries.append(f"{cite}: {item.excerpt}")
            parts.append("Available evidence: " + " ".join(entries))
        else:
            parts.append("Available evidence: no verified supporting passage is attached to this step.")

        missing = [*self.missing_requirements, *self.unresolved_user_facts]
        if missing:
            parts.append("Missing or unresolved: " + "; ".join(dict.fromkeys(missing)) + ".")
        parts.append(self.cannot_conclude)
        return "\n\n".join(parts)


def build_uncertainty_acknowledgement(
    decision: NextActionDecision,
    assessment: EvidenceAssessment,
    step: ReasoningStep,
    *,
    unresolved_user_facts: tuple[str, ...] = (),
    max_evidence_records: int = 3,
    max_excerpt_chars: int = 360,
) -> UncertaintyAcknowledgement:
    """Build an acknowledgement only when Phase 12 selected that action."""
    if decision.action != NextAction.ACKNOWLEDGE_UNCERTAINTY:
        raise ValueError("Uncertainty acknowledgement requires an ACKNOWLEDGE_UNCERTAINTY decision.")
    if (
        decision.step_id != assessment.step_id
        or step.step_id != assessment.step_id
        or decision.sufficiency != assessment.status
    ):
        raise ValueError("Decision and assessment must match the reasoning step ID and sufficiency status.")
    if max_evidence_records < 0 or max_excerpt_chars < 1:
        raise ValueError("Evidence and excerpt bounds are invalid.")
    if any(not isinstance(fact, str) or not fact.strip() for fact in unresolved_user_facts):
        raise ValueError("Unresolved user facts must be non-empty strings.")

    relevant_ids = set(assessment.relevant_evidence_ids)
    records: list[SupportedEvidenceExcerpt] = []
    seen: set[str] = set()
    for raw in step.evidence if max_evidence_records else ():
        if not isinstance(raw, dict):
            continue
        chunk_id = raw.get("chunk_id")
        content = raw.get("section_content")
        if not isinstance(chunk_id, str) or chunk_id not in relevant_ids or chunk_id in seen or not isinstance(content, str):
            continue
        act_title = raw.get("act_title")
        section_id = raw.get("section_id")
        if not act_title or section_id in (None, ""):
            continue
        excerpt = " ".join(content.split())
        if len(excerpt) > max_excerpt_chars:
            excerpt = excerpt[: max_excerpt_chars - 1].rstrip() + "…"
        records.append(
            SupportedEvidenceExcerpt(
                chunk_id=chunk_id,
                act_title=_bounded_text(act_title, 160),
                section_id=_bounded_text(section_id, 80),
                excerpt=excerpt,
                source_url=_bounded_text(raw["source_url"], 500) if raw.get("source_url") else None,
            )
        )
        seen.add(chunk_id)
        if len(records) >= max_evidence_records:
            break

    missing_requirements = tuple(_bounded_text(item, 160) for item in assessment.missing_requirements[:10])
    unresolved = tuple(
        dict.fromkeys(_bounded_text(fact.strip(), 200) for fact in unresolved_user_facts[:10])
    )
    cannot_conclude = (
        f"A definitive conclusion about '{_bounded_text(step.objective, 300)}' cannot be established "
        "from the available evidence and information."
    )
    return UncertaintyAcknowledgement(
        step_id=step.step_id,
        supported_evidence=tuple(records),
        missing_requirements=missing_requirements,
        unresolved_user_facts=unresolved,
        cannot_conclude=cannot_conclude,
        reason_code=decision.reason_code.value,
    )
