"""Inspectable text/metadata rules for matching evidence to step requirements."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.reasoning.schemas import IssueType, ReasoningStep


@dataclass(frozen=True)
class RequirementMatch:
    requirement: str
    matched_evidence_ids: tuple[str, ...] = ()
    ambiguous_evidence_ids: tuple[str, ...] = ()


_OFFENCE_DEFINITION = re.compile(
    r"\b(?:is an? offence|constitutes an? offence|commits? the offence|"
    r"culpable homicide is murder|offence of [a-z ]{2,40})\b",
    re.I,
)
_OFFENCE_ELEMENTS = re.compile(
    r"\b(?:intention|intent|knowledge|causes? death|causes? bodily injury|"
    r"whoever|does any act|voluntarily)\b",
    re.I,
)
_PUNISHMENT_CUE = re.compile(
    r"\b(?:shall be punished|is punishable|punishment|penalty|liable to (?:a )?fine|"
    r"liable to (?:imprisonment|death))\b",
    re.I,
)
_NAMED_OFFENCE = re.compile(r"\b(murder|theft|robbery|culpable homicide|rape|trespass)\b", re.I)
_EXCEPTION_CUE = re.compile(
    r"\b(?:exception|defen[cs]e|justification|not an offence if|not murder if|"
    r"private defence|private defense|provocation)\b",
    re.I,
)
_PROCEDURE_CUE = re.compile(
    r"\b(?:admissible|admitted in evidence|admit(?:ted|s)? in evidence|"
    r"procedure|appeal|bail|trial|filing|jurisdiction|limitation)\b",
    re.I,
)
_LEGAL_RULE_CUE = re.compile(r"\b(?:means|shall|must|may be|is defined|constitutes)\b", re.I)

_ISSUE_REQUIREMENTS: dict[IssueType, tuple[str, ...]] = {
    IssueType.OFFENCE: ("offence_definition", "offence_elements"),
    IssueType.PUNISHMENT: ("punishment_provision",),
    IssueType.EXCEPTION: ("exception_or_defence_provision",),
    IssueType.PROCEDURAL: ("procedural_rule",),
    IssueType.PROVISION: ("statutory_text",),
    IssueType.LEGAL_RULE: ("relevant_legal_rule",),
}


def requirements_for(step: ReasoningStep) -> tuple[str, ...]:
    """Return the explicit requirements for this step, rejecting unknown types."""
    requirements = tuple(step.required_evidence_types)
    if not requirements:
        raise ValueError(f"Step {step.step_id} has no evidence requirements.")
    supported = set(_ISSUE_REQUIREMENTS.get(step.issue_type, ()))
    if not set(requirements).issubset(supported):
        raise ValueError(f"Unsupported evidence requirement for {step.issue_type.value}: {requirements!r}")
    return requirements


def _record_text(record: dict[str, Any]) -> str:
    fields = ("section_content", "section_title", "act_title")
    return " ".join(str(record.get(field) or "") for field in fields)


def _has_named_offence_relation(text: str, offence: str) -> bool:
    # Require a direct “commits <offence> … shall be punished” structure.
    # Mere co-occurrence of a crime name and a punishment phrase is ambiguous.
    escaped = re.escape(offence)
    pattern = re.compile(
        rf"\bcommits\s+(?:the\s+)?{escaped}\b.{{0,100}}\bshall be punished\b",
        re.I | re.S,
    )
    return bool(pattern.search(text))


def match_requirement(
    requirement: str,
    step: ReasoningStep,
    records: tuple[dict[str, Any], ...],
) -> RequirementMatch:
    """Find exact-category matches; ambiguous text cues are reported separately."""
    matched: list[str] = []
    ambiguous: list[str] = []
    for record in records:
        chunk_id = str(record["chunk_id"])
        text = _record_text(record)
        content = str(record.get("section_content") or "")
        if requirement == "offence_definition":
            if _OFFENCE_DEFINITION.search(content):
                matched.append(chunk_id)
        elif requirement == "offence_elements":
            if _OFFENCE_ELEMENTS.search(content):
                matched.append(chunk_id)
        elif requirement == "punishment_provision":
            cue = _PUNISHMENT_CUE.search(content)
            if not cue:
                continue
            offence_names = {item.casefold() for item in _NAMED_OFFENCE.findall(step.retrieval_query)}
            if offence_names and any(_has_named_offence_relation(content, offence) for offence in offence_names):
                matched.append(chunk_id)
            else:
                # A punishment phrase alone may describe attempt, abetment, or another offence.
                ambiguous.append(chunk_id)
        elif requirement == "exception_or_defence_provision":
            if _EXCEPTION_CUE.search(content):
                matched.append(chunk_id)
        elif requirement == "procedural_rule":
            if _PROCEDURE_CUE.search(content):
                matched.append(chunk_id)
        elif requirement == "statutory_text":
            if content.strip() and record.get("section_id") not in (None, ""):
                matched.append(chunk_id)
        elif requirement == "relevant_legal_rule":
            if content.strip() and _LEGAL_RULE_CUE.search(content):
                ambiguous.append(chunk_id)
        else:
            raise ValueError(f"No evidence rule is registered for requirement: {requirement}")
    return RequirementMatch(requirement, tuple(dict.fromkeys(matched)), tuple(dict.fromkeys(ambiguous)))
