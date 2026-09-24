"""Deterministic reasoning-plan construction from Phase 9 query analysis."""

from __future__ import annotations

import re

from src.analysis.schemas import QueryAnalysis, QueryKind
from src.reasoning.schemas import IssueType, ReasoningPlan, ReasoningStep


_OFFENCE_CUES = re.compile(
    r"\b(offen[cs]e|crime|murder|theft|steal|killed|kill|death|attacked|"
    r"injur(?:y|ed)|caused|took|stole|struck|charged|accused)\b",
    re.I,
)
_PUNISHMENT_CUES = re.compile(r"\b(punishment|penalty|sentence|fine|imprisonment)\b", re.I)
_EXCEPTION_CUES = re.compile(
    r"\b(exception|defen[cs]e|justification|provocation|self[- ]defen[cs]e|"
    r"unless|except|provided that|subject to)\b",
    re.I,
)
_PROVISION_CUES = re.compile(r"\b(section|article|act\s+no\.?|under\s+which\s+section)\b", re.I)


def _step(
    number: int,
    issue_type: IssueType,
    objective: str,
    query_prefix: str,
    evidence_types: tuple[str, ...],
    dependencies: tuple[str, ...] = (),
    query: str = "",
) -> ReasoningStep:
    return ReasoningStep(
        step_id=f"step_{number}",
        issue_type=issue_type,
        objective=objective,
        retrieval_query=f"{query_prefix}: {query}",
        required_evidence_types=evidence_types,
        depends_on=dependencies,
    )


def build_reasoning_plan(analysis: QueryAnalysis) -> ReasoningPlan:
    """Build standard legal evidence requests without deciding legal outcomes.

    Query wording is retained in each retrieval query. The planner adds only
    generic evidence categories and never invents an offence name or section.
    """
    query = analysis.normalized_query
    if analysis.kind is QueryKind.UNKNOWN:
        return ReasoningPlan(query=query, analysis_kind=analysis.kind.value, steps=())

    offence_needed = analysis.kind in {QueryKind.FACT_PATTERN, QueryKind.MULTI_ISSUE} or bool(
        _OFFENCE_CUES.search(query)
    )
    punishment_needed = bool(_PUNISHMENT_CUES.search(query))
    exception_needed = bool(_EXCEPTION_CUES.search(query))
    procedural_needed = "procedural_language" in analysis.signals
    provision_needed = analysis.kind is QueryKind.SECTION_LOOKUP or bool(_PROVISION_CUES.search(query))

    specs: list[tuple[IssueType, str, str, tuple[str, ...], tuple[str, ...]]] = []
    if offence_needed:
        specs.append((
            IssueType.OFFENCE,
            "Identify potentially relevant offence provisions and their elements",
            "legal provisions describing offences and their elements relevant to",
            ("offence_definition", "offence_elements"),
            (),
        ))
    if punishment_needed:
        deps = ("step_1",) if offence_needed else ()
        specs.append((
            IssueType.PUNISHMENT,
            "Retrieve punishment provisions relevant to the legal issue",
            "statutory punishment provisions relevant to",
            ("punishment_provision",),
            deps,
        ))
    if exception_needed:
        deps = ("step_1",) if offence_needed else ()
        specs.append((
            IssueType.EXCEPTION,
            "Retrieve potentially relevant statutory exceptions or defences",
            "statutory exceptions or defences relevant to",
            ("exception_or_defence_provision",),
            deps,
        ))
    if procedural_needed:
        specs.append((
            IssueType.PROCEDURAL,
            "Retrieve procedural rules relevant to the user's question",
            "procedural legal rules relevant to",
            ("procedural_rule",),
            (),
        ))
    if provision_needed:
        specs.append((
            IssueType.PROVISION,
            "Retrieve the text and scope of the referenced provision",
            "text and scope of the referenced statutory provision in",
            ("statutory_text",),
            (),
        ))
    if not specs:
        specs.append((
            IssueType.LEGAL_RULE,
            "Retrieve legal rules relevant to the user's question",
            "legal rules relevant to",
            ("relevant_legal_rule",),
            (),
        ))

    steps = tuple(
        _step(index, issue_type, objective, prefix, evidence_types, dependencies, query)
        for index, (issue_type, objective, prefix, evidence_types, dependencies) in enumerate(specs, 1)
    )
    return ReasoningPlan(query=query, analysis_kind=analysis.kind.value, steps=steps)
