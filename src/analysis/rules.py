"""Small, auditable deterministic query signals (not legal conclusions)."""

from __future__ import annotations

import re

from src.analysis.schemas import Complexity, QueryKind


_SECTION_TERMS = re.compile(r"\b(section|article|act\s+no\.?|under\s+which\s+section)\b", re.I)
_FACT_TERMS = re.compile(
    r"\b(is accused|allegedly|after|before|during|when|while|"
    r"because|argued|killed|stole|stolen|stealing|took|attacked|caused|"
    r"struck|signed|paid|entered|arrested|charged|injured|refused|"
    r"failed to)\b",
    re.I,
)
_LEGAL_ISSUES = re.compile(
    r"\b(offen[cs]e|punishment|penalty|liable|liability|whether|"
    r"exception|defence|defense|bail|limitation|jurisdiction|procedure|"
    r"theft|steal|murder|crime|court|law|legal|contract|evidence|"
    r"rights|tort|property)\b",
    re.I,
)
_PROCEDURAL_TERMS = re.compile(
    r"\b(procedure|procedural|admitted|admissible|file|filing|appeal|"
    r"bail|trial|court|evidence|limitation|jurisdiction)\b",
    re.I,
)
_ISSUE_GROUPS = (
    re.compile(r"\b(offen[cs]e|theft|steal|murder|crime|tort)\b", re.I),
    re.compile(r"\b(punishment|penalty|sentence|fine)\b", re.I),
    re.compile(r"\b(exception|defen[cs]e|defence|defense|justification)\b", re.I),
    re.compile(r"\b(procedure|procedural|bail|trial|appeal|limitation|jurisdiction)\b", re.I),
    re.compile(r"\b(contract|liable|liability|rights?)\b", re.I),
)
_CLAUSE_SPLIT = re.compile(r"[;?]+|\b(?:and|also|additionally)\b", re.I)


def analyze_signals(query: str) -> tuple[QueryKind, Complexity, tuple[str, ...]]:
    """Classify query shape from explicit surface features only."""
    if not query.strip():
        raise ValueError("The query must not be empty.")

    signals: list[str] = []
    question_count = query.count("?")
    clause_count = sum(bool(part.strip()) for part in _CLAUSE_SPLIT.split(query))
    if question_count > 1 or clause_count >= 3:
        signals.append("multiple_question_or_clause_markers")
    if _SECTION_TERMS.search(query):
        signals.append("statute_or_section_reference")
    if _FACT_TERMS.search(query):
        signals.append("fact_pattern_language")
    if _LEGAL_ISSUES.search(query):
        signals.append("legal_issue_language")
    if _PROCEDURAL_TERMS.search(query):
        signals.append("procedural_language")
    if re.search(r"\b(if|unless|except|provided that|subject to)\b", query, re.I):
        signals.append("conditional_or_exception_language")

    issue_group_count = sum(bool(pattern.search(query)) for pattern in _ISSUE_GROUPS)
    has_issue_connector = bool(re.search(r"\b(and|also|additionally|as well as)\b", query, re.I))
    multi_issue = question_count > 1 or (issue_group_count >= 2 and has_issue_connector)
    if multi_issue:
        kind = QueryKind.MULTI_ISSUE
    elif _FACT_TERMS.search(query):
        kind = QueryKind.FACT_PATTERN
    elif _SECTION_TERMS.search(query):
        kind = QueryKind.SECTION_LOOKUP
    elif _PROCEDURAL_TERMS.search(query):
        kind = QueryKind.PROCEDURAL_QUESTION
    elif _LEGAL_ISSUES.search(query):
        kind = QueryKind.GENERAL_QUESTION
    else:
        kind = QueryKind.UNKNOWN

    if multi_issue or "conditional_or_exception_language" in signals:
        complexity = Complexity.COMPLEX
    elif "fact_pattern_language" in signals or clause_count > 1:
        complexity = Complexity.COMPOUND
    else:
        complexity = Complexity.SIMPLE
    return kind, complexity, tuple(signals)
