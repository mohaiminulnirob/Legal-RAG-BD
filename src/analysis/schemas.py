"""Validated, serializable structures for Phase 9 query analysis.

Analysis records describe what the query says and signals that may guide later
stages. They do not determine legal outcomes or add facts to the user's account.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class QueryKind(StrEnum):
    GENERAL_QUESTION = "general_question"
    FACT_PATTERN = "fact_pattern"
    MULTI_ISSUE = "multi_issue"
    SECTION_LOOKUP = "section_lookup"
    PROCEDURAL_QUESTION = "procedural_question"
    UNKNOWN = "unknown"


class Complexity(StrEnum):
    SIMPLE = "simple"
    COMPOUND = "compound"
    COMPLEX = "complex"


@dataclass(frozen=True)
class TextSpan:
    """A verbatim substring from the original user query."""

    text: str
    start: int
    end: int

    def validate_against(self, query: str) -> None:
        if self.start < 0 or self.end <= self.start or self.end > len(query):
            raise ValueError("Text span has invalid offsets.")
        if query[self.start:self.end] != self.text:
            raise ValueError("Text span must exactly match its source query.")


@dataclass(frozen=True)
class QueryAnalysis:
    """Analysis output with explicit provenance and deterministic signals."""

    query: str
    normalized_query: str
    kind: QueryKind
    complexity: Complexity
    signals: tuple[str, ...] = ()
    issue_spans: tuple[TextSpan, ...] = ()
    fact_spans: tuple[TextSpan, ...] = ()
    llm_assisted: bool = False
    warnings: tuple[str, ...] = ()
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("The query must not be empty.")
        for span in (*self.issue_spans, *self.fact_spans):
            span.validate_against(self.query)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly output (enum values, lists, and nested spans)."""
        result = asdict(self)
        result["kind"] = self.kind.value
        result["complexity"] = self.complexity.value
        result["signals"] = list(self.signals)
        result["issue_spans"] = [asdict(span) for span in self.issue_spans]
        result["fact_spans"] = [asdict(span) for span in self.fact_spans]
        result["warnings"] = list(self.warnings)
        return result
