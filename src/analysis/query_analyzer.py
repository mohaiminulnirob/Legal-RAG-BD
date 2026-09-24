"""Phase 9 query/case analysis with deterministic defaults and optional LLM help."""

from __future__ import annotations

import argparse
import json
import re
from typing import Protocol

from src.analysis.rules import analyze_signals
from src.analysis.schemas import QueryAnalysis, QueryKind, TextSpan


class AnalysisLLM(Protocol):
    """The narrow LLM capability needed for optional interpretation assistance."""

    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


_SYSTEM_PROMPT = """Analyze the user's query as text, not as a request to answer it.
Return only one JSON object with keys: kind, issue_spans, fact_spans.
kind must be one of: general_question, fact_pattern, multi_issue,
section_lookup, unknown. issue_spans and fact_spans must be arrays of exact,
contiguous substrings copied verbatim from the query. Include only explicit
issues or facts stated by the user. Do not infer missing facts, legal rules,
offences, or outcomes. Do not create a reasoning plan or decide whether
evidence is sufficient."""


def _spans_from_strings(query: str, values: object) -> tuple[TextSpan, ...]:
    if not isinstance(values, list):
        return ()
    spans: list[TextSpan] = []
    used: set[tuple[int, int]] = set()
    for value in values[:12]:
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text or len(text) > 600:
            continue
        start = query.find(text)
        if start < 0 or (start, start + len(text)) in used:
            continue
        spans.append(TextSpan(text, start, start + len(text)))
        used.add((start, start + len(text)))
    return tuple(spans)


def analyze_query(query: str, llm: AnalysisLLM | None = None) -> QueryAnalysis:
    """Analyze query shape; optional LLM suggestions are constrained to source spans.

    The deterministic classification is always available. Malformed or invalid
    LLM output is ignored with a warning, so analysis still works offline.
    """
    if not query.strip():
        raise ValueError("The query must not be empty.")
    if len(query) > 5_000:
        raise ValueError("The query is too long (maximum 5,000 characters).")

    kind, complexity, signals = analyze_signals(query)
    issue_spans: tuple[TextSpan, ...] = ()
    fact_spans: tuple[TextSpan, ...] = ()
    warnings: list[str] = []
    assisted = False
    if llm is not None:
        try:
            raw = llm.generate(
                _SYSTEM_PROMPT,
                "Analyze this query. Copy all span values exactly from it.\n\n"
                f"<query>\n{query}\n</query>",
            )
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ValueError("Expected a JSON object.")
            try:
                suggested_kind = QueryKind(result.get("kind", "unknown"))
            except (ValueError, TypeError):
                suggested_kind = QueryKind.UNKNOWN
            # Semantic classification assists only when surface rules are inconclusive.
            if kind is QueryKind.UNKNOWN and suggested_kind is not QueryKind.UNKNOWN:
                kind = suggested_kind
            issue_spans = _spans_from_strings(query, result.get("issue_spans"))
            fact_spans = _spans_from_strings(query, result.get("fact_spans"))
            assisted = True
        except Exception as error:
            warnings.append(f"LLM assistance unavailable or invalid: {type(error).__name__}.")

    return QueryAnalysis(
        query=query,
        normalized_query=re.sub(r"\s+", " ", query).strip(),
        kind=kind,
        complexity=complexity,
        signals=signals,
        issue_spans=issue_spans,
        fact_spans=fact_spans,
        llm_assisted=assisted,
        warnings=tuple(warnings),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze legal query shape (Phase 9).")
    parser.add_argument("--query", required=True)
    parser.add_argument("--llm-assisted", action="store_true", help="Use Groq for optional span extraction and ambiguous classification.")
    parser.add_argument("--model", help="Groq model override; defaults to the project baseline model.")
    args = parser.parse_args()
    llm: AnalysisLLM | None = None
    if args.llm_assisted:
        from src.rag.llm import DEFAULT_MODEL, GroqResponsesLLM

        llm = GroqResponsesLLM(args.model or DEFAULT_MODEL)
    print(json.dumps(analyze_query(args.query, llm).to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
