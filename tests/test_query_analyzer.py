"""Tests for Phase 9 deterministic query analysis and optional LLM fallback."""

from __future__ import annotations

import unittest

from src.analysis.query_analyzer import analyze_query
from src.analysis.schemas import Complexity, QueryKind


class FakeLLM:
    def __init__(self, output: str) -> None:
        self.output = output

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self.output


class QueryAnalyzerTests(unittest.TestCase):
    def test_representative_query_categories(self) -> None:
        examples = (
            ("What is the punishment for murder?", QueryKind.GENERAL_QUESTION, Complexity.SIMPLE),
            (
                "A person intentionally killed another person. What offence may apply?",
                QueryKind.FACT_PATTERN,
                Complexity.COMPOUND,
            ),
            (
                "A person attacked another during an argument and caused serious injury. What offence and punishment may apply?",
                QueryKind.MULTI_ISSUE,
                Complexity.COMPLEX,
            ),
            (
                "A person took another person's property. Is this theft?",
                QueryKind.FACT_PATTERN,
                Complexity.COMPOUND,
            ),
            (
                "A person caused another person's death during a fight.",
                QueryKind.FACT_PATTERN,
                Complexity.COMPOUND,
            ),
            (
                "Can this evidence be admitted in court?",
                QueryKind.PROCEDURAL_QUESTION,
                Complexity.SIMPLE,
            ),
            ("How can I learn Python?", QueryKind.UNKNOWN, Complexity.SIMPLE),
        )
        for query, expected_kind, expected_complexity in examples:
            with self.subTest(query=query):
                analysis = analyze_query(query)
                self.assertEqual(analysis.kind, expected_kind)
                self.assertEqual(analysis.complexity, expected_complexity)
                self.assertFalse(analysis.llm_assisted)
                self.assertEqual(analysis.schema_version, "1.0")
                self.assertIsInstance(analysis.to_dict()["signals"], list)

    def test_llm_spans_must_be_exact_source_substrings(self) -> None:
        query = "A person took property. Is this theft?"
        llm = FakeLLM(
            '{"kind":"fact_pattern","issue_spans":["this theft"],'
            '"fact_spans":["took property", "invented fact"]}'
        )
        analysis = analyze_query(query, llm)
        self.assertTrue(analysis.llm_assisted)
        self.assertEqual([span.text for span in analysis.issue_spans], ["this theft"])
        self.assertEqual([span.text for span in analysis.fact_spans], ["took property"])
        self.assertEqual(query[analysis.fact_spans[0].start:analysis.fact_spans[0].end], "took property")

    def test_malformed_llm_output_falls_back_to_deterministic_result(self) -> None:
        query = "What is the punishment for murder?"
        deterministic = analyze_query(query)
        assisted = analyze_query(query, FakeLLM("not valid JSON"))
        self.assertEqual(assisted.kind, deterministic.kind)
        self.assertEqual(assisted.complexity, deterministic.complexity)
        self.assertEqual(assisted.signals, deterministic.signals)
        self.assertFalse(assisted.llm_assisted)
        self.assertTrue(assisted.warnings)

    def test_empty_query_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analyze_query("  ")


if __name__ == "__main__":
    unittest.main()
