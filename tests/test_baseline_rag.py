"""Tests for evidence construction and conventional baseline RAG orchestration."""

import unittest
from unittest.mock import patch

from src.rag.baseline import answer_query
from src.rag.context import build_context
from src.rag.llm import OpenAIResponsesLLM


class FakeLLM:
    model = "test-model"

    def __init__(self) -> None:
        self.system_prompt = ""
        self.user_prompt = ""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return "The supplied provision addresses murder. [Penal Code, section 302]"


EVIDENCE = [{
    "chunk_id": "act_penal_section_0302",
    "act_title": "The Penal Code, 1860",
    "act_no": "XLV",
    "act_year": "1860",
    "section_id": "302",
    "section_content": "Whoever commits murder shall be punished with death or imprisonment for life.",
    "source_url": "https://example.test/penal/302",
    "rrf_score": 0.03,
}]


class BaselineRAGTests(unittest.TestCase):
    def test_context_contains_traceable_evidence_fields(self) -> None:
        context = build_context(EVIDENCE)
        self.assertIn("Citation: The Penal Code, 1860, section 302", context)
        self.assertIn("Chunk ID: act_penal_section_0302", context)
        self.assertIn("Source: https://example.test/penal/302", context)

    @patch("src.rag.baseline.hybrid.search", return_value=EVIDENCE)
    def test_baseline_uses_hybrid_evidence_and_returns_citations(self, retrieve) -> None:
        llm = FakeLLM()
        response = answer_query("What punishment applies for murder?", llm)
        self.assertIn("murder", response.answer)
        self.assertEqual(response.citations[0]["chunk_id"], "act_penal_section_0302")
        self.assertIn("Answer only from the supplied evidence", llm.system_prompt)
        self.assertIn("[Evidence 1]", llm.user_prompt)
        retrieve.assert_called_once()

    @patch("src.rag.baseline.hybrid.search", return_value=[])
    def test_baseline_abstains_when_no_evidence_is_retrieved(self, _retrieve) -> None:
        response = answer_query("Unknown question", FakeLLM())
        self.assertTrue(response.insufficient_evidence)
        self.assertEqual(response.citations, [])

    def test_invalid_query_or_retrieval_sizes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            answer_query("", FakeLLM())
        with self.assertRaises(ValueError):
            answer_query("question", FakeLLM(), top_k=5, candidate_k=4)

    @patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False)
    def test_openai_client_requires_an_api_key(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
            OpenAIResponsesLLM()
