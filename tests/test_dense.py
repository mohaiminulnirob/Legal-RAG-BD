"""Tests for persistent dense legal retrieval without downloading a model."""

import tempfile
import unittest
from pathlib import Path

from src.retrieval.dense import index_records, search


class KeywordEmbedder:
    """Deterministic two-dimensional test embedder."""

    def encode(self, texts, **_kwargs):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([float("murder" in lowered or "killed" in lowered), float("contract" in lowered or "agreement" in lowered)])
        return vectors


class DenseRetrieverTests(unittest.TestCase):
    def test_index_and_search_returns_semantic_match(self) -> None:
        records = [
            {
                "chunk_id": "act_penal_section_0302",
                "act_title": "The Penal Code, 1860",
                "act_year": "1860",
                "section_id": "302",
                "section_position": 302,
                "section_content": "Whoever commits murder shall be punished with death or imprisonment for life.",
                "source_file": "act-print-11.json",
                "source_url": "https://example.test/penal",
                "is_repealed": False,
            },
            {
                "chunk_id": "act_contract_section_0010",
                "act_title": "The Contract Act, 1872",
                "act_year": "1872",
                "section_id": "10",
                "section_position": 10,
                "section_content": "All agreements are contracts if made by free consent.",
                "source_file": "act-print-20.json",
                "source_url": "https://example.test/contract",
                "is_repealed": False,
            },
        ]
        # Chroma may retain a SQLite handle briefly on Windows after this test.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary_directory:
            database_path = Path(temporary_directory) / "chroma"
            model = KeywordEmbedder()
            self.assertEqual(index_records(records, model, database_path=database_path), 2)
            results = search("A person killed another person", model, database_path=database_path, top_k=1)

        self.assertEqual(results[0]["chunk_id"], "act_penal_section_0302")
        self.assertGreater(results[0]["similarity"], 0.9)


if __name__ == "__main__":
    unittest.main()
