"""Unit tests for persistent BM25 legal retrieval."""

import json
import tempfile
import unittest
from pathlib import Path

from src.retrieval.bm25 import build_index, search, tokenize


class BM25RetrieverTests(unittest.TestCase):
    def test_tokenize_preserves_legal_section_number(self) -> None:
        self.assertEqual(tokenize("Section 302: Murder's punishment"), ["section", "302", "murder's", "punishment"])

    def test_build_and_search_returns_relevant_section(self) -> None:
        records = [
            {
                "chunk_id": "act_penal_section_0302",
                "act_title": "The Penal Code, 1860",
                "act_year": "1860",
                "section_id": "302",
                "section_position": 302,
                "section_content": "Whoever commits murder shall be punished with death or imprisonment for life.",
                "source_url": "https://example.test/penal",
            },
            {
                "chunk_id": "act_contract_section_0010",
                "act_title": "The Contract Act, 1872",
                "act_year": "1872",
                "section_id": "10",
                "section_position": 10,
                "section_content": "All agreements are contracts if made by free consent.",
                "source_url": "https://example.test/contract",
            },
            {
                "chunk_id": "act_land_section_0001",
                "act_title": "The Land Act, 1885",
                "act_year": "1885",
                "section_id": "1",
                "section_position": 1,
                "section_content": "This Act provides rules concerning land administration.",
                "source_url": "https://example.test/land",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            input_path = directory / "sections.json"
            index_path = directory / "index.pkl"
            input_path.write_text(json.dumps(records), encoding="utf-8")

            self.assertEqual(build_index(input_path, index_path), 3)
            results = search("punishment for murder", index_path, top_k=1)

        self.assertEqual(results[0]["chunk_id"], "act_penal_section_0302")
        self.assertGreater(results[0]["score"], 0)


if __name__ == "__main__":
    unittest.main()
