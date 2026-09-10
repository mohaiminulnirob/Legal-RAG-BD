"""Tests for cross-encoder reranking without downloading a production model."""

import unittest

from src.retrieval.reranker import rerank_candidates


class FixedScorer:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = None

    def predict(self, sentences, **_kwargs):
        self.pairs = list(sentences)
        return self.scores


class RerankerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [
            {"chunk_id": "a", "act_title": "Penal Code", "section_id": "302", "section_content": "Murder punishment", "rrf_score": 0.03, "source_url": "https://example.test/a"},
            {"chunk_id": "b", "act_title": "Penal Code", "section_id": "300", "section_content": "Murder definition", "rrf_score": 0.02, "source_url": "https://example.test/b"},
            {"chunk_id": "c", "act_title": "Contract Act", "section_id": "10", "section_content": "Valid contracts", "rrf_score": 0.01, "source_url": "https://example.test/c"},
        ]

    def test_scores_pairs_and_ranks_candidates(self) -> None:
        scorer = FixedScorer([0.2, 0.9, 0.1])
        results = rerank_candidates("What is murder?", self.candidates, scorer, top_k=2)
        self.assertEqual([item["chunk_id"] for item in results], ["b", "a"])
        self.assertEqual(scorer.pairs[1], ("What is murder?", "Murder definition"))
        self.assertEqual(results[0]["reranker_score"], 0.9)

    def test_preserves_metadata_and_is_deterministic_for_ties(self) -> None:
        results = rerank_candidates("question", self.candidates, FixedScorer([0.5, 0.5, 0.1]), top_k=3)
        self.assertEqual([item["chunk_id"] for item in results], ["a", "b", "c"])
        self.assertEqual(results[0]["source_url"], "https://example.test/a")
        self.assertEqual(results[0]["rrf_score"], 0.03)

    def test_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            rerank_candidates("", self.candidates, FixedScorer([]))
        with self.assertRaises(ValueError):
            rerank_candidates("q", self.candidates, FixedScorer([]), top_k=0)
        with self.assertRaises(ValueError):
            rerank_candidates("q", [self.candidates[0], self.candidates[0]], FixedScorer([0.1, 0.2]))


if __name__ == "__main__":
    unittest.main()
