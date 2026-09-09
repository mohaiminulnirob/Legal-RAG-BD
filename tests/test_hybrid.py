"""Tests for Reciprocal Rank Fusion of legal retrieval results."""

import unittest

from src.retrieval.hybrid import rrf_fuse


def result(chunk_id: str, score: float, *, source: str) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "act_title": "The Penal Code, 1860",
        "section_id": chunk_id[-3:],
        "section_content": f"Content for {chunk_id}",
        "score" if source == "bm25" else "similarity": score,
    }


class HybridRetrieverTests(unittest.TestCase):
    def test_rrf_boosts_documents_present_in_both_retrievers(self) -> None:
        fused = rrf_fuse(
            [result("section_302", 9.0, source="bm25"), result("section_299", 8.0, source="bm25")],
            [result("section_300", 0.9, source="dense"), result("section_302", 0.8, source="dense")],
            rrf_k=60,
        )
        self.assertEqual(fused[0]["chunk_id"], "section_302")
        self.assertEqual(fused[0]["bm25_rank"], 1)
        self.assertEqual(fused[0]["dense_rank"], 2)
        self.assertAlmostEqual(fused[0]["rrf_score"], 1 / 61 + 1 / 62, places=8)

    def test_results_are_merged_and_source_specific_scores_are_preserved(self) -> None:
        fused = rrf_fuse(
            [result("section_302", 4.2, source="bm25")],
            [result("section_302", 0.88, source="dense")],
        )
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["bm25_score"], 4.2)
        self.assertEqual(fused[0]["dense_score"], 0.88)

    def test_single_source_results_and_top_k_are_preserved(self) -> None:
        fused = rrf_fuse(
            [result("section_302", 9.0, source="bm25"), result("section_299", 8.0, source="bm25")],
            [result("section_300", 0.9, source="dense"), result("section_304", 0.8, source="dense")],
            top_k=3,
        )
        self.assertEqual(len(fused), 3)
        self.assertEqual(len({item["chunk_id"] for item in fused}), 3)
        self.assertTrue(any(item["chunk_id"] == "section_299" and "dense_rank" not in item for item in fused))
        self.assertTrue(any(item["chunk_id"] == "section_300" and "bm25_rank" not in item for item in fused))

    def test_invalid_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rrf_fuse([], [], top_k=0)
        with self.assertRaises(ValueError):
            rrf_fuse([], [], rrf_k=-1)


if __name__ == "__main__":
    unittest.main()
