"""Tests for retrieval-evaluation metrics and reproducible result exports."""

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.retrieval_eval import aggregate, evaluate, metric_record, write_results


class RetrievalEvaluationTests(unittest.TestCase):
    def test_metrics_use_first_relevant_rank(self) -> None:
        metrics = metric_record(["x", "target", "target2"], ["target", "target2"])
        self.assertEqual(metrics["first_relevant_rank"], 2)
        self.assertEqual(metrics["recall_at_5"], 1)
        self.assertEqual(metrics["recall_at_10"], 1)
        self.assertEqual(metrics["reciprocal_rank"], 0.5)

    def test_aggregate_and_export_are_deterministic(self) -> None:
        queries = [{"query_id": "q1", "query": "question", "relevant_chunk_ids": ["relevant"]}]
        retrievers = {
            "bm25": lambda _query, _limit: [{"chunk_id": "relevant"}],
            "dense": lambda _query, _limit: [{"chunk_id": "other"}],
        }
        per_query, summary = evaluate(queries, retrievers)
        self.assertEqual(summary["bm25"], {"query_count": 1, "recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0})
        self.assertEqual(summary["dense"]["mrr"], 0.0)
        self.assertEqual(aggregate([{"recall_at_5": 1, "recall_at_10": 0, "reciprocal_rank": 0.5}])["mrr"], 0.5)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            write_results(per_query, summary, output)
            self.assertEqual(json.loads((output / "aggregate_results.json").read_text(encoding="utf-8")), summary)


if __name__ == "__main__":
    unittest.main()