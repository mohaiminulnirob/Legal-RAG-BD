import unittest
from src.evaluation.retrieval_ablation import fuse, metrics
from src.retrieval.hybrid import rrf_fuse


class AblationTests(unittest.TestCase):
    def test_equal_weights_match_production(self):
        b = [{'chunk_id': x, 'score': 1} for x in ('a', 'b', 'c')]
        d = [{'chunk_id': x, 'similarity': .5} for x in ('c', 'd', 'b')]
        expected = rrf_fuse(b, d)
        actual = fuse(b, d)
        self.assertEqual([r['chunk_id'] for r in actual], [r['chunk_id'] for r in expected])
        for a, e in zip(actual, expected):
            self.assertAlmostEqual(a['rrf_score'], e['rrf_score'], places=8)

    def test_weight_and_duplicate_handling(self):
        a, b = {'chunk_id': 'a'}, {'chunk_id': 'b'}
        rows = fuse([a, a], [b], weights=(.5, 1.5))
        self.assertEqual(rows[0]['chunk_id'], 'b')
        self.assertAlmostEqual(rows[1]['rrf_score'], .5 / 61)
        with self.assertRaises(ValueError):
            fuse([], [], weights=(0, 0))

    def test_recall_is_not_hit_rate_and_mrr_is_truncated(self):
        m = metrics([{'chunk_id': 'a'}], ['a', 'b'])
        self.assertEqual(m['recall_at_5'], .5)
        self.assertEqual(m['hit_at_5'], 1)
        m = metrics([{'chunk_id': str(i)} for i in range(11)], ['10'])
        self.assertEqual(m['mrr_at_10'], 0)
