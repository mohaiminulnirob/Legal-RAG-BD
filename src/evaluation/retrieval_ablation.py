"""Offline fusion experiments; does not change production retrieval defaults."""

import argparse
import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.retrieval_eval import DEFAULT_QUERIES, load_queries
from src.retrieval import bm25, dense

POOLS = (20, 50, 100)
CONSTANTS = (20, 40, 60, 100)
WEIGHTS = ((1.0, 1.0), (0.8, 1.2), (0.6, 1.4), (0.5, 1.5))


def fuse(lexical, semantic, *, k=60, weights=(1.0, 1.0)):
    """Weighted RRF with production tie-breaking and one vote per source/ID."""
    if k < 0 or len(weights) != 2 or any(not math.isfinite(w) or w < 0 for w in weights) or not any(weights):
        raise ValueError('Invalid RRF constant or weights')
    merged = {}
    for source, rows, weight in zip(('bm25', 'dense'), (lexical, semantic), weights):
        seen = set()
        for rank, row in enumerate(rows, 1):
            cid = row['chunk_id']
            if cid in seen:
                continue
            seen.add(cid)
            item = merged.setdefault(cid, {'chunk_id': cid, 'rrf_score': 0.0})
            item[source + '_rank'] = rank
            item[source + '_score'] = row.get('score' if source == 'bm25' else 'similarity')
            item['rrf_score'] += weight / (k + rank)
    return sorted(merged.values(), key=lambda r: (-r['rrf_score'], min(r.get('bm25_rank', float('inf')), r.get('dense_rank', float('inf'))), r['chunk_id']))


def metrics(rows, relevant):
    """True recall, hit rate, and reciprocal rank truncated at 10."""
    relevant = set(relevant)
    if not relevant:
        raise ValueError('Relevance labels cannot be empty')
    ids = [r['chunk_id'] for r in rows[:10]]
    rank = next((i for i, cid in enumerate(ids, 1) if cid in relevant), None)
    out = {'mrr_at_10': 1 / rank if rank else 0.0}
    for cutoff in (5, 10):
        hits = relevant.intersection(ids[:cutoff])
        out[f'recall_at_{cutoff}'] = len(hits) / len(relevant)
        out[f'hit_at_{cutoff}'] = float(bool(hits))
    return out


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, default=DEFAULT_QUERIES)
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; existing runs are never overwritten')
    args = parser.parse_args()
    queries = load_queries(args.queries)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    known = {r['chunk_id'] for r in bm25.load_index()['documents']}
    for query in queries:
        if not set(query['relevant_chunk_ids']) <= known:
            raise ValueError(f"Unknown relevance ID in {query['query_id']}")
    settings = [dict(candidate_k=p, rrf_k=k, bm25_weight=a, dense_weight=b)
                for p, k, (a, b) in itertools.product(POOLS, CONSTANTS, WEIGHTS)]
    manifest = {
        'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'query_sha256': digest(args.queries), 'query_count': len(queries),
        'bm25_sha256': digest(bm25.DEFAULT_INDEX), 'embedding_model': dense.MODEL_NAME,
        'settings': settings, 'final_cutoffs': [5, 10],
        'note': 'Development-set tuning, not held-out performance. MRR is truncated at 10. Dense is queried separately at each pool size; ANN prefixes may differ.',
        'source_sha256': {p: digest(p) for p in ('src/retrieval/bm25.py', 'src/retrieval/dense.py', 'src/retrieval/hybrid.py', __file__)},
    }
    write_json(args.output_dir / 'manifest.json', manifest)
    model = dense.create_embedding_model()
    records = []
    # Diagnostic is excluded from benchmark averages and label selection.
    diagnostic = {'query_id': 'diagnostic_murder', 'query': 'What punishment applies for murder?',
                  'relevant_chunk_ids': ['act_act-print-11_section_0344']}
    for index, query in enumerate([*queries, diagnostic], 1):
        lexical = bm25.search(query['query'], top_k=max(POOLS))
        semantic = {p: dense.search(query['query'], model, top_k=p) for p in POOLS}
        runs = []
        for config in settings:
            p = config['candidate_k']
            rows = fuse(lexical[:p], semantic[p], k=config['rrf_k'], weights=(config['bm25_weight'], config['dense_weight']))
            runs.append({'config': config, 'metrics': metrics(rows, query['relevant_chunk_ids']),
                         'first_relevant_rank_in_pool': next((i for i, r in enumerate(rows, 1) if r['chunk_id'] in query['relevant_chunk_ids']), None),
                         'top_10': rows[:10]})
        records.append({'query': query, 'runs': runs})
        # Checkpoint raw ranks for independent reproduction of every fusion result.
        write_json(args.output_dir / f"candidates_{query['query_id']}.json", {
            'query': query, 'bm25': [{k: r[k] for k in ('chunk_id', 'score')} for r in lexical],
            'dense': {p: [{k: r[k] for k in ('chunk_id', 'similarity')} for r in rows] for p, rows in semantic.items()}})
        print(f"Completed {index}/{len(queries)+1}: {query['query_id']}", flush=True)
    summaries = []
    for i, config in enumerate(settings):
        values = [r['runs'][i]['metrics'] for r in records[:-1]]
        summaries.append({'config': config, **{key: sum(v[key] for v in values) / len(values) for key in values[0]}})
    write_json(args.output_dir / 'per_query_results.json', records)
    write_json(args.output_dir / 'aggregate_results.json', summaries)
    manifest['status'] = 'complete'
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(args.output_dir / 'manifest.json', manifest)
    print(f'Completed ablation: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
