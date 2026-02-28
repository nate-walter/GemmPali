#!/usr/bin/env python3
import argparse, glob, json, os
from pathlib import Path
import pyarrow.parquet as pq


def iter_rows(parquet_glob):
    for f in sorted(glob.glob(parquet_glob)):
        t = pq.read_table(f)
        d = t.to_pydict()
        n = t.num_rows
        for i in range(n):
            row = {k: d[k][i] for k in d.keys()}
            yield row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--docvqa-glob', required=True)
    ap.add_argument('--colpali-glob', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    corpus_path = out / 'docvqa_corpus.jsonl'
    query_path = out / 'docvqa_queries.jsonl'
    warmup_path = out / 'phase1_warmup_pairs.jsonl'

    # Build corpus/query split from vidore/docvqa_train
    seen_doc = set()
    q_count = c_count = w_count = 0

    with corpus_path.open('w') as fc, query_path.open('w') as fq:
        for r in iter_rows(args.docvqa_glob):
            image_filename = r.get('image_filename')
            if image_filename and image_filename not in seen_doc:
                seen_doc.add(image_filename)
                fc.write(json.dumps({
                    'doc_id': image_filename,
                    'image_filename': image_filename,
                    'source': r.get('source', 'vidore/docvqa_train')
                }) + '\n')
                c_count += 1

            fq.write(json.dumps({
                'query_id': f"docvqa::{q_count}",
                'query': r.get('query', ''),
                'target_doc_id': image_filename,
                'answer': r.get('answer', ''),
                'source': r.get('source', 'vidore/docvqa_train')
            }) + '\n')
            q_count += 1

    # Build Phase-1 warmup pairs from vidore/colpali_train_set
    with warmup_path.open('w') as fw:
        for r in iter_rows(args.colpali_glob):
            fw.write(json.dumps({
                'pair_id': f"warmup::{w_count}",
                'query': r.get('query', ''),
                'target_doc_id': r.get('image_filename'),
                'answer': r.get('answer', ''),
                'source': r.get('source', 'vidore/colpali_train_set'),
                'meta': {
                    'options': r.get('options'),
                    'page': r.get('page'),
                    'model': r.get('model'),
                    'prompt': r.get('prompt'),
                    'answer_type': r.get('answer_type')
                }
            }) + '\n')
            w_count += 1

    summary = {
        'docvqa_corpus_docs': c_count,
        'docvqa_queries': q_count,
        'phase1_warmup_pairs': w_count,
        'out_dir': str(out)
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
