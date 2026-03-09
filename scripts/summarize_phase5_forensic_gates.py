#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime

OUTDIR = Path('/home/nate/GemmPali/reports/phase5_trackA1_global_3gpu_forensic_v3_gpucanary')
FILES = [
    ('canary10', OUTDIR / 'scorecard_step_0002000_canary10_gpu.json'),
    ('0002000', OUTDIR / 'scorecard_step_0002000.json'),
    ('0004000', OUTDIR / 'scorecard_step_0004000.json'),
    ('0008000', OUTDIR / 'scorecard_step_0008000.json'),
]


def pick(d, path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def fmt(v, nd=4):
    if v is None:
        return '—'
    if isinstance(v, (int, float)):
        return f'{v:.{nd}f}' if isinstance(v, float) else str(v)
    return str(v)

rows = []
for step, fp in FILES:
    if not fp.exists():
        rows.append({
            'step': step,
            'status': 'pending',
            'path': str(fp),
        })
        continue
    d = json.loads(fp.read_text())
    m = pick(d, ['metrics'], {}) or {}
    f = pick(d, ['forensic_metrics'], {}) or {}
    p = pick(d, ['preprocess', 'stats'], {}) or {}
    rows.append({
        'step': step,
        'status': 'ready',
        'samples': pick(d, ['samples']),
        'candidates': pick(d, ['candidates_per_query']),
        'vidore_hit10': f.get('vidore_hit10'),
        'cpcr10': m.get('CPCR@10'),
        'cgi_trr': f.get('cgi_trr'),
        'hit1': m.get('Hit@1'),
        'hit5': m.get('Hit@5'),
        'hit10': m.get('Hit@10'),
        'mrr10': m.get('MRR@10'),
        'ndcg10': m.get('NDCG@10'),
        'latency_ms': m.get('latency_ms'),
        'gpu_decode_count': p.get('gpu_decode_count'),
        'gpu_resize_count': p.get('gpu_resize_count'),
        'cpu_resize_count': p.get('cpu_resize_count'),
        'materialized_count': p.get('materialized_count'),
        'path': str(fp),
    })

md = []
md.append('# Phase5 A1 Forensic Gate Summary (auto-generated)')
md.append('')
md.append(f'- Generated: {datetime.now().isoformat()}')
md.append(f'- Source dir: `{OUTDIR}`')
md.append('')
md.append('## Gate table')
md.append('')
md.append('| Step | Status | Samples | ViDoRe Hit@10 | CPCR@10 | CGI TRR | Hit@1 | Hit@5 | Hit@10 | MRR@10 | NDCG@10 | Latency ms |')
md.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for r in rows:
    md.append(
        f"| {r['step']} | {r['status']} | {fmt(r.get('samples'),0)} | {fmt(r.get('vidore_hit10'))} | {fmt(r.get('cpcr10'))} | {fmt(r.get('cgi_trr'))} | {fmt(r.get('hit1'))} | {fmt(r.get('hit5'))} | {fmt(r.get('hit10'))} | {fmt(r.get('mrr10'))} | {fmt(r.get('ndcg10'))} | {fmt(r.get('latency_ms'),1)} |"
    )

md.append('')
md.append('## Preprocess telemetry')
md.append('')
md.append('| Step | GPU decode count | GPU resize count | CPU resize count | Materialized count |')
md.append('|---|---:|---:|---:|---:|')
for r in rows:
    md.append(
        f"| {r['step']} | {fmt(r.get('gpu_decode_count'),0)} | {fmt(r.get('gpu_resize_count'),0)} | {fmt(r.get('cpu_resize_count'),0)} | {fmt(r.get('materialized_count'),0)} |"
    )

md.append('')
md.append('## Artifact paths')
md.append('')
for r in rows:
    md.append(f"- `{r['step']}` → `{r['path']}` ({r['status']})")

OUT_MD = OUTDIR / 'GATE-SUMMARY.md'
OUT_JSON = OUTDIR / 'GATE-SUMMARY.json'
OUT_MD.write_text('\n'.join(md) + '\n')
OUT_JSON.write_text(json.dumps({'generated_at': datetime.now().isoformat(), 'rows': rows}, indent=2))
print(str(OUT_MD))
print(str(OUT_JSON))
