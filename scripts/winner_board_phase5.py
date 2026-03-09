#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime

OUTDIR = Path('/home/nate/GemmPali/reports/phase5_trackA1_global_3gpu_forensic_v3_gpucanary')
FILES = {
    '2k': OUTDIR / 'scorecard_step_0002000.json',
    '4k': OUTDIR / 'scorecard_step_0004000.json',
    '8k': OUTDIR / 'scorecard_step_0008000.json',
}

GATES = {
    'cpcr10_min': 0.35,
    'cgi_trr_max': 0.40,
}

def load(fp: Path):
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text())
    except Exception:
        return None

def f(v, d=4):
    if v is None:
        return '—'
    if isinstance(v, float):
        return f'{v:.{d}f}'
    return str(v)

rows = {}
for k, fp in FILES.items():
    d = load(fp)
    if not d:
        rows[k] = {'status': 'pending', 'path': str(fp)}
        continue
    m = d.get('metrics', {}) or {}
    fm = d.get('forensic_metrics', {}) or {}
    cpcr = m.get('CPCR@10')
    trr = fm.get('cgi_trr')
    rows[k] = {
        'status': 'ready',
        'path': str(fp),
        'samples': d.get('samples'),
        'hit1': m.get('Hit@1'),
        'hit5': m.get('Hit@5'),
        'hit10': m.get('Hit@10'),
        'mrr10': m.get('MRR@10'),
        'ndcg10': m.get('NDCG@10'),
        'cpcr10': cpcr,
        'vidore_hit10': fm.get('vidore_hit10'),
        'cgi_trr': trr,
        'cpcr_gate': (cpcr is not None and cpcr >= GATES['cpcr10_min']),
        'trr_gate': (trr is not None and trr <= GATES['cgi_trr_max']),
        'trr_evaluable': trr is not None,
        'latency_ms': m.get('latency_ms'),
    }

# Winner picks among ready rows
ready = {k:v for k,v in rows.items() if v.get('status')=='ready'}

def best(metric, higher=True):
    cand = [(k, v.get(metric)) for k,v in ready.items() if isinstance(v.get(metric), (int,float))]
    if not cand:
        return None
    cand.sort(key=lambda x: x[1], reverse=higher)
    return cand[0][0], cand[0][1]

winners = {
    'best_cpcr10': best('cpcr10', higher=True),
    'best_vidore_hit10': best('vidore_hit10', higher=True),
    'best_hit10': best('hit10', higher=True),
    'lowest_latency_ms': best('latency_ms', higher=False),
}

md = []
md.append('# 🏆 Phase5 A1 Winner Board')
md.append('')
md.append(f'- Generated: {datetime.now().isoformat()}')
md.append(f'- Source: `{OUTDIR}`')
md.append('')
md.append('## Gate Scoreboard')
md.append('')
md.append('| Step | Status | Samples | ViDoRe Hit@10 | CPCR@10 | CGI TRR | CPCR Gate (>=0.35) | TRR Gate (<=0.40) | Hit@10 | MRR@10 | NDCG@10 | Latency ms |')
md.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for k in ['2k','4k','8k']:
    r=rows[k]
    md.append(
        f"| {k} | {r.get('status')} | {f(r.get('samples'),0)} | {f(r.get('vidore_hit10'))} | {f(r.get('cpcr10'))} | {f(r.get('cgi_trr'))} | {('✅' if r.get('cpcr_gate') else ('❌' if r.get('cpcr10') is not None else '—'))} | {('✅' if r.get('trr_gate') else ('❌' if r.get('trr_evaluable') else '—'))} | {f(r.get('hit10'))} | {f(r.get('mrr10'))} | {f(r.get('ndcg10'))} | {f(r.get('latency_ms'),1)} |"
    )

md.append('')
md.append('## Winners so far')
md.append('')
for label, val in winners.items():
    if val is None:
        md.append(f'- {label}: pending')
    else:
        md.append(f'- {label}: **{val[0]}** ({f(val[1])})')

md.append('')
md.append('## Notes')
md.append('')
md.append('- Final A1 gate verdict requires all three checkpoints + evaluable CGI TRR.')
md.append('- If CGI TRR is null on a checkpoint, TRR gate is marked not evaluable (—), not fail.')

md_out = OUTDIR / 'WINNER-BOARD.md'
json_out = OUTDIR / 'WINNER-BOARD.json'
md_out.write_text('\n'.join(md) + '\n')
json_out.write_text(json.dumps({'generated_at': datetime.now().isoformat(), 'rows': rows, 'winners': winners}, indent=2))
print(md_out)
print(json_out)
