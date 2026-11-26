import json, statistics as st
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path("evidence/final/runs"); FIGS = Path("evidence/final/figures")
FIGS.mkdir(parents=True, exist_ok=True)
DPI = 600
NAVY, ORANGE, TEAL, GREY = '#0072B2', '#E69F00', '#009E73', '#8A8A8A'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':10,
 'axes.titlesize':10.5,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,
 'axes.edgecolor':'#444444','axes.linewidth':0.8,'grid.color':'#D9D9D9','grid.linewidth':0.6,
 'savefig.bbox':'tight','savefig.facecolor':'white','figure.facecolor':'white'})

runs = [json.loads(p.read_text()) for p in
        sorted(RUNS.glob("run-*.json"), key=lambda p: int(p.stem.split('-')[1]))]
WARMUP = 5
warm = runs[WARMUP:]
print(f"{len(runs)} runs, {len(warm)} measured after discarding {WARMUP} warm-up")

KEYS  = ["puf_latency_ms","storage_latency_ms","consensus_latency_ms","total_latency_ms"]
NAMES = ["PUF key\nderivation","Encrypted\nstorage","VRF-PBFT\nconsensus","End-to-end"]
COLS  = [TEAL, TEAL, NAVY, ORANGE]
def stats(k):
    v = sorted(r[k] for r in warm)
    return dict(mean=st.mean(v), median=st.median(v), p95=v[int(0.95*len(v))-1],
                mn=v[0], mx=v[-1], sd=st.stdev(v) if len(v) > 1 else 0.0)
S = {k: stats(k) for k in KEYS}

# ---- Figure A: stage decomposition ----
fig, axes = plt.subplots(1, 2, figsize=(10.4,3.4), gridspec_kw={'width_ratios':[1.3,1]})
ax = axes[0]; x = np.arange(len(KEYS)); w = 0.36
b1 = ax.bar(x-w/2, [S[k]['median'] for k in KEYS], w, color=COLS,
            edgecolor='white', lw=.7, label='Median')
b2 = ax.bar(x+w/2, [S[k]['p95'] for k in KEYS], w, color=COLS, alpha=.45,
            edgecolor='white', lw=.7, label='95th percentile')
for b,k in zip(b1,KEYS): ax.text(b.get_x()+b.get_width()/2, S[k]['median']+1.2,
                                 f"{S[k]['median']:.2f}", ha='center', fontsize=8)
for b,k in zip(b2,KEYS): ax.text(b.get_x()+b.get_width()/2, S[k]['p95']+1.2,
                                 f"{S[k]['p95']:.2f}", ha='center', fontsize=8, color='#555')
ax.set_xticks(x); ax.set_xticklabels(NAMES, fontsize=8.2)
ax.set_ylabel('Latency (ms)'); ax.set_ylim(0, max(S[k]['p95'] for k in KEYS)*1.35)
ax.legend(frameon=False, fontsize=8.5); ax.grid(axis='y', alpha=.7); ax.set_axisbelow(True)
ax.set_title(f'(a) Stage decomposition, {len(warm)} measured runs', fontsize=9.5, loc='left')

ax = axes[1]
cold = runs[0]['total_latency_ms']
vals = [cold, S['total_latency_ms']['median'], S['total_latency_ms']['p95']]
bars = ax.bar(np.arange(3), vals, .5, color=[GREY, ORANGE, '#F0C069'], edgecolor='white', lw=.7)
for b,v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, v+max(vals)*.03,
                                    f'{v:.1f}', ha='center', fontsize=8.5)
ax.axhline(200, color=TEAL, ls='--', lw=1.1)
ax.text(2.45, 205, '200 ms target', fontsize=7.6, color=TEAL, ha='right')
ax.set_xticks(np.arange(3))
ax.set_xticklabels(['Cold start\n(first request)','Steady state\n(median)',
                    'Steady state\n(95th pct)'], fontsize=8.2)
ax.set_ylabel('End-to-end latency (ms)'); ax.set_ylim(0, max(max(vals),200)*1.18)
ax.grid(axis='y', alpha=.7); ax.set_axisbelow(True)
ax.set_title('(b) Cold start versus steady state', fontsize=9.5, loc='left')
fig.savefig(FIGS/'deploy_stage_latency.png', dpi=DPI); plt.close(fig)

# ---- Figure B: per-run stability ----
fig, ax = plt.subplots(figsize=(7.0,3.2))
idx = np.arange(1, len(runs)+1)
tot = [r['total_latency_ms'] for r in runs]
ax.plot(idx[:WARMUP], tot[:WARMUP], 'o--', color=GREY, ms=4, lw=1.0, label='Warm-up (discarded)')
ax.plot(idx[WARMUP:], tot[WARMUP:], 'o-', color=ORANGE, ms=4, lw=1.2, label='Measured')
ax.axhline(S['total_latency_ms']['median'], color=NAVY, ls=':', lw=1.1)
ax.text(len(runs), S['total_latency_ms']['median']*1.06,
        f"median {S['total_latency_ms']['median']:.2f} ms", fontsize=8, color=NAVY, ha='right')
ax.axhline(200, color=TEAL, ls='--', lw=1.1)
ax.text(1, 205, '200 ms target', fontsize=7.6, color=TEAL)
ax.set_xlabel('Run index'); ax.set_ylabel('End-to-end latency (ms)')
ax.set_ylim(0, max(max(tot),200)*1.15)
ax.legend(frameon=False, fontsize=8.5); ax.grid(alpha=.7); ax.set_axisbelow(True)
fig.savefig(FIGS/'deploy_per_run.png', dpi=DPI); plt.close(fig)

# ---- Figure C: service integrity ----
m = json.loads(Path("evidence/final/consensus-service-metrics.json").read_text())
fig, ax = plt.subplots(figsize=(5.4,3.2))
lab = ['Certificates\nissued','Rejected\nvotes','Logical msgs\nper certificate',
       'Validator calls\nper certificate']
c = m['certificates_issued']
vals = [c, m['rejected_votes'], m['logical_messages']/c, m['validator_http_requests']/c]
bars = ax.bar(np.arange(4), vals, .55,
              color=[ORANGE, TEAL, NAVY, NAVY], edgecolor='white', lw=.7)
for b,v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, v+max(vals)*.02,
                                    f'{v:g}', ha='center', fontsize=8.5)
ax.set_xticks(np.arange(4)); ax.set_xticklabels(lab, fontsize=8.2)
ax.set_ylabel('Count'); ax.set_ylim(0, max(vals)*1.2)
ax.grid(axis='y', alpha=.7); ax.set_axisbelow(True)
ax.set_title('Consensus service integrity counters', fontsize=9.5, loc='left')
fig.savefig(FIGS/'deploy_integrity.png', dpi=DPI); plt.close(fig)

print("written:", *[p.name for p in sorted(FIGS.glob('*.png'))])
for k in KEYS: print(f"  {k}: median {S[k]['median']:.2f}  p95 {S[k]['p95']:.2f}  sd {S[k]['sd']:.2f}")
