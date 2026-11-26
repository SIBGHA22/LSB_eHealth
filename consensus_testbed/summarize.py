"""Aggregate results/raw/*.json into results/summary.csv and results/summary.md (mean +- 95% CI over seeds)."""
import glob, json, math, os, statistics as st, sys, csv
out = sys.argv[1] if len(sys.argv) > 1 else "results"
rows = [json.load(open(f)) for f in glob.glob(os.path.join(out, "raw", "*.json"))]
for r, f in zip(rows, glob.glob(os.path.join(out, "raw", "*.json"))):
    r["delay_ms"] = int(os.path.basename(f).split("_d")[1].split("_")[0])
T = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
def ci(v): return 0.0 if len(v) < 2 else T.get(len(v), 1.96) * st.stdev(v) / math.sqrt(len(v))
M = ["latency_ms_median", "latency_ms_p95", "throughput_tx_s", "messages_per_tx", "cpu_ms_per_tx",
     "unsafe_commits", "false_rejects", "timeouts", "view_changes"]
groups = {}
for r in rows:
    k = (r["nodes"], r["delay_ms"], r["scenario"], r["concurrency"], r["mode"])
    groups.setdefault(k, []).append(r)
rec = []
for k in sorted(groups):
    g = groups[k]; d = dict(zip(["nodes", "delay_ms", "scenario", "concurrency", "mode"], k))
    d["seeds"] = len(g); d["committee"] = g[0]["committee_size"]; d["txs_per_seed"] = g[0]["txs"]
    for m in M:
        v = [x[m] for x in g]; d[m] = round(st.mean(v), 3); d[m + "_ci95"] = round(ci(v), 3)
    rec.append(d)
with open(os.path.join(out, "summary.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rec[0].keys())); w.writeheader(); w.writerows(rec)
lines = ["| N | delay | scenario | clients | mode | k | median ms | p95 ms | tx/s | msgs/tx | CPU ms/tx | unsafe commits | false rejects | timeouts |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for d in rec:
    lines.append(f"| {d['nodes']} | {d['delay_ms']} | {d['scenario']} | {d['concurrency']} | {d['mode']} | {d['committee']} | "
                 f"{d['latency_ms_median']:.2f} ± {d['latency_ms_median_ci95']:.2f} | {d['latency_ms_p95']:.2f} | "
                 f"{d['throughput_tx_s']:.1f} ± {d['throughput_tx_s_ci95']:.1f} | {d['messages_per_tx']:.1f} | {d['cpu_ms_per_tx']:.2f} | "
                 f"{d['unsafe_commits']:.1f} | {d['false_rejects']:.1f} | {d['timeouts']:.1f} |")
open(os.path.join(out, "summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
