"""
scripts/analyze_s3_9_seed1_anomaly.py - S3.9 Category B step 5: read-only scan of EXISTING results for the
"high training success, collapsed eval success" pattern. No training, no checkpoint loading, no new evaluation.

Collapse definition (from the S3.9 spec): training full-success >= 80% AND eval full-success <= 20%.

Sources (all existing): newreward sweep granular CSVs (S3.4g), S3.7 shaping table (log), S3.8 curriculum table (log),
and, if present, the S3.9 extra-seed combined CSV (seeds 2-4 of the shaping ablation).
"""

import csv
import glob
import os
import re
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
runs = []  # dict(source, algo, seed, cond, train, eval)


def add(source, algo, seed, cond, train, ev):
    runs.append({"source": source, "algo": algo, "seed": seed, "cond": cond, "train": train, "eval": ev})


for algo in ("ippo", "mappo"):
    with open(os.path.join(ROOT, "logs", f"{algo}_jamming_sweep_newreward_granular.csv")) as f:
        for r in csv.DictReader(f):
            add("S3.4g sweep", algo, int(r["seed"]), f"jam={float(r['jammer_intensity']):.1f}",
                float(r["full_success_rate"]), float(r["eval_full_success"]))


def last_log(prefix):
    return sorted(glob.glob(os.path.join(ROOT, "logs", "runs", prefix + "_*.log")))[-1]


pat37 = re.compile(r"^(IPPO|MAPPO)\s+\|\s+(\d+)\s+\|\s+(ON|OFF)\s+\|\s+\w+\s+\|\s+\d+\s+\|\s+([\d.]+)%\s+\|\s+([\d.]+)%")
pat38 = re.compile(r"^(IPPO|MAPPO)\s+\|\s+(\d+)\s+\|\s+(DIRECT|CURRICULUM)\s+\|.*?\|\s+([\d.]+)%\s+\|\s+([\d.]+)%\s+\|")
with open(last_log("s3_7_shaping_ablation"), encoding="utf-8", errors="replace") as f:
    for line in f:
        m = pat37.match(line.strip())
        if m:
            add("S3.7 shaping", m[1].lower(), int(m[2]), f"shaping={m[3]} jam=0.0", float(m[4]) / 100, float(m[5]) / 100)
with open(last_log("s3_8_curriculum_ablation"), encoding="utf-8", errors="replace") as f:
    for line in f:
        m = pat38.match(line.strip())
        if m:
            add("S3.8 curriculum", m[1].lower(), int(m[2]), f"{m[3].lower()} jam=0.5", float(m[4]) / 100, float(m[5]) / 100)

extra = os.path.join(ROOT, "logs", "s3_9_shaping_combined_5seeds.csv")
if os.path.exists(extra):
    with open(extra) as f:
        for r in csv.DictReader(f):
            if r["source"] == "S3.9":
                add("S3.9 shaping extra", r["algo"], int(r["seed"]),
                    f"shaping={'ON' if r['shaping'] == 'True' else 'OFF'} jam=0.0", float(r["train_full"]), float(r["eval_full"]))

collapsed = [r for r in runs if r["train"] >= 0.80 and r["eval"] <= 0.20]
print(f"Runs scanned: {len(runs)} | collapsed (train>=80% & eval<=20%): {len(collapsed)}\n")
print("Collapsed runs:")
for r in sorted(collapsed, key=lambda r: (r["algo"], r["seed"], r["source"], r["cond"])):
    print(f"  {r['algo'].upper():<5} seed {r['seed']} | {r['source']:<18} | {r['cond']:<22} | train {r['train']:.0%} -> eval {r['eval']:.0%}")

print("\nCollapse counts by (algo, seed)  [collapsed / total runs]:")
tot, col = defaultdict(int), defaultdict(int)
for r in runs:
    tot[(r["algo"], r["seed"])] += 1
for r in collapsed:
    col[(r["algo"], r["seed"])] += 1
for k in sorted(tot):
    print(f"  {k[0].upper():<5} seed {k[1]}: {col[k]:>2} / {tot[k]:>2}")

print("\nMean eval full-success by (algo, seed) over ALL scanned runs:")
agg = defaultdict(list)
for r in runs:
    agg[(r["algo"], r["seed"])].append(r["eval"])
for k in sorted(agg):
    print(f"  {k[0].upper():<5} seed {k[1]}: {sum(agg[k]) / len(agg[k]):.0%} (n={len(agg[k])})")

print("\nMean (train - eval) gap by seed, both algos pooled:")
gap = defaultdict(list)
for r in runs:
    gap[r["seed"]].append(r["train"] - r["eval"])
for s in sorted(gap):
    print(f"  seed {s}: {sum(gap[s]) / len(gap[s]):+.0%} (n={len(gap[s])})")
