"""
scripts/build_lite_viewer.py - S4.1: bake one replay JSON and the existing sweep-CSV rows into a single
self-contained HTML file (viewer/lite_viewer.html) so it opens by double-click, no server needed
(browsers block fetch() of local JSON from file://).

Usage: python scripts/build_lite_viewer.py [replay.json]
Sweep numbers are copied verbatim from logs/{ippo,mappo}_jamming_sweep_newreward.csv; nothing is computed.
"""

import csv
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
replay_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "replays", "s4_1_sample_mappo_jam50.json")

sweep = {}
for algo in ("ippo", "mappo"):
    with open(os.path.join(ROOT, "logs", f"{algo}_jamming_sweep_newreward.csv"), newline="") as f:
        sweep[algo] = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]

with open(replay_path) as f:
    replay_text = json.dumps(json.load(f), separators=(",", ":"))
with open(os.path.join(ROOT, "viewer", "lite_viewer.template.html"), encoding="utf-8") as f:
    html = f.read()

assert "</script" not in replay_text
html = html.replace("__REPLAY_JSON__", replay_text).replace("__SWEEP_JSON__", json.dumps(sweep, separators=(",", ":")))
out = os.path.join(ROOT, "viewer", "lite_viewer.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print(f"wrote {out} ({os.path.getsize(out) / 1024:.1f} KB) from {os.path.relpath(replay_path, ROOT)}")
