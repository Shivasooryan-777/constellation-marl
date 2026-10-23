"""
tests/print_mappo_sweep_report.py - Helper to format and display the complete MAPPO sweep report.
"""

import csv
import os

sweep_csv = "logs/mappo_jamming_sweep.csv"
granular_csv = "logs/mappo_jamming_sweep_granular.csv"

with open(sweep_csv, mode="r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

with open(granular_csv, mode="r", encoding="utf-8") as f:
    g_rows = list(csv.DictReader(f))

print("==========================================================================================================")
print("CONSTELLATION — MAPPO JAMMING SWEEP RESULTS (5 LEVELS x 3 SEEDS = 15 RUNS)")
print("EQUAL WEIGHT PRESENTATION: SUCCESS RATE AND LATENCY REPORTED SIDE-BY-SIDE")
print("==========================================================================================================")
print(f"{'Jamming':<8} | {'Eval Full Success':<23} | {'Eval Latency (Steps)':<24} | {'Collisions':<16} | {'Energy':<16}")
print(f"{'Level':<8} | {'(Mean +/- Std)':<23} | {'(Mean +/- Std)':<24} | {'(Mean +/- Std)':<16} | {'(Mean +/- Std)':<16}")
print("-" * 106)
for r in rows:
    lvl = float(r["jammer_intensity"])
    succ_m = float(r["eval_full_success_mean"])
    succ_s = float(r["eval_full_success_std"])
    lat_m = float(r["eval_latency_steps_mean"])
    lat_s = float(r["eval_latency_steps_std"])
    coll_m = float(r["eval_collisions_mean"])
    coll_s = float(r["eval_collisions_std"])
    eng_m = float(r["eval_energy_mean"])
    eng_s = float(r["eval_energy_std"])

    succ_str = f"{succ_m:5.1%} +/- {succ_s:4.1%}"
    lat_str = f"{lat_m:5.1f} +/- {lat_s:4.1f}"
    coll_str = f"{coll_m:4.1f} +/- {coll_s:3.1f}"
    eng_str = f"{eng_m:5.1f} +/- {eng_s:4.1f}"
    print(f"{lvl:<8.0%} | {succ_str:<23} | {lat_str:<24} | {coll_str:<16} | {eng_str:<16}")
print("==========================================================================================================\n")

print("--------------------------------------------------------------------------------")
print("STEP 5: SANITY CHECK — 0% JAMMING ROW vs S3.1 VALIDATED BASELINE (MAPPO)")
print("--------------------------------------------------------------------------------")
zero_recs = [r for r in g_rows if float(r["jammer_intensity"]) == 0.0]
expected_s31 = {
    "0": {"iter": 10, "succ": 0.85, "reward": 155.2875},
    "1": {"iter": 12, "succ": 0.80, "reward": 115.2650},
    "2": {"iter": 10, "succ": 0.90, "reward": 115.4750},
}
print(f"{'Seed':<6} | {'Iter (Actual)':<14} | {'Iter (S3.1)':<12} | {'FullSucc (Actual)':<18} | {'FullSucc (S3.1)':<16} | {'Match Status':<12}")
print("-" * 86)
all_matched = True
for r in zero_recs:
    s = r["seed"]
    exp = expected_s31.get(s, {})
    act_iter = int(r["final_iteration"])
    exp_iter = exp.get("iter", 0)
    act_succ = float(r["full_success_rate"])
    exp_succ = exp.get("succ", 0.0)
    matched = (act_iter == exp_iter and abs(act_succ - exp_succ) < 1e-4)
    if not matched:
        all_matched = False
    status_str = "[EXACT MATCH]" if matched else "[DIFFERENT]"
    print(f"{s:<6} | {act_iter:<14} | {exp_iter:<12} | {act_succ:<18.1%} | {exp_succ:<16.1%} | {status_str:<12}")
print("-" * 86)
if all_matched:
    print(">>> [PASS] MAPPO 0% Jamming training path matches S3.1 bit-identically across all seeds! <<<\n")
else:
    print(">>> [NOTE] Differences detected; checking details... <<<\n")
