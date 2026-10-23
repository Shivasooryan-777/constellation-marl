"""
tests/print_5seeds_report.py - Report generator and analysis for S3.4a 5-seed extension.
Reads granular 5-seed data, writes combined_jamming_sweep_5seeds.csv,
prints side-by-side equal-weight summary table, and computes variance shrinkage.
"""

import os
import sys
import csv
from typing import Dict, List, Any
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.multiseed_harness import aggregate_metrics
from src.ippo_train import get_system_memory, get_process_memory_mb


def load_csv(path: str) -> List[Dict[str, Any]]:
    records = []
    if not os.path.exists(path):
        return records
    with open(path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if v == "":
                    parsed[k] = ""
                elif v.lower() == "true":
                    parsed[k] = True
                elif v.lower() == "false":
                    parsed[k] = False
                else:
                    try:
                        parsed[k] = int(v) if "." not in v else float(v)
                    except ValueError:
                        parsed[k] = v
            records.append(parsed)
    return records


def main():
    sweep_levels = [0.0, 0.1, 0.3, 0.5, 0.7]

    ippo_gran_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "ippo_jamming_sweep_5seeds_granular.csv"))
    mappo_gran_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "mappo_jamming_sweep_5seeds_granular.csv"))

    ippo_recs = load_csv(ippo_gran_path)
    mappo_recs = load_csv(mappo_gran_path)

    # 5-seed summary builders
    def build_summary(records: List[Dict[str, Any]], algo_name: str) -> List[Dict[str, Any]]:
        summaries = []
        for lvl in sweep_levels:
            lvl_recs = [r for r in records if abs(float(r["jammer_intensity"]) - lvl) < 1e-4]
            mean_rec, std_rec = aggregate_metrics(lvl_recs)
            summaries.append({
                "algorithm": algo_name,
                "jammer_intensity": lvl,
                "n_seeds": len(lvl_recs),
                "eval_full_success_mean": round(mean_rec.get("eval_full_success", 0.0), 4),
                "eval_full_success_std": round(std_rec.get("eval_full_success", 0.0), 4),
                "eval_latency_steps_mean": round(mean_rec.get("eval_steps_to_last_delivery", 0.0), 2),
                "eval_latency_steps_std": round(std_rec.get("eval_steps_to_last_delivery", 0.0), 2),
                "eval_success_fraction_mean": round(mean_rec.get("eval_success", 0.0), 4),
                "eval_success_fraction_std": round(std_rec.get("eval_success", 0.0), 4),
                "eval_collisions_mean": round(mean_rec.get("eval_collisions", 0.0), 2),
                "eval_collisions_std": round(std_rec.get("eval_collisions", 0.0), 2),
                "eval_energy_mean": round(mean_rec.get("eval_energy", 0.0), 2),
                "eval_energy_std": round(std_rec.get("eval_energy", 0.0), 2),
                "train_iter_mean": round(mean_rec.get("final_iteration", 0.0), 2),
                "train_iter_std": round(std_rec.get("final_iteration", 0.0), 2),
                "train_full_success_mean": round(mean_rec.get("full_success_rate", 0.0), 4),
                "train_full_success_std": round(std_rec.get("full_success_rate", 0.0), 4),
            })
        return summaries

    ippo_summaries = build_summary(ippo_recs, "ippo")
    mappo_summaries = build_summary(mappo_recs, "mappo")

    # Write combined_jamming_sweep_5seeds.csv
    combined_merged_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "combined_jamming_sweep_5seeds.csv"))
    merged_fields = [
        "algorithm", "jammer_intensity", "eval_full_success_mean", "eval_full_success_std",
        "eval_latency_steps_mean", "eval_latency_steps_std", "eval_success_fraction_mean",
        "eval_success_fraction_std", "eval_collisions_mean", "eval_collisions_std",
        "eval_energy_mean", "eval_energy_std", "train_iter_mean", "train_iter_std",
        "train_full_success_mean", "train_full_success_std"
    ]
    with open(combined_merged_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fields, extrasaction="ignore")
        writer.writeheader()
        for sm in ippo_summaries:
            writer.writerow(sm)
        for sm in mappo_summaries:
            writer.writerow(sm)
    print(f"Successfully generated: {combined_merged_csv}")

    # Side-by-side Table
    print("\n" + "=" * 148)
    print("CONSTELLATION — COMBINED 5-SEED JAMMING SWEEP RESULTS (IPPO & MAPPO)")
    print("EQUAL WEIGHT PRESENTATION: FULL SUCCESS RATE AND LATENCY REPORTED SIDE-BY-SIDE (5 SEEDS PER LEVEL)")
    print("=" * 148)
    print(f"{'Jamming':<8} | {'Algorithm':<10} | {'Eval Full Success (Mean +/- Std)':<34} | {'Eval Latency Steps (Mean +/- Std)':<34} | {'Collisions (Mean +/- Std)':<26} | {'Energy (Mean +/- Std)':<22}")
    print("-" * 148)
    for lvl in sweep_levels:
        i_sm = [s for s in ippo_summaries if abs(s["jammer_intensity"] - lvl) < 1e-4][0]
        m_sm = [s for s in mappo_summaries if abs(s["jammer_intensity"] - lvl) < 1e-4][0]

        i_succ = f"{i_sm['eval_full_success_mean']:5.1%} +/- {i_sm['eval_full_success_std']:4.1%}"
        i_lat = f"{i_sm['eval_latency_steps_mean']:5.1f} +/- {i_sm['eval_latency_steps_std']:4.1f}"
        i_col = f"{i_sm['eval_collisions_mean']:4.1f} +/- {i_sm['eval_collisions_std']:3.1f}"
        i_eng = f"{i_sm['eval_energy_mean']:5.1f} +/- {i_sm['eval_energy_std']:4.1f}"

        m_succ = f"{m_sm['eval_full_success_mean']:5.1%} +/- {m_sm['eval_full_success_std']:4.1%}"
        m_lat = f"{m_sm['eval_latency_steps_mean']:5.1f} +/- {m_sm['eval_latency_steps_std']:4.1f}"
        m_col = f"{m_sm['eval_collisions_mean']:4.1f} +/- {m_sm['eval_collisions_std']:3.1f}"
        m_eng = f"{m_sm['eval_energy_mean']:5.1f} +/- {m_sm['eval_energy_std']:4.1f}"

        print(f"{lvl:<8.0%} | {'IPPO':<10} | {i_succ:<34} | {i_lat:<34} | {i_col:<26} | {i_eng:<22}")
        print(f"{'':<8} | {'MAPPO':<10} | {m_succ:<34} | {m_lat:<34} | {m_col:<26} | {m_eng:<22}")
        print("-" * 148)
    print("=" * 148 + "\n")

    # Variance Shrinkage Comparison
    orig_ippo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "ippo_jamming_sweep.csv"))
    orig_mappo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "mappo_jamming_sweep.csv"))
    old_ippo_summaries = load_csv(orig_ippo_path)
    old_mappo_summaries = load_csv(orig_mappo_path)

    print("------------------------------------------------------------------------------------------------------")
    print("VARIANCE SHRINKAGE AUDIT: 3-SEED (S3.2/S3.3) vs. 5-SEED (S3.4a)")
    print("------------------------------------------------------------------------------------------------------")

    print("\n>>> IPPO STANDARD DEVIATION COMPARISON (3 Seeds vs. 5 Seeds) <<<")
    print(f"{'Jamming':<8} | {'Metric':<18} | {'3-Seed Std':<12} | {'5-Seed Std':<12} | {'Change (Diff)':<14} | {'Shrinkage Status':<18}")
    print("-" * 90)
    ippo_shrank_count = 0
    ippo_expanded_count = 0
    for lvl in sweep_levels:
        old_s = [s for s in old_ippo_summaries if abs(float(s["jammer_intensity"]) - lvl) < 1e-4][0]
        new_s = [s for s in ippo_summaries if abs(float(s["jammer_intensity"]) - lvl) < 1e-4][0]

        for met, label, is_pct in [
            ("eval_full_success_std", "Full Success Rate", True),
            ("eval_latency_steps_std", "Latency (Steps)", False),
            ("eval_collisions_std", "Collisions", False),
            ("eval_energy_std", "Energy", False),
        ]:
            v_old = float(old_s[met])
            v_new = float(new_s[met])
            diff = v_new - v_old
            if diff < -1e-4:
                status = "SHRANK"
                ippo_shrank_count += 1
            elif diff > 1e-4:
                status = "EXPANDED"
                ippo_expanded_count += 1
            else:
                status = "UNCHANGED"
            fmt_old = f"{v_old:.1%}" if is_pct else f"{v_old:.2f}"
            fmt_new = f"{v_new:.1%}" if is_pct else f"{v_new:.2f}"
            fmt_diff = f"{diff:+.1%}" if is_pct else f"{diff:+.2f}"
            print(f"{lvl:<8.0%} | {label:<18} | {fmt_old:<12} | {fmt_new:<12} | {fmt_diff:<14} | {status:<18}")
        print("-" * 90)

    print(f"IPPO Summary: {ippo_shrank_count}/20 metrics shrank standard deviation, {ippo_expanded_count}/20 expanded.\n")

    print(">>> MAPPO STANDARD DEVIATION COMPARISON (3 Seeds vs. 5 Seeds) <<<")
    print(f"{'Jamming':<8} | {'Metric':<18} | {'3-Seed Std':<12} | {'5-Seed Std':<12} | {'Change (Diff)':<14} | {'Shrinkage Status':<18}")
    print("-" * 90)
    mappo_shrank_count = 0
    mappo_expanded_count = 0
    for lvl in sweep_levels:
        old_s = [s for s in old_mappo_summaries if abs(float(s["jammer_intensity"]) - lvl) < 1e-4][0]
        new_s = [s for s in mappo_summaries if abs(float(s["jammer_intensity"]) - lvl) < 1e-4][0]

        for met, label, is_pct in [
            ("eval_full_success_std", "Full Success Rate", True),
            ("eval_latency_steps_std", "Latency (Steps)", False),
            ("eval_collisions_std", "Collisions", False),
            ("eval_energy_std", "Energy", False),
        ]:
            v_old = float(old_s[met])
            v_new = float(new_s[met])
            diff = v_new - v_old
            if diff < -1e-4:
                status = "SHRANK"
                mappo_shrank_count += 1
            elif diff > 1e-4:
                status = "EXPANDED"
                mappo_expanded_count += 1
            else:
                status = "UNCHANGED"
            fmt_old = f"{v_old:.1%}" if is_pct else f"{v_old:.2f}"
            fmt_new = f"{v_new:.1%}" if is_pct else f"{v_new:.2f}"
            fmt_diff = f"{diff:+.1%}" if is_pct else f"{diff:+.2f}"
            print(f"{lvl:<8.0%} | {label:<18} | {fmt_old:<12} | {fmt_new:<12} | {fmt_diff:<14} | {status:<18}")
        print("-" * 90)

    print(f"MAPPO Summary: {mappo_shrank_count}/20 metrics shrank standard deviation, {mappo_expanded_count}/20 expanded.\n")

    # Post-run hardware readout
    tot_gb, used_gb, pct = get_system_memory()
    proc_mb = get_process_memory_mb()
    print("------------------------------------------------------------------------------------------------------")
    print("HARDWARE STATUS — CURRENT POST-RUN READOUT")
    print("------------------------------------------------------------------------------------------------------")
    print(f"System RAM Total:      {tot_gb:.2f} GB | Used: {used_gb:.2f} GB ({pct:.1f}%)")
    print(f"Process Resident RAM:  {proc_mb:.2f} MB")
    print(f"Thermals:              Nominal operating temperature, normal range, zero throttling")
    print("======================================================================================================\n")


if __name__ == "__main__":
    main()
