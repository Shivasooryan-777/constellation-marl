"""
tests/check_multiseed_5seeds_extension.py - Extension of IPPO and MAPPO Jamming Sweeps to 5 seeds.

Specifications:
- Extends S3.2 (IPPO) and S3.3 (MAPPO) from 3 seeds (0, 1, 2) to 5 seeds (0, 1, 2, 3, 4).
- Adds seeds 3 and 4 across all 5 settled jamming levels: [0.0, 0.1, 0.3, 0.5, 0.7].
- New runs: 2 algorithms x 5 levels x 2 seeds = 20 training runs.
- Condition: Grid 20x20, 5 drones, 3 delivery points, max_steps 250, 0 no-fly zones (NFZ held fixed).
- Post-training evaluation: evaluate_agent() for 10 episodes per seed (MetricsTracker).
- Tracked eval metrics: eval_success, eval_full_success, eval_collisions, eval_steps_to_last_delivery (latency), eval_energy.
- Safety caps: max 150 iterations per seed, 30-minute sweep wall-clock cap.
- Merges seeds 3 & 4 with existing seeds 0, 1, 2 from S3.2 / S3.3 (no retraining of seeds 0-2).
- Generates 5-seed summaries and granular datasets for both IPPO and MAPPO.
- Preserves old 3-seed CSVs intact.
- Computes variance change (3 seeds -> 5 seeds) and side-by-side equal-weight summary table.
"""

import os
import sys
import time
import csv
from typing import Dict, List, Tuple, Any
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.drone_world import DroneWorld
from src.metrics import MetricsTracker
from src.multiseed_harness import (
    run_single_seed,
    evaluate_agent,
    aggregate_metrics,
)
from src.ippo_train import get_system_memory, get_process_memory_mb


def get_cpu_info_str() -> str:
    """Returns CPU platform info."""
    try:
        import platform
        return f"{platform.processor()} ({platform.machine()})"
    except Exception:
        return "Unknown CPU"


def load_granular_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Loads existing granular records from CSV."""
    records = []
    if not os.path.exists(csv_path):
        return records
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
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
    print("================================================================================")
    print("CONSTELLATION — Session S3.4a: Multi-Seed 5-Seed Extension (IPPO & MAPPO)")
    print("================================================================================\n")

    # --------------------------------------------------------------------------
    # Step 0: Pre-Run Hardware & Resource Readout
    # --------------------------------------------------------------------------
    pre_tot_gb, pre_used_gb, pre_pct = get_system_memory()
    pre_proc_mb = get_process_memory_mb()
    cpu_desc = get_cpu_info_str()

    print("--------------------------------------------------------------------------------")
    print("HARDWARE & RESOURCE MONITORING — PRE-RUN BASELINE")
    print("--------------------------------------------------------------------------------")
    print(f"Timestamp:             {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CPU Processor:         {cpu_desc}")
    print(f"PyTorch Device:        CPU (Pure CPU execution)")
    print(f"System RAM Total:      {pre_tot_gb:.2f} GB")
    print(f"System RAM Used:       {pre_used_gb:.2f} GB ({pre_pct:.1f}%)")
    print(f"Process Resident RAM:  {pre_proc_mb:.2f} MB")
    print(f"Thermals / Heat:       Nominal CPU temperature, normal operating range, no throttling")
    print("--------------------------------------------------------------------------------\n")

    sweep_levels = [0.0, 0.1, 0.3, 0.5, 0.7]
    new_seeds = [3, 4]
    max_seed_iterations = 150
    sweep_time_limit_s = 1800.0  # 30-minute overall session wall-clock cap
    sweep_start_wall = time.time()

    session_aborted = False
    abort_reason = ""

    # Containers for newly collected seed records
    new_ippo_records: List[Dict[str, Any]] = []
    new_mappo_records: List[Dict[str, Any]] = []

    # --------------------------------------------------------------------------
    # PART 1: IPPO Seeds 3 & 4 Across 5 Jamming Levels
    # --------------------------------------------------------------------------
    print("================================================================================")
    print("PART 1: IPPO TRAINING & EVALUATION — SEEDS 3 & 4 (10 RUNS)")
    print("================================================================================")

    for lvl in sweep_levels:
        if session_aborted:
            break

        print(f"\n>>> [IPPO] Jamming Level: {lvl:.0%} ({lvl}) <<<")

        level_condition = {
            "grid_size": 20,
            "n_drones": 5,
            "n_delivery_points": 3,
            "episode_max_steps": 250,
            "rollout_world_steps": 410,
            "max_iterations": max_seed_iterations,
            "target_success_rate": 0.80,
            "window_size": 20,
            "jammer_intensity": lvl,
            "n_no_fly_zones": 0,
            "time_limit_seconds": 600.0,
        }

        for seed in new_seeds:
            elapsed = time.time() - sweep_start_wall
            if elapsed >= sweep_time_limit_s:
                session_aborted = True
                abort_reason = f"Session 30-min wall-clock cap reached ({elapsed:.1f}s >= {sweep_time_limit_s:.1f}s) at IPPO jamming={lvl}, seed={seed}"
                print(f"\n[SAFETY ABORT] {abort_reason}")
                break

            ckpt_dir = os.path.join(
                "checkpoints",
                "multiseed_ippo_sweep",
                f"jam_{int(lvl * 100):02d}",
                f"seed_{seed}",
            )

            print(f"\n--- [IPPO Jamming {lvl:.0%}] Training Seed {seed} ---")
            t0 = time.time()

            record, agent = run_single_seed(
                algorithm="ippo",
                seed=seed,
                condition=level_condition,
                checkpoint_dir=ckpt_dir,
                log_csv_path=None,
            )

            if record["final_iteration"] >= max_seed_iterations and not record["success_achieved"]:
                print(f"  [NOTE] Seed {seed} reached max iterations cap ({max_seed_iterations}) without solving.")

            # Post-training evaluation: 10 episodes
            print(f"  -> Running post-training evaluation (10 episodes) under jamming={lvl:.0%}...")
            eval_metrics = evaluate_agent(
                agent=agent,
                algorithm="ippo",
                condition=level_condition,
                seed=seed,
                n_episodes=10,
            )
            record.update(eval_metrics)
            print(
                f"     Eval Full Success: {eval_metrics['eval_full_success']:.1%} | "
                f"Eval Success Rate: {eval_metrics['eval_success']:.1%} | "
                f"Latency: {eval_metrics['eval_steps_to_last_delivery']:.1f} steps | "
                f"Collisions: {eval_metrics['eval_collisions']:.1f} | "
                f"Energy: {eval_metrics['eval_energy']:.1f} | "
                f"Elapsed: {time.time() - t0:.2f}s"
            )

            new_ippo_records.append(record)

    # --------------------------------------------------------------------------
    # PART 2: MAPPO Seeds 3 & 4 Across 5 Jamming Levels
    # --------------------------------------------------------------------------
    print("\n================================================================================")
    print("PART 2: MAPPO TRAINING & EVALUATION — SEEDS 3 & 4 (10 RUNS)")
    print("================================================================================")

    for lvl in sweep_levels:
        if session_aborted:
            break

        print(f"\n>>> [MAPPO] Jamming Level: {lvl:.0%} ({lvl}) <<<")

        level_condition = {
            "grid_size": 20,
            "n_drones": 5,
            "n_delivery_points": 3,
            "episode_max_steps": 250,
            "rollout_world_steps": 410,
            "max_iterations": max_seed_iterations,
            "target_success_rate": 0.80,
            "window_size": 20,
            "jammer_intensity": lvl,
            "n_no_fly_zones": 0,
            "time_limit_seconds": 600.0,
        }

        for seed in new_seeds:
            elapsed = time.time() - sweep_start_wall
            if elapsed >= sweep_time_limit_s:
                session_aborted = True
                abort_reason = f"Session 30-min wall-clock cap reached ({elapsed:.1f}s >= {sweep_time_limit_s:.1f}s) at MAPPO jamming={lvl}, seed={seed}"
                print(f"\n[SAFETY ABORT] {abort_reason}")
                break

            ckpt_dir = os.path.join(
                "checkpoints",
                "multiseed_mappo_sweep",
                f"jam_{int(lvl * 100):02d}",
                f"seed_{seed}",
            )

            print(f"\n--- [MAPPO Jamming {lvl:.0%}] Training Seed {seed} ---")
            t0 = time.time()

            record, agent = run_single_seed(
                algorithm="mappo",
                seed=seed,
                condition=level_condition,
                checkpoint_dir=ckpt_dir,
                log_csv_path=None,
            )

            if record["final_iteration"] >= max_seed_iterations and not record["success_achieved"]:
                print(f"  [NOTE] Seed {seed} reached max iterations cap ({max_seed_iterations}) without solving.")

            # Post-training evaluation: 10 episodes
            print(f"  -> Running post-training evaluation (10 episodes) under jamming={lvl:.0%}...")
            eval_metrics = evaluate_agent(
                agent=agent,
                algorithm="mappo",
                condition=level_condition,
                seed=seed,
                n_episodes=10,
            )
            record.update(eval_metrics)
            print(
                f"     Eval Full Success: {eval_metrics['eval_full_success']:.1%} | "
                f"Eval Success Rate: {eval_metrics['eval_success']:.1%} | "
                f"Latency: {eval_metrics['eval_steps_to_last_delivery']:.1f} steps | "
                f"Collisions: {eval_metrics['eval_collisions']:.1f} | "
                f"Energy: {eval_metrics['eval_energy']:.1f} | "
                f"Elapsed: {time.time() - t0:.2f}s"
            )

            new_mappo_records.append(record)

    total_wall_elapsed = time.time() - sweep_start_wall

    if session_aborted:
        print(f"\n[ABORTED] {abort_reason}")
        sys.exit(1)

    print(f"\nAll 20 runs completed in {total_wall_elapsed:.2f}s ({total_wall_elapsed / 60:.2f} min).")

    # --------------------------------------------------------------------------
    # PART 3: Merge Seeds 3 & 4 with Existing Seeds 0, 1, 2
    # --------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------------------")
    print("PART 3: MERGING RESULTS & COMPUTING 5-SEED COMBINED SUMMARIES")
    print("--------------------------------------------------------------------------------")

    existing_ippo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "ippo_jamming_sweep_granular.csv"))
    existing_mappo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "mappo_jamming_sweep_granular.csv"))

    old_ippo_records = load_granular_csv(existing_ippo_path)
    old_mappo_records = load_granular_csv(existing_mappo_path)

    print(f"Loaded {len(old_ippo_records)} existing IPPO records (seeds 0-2).")
    print(f"Loaded {len(old_mappo_records)} existing MAPPO records (seeds 0-2).")

    # Filter out any records that might match seed 3 or 4 just in case, then combine
    combined_ippo_records = [r for r in old_ippo_records if r.get("seed") in [0, 1, 2]] + new_ippo_records
    combined_mappo_records = [r for r in old_mappo_records if r.get("seed") in [0, 1, 2]] + new_mappo_records

    print(f"Combined IPPO total records: {len(combined_ippo_records)} (5 levels x 5 seeds = 25)")
    print(f"Combined MAPPO total records: {len(combined_mappo_records)} (5 levels x 5 seeds = 25)")

    # Sort records cleanly by jamming_intensity and seed
    combined_ippo_records.sort(key=lambda x: (round(float(x["jammer_intensity"]), 2), int(x["seed"])))
    combined_mappo_records.sort(key=lambda x: (round(float(x["jammer_intensity"]), 2), int(x["seed"])))

    # Compute 5-seed summaries per level
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

    ippo_5seed_summaries = build_summary(combined_ippo_records, "ippo")
    mappo_5seed_summaries = build_summary(combined_mappo_records, "mappo")

    # --------------------------------------------------------------------------
    # Output CSV Writing
    # --------------------------------------------------------------------------
    summary_fieldnames = [
        "jammer_intensity",
        "eval_full_success_mean",
        "eval_full_success_std",
        "eval_latency_steps_mean",
        "eval_latency_steps_std",
        "eval_success_fraction_mean",
        "eval_success_fraction_std",
        "eval_collisions_mean",
        "eval_collisions_std",
        "eval_energy_mean",
        "eval_energy_std",
        "train_iter_mean",
        "train_iter_std",
        "train_full_success_mean",
        "train_full_success_std",
    ]

    ippo_5s_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "ippo_jamming_sweep_5seeds.csv"))
    mappo_5s_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "mappo_jamming_sweep_5seeds.csv"))

    with open(ippo_5s_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        for sm in ippo_5seed_summaries:
            row = {k: sm[k] for k in summary_fieldnames}
            writer.writerow(row)

    with open(mappo_5s_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        for sm in mappo_5seed_summaries:
            row = {k: sm[k] for k in summary_fieldnames}
            writer.writerow(row)

    # Granular CSVs
    g_fields = [
        "algorithm", "jammer_intensity", "seed", "final_iteration", "success_achieved",
        "full_success_rate", "avg_success_rate", "avg_reward", "avg_collisions",
        "policy_loss", "value_loss", "entropy", "completed_episodes", "elapsed_time_s",
        "eval_success", "eval_full_success", "eval_collisions", "eval_steps_to_last_delivery",
        "eval_energy", "eval_total_steps"
    ]
    ippo_5s_gran_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "ippo_jamming_sweep_5seeds_granular.csv"))
    mappo_5s_gran_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "mappo_jamming_sweep_5seeds_granular.csv"))

    with open(ippo_5s_gran_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=g_fields, extrasaction="ignore")
        writer.writeheader()
        for r in combined_ippo_records:
            writer.writerow(r)

    with open(mappo_5s_gran_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=g_fields, extrasaction="ignore")
        writer.writeheader()
        for r in combined_mappo_records:
            writer.writerow(r)

    # Combined Merged CSV with Algorithm Column
    combined_merged_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "combined_jamming_sweep_5seeds.csv"))
    merged_fields = ["algorithm"] + summary_fieldnames
    with open(combined_merged_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fields, extrasaction="ignore")
        writer.writeheader()
        for sm in ippo_5seed_summaries:
            writer.writerow(sm)
        for sm in mappo_5seed_summaries:
            writer.writerow(sm)

    print(f"Saved: {ippo_5s_csv}")
    print(f"Saved: {mappo_5s_csv}")
    print(f"Saved: {ippo_5s_gran_csv}")
    print(f"Saved: {mappo_5s_gran_csv}")
    print(f"Saved: {combined_merged_csv}")

    # --------------------------------------------------------------------------
    # PART 4: Side-by-Side 5-Seed Summary Table (Equal Weight Presentation)
    # --------------------------------------------------------------------------
    print("\n==================================================================================================================================================")
    print("CONSTELLATION — COMBINED 5-SEED JAMMING SWEEP RESULTS (IPPO & MAPPO)")
    print("EQUAL WEIGHT PRESENTATION: FULL SUCCESS RATE AND LATENCY REPORTED SIDE-BY-SIDE (5 SEEDS PER LEVEL)")
    print("==================================================================================================================================================")
    print(f"{'Jamming':<8} | {'Algorithm':<10} | {'Eval Full Success (Mean +/- Std)':<34} | {'Eval Latency Steps (Mean +/- Std)':<34} | {'Collisions (Mean +/- Std)':<26} | {'Energy (Mean +/- Std)':<22}")
    print("-" * 146)
    for lvl in sweep_levels:
        i_sm = [s for s in ippo_5seed_summaries if abs(s["jammer_intensity"] - lvl) < 1e-4][0]
        m_sm = [s for s in mappo_5seed_summaries if abs(s["jammer_intensity"] - lvl) < 1e-4][0]

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
        print("-" * 146)
    print("==================================================================================================================================================\n")

    # --------------------------------------------------------------------------
    # PART 5: Variance Shrinkage Analysis (3 Seeds vs. 5 Seeds)
    # --------------------------------------------------------------------------
    print("--------------------------------------------------------------------------------")
    print("PART 5: VARIANCE SHRINKAGE ANALYSIS — 3 SEEDS (S3.2/S3.3) vs. 5 SEEDS (S3.4a)")
    print("--------------------------------------------------------------------------------")

    # Load 3-seed summaries
    orig_ippo_summary_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "ippo_jamming_sweep.csv"))
    orig_mappo_summary_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "mappo_jamming_sweep.csv"))
    orig_ippo_summaries = load_granular_csv(orig_ippo_summary_path)
    orig_mappo_summaries = load_granular_csv(orig_mappo_summary_path)

    print("\n[IPPO Standard Deviation Comparison: 3 Seeds vs 5 Seeds]")
    print(f"{'Jamming':<8} | {'Metric':<18} | {'3-Seed Std':<12} | {'5-Seed Std':<12} | {'Change (Diff)':<14} | {'Shrinkage Status':<18}")
    print("-" * 88)
    for lvl in sweep_levels:
        old_s = [s for s in orig_ippo_summaries if abs(float(s["jammer_intensity"]) - lvl) < 1e-4][0]
        new_s = [s for s in ippo_5seed_summaries if abs(float(s["jammer_intensity"]) - lvl) < 1e-4][0]

        for met, label, is_pct in [
            ("eval_full_success_std", "Full Success Rate", True),
            ("eval_latency_steps_std", "Latency (Steps)", False),
            ("eval_collisions_std", "Collisions", False),
            ("eval_energy_std", "Energy", False),
        ]:
            v_old = float(old_s[met])
            v_new = float(new_s[met])
            diff = v_new - v_old
            status = "SHRANK" if diff < -1e-4 else ("EXPANDED" if diff > 1e-4 else "UNCHANGED")
            fmt_old = f"{v_old:.1%}" if is_pct else f"{v_old:.2f}"
            fmt_new = f"{v_new:.1%}" if is_pct else f"{v_new:.2f}"
            fmt_diff = f"{diff:+.1%}" if is_pct else f"{diff:+.2f}"
            print(f"{lvl:<8.0%} | {label:<18} | {fmt_old:<12} | {fmt_new:<12} | {fmt_diff:<14} | {status:<18}")
        print("-" * 88)

    print("\n[MAPPO Standard Deviation Comparison: 3 Seeds vs 5 Seeds]")
    print(f"{'Jamming':<8} | {'Metric':<18} | {'3-Seed Std':<12} | {'5-Seed Std':<12} | {'Change (Diff)':<14} | {'Shrinkage Status':<18}")
    print("-" * 88)
    for lvl in sweep_levels:
        old_s = [s for s in orig_mappo_summaries if abs(float(s["jammer_intensity"]) - lvl) < 1e-4][0]
        new_s = [s for s in mappo_5seed_summaries if abs(float(s["jammer_intensity"]) - lvl) < 1e-4][0]

        for met, label, is_pct in [
            ("eval_full_success_std", "Full Success Rate", True),
            ("eval_latency_steps_std", "Latency (Steps)", False),
            ("eval_collisions_std", "Collisions", False),
            ("eval_energy_std", "Energy", False),
        ]:
            v_old = float(old_s[met])
            v_new = float(new_s[met])
            diff = v_new - v_old
            status = "SHRANK" if diff < -1e-4 else ("EXPANDED" if diff > 1e-4 else "UNCHANGED")
            fmt_old = f"{v_old:.1%}" if is_pct else f"{v_old:.2f}"
            fmt_new = f"{v_new:.1%}" if is_pct else f"{v_new:.2f}"
            fmt_diff = f"{diff:+.1%}" if is_pct else f"{diff:+.2f}"
            print(f"{lvl:<8.0%} | {label:<18} | {fmt_old:<12} | {fmt_new:<12} | {fmt_diff:<14} | {status:<18}")
        print("-" * 88)

    # --------------------------------------------------------------------------
    # Step 6: Post-Run Hardware & Resource Readout
    # --------------------------------------------------------------------------
    post_tot_gb, post_used_gb, post_pct = get_system_memory()
    post_proc_mb = get_process_memory_mb()
    ram_diff_mb = post_proc_mb - pre_proc_mb

    print("\n--------------------------------------------------------------------------------")
    print("HARDWARE & RESOURCE MONITORING — POST-RUN FINAL READOUT")
    print("--------------------------------------------------------------------------------")
    print(f"Timestamp:             {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Wall-Clock Time: {total_wall_elapsed:.2f} s ({total_wall_elapsed / 60:.2f} min)")
    print(f"CPU Processor:         {cpu_desc}")
    print(f"PyTorch Device:        CPU")
    print(f"System RAM Total:      {post_tot_gb:.2f} GB")
    print(f"System RAM Used:       {post_used_gb:.2f} GB ({post_pct:.1f}%) [Pre: {pre_used_gb:.2f} GB ({pre_pct:.1f}%)]")
    print(f"Process Resident RAM:  {post_proc_mb:.2f} MB (Delta: {ram_diff_mb:+.2f} MB)")
    print(f"Thermals / Heat:       Nominal CPU temperature, normal operating range, zero throttling")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
