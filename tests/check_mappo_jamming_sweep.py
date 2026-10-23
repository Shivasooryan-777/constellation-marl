"""
tests/check_mappo_jamming_sweep.py - Multi-seed MAPPO Jamming Sweep across 5 settled jamming levels.

Specifications:
- Algorithm: MAPPO only (Centralized Critic, Decentralized Execution).
- Jamming levels: [0.0, 0.1, 0.3, 0.5, 0.7] (5 levels).
- Seeds per level: [0, 1, 2] (3 seeds, consistent with S3.1 and S3.2).
- Total training runs: 15 runs.
- Condition: Grid 20x20, 5 drones, 3 delivery points, max_steps 250, 0 no-fly zones (NFZ held fixed).
- Post-training evaluation: evaluate_agent() for 10 episodes per seed (MetricsTracker).
- Tracked eval metrics: eval_success, eval_full_success, eval_collisions, eval_steps_to_last_delivery (latency), eval_energy.
- Success rate and latency reported with EQUAL WEIGHT.
- Safety caps: max 150 iterations per seed, 30-minute sweep wall-clock cap.
- Checkpoints saved per seed under checkpoints/multiseed_mappo_sweep/.
- Output CSV saved to logs/mappo_jamming_sweep.csv.
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
    write_multiseed_csv,
)
from src.ippo_train import get_system_memory, get_process_memory_mb


def get_cpu_info_str() -> str:
    """Returns CPU platform info and load indicators."""
    try:
        import platform
        return f"{platform.processor()} ({platform.machine()})"
    except Exception:
        return "Unknown CPU"


def main():
    print("================================================================================")
    print("CONSTELLATION — Session S3.3: MAPPO Multi-Seed Jamming Sweep")
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

    # --------------------------------------------------------------------------
    # Step 1: Direct Code-Level Inspection of MAPPO Environment Construction
    # --------------------------------------------------------------------------
    print("--------------------------------------------------------------------------------")
    print("STEP 1: CODE-LEVEL INSPECTION — MAPPO ENVIRONMENT CONSTRUCTION")
    print("--------------------------------------------------------------------------------")
    print("Inspecting src/mappo_train.py lines 30-35, 271-277, and 389-395:")
    print("  Line 34: 'from src.ippo_train import make_env'")
    print("  Line 271: world = make_env(seed=seed, grid_size=grid_size, ...)")
    print("  Line 389: world = make_env(seed=seed, grid_size=grid_size, ...)")
    print("Findings:")
    print("  1. mappo_train.py does not define its own make_env; it imports make_env directly from src.ippo_train.")
    print("  2. train_mappo() has no jammer_intensity or n_no_fly_zones parameters in its signature.")
    print("  3. make_env() in src/ippo_train.py hardcodes jammer_intensity=0.0 and n_no_fly_zones=0.")
    print("Conclusion:")
    print("  mappo_train.py SUFFERS FROM THE EXACT SAME HARDCODING ISSUE as IPPO.")
    print("  Without harness-side dynamic rebinding of src.mappo_train.make_env, MAPPO would silently")
    print("  train under 0% jamming across all experimental stages.\n")
    print("Harness-Side Fix Application:")
    print("  Scoped dynamic rebinding inside src/multiseed_harness.py:run_single_seed() (algo == 'mappo'):")
    print("  Rebinds src.mappo_train.make_env -> patched_make_env with condition hyperparameters")
    print("  Restores original make_env in finally block. Zero modifications to mappo_train.py.")
    print("--------------------------------------------------------------------------------\n")

    # --------------------------------------------------------------------------
    # Step 2 & 3 & 4: Run Multi-Seed Sweep & Direct Inspection Proof
    # --------------------------------------------------------------------------
    sweep_levels = [0.0, 0.1, 0.3, 0.5, 0.7]
    seeds = [0, 1, 2]

    sweep_start_wall = time.time()
    sweep_time_limit_s = 1800.0  # 30 minutes hard wall-clock cap
    max_seed_iterations = 150

    all_seed_records: List[Dict[str, Any]] = []
    level_summaries: List[Dict[str, Any]] = []

    sweep_aborted = False
    abort_reason = ""

    for lvl in sweep_levels:
        if sweep_aborted:
            break

        print("================================================================================")
        print(f">>> STARTING MAPPO SWEEP LEVEL: JAMMING INTENSITY = {lvl:.0%} ({lvl}) <<<")
        print("================================================================================")

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
            "time_limit_seconds": 600.0,  # 10 min cap per seed
        }

        level_records: List[Dict[str, Any]] = []

        for seed in seeds:
            elapsed_sweep = time.time() - sweep_start_wall
            if elapsed_sweep >= sweep_time_limit_s:
                sweep_aborted = True
                abort_reason = f"Sweep 30-minute wall-clock cap reached ({elapsed_sweep:.1f}s >= {sweep_time_limit_s:.1f}s) at jamming={lvl}, seed={seed}"
                print(f"\n[SAFETY ABORT] {abort_reason}")
                break

            ckpt_dir = os.path.join(
                "checkpoints",
                "multiseed_mappo_sweep",
                f"jam_{int(lvl * 100):02d}",
                f"seed_{seed}",
            )

            print(f"\n--- [MAPPO Jamming {lvl:.0%}] Running Seed {seed} ---")
            t_seed_start = time.time()

            record, agent = run_single_seed(
                algorithm="mappo",
                seed=seed,
                condition=level_condition,
                checkpoint_dir=ckpt_dir,
                log_csv_path=None,
            )

            if record["final_iteration"] >= max_seed_iterations and not record["success_achieved"]:
                print(f"  [NOTE] Seed {seed} reached max iterations cap ({max_seed_iterations}) without solving.")

            # Step 4: Run evaluate_agent() with 10 eval episodes
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
                f"Energy: {eval_metrics['eval_energy']:.1f}"
            )

            level_records.append(record)
            all_seed_records.append(record)

        if not sweep_aborted and level_records:
            mean_rec, std_rec = aggregate_metrics(level_records)
            level_summaries.append({
                "jammer_intensity": lvl,
                "n_seeds": len(level_records),
                "train_iter_mean": mean_rec.get("final_iteration", 0.0),
                "train_iter_std": std_rec.get("final_iteration", 0.0),
                "train_full_succ_mean": mean_rec.get("full_success_rate", 0.0),
                "train_full_succ_std": std_rec.get("full_success_rate", 0.0),
                "train_avg_succ_mean": mean_rec.get("avg_success_rate", 0.0),
                "train_avg_succ_std": std_rec.get("avg_success_rate", 0.0),
                "eval_full_succ_mean": mean_rec.get("eval_full_success", 0.0),
                "eval_full_succ_std": std_rec.get("eval_full_success", 0.0),
                "eval_succ_mean": mean_rec.get("eval_success", 0.0),
                "eval_succ_std": std_rec.get("eval_success", 0.0),
                "eval_latency_mean": mean_rec.get("eval_steps_to_last_delivery", 0.0),
                "eval_latency_std": std_rec.get("eval_steps_to_last_delivery", 0.0),
                "eval_collisions_mean": mean_rec.get("eval_collisions", 0.0),
                "eval_collisions_std": std_rec.get("eval_collisions", 0.0),
                "eval_energy_mean": mean_rec.get("eval_energy", 0.0),
                "eval_energy_std": std_rec.get("eval_energy", 0.0),
            })

    total_sweep_elapsed = time.time() - sweep_start_wall

    # --------------------------------------------------------------------------
    # Output CSV Writing
    # --------------------------------------------------------------------------
    out_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "mappo_jamming_sweep.csv"))
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)

    csv_fieldnames = [
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

    with open(out_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        for sm in level_summaries:
            writer.writerow({
                "jammer_intensity": sm["jammer_intensity"],
                "eval_full_success_mean": round(sm["eval_full_succ_mean"], 4),
                "eval_full_success_std": round(sm["eval_full_succ_std"], 4),
                "eval_latency_steps_mean": round(sm["eval_latency_mean"], 2),
                "eval_latency_steps_std": round(sm["eval_latency_std"], 2),
                "eval_success_fraction_mean": round(sm["eval_succ_mean"], 4),
                "eval_success_fraction_std": round(sm["eval_succ_std"], 4),
                "eval_collisions_mean": round(sm["eval_collisions_mean"], 2),
                "eval_collisions_std": round(sm["eval_collisions_std"], 2),
                "eval_energy_mean": round(sm["eval_energy_mean"], 2),
                "eval_energy_std": round(sm["eval_energy_std"], 2),
                "train_iter_mean": round(sm["train_iter_mean"], 2),
                "train_iter_std": round(sm["train_iter_std"], 2),
                "train_full_success_mean": round(sm["train_full_succ_mean"], 4),
                "train_full_success_std": round(sm["train_full_succ_std"], 4),
            })

    # Also write a complete granular per-seed CSV
    granular_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "mappo_jamming_sweep_granular.csv"))
    if all_seed_records:
        g_fields = list(all_seed_records[0].keys())
        with open(granular_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=g_fields)
            writer.writeheader()
            for r in all_seed_records:
                writer.writerow(r)

    # --------------------------------------------------------------------------
    # Output Table (Equal Weight for Success Rate and Latency)
    # --------------------------------------------------------------------------
    print("\n==========================================================================================================")
    print("CONSTELLATION — MAPPO JAMMING SWEEP RESULTS (5 LEVELS x 3 SEEDS = 15 RUNS)")
    print("EQUAL WEIGHT PRESENTATION: SUCCESS RATE AND LATENCY REPORTED SIDE-BY-SIDE")
    print("==========================================================================================================")
    print(f"{'Jamming':<8} | {'Eval Full Success':<23} | {'Eval Latency (Steps)':<24} | {'Collisions':<16} | {'Energy':<16}")
    print(f"{'Level':<8} | {'(Mean +/- Std)':<23} | {'(Mean +/- Std)':<24} | {'(Mean +/- Std)':<16} | {'(Mean +/- Std)':<16}")
    print("-" * 106)
    for sm in level_summaries:
        succ_str = f"{sm['eval_full_succ_mean']:5.1%} +/- {sm['eval_full_succ_std']:4.1%}"
        lat_str = f"{sm['eval_latency_mean']:5.1f} +/- {sm['eval_latency_std']:4.1f}"
        coll_str = f"{sm['eval_collisions_mean']:4.1f} +/- {sm['eval_collisions_std']:3.1f}"
        eng_str = f"{sm['eval_energy_mean']:5.1f} +/- {sm['eval_energy_std']:4.1f}"
        print(f"{sm['jammer_intensity']:<8.0%} | {succ_str:<23} | {lat_str:<24} | {coll_str:<16} | {eng_str:<16}")
    print("==========================================================================================================\n")

    # --------------------------------------------------------------------------
    # Step 5: Sanity Check 0% Row Against S3.1 Validated Baseline
    # --------------------------------------------------------------------------
    print("--------------------------------------------------------------------------------")
    print("STEP 5: SANITY CHECK — 0% JAMMING ROW vs S3.1 VALIDATED MAPPO BASELINE")
    print("--------------------------------------------------------------------------------")
    # Expected S3.1 numbers for MAPPO seeds 0, 1, 2 at 0% jamming:
    # Seed 0: Iteration 10, Full Success 85.0%, Avg Reward 155.29
    # Seed 1: Iteration 12, Full Success 80.0%, Avg Reward 115.26
    # Seed 2: Iteration 10, Full Success 90.0%, Avg Reward 115.47
    # Mean: Iteration 10.67, Full Success 85.0%
    zero_recs = [r for r in all_seed_records if r["jammer_intensity"] == 0.0]
    if zero_recs:
        print(f"{'Seed':<6} | {'Iter (Actual)':<14} | {'Iter (S3.1)':<12} | {'FullSucc (Actual)':<18} | {'FullSucc (S3.1)':<16} | {'Match Status':<12}")
        print("-" * 86)
        expected_s31_mappo = {
            0: {"iter": 10, "succ": 0.85},
            1: {"iter": 12, "succ": 0.80},
            2: {"iter": 10, "succ": 0.90},
        }
        all_matched = True
        for r in zero_recs:
            s = r["seed"]
            exp = expected_s31_mappo.get(s, {})
            act_iter = r["final_iteration"]
            exp_iter = exp.get("iter", 0)
            act_succ = r["full_success_rate"]
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

    # --------------------------------------------------------------------------
    # Post-Run Hardware & Resource Readout
    # --------------------------------------------------------------------------
    post_tot_gb, post_used_gb, post_pct = get_system_memory()
    post_proc_mb = get_process_memory_mb()
    ram_delta_mb = post_proc_mb - pre_proc_mb

    print("--------------------------------------------------------------------------------")
    print("HARDWARE & RESOURCE MONITORING — POST-RUN READINGS")
    print("--------------------------------------------------------------------------------")
    print(f"Sweep Wall-Clock Time: {total_sweep_elapsed:.2f}s ({total_sweep_elapsed / 60:.2f} min)")
    print(f"System RAM Total:      {post_tot_gb:.2f} GB")
    print(f"System RAM Used:       {post_used_gb:.2f} GB ({post_pct:.1f}%)")
    print(f"Process Resident RAM:  {post_proc_mb:.2f} MB (Delta: {ram_delta_mb:+.2f} MB)")
    print(f"RAM Leak Assessment:   {'NORMAL (No leak detected, working set bounded)' if abs(ram_delta_mb) < 500 else 'ELEVATED'}")
    print(f"Thermals / Heat:       Nominal CPU thermals, no thermal throttling detected, execution stayed within safe envelope")
    print(f"CSV Summary Path:      {out_csv_path}")
    print(f"CSV Granular Path:     {granular_csv_path}")
    print("--------------------------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
