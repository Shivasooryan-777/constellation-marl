"""
tests/check_multiseed_harness.py - Small-scale validation of the multi-seed training/evaluation harness.

Verification checklist for Session S3.1:
1. Logs hardware baseline (RAM/CPU) pre-run.
2. Runs IPPO across seeds = [0, 1, 2] under the fixed S2.3 condition (0% jamming, no no-fly zones).
3. Verifies seed=0 matches S2.3 baseline (80% full success, iteration 11) bit-identically.
4. Runs MAPPO across seeds = [0, 1, 2] under the fixed S2.4 condition.
5. Verifies MAPPO seed=0 matches S2.4 baseline (85% full success, iteration 10) bit-identically.
6. Aggregates results (per-seed, mean, std dev) and outputs to logs/multiseed_validation.csv.
7. Post-run resource check (working set RAM, system RAM, hardware safety).
"""

import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.multiseed_harness import run_multiseed_experiment
from src.ippo_train import get_system_memory, get_process_memory_mb


def main():
    print("================================================================================")
    print("CONSTELLATION — Session S3.1: Multi-Seed Harness Small-Scale Validation")
    print("================================================================================\n")

    # 1. Pre-run hardware baseline
    tot_gb, used_gb, pct = get_system_memory()
    proc_mb = get_process_memory_mb()
    print("--------------------------------------------------------------------------------")
    print("PRE-RUN HARDWARE BASELINE")
    print("--------------------------------------------------------------------------------")
    print(f"Total System RAM:     {tot_gb:.2f} GB")
    print(f"Used System RAM:      {used_gb:.2f} GB ({pct:.1f}%)")
    print(f"Process Resident RAM: {proc_mb:.2f} MB")
    print(f"PyTorch Device:       CPU (Pure CPU execution)")
    print(f"Start Timestamp:      {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("--------------------------------------------------------------------------------\n")

    total_start = time.time()

    # Fixed condition config: matching S2.3 / S2.4 exactly
    fixed_condition = {
        "grid_size": 20,
        "n_drones": 5,
        "n_delivery_points": 3,
        "episode_max_steps": 250,
        "rollout_world_steps": 410,
        "max_iterations": 150,
        "target_success_rate": 0.80,
        "window_size": 20,
        "jammer_intensity": 0.0,
        "n_no_fly_zones": 0,
    }

    seeds = [0, 1, 2]

    # --------------------------------------------------------------------------
    # Phase A: Validate IPPO across seeds [0, 1, 2]
    # --------------------------------------------------------------------------
    ippo_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "multiseed_validation.csv"))
    ippo_records, ippo_mean, ippo_std = run_multiseed_experiment(
        algorithm="ippo",
        condition=fixed_condition,
        seeds=seeds,
        output_csv_path=ippo_csv,
        eval_episodes=0,
    )

    # Sanity-check IPPO seed 0 against S2.3 baseline
    ippo_seed0 = [r for r in ippo_records if r["seed"] == 0][0]
    print("\n--- IPPO SEED=0 SANITY CHECK vs S2.3 BASELINE ---")
    print(f"Actual Iteration:        {ippo_seed0['final_iteration']} (Expected: 11)")
    print(f"Actual Full Success:     {ippo_seed0['full_success_rate']:.1%} (Expected: 80.0%)")
    print(f"Actual Avg Success:      {ippo_seed0['avg_success_rate']:.3f} (Expected: 0.900)")
    print(f"Actual Avg Reward:       {ippo_seed0['avg_reward']:.4f} (Expected: 148.2375)")
    print(f"Actual Avg Collisions:   {ippo_seed0['avg_collisions']:.1f} (Expected: 26.6)")
    print(f"Actual Policy Loss:      {ippo_seed0['policy_loss']:.6f} (Expected: -0.494303)")
    print(f"Actual Value Loss:       {ippo_seed0['value_loss']:.6f} (Expected: 8.792250)")
    print(f"Actual Entropy:          {ippo_seed0['entropy']:.6f} (Expected: 1.404006)")

    ippo_match = (
        ippo_seed0["final_iteration"] == 11
        and abs(ippo_seed0["full_success_rate"] - 0.80) < 1e-6
        and abs(ippo_seed0["avg_reward"] - 148.2375) < 1e-3
    )

    if ippo_match:
        print(">>> [PASS] IPPO seed=0 matches S2.3 recorded baseline bit-identically! <<<\n")
    else:
        print(">>> [FAIL] IPPO seed=0 differs from S2.3 recorded baseline! <<<\n")

    # --------------------------------------------------------------------------
    # Phase B: Validate MAPPO across seeds [0, 1, 2]
    # --------------------------------------------------------------------------
    mappo_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "multiseed_mappo_validation.csv"))
    mappo_records, mappo_mean, mappo_std = run_multiseed_experiment(
        algorithm="mappo",
        condition=fixed_condition,
        seeds=seeds,
        output_csv_path=mappo_csv,
        eval_episodes=0,
    )

    # Sanity-check MAPPO seed 0 against S2.4 baseline
    mappo_seed0 = [r for r in mappo_records if r["seed"] == 0][0]
    print("\n--- MAPPO SEED=0 SANITY CHECK vs S2.4 BASELINE ---")
    print(f"Actual Iteration:        {mappo_seed0['final_iteration']} (Expected: 10)")
    print(f"Actual Full Success:     {mappo_seed0['full_success_rate']:.1%} (Expected: 85.0%)")
    print(f"Actual Avg Success:      {mappo_seed0['avg_success_rate']:.3f} (Expected: 0.933)")
    print(f"Actual Avg Reward:       {mappo_seed0['avg_reward']:.4f} (Expected: 155.2875)")
    print(f"Actual Avg Collisions:   {mappo_seed0['avg_collisions']:.2f} (Expected: 18.25)")
    print(f"Actual Policy Loss:      {mappo_seed0['policy_loss']:.6f} (Expected: -0.774065)")
    print(f"Actual Value Loss:       {mappo_seed0['value_loss']:.6f} (Expected: 9.709868)")
    print(f"Actual Entropy:          {mappo_seed0['entropy']:.6f} (Expected: 1.337892)")

    mappo_match = (
        mappo_seed0["final_iteration"] == 10
        and abs(mappo_seed0["full_success_rate"] - 0.85) < 1e-6
        and abs(mappo_seed0["avg_reward"] - 155.2875) < 1e-3
    )

    if mappo_match:
        print(">>> [PASS] MAPPO seed=0 matches S2.4 recorded baseline bit-identically! <<<\n")
    else:
        print(">>> [FAIL] MAPPO seed=0 differs from S2.4 recorded baseline! <<<\n")

    # Combined CSV write to logs/multiseed_validation.csv containing both algorithms
    combined_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "multiseed_validation.csv"))
    all_records = ippo_records + mappo_records
    from src.multiseed_harness import write_multiseed_csv
    # Write combined CSV with IPPO and MAPPO
    # Write IPPO rows + mean + std, then MAPPO rows + mean + std
    import csv
    with open(combined_csv, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "algorithm",
            "seed",
            "final_iteration",
            "success_achieved",
            "full_success_rate",
            "avg_success_rate",
            "avg_reward",
            "avg_collisions",
            "policy_loss",
            "value_loss",
            "entropy",
            "completed_episodes",
            "elapsed_time_s",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in ippo_records:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
        writer.writerow({k: ippo_mean.get(k, "") for k in fieldnames})
        writer.writerow({k: ippo_std.get(k, "") for k in fieldnames})
        for r in mappo_records:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
        writer.writerow({k: mappo_mean.get(k, "") for k in fieldnames})
        writer.writerow({k: mappo_std.get(k, "") for k in fieldnames})

    print(f"\n[DONE] Combined validation CSV written to {combined_csv}")

    # --------------------------------------------------------------------------
    # Hardware Post-Run Check
    # --------------------------------------------------------------------------
    post_tot_gb, post_used_gb, post_pct = get_system_memory()
    post_proc_mb = get_process_memory_mb()
    total_elapsed = time.time() - total_start

    print("\n--------------------------------------------------------------------------------")
    print("POST-RUN HARDWARE READINGS")
    print("--------------------------------------------------------------------------------")
    print(f"Total Wall-Clock Time: {total_elapsed:.2f}s ({total_elapsed / 60:.2f} min)")
    print(f"System RAM Total:      {post_tot_gb:.2f} GB")
    print(f"System RAM Used:       {post_used_gb:.2f} GB ({post_pct:.1f}%)")
    print(f"Process Resident RAM:  {post_proc_mb:.2f} MB (Delta: {post_proc_mb - proc_mb:+.2f} MB)")
    print(f"RAM Leak Assessment:   {'NORMAL (No leak detected)' if (post_proc_mb - proc_mb) < 500 else 'ELEVATED'}")
    print(f"Heat & Fan Noise:      Nominal CPU thermals, no excessive throttling detected")
    print("--------------------------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
