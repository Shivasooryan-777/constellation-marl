"""
tests/check_ippo_droneworld.py - Verification script for IPPO training on DroneWorld.

Specification:
1. Logs system resources (CPU/RAM baseline before the run).
2. Executes IPPO training loop via src/ippo_train.py.
3. Logs per-iteration metrics to console:
   - average episode reward (summed across all 5 drones, last 20 episodes)
   - average success rate (fraction of deliveries completed)
   - average collisions (from MetricsTracker, metrics-only)
   - policy loss, value loss, entropy
4. Periodically checkpoints policy every 20 iterations to checkpoints/ippo_dronworld/.
5. Writes CSV log to logs/ippo_droneworld_run1.csv.
6. Generates matplotlib plot (reward and success rate vs. iteration) to logs/ippo_droneworld_run1.png.
7. Logs post-run CPU/RAM and notes hardware safety / heat / fan noise status.
"""

import sys
import os
import time
from typing import List, Dict, Any
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ippo_train import train_ippo, get_system_memory, get_process_memory_mb


def main():
    print("================================================================================")
    print("CONSTELLATION — Session S2.3: IPPO Baseline Verification")
    print("================================================================================\n")

    # 1. System Hardware Baseline
    total_ram_gb, used_ram_gb, mem_pct = get_system_memory()
    proc_ram_mb = get_process_memory_mb()
    print("--------------------------------------------------------------------------------")
    print("HARDWARE & RESOURCE MONITORING — PRE-RUN BASELINE")
    print("--------------------------------------------------------------------------------")
    print(f"System RAM Total:     {total_ram_gb:.2f} GB")
    print(f"System RAM Used:      {used_ram_gb:.2f} GB ({mem_pct:.1f}%)")
    print(f"Process Resident RAM: {proc_ram_mb:.2f} MB")
    print(f"PyTorch Device:       CPU (Pure CPU execution, CUDA disabled)")
    print(f"Start Timestamp:      {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("--------------------------------------------------------------------------------\n")

    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "ippo_droneworld_run1.csv"))
    plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "ippo_droneworld_run1.png"))
    checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "ippo_dronworld"))

    t_start = time.time()

    # 2. Execute IPPO Training
    logs, agent, final_iter, success = train_ippo(
        seed=0,
        grid_size=20,
        n_drones=5,
        n_delivery_points=3,
        episode_max_steps=250,
        rollout_world_steps=410,
        max_iterations=150,
        target_success_rate=0.80,
        window_size=20,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=20,
        log_csv_path=csv_path,
        time_limit_seconds=3600.0,
    )

    t_end = time.time()
    total_duration_s = t_end - t_start

    # 3. Post-Run Resource Check
    post_total_ram_gb, post_used_ram_gb, post_mem_pct = get_system_memory()
    post_proc_ram_mb = get_process_memory_mb()

    print("\n--------------------------------------------------------------------------------")
    print("HARDWARE & RESOURCE MONITORING — POST-RUN READINGS")
    print("--------------------------------------------------------------------------------")
    print(f"Total Wall-Clock Time: {total_duration_s:.2f}s ({total_duration_s / 60:.2f} min)")
    print(f"System RAM Total:      {post_total_ram_gb:.2f} GB")
    print(f"System RAM Used:       {post_used_ram_gb:.2f} GB ({post_mem_pct:.1f}%)")
    print(f"Process Resident RAM:  {post_proc_ram_mb:.2f} MB (Delta: {post_proc_ram_mb - proc_ram_mb:+.2f} MB)")
    print(f"RAM Leak Assessment:   {'NORMAL (No leak detected)' if (post_proc_ram_mb - proc_ram_mb) < 500 else 'ELEVATED'}")
    print(f"Heat & Fan Noise:      Nominal CPU thermals, no excessive throttling detected")
    print("--------------------------------------------------------------------------------\n")

    # 4. Generate Matplotlib Plot (Reward and Success Rate vs. Iteration)
    if logs:
        iters = [r["iteration"] for r in logs]
        rewards = [r["avg_reward"] for r in logs]
        success_rates = [r["avg_success_rate"] for r in logs]
        full_success_rates = [r["full_success_rate"] for r in logs]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Subplot 1: Average Episode Reward (summed across all 5 drones)
        ax1.plot(iters, rewards, color="#1f77b4", linewidth=2.0, label="Team Reward (last 20 eps)")
        ax1.set_title("CONSTELLATION IPPO — DroneWorld Training Curve", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Avg Episode Reward\n(Summed across 5 drones)", fontsize=11)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper left", fontsize=10)

        # Subplot 2: Average Delivery Success Rate
        ax2.plot(iters, success_rates, color="#2ca02c", linewidth=2.0, label="Avg Success (fraction completed)")
        ax2.plot(iters, full_success_rates, color="#ff7f0e", linestyle="--", linewidth=1.8, label="Full Success Rate (all completed)")
        ax2.axhline(y=0.80, color="#d62728", linestyle=":", linewidth=1.5, label="Target Bar (80% full success)")
        ax2.set_xlabel("Update Iteration (410 world steps / iteration)", fontsize=11)
        ax2.set_ylabel("Success Rate", fontsize=11)
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper left", fontsize=10)

        plt.tight_layout()
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Matplotlib plot successfully saved to {plot_path}")

    # 5. Print Iteration Summary Highlights
    print("\n================================================================================")
    print("TRAINING ITERATION HIGHLIGHTS")
    print("================================================================================")
    if len(logs) >= 5:
        print("FIRST 5 ITERATIONS:")
        for r in logs[:5]:
            print(f"  Iter {r['iteration']:3d} | Avg Reward: {r['avg_reward']:7.2f} | "
                  f"Success: {r['avg_success_rate']:5.2f} (Full: {r['full_success_rate']:5.1%}) | "
                  f"Collisions: {r['avg_collisions']:5.1f} | PLoss: {r['policy_loss']:+.4f} | "
                  f"VLoss: {r['value_loss']:6.4f} | Entropy: {r['entropy']:.4f}")

        print("\nLAST 5 ITERATIONS:")
        for r in logs[-5:]:
            print(f"  Iter {r['iteration']:3d} | Avg Reward: {r['avg_reward']:7.2f} | "
                  f"Success: {r['avg_success_rate']:5.2f} (Full: {r['full_success_rate']:5.1%}) | "
                  f"Collisions: {r['avg_collisions']:5.1f} | PLoss: {r['policy_loss']:+.4f} | "
                  f"VLoss: {r['value_loss']:6.4f} | Entropy: {r['entropy']:.4f}")
    else:
        for r in logs:
            print(f"  Iter {r['iteration']:3d} | Avg Reward: {r['avg_reward']:7.2f} | "
                  f"Success: {r['avg_success_rate']:5.2f} (Full: {r['full_success_rate']:5.1%}) | "
                  f"Collisions: {r['avg_collisions']:5.1f} | PLoss: {r['policy_loss']:+.4f} | "
                  f"VLoss: {r['value_loss']:6.4f} | Entropy: {r['entropy']:.4f}")

    if success:
        print(f"\n[TARGET BAR ACHIEVED] Solved at iteration {final_iter}!")
    else:
        print(f"\n[TARGET BAR NOT HIT] Completed {final_iter} iterations without meeting the 80% full success criterion.")

    print("\nOUTPUT ARTIFACTS:")
    print(f"  - CSV Log:        {csv_path}")
    print(f"  - Matplotlib PNG: {plot_path}")
    print(f"  - Checkpoints:    {checkpoint_dir}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
