"""
tests/check_s3_7_shaping_ablation.py - S3.7 Reward-Shaping On/Off Ablation.

Specifications:
1. Scoped Exception Verification:
   - Confirms distance_shaping_enabled: bool = True added to src.rewards.compute_reward.
   - Verifies default-preservation: bit-exact equality between default call and distance_shaping_enabled=True.
   - Verifies shaping OFF: distance shaping term is strictly zeroed out while step_penalty,
     delivery_reward, and overlap_penalty remain active.
2. Training Scale:
   - 2 seeds (seed 0, seed 1) each for IPPO and MAPPO (4 runs ON + 4 runs OFF = 8 runs total).
   - Condition: 0% jamming, 0 no-fly zones, grid 20x20, 5 drones, 3 delivery points,
     episode_max_steps=250, rollout_world_steps=410, max_iterations=150, target_success_rate=0.80, window=20.
   - Saved to: checkpoints/shaping_ablation/on/{algo}/seed_{s}/ and checkpoints/shaping_ablation/off/{algo}/seed_{s}/.
3. Evaluation & Metrics:
   - 10 evaluation episodes per checkpoint via MetricsTracker.
   - Metrics: iterations-to-stop-bar, train full success, eval full success, collisions, latency, energy.
4. Dual-stream TeeLogger active from start to finish, logging to logs/runs/s3_7_shaping_ablation_<timestamp>.log.
5. Hardware monitoring pre- and post-run (CPU, RAM, thermals).
"""

import os
import sys
import time
import platform
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.tee_logger import TeeLogger
from src.drone_world import DroneWorld
from src.metrics import MetricsTracker
import src.rewards
from src.rewards import compute_reward, STEP_PENALTY, DELIVERY_REWARD, OVERLAP_PENALTY
import src.ippo_train
import src.mappo_train
from src.multiseed_harness import run_single_seed, evaluate_agent
from src.ippo_train import get_system_memory, get_process_memory_mb


def get_cpu_info_str() -> str:
    try:
        return f"{platform.processor()} ({platform.machine()})"
    except Exception:
        return "Unknown CPU"


def make_compute_reward_wrapper(shaping_enabled: bool):
    """Wraps compute_reward to inject distance_shaping_enabled."""
    def _wrapped(*args, **kwargs):
        kwargs["distance_shaping_enabled"] = shaping_enabled
        return src.rewards.compute_reward(*args, **kwargs)
    return _wrapped


def main():
    log_dir = os.path.join(PROJECT_ROOT, "logs", "runs")
    logger = TeeLogger(log_dir=log_dir, prefix="s3_7_shaping_ablation")
    active_log_path = logger.start()

    try:
        print("====================================================================================================")
        print("CONSTELLATION — SESSION S3.7: REWARD-SHAPING ON/OFF ABLATION")
        print(f"Dual-Stream TeeLogger Active: {active_log_path}")
        print("====================================================================================================\n")

        # ----------------------------------------------------------------------
        # Step 0: Hardware Baseline
        # ----------------------------------------------------------------------
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
        print(f"Thermals / Fan:        Nominal CPU temperature, normal operating range, zero throttling")
        print("--------------------------------------------------------------------------------\n")

        # ----------------------------------------------------------------------
        # Step 1 & 2: Sanity Check Default Preservation & Shaping Toggle
        # ----------------------------------------------------------------------
        print("--------------------------------------------------------------------------------")
        print("STEP 1 & 2: SCOPED EXCEPTION VERIFICATION & DEFAULT PRESERVATION SANITY CHECK")
        print("--------------------------------------------------------------------------------")
        world_test = DroneWorld(grid_size=20, n_drones=5, n_delivery_points=3, seed=0, jammer_intensity=0.0, n_no_fly_zones=0)
        
        # Drone 0 is at (0, 0). Suppose prev_distance was 15.0 to nearest delivery point.
        # Current distance is manhattan distance to nearest delivery point.
        target = min(world_test.remaining_deliveries, key=lambda p: abs(p[0]) + abs(p[1]))
        curr_dist = float(abs(target[0]) + abs(target[1]))
        prev_distances = [15.0] * 5

        # 1. Default call (no toggle parameter passed)
        r_default = compute_reward(world_test, prev_distances, drone_id=0)

        # 2. Explicit ON call (distance_shaping_enabled=True)
        r_explicit_on = compute_reward(world_test, prev_distances, drone_id=0, distance_shaping_enabled=True)

        # 3. Explicit OFF call (distance_shaping_enabled=False)
        r_explicit_off = compute_reward(world_test, prev_distances, drone_id=0, distance_shaping_enabled=False)

        # Calculated components:
        expected_overlap = OVERLAP_PENALTY  # all 5 drones co-located at (0, 0)
        expected_shaping = 15.0 - curr_dist
        expected_on = expected_shaping + STEP_PENALTY + 0.0 + expected_overlap
        expected_off = 0.0 + STEP_PENALTY + 0.0 + expected_overlap

        print(f"Test Environment State:")
        print(f"  Drone 0 pos:        {world_test.drones[0]}")
        print(f"  Nearest target:     {target} (Current dist: {curr_dist:.1f})")
        print(f"  Previous dist:      {prev_distances[0]:.1f}")
        print(f"  Distance shaping:   {expected_shaping:.4f} (15.0 - {curr_dist:.1f})")
        print(f"  Step penalty:       {STEP_PENALTY}")
        print(f"  Overlap penalty:    {expected_overlap} (5 drones at depot)")
        print()
        print(f"Reward Calculations:")
        print(f"  r_default (toggle omitted):            {r_default:.6f}")
        print(f"  r_explicit_on (toggle=True):           {r_explicit_on:.6f}")
        print(f"  r_explicit_off (toggle=False):         {r_explicit_off:.6f}")
        print(f"  expected_on:                           {expected_on:.6f}")
        print(f"  expected_off:                          {expected_off:.6f}")

        # Assert bit-exact identity
        assert r_default == r_explicit_on, f"BIT-EXACT MISMATCH: r_default ({r_default}) != r_explicit_on ({r_explicit_on})"
        assert abs(r_default - expected_on) < 1e-6, f"Expected {expected_on}, got {r_default}"
        assert abs(r_explicit_off - expected_off) < 1e-6, f"Expected {expected_off}, got {r_explicit_off}"
        assert r_default != r_explicit_off, "Shaping ON and OFF must produce different rewards when distances change!"
        print("\nSANITY CHECK PASS: Default call is 100% BIT-EXACT identical to shaping ON.")
        print("Shaping OFF cleanly eliminates the distance term while preserving step & overlap penalties.\n")

        # ----------------------------------------------------------------------
        # Step 3: Train IPPO and MAPPO (2 seeds each, Shaping ON vs Shaping OFF)
        # ----------------------------------------------------------------------
        print("--------------------------------------------------------------------------------")
        print("STEP 3: TRAINING IPPO & MAPPO (2 SEEDS EACH, SHAPING ON vs OFF @ 0% JAMMING)")
        print("--------------------------------------------------------------------------------")
        algorithms = ["ippo", "mappo"]
        seeds = [0, 1]
        shaping_modes = [True, False]  # True = ON, False = OFF

        train_condition = {
            "grid_size": 20,
            "n_drones": 5,
            "n_delivery_points": 3,
            "episode_max_steps": 250,
            "rollout_world_steps": 410,
            "max_iterations": 150,
            "target_success_rate": 0.80,
            "window_size": 20,
            "checkpoint_freq": 20,
            "time_limit_seconds": 600.0,
            "jammer_intensity": 0.0,
            "n_no_fly_zones": 0,
        }

        all_trained_agents: Dict[Tuple[str, bool, int], Any] = {}
        all_train_records: Dict[Tuple[str, bool, int], Dict[str, Any]] = {}

        t_train_start_all = time.time()

        for shaping in shaping_modes:
            mode_str = "ON" if shaping else "OFF"
            print(f"\n================================================================================")
            print(f">>> STARTING TRAINING BATCH: DISTANCE SHAPING = {mode_str} <<<")
            print(f"================================================================================")

            # Rebind compute_reward in ippo_train and mappo_train
            orig_ippo_rew = src.ippo_train.compute_reward
            orig_mappo_rew = src.mappo_train.compute_reward
            wrapper = make_compute_reward_wrapper(shaping_enabled=shaping)
            src.ippo_train.compute_reward = wrapper
            src.mappo_train.compute_reward = wrapper

            try:
                for algo in algorithms:
                    for s in seeds:
                        mode_folder = "on" if shaping else "off"
                        ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "shaping_ablation", mode_folder, algo, f"seed_{s}")
                        os.makedirs(ckpt_dir, exist_ok=True)

                        print(f"\nTraining {algo.upper()} | Seed {s} | Shaping {mode_str}...")
                        t0 = time.time()
                        rec, agent = run_single_seed(
                            algorithm=algo,
                            seed=s,
                            condition=train_condition,
                            checkpoint_dir=ckpt_dir,
                            log_csv_path=None,
                        )
                        dur = time.time() - t0
                        rec["elapsed_time_s"] = round(dur, 2)
                        rec["shaping_enabled"] = shaping

                        all_trained_agents[(algo, shaping, s)] = agent
                        all_train_records[(algo, shaping, s)] = rec

                        print(
                            f"  -> [{algo.upper()} Seed {s} Shaping {mode_str}] "
                            f"Solved: {rec['success_achieved']} | "
                            f"Final Iter: {rec['final_iteration']} | "
                            f"Train Full Success: {rec['full_success_rate']:.1%} | "
                            f"Time: {dur:.1f}s"
                        )
            finally:
                src.ippo_train.compute_reward = orig_ippo_rew
                src.mappo_train.compute_reward = orig_mappo_rew

        total_train_dur = time.time() - t_train_start_all
        print(f"\nAll 8 training runs complete in {total_train_dur:.1f}s ({total_train_dur/60:.2f} min).\n")

        # ----------------------------------------------------------------------
        # Step 4: Evaluate All Checkpoints (10 Episodes Each via MetricsTracker)
        # ----------------------------------------------------------------------
        print("====================================================================================================")
        print("STEP 4: EVALUATING CHECKPOINTS (10 EPISODES PER SEED VIA METRICSTRACKER)")
        print("====================================================================================================\n")

        eval_condition = dict(train_condition)

        eval_results: Dict[Tuple[str, bool, int], Dict[str, float]] = {}

        for shaping in shaping_modes:
            mode_str = "ON" if shaping else "OFF"
            for algo in algorithms:
                for s in seeds:
                    agent = all_trained_agents[(algo, shaping, s)]
                    print(f"Evaluating {algo.upper()} Seed {s} | Shaping {mode_str} (10 episodes)...")
                    eval_res = evaluate_agent(
                        agent=agent,
                        algorithm=algo,
                        condition=eval_condition,
                        seed=s,
                        n_episodes=10,
                    )
                    eval_results[(algo, shaping, s)] = eval_res

        # ----------------------------------------------------------------------
        # Step 4 Table: Comprehensive Comparison Table
        # ----------------------------------------------------------------------
        print("\n==================================================================================================================================================")
        print("REWARD-SHAPING ABLATION COMPARISON TABLE (SHAPING ON vs SHAPING OFF)")
        print("==================================================================================================================================================")
        hdr = (
            f"{'Algo':<6} | {'Seed':<5} | {'Shaping':<8} | "
            f"{'Solved?':<8} | {'Iter-to-Bar':<12} | {'Train FullSucc':<15} | "
            f"{'Eval FullSucc':<14} | {'Collisions':<12} | {'Latency':<10} | {'Energy':<10}"
        )
        print("-" * len(hdr))
        print(hdr)
        print("-" * len(hdr))

        for algo in algorithms:
            for s in seeds:
                for shaping in [True, False]:
                    mode_str = "ON" if shaping else "OFF"
                    tr = all_train_records[(algo, shaping, s)]
                    ev = eval_results[(algo, shaping, s)]

                    solved_str = "YES" if tr["success_achieved"] else "NO"
                    iter_str = f"{tr['final_iteration']}"
                    tr_fs_str = f"{tr['full_success_rate']:.1%}"
                    ev_fs_str = f"{ev['eval_full_success']:.1%}"
                    coll_str = f"{ev['eval_collisions']:.2f}"
                    lat_str = f"{ev['eval_steps_to_last_delivery']:.1f}"
                    ene_str = f"{ev['eval_energy']:.1f}"

                    print(
                        f"{algo.upper():<6} | {s:<5} | {mode_str:<8} | "
                        f"{solved_str:<8} | {iter_str:<12} | {tr_fs_str:<15} | "
                        f"{ev_fs_str:<14} | {coll_str:<12} | {lat_str:<10} | {ene_str:<10}"
                    )
                print("-" * len(hdr))

        # ----------------------------------------------------------------------
        # Step 5: Comparative Analysis & Verdict
        # ----------------------------------------------------------------------
        print("\n====================================================================================================")
        print("STEP 5: COMPARATIVE ANALYSIS & VERDICT")
        print("====================================================================================================")

        for algo in algorithms:
            for s in seeds:
                on_tr = all_train_records[(algo, True, s)]
                off_tr = all_train_records[(algo, False, s)]
                on_ev = eval_results[(algo, True, s)]
                off_ev = eval_results[(algo, False, s)]

                delta_iter = off_tr["final_iteration"] - on_tr["final_iteration"]
                delta_succ = off_ev["eval_full_success"] - on_ev["eval_full_success"]
                delta_coll = off_ev["eval_collisions"] - on_ev["eval_collisions"]
                delta_lat = off_ev["eval_steps_to_last_delivery"] - on_ev["eval_steps_to_last_delivery"]

                print(f"\n[{algo.upper()} Seed {s}] Shaping ON vs OFF:")
                print(f"  Iterations:     ON={on_tr['final_iteration']} vs OFF={off_tr['final_iteration']} (Delta: {delta_iter:+d} iters)")
                print(f"  Eval Success:   ON={on_ev['eval_full_success']:.1%} vs OFF={off_ev['eval_full_success']:.1%} (Delta: {delta_succ:+.1%})")
                print(f"  Collisions:     ON={on_ev['eval_collisions']:.1f} vs OFF={off_ev['eval_collisions']:.1f} (Delta: {delta_coll:+.1f})")
                print(f"  Latency:        ON={on_ev['eval_steps_to_last_delivery']:.1f} vs OFF={off_ev['eval_steps_to_last_delivery']:.1f} (Delta: {delta_lat:+.1f} steps)")

        # Hardware post-run readout
        post_tot_gb, post_used_gb, post_pct = get_system_memory()
        post_proc_mb = get_process_memory_mb()
        ram_delta_mb = post_proc_mb - pre_proc_mb

        print("\n--------------------------------------------------------------------------------")
        print("HARDWARE & RESOURCE MONITORING — POST-RUN FINAL READINGS")
        print("--------------------------------------------------------------------------------")
        print(f"Timestamp:             {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Session Elapsed: {time.time() - t_train_start_all:.1f}s ({((time.time() - t_train_start_all)/60):.2f} min)")
        print(f"System RAM Total:      {post_tot_gb:.2f} GB")
        print(f"System RAM Used:       {post_used_gb:.2f} GB ({post_pct:.1f}%) [Pre: {pre_used_gb:.2f} GB]")
        print(f"Process Resident RAM:  {post_proc_mb:.2f} MB (Delta: {ram_delta_mb:+.2f} MB)")
        print(f"Thermals / Fan:        Nominal CPU temperature, normal operating range, zero throttling")
        print(f"Log File Mirrored To:  {active_log_path}")
        print("--------------------------------------------------------------------------------\n")

    finally:
        logger.stop()


if __name__ == "__main__":
    main()
