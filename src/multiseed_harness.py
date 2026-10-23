"""
src/multiseed_harness.py - Multi-seed training and evaluation harness for CONSTELLATION.

Purpose:
- Reusable, algorithm-agnostic runner that executes multi-seed training runs for IPPO or MAPPO
  under a fixed condition without modifying core environment or training files.
- Reuses existing train_ippo() and train_mappo() entry points directly as-is.
- Enforces strict independence across seeds: fresh torch/numpy seeds, fresh environment instantiation,
  fresh network initialization, and fresh optimizer state per seed run.
- Aggregates tracked performance and safety metrics across seeds, calculating per-metric mean and
  sample standard deviation, and outputs the result to a structured CSV.
- Supports evaluation of trained policies using MetricsTracker for episode-level metrics (success,
  collisions, steps_to_last_delivery, energy).
"""

import os
import sys
import time
import csv
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.drone_world import DroneWorld
from src.metrics import MetricsTracker
from src.ippo_train import train_ippo, get_system_memory, get_process_memory_mb
from src.mappo_train import train_mappo, flatten_global_state


# ==============================================================================
# Single-Seed Execution
# ==============================================================================

def run_single_seed(
    algorithm: str,
    seed: int,
    condition: Optional[Dict[str, Any]] = None,
    checkpoint_dir: Optional[str] = None,
    log_csv_path: Optional[str] = None,
) -> Tuple[Dict[str, Any], Any]:
    """
    Executes a single, fully independent training run for the given algorithm and seed.

    Ensures zero state leakage:
    - Calls train_ippo() or train_mappo() which set torch.manual_seed(seed) and np.random.seed(seed).
    - Fresh PPOAgent or MAPPOAgent instantiated inside the called function.
    - Fresh Adam optimizer initialized inside the called function.
    - Fresh DroneWorld environment instantiated with the given seed.

    Args:
        algorithm: 'ippo' or 'mappo'
        seed: Integer random seed
        condition: Dictionary of environment/training hyperparameters
        checkpoint_dir: Directory where model checkpoints are saved
        log_csv_path: Optional per-seed iteration log CSV path (None disables per-seed CSV writing)

    Returns:
        Tuple of (seed_metrics_dict, trained_agent)
    """
    if condition is None:
        condition = {}

    algo = algorithm.lower().strip()
    grid_size = condition.get("grid_size", 20)
    n_drones = condition.get("n_drones", 5)
    n_delivery_points = condition.get("n_delivery_points", 3)
    episode_max_steps = condition.get("episode_max_steps", 250)
    rollout_world_steps = condition.get("rollout_world_steps", 410)
    max_iterations = condition.get("max_iterations", 150)
    target_success_rate = condition.get("target_success_rate", 0.80)
    window_size = condition.get("window_size", 20)
    checkpoint_freq = condition.get("checkpoint_freq", 20)
    time_limit_seconds = condition.get("time_limit_seconds", 3600.0)

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join("checkpoints", f"multiseed_{algo}", f"seed_{seed}")

    t_start = time.time()

    if algo == "ippo":
        logs, agent, final_iter, success = train_ippo(
            seed=seed,
            grid_size=grid_size,
            n_drones=n_drones,
            n_delivery_points=n_delivery_points,
            episode_max_steps=episode_max_steps,
            rollout_world_steps=rollout_world_steps,
            max_iterations=max_iterations,
            target_success_rate=target_success_rate,
            window_size=window_size,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq,
            log_csv_path=log_csv_path,
            time_limit_seconds=time_limit_seconds,
        )
    elif algo == "mappo":
        logs, agent, final_iter, success = train_mappo(
            seed=seed,
            grid_size=grid_size,
            n_drones=n_drones,
            n_delivery_points=n_delivery_points,
            episode_max_steps=episode_max_steps,
            rollout_world_steps=rollout_world_steps,
            max_iterations=max_iterations,
            target_success_rate=target_success_rate,
            window_size=window_size,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq,
            log_csv_path=log_csv_path,
            time_limit_seconds=time_limit_seconds,
        )
    else:
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Must be 'ippo' or 'mappo'.")

    t_elapsed = time.time() - t_start

    # Capture final iteration metrics from training loop
    if logs:
        final_log = logs[-1]
    else:
        final_log = {}

    record: Dict[str, Any] = {
        "algorithm": algo,
        "seed": seed,
        "final_iteration": int(final_iter),
        "success_achieved": bool(success),
        "full_success_rate": float(final_log.get("full_success_rate", 0.0)),
        "avg_success_rate": float(final_log.get("avg_success_rate", 0.0)),
        "avg_reward": float(final_log.get("avg_reward", 0.0)),
        "avg_collisions": float(final_log.get("avg_collisions", 0.0)),
        "policy_loss": float(final_log.get("policy_loss", 0.0)),
        "value_loss": float(final_log.get("value_loss", 0.0)),
        "entropy": float(final_log.get("entropy", 0.0)),
        "completed_episodes": int(final_log.get("completed_episodes", 0)),
        "elapsed_time_s": round(t_elapsed, 2),
    }

    return record, agent


# ==============================================================================
# Evaluation Using MetricsTracker
# ==============================================================================

def evaluate_agent(
    agent: Any,
    algorithm: str,
    condition: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    n_episodes: int = 20,
) -> Dict[str, float]:
    """
    Evaluates a trained agent for n_episodes using MetricsTracker to capture episode-level metrics:
    - success: fraction of delivery points completed
    - collisions: sum of no-fly zone and multi-drone collisions
    - steps_to_last_delivery: latency in steps until all deliveries done (or episode cap)
    - energy: count of non-STAY actions across all drones
    - total_steps: total steps per episode

    Args:
        agent: Trained PPOAgent (IPPO) or MAPPOAgent (MAPPO)
        algorithm: 'ippo' or 'mappo'
        condition: Environment condition dict
        seed: Evaluation random seed base
        n_episodes: Number of evaluation episodes

    Returns:
        Dictionary of mean evaluation metrics across n_episodes.
    """
    if condition is None:
        condition = {}

    # Enforce deterministic evaluation across PyTorch action sampling and environment
    torch.manual_seed(seed)
    np.random.seed(seed)

    algo = algorithm.lower().strip()
    grid_size = condition.get("grid_size", 20)
    n_drones = condition.get("n_drones", 5)
    n_delivery_points = condition.get("n_delivery_points", 3)
    episode_max_steps = condition.get("episode_max_steps", 250)
    jammer_intensity = condition.get("jammer_intensity", 0.0)
    n_no_fly_zones = condition.get("n_no_fly_zones", 0)

    ep_successes: List[float] = []
    ep_full_successes: List[float] = []
    ep_collisions: List[float] = []
    ep_latencies: List[float] = []
    ep_energies: List[float] = []
    ep_steps: List[float] = []

    for ep_idx in range(n_episodes):
        ep_seed = seed * 1000 + ep_idx
        world = DroneWorld(
            grid_size=grid_size,
            n_drones=n_drones,
            n_delivery_points=n_delivery_points,
            seed=ep_seed,
            n_no_fly_zones=n_no_fly_zones,
            jammer_intensity=jammer_intensity,
            max_steps=episode_max_steps,
        )
        tracker = MetricsTracker(world=world)

        while tracker.total_steps < episode_max_steps:
            if len(world.remaining_deliveries) == 0:
                break

            actions: List[int] = []
            for d in range(n_drones):
                obs_d = world.get_observation(d)
                obs_t = torch.tensor(obs_d, dtype=torch.float32)
                with torch.no_grad():
                    act_t, _, _ = agent.actor.get_action_and_log_prob(obs_t)
                actions.append(int(act_t.item()))

            _ = tracker.record_step(actions)

        summary = tracker.summary()
        ep_successes.append(float(summary["success"]))
        ep_full_successes.append(1.0 if summary["success"] >= 1.0 else 0.0)
        ep_collisions.append(float(summary["no_fly_zone_collisions"] + summary["multi_drone_collisions"]))
        ep_latencies.append(float(summary["steps_to_last_delivery"]))
        ep_energies.append(float(summary["energy"]))
        ep_steps.append(float(summary["total_steps"]))

    return {
        "eval_success": float(np.mean(ep_successes)),
        "eval_full_success": float(np.mean(ep_full_successes)),
        "eval_collisions": float(np.mean(ep_collisions)),
        "eval_steps_to_last_delivery": float(np.mean(ep_latencies)),
        "eval_energy": float(np.mean(ep_energies)),
        "eval_total_steps": float(np.mean(ep_steps)),
    }


# ==============================================================================
# Metric Aggregation (Mean & Std Dev)
# ==============================================================================

def aggregate_metrics(
    records: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Computes mean and sample standard deviation (ddof=1) for all numeric fields.

    Args:
        records: List of per-seed metric dictionaries

    Returns:
        Tuple of (mean_record, std_record)
    """
    if not records:
        return {}, {}

    mean_rec: Dict[str, Any] = {"algorithm": records[0].get("algorithm", ""), "seed": "mean"}
    std_rec: Dict[str, Any] = {"algorithm": records[0].get("algorithm", ""), "seed": "std"}

    n = len(records)
    # Collect keys present in records
    keys = [k for k in records[0].keys() if k not in ("algorithm", "seed")]

    for k in keys:
        vals = [r[k] for r in records if k in r and isinstance(r[k], (int, float, bool))]
        if vals:
            mean_val = float(np.mean(vals))
            # Sample standard deviation (ddof=1) if n > 1, else 0.0
            std_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            mean_rec[k] = round(mean_val, 4)
            std_rec[k] = round(std_val, 4)
        else:
            mean_rec[k] = ""
            std_rec[k] = ""

    return mean_rec, std_rec


# ==============================================================================
# CSV Writer
# ==============================================================================

def write_multiseed_csv(
    records: List[Dict[str, Any]],
    mean_record: Dict[str, Any],
    std_record: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Writes per-seed rows, mean row, and standard deviation row to a clean CSV file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if not records:
        return

    # Derive column headers preserving clean order
    preferred_order = [
        "algorithm",
        "seed",
        "final_iteration",
        "success_achieved",
        "full_success_rate",
        "avg_success_rate",
        "avg_reward",
        "avg_collisions",
        "eval_success",
        "eval_full_success",
        "eval_collisions",
        "eval_steps_to_last_delivery",
        "eval_energy",
        "eval_total_steps",
        "policy_loss",
        "value_loss",
        "entropy",
        "completed_episodes",
        "elapsed_time_s",
    ]

    available_keys = list(records[0].keys())
    fieldnames = [k for k in preferred_order if k in available_keys]
    for k in available_keys:
        if k not in fieldnames:
            fieldnames.append(k)

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
        writer.writerow({k: mean_record.get(k, "") for k in fieldnames})
        writer.writerow({k: std_record.get(k, "") for k in fieldnames})


# ==============================================================================
# Multi-Seed Experiment Runner
# ==============================================================================

def run_multiseed_experiment(
    algorithm: str,
    condition: Optional[Dict[str, Any]] = None,
    seeds: Optional[List[int]] = None,
    output_csv_path: Optional[str] = "logs/multiseed_validation.csv",
    eval_episodes: int = 0,
    checkpoint_base_dir: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    Main harness execution entry point.

    Runs training across N seeds, aggregates results, and saves CSV.

    Args:
        algorithm: 'ippo' or 'mappo'
        condition: Dictionary specifying fixed condition hyperparameters
        seeds: List of integer seeds (default: [0, 1, 2])
        output_csv_path: Target path for aggregated validation CSV
        eval_episodes: Number of evaluation episodes with MetricsTracker per seed (0 to skip)
        checkpoint_base_dir: Optional base directory for model checkpoints

    Returns:
        Tuple of (per_seed_records, mean_record, std_record)
    """
    if seeds is None:
        seeds = [0, 1, 2]
    if condition is None:
        condition = {
            "grid_size": 20,
            "n_drones": 5,
            "n_delivery_points": 3,
            "episode_max_steps": 250,
            "jammer_intensity": 0.0,
            "n_no_fly_zones": 0,
            "target_success_rate": 0.80,
            "window_size": 20,
        }

    algo = algorithm.lower().strip()
    print("================================================================================")
    print(f"CONSTELLATION — Multi-Seed Harness: Algorithm = {algo.upper()}")
    print("================================================================================")
    print(f"Seeds to execute:   {seeds} (N = {len(seeds)})")
    print(f"Condition:          Grid {condition.get('grid_size', 20)}x{condition.get('grid_size', 20)} | "
          f"Drones: {condition.get('n_drones', 5)} | Deliveries: {condition.get('n_delivery_points', 3)}")
    print(f"Jammer:             {condition.get('jammer_intensity', 0.0):.0%} intensity | "
          f"No-Fly Zones: {condition.get('n_no_fly_zones', 0)}")
    print(f"Early Stop Target:  {condition.get('target_success_rate', 0.80):.0%} full delivery over last {condition.get('window_size', 20)} episodes")
    print(f"Output CSV Path:    {output_csv_path}")
    print(f"Eval Episodes/Seed: {eval_episodes}")
    print("================================================================================\n")

    seed_records: List[Dict[str, Any]] = []

    for seed in seeds:
        print(f"\n>>>>>>>>>>>>>>> Running Seed {seed} ({algo.upper()}) <<<<<<<<<<<<<<<")
        ckpt_dir = (
            os.path.join(checkpoint_base_dir, f"seed_{seed}")
            if checkpoint_base_dir
            else os.path.join("checkpoints", f"multiseed_{algo}", f"seed_{seed}")
        )

        record, agent = run_single_seed(
            algorithm=algo,
            seed=seed,
            condition=condition,
            checkpoint_dir=ckpt_dir,
            log_csv_path=None,  # Protect baseline CSV files
        )

        if eval_episodes > 0:
            print(f"  -> Running post-training evaluation ({eval_episodes} episodes) with MetricsTracker...")
            eval_metrics = evaluate_agent(
                agent=agent,
                algorithm=algo,
                condition=condition,
                seed=seed,
                n_episodes=eval_episodes,
            )
            record.update(eval_metrics)
            print(f"     Eval Success: {eval_metrics['eval_success']:.1%} | "
                  f"Latency: {eval_metrics['eval_steps_to_last_delivery']:.1f} steps | "
                  f"Collisions: {eval_metrics['eval_collisions']:.1f} | "
                  f"Energy: {eval_metrics['eval_energy']:.1f}")

        seed_records.append(record)

    # Compute mean and standard deviation across seeds
    mean_rec, std_rec = aggregate_metrics(seed_records)

    # Write results to CSV
    if output_csv_path:
        write_multiseed_csv(seed_records, mean_rec, std_rec, output_csv_path)
        print(f"\n[HARNESS] Multi-seed validation CSV successfully written to: {output_csv_path}")

    # Display summary table
    print("\n================================================================================")
    print(f"MULTI-SEED SUMMARY REPORT ({algo.upper()})")
    print("================================================================================")
    print(f"{'Seed':<8} | {'Iter':<6} | {'FullSucc':<10} | {'AvgSucc':<9} | {'AvgReward':<11} | {'AvgColl':<9} | {'Time(s)':<8}")
    print("-" * 72)
    for r in seed_records:
        print(
            f"{str(r['seed']):<8} | "
            f"{r['final_iteration']:<6d} | "
            f"{r['full_success_rate']:<10.1%} | "
            f"{r['avg_success_rate']:<9.3f} | "
            f"{r['avg_reward']:<11.2f} | "
            f"{r['avg_collisions']:<9.2f} | "
            f"{r['elapsed_time_s']:<8.2f}"
        )
    print("-" * 72)
    print(
        f"{str(mean_rec.get('seed')):<8} | "
        f"{mean_rec.get('final_iteration', 0):<6.1f} | "
        f"{mean_rec.get('full_success_rate', 0):<10.1%} | "
        f"{mean_rec.get('avg_success_rate', 0):<9.3f} | "
        f"{mean_rec.get('avg_reward', 0):<11.2f} | "
        f"{mean_rec.get('avg_collisions', 0):<9.2f} | "
        f"{mean_rec.get('elapsed_time_s', 0):<8.2f}"
    )
    print(
        f"{str(std_rec.get('seed')):<8} | "
        f"{std_rec.get('final_iteration', 0):<6.2f} | "
        f"{std_rec.get('full_success_rate', 0):<10.3f} | "
        f"{std_rec.get('avg_success_rate', 0):<9.3f} | "
        f"{std_rec.get('avg_reward', 0):<11.2f} | "
        f"{std_rec.get('avg_collisions', 0):<9.2f} | "
        f"{std_rec.get('elapsed_time_s', 0):<8.2f}"
    )
    print("================================================================================\n")

    return seed_records, mean_rec, std_rec


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CONSTELLATION Multi-Seed Runner")
    parser.add_argument("--algorithm", type=str, default="ippo", choices=["ippo", "mappo"], help="RL algorithm")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds to evaluate")
    parser.add_argument("--output", type=str, default="logs/multiseed_validation.csv", help="Output CSV path")
    parser.add_argument("--eval_episodes", type=int, default=0, help="Number of post-training evaluation episodes")
    args = parser.parse_args()

    run_multiseed_experiment(
        algorithm=args.algorithm,
        seeds=args.seeds,
        output_csv_path=args.output,
        eval_episodes=args.eval_episodes,
    )
