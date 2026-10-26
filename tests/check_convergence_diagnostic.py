"""
tests/check_convergence_diagnostic.py - S3.4b Convergence Diagnostic.

Investigates whether policies that hit the 80%-moving-average stop bar early
(8-14 iterations) in S3.4a's 50% jamming sweep are genuinely converged,
or were unstable at the evaluation point.

Protocol:
1. Parse granular CSVs for IPPO and MAPPO at 50% jamming condition.
2. Identify the outlier seed whose evaluated success rate deviates most from condition mean.
3. Resume training from original checkpoint for up to 60 additional iterations,
   or until the 20-episode moving average plateaus (< 2 percentage points variation over 20 iters).
4. Save continued checkpoints separately under checkpoints/convergence_check/.
5. Evaluate both original stopped checkpoint and continued checkpoint under identical conditions.
6. Produce side-by-side Before/After table and plain-words assessment.
7. Wrap full execution in TeeLogger from start to finish.
"""

import os
import sys
import time
import csv
import platform
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.tee_logger import TeeLogger
from src.drone_world import DroneWorld
from src.metrics import MetricsTracker
from src.rewards import compute_reward
from src.ppo_core import PPOAgent, compute_gae
from src.mappo_train import MAPPOAgent, flatten_global_state
from src.ippo_train import (
    get_system_memory,
    get_process_memory_mb,
    get_current_distances,
)
from src.multiseed_harness import evaluate_agent


def get_cpu_info_str() -> str:
    """Returns platform CPU info."""
    try:
        return f"{platform.processor()} ({platform.machine()})"
    except Exception:
        return "Unknown CPU"


def load_csv_records(csv_path: str) -> List[Dict[str, Any]]:
    """Loads CSV records into typed dictionaries."""
    records = []
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
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


def identify_outlier_seed(
    records: List[Dict[str, Any]],
    algorithm_name: str,
    target_jamming: float = 0.5,
) -> Tuple[int, Dict[str, Any], float, float]:
    """
    Identifies the outlier seed at target_jamming whose success rate deviates most from condition mean.
    Returns:
        (outlier_seed, outlier_record, full_success_mean, success_frac_mean)
    """
    lvl_records = [
        r for r in records
        if abs(float(r["jammer_intensity"]) - target_jamming) < 1e-4
    ]
    if not lvl_records:
        raise ValueError(f"No records found for {algorithm_name} at jamming={target_jamming}")

    full_successes = [float(r["eval_full_success"]) for r in lvl_records]
    frac_successes = [float(r["eval_success"]) for r in lvl_records]

    mean_full = float(np.mean(full_successes))
    mean_frac = float(np.mean(frac_successes))

    print(f"\n[{algorithm_name.upper()}] 50% Jamming Condition (N = {len(lvl_records)} seeds):")
    print(f"  Condition Mean Eval Full Success Rate:    {mean_full:6.1%}")
    print(f"  Condition Mean Eval Success Fraction:     {mean_frac:6.1%}")
    print(f"  {'Seed':<6} | {'FinalIter':<10} | {'FullSucc':<10} | {'DevFromMean':<12} | {'SuccFrac':<10} | {'DevFromMean':<12}")
    print("  " + "-" * 68)

    max_deviation = -1.0
    outlier_seed = -1
    outlier_rec = None

    for r in lvl_records:
        seed = int(r["seed"])
        fs = float(r["eval_full_success"])
        sf = float(r["eval_success"])
        dev_fs = fs - mean_full
        dev_sf = sf - mean_frac
        abs_dev_fs = abs(dev_fs)

        print(
            f"  {seed:<6d} | {r['final_iteration']:<10d} | {fs:<10.1%} | {dev_fs:<+12.1%} | "
            f"{sf:<10.3f} | {dev_sf:<+12.3f}"
        )

        if abs_dev_fs > max_deviation:
            max_deviation = abs_dev_fs
            outlier_seed = seed
            outlier_rec = r

    print("  " + "-" * 68)
    print(
        f"  --> Identified Outlier: Seed {outlier_seed} "
        f"(Full Success: {outlier_rec['eval_full_success']:.1%} vs Mean {mean_full:.1%}, "
        f"Deviation: {outlier_rec['eval_full_success'] - mean_full:+.1%})\n"
    )

    return outlier_seed, outlier_rec, mean_full, mean_frac


# ==============================================================================
# IPPO Continuation Runner
# ==============================================================================

def run_ippo_continuation(
    checkpoint_load_path: str,
    checkpoint_save_path: str,
    seed: int,
    condition: Dict[str, Any],
    max_additional_iters: int = 60,
    plateau_window_iters: int = 20,
    plateau_threshold: float = 0.02,
) -> Tuple[PPOAgent, List[Dict[str, Any]], str]:
    """
    Resumes IPPO training from checkpoint for up to max_additional_iters,
    stopping early if 20-episode moving average changes by < plateau_threshold over
    plateau_window_iters consecutive iterations.
    """
    print("================================================================================")
    print(f"IPPO CONTINUATION TRAINING: Seed {seed} | Jamming {condition.get('jammer_intensity', 0.5):.0%}")
    print(f"Loading checkpoint: {checkpoint_load_path}")
    print(f"Target save path:   {checkpoint_save_path}")
    print(f"Budget: up to {max_additional_iters} additional iterations | Plateau threshold: {plateau_threshold:.1%}")
    print("================================================================================\n")

    grid_size = condition.get("grid_size", 20)
    n_drones = condition.get("n_drones", 5)
    n_delivery_points = condition.get("n_delivery_points", 3)
    episode_max_steps = condition.get("episode_max_steps", 250)
    rollout_world_steps = condition.get("rollout_world_steps", 410)
    jammer_intensity = condition.get("jammer_intensity", 0.5)
    n_no_fly_zones = condition.get("n_no_fly_zones", 0)
    window_size = condition.get("window_size", 20)

    # Initialize agent
    state_dim = 7
    action_dim = 5
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        epochs_per_rollout=4,
        minibatch_size=64,
    )
    agent.load_checkpoint(checkpoint_load_path)
    print(f"[CHECKPOINT LOADED] Successfully loaded IPPO weights and optimizer state from:\n  {checkpoint_load_path}\n")

    def make_world():
        return DroneWorld(
            grid_size=grid_size,
            n_drones=n_drones,
            n_delivery_points=n_delivery_points,
            seed=seed,
            n_no_fly_zones=n_no_fly_zones,
            jammer_intensity=jammer_intensity,
            max_steps=episode_max_steps,
        )

    world = make_world()
    tracker = MetricsTracker(world=world)

    ep_drone_returns: List[float] = [0.0] * n_drones
    completed_episodes: List[Dict[str, Any]] = []
    continuation_logs: List[Dict[str, Any]] = []
    ma_history: List[float] = []

    stop_reason = f"Completed all {max_additional_iters} additional iterations budget."
    t0 = time.time()

    for add_iter in range(1, max_additional_iters + 1):
        t_iter_start = time.time()

        drone_obs: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
        drone_acts: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
        drone_lps: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
        drone_vals: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
        drone_rews: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
        drone_dones: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]

        for _ in range(rollout_world_steps):
            actions: List[int] = []
            for d in range(n_drones):
                obs_d = world.get_observation(d)
                obs_t = torch.tensor(obs_d, dtype=torch.float32)
                with torch.no_grad():
                    act_t, lp_t, _ = agent.actor.get_action_and_log_prob(obs_t)
                    val_t = agent.critic(obs_t)

                drone_obs[d].append(obs_t)
                drone_acts[d].append(act_t)
                drone_lps[d].append(lp_t)
                drone_vals[d].append(val_t)
                actions.append(int(act_t.item()))

            prev_distances = get_current_distances(world)
            _ = tracker.record_step(actions)

            for d in range(n_drones):
                rew_d = compute_reward(world, prev_distances, d)
                drone_rews[d].append(torch.tensor(rew_d, dtype=torch.float32))
                ep_drone_returns[d] += rew_d

            all_delivered = (len(world.remaining_deliveries) == 0)
            time_up = (world.t >= episode_max_steps)
            episode_done = all_delivered or time_up

            for d in range(n_drones):
                drone_dones[d].append(torch.tensor(1.0 if episode_done else 0.0, dtype=torch.float32))

            if episode_done:
                summary = tracker.summary()
                total_ep_reward = sum(ep_drone_returns)
                completed_episodes.append({
                    "reward": float(total_ep_reward),
                    "success": float(summary["success"]),
                    "is_full_success": bool(summary["success"] >= 1.0),
                    "collisions": int(summary["no_fly_zone_collisions"] + summary["multi_drone_collisions"]),
                    "steps": int(tracker.total_steps),
                })
                world = make_world()
                tracker = MetricsTracker(world=world)
                ep_drone_returns = [0.0] * n_drones

        # GAE Bootstrap
        next_values: List[float] = []
        for d in range(n_drones):
            if episode_done:
                next_values.append(0.0)
            else:
                next_obs = world.get_observation(d)
                next_obs_t = torch.tensor(next_obs, dtype=torch.float32)
                with torch.no_grad():
                    nv = agent.critic(next_obs_t).item()
                next_values.append(nv)

        all_states: List[torch.Tensor] = []
        all_actions: List[torch.Tensor] = []
        all_log_probs: List[torch.Tensor] = []
        all_returns: List[torch.Tensor] = []
        all_advantages: List[torch.Tensor] = []

        for d in range(n_drones):
            b_rews = torch.stack(drone_rews[d])
            b_vals = torch.stack(drone_vals[d])
            b_dones = torch.stack(drone_dones[d])
            adv_d, ret_d = compute_gae(
                b_rews, b_vals, b_dones, next_values[d],
                gamma=agent.gamma, gae_lambda=agent.gae_lambda,
            )
            all_states.append(torch.stack(drone_obs[d]))
            all_actions.append(torch.stack(drone_acts[d]))
            all_log_probs.append(torch.stack(drone_lps[d]))
            all_returns.append(ret_d)
            all_advantages.append(adv_d)

        batch_states = torch.cat(all_states, dim=0)
        batch_actions = torch.cat(all_actions, dim=0)
        batch_log_probs = torch.cat(all_log_probs, dim=0)
        batch_returns = torch.cat(all_returns, dim=0)
        batch_advantages = torch.cat(all_advantages, dim=0)

        p_loss, v_loss, ent = agent.update(
            batch_states, batch_actions, batch_log_probs, batch_returns, batch_advantages
        )

        recent = completed_episodes[-window_size:] if completed_episodes else []
        n_recent = len(recent)
        avg_reward = float(np.mean([e["reward"] for e in recent])) if recent else 0.0
        avg_success = float(np.mean([e["success"] for e in recent])) if recent else 0.0
        avg_collisions = float(np.mean([e["collisions"] for e in recent])) if recent else 0.0
        full_success_rate = (
            float(sum(1 for e in recent if e["is_full_success"]) / float(n_recent))
            if recent else 0.0
        )

        ma_history.append(full_success_rate)

        # Plateau detection: over plateau_window_iters consecutive iterations
        range_str = "N/A"
        delta_str = "N/A"
        is_plateau = False

        if add_iter >= plateau_window_iters and n_recent >= window_size:
            w_ma = ma_history[-plateau_window_iters:]
            ma_range = max(w_ma) - min(w_ma)
            ma_delta = abs(w_ma[-1] - w_ma[0])
            range_str = f"{ma_range:.1%}"
            delta_str = f"{ma_delta:.1%}"

            if ma_range < plateau_threshold:
                is_plateau = True
                stop_reason = (
                    f"Plateau reached at continuation iteration {add_iter}: "
                    f"20-episode moving average changed by {ma_range:.2%} (< {plateau_threshold:.1%}) "
                    f"over {plateau_window_iters} consecutive iterations."
                )

        print(
            f"[IPPO Cont +{add_iter:2d}/60] "
            f"20-ep FullSucc: {full_success_rate:5.1%} | "
            f"AvgSucc: {avg_success:5.2f} | "
            f"Reward: {avg_reward:7.2f} | "
            f"Coll: {avg_collisions:4.1f} | "
            f"20-iter Range: {range_str:<6} | Delta: {delta_str:<6} | "
            f"Eps: {len(completed_episodes):3d} | "
            f"IterTime: {time.time() - t_iter_start:4.2f}s"
        )

        continuation_logs.append({
            "continuation_iteration": add_iter,
            "full_success_rate": full_success_rate,
            "avg_success_rate": avg_success,
            "avg_reward": avg_reward,
            "avg_collisions": avg_collisions,
            "completed_episodes": len(completed_episodes),
        })

        if is_plateau:
            print(f"\n[EARLY STOP TRIGGERED] {stop_reason}\n")
            break

    # Save continued checkpoint
    agent.save_checkpoint(checkpoint_save_path)
    print(f"[CHECKPOINT SAVED] Continued model saved to:\n  {checkpoint_save_path}")
    print(f"[IPPO Continuation Finished] Total elapsed time: {time.time() - t0:.2f}s | Reason: {stop_reason}\n")

    return agent, continuation_logs, stop_reason


# ==============================================================================
# MAPPO Continuation Runner
# ==============================================================================

def run_mappo_continuation(
    checkpoint_load_path: str,
    checkpoint_save_path: str,
    seed: int,
    condition: Dict[str, Any],
    max_additional_iters: int = 60,
    plateau_window_iters: int = 20,
    plateau_threshold: float = 0.02,
) -> Tuple[MAPPOAgent, List[Dict[str, Any]], str]:
    """
    Resumes MAPPO training from checkpoint for up to max_additional_iters,
    stopping early if 20-episode moving average changes by < plateau_threshold over
    plateau_window_iters consecutive iterations.
    """
    print("================================================================================")
    print(f"MAPPO CONTINUATION TRAINING: Seed {seed} | Jamming {condition.get('jammer_intensity', 0.5):.0%}")
    print(f"Loading checkpoint: {checkpoint_load_path}")
    print(f"Target save path:   {checkpoint_save_path}")
    print(f"Budget: up to {max_additional_iters} additional iterations | Plateau threshold: {plateau_threshold:.1%}")
    print("================================================================================\n")

    grid_size = condition.get("grid_size", 20)
    n_drones = condition.get("n_drones", 5)
    n_delivery_points = condition.get("n_delivery_points", 3)
    episode_max_steps = condition.get("episode_max_steps", 250)
    rollout_world_steps = condition.get("rollout_world_steps", 410)
    jammer_intensity = condition.get("jammer_intensity", 0.5)
    n_no_fly_zones = condition.get("n_no_fly_zones", 0)
    window_size = condition.get("window_size", 20)

    # Initialize agent
    local_state_dim = 7
    global_state_dim = 14
    action_dim = 5
    agent = MAPPOAgent(
        local_state_dim=local_state_dim,
        global_state_dim=global_state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        epochs_per_rollout=4,
        minibatch_size=64,
    )
    agent.load_checkpoint(checkpoint_load_path)
    print(f"[CHECKPOINT LOADED] Successfully loaded MAPPO weights and optimizer state from:\n  {checkpoint_load_path}\n")

    def make_world():
        return DroneWorld(
            grid_size=grid_size,
            n_drones=n_drones,
            n_delivery_points=n_delivery_points,
            seed=seed,
            n_no_fly_zones=n_no_fly_zones,
            jammer_intensity=jammer_intensity,
            max_steps=episode_max_steps,
        )

    world = make_world()
    tracker = MetricsTracker(world=world)

    ep_drone_returns: List[float] = [0.0] * n_drones
    completed_episodes: List[Dict[str, Any]] = []
    continuation_logs: List[Dict[str, Any]] = []
    ma_history: List[float] = []

    stop_reason = f"Completed all {max_additional_iters} additional iterations budget."
    t0 = time.time()

    for add_iter in range(1, max_additional_iters + 1):
        t_iter_start = time.time()

        drone_local_obs: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
        drone_global_obs: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
        drone_acts: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
        drone_lps: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
        drone_vals: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
        drone_rews: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
        drone_dones: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]

        for _ in range(rollout_world_steps):
            global_vec = flatten_global_state(world)
            global_t = torch.tensor(global_vec, dtype=torch.float32)
            with torch.no_grad():
                val_global_t = agent.critic(global_t)

            actions: List[int] = []
            for d in range(n_drones):
                obs_d = world.get_observation(d)
                obs_t = torch.tensor(obs_d, dtype=torch.float32)
                with torch.no_grad():
                    act_t, lp_t, _ = agent.actor.get_action_and_log_prob(obs_t)

                drone_local_obs[d].append(obs_t)
                drone_global_obs[d].append(global_t)
                drone_acts[d].append(act_t)
                drone_lps[d].append(lp_t)
                drone_vals[d].append(val_global_t)
                actions.append(int(act_t.item()))

            prev_distances = get_current_distances(world)
            _ = tracker.record_step(actions)

            for d in range(n_drones):
                rew_d = compute_reward(world, prev_distances, d)
                drone_rews[d].append(torch.tensor(rew_d, dtype=torch.float32))
                ep_drone_returns[d] += rew_d

            all_delivered = (len(world.remaining_deliveries) == 0)
            time_up = (world.t >= episode_max_steps)
            episode_done = all_delivered or time_up

            for d in range(n_drones):
                drone_dones[d].append(torch.tensor(1.0 if episode_done else 0.0, dtype=torch.float32))

            if episode_done:
                summary = tracker.summary()
                total_ep_reward = sum(ep_drone_returns)
                completed_episodes.append({
                    "reward": float(total_ep_reward),
                    "success": float(summary["success"]),
                    "is_full_success": bool(summary["success"] >= 1.0),
                    "collisions": int(summary["no_fly_zone_collisions"] + summary["multi_drone_collisions"]),
                    "steps": int(tracker.total_steps),
                })
                world = make_world()
                tracker = MetricsTracker(world=world)
                ep_drone_returns = [0.0] * n_drones

        # MAPPO GAE Bootstrap
        if episode_done:
            next_global_val = 0.0
        else:
            next_global_vec = flatten_global_state(world)
            next_global_t = torch.tensor(next_global_vec, dtype=torch.float32)
            with torch.no_grad():
                next_global_val = agent.critic(next_global_t).item()

        all_local_states: List[torch.Tensor] = []
        all_global_states: List[torch.Tensor] = []
        all_actions: List[torch.Tensor] = []
        all_log_probs: List[torch.Tensor] = []
        all_returns: List[torch.Tensor] = []
        all_advantages: List[torch.Tensor] = []

        for d in range(n_drones):
            b_rews = torch.stack(drone_rews[d])
            b_vals = torch.stack(drone_vals[d])
            b_dones = torch.stack(drone_dones[d])
            adv_d, ret_d = compute_gae(
                b_rews, b_vals, b_dones, next_global_val,
                gamma=agent.gamma, gae_lambda=agent.gae_lambda,
            )
            all_local_states.append(torch.stack(drone_local_obs[d]))
            all_global_states.append(torch.stack(drone_global_obs[d]))
            all_actions.append(torch.stack(drone_acts[d]))
            all_log_probs.append(torch.stack(drone_lps[d]))
            all_returns.append(ret_d)
            all_advantages.append(adv_d)

        batch_local_states = torch.cat(all_local_states, dim=0)
        batch_global_states = torch.cat(all_global_states, dim=0)
        batch_actions = torch.cat(all_actions, dim=0)
        batch_log_probs = torch.cat(all_log_probs, dim=0)
        batch_returns = torch.cat(all_returns, dim=0)
        batch_advantages = torch.cat(all_advantages, dim=0)

        p_loss, v_loss, ent = agent.update(
            batch_local_states, batch_global_states, batch_actions,
            batch_log_probs, batch_returns, batch_advantages,
        )

        recent = completed_episodes[-window_size:] if completed_episodes else []
        n_recent = len(recent)
        avg_reward = float(np.mean([e["reward"] for e in recent])) if recent else 0.0
        avg_success = float(np.mean([e["success"] for e in recent])) if recent else 0.0
        avg_collisions = float(np.mean([e["collisions"] for e in recent])) if recent else 0.0
        full_success_rate = (
            float(sum(1 for e in recent if e["is_full_success"]) / float(n_recent))
            if recent else 0.0
        )

        ma_history.append(full_success_rate)

        # Plateau detection
        range_str = "N/A"
        delta_str = "N/A"
        is_plateau = False

        if add_iter >= plateau_window_iters and n_recent >= window_size:
            w_ma = ma_history[-plateau_window_iters:]
            ma_range = max(w_ma) - min(w_ma)
            ma_delta = abs(w_ma[-1] - w_ma[0])
            range_str = f"{ma_range:.1%}"
            delta_str = f"{ma_delta:.1%}"

            if ma_range < plateau_threshold:
                is_plateau = True
                stop_reason = (
                    f"Plateau reached at continuation iteration {add_iter}: "
                    f"20-episode moving average changed by {ma_range:.2%} (< {plateau_threshold:.1%}) "
                    f"over {plateau_window_iters} consecutive iterations."
                )

        print(
            f"[MAPPO Cont +{add_iter:2d}/60] "
            f"20-ep FullSucc: {full_success_rate:5.1%} | "
            f"AvgSucc: {avg_success:5.2f} | "
            f"Reward: {avg_reward:7.2f} | "
            f"Coll: {avg_collisions:4.1f} | "
            f"20-iter Range: {range_str:<6} | Delta: {delta_str:<6} | "
            f"Eps: {len(completed_episodes):3d} | "
            f"IterTime: {time.time() - t_iter_start:4.2f}s"
        )

        continuation_logs.append({
            "continuation_iteration": add_iter,
            "full_success_rate": full_success_rate,
            "avg_success_rate": avg_success,
            "avg_reward": avg_reward,
            "avg_collisions": avg_collisions,
            "completed_episodes": len(completed_episodes),
        })

        if is_plateau:
            print(f"\n[EARLY STOP TRIGGERED] {stop_reason}\n")
            break

    # Save continued checkpoint
    agent.save_checkpoint(checkpoint_save_path)
    print(f"[CHECKPOINT SAVED] Continued model saved to:\n  {checkpoint_save_path}")
    print(f"[MAPPO Continuation Finished] Total elapsed time: {time.time() - t0:.2f}s | Reason: {stop_reason}\n")

    return agent, continuation_logs, stop_reason


# ==============================================================================
# Main Diagnostic Routine
# ==============================================================================

def main():
    log_dir = os.path.join(PROJECT_ROOT, "logs", "runs")
    logger = TeeLogger(log_dir=log_dir, prefix="s3_4b_convergence_diagnostic")
    active_log_path = logger.start()

    try:
        print("================================================================================")
        print("CONSTELLATION — SESSION S3.4b: CONVERGENCE DIAGNOSTIC")
        print("Investigation: Does the 80% stop bar explain S3.4a's variance at 50% jamming?")
        print(f"Dual-Stream TeeLogger Active: {active_log_path}")
        print("================================================================================\n")

        # Step 0: Hardware Baseline
        tot_ram_gb, used_ram_gb, ram_pct = get_system_memory()
        proc_ram_mb = get_process_memory_mb()
        print("--------------------------------------------------------------------------------")
        print("HARDWARE & RESOURCE READOUT — PRE-RUN")
        print("--------------------------------------------------------------------------------")
        print(f"Timestamp:             {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"CPU:                   {get_cpu_info_str()}")
        print(f"System RAM Total:      {tot_ram_gb:.2f} GB")
        print(f"System RAM Used:       {used_ram_gb:.2f} GB ({ram_pct:.1f}%)")
        print(f"Process RAM Resident:  {proc_ram_mb:.2f} MB")
        print("--------------------------------------------------------------------------------\n")

        # Step 1: Read CSVs and Identify Outliers
        ippo_csv = os.path.join(PROJECT_ROOT, "logs", "ippo_jamming_sweep_5seeds_granular.csv")
        mappo_csv = os.path.join(PROJECT_ROOT, "logs", "mappo_jamming_sweep_5seeds_granular.csv")

        ippo_recs = load_csv_records(ippo_csv)
        mappo_recs = load_csv_records(mappo_csv)

        ippo_seed, ippo_orig_rec, ippo_mean_full, ippo_mean_frac = identify_outlier_seed(
            ippo_recs, "ippo", target_jamming=0.5
        )
        mappo_seed, mappo_orig_rec, mappo_mean_full, mappo_mean_frac = identify_outlier_seed(
            mappo_recs, "mappo", target_jamming=0.5
        )

        condition = {
            "grid_size": 20,
            "n_drones": 5,
            "n_delivery_points": 3,
            "episode_max_steps": 250,
            "rollout_world_steps": 410,
            "jammer_intensity": 0.5,
            "n_no_fly_zones": 0,
            "window_size": 20,
        }

        # Step 2: Continuation Runs
        # Output directory for continuation checkpoints
        conv_ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "convergence_check")
        os.makedirs(conv_ckpt_dir, exist_ok=True)

        ippo_orig_ckpt = os.path.join(
            PROJECT_ROOT, "checkpoints", "multiseed_ippo_sweep", "jam_50", f"seed_{ippo_seed}",
            f"ippo_iteration_{ippo_orig_rec['final_iteration']}_solved.pt"
        )
        ippo_cont_ckpt = os.path.join(
            conv_ckpt_dir, f"ippo_seed_{ippo_seed}_continued.pt"
        )

        mappo_orig_ckpt = os.path.join(
            PROJECT_ROOT, "checkpoints", "multiseed_mappo_sweep", "jam_50", f"seed_{mappo_seed}",
            f"mappo_iteration_{mappo_orig_rec['final_iteration']}_solved.pt"
        )
        mappo_cont_ckpt = os.path.join(
            conv_ckpt_dir, f"mappo_seed_{mappo_seed}_continued.pt"
        )

        # Confirm original checkpoints exist
        assert os.path.exists(ippo_orig_ckpt), f"Missing IPPO checkpoint: {ippo_orig_ckpt}"
        assert os.path.exists(mappo_orig_ckpt), f"Missing MAPPO checkpoint: {mappo_orig_ckpt}"

        print("Original Checkpoints Verified:")
        print(f"  IPPO:  {ippo_orig_ckpt}")
        print(f"  MAPPO: {mappo_orig_ckpt}\n")

        # Execute IPPO Continuation
        ippo_cont_agent, ippo_cont_logs, ippo_stop_reason = run_ippo_continuation(
            checkpoint_load_path=ippo_orig_ckpt,
            checkpoint_save_path=ippo_cont_ckpt,
            seed=ippo_seed,
            condition=condition,
            max_additional_iters=60,
            plateau_window_iters=20,
            plateau_threshold=0.02,
        )

        # Execute MAPPO Continuation
        mappo_cont_agent, mappo_cont_logs, mappo_stop_reason = run_mappo_continuation(
            checkpoint_load_path=mappo_orig_ckpt,
            checkpoint_save_path=mappo_cont_ckpt,
            seed=mappo_seed,
            condition=condition,
            max_additional_iters=60,
            plateau_window_iters=20,
            plateau_threshold=0.02,
        )

        # Step 3: Run evaluate_agent() on Original vs. Continued Policies (10 episodes each)
        print("================================================================================")
        print("EVALUATION PHASE: evaluate_agent() (10 episodes per policy, seed=1, jamming=50%)")
        print("================================================================================\n")

        # Load fresh agent instances for original policies to guarantee zero contamination
        ippo_orig_eval_agent = PPOAgent(
            state_dim=7, action_dim=5, hidden_dim=64, lr=3e-4, gamma=0.99, gae_lambda=0.95
        )
        ippo_orig_eval_agent.load_checkpoint(ippo_orig_ckpt)

        mappo_orig_eval_agent = MAPPOAgent(
            local_state_dim=7, global_state_dim=14, action_dim=5, hidden_dim=64, lr=3e-4, gamma=0.99, gae_lambda=0.95
        )
        mappo_orig_eval_agent.load_checkpoint(mappo_orig_ckpt)

        print("[1/4] Evaluating IPPO Original Stopped Policy (Iter 9)...")
        ippo_orig_eval = evaluate_agent(
            agent=ippo_orig_eval_agent,
            algorithm="ippo",
            condition=condition,
            seed=ippo_seed,
            n_episodes=10,
        )
        print(f"      Eval Full Success: {ippo_orig_eval['eval_full_success']:.1%} | Collisions: {ippo_orig_eval['eval_collisions']:.1f}")

        print("[2/4] Evaluating IPPO Continued Policy...")
        ippo_cont_eval = evaluate_agent(
            agent=ippo_cont_agent,
            algorithm="ippo",
            condition=condition,
            seed=ippo_seed,
            n_episodes=10,
        )
        print(f"      Eval Full Success: {ippo_cont_eval['eval_full_success']:.1%} | Collisions: {ippo_cont_eval['eval_collisions']:.1f}")

        print("[3/4] Evaluating MAPPO Original Stopped Policy (Iter 11)...")
        mappo_orig_eval = evaluate_agent(
            agent=mappo_orig_eval_agent,
            algorithm="mappo",
            condition=condition,
            seed=mappo_seed,
            n_episodes=10,
        )
        print(f"      Eval Full Success: {mappo_orig_eval['eval_full_success']:.1%} | Collisions: {mappo_orig_eval['eval_collisions']:.1f}")

        print("[4/4] Evaluating MAPPO Continued Policy...")
        mappo_cont_eval = evaluate_agent(
            agent=mappo_cont_agent,
            algorithm="mappo",
            condition=condition,
            seed=mappo_seed,
            n_episodes=10,
        )
        print(f"      Eval Full Success: {mappo_cont_eval['eval_full_success']:.1%} | Collisions: {mappo_cont_eval['eval_collisions']:.1f}\n")

        # Step 4: Before / After Table
        ippo_final_iter = int(ippo_orig_rec['final_iteration']) + len(ippo_cont_logs)
        mappo_final_iter = int(mappo_orig_rec['final_iteration']) + len(mappo_cont_logs)

        print("========================================================================================================================")
        print("CONVERGENCE DIAGNOSTIC: BEFORE / AFTER EVALUATION TABLE (50% JAMMING, SEED 1)")
        print("========================================================================================================================")
        header = (
            f"{'Metric':<30} | "
            f"{'IPPO Orig (Iter 9)':<20} | {'IPPO Cont (Iter ' + str(ippo_final_iter) + ')':<20} | {'IPPO Delta':<12} | "
            f"{'MAPPO Orig (Iter 11)':<20} | {'MAPPO Cont (Iter ' + str(mappo_final_iter) + ')':<20} | {'MAPPO Delta':<12}"
        )
        print(header)
        print("-" * len(header))

        metrics_display = [
            ("Full Success Rate (100% deliv)", "eval_full_success", True),
            ("Delivery Fraction (mean / 3)", "eval_success", True),
            ("Collisions (per episode)", "eval_collisions", False),
            ("Latency (steps to last deliv)", "eval_steps_to_last_delivery", False),
            ("Energy (non-STAY actions)", "eval_energy", False),
        ]

        diff_summary = {}

        for label, k, is_pct in metrics_display:
            i_orig = ippo_orig_eval[k]
            i_cont = ippo_cont_eval[k]
            i_diff = i_cont - i_orig

            m_orig = mappo_orig_eval[k]
            m_cont = mappo_cont_eval[k]
            m_diff = m_cont - m_orig

            diff_summary[f"ippo_{k}"] = i_diff
            diff_summary[f"mappo_{k}"] = m_diff

            if is_pct:
                i_orig_s = f"{i_orig:6.1%}"
                i_cont_s = f"{i_cont:6.1%}"
                i_diff_s = f"{i_diff:+6.1%}"
                m_orig_s = f"{m_orig:6.1%}"
                m_cont_s = f"{m_cont:6.1%}"
                m_diff_s = f"{m_diff:+6.1%}"
            else:
                i_orig_s = f"{i_orig:6.1f}"
                i_cont_s = f"{i_cont:6.1f}"
                i_diff_s = f"{i_diff:+6.1f}"
                m_orig_s = f"{m_orig:6.1f}"
                m_cont_s = f"{m_cont:6.1f}"
                m_diff_s = f"{m_diff:+6.1f}"

            row = (
                f"{label:<30} | "
                f"{i_orig_s:<20} | {i_cont_s:<20} | {i_diff_s:<12} | "
                f"{m_orig_s:<20} | {m_cont_s:<20} | {m_diff_s:<12}"
            )
            print(row)

        print("=" * len(header) + "\n")

        # Step 5: Materiality Judgment
        ippo_fs_delta = diff_summary["ippo_eval_full_success"]
        mappo_fs_delta = diff_summary["mappo_eval_full_success"]

        print("--------------------------------------------------------------------------------")
        print("MATERIALITY ASSESSMENT (Threshold: >= 5.0 percentage points change in Full Success)")
        print("--------------------------------------------------------------------------------")
        print(f"IPPO Full Success Delta:  {ippo_fs_delta:+6.1%} -> "
              f"{'MATERIAL CHANGE (Worth Attention)' if abs(ippo_fs_delta) >= 0.05 else 'NOT MATERIAL (Stable within margin)'}")
        print(f"MAPPO Full Success Delta: {mappo_fs_delta:+6.1%} -> "
              f"{'MATERIAL CHANGE (Worth Attention)' if abs(mappo_fs_delta) >= 0.05 else 'NOT MATERIAL (Stable within margin)'}")
        print("--------------------------------------------------------------------------------\n")

        print(f"Log mirrored to persistent file: {active_log_path}")

    finally:
        logger.stop()


if __name__ == "__main__":
    main()
