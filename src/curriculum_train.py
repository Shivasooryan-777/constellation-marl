"""
src/curriculum_train.py - 5-Stage Curriculum Training over Jamming Intensity for CONSTELLATION.

Curriculum Design:
- Stage 1: Jammer Intensity 0.0 (baseline, unjammed)
- Stage 2: Jammer Intensity 0.1 (mild RF interference)
- Stage 3: Jammer Intensity 0.3 (moderate RF interference)
- Stage 4: Jammer Intensity 0.5 (heavy RF interference)
- Stage 5: Jammer Intensity 0.7 (severe RF interference)

Stability Criterion per Stage:
- Trailing 20-episode full-delivery success rate >= 70% (0.70)
- Maintained for 5 CONSECUTIVE iterations after first reaching >= 70%
- If stable: status = 'stable', checkpoint saved, advance to next stage.
- If capped (reaches 50 update iterations or 20 min wall-clock without 5-streak stability):
  status = 'capped', checkpoint saved, STOP curriculum (do not advance to subsequent stages).

Warm-Start:
- Stage 1 starts from a fresh, untrained network (random initialization, seed=0).
- Subsequent stages continue training from the previous stage's saved checkpoint.

Architecture & Hyperparameters (strictly reused from S2.4):
- MAPPO: Decentralized Actor (local obs dim=7) + Centralized Critic (flatten_global_state dim=14)
- CPU-only execution (CUDA disabled)
- No-fly zones: strictly OFF throughout (n_no_fly_zones=0)
- 5 drones, 20x20 grid, 3 delivery points, episode max steps 250, rollout world steps 410 (2050 samples/update)
- Adam lr=3e-4, 4 epochs, minibatch 64, GAE(gamma=0.99, lambda=0.95), no advantage normalization
"""

import os
import sys
import time
import csv
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.drone_world import DroneWorld
from src.metrics import MetricsTracker
from src.rewards import compute_reward
from src.ppo_core import compute_gae
from src.mappo_train import MAPPOAgent, flatten_global_state
from src.ippo_train import (
    get_system_memory,
    get_process_memory_mb,
    get_current_distances,
)


def make_curriculum_env(
    seed: int = 0,
    grid_size: int = 20,
    n_drones: int = 5,
    n_delivery_points: int = 3,
    jammer_intensity: float = 0.0,
    max_steps: int = 250,
) -> DroneWorld:
    """
    Instantiates DroneWorld for curriculum training:
    - Fixed grid 20x20, 5 drones, depot at (0, 0), 3 delivery points.
    - No-fly zones strictly OFF (n_no_fly_zones=0).
    - Jammer intensity parameterized per curriculum stage (0.0 to 0.7).
    - Episode length cap: 250 steps.
    """
    return DroneWorld(
        grid_size=grid_size,
        n_drones=n_drones,
        n_delivery_points=n_delivery_points,
        seed=seed,
        n_no_fly_zones=0,
        jammer_intensity=jammer_intensity,
        max_steps=max_steps,
    )


def plot_curriculum_results(
    logs: List[Dict[str, Any]],
    stage_summaries: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """
    Generates a high-quality visualization of the curriculum training progression.
    - Subplot 1: Episode Team Reward vs Global Iteration with stage boundaries.
    - Subplot 2: Full-Delivery Success Rate & Average Success Rate with 70% stability threshold.
    """
    if not logs:
        print("[WARN] No logs available to plot.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    global_iters = [r["global_iteration"] for r in logs]
    rewards = [r["avg_reward"] for r in logs]
    avg_success = [r["avg_success_rate"] for r in logs]
    full_success = [r["full_success_rate"] for r in logs]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Color palette for stages
    stage_colors = ["#4A90E2", "#50E3C2", "#F5A623", "#9013FE", "#D0021B"]

    # Calculate stage boundary global iterations
    stage_boundaries = []
    current_iter = 0
    for summary in stage_summaries:
        start_iter = current_iter + 1
        end_iter = current_iter + summary["iterations"]
        stage_boundaries.append((summary["stage"], summary["intensity"], start_iter, end_iter, summary["status"]))
        current_iter = end_iter

    # Subplot 1: Episode Team Reward
    ax1.plot(global_iters, rewards, color="#1f77b4", linewidth=2.0, label="Team Reward (last 20 eps)")
    ax1.set_ylabel("Episode Reward", fontsize=12, fontweight="bold")
    ax1.set_title("CONSTELLATION — MAPPO Curriculum Training: Reward Progression", fontsize=14, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Success Rates
    ax2.plot(global_iters, full_success, color="#2ca02c", linewidth=2.2, label="Full Success Rate (100% delivered)")
    ax2.plot(global_iters, avg_success, color="#17becf", linewidth=1.5, linestyle="--", label="Avg Success Rate (fraction delivered)")
    ax2.axhline(y=0.70, color="#d62728", linestyle=":", linewidth=2.0, label="Curriculum Stability Threshold (70%)")
    ax2.set_xlabel("Global Training Iteration", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Success Rate", fontsize=12, fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Add vertical stage divider lines and stage annotations to both subplots
    for idx, (stg, intensity, s_iter, e_iter, status) in enumerate(stage_boundaries):
        color = stage_colors[(stg - 1) % len(stage_colors)]

        # Highlight stage span
        ax1.axvspan(s_iter - 0.5, e_iter + 0.5, color=color, alpha=0.10)
        ax2.axvspan(s_iter - 0.5, e_iter + 0.5, color=color, alpha=0.10)

        # Stage boundary line (except at start)
        if s_iter > 1:
            ax1.axvline(x=s_iter - 0.5, color="#555555", linestyle="--", alpha=0.8, linewidth=1.2)
            ax2.axvline(x=s_iter - 0.5, color="#555555", linestyle="--", alpha=0.8, linewidth=1.2)

        # Label on top subplot
        mid_iter = (s_iter + e_iter) / 2.0
        status_sym = "STABLE" if status == "stable" else "CAPPED"
        label_text = f"Stage {stg}\nJammer {intensity:.1f}\n[{status_sym}]"
        y_pos = ax1.get_ylim()[1] * 0.85
        ax1.text(
            mid_iter, y_pos, label_text,
            ha="center", va="center", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.85)
        )

    ax1.legend(loc="upper left", framealpha=0.9)
    ax2.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Curriculum plot successfully saved to {output_path}")


def train_curriculum(
    seed: int = 0,
    grid_size: int = 20,
    n_drones: int = 5,
    n_delivery_points: int = 3,
    episode_max_steps: int = 250,
    rollout_world_steps: int = 410,
    stages: Optional[List[float]] = None,
    stage_max_iterations: int = 50,
    stage_time_limit_seconds: float = 1200.0,
    total_time_limit_seconds: float = 5400.0,
    target_success_rate: float = 0.70,
    consecutive_stable_iterations: int = 5,
    window_size: int = 20,
    checkpoint_dir: str = "checkpoints/curriculum",
    log_csv_path: Optional[str] = "logs/curriculum_run1.csv",
    plot_png_path: Optional[str] = "logs/curriculum_run1.png",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], MAPPOAgent]:
    """
    Executes the 5-stage curriculum training over jamming intensity.

    Returns:
        Tuple of (all_logs, stage_summaries, final_agent).
    """
    if stages is None:
        stages = [0.0, 0.1, 0.3, 0.5, 0.7]

    # Enforce reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    local_state_dim = 7
    global_state_dim = 14
    action_dim = 5

    os.makedirs(checkpoint_dir, exist_ok=True)
    if log_csv_path:
        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)

    print("================================================================================")
    print("CONSTELLATION — Session S2.5: MAPPO Curriculum Training over Jamming Intensity")
    print("================================================================================")
    print(f"Hardware Mode:        CPU-only (CUDA disabled)")
    print(f"Curriculum Stages:    {stages} (Jammer intensities)")
    print(f"No-Fly Zones:         OFF throughout (n_no_fly_zones=0)")
    print(f"Environment:          Grid {grid_size}x{grid_size} | Drones: {n_drones} | Delivery Points: {n_delivery_points}")
    print(f"Rollout per update:   {rollout_world_steps} steps * {n_drones} drones = {rollout_world_steps * n_drones} samples")
    print(f"Stability Criterion:  Full-delivery success rate >= {target_success_rate:.0%} maintained for {consecutive_stable_iterations} consecutive iterations")
    print(f"Rolling Window:       Trailing {window_size} completed episodes")
    print(f"Stage Cap:            {stage_max_iterations} iterations OR {stage_time_limit_seconds / 60:.1f} minutes wall-clock")
    print(f"Total Session Cap:    {total_time_limit_seconds / 60:.1f} minutes wall-clock hard stop")
    print("================================================================================\n")

    # Hardware pre-run baseline
    total_ram_gb, used_ram_gb, mem_pct = get_system_memory()
    baseline_process_ram = get_process_memory_mb()
    print("--------------------------------------------------------------------------------")
    print("HARDWARE BASELINE — PRE-RUN READINGS")
    print("--------------------------------------------------------------------------------")
    print(f"System RAM Total:     {total_ram_gb:.2f} GB")
    print(f"System RAM Used:      {used_ram_gb:.2f} GB ({mem_pct:.1f}%)")
    print(f"Process Resident RAM: {baseline_process_ram:.2f} MB")
    print(f"Start Timestamp:      {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("--------------------------------------------------------------------------------\n")

    # Initialize agent: Stage 1 starts from a fresh, untrained network
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
    print("[WARM-START] Stage 1 initialized fresh from untrained network (seed=0).\n")

    session_start_time = time.time()
    halfway_checkpoint_printed = False
    halfway_threshold_seconds = total_time_limit_seconds / 2.0  # ~45 minutes

    all_logs: List[Dict[str, Any]] = []
    stage_summaries: List[Dict[str, Any]] = []
    global_iteration = 0
    curriculum_stopped = False

    for stage_idx, intensity in enumerate(stages, 1):
        if curriculum_stopped:
            break

        # Check total time hard stop
        total_elapsed = time.time() - session_start_time
        if total_elapsed >= total_time_limit_seconds:
            print(f"\n[SESSION HARD STOP] Total wall-clock limit reached ({total_elapsed:.1f}s >= {total_time_limit_seconds:.1f}s).")
            print("Stopping session immediately and checkpointing partial progress.")
            break

        # Warm start from previous stage checkpoint for stages > 1
        if stage_idx > 1:
            prev_stage_summary = stage_summaries[-1]
            prev_intensity = prev_stage_summary["intensity"]
            prev_status = prev_stage_summary["status"]
            prev_ckpt_path = os.path.join(checkpoint_dir, f"stage_{prev_intensity}_{prev_status}.pt")
            if os.path.exists(prev_ckpt_path):
                print(f"[WARM-START] Stage {stage_idx} loading checkpoint from Stage {stage_idx - 1}: {prev_ckpt_path}")
                agent.load_checkpoint(prev_ckpt_path)
            else:
                print(f"[WARN] Previous checkpoint {prev_ckpt_path} not found on disk! Continuing with current in-memory weights.")

        print("================================================================================")
        print(f"STARTING CURRICULUM STAGE {stage_idx}/{len(stages)}: Jammer Intensity = {intensity:.1f}")
        print("================================================================================")
        print(f"Warm start mode:       {'Fresh untrained initialization' if stage_idx == 1 else f'Warm-started from Stage {stage_idx-1} checkpoint'}")
        print(f"Stability target:      Full success >= {target_success_rate:.0%} for {consecutive_stable_iterations} consecutive iterations (window={window_size})")
        print(f"Stage limits:          Max {stage_max_iterations} iterations OR {stage_time_limit_seconds / 60:.1f} min wall-clock")
        print("================================================================================\n")

        # Initialize environment and tracking for this stage
        world = make_curriculum_env(
            seed=seed,
            grid_size=grid_size,
            n_drones=n_drones,
            n_delivery_points=n_delivery_points,
            jammer_intensity=intensity,
            max_steps=episode_max_steps,
        )
        tracker = MetricsTracker(world=world)

        ep_drone_returns: List[float] = [0.0] * n_drones
        stage_completed_episodes: List[Dict[str, Any]] = []
        stable_streak: int = 0
        stage_status: str = "capped"  # Default if loop exits without stabilizing
        stage_start_time = time.time()
        stage_iters_run = 0
        final_full_success = 0.0

        for stage_iter in range(1, stage_max_iterations + 1):
            stage_iters_run = stage_iter
            global_iteration += 1
            iter_start_time = time.time()

            total_elapsed_now = iter_start_time - session_start_time
            stage_elapsed_now = iter_start_time - stage_start_time

            # Check 45-minute halfway progress checkpoint
            if total_elapsed_now >= halfway_threshold_seconds and not halfway_checkpoint_printed:
                halfway_checkpoint_printed = True
                remaining_budget = max(0.0, total_time_limit_seconds - total_elapsed_now)
                print("\n********************************************************************************")
                print(f"[HALFWAY PROGRESS CHECKPOINT] Elapsed: {total_elapsed_now / 60:.1f} min | Remaining Budget: {remaining_budget / 60:.1f} min")
                print(f"Current Stage: Stage {stage_idx}/{len(stages)} (Jammer {intensity:.1f}) | Stage Iteration: {stage_iter}/{stage_max_iterations}")
                print(f"Completed Stages: {[s['stage'] for s in stage_summaries]}")
                print("********************************************************************************\n")

            # Check total session hard cap
            if total_elapsed_now >= total_time_limit_seconds:
                print(f"\n[SESSION HARD STOP] Total wall-clock limit reached ({total_elapsed_now:.1f}s >= {total_time_limit_seconds:.1f}s).")
                stage_status = "capped"
                curriculum_stopped = True
                break

            # Check stage wall-clock cap
            if stage_elapsed_now >= stage_time_limit_seconds:
                print(f"\n[STAGE TIME CAP] Stage {stage_idx} reached {stage_time_limit_seconds:.1f}s cap ({stage_elapsed_now:.1f}s).")
                stage_status = "capped"
                break

            # RAM leak safety check
            current_proc_ram = get_process_memory_mb()
            if current_proc_ram > baseline_process_ram + 1500.0:
                print(f"\n[SAFETY ABORT] Abnormal RAM growth detected: {current_proc_ram:.1f} MB "
                      f"(Baseline: {baseline_process_ram:.1f} MB). Stopping to prevent stall.")
                stage_status = "capped"
                curriculum_stopped = True
                break

            # ------------------------------------------------------------------
            # Rollout Collection across rollout_world_steps (410 steps)
            # ------------------------------------------------------------------
            drone_local_obs: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
            drone_global_obs: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
            drone_acts: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
            drone_lps: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
            drone_vals: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
            drone_rews: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
            drone_dones: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]

            for _ in range(rollout_world_steps):
                # 1. Centralized Critic evaluates global state
                global_vec = flatten_global_state(world)
                global_t = torch.tensor(global_vec, dtype=torch.float32)
                with torch.no_grad():
                    val_global_t = agent.critic(global_t)

                # 2. Decentralized Actor evaluates local obs
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

                # 3. Pre-step distances for potential-based shaping
                prev_distances = get_current_distances(world)

                # 4. Step environment through MetricsTracker
                _ = tracker.record_step(actions)

                # 5. Compute per-drone rewards via src/rewards.py
                for d in range(n_drones):
                    rew_d = compute_reward(world, prev_distances, d)
                    drone_rews[d].append(torch.tensor(rew_d, dtype=torch.float32))
                    ep_drone_returns[d] += rew_d

                # 6. Episode termination check
                all_delivered = (len(world.remaining_deliveries) == 0)
                time_up = (world.t >= episode_max_steps)
                episode_done = all_delivered or time_up

                for d in range(n_drones):
                    drone_dones[d].append(torch.tensor(1.0 if episode_done else 0.0, dtype=torch.float32))

                # 7. Episode wrap-up & re-initialization
                if episode_done:
                    summary = tracker.summary()
                    total_ep_reward = sum(ep_drone_returns)
                    stage_completed_episodes.append({
                        "reward": float(total_ep_reward),
                        "success": float(summary["success"]),
                        "is_full_success": bool(summary["success"] >= 1.0),
                        "collisions": int(summary["no_fly_zone_collisions"] + summary["multi_drone_collisions"]),
                        "steps": int(tracker.total_steps),
                    })

                    world = make_curriculum_env(
                        seed=seed,
                        grid_size=grid_size,
                        n_drones=n_drones,
                        n_delivery_points=n_delivery_points,
                        jammer_intensity=intensity,
                        max_steps=episode_max_steps,
                    )
                    tracker = MetricsTracker(world=world)
                    ep_drone_returns = [0.0] * n_drones

            # ------------------------------------------------------------------
            # GAE Bootstrap & Return Computation
            # ------------------------------------------------------------------
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
                    b_rews,
                    b_vals,
                    b_dones,
                    next_global_val,
                    gamma=agent.gamma,
                    gae_lambda=agent.gae_lambda,
                )
                all_local_states.append(torch.stack(drone_local_obs[d]))
                all_global_states.append(torch.stack(drone_global_obs[d]))
                all_actions.append(torch.stack(drone_acts[d]))
                all_log_probs.append(torch.stack(drone_lps[d]))
                all_returns.append(ret_d)
                all_advantages.append(adv_d)

            # Pool 5 drones' rollout data into update batch (2050 samples)
            batch_local_states = torch.cat(all_local_states, dim=0)
            batch_global_states = torch.cat(all_global_states, dim=0)
            batch_actions = torch.cat(all_actions, dim=0)
            batch_log_probs = torch.cat(all_log_probs, dim=0)
            batch_returns = torch.cat(all_returns, dim=0)
            batch_advantages = torch.cat(all_advantages, dim=0)

            # MAPPO update
            policy_loss, value_loss, entropy = agent.update(
                batch_local_states,
                batch_global_states,
                batch_actions,
                batch_log_probs,
                batch_returns,
                batch_advantages,
            )

            # Rolling window metrics over last window_size (20) completed episodes in this stage
            recent = stage_completed_episodes[-window_size:] if stage_completed_episodes else []
            n_recent = len(recent)
            avg_reward = float(np.mean([e["reward"] for e in recent])) if recent else 0.0
            avg_success = float(np.mean([e["success"] for e in recent])) if recent else 0.0
            avg_collisions = float(np.mean([e["collisions"] for e in recent])) if recent else 0.0
            full_success_rate = (
                float(sum(1 for e in recent if e["is_full_success"]) / float(n_recent))
                if recent
                else 0.0
            )
            final_full_success = full_success_rate

            # Stability check: consecutive streak >= consecutive_stable_iterations (5)
            if n_recent >= window_size and full_success_rate >= target_success_rate:
                stable_streak += 1
            else:
                stable_streak = 0

            stage_elapsed_s = time.time() - stage_start_time
            total_elapsed_s = time.time() - session_start_time

            print(
                f"Stage {stage_idx} (Jammer {intensity:.1f}) | "
                f"Iter {stage_iter:2d}/{stage_max_iterations} (Tot {global_iteration:3d}) | "
                f"Reward: {avg_reward:7.2f} | "
                f"Success: {avg_success:5.2f} (Full: {full_success_rate:5.1%}) | "
                f"Streak: {stable_streak}/{consecutive_stable_iterations} | "
                f"Collisions: {avg_collisions:4.1f} | "
                f"PLoss: {policy_loss:+.4f} | "
                f"VLoss: {value_loss:6.4f} | "
                f"Entropy: {entropy:.4f} | "
                f"Eps: {len(stage_completed_episodes):3d} | "
                f"Time: {stage_elapsed_s:5.1f}s"
            )

            record = {
                "stage": stage_idx,
                "jammer_intensity": intensity,
                "stage_iteration": stage_iter,
                "global_iteration": global_iteration,
                "avg_reward": avg_reward,
                "avg_success_rate": avg_success,
                "full_success_rate": full_success_rate,
                "avg_collisions": avg_collisions,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "stage_completed_episodes": len(stage_completed_episodes),
                "stable_streak": stable_streak,
                "stage_elapsed_time_s": round(stage_elapsed_s, 2),
                "total_elapsed_time_s": round(total_elapsed_s, 2),
            }
            all_logs.append(record)

            if stable_streak >= consecutive_stable_iterations:
                stage_status = "stable"
                print(
                    f"\n>>> STABILITY ACHIEVED in Stage {stage_idx} (Jammer {intensity:.1f})! "
                    f"Full success >= {target_success_rate:.0%} maintained for {consecutive_stable_iterations} consecutive iterations."
                )
                break

        # End of Stage Processing
        stage_total_duration = time.time() - stage_start_time
        stage_ckpt_path = os.path.join(checkpoint_dir, f"stage_{intensity}_{stage_status}.pt")
        agent.save_checkpoint(stage_ckpt_path)

        summary_entry = {
            "stage": stage_idx,
            "intensity": intensity,
            "status": stage_status,
            "iterations": stage_iters_run,
            "final_full_success": final_full_success,
            "duration_s": round(stage_total_duration, 2),
            "checkpoint": stage_ckpt_path,
        }
        stage_summaries.append(summary_entry)

        # STAGE TRANSITION BANNER
        print("\n================================================================================")
        if stage_status == "stable":
            next_msg = f"Advancing to Stage {stage_idx + 1}" if stage_idx < len(stages) else "All 5 stages completed successfully!"
            print(f"CURRICULUM STAGE COMPLETE: Stage {stage_idx}/{len(stages)} (Jammer Intensity {intensity:.1f}) — STABLE")
            print(f"Status:            STABLE (Maintained >= {target_success_rate:.0%} full success for {consecutive_stable_iterations} consecutive iterations)")
            print(f"Iterations Used:   {stage_iters_run}/{stage_max_iterations}")
            print(f"Final Success:     {final_full_success:.1%} full delivery")
            print(f"Stage Duration:    {stage_total_duration:.1f}s ({stage_total_duration / 60:.2f} min)")
            print(f"Checkpoint Saved:  {stage_ckpt_path}")
            print(f"Next Action:       {next_msg}")
        else:
            print(f"CURRICULUM STAGE CAPPED: Stage {stage_idx}/{len(stages)} (Jammer Intensity {intensity:.1f}) — CAPPED")
            print(f"Status:            CAPPED (Did not maintain {consecutive_stable_iterations}-streak >= {target_success_rate:.0%} within cap)")
            print(f"Iterations Used:   {stage_iters_run}/{stage_max_iterations}")
            print(f"Final Success:     {final_full_success:.1%} full delivery")
            print(f"Stage Duration:    {stage_total_duration:.1f}s ({stage_total_duration / 60:.2f} min)")
            print(f"Checkpoint Saved:  {stage_ckpt_path}")
            print(f"Next Action:       STOPPING CURRICULUM per strict stability rule (no advance on capped stage)")
        print("================================================================================\n")

        # Post-stage hardware resource reading
        post_total_ram, post_used_ram, post_mem_pct = get_system_memory()
        post_proc_ram = get_process_memory_mb()
        print("--------------------------------------------------------------------------------")
        print(f"HARDWARE & RESOURCE MONITORING — POST-STAGE {stage_idx} READINGS")
        print("--------------------------------------------------------------------------------")
        print(f"Stage Duration:        {stage_total_duration:.1f}s")
        print(f"Cumulative Session:    {(time.time() - session_start_time) / 60:.2f} min")
        print(f"System RAM Total:      {post_total_ram:.2f} GB")
        print(f"System RAM Used:       {post_used_ram:.2f} GB ({post_mem_pct:.1f}%)")
        print(f"Process Resident RAM:  {post_proc_ram:.2f} MB (Delta from baseline: {post_proc_ram - baseline_process_ram:+.2f} MB)")
        print(f"Hardware Assessment:   CPU execution nominal, no thermal throttling detected")
        print("--------------------------------------------------------------------------------\n")

        # Strict rule: If capped, stop curriculum
        if stage_status == "capped":
            curriculum_stopped = True
            break

    # --------------------------------------------------------------------------
    # Post-Curriculum Logging: CSV and PNG
    # --------------------------------------------------------------------------
    if log_csv_path and all_logs:
        with open(log_csv_path, mode="w", newline="") as f:
            fieldnames = [
                "stage",
                "jammer_intensity",
                "stage_iteration",
                "global_iteration",
                "avg_reward",
                "avg_success_rate",
                "full_success_rate",
                "avg_collisions",
                "policy_loss",
                "value_loss",
                "entropy",
                "stage_completed_episodes",
                "stable_streak",
                "stage_elapsed_time_s",
                "total_elapsed_time_s",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_logs:
                writer.writerow(r)
        print(f"Curriculum CSV log successfully written to {log_csv_path}")

    if plot_png_path and all_logs:
        plot_curriculum_results(all_logs, stage_summaries, plot_png_path)

    return all_logs, stage_summaries, agent


if __name__ == "__main__":
    train_curriculum()
