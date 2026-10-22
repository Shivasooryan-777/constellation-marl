"""
src/ippo_train.py - Independent PPO (IPPO) training loop for CONSTELLATION multi-drone delivery.

Architecture:
- Shared weights across all 5 drones: exactly one Actor and one Critic network from src/ppo_core.py.
- Independent decision-making: each drone passes its own local observation (get_observation(drone_id))
  into the shared Actor to sample its action and into the shared Critic to compute its value estimate.
- NO centralized critic: global state queries are strictly forbidden and NOT called anywhere in this file.
- Single fixed jamming/no-fly-zone condition: 0% jammer intensity, 0 no-fly zones active (n_no_fly_zones=0).
- Reward computation via compute_reward() in src/rewards.py with caller-maintained prev_distances.
- GAE advantages and target returns computed per drone trajectory before pooling into the shared PPO batch.
"""

import os
import sys
import time
import csv
import ctypes
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch

from src.drone_world import DroneWorld
from src.metrics import MetricsTracker
from src.rewards import compute_reward
from src.ppo_core import PPOAgent, compute_gae


# ==============================================================================
# Hardware / OS Resource Monitoring Helpers (Windows standard library ctypes)
# ==============================================================================

class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def get_system_memory() -> Tuple[float, float, float]:
    """Returns (total_ram_gb, used_ram_gb, memory_load_pct)."""
    try:
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        total_gb = stat.ullTotalPhys / (1024.0 ** 3)
        avail_gb = stat.ullAvailPhys / (1024.0 ** 3)
        used_gb = total_gb - avail_gb
        pct = float(stat.dwMemoryLoad)
        return total_gb, used_gb, pct
    except Exception:
        return 0.0, 0.0, 0.0


def get_process_memory_mb() -> float:
    """Returns process Working Set (resident RAM) in MB using 64-bit Windows API."""
    try:
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
            wintypes.DWORD,
        ]
        k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL

        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        h_process = k32.GetCurrentProcess()
        success = k32.K32GetProcessMemoryInfo(h_process, ctypes.byref(counters), counters.cb)
        if success:
            return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
        return 0.0
    except Exception:
        return 0.0



# ==============================================================================
# Distance Tracking Helper for Reward Computation
# ==============================================================================

def get_current_distances(world: DroneWorld) -> Dict[int, float]:
    """
    Computes Manhattan distance to nearest undelivered delivery point for each drone.
    Identical to the pattern validated in S2.1 (tests/check_rewards.py).
    """
    distances: Dict[int, float] = {}
    for i in range(world.n_drones):
        pos = world.drones[i]
        if world.remaining_deliveries:
            target = min(
                world.remaining_deliveries,
                key=lambda p: (abs(pos[0] - p[0]) + abs(pos[1] - p[1]), p[0], p[1]),
            )
            dist = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
            distances[i] = float(dist)
        else:
            distances[i] = 0.0
    return distances


def make_env(
    seed: int = 0,
    grid_size: int = 20,
    n_drones: int = 5,
    n_delivery_points: int = 3,
    max_steps: int = 250,
) -> DroneWorld:
    """
    Instantiates DroneWorld under the fixed S2.3 specifications:
    - 20x20 grid, 5 drones, depot at (0, 0), 3 delivery points.
    - No no-fly zones active (n_no_fly_zones=0).
    - 0% jammer intensity (jammer_intensity=0.0).
    - Episode length cap: 250 steps.
    """
    return DroneWorld(
        grid_size=grid_size,
        n_drones=n_drones,
        n_delivery_points=n_delivery_points,
        seed=seed,
        n_no_fly_zones=0,
        jammer_intensity=0.0,
        max_steps=max_steps,
    )


# ==============================================================================
# IPPO Training Loop
# ==============================================================================

def train_ippo(
    seed: int = 0,
    grid_size: int = 20,
    n_drones: int = 5,
    n_delivery_points: int = 3,
    episode_max_steps: int = 250,
    rollout_world_steps: int = 410,
    max_iterations: int = 150,
    target_success_rate: float = 0.80,
    window_size: int = 20,
    checkpoint_dir: str = "checkpoints/ippo_dronworld",
    checkpoint_freq: int = 20,
    log_csv_path: Optional[str] = "logs/ippo_droneworld_run1.csv",
    time_limit_seconds: float = 3600.0,
) -> Tuple[List[Dict[str, Any]], PPOAgent, int, bool]:
    """
    Runs Independent PPO (IPPO) training on DroneWorld.

    Args:
        seed: Random seed for environment and PyTorch.
        grid_size: Dimension of grid (default: 20).
        n_drones: Number of delivery drones (default: 5).
        n_delivery_points: Number of delivery points (default: 3).
        episode_max_steps: Maximum steps per episode (default: 250).
        rollout_world_steps: World steps per rollout update (default: 410 -> 2050 drone steps).
        max_iterations: Maximum PPO update iterations (default: 150).
        target_success_rate: Early stopping success threshold (0.80 = 80% full completion over last 20 eps).
        window_size: Moving window of episodes for averaging metrics (default: 20).
        checkpoint_dir: Directory where policy checkpoints are saved.
        checkpoint_freq: Save checkpoint every N iterations (default: 20).
        log_csv_path: File path to save CSV run log.
        time_limit_seconds: Hardware safety wall-clock cap (default: 3600s = 60 minutes).

    Returns:
        Tuple of (logs, agent, final_iteration, success_bool).
    """
    # Enforce reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # State dimension: 7 floats from get_observation(drone_id)
    # Action dimension: 5 discrete actions (STAY=0, UP=1, DOWN=2, LEFT=3, RIGHT=4)
    state_dim = 7
    action_dim = 5

    # Shared Actor and Critic networks across all 5 drones
    # Hyperparameters strictly match S2.2: gamma=0.99, lambda=0.95, clip=0.2,
    # value_coef=0.5, entropy_coef=0.01, lr=3e-4, 4 epochs, minibatch 64, no advantage normalization.
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

    # Initialize environment and metric tracker
    world = make_env(
        seed=seed,
        grid_size=grid_size,
        n_drones=n_drones,
        n_delivery_points=n_delivery_points,
        max_steps=episode_max_steps,
    )
    tracker = MetricsTracker(world=world)

    # Per-episode tracking across sequential rollouts
    ep_drone_returns: List[float] = [0.0] * n_drones
    completed_episodes: List[Dict[str, Any]] = []
    logs: List[Dict[str, Any]] = []
    success_achieved: bool = False
    solved_iteration: Optional[int] = None

    os.makedirs(checkpoint_dir, exist_ok=True)
    if log_csv_path:
        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)

    print("================================================================================")
    print("CONSTELLATION — Session S2.3: IPPO Training on DroneWorld (CPU-only)")
    print("================================================================================")
    print(f"Environment: Grid {grid_size}x{grid_size} | Drones: {n_drones} | Delivery Points: {n_delivery_points}")
    print(f"Jammer: 0% intensity | No-Fly Zones: None active | Episode cap: {episode_max_steps} steps")
    print(f"Policy: Shared Actor/Critic (state_dim={state_dim}, action_dim={action_dim}, hidden=64)")
    print(f"Rollout: {rollout_world_steps} world-steps * {n_drones} drones = {rollout_world_steps * n_drones} samples/update")
    print(f"Update: 4 epochs, minibatch 64, Adam lr=3e-4, GAE(gamma=0.99, lambda=0.95), no adv norm")
    print(f"Early stop bar: Average full delivery success >= {target_success_rate:.0%} over last {window_size} episodes")
    print(f"Run safety limits: Max {max_iterations} iterations, {time_limit_seconds / 60:.0f}-minute wall-clock cap")
    print("================================================================================\n")

    start_wall_time = time.time()
    baseline_process_ram = get_process_memory_mb()

    for iteration in range(1, max_iterations + 1):
        iter_start_time = time.time()
        elapsed_total = iter_start_time - start_wall_time

        # Wall-clock safety cap check (60 minutes)
        if elapsed_total >= time_limit_seconds:
            print(f"\n[SAFETY LIMIT] Wall-clock cap reached ({elapsed_total:.1f}s >= {time_limit_seconds}s).")
            print("Stopping training and reporting partial results honestly.")
            break

        # RAM leak safety check
        current_proc_ram = get_process_memory_mb()
        if current_proc_ram > baseline_process_ram + 1500.0:
            print(f"\n[SAFETY ABORT] Abnormal RAM growth detected: {current_proc_ram:.1f} MB "
                  f"(Baseline: {baseline_process_ram:.1f} MB). Stopping to avoid system stall.")
            break

        # Per-drone rollout storage
        drone_obs: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
        drone_acts: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
        drone_lps: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
        drone_vals: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
        drone_rews: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
        drone_dones: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]

        # Collect rollout across rollout_world_steps
        for _ in range(rollout_world_steps):
            # 1. Each drone independently observes and samples action from shared policy
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

            # 2. Record pre-step distances for potential-based shaping
            prev_distances = get_current_distances(world)

            # 3. Step environment through MetricsTracker
            _ = tracker.record_step(actions)

            # 4. Compute per-drone reward via src/rewards.py
            for d in range(n_drones):
                rew_d = compute_reward(world, prev_distances, d)
                drone_rews[d].append(torch.tensor(rew_d, dtype=torch.float32))
                ep_drone_returns[d] += rew_d

            # 5. Check episode termination: all deliveries done or max episode steps reached
            all_delivered = (len(world.remaining_deliveries) == 0)
            time_up = (world.t >= episode_max_steps)
            episode_done = all_delivered or time_up

            for d in range(n_drones):
                # 1.0 at episode boundary to isolate GAE advantages
                drone_dones[d].append(torch.tensor(1.0 if episode_done else 0.0, dtype=torch.float32))

            # 6. Episode wrap-up & re-initialization
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

                # Reset environment for the next episode
                world = make_env(
                    seed=seed,
                    grid_size=grid_size,
                    n_drones=n_drones,
                    n_delivery_points=n_delivery_points,
                    max_steps=episode_max_steps,
                )
                tracker = MetricsTracker(world=world)
                ep_drone_returns = [0.0] * n_drones

        # GAE Bootstrap: obtain next_value for each drone
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

        # Compute GAE advantages and target returns per drone trajectory
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
                b_rews,
                b_vals,
                b_dones,
                next_values[d],
                gamma=agent.gamma,
                gae_lambda=agent.gae_lambda,
            )
            all_states.append(torch.stack(drone_obs[d]))
            all_actions.append(torch.stack(drone_acts[d]))
            all_log_probs.append(torch.stack(drone_lps[d]))
            all_returns.append(ret_d)
            all_advantages.append(adv_d)

        # Pool all 5 drones' rollout data into a single shared training batch (2050 samples)
        batch_states = torch.cat(all_states, dim=0)
        batch_actions = torch.cat(all_actions, dim=0)
        batch_log_probs = torch.cat(all_log_probs, dim=0)
        batch_returns = torch.cat(all_returns, dim=0)
        batch_advantages = torch.cat(all_advantages, dim=0)

        # PPO Update: 4 epochs over minibatches of size 64 without advantage normalization
        policy_loss, value_loss, entropy = agent.update(
            batch_states,
            batch_actions,
            batch_log_probs,
            batch_returns,
            batch_advantages,
        )

        # Compute rolling window metrics over the last window_size (20) completed episodes
        recent = completed_episodes[-window_size:] if completed_episodes else []
        n_recent = len(recent)
        avg_reward = float(np.mean([e["reward"] for e in recent])) if recent else 0.0
        avg_success = float(np.mean([e["success"] for e in recent])) if recent else 0.0
        avg_collisions = float(np.mean([e["collisions"] for e in recent])) if recent else 0.0
        full_success_rate = (
            float(sum(1 for e in recent if e["is_full_success"]) / float(n_recent))
            if recent
            else 0.0
        )

        elapsed_iter_s = time.time() - iter_start_time
        total_elapsed_s = time.time() - start_wall_time

        print(
            f"Iter {iteration:3d}/{max_iterations} | "
            f"Avg Reward (last {n_recent:2d} eps): {avg_reward:7.2f} | "
            f"Success Rate: {avg_success:5.2f} (Full: {full_success_rate:5.1%}) | "
            f"Collisions: {avg_collisions:5.1f} | "
            f"PLoss: {policy_loss:+.4f} | "
            f"VLoss: {value_loss:6.4f} | "
            f"Entropy: {entropy:.4f} | "
            f"Eps: {len(completed_episodes):3d} | "
            f"Time: {total_elapsed_s:5.1f}s"
        )

        record = {
            "iteration": iteration,
            "avg_reward": avg_reward,
            "avg_success_rate": avg_success,
            "full_success_rate": full_success_rate,
            "avg_collisions": avg_collisions,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "completed_episodes": len(completed_episodes),
            "elapsed_time_s": round(total_elapsed_s, 2),
        }
        logs.append(record)

        # Checkpoint every checkpoint_freq (20) iterations
        if iteration % checkpoint_freq == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"ippo_iteration_{iteration}.pt")
            agent.save_checkpoint(ckpt_path)
            print(f"  -> Checkpoint saved to {ckpt_path}")

        # Check early stopping condition: full completion in >= 80% of last 20 episodes
        if n_recent >= window_size and full_success_rate >= target_success_rate:
            success_achieved = True
            solved_iteration = iteration
            print(
                f"\nSUCCESS BAR HIT: Full delivery success rate {full_success_rate:.1%} >= {target_success_rate:.0%} "
                f"over last {window_size} episodes reached at iteration {iteration}!"
            )
            solved_ckpt = os.path.join(checkpoint_dir, f"ippo_iteration_{iteration}_solved.pt")
            agent.save_checkpoint(solved_ckpt)
            print(f"  -> Solved model checkpoint saved to {solved_ckpt}")
            break

    # Write CSV log file
    if log_csv_path:
        with open(log_csv_path, mode="w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "iteration",
                    "avg_reward",
                    "avg_success_rate",
                    "full_success_rate",
                    "avg_collisions",
                    "policy_loss",
                    "value_loss",
                    "entropy",
                    "completed_episodes",
                    "elapsed_time_s",
                ],
            )
            writer.writeheader()
            for r in logs:
                writer.writerow(r)
        print(f"\nCSV log successfully written to {log_csv_path}")

    return logs, agent, iteration, success_achieved


if __name__ == "__main__":
    train_ippo()
