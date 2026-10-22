"""
src/mappo_train.py - Centralized Critic Multi-Agent PPO (MAPPO) training loop for CONSTELLATION.

Architecture:
- Decentralized Execution: Shared Actor network mapping local observation (get_observation(drone_id),
  dim 7) to action categorical distribution. Unchanged from S2.2/S2.3.
- Centralized Critic: Single Centralized Critic network consuming global world state (get_global_state(),
  dim 14) and outputting one scalar state-value estimate V(s_global) representing the entire team state.
- Value assignment in GAE: Every drone at timestep t uses the same global value estimate V(s_global_t)
  when computing GAE advantages and target returns.
- Rewards: Unchanged per-drone compute_reward() from src/rewards.py with caller-maintained prev_distances.
- Hyperparameters strictly match S2.2/S2.3: gamma=0.99, lambda=0.95, clip=0.2, value_coef=0.5,
  entropy_coef=0.01, Adam lr=3e-4, 4 epochs, minibatch 64, no advantage normalization.
"""

import os
import sys
import time
import csv
import ctypes
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn

from src.drone_world import DroneWorld
from src.metrics import MetricsTracker
from src.rewards import compute_reward
from src.ppo_core import Actor, Critic, compute_gae
from src.ippo_train import (
    get_system_memory,
    get_process_memory_mb,
    get_current_distances,
    make_env,
)


# ==============================================================================
# Global State Vectorizer (Consumes get_global_state() from S2.1)
# ==============================================================================

def flatten_global_state(world: DroneWorld) -> np.ndarray:
    """
    Flattens world.get_global_state() into a fixed 14-element normalized float32 vector.

    Vector specification (length 14, strictly fixed):
    - 0-9: All 5 drone (x, y) positions, normalized by grid_size (20.0). [10 floats]
    - 10-12: Delivery completion statuses (1.0 if delivered, 0.0 if not) for the
      3 delivery points. [3 floats]
    - 13: Current timestep t normalized by max_steps (250.0). [1 float]

    Total dimension: 10 + 3 + 1 = 14 floats.
    """
    gs = world.get_global_state()
    norm_grid = float(world.grid_size)

    # 1. All 5 drone (x, y) positions normalized to [0, 1]
    drone_coords: List[float] = []
    for x, y in gs["drones"]:
        drone_coords.append(float(x) / norm_grid)
        drone_coords.append(float(y) / norm_grid)

    # 2. 3 delivery statuses (1.0 if delivered, 0.0 if remaining)
    delivery_statuses: List[float] = []
    for p in world.delivery_points:
        is_delivered = 1.0 if gs["delivery_statuses"].get(p, False) else 0.0
        delivery_statuses.append(is_delivered)

    # 3. Normalized simulation time t
    max_steps = float(world.max_steps) if world.max_steps > 0 else 250.0
    norm_t = min(1.0, max(0.0, float(gs["t"]) / max_steps))

    vec = drone_coords + delivery_statuses + [norm_t]
    return np.array(vec, dtype=np.float32)


# ==============================================================================
# MAPPO Agent Class
# ==============================================================================

class MAPPOAgent:
    """
    Multi-Agent PPO Agent with Centralized Critic and Decentralized Actor.
    - Actor: 2 hidden layers of 64 units, Tanh activation (input_dim=7, output_dim=5).
    - Centralized Critic: 2 hidden layers of 64 units, Tanh activation (input_dim=14, output_dim=1).
    - Shared Adam optimizer over both networks, lr=3e-4.
    """

    def __init__(
        self,
        local_state_dim: int = 7,
        global_state_dim: int = 14,
        action_dim: int = 5,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        epochs_per_rollout: int = 4,
        minibatch_size: int = 64,
    ) -> None:
        self.device = torch.device("cpu")
        self.actor = Actor(local_state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(global_state_dim, hidden_dim).to(self.device)

        # Shared Adam optimizer across actor and centralized critic parameters
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs_per_rollout = epochs_per_rollout
        self.minibatch_size = minibatch_size

    def update(
        self,
        local_states: torch.Tensor,
        global_states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Tuple[float, float, float]:
        """
        Executes MAPPO update:
        - Actor evaluated on local states (dim 7).
        - Critic evaluated on global states (dim 14).
        - No advantage normalization (strictly matches S2.2/S2.3 spec).
        """
        batch_size = local_states.size(0)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.epochs_per_rollout):
            indices = torch.randperm(batch_size)
            for start in range(0, batch_size, self.minibatch_size):
                mb_indices = indices[start : start + self.minibatch_size]

                mb_local_states = local_states[mb_indices]
                mb_global_states = global_states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]

                # 1. Decentralized Actor evaluates local observations
                _, new_log_probs, entropy = self.actor.get_action_and_log_prob(
                    mb_local_states, mb_actions
                )

                # 2. Centralized Critic evaluates global states
                new_values = self.critic(mb_global_states)

                # 3. Clipped surrogate objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # 4. Value function loss (MSE against target returns)
                value_loss = 0.5 * torch.mean((new_values - mb_returns) ** 2)

                # 5. Entropy bonus
                entropy_loss = entropy.mean()

                # Total standard PPO loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                num_updates += 1

        return (
            total_policy_loss / num_updates,
            total_value_loss / num_updates,
            total_entropy / num_updates,
        )

    def save_checkpoint(self, path: str) -> None:
        """Saves actor, critic, and optimizer state dictionaries to path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Loads actor, critic, and optimizer state dictionaries from path."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


# ==============================================================================
# MAPPO Training Loop
# ==============================================================================

def train_mappo(
    seed: int = 0,
    grid_size: int = 20,
    n_drones: int = 5,
    n_delivery_points: int = 3,
    episode_max_steps: int = 250,
    rollout_world_steps: int = 410,
    max_iterations: int = 150,
    target_success_rate: float = 0.80,
    window_size: int = 20,
    checkpoint_dir: str = "checkpoints/mappo_droneworld",
    checkpoint_freq: int = 20,
    log_csv_path: Optional[str] = "logs/mappo_droneworld_run1.csv",
    time_limit_seconds: float = 3600.0,
) -> Tuple[List[Dict[str, Any]], MAPPOAgent, int, bool]:
    """
    Runs MAPPO (Centralized Critic, Decentralized Execution) training loop on DroneWorld.

    Returns:
        Tuple of (logs, agent, final_iteration, success_bool).
    """
    # Enforce reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    local_state_dim = 7
    global_state_dim = 14
    action_dim = 5

    # MAPPO agent: Decentralized Actor (7 in) + Centralized Critic (14 in)
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

    # Initialize environment and tracker
    world = make_env(
        seed=seed,
        grid_size=grid_size,
        n_drones=n_drones,
        n_delivery_points=n_delivery_points,
        max_steps=episode_max_steps,
    )
    tracker = MetricsTracker(world=world)

    ep_drone_returns: List[float] = [0.0] * n_drones
    completed_episodes: List[Dict[str, Any]] = []
    logs: List[Dict[str, Any]] = []
    success_achieved: bool = False
    solved_iteration: Optional[int] = None

    os.makedirs(checkpoint_dir, exist_ok=True)
    if log_csv_path:
        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)

    print("================================================================================")
    print("CONSTELLATION — Session S2.4: MAPPO Training on DroneWorld (CPU-only)")
    print("================================================================================")
    print(f"Environment: Grid {grid_size}x{grid_size} | Drones: {n_drones} | Delivery Points: {n_delivery_points}")
    print(f"Jammer: 0% intensity | No-Fly Zones: None active | Episode cap: {episode_max_steps} steps")
    print(f"Policy: Shared Actor (input_dim={local_state_dim}, action_dim={action_dim}, hidden=64)")
    print(f"Critic: Centralized Critic (input_dim={global_state_dim} from get_global_state(), hidden=64)")
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

        # Wall-clock safety cap check
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
        drone_local_obs: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
        drone_global_obs: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
        drone_acts: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
        drone_lps: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
        drone_vals: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
        drone_rews: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]
        drone_dones: List[List[torch.Tensor]] = [[] for _ in range(n_drones)]

        # Collect rollout across rollout_world_steps
        for _ in range(rollout_world_steps):
            # 1. Centralized Critic evaluates the global state at timestep t
            global_vec = flatten_global_state(world)
            global_t = torch.tensor(global_vec, dtype=torch.float32)
            with torch.no_grad():
                val_global_t = agent.critic(global_t)

            # 2. Decentralized Actor evaluates each drone's local observation independently
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
                # In MAPPO: Every drone receives the SAME global value estimate at timestep t
                drone_vals[d].append(val_global_t)
                actions.append(int(act_t.item()))

            # 3. Record pre-step distances for potential-based shaping
            prev_distances = get_current_distances(world)

            # 4. Step environment through MetricsTracker
            _ = tracker.record_step(actions)

            # 5. Compute per-drone reward via src/rewards.py (unchanged)
            for d in range(n_drones):
                rew_d = compute_reward(world, prev_distances, d)
                drone_rews[d].append(torch.tensor(rew_d, dtype=torch.float32))
                ep_drone_returns[d] += rew_d

            # 6. Check episode termination: all deliveries done or max steps reached
            all_delivered = (len(world.remaining_deliveries) == 0)
            time_up = (world.t >= episode_max_steps)
            episode_done = all_delivered or time_up

            for d in range(n_drones):
                # 1.0 at episode boundary to isolate GAE advantages synchronously
                drone_dones[d].append(torch.tensor(1.0 if episode_done else 0.0, dtype=torch.float32))

            # 7. Episode wrap-up & re-initialization
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

                world = make_env(
                    seed=seed,
                    grid_size=grid_size,
                    n_drones=n_drones,
                    n_delivery_points=n_delivery_points,
                    max_steps=episode_max_steps,
                )
                tracker = MetricsTracker(world=world)
                ep_drone_returns = [0.0] * n_drones

        # GAE Bootstrap: obtain next_global_value from Centralized Critic
        if episode_done:
            next_global_val = 0.0
        else:
            next_global_vec = flatten_global_state(world)
            next_global_t = torch.tensor(next_global_vec, dtype=torch.float32)
            with torch.no_grad():
                next_global_val = agent.critic(next_global_t).item()

        # Compute GAE advantages and target returns per drone trajectory
        # Each drone uses its own rewards, but evaluates them against the shared global value baseline
        all_local_states: List[torch.Tensor] = []
        all_global_states: List[torch.Tensor] = []
        all_actions: List[torch.Tensor] = []
        all_log_probs: List[torch.Tensor] = []
        all_returns: List[torch.Tensor] = []
        all_advantages: List[torch.Tensor] = []

        for d in range(n_drones):
            b_rews = torch.stack(drone_rews[d])
            b_vals = torch.stack(drone_vals[d])  # Centralized V(s_global) values
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

        # Pool all 5 drones' rollout data into shared training batch (2050 samples)
        batch_local_states = torch.cat(all_local_states, dim=0)
        batch_global_states = torch.cat(all_global_states, dim=0)
        batch_actions = torch.cat(all_actions, dim=0)
        batch_log_probs = torch.cat(all_log_probs, dim=0)
        batch_returns = torch.cat(all_returns, dim=0)
        batch_advantages = torch.cat(all_advantages, dim=0)

        # MAPPO Update: 4 epochs over minibatches of size 64 without advantage normalization
        policy_loss, value_loss, entropy = agent.update(
            batch_local_states,
            batch_global_states,
            batch_actions,
            batch_log_probs,
            batch_returns,
            batch_advantages,
        )

        # Rolling window metrics over last window_size (20) completed episodes
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

        # Periodic checkpointing to checkpoints/mappo_droneworld/
        if iteration % checkpoint_freq == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"mappo_iteration_{iteration}.pt")
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
            solved_ckpt = os.path.join(checkpoint_dir, f"mappo_iteration_{iteration}_solved.pt")
            agent.save_checkpoint(solved_ckpt)
            print(f"  -> Solved model checkpoint saved to {solved_ckpt}")
            break

    # Save CSV log file
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
    train_mappo()
