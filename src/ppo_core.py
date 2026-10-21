"""
ppo_core.py - From-scratch Proximal Policy Optimization (PPO) core implementation.

Specification:
- Actor Network: 2 hidden layers, 64 units each, Tanh activation, outputs Categorical distribution.
- Critic Network: Separate 2 hidden layers, 64 units each, Tanh activation, outputs scalar state-value V(s).
  (Actor and Critic do NOT share weights).
- Rollout: Fixed 2048 steps collected sequentially.
- Advantage Estimation: Generalized Advantage Estimation (GAE), gamma=0.99, lambda=0.95.
- Objective: Clipped surrogate objective (clip epsilon=0.2), value function loss (MSE, coef 0.5),
  and entropy bonus (coef 0.01) - standard PPO total loss.
- Optimizer: Shared Adam optimizer over actor and critic parameters, lr=3e-4.
- Update: 4 epochs per rollout, minibatch size 64.
- Hardware: Pure CPU PyTorch (no CUDA / GPU code).
"""

import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym


class Actor(nn.Module):
    """
    Actor network mapping state to a categorical action distribution.
    Architecture: 2 hidden layers of 64 units each with Tanh activation.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> Categorical:
        logits = self.net(state)
        return Categorical(logits=logits)

    def get_action_and_log_prob(
        self, state: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.forward(state)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy


class Critic(nn.Module):
    """
    Critic network mapping state to scalar state-value estimate V(s).
    Architecture: 2 hidden layers of 64 units each with Tanh activation.
    Does not share weights with the Actor.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Generalized Advantage Estimation (GAE) and target returns.

    Args:
        rewards: Tensor of shape (T,) with step rewards.
        values: Tensor of shape (T,) with state values V(s_t).
        dones: Tensor of shape (T,) with boolean termination flags.
        next_value: Float scalar bootstrap value for state after the rollout.
        gamma: Discount factor (default: 0.99).
        gae_lambda: GAE lambda parameter (default: 0.95).

    Returns:
        Tuple of (advantages, returns) both of shape (T,).
    """
    num_steps = len(rewards)
    advantages = torch.zeros(num_steps, dtype=torch.float32)
    last_gae = 0.0

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_val = next_value
        else:
            next_val = values[t + 1].item()

        next_non_terminal = 1.0 - dones[t].item()
        delta = rewards[t].item() + gamma * next_val * next_non_terminal - values[t].item()
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


class PPOAgent:
    """
    Single-agent PPO Agent with separate Actor and Critic networks.
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
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
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, hidden_dim).to(self.device)

        # Shared Adam optimizer over separate actor and critic parameter sets
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
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Tuple[float, float, float]:
        """
        Executes PPO update over rollout data for epochs_per_rollout epochs in minibatches.

        Returns:
            Tuple of (mean_policy_loss, mean_value_loss, mean_entropy).
        """
        batch_size = states.size(0)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.epochs_per_rollout):
            indices = torch.randperm(batch_size)
            for start in range(0, batch_size, self.minibatch_size):
                mb_indices = indices[start : start + self.minibatch_size]

                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]

                # Evaluate actions and values under current policy
                _, new_log_probs, entropy = self.actor.get_action_and_log_prob(mb_states, mb_actions)
                new_values = self.critic(mb_states)

                # Clipped surrogate objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss (MSE)
                value_loss = 0.5 * torch.mean((new_values - mb_returns) ** 2)

                # Mean entropy bonus
                entropy_loss = entropy.mean()

                # Total standard PPO loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

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


def train_ppo(
    env_name: str = "CartPole-v1",
    seed: int = 42,
    rollout_steps: int = 2048,
    max_iterations: int = 200,
    target_avg_reward: float = 475.0,
    reward_window: int = 20,
    checkpoint_dir: str = "checkpoints/ppo_cartpole",
    checkpoint_freq: int = 20,
    log_csv_path: Optional[str] = "logs/ppo_cartpole_run1.csv",
) -> Tuple[List[Dict[str, Any]], PPOAgent, int, bool]:
    """
    Runs the PPO training loop on CartPole-v1.

    Args:
        env_name: Gymnasium environment ID (default: "CartPole-v1").
        seed: Random seed for reproducibility.
        rollout_steps: Steps collected per rollout iteration (default: 2048).
        max_iterations: Maximum update iterations (default: 200).
        target_avg_reward: Success bar threshold (default: 475.0).
        reward_window: Episode window for moving average (default: 20).
        checkpoint_dir: Directory to save policy checkpoints.
        checkpoint_freq: Frequency in iterations to save checkpoints (default: 20).
        log_csv_path: Path to write CSV log (optional).

    Returns:
        Tuple of (logs, agent, final_iteration, success_bool).
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize sequential CartPole-v1 environment
    env = gym.make(env_name)
    obs, _ = env.reset(seed=seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

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

    episode_rewards: List[float] = []
    current_episode_return: float = 0.0
    logs: List[Dict[str, Any]] = []
    success: bool = False
    solved_iteration: Optional[int] = None

    os.makedirs(checkpoint_dir, exist_ok=True)
    if log_csv_path:
        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)

    print(f"Starting PPO Training on {env_name} (Sequential rollout, {rollout_steps} steps/update)...")
    print(f"Goal: Average episode reward >= {target_avg_reward} over last {reward_window} episodes (Max {max_iterations} iterations)\n")

    for iteration in range(1, max_iterations + 1):
        states_list: List[torch.Tensor] = []
        actions_list: List[torch.Tensor] = []
        log_probs_list: List[torch.Tensor] = []
        rewards_list: List[torch.Tensor] = []
        dones_list: List[torch.Tensor] = []
        values_list: List[torch.Tensor] = []

        # Collect 2048 steps sequentially
        for _ in range(rollout_steps):
            state_t = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action, log_prob, _ = agent.actor.get_action_and_log_prob(state_t)
                val = agent.critic(state_t)

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            current_episode_return += reward
            done = terminated or truncated

            states_list.append(state_t)
            actions_list.append(action)
            log_probs_list.append(log_prob)
            rewards_list.append(torch.tensor(reward, dtype=torch.float32))
            # True failure termination (terminated) vs time-limit truncation (truncated)
            dones_list.append(torch.tensor(1.0 if terminated else 0.0, dtype=torch.float32))
            values_list.append(val)

            if done:
                episode_rewards.append(current_episode_return)
                current_episode_return = 0.0
                obs, _ = env.reset()
            else:
                obs = next_obs

        # Bootstrap next_value for GAE
        with torch.no_grad():
            next_state_t = torch.tensor(obs, dtype=torch.float32)
            next_val = agent.critic(next_state_t).item()

        # Compute GAE advantages and target returns
        b_rewards = torch.stack(rewards_list)
        b_values = torch.stack(values_list)
        b_dones = torch.stack(dones_list)
        advantages, returns = compute_gae(
            b_rewards,
            b_values,
            b_dones,
            next_val,
            gamma=agent.gamma,
            gae_lambda=agent.gae_lambda,
        )

        b_states = torch.stack(states_list)
        b_actions = torch.stack(actions_list)
        b_old_log_probs = torch.stack(log_probs_list)

        # Update PPO policy and value networks
        policy_loss, value_loss, entropy = agent.update(
            b_states, b_actions, b_old_log_probs, returns, advantages
        )

        # Track moving average reward over last reward_window episodes
        if len(episode_rewards) > 0:
            avg_reward = float(np.mean(episode_rewards[-reward_window:]))
        else:
            avg_reward = 0.0

        print(
            f"Iteration {iteration:3d} | "
            f"Avg Reward (last {min(len(episode_rewards), reward_window):2d}): {avg_reward:6.2f} | "
            f"Policy Loss: {policy_loss:+.4f} | "
            f"Value Loss: {value_loss:6.4f} | "
            f"Entropy: {entropy:.4f} | "
            f"Total Completed Episodes: {len(episode_rewards)}"
        )

        record = {
            "iteration": iteration,
            "avg_reward": avg_reward,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
        }
        logs.append(record)

        # Periodic checkpointing
        if iteration % checkpoint_freq == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"ppo_iteration_{iteration}.pt")
            agent.save_checkpoint(ckpt_path)
            print(f"  -> Checkpoint saved to {ckpt_path}")

        # Check success condition: stop immediately once hit
        if len(episode_rewards) >= reward_window and avg_reward >= target_avg_reward:
            success = True
            solved_iteration = iteration
            print(
                f"\nSUCCESS BAR HIT: Average reward {avg_reward:.2f} >= {target_avg_reward} "
                f"over last {reward_window} episodes reached at iteration {iteration}!"
            )
            # Save final checkpoint at success
            final_ckpt = os.path.join(checkpoint_dir, f"ppo_iteration_{iteration}_solved.pt")
            agent.save_checkpoint(final_ckpt)
            print(f"  -> Solved model checkpoint saved to {final_ckpt}")
            break

    env.close()

    # Save CSV log if path specified
    if log_csv_path:
        import csv

        with open(log_csv_path, mode="w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["iteration", "avg_reward", "policy_loss", "value_loss", "entropy"],
            )
            writer.writeheader()
            for r in logs:
                writer.writerow(r)
        print(f"\nCSV log successfully written to {log_csv_path}")

    return logs, agent, iteration, success

