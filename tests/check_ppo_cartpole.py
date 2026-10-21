"""
check_ppo_cartpole.py - Verification script for PPO core on CartPole-v1.

Verifications:
1. Executes PPO training loop on CartPole-v1 from scratch.
2. Logs per update iteration: average episode reward (last 20 episodes), policy loss,
   value loss, entropy.
3. Checkpoints policy every 20 iterations to checkpoints/ppo_cartpole/.
4. Stops immediately once average reward reaches 475+ over last 20 episodes or at 200 iterations.
5. Saves CSV log to logs/ppo_cartpole_run1.csv.
6. Generates and saves matplotlib plot to logs/ppo_cartpole_run1.png.
"""

import sys
import os
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ppo_core import train_ppo


def main():
    print("================================================================================")
    print("CONSTELLATION - PPO Core Validation on CartPole-v1 (Session S2.2)")
    print("================================================================================\n")

    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "ppo_cartpole_run1.csv"))
    plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "ppo_cartpole_run1.png"))
    checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "ppo_cartpole"))

    logs, agent, final_iter, success = train_ppo(
        env_name="CartPole-v1",
        seed=42,
        rollout_steps=2048,
        max_iterations=200,
        target_avg_reward=475.0,
        reward_window=20,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=20,
        log_csv_path=csv_path,
    )

    # Generate matplotlib plot of average episode reward vs update iteration
    iterations = [entry["iteration"] for entry in logs]
    avg_rewards = [entry["avg_reward"] for entry in logs]

    plt.figure(figsize=(9, 5))
    plt.plot(iterations, avg_rewards, color="#1f77b4", linewidth=2.0, label="PPO Avg Reward (20 eps)")
    plt.axhline(y=475.0, color="#d62728", linestyle="--", linewidth=1.5, label="Success Bar (475.0)")

    plt.title("CartPole-v1 PPO Training Curve (Run 1)", fontsize=13, fontweight="bold")
    plt.xlabel("Update Iteration (2048 steps / iteration)", fontsize=11)
    plt.ylabel("Average Episode Reward (last 20 episodes)", fontsize=11)
    plt.ylim(0, 520)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Training plot successfully saved to {plot_path}")

    print("\n================================================================================")
    print("FINAL RUN SUMMARY")
    print("================================================================================")
    print(f"Total Update Iterations: {final_iter}")
    final_avg = logs[-1]["avg_reward"] if logs else 0.0
    print(f"Final 20-Episode Average Reward: {final_avg:.2f} / 500.00")
    print(f"Success Bar (>= 475.0) Hit: {'YES' if success else 'NO'}")
    if success:
        print(f"Solved Iteration: {final_iter}")
    print(f"CSV log location: {csv_path}")
    print(f"Plot location:    {plot_path}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
