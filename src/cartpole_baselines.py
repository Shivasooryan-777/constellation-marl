import os
import csv
import numpy as np
import gymnasium as gym

def run_baselines():
    env = gym.make("CartPole-v1")
    records = []

    # Policy A: random
    for s in range(20):
        obs, info = env.reset(seed=s)
        env.action_space.seed(s)
        ep_return = 0.0
        ep_length = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            ep_length += 1
            done = terminated or truncated
        records.append({
            "policy": "random",
            "seed": s,
            "episode_return": ep_return,
            "episode_length": ep_length
        })

    # Policy B: rule-based
    for s in range(20):
        obs, info = env.reset(seed=s)
        ep_return = 0.0
        ep_length = 0
        done = False
        while not done:
            # obs: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
            pole_angle = obs[2]
            pole_angular_velocity = obs[3]
            action = 1 if (pole_angle + pole_angular_velocity) > 0 else 0
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            ep_length += 1
            done = terminated or truncated
        records.append({
            "policy": "rule",
            "seed": s,
            "episode_return": ep_return,
            "episode_length": ep_length
        })

    env.close()

    # Write records to CSV
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "cartpole_baselines.csv")

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["policy", "seed", "episode_return", "episode_length"])
        writer.writeheader()
        writer.writerows(records)

    # Print summary for each policy
    for pol in ["random", "rule"]:
        pol_records = [r for r in records if r["policy"] == pol]
        returns = [r["episode_return"] for r in pol_records]
        lengths = [r["episode_length"] for r in pol_records]

        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        mean_length = np.mean(lengths)

        print(f"Policy: {pol}")
        print(f"  Episode Return: mean = {mean_return:.2f}, std = {std_return:.2f}")
        print(f"  Episode Length: mean = {mean_length:.2f}")

    print("BASELINES DONE")

if __name__ == "__main__":
    run_baselines()
