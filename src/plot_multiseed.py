import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    baselines_csv = os.path.join(base_dir, "logs", "cartpole_baselines.csv")
    seeds_csv = os.path.join(base_dir, "logs", "cartpole_reinforce_seeds.csv")
    plot_png = os.path.join(base_dir, "logs", "cartpole_reinforce_seeds.png")

    # Read baselines from logs/cartpole_baselines.csv
    random_returns = []
    rule_returns = []
    with open(baselines_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["policy"] == "random":
                random_returns.append(float(row["episode_return"]))
            elif row["policy"] == "rule":
                rule_returns.append(float(row["episode_return"]))

    mean_random = float(np.mean(random_returns))
    mean_rule = float(np.mean(rule_returns))

    # Read multi-seed REINFORCE data
    seed_episodes = {}
    with open(seeds_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seed = int(row["seed"])
            ep = int(row["episode"])
            ma50 = float(row["moving_avg_50"])
            if seed not in seed_episodes:
                seed_episodes[seed] = {}
            seed_episodes[seed][ep] = ma50

    seeds = sorted(list(seed_episodes.keys()))
    all_eps = sorted(list(set.union(*[set(seed_episodes[s].keys()) for s in seeds])))

    episodes_list = []
    mean_ma_list = []
    std_ma_list = []

    for ep in all_eps:
        vals = [seed_episodes[s][ep] for s in seeds if ep in seed_episodes[s]]
        if len(vals) > 0:
            episodes_list.append(ep)
            mean_ma_list.append(np.mean(vals))
            std_ma_list.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

    episodes_arr = np.array(episodes_list)
    mean_ma_arr = np.array(mean_ma_list)
    std_ma_arr = np.array(std_ma_list)

    plt.figure(figsize=(10, 6))

    # One thin line per individual seed
    colors = ["#2ca02c", "#ff7f0e", "#9467bd"]
    for idx, s in enumerate(seeds):
        eps = sorted(seed_episodes[s].keys())
        vals = [seed_episodes[s][e] for e in eps]
        plt.plot(eps, vals, label=f"Seed {s}", linewidth=1.0, alpha=0.45, linestyle="--", color=colors[idx % len(colors)])

    # Mean curve across seeds
    plt.plot(episodes_arr, mean_ma_arr, label="REINFORCE Mean (50-ep MA)", color="#1f77b4", linewidth=2.5)

    # Shaded band (+/- 1 std dev)
    plt.fill_between(
        episodes_arr,
        np.maximum(0, mean_ma_arr - std_ma_arr),
        mean_ma_arr + std_ma_arr,
        color="#1f77b4",
        alpha=0.25,
        label="REINFORCE ±1 Std Dev"
    )

    # Baselines (horizontal dashed lines)
    plt.axhline(mean_random, color="#d62728", linestyle="--", linewidth=1.5, label=f"S0.2 Random Baseline Mean ({mean_random:.1f})")
    plt.axhline(mean_rule, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"S0.2 Rule Baseline Mean ({mean_rule:.1f})")

    # Project stop target reference line
    plt.axhline(195.0, color="#7f7f7f", linestyle=":", linewidth=1.2, label="Project stop target (195)")

    plt.title("CartPole-v1: REINFORCE Multi-Seed Learning Curves vs Baselines", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("50-Episode Moving Average Return", fontsize=12)
    plt.xlim(1, max(episodes_arr) if len(episodes_arr) > 0 else 300)
    plt.ylim(bottom=0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", fontsize=10, framealpha=0.9)
    plt.tight_layout()

    plt.savefig(plot_png, dpi=200)
    plt.close()
    print(f"Plot successfully saved to {plot_png}")

if __name__ == "__main__":
    main()
