"""
tests/check_s3_9_shaping_extra_seeds.py - S3.9 Category B step 4: reward-shaping ablation, extra seeds.

Re-runs the S3.7 protocol UNCHANGED (0% jamming, 0 no-fly zones, 20x20, 5 drones, 3 deliveries, 250-step episodes,
410 world-steps/rollout, max 150 iterations, stop bar 80% over 20 episodes, 10 eval episodes) for NEW seeds
2, 3, 4 (pre-declared before any result was seen; no seed is dropped or replaced), both algorithms, shaping ON/OFF.
S3.7's seeds 0 and 1 are NOT re-run; their numbers are parsed from logs/runs/s3_7_shaping_ablation_*.log.

Pre-declared verdict rule (fixed before running, applied to the combined 5 seeds x 2 algos = 10 ON/OFF pairs,
metric = eval full success, OFF minus ON):
  ROBUST   : exact paired sign-flip permutation test two-sided p < 0.05 AND sign of mean delta agrees in both algos
  REVERSED : same test significant but with OFF better than ON
  MIXED    : otherwise (not significant at N=10 pairs) - reported as inconclusive, not tuned further.

Zero protected-file edits: the existing distance_shaping_enabled toggle is injected through the same
compute_reward wrapper used in S3.7. New output only: checkpoints/s3_9_shaping_extra/, logs/s3_9_shaping_extra_seeds.csv,
logs/runs/s3_9_shaping_extra_seeds_<ts>.log.
"""

import csv
import glob
import itertools
import os
import re
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.tee_logger import TeeLogger
import src.rewards
import src.ippo_train
import src.mappo_train
from src.multiseed_harness import run_single_seed, evaluate_agent
from src.ippo_train import get_system_memory, get_process_memory_mb

NEW_SEEDS = [2, 3, 4]
ALGOS = ["ippo", "mappo"]
CONDITION = {
    "grid_size": 20, "n_drones": 5, "n_delivery_points": 3, "episode_max_steps": 250,
    "rollout_world_steps": 410, "max_iterations": 150, "target_success_rate": 0.80, "window_size": 20,
    "checkpoint_freq": 20, "time_limit_seconds": 600.0, "jammer_intensity": 0.0, "n_no_fly_zones": 0,
}


def wrapper_for(shaping: bool):
    def _wrapped(*args, **kwargs):
        kwargs["distance_shaping_enabled"] = shaping
        return src.rewards.compute_reward(*args, **kwargs)
    return _wrapped


def hw(tag: str, t_wall0: float, cpu0: float) -> None:
    tot, used, pct = get_system_memory()
    wall = time.time() - t_wall0
    cpu = time.process_time() - cpu0
    print(f"[HW {tag}] {time.strftime('%Y-%m-%d %H:%M:%S')} | RAM used {used:.2f}/{tot:.2f} GB ({pct:.1f}%) | "
          f"process RSS {get_process_memory_mb():.1f} MB | elapsed {wall:.1f}s | process CPU time {cpu:.1f}s "
          f"({(100 * cpu / wall) if wall > 0 else 0:.0f}% of one core) | thermals: not measurable from Python on this host")


def parse_s37() -> List[Dict[str, Any]]:
    """Parse the S3.7 comparison table from its log (seeds 0, 1)."""
    path = sorted(glob.glob(os.path.join(PROJECT_ROOT, "logs", "runs", "s3_7_shaping_ablation_*.log")))[-1]
    rows = []
    pat = re.compile(r"^(IPPO|MAPPO)\s+\|\s+(\d+)\s+\|\s+(ON|OFF)\s+\|\s+(YES|NO)\s+\|\s+(\d+)\s+\|\s+([\d.]+)%\s+\|\s+"
                     r"([\d.]+)%\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)")
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                rows.append({"algo": m[1].lower(), "seed": int(m[2]), "shaping": m[3] == "ON", "solved": m[4] == "YES",
                             "iters": int(m[5]), "train_full": float(m[6]) / 100, "eval_full": float(m[7]) / 100,
                             "collisions": float(m[8]), "latency": float(m[9]), "energy": float(m[10]),
                             "source": "S3.7"})
    assert len(rows) == 8, f"expected 8 S3.7 rows, parsed {len(rows)} from {path}"
    return rows


def sign_flip_p(deltas: np.ndarray) -> float:
    """Exact two-sided paired sign-flip permutation test on mean(delta)."""
    obs = abs(deltas.mean())
    n = len(deltas)
    count = 0
    for signs in itertools.product([1, -1], repeat=n):
        if abs((np.array(signs) * deltas).mean()) >= obs - 1e-12:
            count += 1
    return count / float(2 ** n)


def main() -> None:
    logger = TeeLogger(log_dir=os.path.join(PROJECT_ROOT, "logs", "runs"), prefix="s3_9_shaping_extra_seeds")
    path = logger.start()
    try:
        t_wall0, cpu0 = time.time(), time.process_time()
        print("S3.9 Category B step 4: reward-shaping ablation extra seeds", NEW_SEEDS, "| log:", path)
        print("Condition (matches S3.7 exactly): 0% jamming, 0 NFZ,", CONDITION)
        hw("PRE-RUN", t_wall0, cpu0)

        new_rows: List[Dict[str, Any]] = []
        for shaping in (True, False):
            mode = "ON" if shaping else "OFF"
            orig_i, orig_m = src.ippo_train.compute_reward, src.mappo_train.compute_reward
            src.ippo_train.compute_reward = src.mappo_train.compute_reward = wrapper_for(shaping)
            try:
                for algo in ALGOS:
                    for s in NEW_SEEDS:
                        ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "s3_9_shaping_extra", mode.lower(), algo, f"seed_{s}")
                        os.makedirs(ckpt, exist_ok=True)
                        print(f"\nTraining {algo.upper()} | seed {s} | shaping {mode}")
                        t0 = time.time()
                        rec, agent = run_single_seed(algorithm=algo, seed=s, condition=CONDITION,
                                                     checkpoint_dir=ckpt, log_csv_path=None)
                        # evaluate with the real (unwrapped) env; evaluation does not use reward
                        ev = evaluate_agent(agent=agent, algorithm=algo, condition=dict(CONDITION), seed=s, n_episodes=10)
                        row = {"algo": algo, "seed": s, "shaping": shaping, "solved": bool(rec["success_achieved"]),
                               "iters": int(rec["final_iteration"]), "train_full": float(rec["full_success_rate"]),
                               "eval_full": float(ev["eval_full_success"]), "collisions": float(ev["eval_collisions"]),
                               "latency": float(ev["eval_steps_to_last_delivery"]), "energy": float(ev["eval_energy"]),
                               "source": "S3.9"}
                        new_rows.append(row)
                        print(f"  -> solved={row['solved']} iters={row['iters']} train_full={row['train_full']:.1%} "
                              f"eval_full={row['eval_full']:.1%} coll={row['collisions']:.1f} lat={row['latency']:.1f} "
                              f"({time.time() - t0:.1f}s)")
            finally:
                src.ippo_train.compute_reward, src.mappo_train.compute_reward = orig_i, orig_m
            hw(f"POST-SHAPING-{mode}", t_wall0, cpu0)

        rows = parse_s37() + new_rows
        csv_path = os.path.join(PROJECT_ROOT, "logs", "s3_9_shaping_combined_5seeds.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCombined rows (S3.7 seeds 0-1 parsed from log + S3.9 seeds 2-4) written to {csv_path}")

        # ---------------- tables ----------------
        def get(algo, seed, shaping):
            return next(r for r in rows if r["algo"] == algo and r["seed"] == seed and r["shaping"] == shaping)

        print("\n" + "=" * 128)
        print("PER-RUN TABLE (combined 5 seeds)")
        print("=" * 128)
        print(f"{'Algo':<6}|{'Seed':<5}|{'Src':<5}|{'Shaping':<8}|{'Solved':<7}|{'Iters':<6}|{'TrainFull':<10}|"
              f"{'EvalFull':<9}|{'Collisions':<11}|{'Latency':<8}|{'Energy':<7}")
        for algo in ALGOS:
            for s in range(5):
                for sh in (True, False):
                    r = get(algo, s, sh)
                    print(f"{algo.upper():<6}|{s:<5}|{r['source']:<5}|{'ON' if sh else 'OFF':<8}|{'Y' if r['solved'] else 'N':<7}|"
                          f"{r['iters']:<6}|{r['train_full']:<10.1%}|{r['eval_full']:<9.1%}|{r['collisions']:<11.1f}|"
                          f"{r['latency']:<8.1f}|{r['energy']:<7.1f}")

        print("\n" + "=" * 128)
        print("AGGREGATE: mean over seeds (OFF - ON deltas are paired per seed)")
        print("=" * 128)
        metrics = [("iters", "iters-to-bar", +1), ("eval_full", "eval full success", -1),
                   ("collisions", "eval collisions", +1), ("latency", "eval latency", +1)]
        print(f"{'Group':<12}{'Metric':<20}{'ON mean':>10}{'OFF mean':>10}{'mean(OFF-ON)':>14}{'pairs OFF worse':>17}")
        groups = [("IPPO", ["ippo"]), ("MAPPO", ["mappo"]), ("BOTH", ALGOS)]
        for gname, galgos in groups:
            for key, label, worse_sign in metrics:
                # "worse" = more iters/collisions/latency, or less success
                on = np.array([get(a, s, True)[key] for a in galgos for s in range(5)], dtype=float)
                off = np.array([get(a, s, False)[key] for a in galgos for s in range(5)], dtype=float)
                d = off - on
                worse = int(np.sum(d * worse_sign > 0)) if worse_sign > 0 else int(np.sum(d < 0))
                print(f"{gname:<12}{label:<20}{on.mean():>10.2f}{off.mean():>10.2f}{d.mean():>+14.2f}{worse:>11d}/{len(d)}")

        # ---------------- verdict ----------------
        print("\n" + "=" * 128)
        print("VERDICT (pre-declared rule): metric = eval full success, delta = OFF - ON")
        print("=" * 128)
        pooled = np.array([get(a, s, False)["eval_full"] - get(a, s, True)["eval_full"] for a in ALGOS for s in range(5)])
        per_algo_mean = {a: float(np.mean([get(a, s, False)["eval_full"] - get(a, s, True)["eval_full"] for s in range(5)]))
                         for a in ALGOS}
        p = sign_flip_p(pooled)
        for a in ALGOS:
            ds = [get(a, s, False)["eval_full"] - get(a, s, True)["eval_full"] for s in range(5)]
            print(f"  {a.upper()}: deltas by seed {['%+.0f%%' % (100 * x) for x in ds]} mean {per_algo_mean[a]:+.1%}")
        print(f"  pooled mean delta {pooled.mean():+.1%} over {len(pooled)} pairs; "
              f"ON better in {int(np.sum(pooled < 0))}, OFF better in {int(np.sum(pooled > 0))}, ties {int(np.sum(pooled == 0))}")
        print(f"  exact paired sign-flip permutation test, two-sided p = {p:.4f}")
        agree = (per_algo_mean["ippo"] < 0) == (per_algo_mean["mappo"] < 0)
        if p < 0.05 and agree and pooled.mean() < 0:
            verdict = "ROBUST (shaping ON gives better final eval quality)"
        elif p < 0.05 and agree and pooled.mean() > 0:
            verdict = "REVERSED (shaping OFF gives better final eval quality)"
        else:
            verdict = "MIXED / INCONCLUSIVE at this N"
        print(f"  VERDICT: {verdict}")

        hw("FINAL", t_wall0, cpu0)
    finally:
        logger.stop()


if __name__ == "__main__":
    main()
