"""
tests/check_trajectory_inspection.py - S3.4d Trajectory Inspection.

Deep-dive trajectory and position inspection comparing IPPO Seed 0
Original Checkpoint (Iter 11) vs. Continued Checkpoint (Iter 50) at 50% jamming.

Inspects:
- Step-by-step positions and movement of all 5 drones
- Exact delivery events (which target, step t, which drone)
- Spatial and temporal distribution of collision (co-location) events
- Oscillation patterns, jammer-zone traps, and flocking/swarming dynamics
- Direct empirical verdict on "pathological clustering" hypothesis
"""

import os
import sys
import time
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import Counter
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.drone_world import DroneWorld, STAY, UP, DOWN, LEFT, RIGHT
from src.metrics import MetricsTracker
from src.ppo_core import PPOAgent


ACTION_NAMES = {STAY: "STAY", UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"}


def load_agent(checkpoint_path: str) -> PPOAgent:
    """Loads PPOAgent from checkpoint."""
    agent = PPOAgent(
        state_dim=7, action_dim=5, hidden_dim=64, lr=3e-4, gamma=0.99, gae_lambda=0.95
    )
    agent.load_checkpoint(checkpoint_path)
    return agent


def run_detailed_rollout(
    agent: PPOAgent,
    ep_seed: int,
    grid_size: int = 20,
    n_drones: int = 5,
    n_delivery_points: int = 3,
    max_steps: int = 250,
    jammer_intensity: float = 0.5,
) -> Dict[str, Any]:
    """
    Executes a single evaluation episode with full per-step telemetry.
    """
    torch.manual_seed(ep_seed)
    np.random.seed(ep_seed)

    world = DroneWorld(
        grid_size=grid_size,
        n_drones=n_drones,
        n_delivery_points=n_delivery_points,
        seed=ep_seed,
        n_no_fly_zones=0,
        jammer_intensity=jammer_intensity,
        max_steps=max_steps,
    )
    tracker = MetricsTracker(world=world)

    history: List[Dict[str, Any]] = []
    deliveries_logged: List[Dict[str, Any]] = []
    co_location_cells: Counter = Counter()
    max_colocated_drones: int = 1
    total_colocation_steps: int = 0
    jammer_center = (grid_size // 2, grid_size // 2)

    prev_delivered_set: Set[Tuple[int, int]] = set()

    for step in range(max_steps):
        if len(world.remaining_deliveries) == 0:
            break

        # Observations and actions
        obs_list = []
        actions = []
        for d in range(n_drones):
            obs_d = world.get_observation(d)
            obs_t = torch.tensor(obs_d, dtype=torch.float32)
            with torch.no_grad():
                act_t, _, _ = agent.actor.get_action_and_log_prob(obs_t)
            obs_list.append(obs_d)
            actions.append(int(act_t.item()))

        # Prior positions
        drones_before = list(world.drones)

        # Step environment
        _ = tracker.record_step(actions)

        # Post positions
        drones_after = list(world.drones)

        # Check co-location on this step
        cell_counts = Counter(drones_after)
        has_colocation = any(c >= 2 for c in cell_counts.values())
        if has_colocation:
            total_colocation_steps += 1
            for pos, count in cell_counts.items():
                if count >= 2:
                    co_location_cells[pos] += 1
                    if count > max_colocated_drones:
                        max_colocated_drones = count

        # Check new delivery events
        current_delivered = set(world.delivered)
        new_deliveries = current_delivered - prev_delivered_set
        for p in new_deliveries:
            delivering_drones = [d for d in range(n_drones) if drones_after[d] == p]
            deliveries_logged.append({
                "target": p,
                "step": step + 1,
                "drones": delivering_drones,
            })
        prev_delivered_set = current_delivered

        history.append({
            "step": step + 1,
            "drones": drones_after,
            "actions": actions,
            "cell_counts": dict(cell_counts),
            "remaining": list(world.remaining_deliveries),
        })

    summary = tracker.summary()
    return {
        "ep_seed": ep_seed,
        "delivery_points": list(world.delivery_points),
        "deliveries_completed": len(world.delivered),
        "total_deliveries": len(world.delivery_points),
        "full_success": bool(len(world.delivered) == len(world.delivery_points)),
        "total_steps": tracker.total_steps,
        "co_location_steps": total_colocation_steps,
        "max_colocated_drones": max_colocated_drones,
        "co_location_cell_counts": co_location_cells,
        "delivery_events": deliveries_logged,
        "history": history,
        "summary": summary,
    }


def analyze_policy_rollouts(agent: PPOAgent, label: str, n_episodes: int = 5) -> List[Dict[str, Any]]:
    print("=" * 100)
    print(f"TELEMETRY ROLLOUTS FOR: {label.upper()} (N = {n_episodes} EPISODES)")
    print("=" * 100)

    ep_results = []
    for ep_idx in range(n_episodes):
        res = run_detailed_rollout(agent, ep_seed=ep_idx)
        ep_results.append(res)

        print(f"\n--- Episode {ep_idx} (Seed {ep_idx}) ---")
        print(f"  Delivery Points:      {res['delivery_points']}")
        print(f"  Deliveries Completed: {res['deliveries_completed']}/{res['total_deliveries']} "
              f"({'FULL SUCCESS' if res['full_success'] else 'FAILED'}) in {res['total_steps']} steps")
        print(f"  Delivery Events:      {res['delivery_events']}")
        print(f"  Co-location Steps:    {res['co_location_steps']}/{res['total_steps']} "
              f"({res['co_location_steps'] / res['total_steps']:.1%}) | Max Drones on 1 Cell: {res['max_colocated_drones']}")

        # Top collision locations
        top_cells = res['co_location_cell_counts'].most_common(5)
        top_cells_str = ", ".join(f"{pos}: {cnt} steps" for pos, cnt in top_cells)
        print(f"  Top Co-located Cells: {top_cells_str if top_cells else 'None'}")

        # Sample snapshot trace every 25 steps
        print(f"  Trajectory Snapshots (Step: D0, D1, D2, D3, D4 | Remaining Targets):")
        sample_steps = [1, 20, 50, 75, 100, 150, 200, res['total_steps']]
        sample_steps = sorted(list(set(s for s in sample_steps if s <= res['total_steps'])))
        for s in sample_steps:
            h = res['history'][s - 1]
            d_pos_str = " ".join(f"D{i}:{h['drones'][i]}" for i in range(5))
            # Check if any cells have duplicates
            dups = [pos for pos, cnt in h['cell_counts'].items() if cnt >= 2]
            dup_str = f" [CO-LOCATED at {dups}]" if dups else ""
            print(f"    t={s:3d}: {d_pos_str} | Remaining: {h['remaining']}{dup_str}")

    return ep_results


def compare_diagnostics(orig_results: List[Dict[str, Any]], cont_results: List[Dict[str, Any]]):
    print("\n" + "=" * 100)
    print("COMPARATIVE TELEMETRY SYNTHESIS: ORIGINAL (ITER 11) vs. CONTINUED (ITER 50)")
    print("=" * 100)

    n_eps = len(orig_results)

    # Episode-by-episode comparison
    print(f"\n{'Ep':<4} | {'Targets Reached (Orig)':<24} | {'Targets Reached (Cont)':<24} | {'Steps (O/C)':<14} | {'Coloc Steps (O/C)':<18} | {'Max Co-located (O/C)':<20}")
    print("-" * 114)
    for i in range(n_eps):
        o = orig_results[i]
        c = cont_results[i]
        o_succ = f"{o['deliveries_completed']}/{o['total_deliveries']} ({'DONE' if o['full_success'] else 'FAIL'})"
        c_succ = f"{c['deliveries_completed']}/{c['total_deliveries']} ({'DONE' if c['full_success'] else 'FAIL'})"
        steps = f"{o['total_steps']:3d} / {c['total_steps']:3d}"
        coloc = f"{o['co_location_steps']:3d} / {c['co_location_steps']:3d}"
        max_d = f"{o['max_colocated_drones']} drones / {c['max_colocated_drones']} drones"
        print(f"{i:<4d} | {o_succ:<24} | {c_succ:<24} | {steps:<14} | {coloc:<18} | {max_d:<20}")
    print("-" * 114)

    # Spatial clustering breakdown
    def categorize_cells(cell_counts: Counter, delivery_points: List[Tuple[int, int]]) -> Dict[str, int]:
        depot = (0, 0)
        jammer_center = (10, 10)
        deliv_set = set(delivery_points)
        cat = {"depot": 0, "delivery_target": 0, "jammer_core": 0, "transit_other": 0}
        for pos, cnt in cell_counts.items():
            if pos == depot:
                cat["depot"] += cnt
            elif pos in deliv_set:
                cat["delivery_target"] += cnt
            elif abs(pos[0] - jammer_center[0]) + abs(pos[1] - jammer_center[1]) <= 3:
                cat["jammer_core"] += cnt
            else:
                cat["transit_other"] += cnt
        return cat

    orig_total_cells = Counter()
    cont_total_cells = Counter()
    for o in orig_results:
        orig_total_cells.update(o['co_location_cell_counts'])
    for c in cont_results:
        cont_total_cells.update(c['co_location_cell_counts'])

    # All delivery points across the 5 episodes
    all_deliv = [p for o in orig_results for p in o['delivery_points']]
    orig_cat = categorize_cells(orig_total_cells, all_deliv)
    cont_cat = categorize_cells(cont_total_cells, all_deliv)

    print("\nSPATIAL BREAKDOWN OF CO-LOCATION TIMESTEPS (Total Across 5 Episodes):")
    print(f"  {'Category':<25} | {'Original (Iter 11)':<20} | {'Continued (Iter 50)':<20} | {'Change':<10}")
    print("  " + "-" * 80)
    for k in ["depot", "delivery_target", "jammer_core", "transit_other"]:
        ov = orig_cat[k]
        cv = cont_cat[k]
        print(f"  {k:<25} | {ov:<20d} | {cv:<20d} | {cv - ov:+10d}")
    print("  " + "-" * 80)

    # Movement diversity & oscillation analysis
    def compute_movement_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
        total_steps_all = 0
        stay_count = 0
        reversals = 0  # Action immediately reversed (e.g. UP then DOWN, LEFT then RIGHT)
        opposite = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

        for res in results:
            for d in range(5):
                d_actions = [h['actions'][d] for h in res['history']]
                total_steps_all += len(d_actions)
                stay_count += sum(1 for a in d_actions if a == STAY)
                for t in range(1, len(d_actions)):
                    prev_a = d_actions[t - 1]
                    curr_a = d_actions[t]
                    if prev_a in opposite and curr_a == opposite[prev_a]:
                        reversals += 1

        return {
            "stay_fraction": stay_count / total_steps_all if total_steps_all > 0 else 0.0,
            "reversal_fraction": reversals / total_steps_all if total_steps_all > 0 else 0.0,
        }

    orig_mov = compute_movement_metrics(orig_results)
    cont_mov = compute_movement_metrics(cont_results)

    print("\nDRONE BEHAVIORAL PATTERN METRICS:")
    print(f"  {'Metric':<35} | {'Original (Iter 11)':<20} | {'Continued (Iter 50)':<20}")
    print("  " + "-" * 80)
    print(f"  {'STAY Action Fraction':<35} | {orig_mov['stay_fraction']:<20.1%} | {cont_mov['stay_fraction']:<20.1%}")
    print(f"  {'Oscillation Reversal Fraction':<35} | {orig_mov['reversal_fraction']:<20.1%} | {cont_mov['reversal_fraction']:<20.1%}")
    print("  " + "-" * 80 + "\n")


def main():
    orig_ckpt = os.path.join(
        PROJECT_ROOT, "checkpoints", "multiseed_ippo_sweep", "jam_50", "seed_0", "ippo_iteration_11_solved.pt"
    )
    cont_ckpt = os.path.join(
        PROJECT_ROOT, "checkpoints", "convergence_check", "ippo_seed_0_continued.pt"
    )

    assert os.path.exists(orig_ckpt), f"Missing original checkpoint: {orig_ckpt}"
    assert os.path.exists(cont_ckpt), f"Missing continued checkpoint: {cont_ckpt}"

    print("================================================================================")
    print("CONSTELLATION — SESSION S3.4d: TRAJECTORY & TELEMETRY INSPECTION")
    print("Comparing IPPO Seed 0: Original (Iter 11) vs. Continued (Iter 50)")
    print(f"Original Checkpoint:  {orig_ckpt}")
    print(f"Continued Checkpoint: {cont_ckpt}")
    print("================================================================================\n")

    agent_orig = load_agent(orig_ckpt)
    agent_cont = load_agent(cont_ckpt)

    n_eval_episodes = 5
    orig_results = analyze_policy_rollouts(agent_orig, label="Original Policy (Iter 11)", n_episodes=n_eval_episodes)
    cont_results = analyze_policy_rollouts(agent_cont, label="Continued Policy (Iter 50)", n_episodes=n_eval_episodes)

    compare_diagnostics(orig_results, cont_results)


if __name__ == "__main__":
    main()
