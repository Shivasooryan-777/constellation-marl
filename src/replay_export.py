"""
replay_export.py - Export evaluation episodes from a saved checkpoint as a JSON replay for the Phase 4 viewer.

New, standalone script (no protected files modified). Loads only the actor from an IPPO or MAPPO checkpoint
(both store {"actor", "critic", "optimizer"} with the same Actor(7, 5, 64) architecture), rolls out episodes
in a DroneWorld using the same stochastic policy sampling as the existing evaluation code, and writes JSON.

Usage:
    python src/replay_export.py --checkpoint <path.pt> --algo ippo|mappo --out replays/<name>.json
        [--seed 0] [--episodes 1] [--jammer 0.0] [--nfz 3] [--max-steps 250]

Replay schema (schema_version 1) - all coordinates are [x, y], origin top-left, x right, y down:
{
  "schema_version": 1,
  "meta": {algo, checkpoint, created, grid_size, n_drones, n_delivery_points, max_steps, jammer_intensity,
           n_no_fly_zones, policy_sampling: "stochastic", torch_seed},
  "actions_legend": {"0": "STAY", "1": "UP", "2": "DOWN", "3": "LEFT", "4": "RIGHT"},
  "episodes": [
    {
      "episode_index": int, "world_seed": int,
      "depot": [x, y],
      "delivery_points": [[x, y], ...],
      "jammer": null | {"position": [x, y], "intensity": float, "falloff_radius": float},
      "no_fly_zone_meta": [{"zone_id": int, "pattern_id": str}, ...],
      "summary": {success, no_fly_zone_collisions, multi_drone_collisions, steps_to_last_delivery,
                  total_steps, energy},
      "frames": [   # frames[0] is the initial state (t=0, no actions); frames[k] is the state after step k
        {
          "t": int,
          "drones": [[x, y], ...],                 # position of each drone after this step
          "intended_actions": [int, ...] | null,   # policy output (null for frame 0)
          "executed_actions": [int, ...] | null,   # after jamming scramble (jammed drones -> 0/STAY)
          "jammed": [bool, ...] | null,            # intended != executed because of the jammer
          "jam_prob": [float, ...],                # jam probability at each drone's position BEFORE the step
          "no_fly_zones": [[x, y], ...],           # zone cells at this frame's t (order = no_fly_zone_meta)
          "newly_delivered": [[x, y], ...],        # delivery points completed on this step
          "delivered": [[x, y], ...],              # all delivery points completed so far
          "multi_drone_collision_cells": [[x, y], ...],  # cells with >=2 drones this step
          "nfz_collision_drones": [int, ...]       # drone ids standing on a zone cell this step
        }, ...
      ]
    }
  ]
}
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time
from typing import Any, Dict, List

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.drone_world import DroneWorld
from src.metrics import MetricsTracker
from src.ppo_core import Actor

SCHEMA_VERSION = 1


def _xy(p) -> List[int]:
    return [int(p[0]), int(p[1])]


def load_actor(checkpoint_path: str) -> Actor:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    actor = Actor(7, 5, 64)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


def _jam_probs(world: DroneWorld) -> List[float]:
    if world.jammer is None:
        return [0.0 for _ in world.drones]
    return [float(world.jammer.jam_probability(p)) for p in world.drones]


def _initial_frame(world: DroneWorld) -> Dict[str, Any]:
    return {
        "t": world.t,
        "drones": [_xy(p) for p in world.drones],
        "intended_actions": None,
        "executed_actions": None,
        "jammed": None,
        "jam_prob": _jam_probs(world),
        "no_fly_zones": [_xy(z.position_at(world.t)) for z in world.no_fly_zones],
        "newly_delivered": [],
        "delivered": [],
        "multi_drone_collision_cells": [],
        "nfz_collision_drones": [],
    }


def rollout_episode(actor: Actor, world_seed: int, episode_index: int, grid_size: int, n_drones: int,
                    n_delivery_points: int, jammer_intensity: float, n_no_fly_zones: int,
                    max_steps: int) -> Dict[str, Any]:
    world = DroneWorld(
        grid_size=grid_size, n_drones=n_drones, n_delivery_points=n_delivery_points, seed=world_seed,
        n_no_fly_zones=n_no_fly_zones, include_held_out=False, jammer_intensity=jammer_intensity,
        max_steps=max_steps,
    )
    tracker = MetricsTracker(world=world)
    frames = [_initial_frame(world)]

    # Same loop shape as the existing evaluation code (S3.6): stop at max_steps or when all delivered.
    while tracker.total_steps < max_steps and len(world.remaining_deliveries) > 0:
        jam_prob = _jam_probs(world)
        intended: List[int] = []
        for d in range(n_drones):
            obs_t = torch.tensor(world.get_observation(d), dtype=torch.float32)
            with torch.no_grad():
                act_t, _, _ = actor.get_action_and_log_prob(obs_t)
            intended.append(int(act_t.item()))

        executed = list(intended)  # world.step() rewrites this list in place with post-jam actions
        with contextlib.redirect_stdout(io.StringIO()):  # silence per-step NFZ warnings
            tracker.record_step(executed)

        cell_counts: Dict[Any, int] = {}
        for p in world.drones:
            cell_counts[p] = cell_counts.get(p, 0) + 1
        zones_now = [z.position_at(world.t) for z in world.no_fly_zones]
        frames.append({
            "t": world.t,
            "drones": [_xy(p) for p in world.drones],
            "intended_actions": intended,
            "executed_actions": [int(a) for a in executed],
            "jammed": [i != e for i, e in zip(intended, executed)],
            "jam_prob": jam_prob,
            "no_fly_zones": [_xy(z) for z in zones_now],
            "newly_delivered": [_xy(p) for p in sorted(world.just_delivered)],
            "delivered": [_xy(p) for p in sorted(world.delivered)],
            "multi_drone_collision_cells": [_xy(c) for c, n in cell_counts.items() if n >= 2],
            "nfz_collision_drones": [i for i, p in enumerate(world.drones) if p in zones_now],
        })

    return {
        "episode_index": episode_index,
        "world_seed": world_seed,
        "depot": _xy(world.depot),
        "delivery_points": [_xy(p) for p in world.delivery_points],
        "jammer": None if world.jammer is None else {
            "position": _xy(world.jammer.position),
            "intensity": float(world.jammer.intensity),
            "falloff_radius": float(world.jammer.falloff_radius),
        },
        "no_fly_zone_meta": [{"zone_id": z.zone_id, "pattern_id": z.pattern_id} for z in world.no_fly_zones],
        "summary": tracker.summary(),
        "frames": frames,
    }


def export_replay(checkpoint: str, algo: str, out_path: str, seed: int = 0, episodes: int = 1,
                  jammer: float = 0.0, nfz: int = 3, max_steps: int = 250, grid_size: int = 20,
                  n_drones: int = 5, n_delivery_points: int = 3) -> Dict[str, Any]:
    torch.manual_seed(seed)
    actor = load_actor(checkpoint)
    replay = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "algo": algo, "checkpoint": checkpoint.replace("\\", "/"), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "grid_size": grid_size, "n_drones": n_drones, "n_delivery_points": n_delivery_points,
            "max_steps": max_steps, "jammer_intensity": jammer, "n_no_fly_zones": nfz,
            "policy_sampling": "stochastic", "torch_seed": seed,
        },
        "actions_legend": {"0": "STAY", "1": "UP", "2": "DOWN", "3": "LEFT", "4": "RIGHT"},
        "episodes": [
            rollout_episode(actor, seed * 1000 + i, i, grid_size, n_drones, n_delivery_points, jammer, nfz, max_steps)
            for i in range(episodes)
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(replay, f, separators=(",", ":"))
    return replay


def main() -> None:
    ap = argparse.ArgumentParser(description="Export evaluation episodes from a checkpoint as a JSON replay.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--algo", required=True, choices=["ippo", "mappo"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--jammer", type=float, default=0.0)
    ap.add_argument("--nfz", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=250)
    a = ap.parse_args()
    r = export_replay(a.checkpoint, a.algo, a.out, a.seed, a.episodes, a.jammer, a.nfz, a.max_steps)
    for ep in r["episodes"]:
        print(f"episode {ep['episode_index']}: {len(ep['frames']) - 1} steps, summary={ep['summary']}")
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
