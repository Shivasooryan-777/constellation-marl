"""
tests/check_s3_9_replay_export.py - S3.9 Step 2: end-to-end check of src/replay_export.py on real checkpoints
(one IPPO, one MAPPO). Verifies schema, frame consistency, and that frame-derived counts reconcile with the
MetricsTracker summary stored in the replay. Writes only to replays/.
"""

import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.replay_export import export_replay

CASES = [
    ("ippo", "checkpoints/ippo_dronworld/ippo_iteration_11_solved.pt", "replays/s3_9_check_ippo.json"),
    ("mappo", "checkpoints/mappo_droneworld/mappo_iteration_10_solved.pt", "replays/s3_9_check_mappo.json"),
]

FRAME_KEYS = {"t", "drones", "intended_actions", "executed_actions", "jammed", "jam_prob", "no_fly_zones",
              "newly_delivered", "delivered", "multi_drone_collision_cells", "nfz_collision_drones"}


def check(algo: str, ckpt: str, out: str) -> None:
    export_replay(os.path.join(PROJECT_ROOT, ckpt), algo, os.path.join(PROJECT_ROOT, out),
                  seed=0, episodes=3, jammer=0.5, nfz=3)
    with open(os.path.join(PROJECT_ROOT, out)) as f:
        r = json.load(f)  # also proves the file is valid JSON
    assert r["schema_version"] == 1 and len(r["episodes"]) == 3
    for ep in r["episodes"]:
        frames = ep["frames"]
        assert frames[0]["t"] == 0 and frames[0]["intended_actions"] is None
        n = len(ep["no_fly_zone_meta"])
        multi = nfz = energy = 0
        for k, fr in enumerate(frames):
            assert set(fr) == FRAME_KEYS, f"frame keys mismatch at {k}"
            assert fr["t"] == k and len(fr["drones"]) == 5 and len(fr["no_fly_zones"]) == n
            if k > 0:
                multi += len(fr["multi_drone_collision_cells"])
                nfz += len(fr["nfz_collision_drones"])
                energy += sum(1 for a in fr["executed_actions"] if a != 0)
                assert fr["jammed"] == [i != e for i, e in zip(fr["intended_actions"], fr["executed_actions"])]
                assert all(e == 0 for e, j in zip(fr["executed_actions"], fr["jammed"]) if j)
        s = ep["summary"]
        assert multi == s["multi_drone_collisions"], (multi, s)
        assert nfz == s["no_fly_zone_collisions"], (nfz, s)
        assert energy == s["energy"], (energy, s)
        assert len(frames) - 1 == s["total_steps"]
        assert len(frames[-1]["delivered"]) == round(s["success"] * len(ep["delivery_points"]))
    jam_steps = sum(any(fr["jammed"]) for ep in r["episodes"] for fr in ep["frames"][1:])
    print(f"[PASS] {algo}: 3 episodes, summaries reconcile with frames; steps with >=1 jammed drone: {jam_steps}")


if __name__ == "__main__":
    for c in CASES:
        check(*c)
    print("ALL PASS")
