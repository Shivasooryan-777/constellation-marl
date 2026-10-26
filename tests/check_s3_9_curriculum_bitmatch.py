"""
tests/check_s3_9_curriculum_bitmatch.py - S3.9 Step 3: numeric bit-match for S3.8a.

Runs the canonical src/curriculum_train.py::train_curriculum (MAPPO) and the test-side
tests/check_s3_8_curriculum_ablation.py::train_curriculum_ippo (IPPO) on identical seeds and compares
per-step quantities for exact equality (==, no tolerance).

What can and cannot be bit-matched (stated up front):
- Both runners build their Actor first, right after torch.manual_seed(seed), with the same Actor(7, 5, 64)
  architecture, so initial actor weights are identical. Critic calls consume no RNG. Therefore, in the FIRST
  rollout (before any update), the sampled actions, log-probs, observations, jam draws, drone positions,
  rewards and episode boundaries must be bit-identical between the two runners. That is what is checked here
  (per-step, all 410 steps x 5 drones, including episode resets).
- After the first update the two runs legitimately diverge: MAPPO's critic takes the 14-d global state and
  IPPO's the 7-d local obs, so values/advantages/losses differ by design. Loss values are printed as
  informational only and are NOT compared.
- Stage warm-start / stability logic is plain Python that does not depend on the algorithm; it is not
  exercised numerically here (1 iteration per run) and remains verified by code comparison only.

Protected files are not modified. Spies wrap DroneWorld.step, compute_reward and Actor.get_action_and_log_prob
via monkeypatching inside this test only. Output checkpoints go to checkpoints/s3_9_bitmatch/ (new path).
"""

import importlib.util
import os
import sys
import time
from typing import Any, Dict, List

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.tee_logger import TeeLogger
import src.curriculum_train as ct
import src.drone_world as dw
import src.ppo_core as ppo_core
from src.ippo_train import get_system_memory, get_process_memory_mb

_spec = importlib.util.spec_from_file_location(
    "s3_8_mod", os.path.join(PROJECT_ROOT, "tests", "check_s3_8_curriculum_ablation.py"))
s38 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s38)

ROLLOUT_STEPS = 410
CONFIGS = [(0, 0.5), (1, 0.5), (0, 0.0)]  # (seed, jammer intensity); one stage, one iteration each


class Spy:
    def __init__(self, align_rng_seed: Any = None) -> None:
        self.events: List[Any] = []
        self.align_rng_seed = align_rng_seed

    def install(self) -> None:
        spy = self
        self._orig_critic_init = ppo_core.Critic.__init__
        self._orig_step = dw.DroneWorld.step
        self._orig_act = ppo_core.Actor.get_action_and_log_prob
        self._orig_rew_ct = ct.compute_reward
        self._orig_rew_s38 = s38.compute_reward

        def step(world, actions):
            pre = list(actions)
            status = spy._orig_step(world, actions)
            spy.events.append(("step", world.t, tuple(pre), tuple(actions), tuple(world.drones),
                               tuple(sorted(world.delivered))))
            return status

        def act(actor, state, action=None):
            a, lp, ent = spy._orig_act(actor, state, action)
            if action is None and state.dim() == 1:  # rollout sampling only (updates pass minibatches)
                spy.events.append(("act", tuple(state.tolist()), int(a.item()), float(lp.item())))
            return a, lp, ent

        def wrap_rew(orig):
            def rew(world, prev_distances, d):
                r = orig(world, prev_distances, d)
                spy.events.append(("rew", d, r))
                return r
            return rew

        def critic_init(critic, *a, **kw):
            spy._orig_critic_init(critic, *a, **kw)
            # Critic construction consumes a dimension-dependent amount of RNG (14-d MAPPO vs 7-d IPPO input),
            # which offsets the later action-sampling stream. Actor weights are already built (identical).
            # Optionally re-seed here so both runners sample from the same RNG state.
            if spy.align_rng_seed is not None:
                torch.manual_seed(spy.align_rng_seed)

        ppo_core.Critic.__init__ = critic_init
        dw.DroneWorld.step = step
        ppo_core.Actor.get_action_and_log_prob = act
        ct.compute_reward = wrap_rew(self._orig_rew_ct)
        s38.compute_reward = wrap_rew(self._orig_rew_s38)

    def uninstall(self) -> None:
        ppo_core.Critic.__init__ = self._orig_critic_init
        dw.DroneWorld.step = self._orig_step
        ppo_core.Actor.get_action_and_log_prob = self._orig_act
        ct.compute_reward = self._orig_rew_ct
        s38.compute_reward = self._orig_rew_s38


def run(kind: str, seed: int, intensity: float, align: bool):
    spy = Spy(align_rng_seed=(seed + 777) if align else None)
    spy.install()
    ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "s3_9_bitmatch", kind, f"seed_{seed}_jam_{intensity}_{'aligned' if align else 'raw'}")
    try:
        if kind == "mappo":
            logs, summaries, _ = ct.train_curriculum(
                seed=seed, stages=[intensity], stage_max_iterations=1, rollout_world_steps=ROLLOUT_STEPS,
                checkpoint_dir=ckpt, log_csv_path=None, plot_png_path=None)
        else:
            logs, summaries, _ = s38.train_curriculum_ippo(
                seed=seed, stages=[intensity], stage_max_iterations=1, rollout_world_steps=ROLLOUT_STEPS,
                checkpoint_dir=ckpt, log_csv_path=None)
    finally:
        spy.uninstall()
    # keep only rollout-phase events: the first ROLLOUT_STEPS env steps (drop anything after, e.g. GAE bootstrap)
    return spy.events, logs[0]

def cut(ev: List[Any]) -> List[Any]:
    """Events up to and including the ROLLOUT_STEPS-th env step (and its trailing reward events).
    Excludes the GAE bootstrap value call and everything the update does."""
    n = 0
    for k, e in enumerate(ev):
        if e[0] == "step":
            n += 1
            if n == ROLLOUT_STEPS:
                j = k + 1
                while j < len(ev) and ev[j][0] == "rew":
                    j += 1
                return ev[:j]
    raise AssertionError("fewer than 410 steps recorded")


def compare(seed: int, intensity: float, align: bool) -> bool:
    ev_m, log_m = run("mappo", seed, intensity, align)
    ev_i, log_i = run("ippo", seed, intensity, align)
    cm, ci = cut(ev_m), cut(ev_i)
    n_steps = sum(1 for e in cm if e[0] == "step")
    n_act = sum(1 for e in cm if e[0] == "act")
    n_rew = sum(1 for e in cm if e[0] == "rew")
    first_diff = next((k for k, (a, b) in enumerate(zip(cm, ci)) if a != b), None)
    ev_equal = (len(cm) == len(ci)) and first_diff is None

    # Iteration-1 log fields that derive only from the rollout (not from the update or wall-clock).
    fields = ["stage", "jammer_intensity", "stage_iteration", "avg_reward", "avg_success_rate",
              "full_success_rate", "avg_collisions", "stage_completed_episodes", "stable_streak"]
    log_equal = all(log_m[f] == log_i[f] for f in fields)

    ok = ev_equal and log_equal
    print(f"\n[seed={seed} jam={intensity}] {'PASS' if ok else 'FAIL'}")
    print(f"  compared: {n_steps} env steps, {n_act} sampled (obs, action, log_prob) records, "
          f"{n_rew} per-drone rewards -> event streams identical: {ev_equal}"
          + (f" (first diff at event {first_diff}: {cm[first_diff]} vs {ci[first_diff]})" if first_diff is not None else ""))
    print(f"  iteration-1 rollout-derived log fields identical: {log_equal} "
          f"(episodes completed={log_m['stage_completed_episodes']}, avg_reward={log_m['avg_reward']:.4f}, "
          f"full_success={log_m['full_success_rate']:.2f})")
    print(f"  informational (expected to differ, different critic): policy_loss MAPPO={log_m['policy_loss']:+.5f} "
          f"IPPO={log_i['policy_loss']:+.5f} | value_loss MAPPO={log_m['value_loss']:.4f} IPPO={log_i['value_loss']:.4f}")
    return ok


def main() -> None:
    logger = TeeLogger(log_dir=os.path.join(PROJECT_ROOT, "logs", "runs"), prefix="s3_9_curriculum_bitmatch")
    path = logger.start()
    try:
        t0 = time.time()
        tot, used, pct = get_system_memory()
        print(f"S3.9 Step 3 - S3.8a numeric bit-match | log: {path}")
        print(f"PRE-RUN: RAM {used:.2f}/{tot:.2f} GB ({pct:.1f}%), process {get_process_memory_mb():.1f} MB, "
              f"{time.strftime('%Y-%m-%d %H:%M:%S')}")

        mode_results = {}
        for align in (False, True):
            print("\n" + "=" * 100)
            print("MODE: " + ("RAW (production RNG streams, no realignment) - informational, expected to FAIL on RNG offset"
                              if not align else
                              "RNG-ALIGNED (test-only: torch RNG re-seeded right after Critic construction) - the actual check"))
            print("=" * 100)
            mode_results[align] = all([compare(s, j, align) for s, j in CONFIGS])

        tot, used, pct = get_system_memory()
        print(f"\nPOST-RUN: RAM {used:.2f}/{tot:.2f} GB ({pct:.1f}%), process {get_process_memory_mb():.1f} MB, "
              f"elapsed {time.time() - t0:.1f}s")
        print(f"\nRAW mode (informational): {'PASS' if mode_results[False] else 'FAIL'}")
        print(f"OVERALL S3.8a NUMERIC BIT-MATCH (RNG-aligned, rollout mechanics, iteration 1): "
              f"{'PASS' if mode_results[True] else 'FAIL'}")
    finally:
        logger.stop()
    sys.exit(0 if mode_results[True] else 1)


if __name__ == "__main__":
    main()
