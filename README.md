# CONSTELLATION

Civilian delivery-drone swarm coordination under communication jamming: a from-scratch, CPU-only, 2D multi-agent reinforcement learning study.

## What this is, and what it isn't

**This is** a multi-agent RL research project. Five simulated delivery drones on a 20x20 grid learn to complete three deliveries while a scripted jammer randomly overrides their movement commands. The project implements PPO from scratch, compares two multi-agent variants (IPPO and MAPPO), and runs a jamming-intensity sweep plus three ablations, reporting the results as measured, including the inconclusive and negative ones.

**This is not** a flight simulator, a physics model, or a deployable system. The world is a discrete 2D grid with five movement actions. It makes no real-time, production, or deployment claims. The framing is civilian logistics only.

## Highlights

- **PPO written from scratch** in PyTorch (actor/critic, GAE, clipped objective), validated on CartPole (`gymnasium` is used for that validation only).
- **IPPO and MAPPO**: independent PPO versus PPO with a centralized critic over global state.
- **Curriculum learning** over jammer intensity, with an ablation of whether it helps.
- **Scripted jammer** (distance-based falloff, 0 to 70% intensity) and **moving no-fly zones** (deterministic patterns, one held-out pattern reserved).
- **Classical Planner (no coordination)** baseline: a rule-based, non-learning planner in which each drone greedily heads for its nearest undelivered point.
- **Full jamming-sweep evaluation**: 2 algorithms x 5 jamming levels x 5 seeds = 50 training runs.
- **Replay export (schema v1) and a self-contained replay viewer** with Top-down (Lite) and Isometric modes and a dark theme.

## Screenshots

<!-- The repository owner will supply the actual image files below (docs/images/ does not exist yet). -->

![Lite mode - Top-down replay with averaged results](docs/images/viewer-topdown.png)

*Lite mode: top-down replay of a MAPPO episode at 50% jamming, alongside the sweep results.*

![Isometric mode - dark theme](docs/images/viewer-isometric-dark.png)

*Isometric mode with the dark theme enabled.*

## How it works

**Environment** (`src/drone_world.py`). A 20x20 grid with a depot at (0, 0), 5 drones starting at the depot, and 3 delivery points placed by seeded rejection sampling (each at least 12 Manhattan distance from the depot and from each other). An episode ends when all deliveries are complete or the step cap is reached (250 steps in the sweep and ablations). A single jammer sits at the grid center; the probability that a drone's action is replaced by STAY is `intensity * max(0, 1 - manhattan_distance / falloff_radius)`, so jamming is strongest near the center.

**Actions.** Discrete, per drone: STAY, UP, DOWN, LEFT, RIGHT.

**Observations.** Each drone sees 7 local values: its own (x, y); the relative vector to its nearest undelivered point; the jam probability at its own cell; the fraction of deliveries remaining; and the fraction of episode time remaining. Drones do not observe each other. The MAPPO critic (training only) additionally sees a 14-value global state: all drone positions, delivery statuses, and the timestep.

**Reward** (`src/rewards.py`). Per drone: potential-based distance shaping toward the nearest undelivered point, a small per-step penalty (-0.01), a terminal reward (+10) for completing a delivery, and a small penalty (-0.1) for sharing a cell with another drone. Collisions are otherwise tracked as metrics only. No-fly zones do not appear in the reward (see Known limitations).

**Training.** PPO with a 64x64 tanh actor and critic, Adam (lr 3e-4), gamma 0.99, GAE lambda 0.95, clip 0.2, 4 epochs, minibatch 64. Each update uses a 410-step rollout across 5 drones (2,050 samples). In IPPO all drones share one actor and pool their data; MAPPO keeps the same actor but gives the critic the global state. A run counts as "solved" when full-delivery success reaches 80% over the last 20 training episodes (cap: 150 iterations).

**Evaluation.** After training, each run is evaluated on 10 episodes. Reported metrics: full-delivery success (all 3 deliveries completed), delivery fraction, collisions, latency (steps to last delivery; 250 if not finished), and energy (count of non-STAY actions). Sweep cells report mean and standard deviation across 5 seeds (0 to 4). **The sweep and ablations were run with no-fly zones disabled (0 zones).**

## Results

All numbers below are copied from files in `logs/`; none are estimated. Each cell is the mean across 5 seeds, with the standard deviation across seeds in parentheses. With 10 evaluation episodes per run and 5 seeds, the seed-to-seed spread is large, which shapes how every comparison below should be read.

### Jamming sweep, full-delivery success (`logs/combined_jamming_sweep_newreward.csv`)

| Jamming | IPPO | MAPPO |
|---|---|---|
| 0% | 76% (26%) | 86% (15%) |
| 10% | 68% (39%) | 72% (31%) |
| 30% | 68% (19%) | 74% (38%) |
| 50% | 74% (33%) | 80% (23%) |
| 70% | 70% (41%) | 60% (19%) |

Under the frozen reward, performance does not fall off monotonically with jamming intensity; the seed-to-seed spread is as large as any trend. Collision, latency and energy columns are in the same CSV.

### IPPO vs MAPPO

Source: `logs/combined_jamming_sweep_newreward.csv`; analysis in `scripts/analyze_s3_4h_ablation.py`.

**No consistent coordination advantage was found.** The success-rate differences above are all within one standard deviation of each other. The rankings are not stable: dropping seed 1 (the weakest seed, see below) flips the success-rate winner at 10% (to a tie) and at 50% (to IPPO), and flips several collision, latency and energy winners. The one partial exception is 30% jamming, where MAPPO leads on success in both views (74% vs 68% with all seeds; 90% vs 75% without seed 1), though that gap was not tested for significance and the standard deviations overlap in the 5-seed view. The earlier sweep under the original reward (`logs/combined_jamming_sweep_5seeds.csv`) also gave different winners at several levels.

### Ablation: reward shaping (`logs/s3_9_shaping_combined_5seeds.csv`)

Distance shaping ON vs OFF, 0% jamming, IPPO and MAPPO, seeds 0 to 4 (10 paired runs).

| Metric | Shaping ON | Shaping OFF |
|---|---|---|
| Iterations to reach the 80% bar (mean) | 11.3 | 34.3 |
| Eval full success (mean) | 81% | 64% |

**Convergence speed improves robustly**: shaping ON reached the bar sooner in all 10 of 10 pairs, and one OFF run (IPPO seed 4) never reached it within 150 iterations. **Final policy quality is statistically inconclusive at 5 seeds**: ON was better on eval success in 6 of 10 pairs, OFF in 4, and an exact paired sign-flip permutation test gave two-sided p = 0.28 (`logs/s3_ablation_summary.csv`). The supported claim is faster convergence, not a proven final-quality benefit.

### Ablation: curriculum (`logs/s3_ablation_summary.csv`, extracted from the S3.8 run log)

Curriculum (four jammer stages) vs training directly at 50% jamming; 2 algorithms x 2 seeds (4 runs per arm), evaluated at 50% jamming. This is a small ablation.

| Arm (mean of 4 runs) | Training iterations | Training time | Eval full success | Collisions / episode |
|---|---|---|---|---|
| Direct | 11.0 | 16.4 s | 57.5% | 57.9 |
| Curriculum | 35.5 | 44.5 s | 45.0% | 137.9 |

**Curriculum does not earn its place at 50% jamming in this setup**: it cost about 3x the training, produced more collisions, and gave no better outcome.

### Reward fix: co-location penalty (`logs/s3_ablation_summary.csv`, extracted from the S3.4e run log; `logs/combined_jamming_sweep_5seeds.csv` vs `logs/combined_jamming_sweep_newreward.csv`)

A small co-location penalty (-0.1) was added after early training showed a policy degrading in collisions. On a single seed at 50% jamming (10 episodes), it improved IPPO from 40% to 70% full success and MAPPO from 20% to 50%. When the full 50-run sweep was retrained under the new reward, the effect was **mixed, not a clean win**: some cells improved (for example IPPO at 50%: 66% to 74%; MAPPO at 50%: 58% to 80%) and others got worse (IPPO at 10%: 84% to 68%; MAPPO at 70%: 88% to 60%). We do not describe the fix as validated.

### Classical Planner (no coordination)

The baseline is implemented (`src/classical_planner.py`) and has a correctness check (`tests/check_classical_planner.py`), but no recorded head-to-head result table exists in `logs/`. **Placeholder: baseline-vs-learned comparison numbers to be added once a logged run exists.**

## Known limitations

These are documented scope boundaries and open findings, not hidden caveats.

- **No-fly zones are not part of the learning problem.** They are simulated and counted as a metric, but the 7-value observation contains no zone information and the reward has no zone term. Policies therefore have no learned avoidance, and generalization to unseen zone patterns cannot exist or be tested. This is a deliberate scope decision, neither a bug nor a planned feature (`docs/limitations.md`). The sweep and ablations ran with zones disabled.
- **The seed-1 anomaly is an open hypothesis, not a solved problem.** Seed 1 is the weakest seed: in a scan of existing runs, IPPO seed 1 had 4 of 9 runs with high training success but near-collapsed evaluation success (mean eval full success 34%), and MAPPO seed 1 had 2 of 9 (62%) (`logs/s3_ablation_summary.csv`). The working hypothesis is a generalization effect of training on a single layout, seen across both algorithms. It has not been investigated further.
- **Small samples.** 10 evaluation episodes per run and 2 to 5 seeds per condition. Most differences between conditions are within noise, which is why several results above are reported as inconclusive.
- **Simplified world.** A discrete 2D grid with a scripted, fixed-position jammer. Learned communication and an adaptive jammer are out of scope for this version.

## Tech stack

- Python 3.13, PyTorch (CPU-only build), NumPy, matplotlib
- `gymnasium`, used only for CartPole validation of the PPO core
- Pure HTML/JS viewer, no server or external services
- No paid services; $0 budget

Versions are pinned in `requirements.txt` (torch 2.14.1, numpy 2.5.4, matplotlib 3.11.2, gymnasium 1.4.0), as recorded during project setup on Python 3.13.14; the CPU-only PyTorch build was used.

## How to run

Run commands from the project root on a machine with Python 3.13. Training is CPU-only; a single run takes seconds to about a minute here, and the full 50-run sweep took about 17 minutes on the development machine (`logs/s3_ablation_summary.csv`).

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate            # Windows; use source .venv/bin/activate elsewhere
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu   # CPU-only build
pip install -r requirements.txt

# Train
python -m src.ippo_train
python -m src.mappo_train
python -m src.curriculum_train

# Multi-seed run
python -m src.multiseed_harness --algorithm ippo --seeds 0 1 2

# Full jamming sweep (50 runs; writes logs/*_jamming_sweep_newreward*.csv)
python -m tests.check_s3_4g_full_sweep_newreward

# Export a replay from a checkpoint, then rebuild the viewer
python src/replay_export.py --checkpoint <path.pt> --algo mappo --out replays/my_run.json --jammer 0.5
python scripts/build_lite_viewer.py replays/my_run.json
```

To view a replay, open `viewer/lite_viewer.html` by double-clicking it; it is self-contained and needs no server. A sample replay (`replays/s4_1_sample_mappo_jam50.json`) is used by default.

Note: checkpoints (`*.pt`) and raw run logs (`logs/runs/`) are git-ignored, so checkpoints must be produced by training locally. Numbers quoted from raw run logs are preserved in `logs/s3_ablation_summary.csv`.

## Project structure

```
src/        environment, PPO core, IPPO/MAPPO/curriculum training, rewards, metrics,
            classical planner, replay export
scripts/    ablation analysis scripts, viewer build script
tests/      verification and experiment scripts (check_*.py)
logs/       result CSVs and plots (logs/runs/ holds per-session text logs)
replays/    exported episode replays (JSON, schema v1)
viewer/     self-contained Lite / Isometric replay viewer
docs/       known limitations
```

## Roadmap

- **Phase 5 write-up**: pending. It will draw on this README's results and limitations.
- **Out of scope for this version (parked deliberately, not missing features)**: learned inter-agent communication, an adaptive jammer, and no-fly-zone perception.

## License and citation

   ## License

   This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.