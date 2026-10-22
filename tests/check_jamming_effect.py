"""
tests/check_jamming_effect.py - Verification test for Session S2.6: Jamming as Action Unreliability.

Verifications:
1. (Part A) Trajectory Comparison:
   - High-jamming drone (placed at (9, 10), adjacent to jammer at (10, 10)) vs.
     Low-jamming drone (placed at (0, 0), outside the falloff radius).
   - Both choose RIGHT (action 4) for 15 consecutive steps.
   - High-jamming drone has commands scrambled to STAY; actual trajectory deviates noticeably.
   - Low-jamming drone has 0 scrambles; trajectory matches "RIGHT every step" exactly.
2. (Part B) Seeded Determinism:
   - Two separate DroneWorld instances with the same seed (seed=42) and same actions.
   - Confirms that the step-by-step scramble decisions and positions are 100% identical.
3. (Part C) MetricsTracker Energy Accounting:
   - Confirms that MetricsTracker energy counts only effectively executed non-STAY actions,
     ignoring scrambled-to-STAY actions (actual energy < chosen non-STAY actions).
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.drone_world import DroneWorld, RIGHT, STAY
from src.metrics import MetricsTracker


def test_part_a_trajectory_comparison():
    print("================================================================================")
    print("TEST PART A: High-Jamming vs. Low-Jamming Trajectory Comparison")
    print("================================================================================")
    seed = 42
    n_steps = 15
    intensity = 0.7

    # 1. High-Jamming Environment: Drone 0 starts at (9, 10), adjacent to Jammer at (10, 10)
    world_high = DroneWorld(seed=seed, jammer_intensity=intensity, n_no_fly_zones=0)
    world_high.drones[0] = (9, 10)
    high_start_pos = world_high.drones[0]
    high_jammer_pos = world_high.jammer.position
    high_initial_prob = world_high.jammer.jam_probability(high_start_pos)

    # 2. Low-Jamming Environment: Drone 0 starts at (0, 0), far from Jammer at (10, 10)
    world_low = DroneWorld(seed=seed, jammer_intensity=intensity, n_no_fly_zones=0)
    world_low.drones[0] = (0, 0)
    low_start_pos = world_low.drones[0]
    low_initial_prob = world_low.jammer.jam_probability(low_start_pos)

    print(f"Jammer Configuration: Center={high_jammer_pos}, Intensity={intensity:.2f}, Falloff Radius=10.0")
    print(f"High-Jamming Drone Start: {high_start_pos} | Initial Jam Probability: {high_initial_prob:.2%}")
    print(f"Low-Jamming Drone Start:  {low_start_pos} | Initial Jam Probability: {low_initial_prob:.2%}")
    print(f"Chosen Action for all {n_steps} steps: RIGHT (4)\n")

    print(f"{'Step':<5} | {'High-Jam Pos':<14} | {'High JamProb':<12} | {'High Executed':<13} | {'High Scrambled?':<15} | {'Low-Jam Pos':<13} | {'Low Executed'}")
    print("-" * 95)

    high_scrambles = 0
    low_scrambles = 0
    high_positions = [world_high.drones[0]]
    low_positions = [world_low.drones[0]]

    for step in range(1, n_steps + 1):
        # Record pre-step positions and jam probabilities
        curr_high_pos = world_high.drones[0]
        curr_low_pos = world_low.drones[0]
        high_prob = world_high.jammer.jam_probability(curr_high_pos)
        low_prob = world_low.jammer.jam_probability(curr_low_pos)

        # Both choose RIGHT for drone 0, STAY for others
        high_actions = [RIGHT, STAY, STAY, STAY, STAY]
        low_actions = [RIGHT, STAY, STAY, STAY, STAY]

        status_high = world_high.step(high_actions)
        status_low = world_low.step(low_actions)

        exec_high = status_high[0]["action"]
        exec_low = status_low[0]["action"]

        scrambled_high = (exec_high == STAY)
        scrambled_low = (exec_low == STAY)

        if scrambled_high:
            high_scrambles += 1
        if scrambled_low:
            low_scrambles += 1

        high_positions.append(world_high.drones[0])
        low_positions.append(world_low.drones[0])

        high_act_str = "RIGHT" if exec_high == RIGHT else "STAY (scrambled)"
        low_act_str = "RIGHT" if exec_low == RIGHT else "STAY (scrambled)"
        scram_str = "YES [SCRAMBLED]" if scrambled_high else "NO"

        print(f"{step:<5} | {str(world_high.drones[0]):<14} | {high_prob:<12.2%} | {high_act_str:<13} | {scram_str:<15} | {str(world_low.drones[0]):<13} | {low_act_str}")

    print("-" * 95)
    print(f"High-Jamming Summary: {high_scrambles}/{n_steps} actions SCRAMBLED to STAY. Net movement: {world_high.drones[0][0] - high_start_pos[0]} cells (expected 15 without jamming).")
    print(f"Low-Jamming Summary:  {low_scrambles}/{n_steps} actions scrambled. Net movement: {world_low.drones[0][0] - low_start_pos[0]} cells (expected 15).")

    # Assertions
    assert high_scrambles > 0, "High-jamming drone must experience scrambled commands!"
    assert world_high.drones[0] != (high_start_pos[0] + n_steps, high_start_pos[1]), "High-jamming drone trajectory must deviate from 15 RIGHT moves!"
    assert low_scrambles == 0, "Low-jamming drone outside falloff radius must experience 0 scrambles!"
    assert world_low.drones[0] == (low_start_pos[0] + n_steps, low_start_pos[1]), "Low-jamming drone must complete all 15 RIGHT moves exactly!"

    print("\n>>> [PASS] Part A passed: High-jamming trajectory noticeably scrambled, low-jamming trajectory 100% reliable.\n")


def test_part_b_determinism():
    print("================================================================================")
    print("TEST PART B: Seeded Determinism Across Independent Runs")
    print("================================================================================")
    seed = 42
    n_steps = 20
    intensity = 0.7

    # Run 1
    world1 = DroneWorld(seed=seed, jammer_intensity=intensity, n_no_fly_zones=0)
    world1.drones[0] = (9, 10)
    history1 = []
    for _ in range(n_steps):
        actions = [RIGHT, STAY, STAY, STAY, STAY]
        status = world1.step(actions)
        history1.append((status[0]["action"], world1.drones[0]))

    # Run 2 with identical seed
    world2 = DroneWorld(seed=seed, jammer_intensity=intensity, n_no_fly_zones=0)
    world2.drones[0] = (9, 10)
    history2 = []
    for _ in range(n_steps):
        actions = [RIGHT, STAY, STAY, STAY, STAY]
        status = world2.step(actions)
        history2.append((status[0]["action"], world2.drones[0]))

    print(f"Comparing {n_steps} steps of two independent runs with seed={seed} and jammer_intensity={intensity}:")
    for i, ((act1, pos1), (act2, pos2)) in enumerate(zip(history1, history2), 1):
        act_str = "RIGHT" if act1 == RIGHT else "STAY [SCRAMBLED]"
        print(f"  Step {i:2d}: Run1=(action={act1}, pos={pos1}) | Run2=(action={act2}, pos={pos2}) | Match: {act1 == act2 and pos1 == pos2}")
        assert act1 == act2, f"Action mismatch at step {i}: {act1} vs {act2}"
        assert pos1 == pos2, f"Position mismatch at step {i}: {pos1} vs {pos2}"

    print(f"\nAll {n_steps} steps matched bit-for-bit between Run 1 and Run 2.")
    print(">>> [PASS] Part B passed: Jamming action-scrambling is 100% deterministic.\n")


def test_part_c_metrics_tracker_energy():
    print("================================================================================")
    print("TEST PART C: MetricsTracker Energy Accounting for Scrambled Actions")
    print("================================================================================")
    seed = 42
    n_steps = 15
    intensity = 0.7

    world = DroneWorld(seed=seed, jammer_intensity=intensity, n_no_fly_zones=0)
    world.drones[0] = (9, 10)
    tracker = MetricsTracker(world=world)

    chosen_non_stay_actions = 0
    actual_effective_non_stay_actions = 0

    for step in range(1, n_steps + 1):
        # Chosen action is non-STAY (RIGHT = 4)
        actions = [RIGHT, STAY, STAY, STAY, STAY]
        chosen_non_stay_actions += 1

        status = tracker.record_step(actions)
        effective_act = status[0]["action"]
        if effective_act != STAY:
            actual_effective_non_stay_actions += 1

    summary = tracker.summary()
    recorded_energy = summary["energy"]

    print(f"Total steps executed:                      {n_steps}")
    print(f"Naive non-STAY actions CHOSEN by policy:   {chosen_non_stay_actions}")
    print(f"Actual non-STAY actions EXECUTED:          {actual_effective_non_stay_actions}")
    print(f"MetricsTracker reported energy:            {recorded_energy}")

    # Assertions
    assert recorded_energy < chosen_non_stay_actions, (
        f"MetricsTracker energy ({recorded_energy}) must be strictly less than chosen actions ({chosen_non_stay_actions}) "
        f"because scrambled actions do not expend physical motor energy!"
    )
    assert recorded_energy == actual_effective_non_stay_actions, (
        f"MetricsTracker energy ({recorded_energy}) must exactly match effectively executed non-STAY actions ({actual_effective_non_stay_actions})!"
    )

    print("\n>>> [PASS] Part C passed: MetricsTracker correctly excludes scrambled-to-STAY actions from energy expenditure.\n")


def main():
    print("================================================================================")
    print("CONSTELLATION — Session S2.6: Action Unreliability Mechanism Verification")
    print("================================================================================\n")

    test_part_a_trajectory_comparison()
    test_part_b_determinism()
    test_part_c_metrics_tracker_energy()

    print("================================================================================")
    print("ALL S2.6 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("================================================================================")


if __name__ == "__main__":
    main()
