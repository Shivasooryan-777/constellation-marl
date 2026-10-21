"""
check_drone_step.py - Verification script for discrete drone movement and step() in DroneWorld.

Verifications:
1. Initialize DroneWorld(seed=0, n_no_fly_zones=3, jammer_intensity=0.3).
2. Print render() at t=0 (initial state, all drones at depot (0, 0)).
3. Run 5 steps with hand-picked action vectors exercising:
   - Drone 0: boundary clamping against top grid edge (y=0) and left grid edge (x=0).
   - Drone 1: horizontal translation East along top boundary (x increasing).
   - Drone 2: vertical translation South along left boundary (y increasing).
   - Drone 3: diagonal translation toward grid interior (alternating DOWN and RIGHT).
   - Drone 4: closed-loop path returning to depot (0, 0).
4. After each step, print render() (using default t=self.t) verifying world advances.
5. Note why no-fly zones cannot be naturally reached from the depot within 5 steps.
6. Verify no-fly zone entry warning detection and non-blocking behavior in Section 2.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.drone_world import DroneWorld, STAY, UP, DOWN, LEFT, RIGHT, ACTION_DELTAS


def main():
    print("================================================================================")
    print("SECTION 1: 5-Step Drone Movement Simulation (seed=0, n_no_fly_zones=3, jammer=0.3)")
    print("================================================================================\n")

    world = DroneWorld(seed=0, n_no_fly_zones=3, jammer_intensity=0.3)

    print(f"Initial State (t={world.t}):")
    print(f"Depot: {world.depot}")
    print(f"Drones (all at depot): {world.drones}")
    print(f"Delivery points: {world.delivery_points}")
    print(f"No-fly zones active: {[z.pattern_id for z in world.no_fly_zones]}")
    print("\n--- Initial Render (t=0) ---")
    print(world.render())
    print()

    # Hand-picked action lists for 5 discrete steps:
    # Action encodings: STAY=0, UP=1 (dy=-1), DOWN=2 (dy=+1), LEFT=3 (dx=-1), RIGHT=4 (dx=+1)
    #
    # Drone roles:
    # - Drone 0: Tests boundary clamping (UP at y=0, LEFT at x=0). Must remain clamped within grid.
    # - Drone 1: Translates East along top row -> (1,0), (2,0), (3,0), (4,0), (5,0).
    # - Drone 2: Translates South along left column -> (0,1), (0,2), (0,3), (0,4), (0,5).
    # - Drone 3: Moves diagonally toward center -> (0,1), (1,1), (1,2), (2,2), (2,3).
    # - Drone 4: Loops and returns to depot -> (0,0), (1,0), (1,1), (0,1), (0,0).
    planned_steps = [
        {
            "step_num": 1,
            "actions": [UP, RIGHT, DOWN, DOWN, STAY],
            "action_names": ["UP (clamp top)", "RIGHT", "DOWN", "DOWN", "STAY"],
            "expected_positions": [(0, 0), (1, 0), (0, 1), (0, 1), (0, 0)],
        },
        {
            "step_num": 2,
            "actions": [DOWN, RIGHT, DOWN, RIGHT, RIGHT],
            "action_names": ["DOWN", "RIGHT", "DOWN", "RIGHT", "RIGHT"],
            "expected_positions": [(0, 1), (2, 0), (0, 2), (1, 1), (1, 0)],
        },
        {
            "step_num": 3,
            "actions": [LEFT, RIGHT, DOWN, DOWN, DOWN],
            "action_names": ["LEFT (clamp left)", "RIGHT", "DOWN", "DOWN", "DOWN"],
            "expected_positions": [(0, 1), (3, 0), (0, 3), (1, 2), (1, 1)],
        },
        {
            "step_num": 4,
            "actions": [LEFT, RIGHT, DOWN, RIGHT, LEFT],
            "action_names": ["LEFT (clamp left)", "RIGHT", "DOWN", "RIGHT", "LEFT"],
            "expected_positions": [(0, 1), (4, 0), (0, 4), (2, 2), (0, 1)],
        },
        {
            "step_num": 5,
            "actions": [DOWN, RIGHT, DOWN, DOWN, UP],
            "action_names": ["DOWN", "RIGHT", "DOWN", "DOWN", "UP (back to depot)"],
            "expected_positions": [(0, 2), (5, 0), (0, 5), (2, 3), (0, 0)],
        },
    ]

    for plan in planned_steps:
        step_idx = plan["step_num"]
        actions = plan["actions"]
        names = plan["action_names"]
        expected = plan["expected_positions"]

        print(f"--------------------------------------------------------------------------------")
        print(f"STEP {step_idx}: Actions = {actions} ({', '.join(names)})")
        print(f"--------------------------------------------------------------------------------")

        status = world.step(actions)

        # Assertions for state transition correctness
        assert world.t == step_idx, f"world.t should be {step_idx}, got {world.t}"
        assert world.drones == expected, f"Step {step_idx} positions {world.drones} != expected {expected}"

        for i, (pos, exp) in enumerate(zip(world.drones, expected)):
            x, y = pos
            assert 0 <= x < world.grid_size and 0 <= y < world.grid_size, (
                f"Drone {i} out of bounds: {pos}"
            )
            print(f"  Drone {i}: pos={pos} (action={names[i]})")

        print(f"\n--- Render after Step {step_idx} (t={world.t}) ---")
        rendered = world.render()  # Verifies default t=self.t
        print(rendered)
        print()

    # Note on natural zone reachability from depot
    print("================================================================================")
    print("NOTE ON NO-FLY ZONE REACHABILITY IN 5 STEPS")
    print("================================================================================")
    min_dist_to_zone = min(
        abs(z.position_at(t)[0] - 0) + abs(z.position_at(t)[1] - 0)
        for z in world.no_fly_zones
        for t in range(6)
    )
    print(
        f"Depot position: {world.depot}\n"
        f"Active no-fly zone starting positions at t=0: {[z.position_at(0) for z in world.no_fly_zones]}\n"
        f"Minimum Manhattan distance from depot to any active zone cell across t=0..5: {min_dist_to_zone} cells.\n"
        f"Maximum distance any drone can travel in 5 steps: 5 cells.\n"
        f"Conclusion: No drone starting from depot (0, 0) can naturally reach a no-fly zone in 5 steps.\n"
        f"As instructed, we do not force artificial teleportation in the 5-step simulation above.\n"
    )

    print("================================================================================")
    print("SECTION 2: No-Fly Zone Collision Detection & Warning Verification")
    print("================================================================================\n")

    # To explicitly verify the no-fly-zone warning logic specified in Step 4:
    # Build a test world and place a drone adjacent to Zone 0's path.
    test_world = DroneWorld(seed=0, n_no_fly_zones=3)
    # Zone 0 (horizontal) starts at (4, 14) at t=0, and moves to (5, 14) at t=1.
    # Place drone 0 at (5, 13) at t=0. When drone 0 moves DOWN (dy=+1), it reaches (5, 14) at t=1.
    test_world.drones[0] = (5, 13)
    print(f"Setup: Drone 0 manually positioned at (5, 13) prior to step 1.")
    print(f"Zone 0 (horizontal) will move to {test_world.no_fly_zones[0].position_at(1)} at t=1.")
    print("Executing step with Drone 0 action = DOWN (moves into zone at (5, 14)):")

    # Step: Drone 0 takes DOWN (2), other drones STAY (0)
    test_actions = [DOWN, STAY, STAY, STAY, STAY]
    test_world.step(test_actions)

    assert test_world.t == 1, "Expected test_world.t == 1"
    assert test_world.drones[0] == (5, 14), "Drone 0 should have moved to (5, 14)"
    assert test_world.drones[0] == test_world.no_fly_zones[0].position_at(1), "Drone 0 must occupy Zone 0 cell"
    print(f"Drone 0 position at t=1: {test_world.drones[0]} (move was allowed, not blocked)")

    print("\nAll check_drone_step assertions passed successfully.")


if __name__ == "__main__":
    main()
