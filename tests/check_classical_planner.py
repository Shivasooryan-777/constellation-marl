"""
check_classical_planner.py - Verification script for Classical Planner (no coordination).

Baseline description:
Evaluates the "Classical Planner (no coordination)" comparison baseline on DroneWorld:
- Each drone independently targets its nearest undelivered delivery point by Manhattan distance.
- Replanned fresh at every single step with zero memory and zero inter-drone communication.
- Greedily avoids known no-fly zones at t + 1.
- Evaluated end-to-end through MetricsTracker using record_step(actions).
- Max episode step cap: 150 steps (test-script choice to prevent infinite loops, not a planner limit).
- Output: Prints first 10 steps and final 5 steps, followed by tracker.summary().
"""

import sys
import os
from typing import List, Dict, Any, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.drone_world import DroneWorld, STAY, UP, DOWN, LEFT, RIGHT
from src.metrics import MetricsTracker
from src.classical_planner import ClassicalPlanner

ACTION_NAMES = {
    STAY: "STAY",
    UP: "UP",
    DOWN: "DOWN",
    LEFT: "LEFT",
    RIGHT: "RIGHT",
}


def main():
    print("================================================================================")
    print("CLASSICAL PLANNER (NO COORDINATION) - EPISODE SIMULATION")
    print("================================================================================")
    print("Configuration:")
    print("  Seed: 0 | No-fly zones: 3 | Jammer intensity: 0.3 | Episode step cap: 150")
    print("  Policy: Classical Planner (no coordination) - shortest path greedy per drone\n")

    world = DroneWorld(seed=0, n_no_fly_zones=3, jammer_intensity=0.3)
    tracker = MetricsTracker(world=world)
    planner = ClassicalPlanner()

    print(f"Initial State (t=0):")
    print(f"  Depot: {world.depot}")
    print(f"  Drones: {world.drones}")
    print(f"  Delivery points: {world.delivery_points}")
    print(f"  Remaining deliveries: {world.remaining_deliveries}")
    print(f"  Active no-fly zone patterns: {[z.pattern_id for z in world.no_fly_zones]}\n")

    max_steps = 150
    step_history: List[Dict[str, Any]] = []

    print("--- Executing Simulation Loop ---")

    while tracker.total_steps < max_steps:
        # Check termination condition: all delivery points delivered
        if len(world.remaining_deliveries) == 0:
            break

        current_t = world.t
        # Compute actions from Classical Planner (no coordination)
        actions = planner.plan(world)
        action_strs = [ACTION_NAMES.get(a, str(a)) for a in actions]

        # Record step through MetricsTracker
        status = tracker.record_step(actions)

        # Store step snapshot for selective display
        step_record = {
            "step": tracker.total_steps,
            "actions": actions,
            "action_names": action_strs,
            "drones": list(world.drones),
            "delivered_count": len(world.delivered),
            "total_deliveries": len(world.delivery_points),
            "remaining": list(world.remaining_deliveries),
        }
        step_history.append(step_record)

        # Print first 10 steps in real-time
        if step_record["step"] <= 10:
            print(
                f"Step {step_record['step']:3d}: "
                f"Actions=[{', '.join(action_strs)}] | "
                f"Drones={world.drones} | "
                f"Delivered={step_record['delivered_count']}/{step_record['total_deliveries']}"
            )

    total_executed = len(step_history)

    # Print omitted steps indicator and final 5 steps if episode ran longer than 10 steps
    if total_executed > 10:
        start_final = max(11, total_executed - 4)
        if start_final > 11:
            omitted_count = start_final - 11
            print(f"... [{omitted_count} intermediate steps omitted for readability] ...")
        for rec in step_history[start_final - 1 :]:
            print(
                f"Step {rec['step']:3d}: "
                f"Actions=[{', '.join(rec['action_names'])}] | "
                f"Drones={rec['drones']} | "
                f"Delivered={rec['delivered_count']}/{rec['total_deliveries']}"
            )

    print()
    if len(world.remaining_deliveries) == 0:
        print(f"Episode terminated: All {len(world.delivery_points)} delivery points completed at step {tracker.total_steps}!")
    else:
        print(f"Episode terminated: Reached maximum step cap of {max_steps} steps.")

    # Retrieve and print final metrics summary
    summary_dict = tracker.summary()
    print("\n================================================================================")
    print("METRICS TRACKER SUMMARY - CLASSICAL PLANNER (NO COORDINATION)")
    print("================================================================================")
    for k, v in summary_dict.items():
        print(f"  {k:25s}: {v} (type: {type(v).__name__})")
    print()

    # Programmatic sanity assertions
    assert isinstance(summary_dict["success"], float), "success must be float"
    assert 0.0 <= summary_dict["success"] <= 1.0, "success must be in [0.0, 1.0]"
    assert isinstance(summary_dict["no_fly_zone_collisions"], int), "no_fly_zone_collisions must be int"
    assert isinstance(summary_dict["multi_drone_collisions"], int), "multi_drone_collisions must be int"
    assert isinstance(summary_dict["steps_to_last_delivery"], int), "steps_to_last_delivery must be int"
    assert isinstance(summary_dict["total_steps"], int), "total_steps must be int"
    assert isinstance(summary_dict["energy"], int), "energy must be int"
    assert summary_dict["total_steps"] == total_executed, "total_steps must match executed steps"
    assert summary_dict["energy"] > 0, "energy must be positive after drone travel"

    print("All check_classical_planner assertions passed successfully.")


if __name__ == "__main__":
    main()
