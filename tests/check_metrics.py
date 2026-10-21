"""
check_metrics.py - Verification script for MetricsTracker in CONSTELLATION.

Verifications:
1. Note why no-fly zones and delivery points are not naturally reachable from the depot
   (0, 0) within 10 steps (Manhattan distances >= 18 and >= 22 cells).
2. Section 1 (Primary 10-step simulation):
   - Build DroneWorld(seed=0, n_no_fly_zones=3, jammer_intensity=0.3) wrapped in MetricsTracker.
   - Using S1.4's manual setup style prior to step 1:
     * Drone 0 placed at (12, 11) adjacent to delivery point P0 (12, 13).
     * Drone 1 placed at (5, 13) adjacent to Zone 0 path (5, 14) at t=1.
     * Drone 2 placed at (1, 0) and Drone 3 placed at (0, 1) to test mid-air convergence at (1, 1).
     * Drone 4 remains at depot (0, 0).
   - Execute 10 discrete steps with hand-picked action vectors:
     * Step 1: Drone 1 enters Zone 0 (no-fly-zone collision); Drones 2 & 3 converge to (1, 1) (mid-air multi-drone collision).
     * Step 2: Drone 0 reaches delivery point (12, 13) (delivery completion); Drones 1 & 3 clear collision cells.
     * Step 3: Drone 3 returns to depot (0, 0) while Drone 4 is waiting (depot multi-drone collision).
     * Step 4: Drone 3 exits depot back to (1, 0).
     * Steps 5-10: Stationary holding (STAY actions) to complete a 10-step episode.
   - Inspect and verify summary() fields: success (float), no_fly_zone_collisions (int),
     multi_drone_collisions (int), steps_to_last_delivery (int), total_steps (int), energy (int).
3. Section 2 (Full-completion latency verification):
   - Demonstrates that when all delivery points are completed, steps_to_last_delivery
     reflects the exact step of the final delivery rather than defaulting to total_steps.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.drone_world import DroneWorld, STAY, UP, DOWN, LEFT, RIGHT
from src.metrics import MetricsTracker


def main():
    print("================================================================================")
    print("NATURAL REACHABILITY ANALYSIS (seed=0, 10-Step Horizon)")
    print("================================================================================")
    init_world = DroneWorld(seed=0, n_no_fly_zones=3, jammer_intensity=0.3)
    depot = init_world.depot
    deliv_dists = [abs(p[0] - depot[0]) + abs(p[1] - depot[1]) for p in init_world.delivery_points]
    zone_dists = [
        abs(z.position_at(t)[0] - depot[0]) + abs(z.position_at(t)[1] - depot[1])
        for z in init_world.no_fly_zones
        for t in range(11)
    ]
    min_deliv = min(deliv_dists)
    min_zone = min(zone_dists)

    print(f"Depot position: {depot}")
    print(f"Delivery points at t=0: {init_world.delivery_points}")
    print(f"Minimum Manhattan distance from depot to any delivery point: {min_deliv} cells")
    print(f"Minimum Manhattan distance from depot to any active zone cell across t=0..10: {min_zone} cells")
    print(f"Maximum travel distance for any drone in 10 steps: 10 cells")
    print("Conclusion: Natural reachability from depot (0, 0) within 10 steps is mathematically impossible.")
    print("Following S1.4's documented approach, we use manual positioning prior to step 1")
    print("to demonstrate and verify delivery completion, zone collision, and multi-drone collision.\n")

    # ==============================================================================
    # SECTION 1: 10-Step Sequence with MetricsTracker
    # ==============================================================================
    print("================================================================================")
    print("SECTION 1: 10-Step MetricsTracker Verification")
    print("================================================================================\n")

    world = DroneWorld(seed=0, n_no_fly_zones=3, jammer_intensity=0.3)
    tracker = MetricsTracker(world=world)

    # Manual setup prior to step 1 (t=0)
    world.drones[0] = (12, 11)  # 2 steps north of delivery point P0 (12, 13)
    world.drones[1] = (5, 13)   # 1 step north of Zone 0's t=1 position (5, 14)
    world.drones[2] = (1, 0)    # 1 step north of collision cell (1, 1)
    world.drones[3] = (0, 1)    # 1 step west of collision cell (1, 1)
    world.drones[4] = (0, 0)    # Staging at depot

    print(f"Initial Setup (t=0):")
    print(f"  Drone positions: {world.drones}")
    print(f"  Delivery points: {world.delivery_points}")
    print(f"  Remaining deliveries: {world.remaining_deliveries}")
    print(f"  Delivered: {world.delivered}")
    print()

    # Hand-picked 10-step action sequence:
    # Action constants: STAY=0, UP=1, DOWN=2, LEFT=3, RIGHT=4
    #
    # Step 1:
    # - Drone 0: DOWN -> (12, 12) (moving toward P0)
    # - Drone 1: DOWN -> (5, 14) (enters Zone 0 at (5, 14), triggers NFZ warning & collision)
    # - Drone 2: DOWN -> (1, 1) (enters (1, 1))
    # - Drone 3: RIGHT -> (1, 1) (enters (1, 1), triggers mid-air multi-drone collision with Drone 2)
    # - Drone 4: STAY -> (0, 0) (holding at depot)
    # Non-STAY actions: 4 (energy +4)
    #
    # Step 2:
    # - Drone 0: DOWN -> (12, 13) (reaches P0! Delivery completion registered)
    # - Drone 1: UP -> (5, 13) (exits Zone 0)
    # - Drone 2: STAY -> (1, 1) (holding)
    # - Drone 3: UP -> (1, 0) (exits collision cell)
    # - Drone 4: STAY -> (0, 0) (holding)
    # Non-STAY actions: 3 (energy +3, cumulative = 7)
    #
    # Step 3 (Depot Collision):
    # - Drone 0: STAY -> (12, 13)
    # - Drone 1: STAY -> (5, 13)
    # - Drone 2: STAY -> (1, 1)
    # - Drone 3: LEFT -> (0, 0) (returns to depot where Drone 4 is waiting!)
    # - Drone 4: STAY -> (0, 0) (left at depot)
    # Drones 3 and 4 simultaneously occupy depot (0, 0) -> multi-drone collision at depot!
    # Non-STAY actions: 1 (energy +1, cumulative = 8)
    #
    # Step 4:
    # - Drone 3: RIGHT -> (1, 0) (exits depot to resolve depot overlap)
    # - All other drones STAY
    # Non-STAY actions: 1 (energy +1, cumulative = 9)
    #
    # Steps 5 to 10:
    # - All drones STAY -> 0 non-STAY actions each step (energy stays at 9)
    step_actions = [
        ([DOWN, DOWN, DOWN, RIGHT, STAY], "Step 1: NFZ collision (D1 at (5, 14)) + Mid-air collision (D2, D3 at (1, 1))"),
        ([DOWN, UP, STAY, UP, STAY],      "Step 2: Delivery completion (D0 at P0 (12, 13))"),
        ([STAY, STAY, STAY, LEFT, STAY],  "Step 3: Depot collision (D3 moves LEFT to (0, 0), D4 left at (0, 0))"),
        ([STAY, STAY, STAY, RIGHT, STAY], "Step 4: D3 exits depot to (1, 0)"),
        ([STAY, STAY, STAY, STAY, STAY],  "Step 5: Holding (all STAY)"),
        ([STAY, STAY, STAY, STAY, STAY],  "Step 6: Holding (all STAY)"),
        ([STAY, STAY, STAY, STAY, STAY],  "Step 7: Holding (all STAY)"),
        ([STAY, STAY, STAY, STAY, STAY],  "Step 8: Holding (all STAY)"),
        ([STAY, STAY, STAY, STAY, STAY],  "Step 9: Holding (all STAY)"),
        ([STAY, STAY, STAY, STAY, STAY],  "Step 10: Holding (all STAY)"),
    ]

    for step_idx, (actions, description) in enumerate(step_actions, start=1):
        print(f"--- Step {step_idx} ({description}) ---")
        print(f"  Actions: {actions}")
        status = tracker.record_step(actions)
        print(f"  Drones post-step: {world.drones}")
        print(f"  Delivered set: {world.delivered}")
        print(f"  Remaining deliveries: {world.remaining_deliveries}")
        print(f"  Counters after step {step_idx}: NFZ_collisions={tracker.no_fly_zone_collisions}, "
              f"multi_drone_collisions={tracker.multi_drone_collisions}, energy={tracker.energy}")
        if step_idx == 3:
            assert world.drones[3] == world.depot and world.drones[4] == world.depot, "Step 3 must have D3 and D4 at depot"
            print(f"  --> CONFIRMATION: Drones 3 and 4 occupy depot (0, 0) at t=3, correctly counted in multi_drone_collisions!")
        print()

    # Retrieve and print summary dictionary
    summary_dict = tracker.summary()
    print("================================================================================")
    print("METRICS TRACKER SUMMARY (Section 1 - 10 Steps)")
    print("================================================================================")
    for k, v in summary_dict.items():
        print(f"  {k:25s}: {v} (type: {type(v).__name__})")
    print()

    # Hand-inspection and programmatic verification of Section 1
    print("--- Verification of Section 1 Summary ---")
    expected_success = 1.0 / 3.0
    expected_nfz_collisions = 1
    expected_multi_drone_collisions = 2  # 1 mid-air at (1, 1) + 1 depot at (0, 0)
    expected_total_steps = 10
    expected_steps_to_last = 10  # 1 of 3 delivered; not all delivered -> total_steps
    expected_energy = 9          # 4 (step 1) + 3 (step 2) + 1 (step 3) + 1 (step 4)

    print(f"  [CHECK] success: {summary_dict['success']:.4f} == {expected_success:.4f} -> OK")
    print(f"  [CHECK] no_fly_zone_collisions: {summary_dict['no_fly_zone_collisions']} == {expected_nfz_collisions} -> OK")
    print(f"  [CHECK] multi_drone_collisions: {summary_dict['multi_drone_collisions']} == {expected_multi_drone_collisions} (includes depot collision) -> OK")
    print(f"  [CHECK] steps_to_last_delivery: {summary_dict['steps_to_last_delivery']} == {expected_steps_to_last} -> OK")
    print(f"  [CHECK] total_steps: {summary_dict['total_steps']} == {expected_total_steps} -> OK")
    print(f"  [CHECK] energy: {summary_dict['energy']} == {expected_energy} -> OK")

    assert isinstance(summary_dict["success"], float)
    assert 0.0 <= summary_dict["success"] <= 1.0
    assert abs(summary_dict["success"] - expected_success) < 1e-6
    assert isinstance(summary_dict["no_fly_zone_collisions"], int)
    assert summary_dict["no_fly_zone_collisions"] == expected_nfz_collisions
    assert isinstance(summary_dict["multi_drone_collisions"], int)
    assert summary_dict["multi_drone_collisions"] == expected_multi_drone_collisions
    assert isinstance(summary_dict["steps_to_last_delivery"], int)
    assert summary_dict["steps_to_last_delivery"] == expected_steps_to_last
    assert isinstance(summary_dict["total_steps"], int)
    assert summary_dict["total_steps"] == expected_total_steps
    assert isinstance(summary_dict["energy"], int)
    assert summary_dict["energy"] == expected_energy
    print("  All Section 1 assertions passed successfully.\n")

    # ==============================================================================
    # SECTION 2: All-Deliveries Completed Scenario (Latency Verification)
    # ==============================================================================
    print("================================================================================")
    print("SECTION 2: All-Deliveries Completed Latency Verification")
    print("================================================================================\n")

    world2 = DroneWorld(seed=0, n_no_fly_zones=3, jammer_intensity=0.3)
    tracker2 = MetricsTracker(world=world2)

    # Position 3 drones each 1 step adjacent to their respective delivery targets
    p0, p1, p2 = world2.delivery_points  # [(12, 13), (18, 6), (3, 19)]
    world2.drones[0] = (p0[0], p0[1] - 1)  # (12, 12) -> 1 step UP from P0
    world2.drones[1] = (p1[0], p1[1] - 1)  # (18, 5) -> 1 step UP from P1
    world2.drones[2] = (p2[0], p2[1] - 1)  # (3, 18) -> 1 step UP from P2
    world2.drones[3] = (1, 0)
    world2.drones[4] = (0, 0)

    # Step 1: Drone 0 delivers P0 at t=1
    tracker2.record_step([DOWN, STAY, STAY, STAY, STAY])
    print(f"Step 1: Drone 0 delivered P0. Delivered count: {len(world2.delivered)}/3")

    # Step 2: Drone 1 delivers P1 at t=2
    tracker2.record_step([STAY, DOWN, STAY, STAY, STAY])
    print(f"Step 2: Drone 1 delivered P1. Delivered count: {len(world2.delivered)}/3")

    # Step 3: Drone 2 delivers P2 at t=3 (all delivered!)
    tracker2.record_step([STAY, STAY, DOWN, STAY, STAY])
    print(f"Step 3: Drone 2 delivered P2. Delivered count: {len(world2.delivered)}/3 (ALL DELIVERED)")

    # Steps 4 & 5: Stationary holding
    tracker2.record_step([STAY, STAY, STAY, STAY, STAY])
    tracker2.record_step([STAY, STAY, STAY, STAY, STAY])

    summary2 = tracker2.summary()
    print("\nSummary when all deliveries complete (total_steps=5):")
    for k, v in summary2.items():
        print(f"  {k:25s}: {v} (type: {type(v).__name__})")

    assert summary2["success"] == 1.0, f"Expected success=1.0, got {summary2['success']}"
    assert summary2["steps_to_last_delivery"] == 3, f"Expected steps_to_last_delivery=3, got {summary2['steps_to_last_delivery']}"
    assert summary2["total_steps"] == 5, f"Expected total_steps=5, got {summary2['total_steps']}"
    assert summary2["energy"] == 3, f"Expected energy=3, got {summary2['energy']}"
    print("\n  [CHECK] steps_to_last_delivery correctly reflects step 3 (< total_steps 5) when all delivered.")
    print("  All Section 2 assertions passed successfully.\n")

    print("================================================================================")
    print("ALL CHECK_METRICS VERIFICATIONS PASSED SUCCESSFULLY")
    print("================================================================================")


if __name__ == "__main__":
    main()
