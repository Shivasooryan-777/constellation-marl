"""
check_observations.py - Verification script for DroneWorld observation vector and global state.

Verifications:
1. Initialize DroneWorld(seed=0, n_no_fly_zones=3, jammer_intensity=0.3).
2. Query and print the local observation vector for all 5 drones at t=0.
3. Validate vector type, length (7), shape ((7,)), and float dtype.
4. Verify that all values fall within their expected mathematical ranges:
   - own (x, y) normalized by grid size: [0.0, 1.0]
   - relative vector (dx, dy) to nearest undelivered delivery point normalized by grid size: [-1.0, 1.0]
   - jam probability: [0.0, 1.0]
   - fraction of deliveries remaining: [0.0, 1.0]
   - fraction of episode time remaining: [0.0, 1.0]
5. Verify get_global_state() returns all drone positions, delivery point statuses, and current t.
"""

import sys
import os
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.drone_world import DroneWorld


def main():
    print("================================================================================")
    print("CONSTELLATION - Observation Vector & Global State Check (Session S2.1)")
    print("================================================================================\n")

    world = DroneWorld(seed=0, n_no_fly_zones=3, jammer_intensity=0.3, max_steps=150)

    print(f"World Configuration:")
    print(f"  Grid Size: {world.grid_size}x{world.grid_size}")
    print(f"  Depot: {world.depot}")
    print(f"  Drones count: {world.n_drones}")
    print(f"  Delivery points: {world.delivery_points}")
    print(f"  Jammer: intensity={world.jammer.intensity}, pos={world.jammer.position}")
    print(f"  Max Steps: {world.max_steps}")
    print(f"  Initial t: {world.t}\n")

    print("--------------------------------------------------------------------------------")
    print("SECTION 1: Local Observation Vectors at t=0 for All 5 Drones")
    print("--------------------------------------------------------------------------------")

    feature_names = [
        "1. own_x (norm)",
        "2. own_y (norm)",
        "3. rel_dx to nearest delivery (norm)",
        "4. rel_dy to nearest delivery (norm)",
        "5. jam probability at pos",
        "6. fraction deliveries remaining",
        "7. fraction time remaining",
    ]

    for drone_id in range(world.n_drones):
        obs = world.get_observation(drone_id)

        print(f"\nDrone {drone_id} Observation Vector:")
        print(f"  Raw Vector: {obs}")
        print(f"  Shape: {obs.shape} | Length: {len(obs)} | Dtype: {obs.dtype}")
        print("  Feature Breakdown:")
        for name, val in zip(feature_names, obs):
            print(f"    - {name:38s}: {val:+.4f}")

        # Assertions
        assert isinstance(obs, np.ndarray), f"Drone {drone_id}: observation must be np.ndarray"
        assert obs.shape == (7,), f"Drone {drone_id}: expected shape (7,), got {obs.shape}"
        assert len(obs) == 7, f"Drone {drone_id}: expected length 7, got {len(obs)}"
        assert np.issubdtype(obs.dtype, np.floating), f"Drone {drone_id}: expected float dtype"

        # Range checks
        assert 0.0 <= obs[0] <= 1.0, f"Drone {drone_id}: own_x={obs[0]} not in [0.0, 1.0]"
        assert 0.0 <= obs[1] <= 1.0, f"Drone {drone_id}: own_y={obs[1]} not in [0.0, 1.0]"
        assert -1.0 <= obs[2] <= 1.0, f"Drone {drone_id}: rel_dx={obs[2]} not in [-1.0, 1.0]"
        assert -1.0 <= obs[3] <= 1.0, f"Drone {drone_id}: rel_dy={obs[3]} not in [-1.0, 1.0]"
        assert 0.0 <= obs[4] <= 1.0, f"Drone {drone_id}: jam_prob={obs[4]} not in [0.0, 1.0]"
        assert 0.0 <= obs[5] <= 1.0, f"Drone {drone_id}: deliveries_remaining={obs[5]} not in [0.0, 1.0]"
        assert 0.0 <= obs[6] <= 1.0, f"Drone {drone_id}: time_remaining={obs[6]} not in [0.0, 1.0]"

    print("\n--------------------------------------------------------------------------------")
    print("SECTION 2: Global State Verification (for Centralized Critic)")
    print("--------------------------------------------------------------------------------")
    global_state = world.get_global_state()
    print(f"Global State Dictionary:")
    print(f"  drones (positions): {global_state['drones']}")
    print(f"  delivery_statuses:  {global_state['delivery_statuses']}")
    print(f"  timestep t:         {global_state['t']}")

    assert "drones" in global_state or "drone_positions" in global_state
    assert "delivery_statuses" in global_state
    assert "t" in global_state
    assert len(global_state["drones"]) == world.n_drones
    assert len(global_state["delivery_statuses"]) == len(world.delivery_points)
    assert global_state["t"] == 0

    print("\n--------------------------------------------------------------------------------")
    print("CONFIRMATION SUMMARY")
    print("--------------------------------------------------------------------------------")
    print(f"Observation vector length: {len(obs)}")
    print("All 5 observation vectors verified successfully.")
    print("All observation values fall strictly within their defined ranges.")
    print("Global state verified successfully.")
    print("CHECK OBSERVATIONS PASSED.")


if __name__ == "__main__":
    main()
