"""
check_jammer.py - Verification script for scripted Jammer in DroneWorld.

Verifications:
1. Build four DroneWorld(seed=0) instances with jammer_intensity = 0.0, 0.3, 0.5, 0.7.
2. For each, print render(t=0) including the jammer summary line (or no summary line when 0.0).
3. For jammer_intensity=0.5, print jam_probability at 5 hand-picked positions:
   - Jammer's own cell (grid_size // 2, grid_size // 2)
   - Depot (0, 0)
   - Delivery point P0
   - Grid corner (0, grid_size - 1)
   - Grid corner (grid_size - 1, grid_size - 1)
4. Verify peak probability equals intensity, decreases monotonically with Manhattan distance,
   and reaches 0.0 at or beyond falloff_radius.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.drone_world import DroneWorld


def main():
    intensities = [0.0, 0.3, 0.5, 0.7]
    worlds = {}

    print("================================================================================")
    print("SECTION 1: DroneWorld Renders across Jammer Intensities (0.0, 0.3, 0.5, 0.7)")
    print("================================================================================\n")

    for intensity in intensities:
        print(f"--- DroneWorld(seed=0, jammer_intensity={intensity}) ---")
        world = DroneWorld(seed=0, jammer_intensity=intensity)
        worlds[intensity] = world

        rendered = world.render(t=0)
        print(rendered)
        print()

        # Sanity assertions
        if intensity == 0.0:
            assert world.jammer is None, "jammer_intensity=0.0 must produce self.jammer = None"
            assert "Jammer:" not in rendered, "No jammer summary line should appear when intensity=0.0"
        else:
            assert world.jammer is not None, f"Expected active jammer for intensity={intensity}"
            assert world.jammer.position == (world.grid_size // 2, world.grid_size // 2)
            assert abs(world.jammer.intensity - intensity) < 1e-6
            assert world.jammer.falloff_radius == float(world.grid_size // 2)
            assert "Jammer:" in rendered, "Jammer summary line must appear when jammer is active"

    print("================================================================================")
    print("SECTION 2: 5-Position Jammer Falloff Demonstration (jammer_intensity=0.5)")
    print("================================================================================\n")

    world_05 = worlds[0.5]
    jammer = world_05.jammer
    assert jammer is not None

    jammer_pos = jammer.position
    depot_pos = world_05.depot
    delivery_p0 = world_05.delivery_points[0]
    corner_top_left = (0, world_05.grid_size - 1)
    corner_top_right = (world_05.grid_size - 1, world_05.grid_size - 1)

    test_positions = [
        ("Jammer's own cell", jammer_pos),
        (f"Delivery point P0", delivery_p0),
        ("Depot", depot_pos),
        (f"Grid corner (0, {world_05.grid_size - 1})", corner_top_left),
        (f"Grid corner ({world_05.grid_size - 1}, {world_05.grid_size - 1})", corner_top_right),
    ]

    print(f"Jammer configuration: position={jammer.position}, intensity={jammer.intensity}, falloff_radius={jammer.falloff_radius}")
    print(f"Falloff formula: jam_prob(pos) = intensity * max(0, 1 - manhattan_dist / falloff_radius)\n")

    for name, pos in test_positions:
        dist = jammer.manhattan_distance(pos)
        prob = jammer.jam_probability(pos)
        print(f"  {name:30s} pos={pos!s:10s} | Manhattan dist = {dist:2d} | jam_probability = {prob:.4f}")

    # Explicit assertions for the 5 positions
    # 1. Jammer's own cell: dist = 0, jam_prob == intensity == 0.5
    assert abs(jammer.jam_probability(jammer_pos) - 0.5) < 1e-6, "Prob at jammer cell must equal intensity"

    # 2. Delivery point P0 (12, 13): dist = |12-10| + |13-10| = 5, jam_prob = 0.5 * (1 - 5/10) = 0.25
    assert abs(jammer.jam_probability(delivery_p0) - 0.25) < 1e-6, "Prob at (12, 13) must equal 0.25"

    # 3. Depot (0, 0): dist = 20 >= falloff_radius (10) -> jam_prob == 0.0
    assert jammer.jam_probability(depot_pos) == 0.0, "Prob at depot must equal 0.0"

    # 4. Corner (0, 19): dist = 10 + 9 = 19 >= 10 -> jam_prob == 0.0
    assert jammer.jam_probability(corner_top_left) == 0.0, "Prob at far corner must equal 0.0"

    # 5. Corner (19, 19): dist = 9 + 9 = 18 >= 10 -> jam_prob == 0.0
    assert jammer.jam_probability(corner_top_right) == 0.0, "Prob at far corner must equal 0.0"

    # Monotonicity test: along horizontal line from center (10, 10) to (19, 10)
    prev_prob = jammer.jam_probability(jammer_pos)
    for step in range(1, 10):
        pos = (jammer_pos[0] + step, jammer_pos[1])
        curr_prob = jammer.jam_probability(pos)
        assert curr_prob <= prev_prob, f"Probability must decrease monotonically with distance (step={step})"
        if step >= int(jammer.falloff_radius):
            assert curr_prob == 0.0, f"Probability must reach 0.0 at or beyond falloff_radius (step={step})"
        prev_prob = curr_prob

    print("\nAll jammer verification assertions passed successfully.")


if __name__ == "__main__":
    main()
