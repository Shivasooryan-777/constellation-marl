"""
check_no_fly_zones.py - Verification script for moving NoFlyZones in DroneWorld.

Steps checked:
1. Build DroneWorld(seed=0) with default n_no_fly_zones=3.
2. Print render(t) for t = 0, 1, 2, 3, 4 showing zone movement across frames.
3. Print each zone's pattern id and its (x,y) position at each t value as plain lists.
4. Separately, demonstrate that include_held_out=True produces a 4th zone (printing pattern id and t=0 position).
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.drone_world import DroneWorld


def main():
    print("=== Default DroneWorld (seed=0, n_no_fly_zones=3) ===")
    world = DroneWorld(seed=0)
    print("Depot coords:", [world.depot])
    print("Delivery point coords:", world.delivery_points)
    print("Active zone pattern IDs:", [z.pattern_id for z in world.no_fly_zones])
    print(f"Total active zones: {len(world.no_fly_zones)}")
    print()

    # Verify positions at t=0 are valid and non-overlapping
    t0_positions = [z.position_at(0) for z in world.no_fly_zones]
    assert len(set(t0_positions)) == len(t0_positions), "Zones must start at non-overlapping cells."
    for pos in t0_positions:
        assert pos != world.depot, "Zone cannot start on depot."
        assert pos not in world.delivery_points, "Zone cannot start on delivery point."

    # Frames for t = 0, 1, 2, 3, 4
    for t in range(5):
        print(f"--- Frame t = {t} ---")
        grid_str = world.render(t=t)
        print(grid_str)
        
        pattern_ids = [z.pattern_id for z in world.no_fly_zones]
        positions = [z.position_at(t) for z in world.no_fly_zones]
        print(f"Pattern IDs at t={t}: {pattern_ids}")
        print(f"Positions at t={t}: {positions}")
        for z in world.no_fly_zones:
            print(f"  Zone {z.zone_id} ({z.pattern_id}): {z.position_at(t)}")
        print()

        # Sanity check: no zone lands on depot or delivery points
        for pos in positions:
            assert pos != world.depot, f"Zone occupied depot at t={t}"
            assert pos not in world.delivery_points, f"Zone occupied delivery point at t={t}"

        # Sanity check: position_at is a pure function of t
        for z in world.no_fly_zones:
            assert z.position_at(t) == z.position_at(t), "position_at must be deterministic and pure."

    print("=== Held-Out Pattern Demonstration (include_held_out=True) ===")
    world_held_out = DroneWorld(seed=0, include_held_out=True)
    held_out_pattern_ids = [z.pattern_id for z in world_held_out.no_fly_zones]
    held_out_positions_t0 = [z.position_at(0) for z in world_held_out.no_fly_zones]

    print(f"Total zones with include_held_out=True: {len(world_held_out.no_fly_zones)}")
    print(f"All zone pattern IDs: {held_out_pattern_ids}")
    print(f"All zone positions at t=0: {held_out_positions_t0}")
    
    assert len(world_held_out.no_fly_zones) == 4, "Expected exactly 4 zones when include_held_out=True."
    fourth_zone = world_held_out.no_fly_zones[3]
    print(f"4th zone (held-out):")
    print(f"  Pattern ID: {fourth_zone.pattern_id}")
    print(f"  Position at t=0: {fourth_zone.position_at(0)}")

    assert fourth_zone.pattern_id == "diagonal", "Expected 4th zone pattern to be 'diagonal'."
    assert fourth_zone.position_at(0) != world_held_out.depot, "Held-out zone cannot start on depot."
    assert fourth_zone.position_at(0) not in world_held_out.delivery_points, "Held-out zone cannot start on delivery point."
    print("\nAll verification assertions passed successfully.")


if __name__ == "__main__":
    main()
