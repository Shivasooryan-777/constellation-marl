"""
check_drone_world.py - Verification script for static DroneWorld scaffolding.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.drone_world import DroneWorld


def main():
    world = DroneWorld(seed=0)
    print(world.render())
    print("Depot coords:", [world.depot])
    print("Delivery point coords:", world.delivery_points)
    print("Drone coords:", world.drones)


if __name__ == "__main__":
    main()
