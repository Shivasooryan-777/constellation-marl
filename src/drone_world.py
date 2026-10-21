"""
drone_world.py - Static 2D drone delivery world state scaffolding for CONSTELLATION.

Civilian logistics framing only:
- 2D grid
- Fixed depot at (0, 0)
- Delivery points placed via seeded reject-and-resample with minimum Manhattan distance
- Drones initialized at the depot
- No movement, no jammer, no no-fly zones, no RL/torch usage.
"""

from typing import List, Tuple
import random


class DroneWorld:
    """
    Static world state representing a 2D delivery grid, depot, delivery points, and drones.
    """

    def __init__(
        self,
        grid_size: int = 20,
        n_drones: int = 5,
        n_delivery_points: int = 3,
        seed: int = 0,
        min_delivery_distance: int = 12,
    ) -> None:
        self.grid_size = grid_size
        self.n_drones = n_drones
        self.n_delivery_points = n_delivery_points
        self.seed = seed
        self.min_delivery_distance = min_delivery_distance

        # Place depot at (0, 0)
        self.depot: Tuple[int, int] = (0, 0)
        self.depot_coords: List[Tuple[int, int]] = [self.depot]

        # Place delivery points using seeded rejection sampling
        self.delivery_points: List[Tuple[int, int]] = self._place_delivery_points()

        # Place all drones at the depot
        self.drones: List[Tuple[int, int]] = [self.depot for _ in range(self.n_drones)]

    def _manhattan_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _place_delivery_points(self) -> List[Tuple[int, int]]:
        rng = random.Random(self.seed)
        points: List[Tuple[int, int]] = []
        max_attempts = 10000
        attempts = 0

        while len(points) < self.n_delivery_points and attempts < max_attempts:
            attempts += 1
            candidate = (rng.randint(0, self.grid_size - 1), rng.randint(0, self.grid_size - 1))

            # Must satisfy minimum Manhattan distance of 12 from depot
            if self._manhattan_distance(candidate, self.depot) < self.min_delivery_distance:
                continue

            # Must satisfy minimum Manhattan distance of 12 from each already placed delivery point
            if any(self._manhattan_distance(candidate, p) < self.min_delivery_distance for p in points):
                continue

            points.append(candidate)

        if len(points) < self.n_delivery_points:
            raise RuntimeError(
                f"Could not place {self.n_delivery_points} delivery points with minimum Manhattan "
                f"distance {self.min_delivery_distance} after {max_attempts} attempts."
            )

        return points

    def render(self, sep: str = "") -> str:
        """
        Renders the 20x20 grid as an ASCII string.

        Symbols:
            D: Depot
            P: Delivery point
            A: Drone (when active outside depot)
            .: Empty cell
        """
        delivery_set = set(self.delivery_points)
        # Active drones away from depot
        active_drone_set = {d for d in self.drones if d != self.depot}

        lines = []
        for y in range(self.grid_size):
            row = []
            for x in range(self.grid_size):
                cell = (x, y)
                if cell == self.depot:
                    row.append("D")
                elif cell in delivery_set:
                    row.append("P")
                elif cell in active_drone_set:
                    row.append("A")
                else:
                    row.append(".")
            lines.append(sep.join(row))

        return "\n".join(lines)
