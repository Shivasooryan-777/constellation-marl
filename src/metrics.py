"""
metrics.py - Episode metrics tracking for CONSTELLATION drone world.

Civilian logistics framing:
Monitors and logs key performance and safety metrics for delivery drone operations:
- success: fraction of delivery points completed by end of episode (0.0 to 1.0)
- collisions:
    * no_fly_zone_collisions: count of no-fly-zone entry events (occupying an active zone cell at t)
    * multi_drone_collisions: count of multi-drone same-cell events (>=2 drones on the same cell at t)
- latency:
    * steps_to_last_delivery: steps until the last delivery was completed (or total episode steps if not all delivered)
    * total_steps: total steps elapsed across the episode
- energy: count of non-STAY actions issued across all drones across the episode (movement cost proxy)

Architecture:
MetricsTracker is decoupled from DroneWorld. It wraps a DroneWorld instance, observing
transitions from the outside via record_step(actions). This allows the same tracker
to benchmark classical planners, heuristics, and RL policies without modifying environment dynamics.
"""

from typing import List, Dict, Any, Tuple, Optional
from src.drone_world import DroneWorld, STAY


class MetricsTracker:
    """
    External observer that tracks episode metrics for a DroneWorld instance.

    Args:
        world: DroneWorld instance to monitor.
    """

    def __init__(self, world: DroneWorld) -> None:
        self.world: DroneWorld = world

        # Internal metric counters
        self.total_steps: int = 0
        self.energy: int = 0
        self.no_fly_zone_collisions: int = 0
        self.multi_drone_collisions: int = 0

        # Delivery latency tracking
        self.last_delivery_step: Optional[int] = None
        self._step_all_delivered: Optional[int] = None
        self._prev_delivered_count: int = len(self.world.delivered)

        # Check if already fully delivered at start (edge case)
        if (
            len(self.world.delivery_points) > 0
            and len(self.world.delivered) == len(self.world.delivery_points)
        ):
            self._step_all_delivered = 0

    def record_step(self, actions: List[int]) -> List[Dict[str, Any]]:
        """
        Executes a step on the wrapped DroneWorld instance and updates metric counters.

        Args:
            actions: List of length n_drones containing discrete action integers (0..4).

        Returns:
            The list of per-drone transition status dicts returned by world.step(actions).
        """
        # Execute transition on the world
        status = self.world.step(actions)

        # Advance total step counter
        self.total_steps += 1

        # 1. Energy: count non-STAY actions issued across all drones
        self.energy += sum(1 for a in actions if a != STAY)

        # 2. No-fly zone collisions: count no-fly zone entry events
        # Reuses S1.4 warning condition: drone occupying an active no-fly zone cell at world.t
        for pos in self.world.drones:
            for zone in self.world.no_fly_zones:
                if zone.position_at(self.world.t) == pos:
                    self.no_fly_zone_collisions += 1

        # 3. Multi-drone collisions: count multi-drone same-cell events
        # Evaluates cells occupied by two or more drones at time t (no cells exempted, including depot (0, 0))
        cell_counts: Dict[Tuple[int, int], int] = {}
        for pos in self.world.drones:
            cell_counts[pos] = cell_counts.get(pos, 0) + 1

        for count in cell_counts.values():
            if count >= 2:
                self.multi_drone_collisions += 1

        # 4. Delivery tracking for latency
        curr_delivered_count = len(self.world.delivered)
        if curr_delivered_count > self._prev_delivered_count:
            self.last_delivery_step = self.total_steps
            if (
                curr_delivered_count == len(self.world.delivery_points)
                and self._step_all_delivered is None
            ):
                self._step_all_delivered = self.total_steps
            self._prev_delivered_count = curr_delivered_count

        return status

    def summary(self) -> Dict[str, Any]:
        """
        Produces a per-episode summary dictionary of the tracked metrics.

        Returns:
            Dict containing:
            - success: fraction of delivery points completed (0.0 to 1.0) [float]
            - no_fly_zone_collisions: count of no-fly-zone entries [int]
            - multi_drone_collisions: count of multi-drone same-cell events [int]
            - steps_to_last_delivery: steps until last delivery, or total_steps if not all delivered [int]
            - total_steps: total steps elapsed [int]
            - energy: count of non-STAY actions [int]
        """
        n_total = len(self.world.delivery_points)
        n_delivered = len(self.world.delivered)

        success = float(n_delivered / n_total) if n_total > 0 else 1.0

        all_delivered = (n_delivered == n_total) and (n_total > 0)
        if all_delivered and self._step_all_delivered is not None:
            latency = self._step_all_delivered
        else:
            latency = self.total_steps

        return {
            "success": float(success),
            "no_fly_zone_collisions": int(self.no_fly_zone_collisions),
            "multi_drone_collisions": int(self.multi_drone_collisions),
            "steps_to_last_delivery": int(latency),
            "total_steps": int(self.total_steps),
            "energy": int(self.energy),
        }
