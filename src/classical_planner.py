"""
classical_planner.py - Classical Planner (no coordination) baseline for CONSTELLATION.

Civilian logistics framing:
A non-learning, rule-based baseline comparison point for multi-drone delivery operations.
Each drone acts completely independently:
- No inter-drone communication or coordination.
- No memory of past trajectories or commitments across steps (replanned fresh every single step).
- Greedily targets the single nearest undelivered delivery point by Manhattan distance.
- Takes one distance-reducing step toward its target, greedily avoiding known no-fly zones
  at the next time step (t + 1).

Tie-break and movement priority rules:
1. Target selection: Lowest Manhattan distance to an undelivered delivery point.
   Ties broken deterministically by coordinate order (p[0], p[1]).
2. Movement priority:
   - Evaluates distance-reducing candidate moves.
   - Fixed priority order: Horizontal move (RIGHT if tx > x else LEFT if tx < x) is evaluated
     before Vertical move (DOWN if ty > y else UP if ty < y).
   - Any move landing on an active no-fly zone at t+1 is excluded.
   - If no distance-reducing move is safe (or already at target), the drone takes STAY (0).
"""

from typing import List, Tuple, Set, Optional
from src.drone_world import DroneWorld, STAY, UP, DOWN, LEFT, RIGHT, ACTION_DELTAS


class ClassicalPlanner:
    """
    Classical Planner (no coordination) baseline.

    Produces joint actions for all drones in a DroneWorld instance,
    re-evaluating state greedily at each step without inter-agent communication.
    """

    def __init__(self) -> None:
        pass

    def plan_drone(
        self,
        curr_pos: Tuple[int, int],
        remaining_deliveries: List[Tuple[int, int]],
        zone_positions_next: Set[Tuple[int, int]],
    ) -> int:
        """
        Plans a single discrete action for one drone.

        Args:
            curr_pos: Current (x, y) coordinate of the drone.
            remaining_deliveries: List of coordinates of undelivered delivery points.
            zone_positions_next: Set of coordinates occupied by no-fly zones at t + 1.

        Returns:
            Integer action: 0=STAY, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT.
        """
        # If no undelivered delivery points remain, stay in place
        if not remaining_deliveries:
            return STAY

        curr_x, curr_y = curr_pos

        # 1. Target Selection:
        # Pick the nearest undelivered delivery point by Manhattan distance.
        # Tie-break: deterministic coordinate order (p[0], p[1]).
        target = min(
            remaining_deliveries,
            key=lambda p: (abs(curr_x - p[0]) + abs(curr_y - p[1]), p[0], p[1]),
        )
        tx, ty = target

        dx = tx - curr_x
        dy = ty - curr_y

        # Already at target (will be marked delivered after step)
        if dx == 0 and dy == 0:
            return STAY

        # 2. Movement Rule & Tie-Break:
        # Evaluate distance-reducing directional moves in fixed priority order:
        # Horizontal (RIGHT/LEFT reducing x-distance) before Vertical (DOWN/UP reducing y-distance).
        candidate_moves: List[int] = []

        # Priority 1: Horizontal move reducing x-distance
        if dx > 0:
            candidate_moves.append(RIGHT)
        elif dx < 0:
            candidate_moves.append(LEFT)

        # Priority 2: Vertical move reducing y-distance
        if dy > 0:
            candidate_moves.append(DOWN)
        elif dy < 0:
            candidate_moves.append(UP)

        # 3. No-Fly Zone Collision Avoidance:
        # Check candidate moves against zone positions at t + 1.
        # Return the highest-priority move that does not land on a no-fly zone.
        for action in candidate_moves:
            delta_x, delta_y = ACTION_DELTAS[action]
            next_pos = (curr_x + delta_x, curr_y + delta_y)
            if next_pos not in zone_positions_next:
                return action

        # If no distance-reducing move is safe, fall back to STAY
        return STAY

    def plan(self, world: DroneWorld) -> List[int]:
        """
        Plans discrete actions for all drones in the given DroneWorld.

        Args:
            world: Current DroneWorld instance.

        Returns:
            List of length n_drones containing discrete actions (0..4).
        """
        next_t = world.t + 1
        zone_positions_next: Set[Tuple[int, int]] = {
            zone.position_at(next_t) for zone in world.no_fly_zones
        }

        actions: List[int] = []
        for pos in world.drones:
            action = self.plan_drone(
                curr_pos=pos,
                remaining_deliveries=world.remaining_deliveries,
                zone_positions_next=zone_positions_next,
            )
            actions.append(action)

        return actions

    def __call__(self, world: DroneWorld) -> List[int]:
        """Allows calling the planner instance directly: planner(world)."""
        return self.plan(world)


def plan_classical(world: DroneWorld) -> List[int]:
    """Functional convenience entry point for Classical Planner (no coordination)."""
    return ClassicalPlanner().plan(world)
