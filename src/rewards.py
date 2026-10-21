"""
rewards.py - Reward computation for CONSTELLATION multi-drone delivery environment.

Civilian logistics framing:
Calculates per-agent scalar rewards based on potential-based distance shaping,
a small per-step progress penalty, and a large terminal reward for completing a delivery.

SPEC:
reward = (prev_distance_to_nearest_undelivered_target - current_distance_to_same_target)
        + small_constant_per_step_penalty
        + large_terminal_reward_if_this_drone_just_completed_a_delivery

Safety / Collisions:
Collisions (no-fly zone entries, multi-drone co-locations) are tracked strictly as metrics
by MetricsTracker and MUST NEVER leak into the reward signal.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
from src.drone_world import DroneWorld

# Module-level reward constants
STEP_PENALTY: float = -0.01
DELIVERY_REWARD: float = 10.0


def compute_reward(
    world: DroneWorld,
    prev_distances: Union[Dict[int, Any], List[Any], np.ndarray],
    drone_id: int,
    step_penalty: float = STEP_PENALTY,
    delivery_reward: float = DELIVERY_REWARD,
) -> float:
    """
    Computes scalar reward for drone_id following a simulation step.

    reward = (prev_distance_to_nearest_undelivered_target - current_distance_to_same_target)
            + small_constant_per_step_penalty
            + large_terminal_reward_if_this_drone_just_completed_a_delivery

    Args:
        world: DroneWorld instance (in its state immediately following step()).
        prev_distances: A dict, list, or NumPy array maintained by the caller between steps,
                        indexed by drone_id. Each entry may be a numeric distance or a tuple/dict
                        specifying (distance, target_coord).
        drone_id: Integer index of the drone [0, world.n_drones - 1].
        step_penalty: Constant penalty added each step (default: -0.01).
        delivery_reward: Terminal bump awarded when this drone completes a delivery (default: 10.0).

    Returns:
        float: Computed scalar reward for drone_id.
    """
    if drone_id < 0 or drone_id >= world.n_drones:
        raise ValueError(f"Invalid drone_id {drone_id}. Must be in [0, {world.n_drones - 1}].")

    curr_pos = world.drones[drone_id]

    # Extract previous distance and target (if provided) from caller-maintained prev_distances
    entry = prev_distances[drone_id]
    target: Optional[Tuple[int, int]] = None

    if isinstance(entry, (tuple, list)) and len(entry) == 2:
        prev_dist = float(entry[0])
        target = tuple(entry[1])  # type: ignore
    elif isinstance(entry, dict):
        prev_dist = float(entry["distance"])
        if "target" in entry and entry["target"] is not None:
            target = tuple(entry["target"])  # type: ignore
    else:
        prev_dist = float(entry)

    # Check if this drone just completed a delivery on this step
    just_delivered = getattr(world, "just_delivered", set())
    completed_delivery = curr_pos in just_delivered

    # Compute current distance to nearest undelivered target as of state s' (Φ(s') = -curr_dist)
    # Genuinely recomputed fresh against remaining deliveries in state s' (no special-casing by movement)
    if completed_delivery:
        curr_dist = 0.0
    elif world.remaining_deliveries:
        nearest = min(
            world.remaining_deliveries,
            key=lambda p: (abs(curr_pos[0] - p[0]) + abs(curr_pos[1] - p[1]), p[0], p[1]),
        )
        curr_dist = float(abs(curr_pos[0] - nearest[0]) + abs(curr_pos[1] - nearest[1]))
    else:
        # No undelivered targets remain
        curr_dist = 0.0

    distance_shaping = prev_dist - curr_dist

    # Large terminal reward if this drone just completed a delivery
    terminal_reward = float(delivery_reward) if completed_delivery else 0.0

    # Total reward: (Φ(s') - Φ(s)) + step_penalty + terminal_bump (strictly no collision penalty)
    reward = distance_shaping + float(step_penalty) + terminal_reward
    return float(reward)
