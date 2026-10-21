"""
drone_world.py - 2D drone delivery world state with moving no-fly zones, scripted jammer,
and discrete movement transitions for CONSTELLATION.

Civilian logistics framing only:
- 2D grid
- Fixed depot at (0, 0)
- Delivery points placed via seeded reject-and-resample with minimum Manhattan distance
- Drones initialized at the depot
- Moving No-Fly Zones: deterministic, repeating paths as a pure function of integer time step t
- Pattern pool with reservable held-out pattern for unseen testing
- Scripted Jammer: fixed at grid center, configurable intensity (0% to 70%), deterministic distance-based falloff
- Discrete drone actions (stay, up, down, left, right) with boundary clamping and step() state transitions
- Delivery-completion detection tracking remaining vs delivered points
- No multi-drone collision rules, no reward shaping, no RL/torch usage yet.
"""

from typing import List, Tuple, Set, Optional, Dict, Any
import random
import numpy as np



# ==============================================================================
# Action Definitions and Coordinate Conventions
# ==============================================================================
# Coordinate system:
# Standard 2D grid convention:
# - Origin (0, 0) is at top-left.
# - x is the horizontal coordinate (column index), increasing to the right (East) [0, grid_size - 1].
# - y is the vertical coordinate (row index), increasing downwards (South) [0, grid_size - 1].
#
# Discrete Actions:
# - STAY (0): drone remains in place, delta = (0, 0)
# - UP (1): moves north, decreasing y-coordinate, delta = (0, -1)
# - DOWN (2): moves south, increasing y-coordinate, delta = (0, +1)
# - LEFT (3): moves west, decreasing x-coordinate, delta = (-1, 0)
# - RIGHT (4): moves east, increasing x-coordinate, delta = (+1, 0)
# ==============================================================================

ACTION_STAY = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4

# Short aliases
STAY = ACTION_STAY
UP = ACTION_UP
DOWN = ACTION_DOWN
LEFT = ACTION_LEFT
RIGHT = ACTION_RIGHT

ACTION_DELTAS: Dict[int, Tuple[int, int]] = {
    ACTION_STAY: (0, 0),
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}



class NoFlyZone:
    """
    Represents a dynamic 1-cell no-fly zone moving in a deterministic repeating pattern.

    Movement is a pure function of integer time step t: same t always produces the same cell.
    No-fly zones never occupy the depot or delivery points (reflecting away if encountered).
    """

    def __init__(
        self,
        zone_id: int,
        pattern_id: str,
        start_pos: Tuple[int, int],
        seed: int,
        grid_size: int = 20,
        obstacles: Optional[Set[Tuple[int, int]]] = None,
        span: int = 4,
    ) -> None:
        self.zone_id = zone_id
        self.pattern_id = pattern_id
        self.start_pos = start_pos
        self.seed = seed
        self.grid_size = grid_size
        self.obstacles: Set[Tuple[int, int]] = set(obstacles) if obstacles else set()
        self.span = span
        self.path: List[Tuple[int, int]] = self._generate_path()

    def position_at(self, t: int) -> Tuple[int, int]:
        """
        Pure function of integer time step t returning the occupied cell (x, y).
        """
        if not self.path:
            return self.start_pos
        return self.path[t % len(self.path)]

    def _generate_path(self) -> List[Tuple[int, int]]:
        """
        Generates the repeating cycle of coordinates according to pattern_id.
        All patterns guarantee:
        1. Moves by exactly one grid cell per time step.
        2. Never occupies an obstacle cell (depot or delivery point).
        3. Stays within [0, grid_size - 1] grid boundaries.
        """
        if self.pattern_id == "horizontal":
            return self._generate_horizontal_path()
        elif self.pattern_id == "vertical":
            return self._generate_vertical_path()
        elif self.pattern_id == "box_loop":
            return self._generate_box_loop_path()
        elif self.pattern_id == "diagonal":
            return self._generate_diagonal_path()
        else:
            raise ValueError(f"Unknown pattern_id: {self.pattern_id}")

    def _generate_linear_path(self, dx: int, dy: int) -> List[Tuple[int, int]]:
        """
        Builds a linear back-and-forth path along (dx, dy).
        Advances cell by cell; reflects (reverses direction) upon reaching span or hitting
        grid boundaries / obstacles.
        """
        steps = [self.start_pos]
        x, y = self.start_pos
        for _ in range(self.span):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and (nx, ny) not in self.obstacles:
                steps.append((nx, ny))
                x, y = nx, ny
            else:
                break

        # If forward direction was immediately blocked, attempt the reverse direction
        if len(steps) < 3:
            alt_steps = [self.start_pos]
            x, y = self.start_pos
            for _ in range(self.span):
                nx, ny = x - dx, y - dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and (nx, ny) not in self.obstacles:
                    alt_steps.append((nx, ny))
                    x, y = nx, ny
                else:
                    break
            if len(alt_steps) > len(steps):
                steps = alt_steps

        if len(steps) <= 1:
            return steps

        # Construct full repeating cycle: [c0, c1, ..., cm, cm-1, ..., c1]
        cycle = steps + steps[-2:0:-1]
        return cycle

    def _generate_horizontal_path(self) -> List[Tuple[int, int]]:
        """
        Pattern 1 - Horizontal back-and-forth:
        Moves horizontally along row (dx=1, dy=0), reflecting at boundaries or obstacles.
        """
        return self._generate_linear_path(dx=1, dy=0)

    def _generate_vertical_path(self) -> List[Tuple[int, int]]:
        """
        Pattern 2 - Vertical back-and-forth:
        Moves vertically along column (dx=0, dy=1), reflecting at boundaries or obstacles.
        """
        return self._generate_linear_path(dx=0, dy=1)

    def _generate_diagonal_path(self) -> List[Tuple[int, int]]:
        """
        Pattern 4 (Held Out) - Diagonal back-and-forth:
        Moves diagonally (dx=1, dy=1), reflecting at boundaries or obstacles.
        """
        return self._generate_linear_path(dx=1, dy=1)

    def _generate_box_loop_path(self) -> List[Tuple[int, int]]:
        """
        Pattern 3 - Box perimeter loop:
        Moves in a clockwise rectangular perimeter: East -> South -> West -> North.
        Tests 4 orientations anchored at start_pos to find a clear path avoiding obstacles.
        """
        w, h = 2, 2
        x0, y0 = self.start_pos

        # Orientation candidates: (dx_sign, dy_sign)
        # 1. Top-Left: (+1, +1) -> East, South, West, North
        # 2. Bottom-Left: (+1, -1) -> North, East, South, West
        # 3. Top-Right: (-1, +1) -> South, West, North, East
        # 4. Bottom-Right: (-1, -1) -> West, North, East, South
        candidates = []

        # Candidate 1: Top-Left anchor
        c1 = [self.start_pos]
        for i in range(1, w + 1):
            c1.append((x0 + i, y0))
        for i in range(1, h + 1):
            c1.append((x0 + w, y0 + i))
        for i in range(1, w + 1):
            c1.append((x0 + w - i, y0 + h))
        for i in range(1, h):
            c1.append((x0, y0 + h - i))
        candidates.append(c1)

        # Candidate 2: Bottom-Left anchor
        c2 = [self.start_pos]
        for i in range(1, h + 1):
            c2.append((x0, y0 - i))
        for i in range(1, w + 1):
            c2.append((x0 + i, y0 - h))
        for i in range(1, h + 1):
            c2.append((x0 + w, y0 - h + i))
        for i in range(1, w):
            c2.append((x0 + w - i, y0))
        candidates.append(c2)

        # Candidate 3: Top-Right anchor
        c3 = [self.start_pos]
        for i in range(1, h + 1):
            c3.append((x0, y0 + i))
        for i in range(1, w + 1):
            c3.append((x0 - i, y0 + h))
        for i in range(1, h + 1):
            c3.append((x0 - w, y0 + h - i))
        for i in range(1, w):
            c3.append((x0 - w + i, y0))
        candidates.append(c3)

        # Candidate 4: Bottom-Right anchor
        c4 = [self.start_pos]
        for i in range(1, w + 1):
            c4.append((x0 - i, y0))
        for i in range(1, h + 1):
            c4.append((x0 - w, y0 - i))
        for i in range(1, w + 1):
            c4.append((x0 - w + i, y0 - h))
        for i in range(1, h):
            c4.append((x0, y0 - h + i))
        candidates.append(c4)

        for candidate in candidates:
            if all(
                0 <= cx < self.grid_size and 0 <= cy < self.grid_size and (cx, cy) not in self.obstacles
                for (cx, cy) in candidate
            ):
                return candidate

        # Fallback: if all box orientations are constrained, use linear reflection
        return self._generate_horizontal_path()


class Jammer:
    """
    Scripted communication jammer at a fixed grid location with deterministic distance-based falloff.

    Civilian logistics framing: models ambient RF interference / communication degradation.
    Stores position, intensity, and falloff_radius.
    Calculates jam_probability(pos) = intensity * max(0, 1 - manhattan_distance(pos, jammer_pos) / falloff_radius),
    clamped strictly within [0.0, intensity].
    """

    def __init__(
        self,
        position: Tuple[int, int],
        intensity: float,
        falloff_radius: float,
    ) -> None:
        self.position: Tuple[int, int] = position
        self.intensity: float = float(intensity)
        self.falloff_radius: float = float(falloff_radius)

    def manhattan_distance(self, pos: Tuple[int, int]) -> int:
        """Calculates Manhattan distance from the jammer to pos."""
        return abs(pos[0] - self.position[0]) + abs(pos[1] - self.position[1])

    def jam_probability(self, pos: Tuple[int, int]) -> float:
        """
        Calculates the effective communication jamming probability at pos.
        Formula: intensity * max(0.0, 1.0 - manhattan_distance(pos, jammer_pos) / falloff_radius)
        Clamped to [0.0, intensity].
        """
        dist = self.manhattan_distance(pos)
        if self.falloff_radius <= 0.0:
            raw_prob = self.intensity if dist == 0 else 0.0
        else:
            raw_prob = self.intensity * max(0.0, 1.0 - (dist / self.falloff_radius))
        return max(0.0, min(float(self.intensity), float(raw_prob)))


class DroneWorld:
    """
    Static world state representing a 2D delivery grid, depot, delivery points,
    drones, moving no-fly zones, and scripted jammer.
    """

    AVAILABLE_PATTERNS: List[str] = ["horizontal", "vertical", "box_loop", "diagonal"]
    HELD_OUT_PATTERNS: List[str] = ["diagonal"]

    # Action constants exposed on class for convenience
    STAY: int = STAY
    UP: int = UP
    DOWN: int = DOWN
    LEFT: int = LEFT
    RIGHT: int = RIGHT

    def __init__(
        self,
        grid_size: int = 20,
        n_drones: int = 5,
        n_delivery_points: int = 3,
        seed: int = 0,
        min_delivery_distance: int = 12,
        n_no_fly_zones: int = 3,
        include_held_out: bool = False,
        jammer_intensity: float = 0.0,
        max_steps: int = 150,
    ) -> None:
        self.grid_size = grid_size
        self.n_drones = n_drones
        self.n_delivery_points = n_delivery_points
        self.seed = seed
        self.min_delivery_distance = min_delivery_distance
        self.n_no_fly_zones = n_no_fly_zones
        self.include_held_out = include_held_out
        self.held_out_patterns: List[str] = list(self.HELD_OUT_PATTERNS)
        self.jammer_intensity = jammer_intensity
        self.max_steps = max_steps
        self.max_episode_steps = max_steps

        # Internal simulation time step, incremented each step() call
        self.t: int = 0

        # Place depot at (0, 0)
        self.depot: Tuple[int, int] = (0, 0)
        self.depot_coords: List[Tuple[int, int]] = [self.depot]

        # Place delivery points using seeded rejection sampling
        self.delivery_points: List[Tuple[int, int]] = self._place_delivery_points()

        # Track completed and remaining delivery points (state tracking only, no rewards)
        self.delivered: Set[Tuple[int, int]] = set()
        self.remaining_deliveries: List[Tuple[int, int]] = list(self.delivery_points)
        self.just_delivered: Set[Tuple[int, int]] = set()

        # Place all drones at the depot
        self.drones: List[Tuple[int, int]] = [self.depot for _ in range(self.n_drones)]
        self.prev_drones: List[Tuple[int, int]] = list(self.drones)

        # Initialize moving no-fly zones
        self.no_fly_zones: List[NoFlyZone] = self._init_no_fly_zones()

        # Initialize Jammer at grid center when intensity > 0.0
        if self.jammer_intensity > 0.0:
            center = (self.grid_size // 2, self.grid_size // 2)
            falloff_radius = float(self.grid_size // 2)
            self.jammer: Optional[Jammer] = Jammer(
                position=center,
                intensity=self.jammer_intensity,
                falloff_radius=falloff_radius,
            )
        else:
            self.jammer = None

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

    def _init_no_fly_zones(self) -> List[NoFlyZone]:
        """
        Initializes seeded moving no-fly zones.
        Selects n_no_fly_zones non-held-out patterns.
        If include_held_out is True, additionally appends the held-out pattern(s).
        Each zone starts at a valid, non-overlapping cell (not depot, not delivery point,
        and not another zone's starting cell).
        """
        # Active patterns (excluding held-out)
        active_pool = [p for p in self.AVAILABLE_PATTERNS if p not in self.HELD_OUT_PATTERNS]

        # Deterministic selection of active patterns
        if self.n_no_fly_zones <= len(active_pool):
            patterns_to_place = active_pool[: self.n_no_fly_zones]
        else:
            rng_pat = random.Random(self.seed + 10)
            patterns_to_place = [rng_pat.choice(active_pool) for _ in range(self.n_no_fly_zones)]

        # Append held-out pattern(s) if explicitly requested
        if self.include_held_out:
            patterns_to_place = patterns_to_place + self.held_out_patterns

        obstacles = {self.depot} | set(self.delivery_points)
        zones: List[NoFlyZone] = []
        used_starts: Set[Tuple[int, int]] = set()

        rng = random.Random(self.seed + 100)
        max_attempts = 10000

        for i, pattern_id in enumerate(patterns_to_place):
            zone_seed = self.seed * 1000 + i
            placed = False
            attempts = 0

            while attempts < max_attempts:
                attempts += 1
                candidate = (rng.randint(0, self.grid_size - 1), rng.randint(0, self.grid_size - 1))

                # Must not start on depot, delivery points, or existing zone starts
                if candidate in obstacles or candidate in used_starts:
                    continue

                zone = NoFlyZone(
                    zone_id=i,
                    pattern_id=pattern_id,
                    start_pos=candidate,
                    seed=zone_seed,
                    grid_size=self.grid_size,
                    obstacles=obstacles,
                    span=4,
                )

                # Ensure path is valid and changes across t=0..4
                if len(zone.path) >= 4:
                    used_starts.add(candidate)
                    zones.append(zone)
                    placed = True
                    break

            if not placed:
                raise RuntimeError(
                    f"Could not place no-fly zone {i} with pattern {pattern_id} after {max_attempts} attempts."
                )

        return zones

    def step(self, actions: List[int]) -> List[Dict[str, Any]]:
        """
        Executes one discrete simulation step for all drones.

        Args:
            actions: List of length n_drones with integer actions
                     (0=STAY, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT).

        Process:
            1. For each drone i, compute candidate position: (curr_x + dx, curr_y + dy).
            2. Clamp candidate position to grid boundaries [0, grid_size - 1].
            3. Update stored drone positions.
            4. Increment self.t by 1.
            5. Check for delivery completions: if any drone occupies an undelivered delivery
               point coordinate, add that coordinate to self.delivered and update
               self.remaining_deliveries (state tracking only; no rewards).
            6. Check each drone's new position against active no-fly zones at the new self.t.
               If a drone lands on an active no-fly zone cell, log a warning:
               "Warning: drone {i} entered no-fly zone (pattern {zone.pattern_id}) at {pos}, t={self.t}"
               (The move is not blocked or altered; penalties/rewards are future work).

        Returns:
            A list of per-drone status dictionaries containing drone index, previous position,
            action taken, and new position.
        """
        if len(actions) != self.n_drones:
            raise ValueError(f"Expected {self.n_drones} actions, got {len(actions)}")

        # Store positions prior to movement for transition tracking
        self.prev_drones = list(self.drones)
        self.just_delivered = set()

        status: List[Dict[str, Any]] = []
        for i in range(self.n_drones):
            action = actions[i]
            if action not in ACTION_DELTAS:
                raise ValueError(f"Invalid action {action} for drone {i}. Must be 0..4.")

            curr_x, curr_y = self.drones[i]
            dx, dy = ACTION_DELTAS[action]

            # Boundary clamping: ensure drones cannot leave the grid
            new_x = max(0, min(self.grid_size - 1, curr_x + dx))
            new_y = max(0, min(self.grid_size - 1, curr_y + dy))
            new_pos = (new_x, new_y)

            self.drones[i] = new_pos
            status.append({
                "drone_id": i,
                "prev_pos": (curr_x, curr_y),
                "action": action,
                "pos": new_pos,
            })

        # Advance internal simulation time step
        self.t += 1

        # Check for delivery completions at the new drone positions (prior to zone collision warnings)
        for pos in self.drones:
            if pos in self.delivery_points and pos not in self.delivered:
                self.delivered.add(pos)
                self.just_delivered.add(pos)
        self.remaining_deliveries = [p for p in self.delivery_points if p not in self.delivered]

        # Check for no-fly zone collisions at the new time step
        for i, pos in enumerate(self.drones):
            for zone in self.no_fly_zones:
                if zone.position_at(self.t) == pos:
                    print(
                        f"Warning: drone {i} entered no-fly zone (pattern {zone.pattern_id}) "
                        f"at {pos}, t={self.t}"
                    )

        return status

    def render(self, t: Optional[int] = None, sep: str = "") -> str:
        """
        Renders the 20x20 grid as an ASCII string at time step t.
        If t is None, defaults to self.t (the current internal simulation time step).

        Symbols:
            D: Depot
            P: Delivery point
            A: Drone (when active outside depot)
            X: No-fly zone
            .: Empty cell

        If a zone cell coincides with a drone/depot/delivery cell at some t,
        X is displayed with priority, and a warning is logged to stdout.
        """
        if t is None:
            t = self.t
        delivery_set = set(self.delivery_points)
        active_drone_set = {d for d in self.drones if d != self.depot}
        drone_set = set(self.drones)

        # Compute no-fly zone positions at time t
        zone_positions = set()
        for zone in self.no_fly_zones:
            pos = zone.position_at(t)
            zone_positions.add(pos)
            # Visual/logging check for overlaps
            if pos == self.depot:
                print(f"Warning: No-fly zone {zone.zone_id} ({zone.pattern_id}) coincides with depot at {pos} (t={t})")
            if pos in delivery_set:
                print(f"Warning: No-fly zone {zone.zone_id} ({zone.pattern_id}) coincides with delivery point at {pos} (t={t})")
            if pos in drone_set:
                print(f"Warning: No-fly zone {zone.zone_id} ({zone.pattern_id}) coincides with drone at {pos} (t={t})")

        # Check for zone-vs-zone overlaps
        for i in range(len(self.no_fly_zones)):
            for j in range(i + 1, len(self.no_fly_zones)):
                z1 = self.no_fly_zones[i]
                z2 = self.no_fly_zones[j]
                if z1.position_at(t) == z2.position_at(t):
                    pos = z1.position_at(t)
                    print(
                        f"Warning: zone {z1.zone_id} ({z1.pattern_id}) and zone {z2.zone_id} ({z2.pattern_id}) "
                        f"overlap at {pos}, t={t}"
                    )

        lines = []
        for y in range(self.grid_size):
            row = []
            for x in range(self.grid_size):
                cell = (x, y)
                if cell in zone_positions:
                    row.append("X")
                elif cell == self.depot:
                    row.append("D")
                elif cell in delivery_set:
                    row.append("P")
                elif cell in active_drone_set:
                    row.append("A")
                else:
                    row.append(".")
            lines.append(sep.join(row))

        if self.jammer is not None:
            depot_prob = self.jammer.jam_probability(self.depot)
            delivery_parts = [
                f"{p}: {self.jammer.jam_probability(p):.2f}"
                for p in self.delivery_points
            ]
            lines.append(
                f"Jammer: pos={self.jammer.position}, intensity={self.jammer.intensity:.2f} | "
                f"jam_prob(depot {self.depot})={depot_prob:.2f} | "
                f"jam_prob(delivery)={', '.join(delivery_parts)}"
            )

        return "\n".join(lines)

    def get_observation(self, drone_id: int) -> np.ndarray:
        """
        Returns the local observation vector for drone_id as a NumPy float array.

        Vector components (length 7, normalized 0-1 where applicable, ordered):
        1-2. own (x, y), normalized by grid size
        3-4. relative vector (dx, dy) to nearest undelivered delivery point, normalized by grid size
        5. jam probability at the drone's own current position
        6. fraction of deliveries remaining (remaining / total)
        7. fraction of episode time remaining (based on current t vs max episode steps)

        Local/per-agent only: does NOT include other drones' positions.
        """
        if drone_id < 0 or drone_id >= self.n_drones:
            raise ValueError(f"Invalid drone_id {drone_id}. Must be in [0, {self.n_drones - 1}].")

        curr_x, curr_y = self.drones[drone_id]

        # 1-2. Own (x, y) normalized by grid size
        norm_x = float(curr_x) / float(self.grid_size)
        norm_y = float(curr_y) / float(self.grid_size)

        # 3-4. Relative vector (dx, dy) to nearest undelivered delivery point, normalized by grid size
        if self.remaining_deliveries:
            target = min(
                self.remaining_deliveries,
                key=lambda p: (abs(curr_x - p[0]) + abs(curr_y - p[1]), p[0], p[1]),
            )
            dx = float(target[0] - curr_x) / float(self.grid_size)
            dy = float(target[1] - curr_y) / float(self.grid_size)
        else:
            dx = 0.0
            dy = 0.0

        # 5. Jam probability at the drone's own current position
        if self.jammer is not None:
            jam_prob = float(self.jammer.jam_probability((curr_x, curr_y)))
        else:
            jam_prob = 0.0

        # 6. Fraction of deliveries remaining (remaining / total)
        total_deliveries = len(self.delivery_points)
        deliveries_remaining = (
            float(len(self.remaining_deliveries)) / float(total_deliveries)
            if total_deliveries > 0
            else 0.0
        )

        # 7. Fraction of episode time remaining (current t vs max episode steps)
        max_steps = float(self.max_steps)
        if max_steps > 0:
            time_remaining = max(0.0, min(1.0, float(max_steps - self.t) / max_steps))
        else:
            time_remaining = 0.0

        return np.array(
            [norm_x, norm_y, dx, dy, jam_prob, deliveries_remaining, time_remaining],
            dtype=np.float32,
        )

    def get_global_state(self) -> Dict[str, Any]:
        """
        Returns global world state for the centralized critic.

        Contains:
        - all drone positions
        - all delivery point statuses (delivered/not)
        - current timestep t
        """
        return {
            "drones": list(self.drones),
            "drone_positions": list(self.drones),
            "delivery_statuses": {p: (p in self.delivered) for p in self.delivery_points},
            "delivery_points": [(p, p in self.delivered) for p in self.delivery_points],
            "t": self.t,
        }

