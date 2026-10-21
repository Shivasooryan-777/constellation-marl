"""
drone_world.py - 2D drone delivery world state with moving no-fly zones and scripted jammer for CONSTELLATION.

Civilian logistics framing only:
- 2D grid
- Fixed depot at (0, 0)
- Delivery points placed via seeded reject-and-resample with minimum Manhattan distance
- Drones initialized at the depot
- Moving No-Fly Zones: deterministic, repeating paths as a pure function of integer time step t
- Pattern pool with reservable held-out pattern for unseen testing
- Scripted Jammer: fixed at grid center, configurable intensity (0% to 70%), deterministic distance-based falloff
- No drone movement logic yet, no collision/penalty logic yet, no RL/torch usage.
"""

from typing import List, Tuple, Set, Optional
import random


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

        # Place depot at (0, 0)
        self.depot: Tuple[int, int] = (0, 0)
        self.depot_coords: List[Tuple[int, int]] = [self.depot]

        # Place delivery points using seeded rejection sampling
        self.delivery_points: List[Tuple[int, int]] = self._place_delivery_points()

        # Place all drones at the depot
        self.drones: List[Tuple[int, int]] = [self.depot for _ in range(self.n_drones)]

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

    def render(self, t: int = 0, sep: str = "") -> str:
        """
        Renders the 20x20 grid as an ASCII string at time step t.

        Symbols:
            D: Depot
            P: Delivery point
            A: Drone (when active outside depot)
            X: No-fly zone
            .: Empty cell

        If a zone cell coincides with a drone/depot/delivery cell at some t,
        X is displayed with priority, and a warning is logged to stdout.
        """
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
