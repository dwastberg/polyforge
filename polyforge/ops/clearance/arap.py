import numpy as np
import shapely
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import factorized
from shapely.geometry import Polygon, Point
from shapely.ops import nearest_points
from shapely.geometry import JOIN_STYLE


class ARAPLiteSolver:
    """
    ARAP-lite solver with reusable sparse factorization.
    """

    def __init__(
        self,
        coords: np.ndarray,
        constraint_indices: list[int],
        constraint_weight: float = 1000.0,
    ):
        self.n = len(coords)
        self.dim = 2
        self.size = self.n * self.dim
        self.constraint_indices = constraint_indices
        self.weight = constraint_weight

        self.A = self._build_matrix()
        self.solve = factorized(self.A)  # <-- expensive step (once)

    def _idx(self, i, d):
        return i * self.dim + d

    def _build_matrix(self) -> csc_matrix:
        A = lil_matrix((self.size, self.size))

        # Edge rigidity terms
        for i in range(self.n):
            j = (i + 1) % self.n
            for d in range(self.dim):
                A[self._idx(i, d), self._idx(i, d)] += 1
                A[self._idx(i, d), self._idx(j, d)] -= 1
                A[self._idx(j, d), self._idx(j, d)] += 1
                A[self._idx(j, d), self._idx(i, d)] -= 1

        # Constraint structure (positions go in RHS only)
        for vi in self.constraint_indices:
            for d in range(self.dim):
                A[self._idx(vi, d), self._idx(vi, d)] += self.weight

        # Return as CSC format - factorized() prefers this and avoids conversion warning
        return A.tocsc()

    def solve_positions(
        self,
        coords: np.ndarray,
        constraint_targets: dict[int, np.ndarray],
    ) -> np.ndarray:
        """
        Solve ARAP-lite deformation for updated constraint targets.
        """
        b = np.zeros(self.size)

        # Edge RHS
        for i in range(self.n):
            j = (i + 1) % self.n
            for d in range(self.dim):
                diff = coords[i, d] - coords[j, d]
                b[self._idx(i, d)] += diff
                b[self._idx(j, d)] -= diff

        # Constraint RHS
        for vi, target in constraint_targets.items():
            for d in range(self.dim):
                b[self._idx(vi, d)] += self.weight * target[d]

        x = self.solve(b)
        return x.reshape((self.n, self.dim))

def _widen_notch_arap(
    polygon: Polygon,
    min_clearance: float,
    original_holes: tuple,
    original_area: float,
    tolerance: float = 1e-9,
    max_iterations: int = 5,
) -> Polygon | None:
    """Widen a notch-shaped narrow passage using minimum_clearance_line and ARAP.

    This handles the case where erosion fills in a notch rather than splitting
    the polygon. We use minimum_clearance_line to find the narrow point directly.
    """
    current = polygon
    coords = np.asarray(current.exterior.coords[:-1])

    for iteration in range(max_iterations):
        # Check current clearance
        try:
            current_clearance = current.minimum_clearance
            if current_clearance >= min_clearance:
                return current
        except Exception:
            return None

        # Get minimum clearance line - this tells us exactly where the narrow point is
        try:
            clearance_line = shapely.minimum_clearance_line(current)
        except Exception:
            return None

        if clearance_line is None or clearance_line.is_empty:
            return None

        line_coords = list(clearance_line.coords)
        if len(line_coords) != 2:
            return None

        p1 = np.array(line_coords[0])
        p2 = np.array(line_coords[1])

        delta = np.linalg.norm(p2 - p1)
        if delta < tolerance:
            return None

        required = min_clearance - delta
        if required <= tolerance:
            return current

        direction = (p2 - p1) / delta
        move = 0.5 * required * direction

        # Lift to boundary vertices
        b1 = np.array(nearest_points(Point(p1), current.boundary)[1].coords[0])
        b2 = np.array(nearest_points(Point(p2), current.boundary)[1].coords[0])

        def nearest_vertex(pt):
            return int(np.argmin(np.linalg.norm(coords - pt, axis=1)))

        v1 = nearest_vertex(b1)
        v2 = nearest_vertex(b2)

        # Build ARAP solver with these constraint vertices
        solver = ARAPLiteSolver(
            coords,
            constraint_indices=[v1, v2],
            constraint_weight=1000.0,
        )

        # Move vertices apart
        constraints = {
            v1: coords[v1] - move,
            v2: coords[v2] + move,
        }

        new_coords = solver.solve_positions(coords, constraints)

        candidate = Polygon(
            np.vstack([new_coords, new_coords[0]]),
            holes=[ring.coords[:] for ring in original_holes]
        )

        if not candidate.is_valid:
            return None

        if candidate.area < 0.9 * original_area:
            return None

        # Prepare for next iteration
        current = candidate
        coords = new_coords

    return current


def widen_narrow_passage_offset_arap_lite(
    polygon: Polygon,
    min_clearance: float,
    *,
    max_iterations: int = 5,
    join_style: int = JOIN_STYLE.mitre,
    tolerance: float = 1e-9,
) -> Polygon | None:

    if not polygon.is_valid or polygon.is_empty:
        return None

    r = min_clearance / 2.0
    original_holes = polygon.interiors
    current = polygon
    original_area = polygon.area

    # --- Extract initial coordinates (once) ---
    coords = np.asarray(current.exterior.coords[:-1])

    for iteration in range(max_iterations):
        # Check if we already meet the clearance requirement
        try:
            current_clearance = current.minimum_clearance
            if current_clearance >= min_clearance:
                return current
        except Exception:
            pass

        eroded = current.buffer(-r, join_style=join_style)

        if eroded.is_empty:
            return None

        if eroded.geom_type == "Polygon":
            # Erosion didn't split - this could be a notch case
            # Try direct widening using minimum_clearance_line
            result = _widen_notch_arap(current, min_clearance, original_holes, original_area, tolerance)
            if result is not None:
                return result
            return current

        if eroded.geom_type != "MultiPolygon":
            return None

        # --- Find closest eroded components ---
        components = list(eroded.geoms)
        min_dist = np.inf
        pair = None

        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                d = components[i].distance(components[j])
                if d < min_dist:
                    min_dist = d
                    pair = (components[i], components[j])

        if pair is None:
            return None

        c1, c2 = pair
        p1, p2 = nearest_points(c1, c2)

        p1 = np.array(p1.coords[0])
        p2 = np.array(p2.coords[0])

        delta = np.linalg.norm(p2 - p1)
        if delta < tolerance:
            return None

        required = min_clearance - delta
        if required <= tolerance:
            return current

        direction = (p2 - p1) / delta
        move = 0.5 * required * direction

        # --- Lift to boundary ---
        b1 = np.array(nearest_points(Point(p1), current.boundary)[1].coords[0])
        b2 = np.array(nearest_points(Point(p2), current.boundary)[1].coords[0])

        def nearest_vertex(pt):
            return int(np.argmin(np.linalg.norm(coords - pt, axis=1)))

        v1 = nearest_vertex(b1)
        v2 = nearest_vertex(b2)

        # --- Build solver ONCE ---
        if iteration == 0:
            solver = ARAPLiteSolver(
                coords,
                constraint_indices=[v1, v2],
                constraint_weight=1000.0,
            )

        constraints = {
            v1: coords[v1] - move,
            v2: coords[v2] + move,
        }

        # --- Solve deformation ---
        new_coords = solver.solve_positions(coords, constraints)


        candidate = Polygon(
            np.vstack([new_coords, new_coords[0]]),
            holes=[ring.coords[:] for ring in original_holes]
        )

        if not candidate.is_valid:
            return None

        if candidate.area < 0.9 * original_area:
            return None

        # Prepare for next iteration
        current = candidate
        coords = new_coords

    return None
