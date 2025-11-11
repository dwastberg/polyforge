"""Coordinate-level simplification helpers.

These functions operate purely on NumPy arrays so they can be reused by any
geometry wrapper (process_geometry, pipeline steps, etc.) without pulling in
the heavy orchestration layers. They contain the logic that used to live inside
``polyforge.simplify``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from simplification.cutil import (
    simplify_coords as _rdp_simplify,
    simplify_coords_vw as _vw_simplify,
    simplify_coords_vwp as _vwp_simplify,
)


def simplify_rdp_coords(vertices: np.ndarray, epsilon: float) -> np.ndarray:
    """Apply the Ramer-Douglas-Peucker algorithm to a coordinate array."""
    if len(vertices) < 2:
        return vertices.copy()

    result = _rdp_simplify(vertices, epsilon)
    return np.array(result) if not isinstance(result, np.ndarray) else result


def simplify_vw_coords(vertices: np.ndarray, threshold: float) -> np.ndarray:
    """Apply the Visvalingam-Whyatt simplifier to a coordinate array."""
    if len(vertices) < 2:
        return vertices.copy()

    result = _vw_simplify(vertices, threshold)
    return np.array(result) if not isinstance(result, np.ndarray) else result


def simplify_vwp_coords(vertices: np.ndarray, threshold: float) -> np.ndarray:
    """Apply the topology-preserving VW simplifier to a coordinate array."""
    if len(vertices) < 2:
        return vertices.copy()

    result = _vwp_simplify(vertices, threshold)
    return np.array(result) if not isinstance(result, np.ndarray) else result


def snap_short_edges(
    vertices: np.ndarray,
    min_length: float,
    snap_mode: Literal["midpoint", "first", "last"] = "midpoint",
) -> np.ndarray:
    """Collapse edges shorter than ``min_length`` according to ``snap_mode``."""
    if len(vertices) < 2:
        return vertices.copy()

    snap_handlers = {
        "midpoint": lambda current, nxt: (current + nxt) / 2.0,
        "first": lambda current, _: current,
        "last": lambda _, nxt: nxt,
    }

    if snap_mode not in snap_handlers:
        raise ValueError(f"Unknown snap_mode: {snap_mode}")

    is_closed = np.allclose(vertices[0], vertices[-1])
    working = vertices[:-1] if is_closed else vertices
    snapper = snap_handlers[snap_mode]

    collapsed: list[np.ndarray] = []
    i = 0
    n = len(working)

    while i < n:
        current = working[i].copy()
        j = i + 1

        while j < n:
            next_vertex = working[j]
            if np.linalg.norm(next_vertex - current) >= min_length:
                break
            current = np.array(snapper(current, next_vertex), dtype=float, copy=True)
            j += 1

        collapsed.append(current)
        i = j

    if not collapsed:
        return vertices[:2].copy()

    collapsed_array = np.vstack(collapsed)

    if is_closed:
        if len(collapsed_array) == 1:
            collapsed_array = np.vstack([collapsed_array, collapsed_array[0]])
        else:
            first = collapsed_array[0]
            last = collapsed_array[-1]
            if np.linalg.norm(last - first) < min_length:
                merged = np.array(snapper(first, last), dtype=float, copy=True)
                collapsed_array[0] = merged
                collapsed_array[-1] = merged
            else:
                collapsed_array = np.vstack([collapsed_array, collapsed_array[0]])

    if len(collapsed_array) < 2:
        return vertices[:2].copy()

    return collapsed_array


def remove_duplicate_vertices(
    vertices: np.ndarray,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Remove consecutive duplicate vertices within the given tolerance."""
    if len(vertices) < 2:
        return vertices.copy()

    is_closed = np.allclose(vertices[0], vertices[-1])
    result = [vertices[0]]

    for i in range(1, len(vertices)):
        distance = np.linalg.norm(vertices[i] - result[-1])
        if distance > tolerance:
            result.append(vertices[i])

    if is_closed and len(result) > 1 and not np.allclose(result[0], result[-1]):
        result.append(result[0].copy())

    if len(result) < 2:
        return vertices[:2].copy()

    return np.array(result)


__all__ = [
    "simplify_rdp_coords",
    "simplify_vw_coords",
    "simplify_vwp_coords",
    "snap_short_edges",
    "remove_duplicate_vertices",
]
