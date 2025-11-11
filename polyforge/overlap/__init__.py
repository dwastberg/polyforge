"""Batch overlap resolution utilities."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import split as shapely_split, unary_union
from shapely.strtree import STRtree

from ..core.geometry_utils import to_single_polygon
from ..core.types import OverlapStrategy

_AREA_EPS = 1e-10


def remove_overlaps(
    polygons: List[Polygon],
    overlap_strategy: OverlapStrategy = OverlapStrategy.SPLIT,
    max_iterations: int = 100,
) -> List[Polygon]:
    """Remove overlaps from a list of polygons using spatial indexing."""
    if not polygons:
        return []

    result = list(polygons)
    changed = True
    iteration = 0

    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        tree = STRtree(result)
        overlapping_pairs = []
        seen = set()

        for i, poly_i in enumerate(result):
            for j in tree.query(poly_i, predicate="intersects"):
                if j <= i or (i, j) in seen:
                    continue
                seen.add((i, j))
                poly_j = result[j]

                if poly_i.touches(poly_j):
                    continue

                overlap = poly_i.intersection(poly_j)
                if getattr(overlap, "area", 0.0) > _AREA_EPS:
                    overlapping_pairs.append((i, j))

        if not overlapping_pairs:
            break

        processed = set()
        independent_pairs = []

        for i, j in overlapping_pairs:
            if i in processed or j in processed:
                continue
            processed.add(i)
            processed.add(j)
            independent_pairs.append((i, j))

        for i, j in independent_pairs:
            result[i], result[j] = resolve_overlap_pair(
                result[i],
                result[j],
                strategy=overlap_strategy,
            )
            changed = True

    return result


def count_overlaps(polygons: List[Polygon], min_area_threshold: float = 1e-10) -> int:
    """Count the number of overlapping polygon pairs."""
    if not polygons:
        return 0

    tree = STRtree(polygons)
    count = 0
    seen = set()

    for i, poly_i in enumerate(polygons):
        for j in tree.query(poly_i, predicate="intersects"):
            if j <= i or (i, j) in seen:
                continue
            seen.add((i, j))
            overlap = poly_i.intersection(polygons[j])
            if getattr(overlap, "area", 0.0) > min_area_threshold:
                count += 1

    return count


def find_overlapping_groups(
    polygons: List[Polygon],
    min_area_threshold: float = 1e-10,
) -> List[List[int]]:
    """Return components of polygons where overlaps are present."""
    if not polygons:
        return []

    tree = STRtree(polygons)
    adjacency = {i: set() for i in range(len(polygons))}

    for i, poly_i in enumerate(polygons):
        for j in tree.query(poly_i, predicate="intersects"):
            if j == i:
                continue
            overlap = poly_i.intersection(polygons[j])
            if getattr(overlap, "area", 0.0) > min_area_threshold:
                adjacency[i].add(j)
                adjacency[j].add(i)

    visited = set()
    groups = []

    def dfs(node: int, acc: List[int]):
        visited.add(node)
        acc.append(node)
        for neighbor in adjacency.get(node, ()):
            if neighbor not in visited:
                dfs(neighbor, acc)

    for idx in range(len(polygons)):
        if idx not in visited:
            component: List[int] = []
            dfs(idx, component)
            groups.append(sorted(component))

    return groups


def resolve_overlap_pair(
    poly1: Polygon,
    poly2: Polygon,
    strategy: OverlapStrategy = OverlapStrategy.SPLIT,
) -> Tuple[Polygon, Polygon]:
    """
    Resolve an overlap between two polygons using the requested strategy.

    This is the canonical entry point used by both `split_overlap` and
    `remove_overlaps`. It handles containment checks, strategy-specific
    allocation, split-line construction, and fallbacks.
    """
    result = _build_overlap_data(poly1, poly2)
    if result is None:
        return poly1, poly2

    overlap, poly1_only, poly2_only = result

    if strategy == OverlapStrategy.LARGEST:
        prefer_first = poly1.area >= poly2.area
        return _assign_entire_overlap(poly1_only, poly2_only, overlap, prefer_first)
    if strategy == OverlapStrategy.SMALLEST:
        prefer_first = poly1.area <= poly2.area
        return _assign_entire_overlap(poly1_only, poly2_only, overlap, prefer_first)

    return _split_equally(poly1, poly2, overlap, poly1_only, poly2_only)


def _build_overlap_data(
    poly1: Polygon, poly2: Polygon
) -> Optional[Tuple[Polygon, BaseGeometry, BaseGeometry]]:
    """Extract overlap and non-overlapping parts of two polygons.

    Returns:
        Tuple of (overlap, poly1_only, poly2_only) or None if no significant overlap
    """
    if not poly1.intersects(poly2):
        return None

    overlap = poly1.intersection(poly2)
    if overlap.is_empty or getattr(overlap, "area", 0.0) < _AREA_EPS:
        return None

    if poly1.contains(poly2) or poly2.contains(poly1):
        return None

    if isinstance(overlap, MultiPolygon):
        merged = unary_union(overlap)
        if isinstance(merged, MultiPolygon):
            overlap = max(merged.geoms, key=lambda p: p.area)
        else:
            overlap = merged

    overlap = to_single_polygon(overlap)
    poly1_only = poly1.difference(overlap)
    poly2_only = poly2.difference(overlap)

    return overlap, poly1_only, poly2_only


def _assign_entire_overlap(
    poly1_only: BaseGeometry,
    poly2_only: BaseGeometry,
    overlap: Polygon,
    prefer_first: bool,
) -> Tuple[Polygon, Polygon]:
    """Assign entire overlap to one polygon based on preference."""
    if prefer_first:
        new_poly1 = _safe_union(poly1_only, overlap)
        new_poly2 = _to_polygon(poly2_only)
    else:
        new_poly1 = _to_polygon(poly1_only)
        new_poly2 = _safe_union(poly2_only, overlap)
    return new_poly1, new_poly2


def _split_equally(
    poly1: Polygon,
    poly2: Polygon,
    overlap: Polygon,
    poly1_only: BaseGeometry,
    poly2_only: BaseGeometry,
) -> Tuple[Polygon, Polygon]:
    """Split overlap equally between two polygons using geometric splitting."""
    centroid1 = _geometry_centroid(poly1_only) or poly1.centroid
    centroid2 = _geometry_centroid(poly2_only) or poly2.centroid

    try:
        cutting_line = _build_cutting_line(overlap, centroid1, centroid2)
        split_result = shapely_split(overlap, cutting_line)
        pieces = [
            geom
            for geom in split_result.geoms
            if isinstance(geom, Polygon) and geom.area > _AREA_EPS
        ]

        if len(pieces) < 2:
            return _fallback_split(poly1, poly2, overlap, poly1_only, poly2_only)

        piece1, piece2 = _assign_pieces_to_polygons(pieces, centroid1, centroid2)
        new_poly1 = _safe_union(poly1_only, piece1)
        new_poly2 = _safe_union(poly2_only, piece2)
        return new_poly1, new_poly2
    except Exception:
        return _fallback_split(poly1, poly2, overlap, poly1_only, poly2_only)


def _build_cutting_line(
    overlap: Polygon, centroid1: Point, centroid2: Point
) -> LineString:
    """Build a line that cuts through the overlap perpendicular to centroids."""
    direction = np.array(
        [centroid2.x - centroid1.x, centroid2.y - centroid1.y], dtype=float
    )
    if np.linalg.norm(direction) < 1e-10:
        direction = _get_overlap_longest_axis(overlap)
    direction = direction / np.linalg.norm(direction)
    perp = np.array([-direction[1], direction[0]])

    bounds = overlap.bounds
    diagonal = np.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1])
    extension = diagonal * 2.0
    center = np.array([overlap.centroid.x, overlap.centroid.y])

    cut_p1 = center - perp * extension
    cut_p2 = center + perp * extension
    return LineString([cut_p1.tolist(), cut_p2.tolist()])


def _assign_pieces_to_polygons(
    pieces: List[Polygon],
    centroid1: Point,
    centroid2: Point,
) -> Tuple[Polygon, Polygon]:
    """Assign split pieces to the two original polygons based on proximity."""
    if len(pieces) == 2:
        dist1_to_first = centroid1.distance(pieces[0].centroid)
        dist1_to_second = centroid1.distance(pieces[1].centroid)
        if dist1_to_first <= dist1_to_second:
            return pieces[0], pieces[1]
        return pieces[1], pieces[0]

    pieces1 = []
    pieces2 = []
    for piece in pieces:
        dist_to_1 = centroid1.distance(piece.centroid)
        dist_to_2 = centroid2.distance(piece.centroid)
        if dist_to_1 <= dist_to_2:
            pieces1.append(piece)
        else:
            pieces2.append(piece)

    poly1_part = unary_union(pieces1) if pieces1 else Polygon()
    poly2_part = unary_union(pieces2) if pieces2 else Polygon()

    return (
        to_single_polygon(poly1_part) if not poly1_part.is_empty else Polygon(),
        to_single_polygon(poly2_part) if not poly2_part.is_empty else Polygon(),
    )


def _fallback_split(
    poly1: Polygon,
    poly2: Polygon,
    overlap: Polygon,
    poly1_only: BaseGeometry,
    poly2_only: BaseGeometry,
) -> Tuple[Polygon, Polygon]:
    """Fallback strategy using buffering when geometric split fails."""
    # Use buffering approach to erode overlap and share area.
    half_overlap_area = overlap.area / 2.0
    buffer_dist = -np.sqrt(max(half_overlap_area, 0.0) / np.pi) * 0.5

    try:
        new_poly1 = poly1.buffer(buffer_dist / 2.0)
        new_poly2 = poly2.buffer(buffer_dist / 2.0)
        if (
            isinstance(new_poly1, Polygon)
            and isinstance(new_poly2, Polygon)
            and new_poly1.is_valid
            and new_poly2.is_valid
        ):
            return new_poly1, new_poly2
    except Exception:
        pass

    return poly1, poly2


def _safe_union(base_geom: BaseGeometry, addition: BaseGeometry) -> Polygon:
    """Safely union two geometries, handling empty cases."""
    if base_geom.is_empty and addition.is_empty:
        return Polygon()
    if base_geom.is_empty:
        return _to_polygon(addition)
    if addition.is_empty:
        return _to_polygon(base_geom)
    union = unary_union([base_geom, addition])
    return _to_polygon(union)


def _to_polygon(geometry: BaseGeometry) -> Polygon:
    """Convert any geometry to a single Polygon, taking largest if MultiPolygon."""
    if isinstance(geometry, Polygon):
        return geometry
    if isinstance(geometry, MultiPolygon) and geometry.geoms:
        return to_single_polygon(geometry)
    if hasattr(geometry, "geoms"):
        # GeometryCollection or similar
        polygons = [g for g in geometry.geoms if isinstance(g, Polygon)]
        if polygons:
            return to_single_polygon(
                polygons[0] if len(polygons) == 1 else unary_union(polygons)
            )
    return Polygon()


def _geometry_centroid(geometry: BaseGeometry) -> Optional[Point]:
    """Get centroid of geometry, handling various geometry types."""
    if geometry is None or geometry.is_empty:
        return None
    if isinstance(geometry, Polygon):
        return geometry.centroid
    if isinstance(geometry, MultiPolygon) and geometry.geoms:
        largest = max(geometry.geoms, key=lambda p: p.area)
        return largest.centroid
    if hasattr(geometry, "geoms"):
        polygons = [g for g in geometry.geoms if isinstance(g, Polygon)]
        if polygons:
            largest = max(polygons, key=lambda p: p.area)
            return largest.centroid
    try:
        return geometry.centroid
    except Exception:
        return None


def _get_overlap_longest_axis(overlap: Polygon) -> np.ndarray:
    """Get the longest axis direction of the overlap bounding box."""
    bounds = overlap.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    if width >= height:
        return np.array([1.0, 0.0])
    return np.array([0.0, 1.0])


__all__ = [
    "remove_overlaps",
    "count_overlaps",
    "find_overlapping_groups",
    "resolve_overlap_pair",
]
