"""Overlap resolution utilities.

The previous package structure split these helpers between ``overlap/__init__.py``
and ``overlap/engine.py``. Consolidating them into a single module keeps all of
the overlap-specific logic in one place and eliminates the extra wrapper file
(``split.py``) that simply forwarded to ``resolve_overlap_pair``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import split as shapely_split, unary_union
from shapely.strtree import STRtree

from .core.geometry_utils import to_single_polygon
from .core.types import OverlapStrategy, coerce_enum

_AREA_EPS = 1e-10


@dataclass
class OverlapContext:
    poly1: Polygon
    poly2: Polygon
    overlap: Polygon
    poly1_only: BaseGeometry
    poly2_only: BaseGeometry


def split_overlap(
    poly1: Polygon,
    poly2: Polygon,
    overlap_strategy: OverlapStrategy = OverlapStrategy.SPLIT,
) -> Tuple[Polygon, Polygon]:
    """Backwards-compatible helper that simply calls :func:`resolve_overlap_pair`."""
    strategy = coerce_enum(overlap_strategy, OverlapStrategy)
    return resolve_overlap_pair(poly1, poly2, overlap_strategy=strategy)


def resolve_overlap_pair(
    poly1: Polygon,
    poly2: Polygon,
    overlap_strategy: OverlapStrategy = OverlapStrategy.SPLIT,
) -> Tuple[Polygon, Polygon]:
    """Resolve an overlap between two polygons using the requested strategy."""
    overlap_strategy = coerce_enum(overlap_strategy, OverlapStrategy)

    # Handle full containment separately so we don't loop forever with an unchanged pair.
    containment = _detect_containment(poly1, poly2)
    if containment is not None:
        return _resolve_containment(containment, overlap_strategy)

    ctx = _build_context(poly1, poly2)
    if ctx is None:
        return poly1, poly2

    if overlap_strategy == OverlapStrategy.LARGEST:
        prefer_first = poly1.area >= poly2.area
        return _assign_entire_overlap(ctx, prefer_first)
    if overlap_strategy == OverlapStrategy.SMALLEST:
        prefer_first = poly1.area <= poly2.area
        return _assign_entire_overlap(ctx, prefer_first)

    return _split_equally(ctx)


def remove_overlaps(
    polygons: List[Polygon],
    overlap_strategy: OverlapStrategy = OverlapStrategy.SPLIT,
    max_iterations: int = 100,
) -> List[Polygon]:
    """Remove overlaps from a list of polygons using the shared overlap engine."""
    if not polygons:
        return []

    strategy = coerce_enum(overlap_strategy, OverlapStrategy)
    result = list(polygons)
    iteration = 0
    while iteration < max_iterations:
        tree = STRtree(result)
        overlaps = []
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
                area = getattr(overlap, "area", 0.0)
                if area > 1e-10:
                    overlaps.append((area, i, j))

        if not overlaps:
            break

        # Resolve pairs in descending overlap area to prioritize the worst offenders first.
        overlaps.sort(key=lambda item: item[0], reverse=True)
        changed = False

        for _, i, j in overlaps:
            poly_i = result[i]
            poly_j = result[j]

            if poly_i.touches(poly_j):
                continue

            current_overlap = poly_i.intersection(poly_j)
            if getattr(current_overlap, "area", 0.0) <= 1e-10:
                continue

            result[i], result[j] = resolve_overlap_pair(
                poly_i,
                poly_j,
                overlap_strategy=strategy,
            )
            changed = True

        if not changed:
            break

        iteration += 1

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


def _detect_containment(poly1: Polygon, poly2: Polygon) -> Optional[Tuple[Polygon, Polygon, bool]]:
    """Return (outer, inner, outer_is_first) if one polygon fully contains the other."""
    if poly1.contains(poly2):
        return poly1, poly2, True
    if poly2.contains(poly1):
        return poly2, poly1, False
    return None


def _resolve_containment(
    containment: Tuple[Polygon, Polygon, bool],
    overlap_strategy: OverlapStrategy,
) -> Tuple[Polygon, Polygon]:
    outer, inner, outer_is_first = containment

    if overlap_strategy == OverlapStrategy.LARGEST:
        # Assign the entire overlap to the outer polygon; drop the inner to avoid double counting.
        outer_result = outer
        inner_result = Polygon()
    else:
        # Default and SMALLEST: preserve the inner and carve a hole from the outer.
        carved = outer.difference(inner)
        if isinstance(carved, MultiPolygon):
            carved = to_single_polygon(carved)
        if not isinstance(carved, Polygon):
            carved = Polygon()
        outer_result = carved
        inner_result = inner

    if outer_is_first:
        return outer_result, inner_result
    return inner_result, outer_result


def _build_context(poly1: Polygon, poly2: Polygon) -> Optional[OverlapContext]:
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

    return OverlapContext(
        poly1=poly1,
        poly2=poly2,
        overlap=overlap,
        poly1_only=poly1.difference(overlap),
        poly2_only=poly2.difference(overlap),
    )


def _assign_entire_overlap(ctx: OverlapContext, prefer_first: bool) -> Tuple[Polygon, Polygon]:
    if prefer_first:
        new_poly1 = _safe_union(ctx.poly1_only, ctx.overlap)
        new_poly2 = _to_polygon(ctx.poly2_only)
    else:
        new_poly1 = _to_polygon(ctx.poly1_only)
        new_poly2 = _safe_union(ctx.poly2_only, ctx.overlap)
    return new_poly1, new_poly2


def _split_equally(ctx: OverlapContext) -> Tuple[Polygon, Polygon]:
    centroid1 = _geometry_centroid(ctx.poly1_only) or ctx.poly1.centroid
    centroid2 = _geometry_centroid(ctx.poly2_only) or ctx.poly2.centroid

    try:
        cutting_line = _build_cutting_line(ctx.overlap, centroid1, centroid2)
        split_result = shapely_split(ctx.overlap, cutting_line)
        pieces = [
            geom for geom in split_result.geoms if isinstance(geom, Polygon) and geom.area > _AREA_EPS
        ]

        if len(pieces) < 2:
            return _fallback_split(ctx)

        piece1, piece2 = _assign_pieces_to_polygons(pieces, centroid1, centroid2)
        new_poly1 = _safe_union(ctx.poly1_only, piece1)
        new_poly2 = _safe_union(ctx.poly2_only, piece2)
        return new_poly1, new_poly2
    except Exception:
        return _fallback_split(ctx)


def _build_cutting_line(overlap: Polygon, centroid1: Point, centroid2: Point) -> LineString:
    direction = np.array([centroid2.x - centroid1.x, centroid2.y - centroid1.y], dtype=float)
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


def _fallback_split(ctx: OverlapContext) -> Tuple[Polygon, Polygon]:
    """Last-resort overlap resolution: subtract the overlap from the polygon
    that has the smaller non-overlapping remainder, so the larger polygon
    keeps more of its original shape."""
    try:
        # Subtract overlap from each polygon and compare what remains
        remainder1 = ctx.poly1.difference(ctx.overlap)
        remainder2 = ctx.poly2.difference(ctx.overlap)

        r1_area = getattr(remainder1, "area", 0.0)
        r2_area = getattr(remainder2, "area", 0.0)

        # Subtract overlap from the polygon with the smaller remainder
        # (i.e., the one that "owns" less non-overlapping area)
        if r1_area <= r2_area:
            new_poly1 = _to_polygon(remainder1) if not remainder1.is_empty else Polygon()
            new_poly2 = ctx.poly2
        else:
            new_poly1 = ctx.poly1
            new_poly2 = _to_polygon(remainder2) if not remainder2.is_empty else Polygon()

        if new_poly1.is_valid and new_poly2.is_valid:
            return new_poly1, new_poly2
    except Exception:
        pass

    return ctx.poly1, ctx.poly2


def _safe_union(base_geom: BaseGeometry, addition: BaseGeometry) -> Polygon:
    if base_geom.is_empty and addition.is_empty:
        return Polygon()
    if base_geom.is_empty:
        return _to_polygon(addition)
    if addition.is_empty:
        return _to_polygon(base_geom)
    union = unary_union([base_geom, addition])
    return _to_polygon(union)


def _to_polygon(geometry: BaseGeometry) -> Polygon:
    if isinstance(geometry, Polygon):
        return geometry
    if isinstance(geometry, MultiPolygon) and geometry.geoms:
        return to_single_polygon(geometry)
    if hasattr(geometry, "geoms"):
        polys = [geom for geom in geometry.geoms if isinstance(geom, Polygon)]
        if polys:
            return to_single_polygon(unary_union(polys))
    return Polygon()


def _geometry_centroid(geometry: BaseGeometry) -> Optional[Point]:
    if geometry.is_empty:
        return None
    centroid = geometry.centroid
    if isinstance(centroid, Point):
        return centroid
    if hasattr(centroid, "geoms"):
        for geom in centroid.geoms:
            if isinstance(geom, Point):
                return geom
    return None


def _get_overlap_longest_axis(overlap: Polygon) -> np.ndarray:
    bounds = overlap.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    if height > width:
        return np.array([0.0, 1.0])
    return np.array([1.0, 0.0])


__all__ = [
    "count_overlaps",
    "find_overlapping_groups",
    "remove_overlaps",
    "resolve_overlap_pair",
    "split_overlap",
]
