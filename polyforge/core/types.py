"""Type definitions for polyforge operations.

This module defines enums for strategy parameters throughout the library.
"""

from enum import Enum


class OverlapStrategy(Enum):
    """Strategy for resolving overlaps between polygons.

    Attributes:
        SPLIT: Split overlap 50/50 between polygons
        LARGEST: Assign entire overlap to larger polygon
        SMALLEST: Assign entire overlap to smaller polygon

    Examples:
        >>> from polyforge import remove_overlaps, OverlapStrategy
        >>> result = remove_overlaps(polygons, overlap_strategy=OverlapStrategy.SPLIT)
    """
    SPLIT = 'split'
    LARGEST = 'largest'
    SMALLEST = 'smallest'


class MergeStrategy(Enum):
    """Strategy for merging close polygons.

    Attributes:
        SIMPLE_BUFFER: Classic expand-contract (fast, may change shape)
        SELECTIVE_BUFFER: Buffer only near gaps (good balance, default)
        VERTEX_MOVEMENT: Move vertices toward each other (precise)
        BOUNDARY_EXTENSION: Extend parallel edges (good for architectural features)
        CONVEX_BRIDGES: Convex hull bridges (smooth connections)

    Examples:
        >>> from polyforge import merge_close_polygons, MergeStrategy
        >>> result = merge_close_polygons(polygons, merge_strategy=MergeStrategy.SELECTIVE_BUFFER)
    """
    SIMPLE_BUFFER = 'simple_buffer'
    SELECTIVE_BUFFER = 'selective_buffer'
    VERTEX_MOVEMENT = 'vertex_movement'
    BOUNDARY_EXTENSION = 'boundary_extension'
    CONVEX_BRIDGES = 'convex_bridges'


class RepairStrategy(Enum):
    """Strategy for repairing invalid geometries.

    Attributes:
        AUTO: Automatically detect and repair (tries multiple strategies)
        BUFFER: Use buffer(0) trick (fast, good for self-intersections)
        SIMPLIFY: Simplify and rebuild (reduces complexity)
        RECONSTRUCT: Reconstruct from scratch using convex hull
        STRICT: Only conservative fixes that preserve shape

    Examples:
        >>> from polyforge import repair_geometry, RepairStrategy
        >>> fixed = repair_geometry(invalid_poly, repair_strategy=RepairStrategy.AUTO)
        >>> # For self-intersections, buffer is often best
        >>> fixed = repair_geometry(invalid_poly, repair_strategy=RepairStrategy.BUFFER)
    """
    AUTO = 'auto'
    BUFFER = 'buffer'
    SIMPLIFY = 'simplify'
    RECONSTRUCT = 'reconstruct'
    STRICT = 'strict'


class SimplifyAlgorithm(Enum):
    """Algorithm for geometry simplification.

    Attributes:
        RDP: Ramer-Douglas-Peucker (fast, good general purpose)
        VW: Visvalingam-Whyatt (slower, better visual quality)
        VWP: Topology-preserving Visvalingam-Whyatt (slowest, guaranteed valid)

    Examples:
        >>> from polyforge import simplify_rdp, SimplifyAlgorithm
        >>> # Enum can be used to document algorithm choice
        >>> algorithm = SimplifyAlgorithm.RDP
        >>> if algorithm == SimplifyAlgorithm.RDP:
        ...     result = simplify_rdp(polygon, epsilon=1.0)
    """
    RDP = 'rdp'
    VW = 'vw'
    VWP = 'vwp'


class CollapseMode(Enum):
    """Mode for collapsing short edges.

    Attributes:
        MIDPOINT: Snap both vertices to their midpoint (default, balanced)
        FIRST: Keep first vertex, remove second (preserves start)
        LAST: Remove first vertex, keep second (preserves end)

    Examples:
        >>> from polyforge import collapse_short_edges, CollapseMode
        >>> result = collapse_short_edges(poly, min_length=0.1, snap_mode=CollapseMode.MIDPOINT)
    """
    MIDPOINT = 'midpoint'
    FIRST = 'first'
    LAST = 'last'


class HoleStrategy(Enum):
    """Strategy for fixing holes too close to exterior.

    Attributes:
        REMOVE: Remove holes that are too close (default, safest)
        SHRINK: Make holes smaller via negative buffer
        MOVE: Move holes away from exterior (experimental)

    Examples:
        >>> from polyforge.clearance import fix_hole_too_close
        >>> from polyforge.core.types import HoleStrategy
        >>> fixed = fix_hole_too_close(poly, min_clearance=1.0, strategy=HoleStrategy.REMOVE)
    """
    REMOVE = 'remove'
    SHRINK = 'shrink'
    MOVE = 'move'


class PassageStrategy(Enum):
    """Strategy for fixing narrow passages.

    Attributes:
        WIDEN: Move vertices apart to widen passage (default, preserves single polygon)
        SPLIT: Split into separate polygons at narrow point

    Examples:
        >>> from polyforge.clearance import fix_narrow_passage
        >>> from polyforge.core.types import PassageStrategy
        >>> fixed = fix_narrow_passage(poly, min_clearance=1.0, strategy=PassageStrategy.WIDEN)
    """
    WIDEN = 'widen'
    SPLIT = 'split'


class IntrusionStrategy(Enum):
    """Strategy for fixing sharp intrusions.

    Attributes:
        FILL: Fill intrusion with straight edge (default)
        SMOOTH: Apply smoothing to widen intrusion
        SIMPLIFY: Use vertex simplification

    Examples:
        >>> from polyforge.clearance import fix_sharp_intrusion
        >>> from polyforge.core.types import IntrusionStrategy
        >>> fixed = fix_sharp_intrusion(poly, min_clearance=1.0, strategy=IntrusionStrategy.FILL)
    """
    FILL = 'fill'
    SMOOTH = 'smooth'
    SIMPLIFY = 'simplify'


class IntersectionStrategy(Enum):
    """Strategy for fixing near self-intersections.

    Attributes:
        SIMPLIFY: Remove vertices causing near-intersection (default)
        BUFFER: Use buffer to fix topology
        SMOOTH: Apply smoothing

    Examples:
        >>> from polyforge.clearance import fix_near_self_intersection
        >>> from polyforge.core.types import IntersectionStrategy
        >>> fixed = fix_near_self_intersection(poly, min_clearance=1.0, strategy=IntersectionStrategy.SIMPLIFY)
    """
    SIMPLIFY = 'simplify'
    BUFFER = 'buffer'
    SMOOTH = 'smooth'


class EdgeStrategy(Enum):
    """Strategy for fixing parallel close edges.

    Attributes:
        SIMPLIFY: Simplify to remove close parallel edges (default)
        BUFFER: Use buffer to adjust edges

    Examples:
        >>> from polyforge.clearance import fix_parallel_close_edges
        >>> from polyforge.core.types import EdgeStrategy
        >>> fixed = fix_parallel_close_edges(poly, min_clearance=1.0, strategy=EdgeStrategy.SIMPLIFY)
    """
    SIMPLIFY = 'simplify'
    BUFFER = 'buffer'


__all__ = [
    'OverlapStrategy',
    'MergeStrategy',
    'RepairStrategy',
    'SimplifyAlgorithm',
    'CollapseMode',
    'HoleStrategy',
    'PassageStrategy',
    'IntrusionStrategy',
    'IntersectionStrategy',
    'EdgeStrategy',
]
