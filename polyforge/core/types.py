"""Type definitions for polyforge operations.

This module defines enums for strategy parameters throughout the library.
"""

from enum import Enum
from typing import Type, TypeVar, Union

EnumT = TypeVar("EnumT", bound=Enum)


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

    Note: Internal use only. Users should call simplification functions directly
    (simplify_rdp, simplify_vw, simplify_vwp) rather than using this enum.

    Attributes:
        RDP: Ramer-Douglas-Peucker (fast, good general purpose)
        VW: Visvalingam-Whyatt (slower, better visual quality)
        VWP: Topology-preserving Visvalingam-Whyatt (slowest, guaranteed valid)
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

    Note: Internal use only - not exported in public API.
    """
    REMOVE = 'remove'
    SHRINK = 'shrink'
    MOVE = 'move'


class PassageStrategy(Enum):
    """Strategy for fixing narrow passages.

    Note: Internal use only - not exported in public API.
    """
    WIDEN = 'widen'
    SPLIT = 'split'
    ARAP = 'arap'


class IntrusionStrategy(Enum):
    """Strategy for fixing sharp intrusions.

    Note: Internal use only - not exported in public API.
    """
    FILL = 'fill'
    SMOOTH = 'smooth'
    SIMPLIFY = 'simplify'


class IntersectionStrategy(Enum):
    """Strategy for fixing near self-intersections.

    Note: Internal use only - not exported in public API.
    """
    SIMPLIFY = 'simplify'
    BUFFER = 'buffer'
    SMOOTH = 'smooth'


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
    'coerce_enum',
]


def coerce_enum(value: Union[EnumT, str], enum_cls: Type[EnumT]) -> EnumT:
    """Convert ``value`` to ``enum_cls`` without forcing callers to import enums."""
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        return enum_cls(value)
    raise ValueError(f"Cannot coerce {value!r} to {enum_cls.__name__}")
