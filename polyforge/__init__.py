"""Polyforge - Polygon processing and manipulation library.

This library provides utilities for processing, simplifying, and manipulating
polygon geometries using Shapely.
"""


# Simplification functions
from .simplify import (
    simplify_rdp,
    simplify_vw,
    simplify_vwp,
    collapse_short_edges,
    deduplicate_vertices,
    remove_small_holes,
)

# Clearance fixing functions
from .clearance import (
    fix_clearance,
    fix_hole_too_close,
    fix_narrow_protrusion,
    remove_narrow_protrusions,
    fix_sharp_intrusion,
    fix_narrow_passage,
    fix_near_self_intersection,
    fix_parallel_close_edges,
)

# Split overlap functions
from .split import split_overlap

# Overlap removal functions
from .overlap import remove_overlaps, count_overlaps, find_overlapping_groups

# Merge functions
from .merge import merge_close_polygons

# Topology functions
from .topology import align_boundaries

# Geometry repair functions
from .fix import (
    repair_geometry,
    analyze_geometry,
    batch_repair_geometries,
)

# Core types (enums)
from .core import (
    OverlapStrategy,
    MergeStrategy,
    RepairStrategy,
    SimplifyAlgorithm,
    CollapseMode,
)

# Core exceptions
from .core import (
    PolyforgeError,
    ValidationError,
    RepairError,
    OverlapResolutionError,
    MergeError,
    ClearanceError,
    ConfigurationError,
)

__all__ = [

    # Simplification
    'simplify_rdp',
    'simplify_vw',
    'simplify_vwp',
    'collapse_short_edges',
    'deduplicate_vertices',
    'remove_small_holes',

    # Clearance fixing
    'fix_clearance',
    'fix_hole_too_close',
    'fix_narrow_protrusion',
    'remove_narrow_protrusions',
    'fix_sharp_intrusion',
    'fix_narrow_passage',
    'fix_near_self_intersection',
    'fix_parallel_close_edges',

    # Overlap handling
    'split_overlap',
    'remove_overlaps',
    'count_overlaps',
    'find_overlapping_groups',

    # Merge
    'merge_close_polygons',

    # Topology
    'align_boundaries',

    # Geometry repair
    'repair_geometry',
    'analyze_geometry',
    'batch_repair_geometries',

    # Core types (enums)
    'OverlapStrategy',
    'MergeStrategy',
    'RepairStrategy',
    'SimplifyAlgorithm',
    'CollapseMode',

    # Core exceptions
    'PolyforgeError',
    'ValidationError',
    'RepairError',
    'OverlapResolutionError',
    'MergeError',
    'ClearanceError',
    'ConfigurationError',
]
