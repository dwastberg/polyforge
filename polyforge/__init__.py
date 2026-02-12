"""Polyforge - Polygon processing and manipulation library.

This library provides utilities for processing, simplifying, and manipulating
polygon geometries using Shapely.
"""

__version__ = "0.1.0a2"

# Simplification functions
from .simplify import (
    simplify_rdp,
    simplify_vw,
    simplify_vwp,
    collapse_short_edges,
    deduplicate_vertices,
    remove_small_holes,
    remove_narrow_holes,
)

# Clearance fixing functions
from .clearance import (
    fix_clearance,
    diagnose_clearance,
    ClearanceIssue,
    ClearanceDiagnosis,
    ClearanceFixSummary,
    fix_hole_too_close,
    fix_narrow_protrusion,
    remove_narrow_protrusions,
    fix_sharp_intrusion,
    fix_narrow_passage,
    fix_near_self_intersection,
    fix_parallel_close_edges,
)

# Overlap handling functions
from .overlap import (
    split_overlap,
    remove_overlaps,
    count_overlaps,
    find_overlapping_groups,
    resolve_overlap_pair,
)

# Merge functions
from .merge import merge_close_polygons, find_close_polygon_groups

# Topology functions
from .topology import align_boundaries

# Geometry repair functions
from .repair import (
    repair_geometry,
    analyze_geometry,
    batch_repair_geometries,
)

# Robust constraint-aware repair
from .repair.robust import (
    robust_fix_geometry,
    robust_fix_batch,
)

# Core types (enums)
from .core import (
    OverlapStrategy,
    MergeStrategy,
    RepairStrategy,
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
    FixWarning,
)

# Constraint framework
from .core import (
    GeometryConstraints,
    ConstraintStatus,
    ConstraintViolation,
    ConstraintType,
    MergeConstraints,
)

__all__ = [

    # Simplification
    'simplify_rdp',
    'simplify_vw',
    'simplify_vwp',
    'collapse_short_edges',
    'deduplicate_vertices',
    'remove_small_holes',
    'remove_narrow_holes',

    # Clearance fixing
    'fix_clearance',
    'diagnose_clearance',
    'ClearanceIssue',
    'ClearanceDiagnosis',
    'ClearanceFixSummary',
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
    'resolve_overlap_pair',

    # Merge
    'merge_close_polygons',
    'find_close_polygon_groups',

    # Topology
    'align_boundaries',

    # Geometry repair
    'repair_geometry',
    'analyze_geometry',
    'batch_repair_geometries',

    # Robust constraint-aware repair
    'robust_fix_geometry',
    'robust_fix_batch',

    # Core types (enums)
    'OverlapStrategy',
    'MergeStrategy',
    'RepairStrategy',
    'CollapseMode',

    # Core exceptions
    'PolyforgeError',
    'ValidationError',
    'RepairError',
    'OverlapResolutionError',
    'MergeError',
    'ClearanceError',
    'ConfigurationError',
    'FixWarning',

    # Constraint framework
    'GeometryConstraints',
    'ConstraintStatus',
    'ConstraintViolation',
    'ConstraintType',
    'MergeConstraints',
]
