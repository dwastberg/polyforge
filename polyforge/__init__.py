"""Polyforge - Polygon processing and manipulation library.

This library provides utilities for processing, simplifying, and manipulating
polygon geometries using Shapely.
"""


# Simplification functions
from .simplify import (
    simplify_rdp,
    simplify_vw,
    simplify_vwp,
    snap_short_edges,
    remove_duplicate_vertices,
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
from .topology import conform_boundaries

# Geometry fixing functions
from .fix import fix_geometry, diagnose_geometry, batch_fix_geometries, GeometryFixError

__all__ = [

    # Simplification
    'simplify_rdp',
    'simplify_vw',
    'simplify_vwp',
    'snap_short_edges',
    'remove_duplicate_vertices',
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
    'conform_boundaries',

    # Geometry fixing
    'fix_geometry',
    'diagnose_geometry',
    'batch_fix_geometries',
    'GeometryFixError',
]
