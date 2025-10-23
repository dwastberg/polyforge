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
    fix_hole_too_close,
    fix_narrow_protrusion,
    fix_sharp_intrusion,
    fix_narrow_passage,
    fix_near_self_intersection,
    fix_parallel_close_edges,
)

# Split overlap functions
from .split import split_overlap

__all__ = [

    # Simplification
    'simplify_rdp',
    'simplify_vw',
    'simplify_vwp',
    'snap_short_edges',
    'remove_duplicate_vertices',
    'remove_small_holes',

    # Clearance fixing
    'fix_hole_too_close',
    'fix_narrow_protrusion',
    'fix_sharp_intrusion',
    'fix_narrow_passage',
    'fix_near_self_intersection',
    'fix_parallel_close_edges',

    # Overlap splitting
    'split_overlap',
]
