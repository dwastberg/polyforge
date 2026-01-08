"""Geometry clearance fixing functions.

This module provides functions for fixing geometries with low minimum clearance
by applying minimal geometric modifications. Each function targets a specific
type of clearance issue.

The minimum_clearance (from Shapely) represents the smallest distance by which
a vertex could be moved to create an invalid geometry. Higher values indicate
more robust, stable geometries.
"""

# Import public API functions from the ops layer
from polyforge.ops.clearance import (
    fix_hole_too_close,
    fix_narrow_protrusion,
    fix_sharp_intrusion,
    remove_narrow_protrusions,
    fix_narrow_passage,
    fix_near_self_intersection,
    fix_parallel_close_edges,
)
from .fix_clearance import (
    fix_clearance,
    diagnose_clearance,
    ClearanceIssue,
    ClearanceDiagnosis,
    ClearanceFixSummary,
)

__all__ = [
    # Main public API
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
]
