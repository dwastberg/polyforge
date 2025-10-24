"""Geometry clearance fixing functions.

This module provides functions for fixing geometries with low minimum clearance
by applying minimal geometric modifications. Each function targets a specific
type of clearance issue.

The minimum_clearance (from Shapely) represents the smallest distance by which
a vertex could be moved to create an invalid geometry. Higher values indicate
more robust, stable geometries.
"""

# Import public API functions
from .holes import fix_hole_too_close
from .protrusions import fix_narrow_protrusion, fix_sharp_intrusion
from .remove_protrusions import remove_narrow_protrusions
from .passages import (
    fix_narrow_passage,
    fix_near_self_intersection,
    fix_parallel_close_edges
)

# Import utility functions for backwards compatibility with tests
from .utils import (
    _find_nearest_vertex_index,
    _find_nearest_edge_index,
    _point_to_segment_distance,
    _get_vertex_neighborhood,
    _calculate_curvature_at_vertex,
    _remove_vertices_between
)

__all__ = [
    # Main public API
    'fix_hole_too_close',
    'fix_narrow_protrusion',
    'remove_narrow_protrusions',
    'fix_sharp_intrusion',
    'fix_narrow_passage',
    'fix_near_self_intersection',
    'fix_parallel_close_edges',
    # Utility functions (for advanced users and tests)
    # '_find_nearest_vertex_index',
    # '_find_nearest_edge_index',
    # '_point_to_segment_distance',
    # '_get_vertex_neighborhood',
    # '_calculate_curvature_at_vertex',
    # '_remove_vertices_between',
]
