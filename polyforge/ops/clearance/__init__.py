"""Low-level clearance helpers used by the high-level API."""

from .holes import fix_hole_too_close
from .protrusions import fix_narrow_protrusion, fix_sharp_intrusion, fill_narrow_wedge
from .remove_protrusions import remove_narrow_protrusions
from .passages import (
    fix_narrow_passage,
    fix_near_self_intersection,
    fix_parallel_close_edges,
    _erode_dilate_fix,
)
from .utils import (
    _find_nearest_vertex_index,
    _find_nearest_edge_index,
    _point_to_segment_distance,
    _point_to_line_perpendicular_distance,
    _get_vertex_neighborhood,
    _calculate_curvature_at_vertex,
    _remove_vertices_between,
)

__all__ = [
    "fix_hole_too_close",
    "fix_narrow_protrusion",
    "fix_sharp_intrusion",
    "fill_narrow_wedge",
    "remove_narrow_protrusions",
    "fix_narrow_passage",
    "fix_near_self_intersection",
    "fix_parallel_close_edges",
    "_erode_dilate_fix",
    "_find_nearest_vertex_index",
    "_find_nearest_edge_index",
    "_point_to_segment_distance",
    "_point_to_line_perpendicular_distance",
    "_get_vertex_neighborhood",
    "_calculate_curvature_at_vertex",
    "_remove_vertices_between",
]
