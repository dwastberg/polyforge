"""Merge operations built on the shared ops layer."""

from polyforge.ops import merge_edge_detection as _edge
from polyforge.ops import merge_ops as _merge_ops
from polyforge.ops import merge_simple_buffer as _simple
from polyforge.ops import merge_selective_buffer as _selective
from polyforge.ops import merge_vertex_movement as _movement
from polyforge.ops import merge_boundary_extension as _boundary
from polyforge.ops import merge_convex_bridges as _bridges

find_parallel_close_edges = _edge.find_parallel_close_edges
filter_redundant_parallel_pairs = _edge.filter_redundant_parallel_pairs
merge_simple_buffer = _simple.merge_simple_buffer
merge_selective_buffer = _selective.merge_selective_buffer
merge_vertex_movement = _movement.merge_vertex_movement
merge_boundary_extension = _boundary.merge_boundary_extension
merge_convex_bridges = _bridges.merge_convex_bridges
find_close_boundary_pairs = _merge_ops.find_close_boundary_pairs
get_boundary_points_near = _merge_ops.get_boundary_points_near
insert_connection_vertices = _merge_ops.insert_connection_vertices

__all__ = [
    "find_parallel_close_edges",
    "filter_redundant_parallel_pairs",
    "merge_simple_buffer",
    "merge_selective_buffer",
    "merge_vertex_movement",
    "merge_boundary_extension",
    "merge_convex_bridges",
    "find_close_boundary_pairs",
    "get_boundary_points_near",
    "insert_connection_vertices",
]
