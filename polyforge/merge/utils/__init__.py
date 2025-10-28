"""Utility functions for polygon merging."""

from .vertex_insertion import insert_connection_vertices
from .boundary_analysis import find_close_boundary_pairs, get_boundary_points_near
from .edge_detection import find_parallel_close_edges, filter_redundant_parallel_pairs

__all__ = [
    'insert_connection_vertices',
    'find_close_boundary_pairs',
    'get_boundary_points_near',
    'find_parallel_close_edges',
    'filter_redundant_parallel_pairs',
]
