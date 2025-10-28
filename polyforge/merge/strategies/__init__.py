"""Merge strategy implementations."""

from .simple_buffer import merge_simple_buffer
from .selective_buffer import merge_selective_buffer
from .vertex_movement import merge_vertex_movement
from .boundary_extension import merge_boundary_extension
from .convex_bridges import merge_convex_bridges

__all__ = [
    'merge_simple_buffer',
    'merge_selective_buffer',
    'merge_vertex_movement',
    'merge_boundary_extension',
    'merge_convex_bridges',
]
