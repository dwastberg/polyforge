"""Public API for polygon merging operations.

This module provides efficient algorithms for merging polygons that are
either overlapping or within a specified distance (margin) of each other.
Uses spatial indexing for O(n log n) performance.
"""

from .core import merge_close_polygons, find_close_polygon_groups

__all__ = ['merge_close_polygons', 'find_close_polygon_groups']
