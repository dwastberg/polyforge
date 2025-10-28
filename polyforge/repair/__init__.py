"""Public API for geometry repair operations."""

from .core import repair_geometry, batch_repair_geometries
from .analysis import analyze_geometry

__all__ = [
    'repair_geometry',
    'analyze_geometry',
    'batch_repair_geometries',
]
