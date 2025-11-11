"""Backward-compatible imports for cleanup helpers.

The real implementations now live under :mod:`polyforge.ops.cleanup_ops`. This
module exists so legacy imports (``from polyforge.core import cleanup``) keep
working while the codebase migrates toward the ops-first architecture.
"""

from polyforge.ops.cleanup_ops import (
    CleanupConfig,
    cleanup_polygon,
    remove_small_holes,
    remove_narrow_holes,
)

__all__ = [
    "CleanupConfig",
    "cleanup_polygon",
    "remove_small_holes",
    "remove_narrow_holes",
]
