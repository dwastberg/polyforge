"""Core types and utilities for polyforge.

This module provides type definitions, enums, exceptions, and core utilities
used throughout the library.
"""

from .types import (
    OverlapStrategy,
    MergeStrategy,
    RepairStrategy,
    SimplifyAlgorithm,
    CollapseMode,
    HoleStrategy,
    PassageStrategy,
    IntrusionStrategy,
    IntersectionStrategy,
    EdgeStrategy,
)

from .errors import (
    PolyforgeError,
    ValidationError,
    RepairError,
    OverlapResolutionError,
    MergeError,
    ClearanceError,
    ConfigurationError,
)

__all__ = [
    # Strategy enums
    'OverlapStrategy',
    'MergeStrategy',
    'RepairStrategy',
    'SimplifyAlgorithm',
    'CollapseMode',
    'HoleStrategy',
    'PassageStrategy',
    'IntrusionStrategy',
    'IntersectionStrategy',
    'EdgeStrategy',

    # Exceptions
    'PolyforgeError',
    'ValidationError',
    'RepairError',
    'OverlapResolutionError',
    'MergeError',
    'ClearanceError',
    'ConfigurationError',
]
