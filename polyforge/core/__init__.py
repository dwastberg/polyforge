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
    coerce_enum,
)

from .errors import (
    PolyforgeError,
    ValidationError,
    RepairError,
    OverlapResolutionError,
    MergeError,
    ClearanceError,
    ConfigurationError,
    FixWarning,
)

from .constraints import (
    ConstraintType,
    ConstraintViolation,
    ConstraintStatus,
    GeometryConstraints,
    MergeConstraints,
)

__all__ = [
    # Strategy enums (public)
    'OverlapStrategy',
    'MergeStrategy',
    'RepairStrategy',
    'SimplifyAlgorithm',
    'CollapseMode',
    'coerce_enum',

    # Exceptions
    'PolyforgeError',
    'ValidationError',
    'RepairError',
    'OverlapResolutionError',
    'MergeError',
    'ClearanceError',
    'ConfigurationError',
    'FixWarning',

    # Constraint framework
    'ConstraintType',
    'ConstraintViolation',
    'ConstraintStatus',
    'GeometryConstraints',
    'MergeConstraints',
]
