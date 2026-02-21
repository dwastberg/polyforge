from __future__ import annotations

from enum import Enum
from typing import TypeVar

EnumT = TypeVar("EnumT", bound=Enum)


class OverlapStrategy(Enum):
    SPLIT = "split"
    LARGEST = "largest"
    SMALLEST = "smallest"


class MergeStrategy(Enum):
    SIMPLE_BUFFER = "simple_buffer"
    SELECTIVE_BUFFER = "selective_buffer"
    VERTEX_MOVEMENT = "vertex_movement"
    BOUNDARY_EXTENSION = "boundary_extension"
    CONVEX_BRIDGES = "convex_bridges"


class RepairStrategy(Enum):
    AUTO = "auto"
    BUFFER = "buffer"
    SIMPLIFY = "simplify"
    RECONSTRUCT = "reconstruct"
    STRICT = "strict"


class SimplifyAlgorithm(Enum):
    """
    RDP: Ramer-Douglas-Peucker (fast, good general purpose)
    VW: Visvalingam-Whyatt (slower, better visual quality)
    VWP: Topology-preserving Visvalingam-Whyatt (slowest, guaranteed valid)
    """

    RDP = "rdp"
    VW = "vw"
    VWP = "vwp"


class CollapseMode(Enum):
    MIDPOINT = "midpoint"
    FIRST = "first"
    LAST = "last"


class HoleStrategy(Enum):
    REMOVE = "remove"
    SHRINK = "shrink"
    MOVE = "move"


class PassageStrategy(Enum):
    WIDEN = "widen"
    SPLIT = "split"
    ARAP = "arap"


class IntrusionStrategy(Enum):
    FILL = "fill"
    SMOOTH = "smooth"
    SIMPLIFY = "simplify"


class IntersectionStrategy(Enum):
    SIMPLIFY = "simplify"
    BUFFER = "buffer"
    SMOOTH = "smooth"


__all__ = [
    "OverlapStrategy",
    "MergeStrategy",
    "RepairStrategy",
    "SimplifyAlgorithm",
    "CollapseMode",
    "HoleStrategy",
    "PassageStrategy",
    "IntrusionStrategy",
    "IntersectionStrategy",
    "coerce_enum",
]


def coerce_enum(value: EnumT | str, enum_cls: type[EnumT]) -> EnumT:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        return enum_cls(value)
    raise ValueError(f"Cannot coerce {value!r} to {enum_cls.__name__}")
