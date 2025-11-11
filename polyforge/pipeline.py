"""Minimal pipeline runner used by the new repair implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from shapely.geometry.base import BaseGeometry

from .core.constraints import ConstraintStatus, GeometryConstraints, MergeConstraints

PipelineStep = Callable[[BaseGeometry, "PipelineContext"], "StepResult"]


@dataclass
class FixConfig:
    """Subset of constraint settings needed by the pipeline steps."""

    min_clearance: Optional[float] = None
    min_area_ratio: float = 0.0
    must_be_valid: bool = True
    allow_multipolygon: bool = True
    cleanup: bool = True


@dataclass
class PipelineContext:
    """Runtime context shared across pipeline steps."""

    original: BaseGeometry
    constraints: GeometryConstraints
    config: FixConfig
    merge_constraints: Optional[MergeConstraints] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class StepResult:
    """Outcome of running a single pipeline step."""

    name: str
    geometry: BaseGeometry
    changed: bool
    message: str = ""


def config_from_constraints(constraints: GeometryConstraints) -> FixConfig:
    """Helper that translates the old constraint dataclass into ``FixConfig``."""
    return FixConfig(
        min_clearance=constraints.min_clearance,
        min_area_ratio=constraints.min_area_ratio,
        must_be_valid=constraints.must_be_valid,
        allow_multipolygon=constraints.allow_multipolygon,
        cleanup=True,
    )


def run_steps(
    initial_geometry: BaseGeometry,
    steps: List[PipelineStep],
    context: PipelineContext,
    max_passes: int = 10,
) -> Tuple[BaseGeometry, ConstraintStatus, List[StepResult]]:
    """Execute the supplied steps until constraints are satisfied or progress stalls."""
    geometry = initial_geometry
    history: List[StepResult] = []

    for _ in range(max_passes):
        iteration_changed = False
        for step in steps:
            result = step(geometry, context)
            geometry = result.geometry
            history.append(result)
            iteration_changed = iteration_changed or result.changed

        status = context.constraints.check(geometry, context.original)
        if status.all_satisfied():
            return geometry, status, history

        if not iteration_changed:
            return geometry, status, history

    status = context.constraints.check(geometry, context.original)
    return geometry, status, history


__all__ = [
    "FixConfig",
    "PipelineContext",
    "PipelineStep",
    "StepResult",
    "config_from_constraints",
    "run_steps",
]
