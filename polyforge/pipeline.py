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
    original: BaseGeometry
    constraints: GeometryConstraints
    config: FixConfig
    merge_constraints: Optional[MergeConstraints] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    # Metric caching for performance (hidden from repr)
    _geometry_cache: Optional[BaseGeometry] = field(default=None, repr=False)
    _metrics_cache: Optional[Dict] = field(default=None, repr=False)

    def get_metrics(self, geometry: BaseGeometry) -> Dict:
        """Get metrics for geometry with caching.

        Only recalculates metrics if geometry has changed since last call.
        Uses geometry.equals() for comparison to detect actual changes.

        Args:
            geometry: Geometry to measure

        Returns:
            Dict with keys: is_valid, is_empty, clearance, area, area_ratio
        """
        from .metrics import measure_geometry

        # Check if we can use cached metrics
        if self._geometry_cache is not None and geometry.equals(self._geometry_cache):
            return self._metrics_cache

        # Calculate new metrics and cache them
        skip_clearance = (self.constraints.min_clearance is None)
        metrics = measure_geometry(geometry, self.original, skip_clearance=skip_clearance)

        self._geometry_cache = geometry
        self._metrics_cache = metrics

        return metrics


@dataclass
class StepResult:
    """Result of a pipeline step."""
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
    """Execute the supplied steps until constraints are satisfied or progress stalls.
    
    Args:
        initial_geometry: Geometry to fix
        steps: List of pipeline steps to execute
        context: Pipeline context
        max_passes: Maximum number of passes through the steps

    Returns:
        Tuple of (final_geometry, final_status, history)
    """
    geometry = initial_geometry
    history: List[StepResult] = []

    for _ in range(max_passes):
        iteration_changed = False
        for step in steps:
            result = step(geometry, context)
            geometry = result.geometry
            history.append(result)
            iteration_changed = iteration_changed or result.changed

        # Use cached metrics for end-of-iteration check
        metrics = context.get_metrics(geometry)
        status = context.constraints.check(geometry, context.original, metrics=metrics)
        if status.all_satisfied():
            return geometry, status, history

        if not iteration_changed:
            return geometry, status, history

    # Final check also uses caching
    metrics = context.get_metrics(geometry)
    status = context.constraints.check(geometry, context.original, metrics=metrics)
    return geometry, status, history


__all__ = [
    "FixConfig",
    "PipelineContext",
    "PipelineStep",
    "StepResult",
    "config_from_constraints",
    "run_steps",
]
