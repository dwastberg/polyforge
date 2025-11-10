"""Composable fix stages used by the robust repair orchestrators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.geometry.base import BaseGeometry

from ..core.constraints import GeometryConstraints, MergeConstraints
from ..core.types import RepairStrategy
from ..merge import merge_close_polygons
from ..core.cleanup import CleanupConfig, cleanup_polygon
from .core import repair_geometry
from .transaction import FixTransaction
from ..clearance.fix_clearance import fix_clearance


Predicate = Callable[["StageContext"], bool]
Runner = Callable[["StageContext"], "StageResult"]


@dataclass
class StageResult:
    """Outcome of running a single stage."""

    name: str
    executed: bool
    committed: bool
    changed: bool
    message: str = ""
    error: Optional[str] = None

    def summary(self) -> str:
        """Human readable description for logs / warnings."""
        if not self.executed:
            return f"{self.name}: skipped"
        status = "committed" if self.committed else "rolled back"
        if not self.changed:
            status = "no-op"
        base = f"{self.name}: {status}"
        if self.message:
            base = f"{base} ({self.message})"
        if self.error:
            base = f"{base} error={self.error}"
        return base


@dataclass
class StageContext:
    """Runtime context shared between stages."""

    transaction: FixTransaction
    constraints: GeometryConstraints
    merge_constraints: Optional[MergeConstraints] = None
    verbose: bool = False
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def geometry(self) -> BaseGeometry:
        return self.transaction.current_geometry

    @property
    def status(self):
        return self.transaction.current_status


@dataclass
class FixStage:
    """Single unit of work in the repair pipeline."""

    name: str
    predicate: Predicate
    runner: Runner

    def should_run(self, ctx: StageContext) -> bool:
        try:
            return self.predicate(ctx)
        except Exception:
            return False

    def execute(self, ctx: StageContext) -> StageResult:
        if not self.should_run(ctx):
            return StageResult(
                name=self.name,
                executed=False,
                committed=False,
                changed=False,
                message="predicate skipped",
            )

        try:
            return self.runner(ctx)
        except Exception as exc:  # pragma: no cover - defensive
            return StageResult(
                name=self.name,
                executed=True,
                committed=False,
                changed=False,
                error=str(exc),
            )


def validity_stage() -> FixStage:
    """Stage that repairs invalid geometries."""

    def predicate(ctx: StageContext) -> bool:
        return ctx.constraints.must_be_valid and not ctx.status.validity

    def runner(ctx: StageContext) -> StageResult:
        committed = ctx.transaction.try_fix(
            repair_geometry,
            fix_name="repair_validity",
            repair_strategy=RepairStrategy.AUTO,
        )
        message = "repaired invalid geometry" if committed else "repair rolled back"
        return StageResult(
            name="validity_repair",
            executed=True,
            committed=committed,
            changed=committed,
            message=message,
        )

    return FixStage("validity_repair", predicate, runner)


def clearance_stage() -> FixStage:
    """Stage that improves minimum clearance when requested."""

    def predicate(ctx: StageContext) -> bool:
        target = ctx.constraints.min_clearance
        if target is None:
            return False
        if not isinstance(ctx.geometry, (Polygon, MultiPolygon)):
            return False
        clearance = ctx.status.clearance
        if clearance is None:
            return False
        return clearance + 1e-9 < target

    def runner(ctx: StageContext) -> StageResult:
        def clearance_fix(geometry: BaseGeometry) -> BaseGeometry:
            return _apply_clearance_fix(geometry, ctx.constraints.min_clearance)

        committed = ctx.transaction.try_fix(
            clearance_fix,
            fix_name="fix_clearance",
        )
        message = (
            "clearance improved"
            if committed
            else "clearance fix rolled back or no improvement"
        )
        return StageResult(
            name="clearance_fix",
            executed=True,
            committed=committed,
            changed=committed,
            message=message,
        )

    return FixStage("clearance_fix", predicate, runner)


def component_merge_stage(config: MergeConstraints) -> FixStage:
    """Stage that merges MultiPolygon components when configured."""

    def predicate(ctx: StageContext) -> bool:
        if not config.enabled:
            return False
        geom = ctx.geometry
        return isinstance(geom, MultiPolygon) and len(geom.geoms) > 1

    def runner(ctx: StageContext) -> StageResult:
        def merge_components(geometry: BaseGeometry) -> BaseGeometry:
            if isinstance(geometry, Polygon):
                polygons = [geometry]
            elif isinstance(geometry, MultiPolygon):
                polygons = list(geometry.geoms)
            else:
                return geometry

            merged = merge_close_polygons(
                polygons,
                margin=config.margin,
                merge_strategy=config.merge_strategy,
                preserve_holes=config.preserve_holes,
                insert_vertices=config.insert_vertices,
            )

            if not merged:
                return geometry

            if len(merged) == 1:
                return merged[0]

            return MultiPolygon(merged)

        committed = ctx.transaction.try_fix(
            merge_components,
            fix_name="merge_components",
        )
        message = (
            "components merged"
            if committed
            else "merge rolled back or not applicable"
        )
        return StageResult(
            name="component_merge",
            executed=True,
            committed=committed,
            changed=committed,
            message=message,
        )

    return FixStage("component_merge", predicate, runner)


def cleanup_stage() -> FixStage:
    """Stage that performs lightweight cleanup (holes, zero-width slivers)."""

    def predicate(ctx: StageContext) -> bool:
        return isinstance(ctx.geometry, (Polygon, MultiPolygon))

    def runner(ctx: StageContext) -> StageResult:
        geometry = ctx.geometry
        cleaned = _cleanup_geometry(geometry, ctx.constraints)
        if cleaned is None or cleaned.equals(geometry):
            return StageResult(
                name="cleanup",
                executed=True,
                committed=False,
                changed=False,
                message="cleanup not needed",
            )

        def cleanup_fix(_: BaseGeometry) -> BaseGeometry:
            return _cleanup_geometry(_, ctx.constraints)

        committed = ctx.transaction.try_fix(cleanup_fix, fix_name="cleanup")
        message = "cleanup applied" if committed else "cleanup rolled back"
        return StageResult(
            name="cleanup",
            executed=True,
            committed=committed,
            changed=committed,
            message=message,
        )

    return FixStage("cleanup", predicate, runner)


def build_default_stages(
    merge_constraints: Optional[MergeConstraints] = None,
) -> List[FixStage]:
    """Create the default ordered stage list."""
    stages = [validity_stage(), clearance_stage()]

    if merge_constraints and merge_constraints.enabled:
        stages.append(component_merge_stage(merge_constraints))

    stages.append(cleanup_stage())
    return stages


def execute_stage_pipeline(
    transaction: FixTransaction,
    constraints: GeometryConstraints,
    merge_constraints: Optional[MergeConstraints] = None,
    stages: Optional[List[FixStage]] = None,
    max_iterations: int = 20,
    verbose: bool = False,
) -> List[StageResult]:
    """Run stages until constraints satisfied, stagnation, or iteration limit."""
    if stages is None:
        stages = build_default_stages(merge_constraints)

    ctx = StageContext(
        transaction=transaction,
        constraints=constraints,
        merge_constraints=merge_constraints,
        verbose=verbose,
    )

    history: List[StageResult] = []
    iteration = 0

    while iteration < max_iterations and not ctx.status.all_satisfied():
        iteration += 1
        iteration_progress = False

        for stage in stages:
            result = stage.execute(ctx)
            history.append(result)

            if verbose and result.executed:
                print(f"[Stage {result.name}] {result.summary()}")

            if result.changed:
                iteration_progress = True

            if ctx.status.all_satisfied():
                break

        if not iteration_progress:
            break

    return history


def _cleanup_geometry(
    geometry: BaseGeometry,
    constraints: GeometryConstraints,
) -> BaseGeometry:
    """Apply shared cleanup rules to polygons."""
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry

    config = CleanupConfig(
        min_zero_area=1e-10,
        hole_area_threshold=constraints.min_hole_area if constraints.min_hole_area and constraints.min_hole_area > 0 else None,
        hole_aspect_ratio=constraints.max_hole_aspect_ratio,
        hole_min_width=constraints.min_hole_width,
        preserve_holes=True,
    )

    cleaned = cleanup_polygon(geometry, config)
    try:
        # Remove near-zero-width slivers using a tiny buffer pass
        if cleaned.is_valid and not cleaned.is_empty:
            try:
                clearance = cleaned.minimum_clearance
            except Exception:  # pragma: no cover - Shapely edge case
                clearance = None

            if clearance is not None and clearance < 0.01:
                eroded = cleaned.buffer(-0.01, join_style=2)
                if (
                    isinstance(eroded, (Polygon, MultiPolygon))
                    and eroded.is_valid
                    and not eroded.is_empty
                    and eroded.area > 0
                ):
                    dilated = eroded.buffer(0.01, join_style=2)
                    if (
                        isinstance(dilated, (Polygon, MultiPolygon))
                        and dilated.is_valid
                        and not dilated.is_empty
                    ):
                        area_loss = (
                            (cleaned.area - dilated.area) / cleaned.area
                            if cleaned.area > 0
                            else 0
                        )
                        if area_loss < 0.01:
                            cleaned = dilated

            buffered = cleaned.buffer(0)
            if buffered.is_valid and not buffered.is_empty:
                cleaned = buffered

    except Exception:
        return geometry

    return cleaned


def _apply_clearance_fix(geometry: BaseGeometry, min_clearance: Optional[float]) -> BaseGeometry:
    if min_clearance is None or min_clearance <= 0:
        return geometry

    if isinstance(geometry, Polygon):
        return fix_clearance(geometry, min_clearance)

    if isinstance(geometry, MultiPolygon):
        united = unary_union(geometry)
        if isinstance(united, Polygon):
            return fix_clearance(united, min_clearance)
        if isinstance(united, MultiPolygon):
            fixed_parts = [fix_clearance(poly, min_clearance) for poly in united.geoms]
            valid_parts = [poly for poly in fixed_parts if isinstance(poly, Polygon) and not poly.is_empty]
            if not valid_parts:
                return geometry
            return MultiPolygon(valid_parts)
        return geometry

    return geometry


__all__ = [
    "FixStage",
    "StageContext",
    "StageResult",
    "build_default_stages",
    "execute_stage_pipeline",
]
