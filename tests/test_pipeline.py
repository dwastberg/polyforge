"""Unit tests for the lightweight pipeline helpers."""

import pytest
from shapely.affinity import scale
from shapely.geometry import Polygon

from polyforge.core import GeometryConstraints
from polyforge.pipeline import (
    FixConfig,
    PipelineContext,
    StepResult,
    config_from_constraints,
    run_steps,
)


def _evaluate_candidate(
    name: str,
    current: Polygon,
    candidate: Polygon,
    ctx: PipelineContext,
    message: str,
) -> StepResult:
    current_status = ctx.constraints.check(current, ctx.original)
    candidate_status = ctx.constraints.check(candidate, ctx.original)
    if candidate_status.is_better_or_equal(current_status):
        changed = not candidate.equals(current)
        return StepResult(name, candidate, changed, message)
    return StepResult(name, current, False, "rejected")


def test_pipeline_rejects_area_regressions():
    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    constraints = GeometryConstraints(min_area_ratio=0.95)
    config = config_from_constraints(constraints)
    ctx = PipelineContext(
        original=square,
        constraints=constraints,
        config=config,
    )

    def shrink_step(geometry, context):
        candidate = scale(geometry, xfact=0.8, yfact=0.8)
        return _evaluate_candidate("shrink", geometry, candidate, context, "shrink attempt")

    result, status, history = run_steps(square, [shrink_step], ctx, max_passes=1)
    assert result.equals(square)
    assert not history[-1].changed
    assert status.area_ratio == 1.0


def test_pipeline_accepts_clearance_improvement():
    thin = Polygon([(0, 0), (5, 0), (5, 0.1), (0, 0.1)])
    constraints = GeometryConstraints(min_clearance=1.0)
    config = config_from_constraints(constraints)
    ctx = PipelineContext(
        original=thin,
        constraints=constraints,
        config=config,
    )

    buffer_amount = 1.0

    def widen_step(geometry, context):
        candidate = geometry.buffer(buffer_amount, join_style=2)
        return _evaluate_candidate("widen", geometry, candidate, context, "buffered")

    result, status, history = run_steps(thin, [widen_step], ctx, max_passes=1)
    assert not result.equals(thin)
    assert history[-1].changed
    assert status.clearance is not None
    assert status.clearance >= 1.0


def test_pipeline_context_metric_cache(monkeypatch):
    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    constraints = GeometryConstraints()
    ctx = PipelineContext(
        original=square,
        constraints=constraints,
        config=config_from_constraints(constraints),
    )

    call_count = {"count": 0}

    def fake_measure(geometry, original, skip_clearance):
        call_count["count"] += 1
        return {
            "is_valid": True,
            "is_empty": False,
            "clearance": None,
            "area": geometry.area,
            "area_ratio": 1.0,
        }

    monkeypatch.setattr("polyforge.metrics.measure_geometry", fake_measure)

    ctx.get_metrics(square)
    ctx.get_metrics(square)

    assert call_count["count"] == 1


def test_run_steps_uses_metric_cache(monkeypatch):
    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    constraints = GeometryConstraints()
    config = config_from_constraints(constraints)
    ctx = PipelineContext(
        original=square,
        constraints=constraints,
        config=config,
    )

    call_count = {"count": 0}

    def fake_measure(geometry, original, skip_clearance):
        call_count["count"] += 1
        return {
            "is_valid": True,
            "is_empty": False,
            "clearance": None,
            "area": geometry.area,
            "area_ratio": 1.0,
        }

    monkeypatch.setattr("polyforge.metrics.measure_geometry", fake_measure)

    def noop_step(geometry, context):
        return StepResult("noop", geometry, False, "no change")

    run_steps(square, [noop_step], ctx, max_passes=1)

    # Only the first metrics computation should invoke measure_geometry
    assert call_count["count"] == 1
