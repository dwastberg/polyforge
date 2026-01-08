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


class TestMultiStepPipeline:
    """Tests for realistic multi-step pipeline scenarios."""

    def test_multiple_steps_in_sequence(self):
        """Test pipeline with multiple steps."""
        square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        constraints = GeometryConstraints()
        config = config_from_constraints(constraints)
        ctx = PipelineContext(original=square, constraints=constraints, config=config)

        step_order = []

        def step_a(geometry, context):
            step_order.append("a")
            return StepResult("step_a", geometry, False, "no change")

        def step_b(geometry, context):
            step_order.append("b")
            return StepResult("step_b", geometry, False, "no change")

        result, status, history = run_steps(square, [step_a, step_b], ctx, max_passes=1)

        assert step_order == ["a", "b"]
        assert result.equals(square)

    def test_pipeline_stops_when_satisfied(self):
        """Test that pipeline stops early when all constraints are satisfied."""
        square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        # Constraints already satisfied by the square
        constraints = GeometryConstraints(min_clearance=0.1)
        config = config_from_constraints(constraints)
        ctx = PipelineContext(original=square, constraints=constraints, config=config)

        step_calls = {"count": 0}

        def counting_step(geometry, context):
            step_calls["count"] += 1
            return StepResult("count", geometry, False, "no change")

        result, status, history = run_steps(square, [counting_step], ctx, max_passes=10)

        # Should stop early because constraints already satisfied
        # At most 1 pass (check, find satisfied, stop)
        assert step_calls["count"] <= 2

    def test_pipeline_iterates_when_making_progress(self):
        """Test that pipeline continues when making progress."""
        thin = Polygon([(0, 0), (10, 0), (10, 0.5), (0, 0.5)])
        constraints = GeometryConstraints(min_clearance=2.0)
        config = config_from_constraints(constraints)
        ctx = PipelineContext(original=thin, constraints=constraints, config=config)

        step_calls = {"count": 0}

        def improving_step(geometry, context):
            step_calls["count"] += 1
            if step_calls["count"] <= 3:
                # Keep improving until we hit target
                candidate = geometry.buffer(0.5, join_style=2)
                return _evaluate_candidate("improve", geometry, candidate, context, "improved")
            return StepResult("improve", geometry, False, "no more improvement")

        result, status, history = run_steps(thin, [improving_step], ctx, max_passes=10)

        # Should have called the step multiple times
        assert step_calls["count"] >= 2

    def test_pipeline_handles_invalid_step_result(self):
        """Test pipeline handles steps returning invalid geometry gracefully."""
        square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        constraints = GeometryConstraints()
        config = config_from_constraints(constraints)
        ctx = PipelineContext(original=square, constraints=constraints, config=config)

        def breaking_step(geometry, context):
            # Return empty polygon (invalid result)
            return StepResult("break", Polygon(), True, "broken")

        def recovery_step(geometry, context):
            return StepResult("recover", geometry, False, "no change")

        # Pipeline should handle this gracefully
        result, status, history = run_steps(
            square, [breaking_step, recovery_step], ctx, max_passes=1
        )

        # Result should still be valid
        assert result.is_valid


class TestFixConfigFromConstraints:
    """Tests for config_from_constraints helper."""

    def test_extracts_clearance(self):
        """Test that min_clearance is extracted."""
        constraints = GeometryConstraints(min_clearance=2.5)
        config = config_from_constraints(constraints)
        assert config.min_clearance == 2.5

    def test_extracts_area_ratio(self):
        """Test that min_area_ratio is extracted."""
        constraints = GeometryConstraints(min_area_ratio=0.9)
        config = config_from_constraints(constraints)
        assert config.min_area_ratio == 0.9

    def test_default_values(self):
        """Test default values from empty constraints."""
        constraints = GeometryConstraints()
        config = config_from_constraints(constraints)
        # min_clearance defaults to None in FixConfig
        assert config.min_clearance is None
        assert config.min_area_ratio == 0.0
        assert config.must_be_valid is True
        assert config.allow_multipolygon is True

    def test_must_be_valid_extracted(self):
        """Test that must_be_valid is extracted."""
        constraints = GeometryConstraints(must_be_valid=False)
        config = config_from_constraints(constraints)
        assert config.must_be_valid is False


class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_step_result_creation(self):
        """Test creating a StepResult."""
        result = StepResult("test_step", Polygon(), True, "test message")

        assert result.name == "test_step"
        assert result.changed is True
        assert result.message == "test message"

    def test_step_result_with_polygon(self):
        """Test StepResult with actual geometry."""
        square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        result = StepResult("process", square, False, "unchanged")

        assert result.geometry.equals(square)
        assert result.changed is False


class TestPipelineContext:
    """Tests for PipelineContext."""

    def test_context_creation(self):
        """Test creating a PipelineContext."""
        square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        constraints = GeometryConstraints(min_clearance=1.0)
        config = config_from_constraints(constraints)

        ctx = PipelineContext(original=square, constraints=constraints, config=config)

        assert ctx.original.equals(square)
        assert ctx.constraints.min_clearance == 1.0
        assert ctx.config.min_clearance == 1.0

    def test_context_stores_merge_constraints(self):
        """Test that merge_constraints can be stored."""
        from polyforge.core import MergeConstraints

        square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        constraints = GeometryConstraints()
        config = config_from_constraints(constraints)
        merge = MergeConstraints(enabled=True, margin=2.0)

        ctx = PipelineContext(
            original=square,
            constraints=constraints,
            config=config,
            merge_constraints=merge,
        )

        assert ctx.merge_constraints.enabled is True
        assert ctx.merge_constraints.margin == 2.0
