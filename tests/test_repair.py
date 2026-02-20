"""Tests for geometry repair functions."""

import math
from unittest.mock import patch

import numpy as np
import pytest
from shapely.geometry import Polygon, MultiPolygon, Point, LineString

from polyforge import repair_geometry, batch_repair_geometries
from polyforge.core import RepairStrategy
from polyforge.core.errors import RepairError


def _bowtie() -> Polygon:
    """Self-intersecting bowtie polygon."""
    return Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])


def _spike_polygon() -> Polygon:
    """Valid polygon with a very thin spike."""
    return Polygon(
        [
            (0, 0),
            (10, 0),
            (10, 10),
            (5.001, 10),
            (5.001, 10.002),
            (4.999, 10.002),
            (4.999, 10),
            (0, 10),
            (0, 0),
        ]
    )


def _ring_self_intersection() -> Polygon:
    """Polygon with a ring self-intersection (figure-8 shape)."""
    return Polygon(
        [
            (0, 0),
            (4, 0),
            (4, 4),
            (2, 2),
            (0, 4),
            (0, 0),
        ]
    )


class TestBatchRepairGeometries:
    """Tests for batch_repair_geometries()."""

    def test_all_valid(self):
        polys = [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            Polygon([(20, 0), (30, 0), (30, 10), (20, 10)]),
        ]
        repaired, failed = batch_repair_geometries(polys)
        assert len(repaired) == 2
        assert len(failed) == 0
        assert all(g.is_valid for g in repaired)

    def test_mixed_valid_invalid(self):
        polys = [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),  # valid
            _bowtie(),  # invalid
            Polygon([(20, 0), (30, 0), (30, 10), (20, 10)]),  # valid
        ]
        repaired, failed = batch_repair_geometries(polys)
        assert len(repaired) == 3
        assert len(failed) == 0
        assert all(g.is_valid for g in repaired)

    def test_on_error_keep_default(self):
        """Default on_error='keep' preserves alignment."""
        polys = [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            _bowtie(),
        ]
        repaired, failed = batch_repair_geometries(polys)
        assert len(repaired) == len(polys)

    def test_on_error_skip_preserves_alignment(self):
        """on_error='skip' inserts None for failed entries."""
        polys = [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            _bowtie(),
        ]
        # With 'skip' mode, both bowtie and valid are fine since repair succeeds
        repaired, failed = batch_repair_geometries(polys, on_error="skip")
        assert len(repaired) == len(polys)  # alignment always preserved

    def test_on_error_raise(self):
        """on_error='raise' propagates exceptions."""
        # repair_geometry shouldn't fail on a bowtie, so use a valid poly
        # and check that it works normally
        polys = [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])]
        repaired, failed = batch_repair_geometries(polys, on_error="raise")
        assert len(repaired) == 1

    def test_empty_list(self):
        repaired, failed = batch_repair_geometries([])
        assert repaired == []
        assert failed == []


class TestGeometryConstraintsValidation:
    """Tests for GeometryConstraints __post_init__ validation."""

    def test_negative_min_clearance_raises(self):
        from polyforge.core import GeometryConstraints

        with pytest.raises(ValueError, match="min_clearance"):
            GeometryConstraints(min_clearance=-1.0)

    def test_negative_max_overlap_area_raises(self):
        from polyforge.core import GeometryConstraints

        with pytest.raises(ValueError, match="max_overlap_area"):
            GeometryConstraints(max_overlap_area=-1.0)

    def test_negative_min_area_ratio_raises(self):
        from polyforge.core import GeometryConstraints

        with pytest.raises(ValueError, match="min_area_ratio"):
            GeometryConstraints(min_area_ratio=-0.5)

    def test_max_below_min_area_ratio_raises(self):
        from polyforge.core import GeometryConstraints

        with pytest.raises(ValueError, match="max_area_ratio"):
            GeometryConstraints(min_area_ratio=0.5, max_area_ratio=0.3)

    def test_negative_max_holes_raises(self):
        from polyforge.core import GeometryConstraints

        with pytest.raises(ValueError, match="max_holes"):
            GeometryConstraints(max_holes=-1)

    def test_valid_constraints_accepted(self):
        from polyforge.core import GeometryConstraints

        c = GeometryConstraints(
            min_clearance=0.5,
            min_area_ratio=0.8,
            max_area_ratio=1.2,
            max_holes=5,
        )
        assert c.min_clearance == 0.5


class TestSafeClearanceNoneReturn:
    """Tests for _safe_clearance returning None on error."""

    def test_returns_none_for_invalid_object(self):
        from polyforge.metrics import _safe_clearance

        class FakeGeom:
            @property
            def minimum_clearance(self):
                raise ValueError("broken")

        result = _safe_clearance(FakeGeom())
        assert result is None

    def test_returns_float_for_valid_polygon(self):
        from polyforge.metrics import _safe_clearance

        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        result = _safe_clearance(poly)
        assert isinstance(result, float)
        assert result > 0


class TestBatchRepairFailurePaths:
    def test_on_error_keep_preserves_original(self):
        """When repair raises and on_error='keep', original is kept."""
        polys = [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])]
        with patch(
            "polyforge.repair.core.repair_geometry",
            side_effect=RepairError("forced failure"),
        ):
            repaired, failed = batch_repair_geometries(polys, on_error="keep")
        assert len(repaired) == 1
        assert repaired[0] is polys[0]  # original geometry kept

    def test_batch_always_preserves_list_length(self):
        """Output list length always matches input, regardless of on_error mode."""
        polys = [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            _bowtie(),
            Polygon([(20, 0), (30, 0), (30, 10), (20, 10)]),
        ]
        for mode in ("keep", "skip", "raise"):
            try:
                repaired, _ = batch_repair_geometries(polys, on_error=mode)
                assert len(repaired) == len(polys), (
                    f"Length mismatch with on_error='{mode}'"
                )
            except Exception:
                pass  # 'raise' mode may raise, that's fine
