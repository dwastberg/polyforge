"""Tests for geometry fixing functions."""

import pytest
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.validation import explain_validity
from polyforge.repair import (
    repair_geometry,
    analyze_geometry,
    batch_repair_geometries,
)
from polyforge.core.errors import RepairError
from polyforge.core.types import RepairStrategy


class TestDiagnoseGeometry:
    """Test suite for diagnose_geometry function."""

    def test_diagnose_valid_geometry(self):
        """Test diagnosing valid geometry."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        diagnosis = analyze_geometry(poly)

        assert diagnosis["is_valid"] is True
        assert diagnosis["geometry_type"] == "Polygon"
        assert not diagnosis["is_empty"]
        assert diagnosis["area"] == 1.0

    def test_diagnose_self_intersection(self):
        """Test diagnosing self-intersecting polygon."""
        poly = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])

        diagnosis = analyze_geometry(poly)

        assert diagnosis["is_valid"] is False
        assert "Self-intersection" in diagnosis["issues"]
        assert any("buffer" in s.lower() for s in diagnosis["suggestions"])

    def test_diagnose_duplicate_vertices(self):
        """Test diagnosing duplicate vertices."""
        # Create polygon with duplicates
        poly = Polygon([(0, 0), (1, 0), (1, 0), (1, 1), (0, 1)])

        diagnosis = analyze_geometry(poly)

        # Check for duplicate-related issues
        assert "Consecutive duplicate vertices" in diagnosis["issues"]

    def test_diagnose_includes_all_fields(self):
        """Test that diagnosis includes all expected fields."""
        poly = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])

        diagnosis = analyze_geometry(poly)

        assert "is_valid" in diagnosis
        assert "validity_message" in diagnosis
        assert "issues" in diagnosis
        assert "suggestions" in diagnosis
        assert "geometry_type" in diagnosis
        assert "is_empty" in diagnosis
        assert "area" in diagnosis

    def test_diagnose_empty_geometry(self):
        """Test diagnosing empty geometry."""
        poly = Polygon()

        diagnosis = analyze_geometry(poly)

        assert diagnosis["is_empty"] is True


class TestBatchFixGeometries:
    """Test suite for batch_fix_geometries function."""

    def test_batch_fix_all_valid(self):
        """Test batch fixing when all geometries are valid."""
        polys = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            Polygon([(0, 2), (1, 2), (1, 3), (0, 3)]),
        ]

        fixed, failed = batch_repair_geometries(polys)

        assert len(fixed) == 3
        assert len(failed) == 0
        assert all(p.is_valid for p in fixed)

    def test_batch_fix_some_invalid(self):
        """Test batch fixing with some invalid geometries."""
        polys = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # Valid
            Polygon([(0, 0), (2, 2), (2, 0), (0, 2)]),  # Invalid (bow-tie)
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),  # Valid
        ]

        fixed, failed = batch_repair_geometries(polys)

        assert len(fixed) == 3
        assert len(failed) == 0
        assert all(p.is_valid for p in fixed)

    def test_batch_fix_skip_unfixable(self):
        """Test batch fixing with skip on error."""
        polys = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(0, 0), (2, 2), (2, 0), (0, 2)]),
        ]

        # Use strict strategy which might fail
        fixed, failed = batch_repair_geometries(
            polys, repair_strategy=RepairStrategy.STRICT, on_error="skip"
        )

        # Should skip unfixable ones
        assert len(fixed) <= len(polys)

    def test_batch_fix_keep_on_error(self):
        """Test batch fixing with keep on error."""
        polys = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(0, 0), (2, 2), (2, 0), (0, 2)]),
        ]

        fixed, failed = batch_repair_geometries(
            polys, repair_strategy=RepairStrategy.STRICT, on_error="keep"
        )

        # Should keep all, even if invalid
        assert len(fixed) == len(polys)

    def test_batch_fix_empty_list(self):
        """Test batch fixing with empty list."""
        fixed, failed = batch_repair_geometries([])

        assert len(fixed) == 0
        assert len(failed) == 0

    def test_batch_fix_with_verbose(self):
        """Test batch fixing with verbose mode."""
        polys = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(0, 0), (2, 2), (2, 0), (0, 2)]),
        ]

        fixed, failed = batch_repair_geometries(polys, verbose=True)

        assert len(fixed) == 2

    def test_batch_fix_preserves_order(self):
        """Test that batch fixing preserves order."""
        polys = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            Polygon([(0, 2), (1, 2), (1, 3), (0, 3)]),
        ]

        fixed, failed = batch_repair_geometries(polys)

        assert len(fixed) == 3
        # Check that they're in the same order
        for i, poly in enumerate(fixed):
            # Original and fixed should have same centroid (roughly)
            assert abs(poly.centroid.x - polys[i].centroid.x) < 1.0


if __name__ == "__main__":
    pytest.main()
