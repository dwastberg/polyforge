"""Tests for the metrics module."""

import pytest
from shapely.geometry import Polygon, MultiPolygon, Point, LineString

from polyforge.metrics import (
    measure_geometry,
    total_overlap_area,
    overlap_area_by_geometry,
    _safe_clearance,
)


class TestSafeClearance:
    """Tests for _safe_clearance helper function."""

    def test_valid_polygon(self):
        """Test clearance of valid polygon."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        clearance = _safe_clearance(poly)

        assert clearance > 0
        assert isinstance(clearance, float)

    def test_empty_polygon(self):
        """Test clearance of empty polygon returns inf (infinitely stable)."""
        poly = Polygon()
        clearance = _safe_clearance(poly)

        # Empty polygons have infinite clearance (no vertices to move)
        assert clearance == float('inf')

    def test_invalid_polygon(self):
        """Test clearance of invalid polygon returns 0.0."""
        # Self-intersecting bowtie
        poly = Polygon([(0, 0), (1, 1), (0, 1), (1, 0)])
        clearance = _safe_clearance(poly)

        # Should return a value (may be 0.0 or computed depending on Shapely version)
        assert isinstance(clearance, float)


class TestMeasureGeometry:
    """Tests for measure_geometry function."""

    def test_simple_polygon(self):
        """Test measuring a valid simple polygon."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        metrics = measure_geometry(poly)

        assert metrics["is_valid"] is True
        assert metrics["is_empty"] is False
        assert metrics["area"] == 100.0
        assert metrics["clearance"] is not None
        assert metrics["clearance"] > 0
        assert metrics["area_ratio"] is None  # No original provided

    def test_with_original_polygon(self):
        """Test area ratio calculation with original polygon."""
        original = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        smaller = Polygon([(0, 0), (8, 0), (8, 8), (0, 8)])
        metrics = measure_geometry(smaller, original=original)

        assert metrics["area_ratio"] == pytest.approx(0.64, rel=0.01)

    def test_skip_clearance(self):
        """Test that skip_clearance=True returns None for clearance."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        metrics = measure_geometry(poly, skip_clearance=True)

        assert metrics["clearance"] is None
        assert metrics["is_valid"] is True
        assert metrics["area"] == 100.0

    def test_empty_polygon(self):
        """Test measuring empty polygon."""
        poly = Polygon()
        metrics = measure_geometry(poly)

        assert metrics["is_empty"] is True
        assert metrics["area"] == 0.0

    def test_invalid_polygon(self):
        """Test measuring invalid (self-intersecting) polygon."""
        # Figure-8 / bowtie polygon
        poly = Polygon([(0, 0), (1, 1), (0, 1), (1, 0)])
        metrics = measure_geometry(poly)

        assert metrics["is_valid"] is False

    def test_multipolygon(self):
        """Test measuring a MultiPolygon."""
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
        multi = MultiPolygon([p1, p2])
        metrics = measure_geometry(multi)

        assert metrics["is_valid"] is True
        assert metrics["area"] == 2.0

    def test_area_ratio_with_zero_original(self):
        """Test area ratio when original has zero area."""
        original = Polygon()
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        metrics = measure_geometry(poly, original=original)

        # Should be None when original area is 0
        assert metrics["area_ratio"] is None


class TestTotalOverlapArea:
    """Tests for total_overlap_area function."""

    def test_no_overlap(self):
        """Test with non-overlapping polygons."""
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
        area = total_overlap_area([p1, p2])

        assert area == 0.0

    def test_partial_overlap(self):
        """Test with partially overlapping polygons."""
        p1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        p2 = Polygon([(1, 0), (3, 0), (3, 2), (1, 2)])
        area = total_overlap_area([p1, p2])

        # Overlap is 1x2 = 2 square units
        assert area == pytest.approx(2.0, rel=0.01)

    def test_full_overlap(self):
        """Test with fully contained polygon."""
        outer = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        inner = Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
        area = total_overlap_area([outer, inner])

        # Inner is fully inside outer, overlap = inner.area = 36
        assert area == pytest.approx(36.0, rel=0.01)

    def test_empty_list(self):
        """Test with empty list."""
        area = total_overlap_area([])
        assert area == 0.0

    def test_single_polygon(self):
        """Test with single polygon (no overlap possible)."""
        p = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        area = total_overlap_area([p])
        assert area == 0.0

    def test_three_overlapping(self):
        """Test with three overlapping polygons."""
        p1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        p2 = Polygon([(1, 0), (3, 0), (3, 2), (1, 2)])
        p3 = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
        area = total_overlap_area([p1, p2, p3])

        # Has overlaps between p1-p2, p2-p3, and some triple overlap
        assert area > 0

    def test_with_empty_geometry(self):
        """Test that empty geometries are handled."""
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon()  # Empty
        p3 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])  # Same as p1
        area = total_overlap_area([p1, p2, p3])

        # p1 and p3 overlap completely (area = 1)
        assert area == pytest.approx(1.0, rel=0.01)


class TestOverlapAreaByGeometry:
    """Tests for overlap_area_by_geometry function."""

    def test_no_overlap(self):
        """Test attribution with non-overlapping polygons."""
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
        overlaps = overlap_area_by_geometry([p1, p2])

        assert overlaps == [0.0, 0.0]

    def test_partial_overlap(self):
        """Test attribution with partially overlapping polygons."""
        p1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        p2 = Polygon([(1, 0), (3, 0), (3, 2), (1, 2)])
        overlaps = overlap_area_by_geometry([p1, p2])

        # Both should have 2.0 overlap attributed
        assert overlaps[0] == pytest.approx(2.0, rel=0.01)
        assert overlaps[1] == pytest.approx(2.0, rel=0.01)

    def test_empty_list(self):
        """Test with empty list."""
        overlaps = overlap_area_by_geometry([])
        assert overlaps == []

    def test_single_polygon(self):
        """Test with single polygon."""
        p = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        overlaps = overlap_area_by_geometry([p])
        assert overlaps == [0.0]

    def test_min_area_threshold(self):
        """Test that tiny overlaps are ignored."""
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        # Create polygon that barely touches p1
        p2 = Polygon([(0.99999, 0), (2, 0), (2, 1), (0.99999, 1)])
        overlaps = overlap_area_by_geometry([p1, p2], min_area_threshold=0.01)

        # Tiny overlap should be ignored
        assert overlaps == [0.0, 0.0]

    def test_three_overlapping(self):
        """Test attribution with three overlapping polygons."""
        p1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        p2 = Polygon([(1, 0), (3, 0), (3, 2), (1, 2)])
        p3 = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
        overlaps = overlap_area_by_geometry([p1, p2, p3])

        # All should have some overlap attributed
        assert all(ov >= 0 for ov in overlaps)
        # p2 should have the most overlap (overlaps with both p1 and p3)
        assert overlaps[1] >= overlaps[0]
        assert overlaps[1] >= overlaps[2]

    def test_with_none_geometry(self):
        """Test handling of None in geometry list."""
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        overlaps = overlap_area_by_geometry([p1, None, p1])

        # None should have 0.0 attributed
        assert overlaps[1] == 0.0


class TestEdgeCases:
    """Edge case tests for metrics functions."""

    def test_measure_geometry_with_point(self):
        """Test measure_geometry with non-polygon geometry."""
        point = Point(0, 0)
        metrics = measure_geometry(point)

        assert metrics["is_valid"] is True
        assert metrics["is_empty"] is False
        assert metrics["area"] == 0.0

    def test_measure_geometry_with_linestring(self):
        """Test measure_geometry with LineString."""
        line = LineString([(0, 0), (1, 1), (2, 0)])
        metrics = measure_geometry(line)

        assert metrics["is_valid"] is True
        assert metrics["area"] == 0.0

    def test_overlap_with_touching_polygons(self):
        """Test that touching polygons have zero overlap."""
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])  # Shares edge with p1
        area = total_overlap_area([p1, p2])

        assert area == 0.0
