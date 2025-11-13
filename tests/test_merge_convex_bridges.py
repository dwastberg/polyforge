"""Tests for the convex_bridges merge strategy.

Tests specifically for the bug fix that prevented diagonal bridges
from being created when merging separated rectangular polygons.
"""

import pytest
from shapely.geometry import Polygon
from polyforge import merge_close_polygons
from polyforge.core.types import MergeStrategy
from polyforge.ops.merge_convex_bridges import _bridge_for_polygon_pair


class TestConvexBridgesFix:
    """Test the convex_bridges strategy bug fix."""

    def test_bridge_helper_returns_polygon(self):
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = Polygon([(3, 0), (5, 0), (5, 2), (3, 2)])
        bridge = _bridge_for_polygon_pair(poly1, poly2, margin=2.0)
        assert bridge is not None
        assert bridge.is_valid
        assert bridge.area > 0

    def test_separated_rectangles_no_diagonal_cuts(self):
        """Test that separated rectangles merge without diagonal cuts.

        This is a regression test for the bug where convex_bridges would
        create diagonal cuts across the entire gap between polygons.
        """
        # Two separated rectangles
        poly1 = Polygon([(0, 5), (9, 5), (9, 15), (0, 15)])
        poly2 = Polygon([(11, 0), (21, 0), (21, 10), (11, 10)])

        result = merge_close_polygons([poly1, poly2], margin=5.0, merge_strategy=MergeStrategy.CONVEX_BRIDGES)

        assert len(result) == 1
        merged = result[0]

        # Should be valid
        assert merged.is_valid

        # Area should be reasonable (original areas + small bridge)
        # Not the huge diagonal-cut area from the bug
        original_area = poly1.area + poly2.area  # 100 + 100 = 200
        assert merged.area < original_area + 50  # Bridge should be small
        assert merged.area > original_area  # But should connect them

        # Should not have too many vertices (the buggy version created complex geometry)
        assert len(merged.exterior.coords) < 20

    def test_irregular_polygons_smooth_bridge(self):
        """Test that irregular polygons get smooth convex bridges."""
        poly1 = Polygon([(0, 0), (5, 0), (5, 5), (3, 7), (0, 5)])
        poly2 = Polygon([(7, 2), (12, 2), (12, 8), (7, 8)])

        result = merge_close_polygons([poly1, poly2], margin=3.0, merge_strategy=MergeStrategy.CONVEX_BRIDGES)

        assert len(result) == 1
        merged = result[0]
        assert merged.is_valid

        # Check that bridge was created
        original_area = poly1.area + poly2.area
        assert merged.area > original_area

    def test_three_polygons_in_row(self):
        """Test merging three polygons in a row."""
        poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        poly2 = Polygon([(7, 0), (12, 0), (12, 5), (7, 5)])
        poly3 = Polygon([(14, 0), (19, 0), (19, 5), (14, 5)])

        result = merge_close_polygons([poly1, poly2, poly3], margin=3.0, merge_strategy=MergeStrategy.CONVEX_BRIDGES)

        assert len(result) == 1
        merged = result[0]
        assert merged.is_valid

        # All three should be connected
        original_area = poly1.area + poly2.area + poly3.area  # 75
        assert merged.area > original_area  # Bridges add area
        assert merged.area < original_area + 20  # But not too much

    def test_overlapping_polygons(self):
        """Test that overlapping polygons are handled correctly."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])

        result = merge_close_polygons([poly1, poly2], margin=0.0, merge_strategy=MergeStrategy.CONVEX_BRIDGES)

        assert len(result) == 1
        merged = result[0]
        assert merged.is_valid

        # Area should be union of the two
        expected_area = poly1.area + poly2.area - poly1.intersection(poly2).area
        assert abs(merged.area - expected_area) < 0.01

    def test_touching_polygons(self):
        """Test polygons that touch at a point."""
        poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        poly2 = Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])

        result = merge_close_polygons([poly1, poly2], margin=0.0, merge_strategy=MergeStrategy.CONVEX_BRIDGES)

        assert len(result) == 1
        merged = result[0]
        assert merged.is_valid

        # Should merge into one polygon
        expected_area = poly1.area + poly2.area
        assert abs(merged.area - expected_area) < 0.01

    def test_distant_polygons_not_merged(self):
        """Test that polygons beyond margin are not merged."""
        poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        poly2 = Polygon([(20, 0), (25, 0), (25, 5), (20, 5)])

        result = merge_close_polygons([poly1, poly2], margin=5.0, merge_strategy=MergeStrategy.CONVEX_BRIDGES)

        # Should remain as two separate polygons
        assert len(result) == 2

    def test_single_polygon_unchanged(self):
        """Test that a single polygon is returned unchanged."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        result = merge_close_polygons([poly], margin=5.0, merge_strategy=MergeStrategy.CONVEX_BRIDGES)

        assert len(result) == 1
        assert result[0].equals(poly)

    def test_empty_list(self):
        """Test that empty list returns empty list."""
        result = merge_close_polygons([], margin=5.0, merge_strategy=MergeStrategy.CONVEX_BRIDGES)
        assert result == []


class TestConvexBridgesWithMapping:
    """Test convex_bridges with return_mapping=True."""

    def test_mapping_two_merged(self):
        """Test that mapping correctly tracks merged polygons."""
        poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        poly2 = Polygon([(7, 0), (12, 0), (12, 5), (7, 5)])

        result, mapping = merge_close_polygons(
            [poly1, poly2], margin=3.0, merge_strategy=MergeStrategy.CONVEX_BRIDGES, return_mapping=True
        )

        assert len(result) == 1
        assert len(mapping) == 1
        # Both original polygons should be in the mapping
        assert set(mapping[0]) == {0, 1}

    def test_mapping_with_isolated(self):
        """Test mapping when some polygons are isolated."""
        poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        poly2 = Polygon([(7, 0), (12, 0), (12, 5), (7, 5)])
        poly3 = Polygon([(100, 0), (105, 0), (105, 5), (100, 5)])  # Far away

        result, mapping = merge_close_polygons(
            [poly1, poly2, poly3], margin=3.0, merge_strategy=MergeStrategy.CONVEX_BRIDGES, return_mapping=True
        )

        assert len(result) == 2  # One merged, one isolated
        assert len(mapping) == 2

        # Find which result is the merged one
        merged_mapping = [m for m in mapping if len(m) > 1][0]
        isolated_mapping = [m for m in mapping if len(m) == 1][0]

        assert set(merged_mapping) == {0, 1}
        assert isolated_mapping == [2]
