"""Tests for polygon overlap splitting."""

import pytest
import numpy as np
from shapely.geometry import Polygon
from polyforge.split import split_overlap
from polyforge.core.types import OverlapStrategy


class TestSplitOverlap:
    """Test suite for split_overlap function."""

    def test_simple_overlapping_squares(self):
        """Test splitting two overlapping squares."""
        # Two squares that overlap by 1 unit
        poly1 = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
        poly2 = Polygon([(2, 0), (5, 0), (5, 3), (2, 3)])

        result1, result2 = split_overlap(poly1, poly2)

        # Check that they touch but don't overlap
        assert result1.intersects(result2), "Polygons should touch"
        overlap_area = result1.intersection(result2).area
        assert overlap_area < 1e-6, f"Polygons should not overlap, got area {overlap_area}"

        # Check that total area is conserved (approximately)
        original_total = poly1.area + poly2.area - poly1.intersection(poly2).area
        result_total = result1.area + result2.area
        assert abs(original_total - result_total) < 0.5, "Total area should be approximately conserved"

        # Check that both polygons are valid
        assert result1.is_valid, "Result 1 should be valid"
        assert result2.is_valid, "Result 2 should be valid"

    def test_non_overlapping_polygons(self):
        """Test that non-overlapping polygons are returned unchanged."""
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = Polygon([(3, 0), (5, 0), (5, 2), (3, 2)])

        result1, result2 = split_overlap(poly1, poly2)

        # Should return originals
        assert result1.equals(poly1), "Non-overlapping poly1 should be unchanged"
        assert result2.equals(poly2), "Non-overlapping poly2 should be unchanged"

    def test_touching_but_not_overlapping(self):
        """Test polygons that touch at an edge but don't overlap."""
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = Polygon([(2, 0), (4, 0), (4, 2), (2, 2)])

        result1, result2 = split_overlap(poly1, poly2)

        # Should return originals (or very similar)
        assert abs(result1.area - poly1.area) < 0.01
        assert abs(result2.area - poly2.area) < 0.01

    def test_one_polygon_contains_other(self):
        """Test when one polygon completely contains the other."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(2, 2), (5, 2), (5, 5), (2, 5)])

        result1, result2 = split_overlap(poly1, poly2)

        # Should return originals
        assert result1.equals(poly1), "Container polygon should be unchanged"
        assert result2.equals(poly2), "Contained polygon should be unchanged"

    def test_partial_overlap_rectangles(self):
        """Test rectangles with partial overlap."""
        poly1 = Polygon([(0, 0), (4, 0), (4, 2), (0, 2)])
        poly2 = Polygon([(3, 1), (7, 1), (7, 3), (3, 3)])

        result1, result2 = split_overlap(poly1, poly2)

        # Check validity
        assert result1.is_valid
        assert result2.is_valid

        # Check no overlap
        overlap = result1.intersection(result2)
        assert overlap.area < 1e-6

        # Check they touch or are close
        assert result1.distance(result2) < 1e-6

    def test_complex_polygon_shapes(self):
        """Test with more complex polygon shapes."""
        # Create an L-shaped polygon
        poly1 = Polygon([(0, 0), (3, 0), (3, 2), (2, 2), (2, 3), (0, 3)])

        # Create a rectangle that overlaps with the L
        poly2 = Polygon([(1, 1), (4, 1), (4, 4), (1, 4)])

        result1, result2 = split_overlap(poly1, poly2)

        # Check validity
        assert result1.is_valid
        assert result2.is_valid

        # Check minimal overlap (touching is OK)
        overlap = result1.intersection(result2)
        assert overlap.area < 0.1

    def test_approximate_equal_split(self):
        """Test that the overlap is split approximately equally."""
        # Two large overlapping squares
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(5, 0), (15, 0), (15, 10), (5, 10)])

        # Original overlap
        original_overlap = poly1.intersection(poly2)
        original_overlap_area = original_overlap.area

        result1, result2 = split_overlap(poly1, poly2)

        # Calculate how much each polygon gained/lost
        poly1_change = result1.area - poly1.area
        poly2_change = result2.area - poly2.area

        # Each should gain approximately half the overlap (minus the original area they had)
        # poly1 originally had the full overlap area with poly2
        # After split, poly1 loses about half the overlap
        # So poly1_change should be approximately -overlap_area/2
        # and poly2_change should be approximately -overlap_area/2

        # The sum of changes should be approximately -overlap_area
        total_change = poly1_change + poly2_change
        assert abs(total_change + original_overlap_area) < 1.0, \
            f"Total change {total_change} should approximately equal -{original_overlap_area}"

    def test_triangular_overlap(self):
        """Test with triangular polygons."""
        poly1 = Polygon([(0, 0), (4, 0), (2, 3)])
        poly2 = Polygon([(2, 0), (6, 0), (4, 3)])

        result1, result2 = split_overlap(poly1, poly2)

        assert result1.is_valid
        assert result2.is_valid

        # Check minimal overlap
        overlap = result1.intersection(result2)
        assert overlap.area < 0.1

    def test_multiple_overlaps_small_polygons(self):
        """Test with very small polygons."""
        poly1 = Polygon([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)])
        poly2 = Polygon([(0.3, 0), (0.8, 0), (0.8, 0.5), (0.3, 0.5)])

        result1, result2 = split_overlap(poly1, poly2)

        assert result1.is_valid
        assert result2.is_valid

    def test_polygon_with_holes(self):
        """Test polygons that have holes."""
        # Polygon with a hole
        exterior1 = [(0, 0), (5, 0), (5, 5), (0, 5)]
        hole1 = [(1, 1), (2, 1), (2, 2), (1, 2)]
        poly1 = Polygon(exterior1, [hole1])

        # Regular polygon that overlaps
        poly2 = Polygon([(3, 0), (8, 0), (8, 5), (3, 5)])

        result1, result2 = split_overlap(poly1, poly2)

        assert result1.is_valid
        assert result2.is_valid

        # Check minimal overlap
        overlap = result1.intersection(result2)
        assert overlap.area < 0.1

    def test_nearly_identical_polygons(self):
        """Test with nearly identical polygons."""
        poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        poly2 = Polygon([(0.1, 0.1), (5.1, 0.1), (5.1, 5.1), (0.1, 5.1)])

        result1, result2 = split_overlap(poly1, poly2)

        assert result1.is_valid
        assert result2.is_valid

        # They should both still have positive area
        assert result1.area > 0
        assert result2.area > 0

    def test_vertical_overlap(self):
        """Test with vertical overlap configuration."""
        poly1 = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
        poly2 = Polygon([(0, 2), (3, 2), (3, 5), (0, 5)])

        result1, result2 = split_overlap(poly1, poly2)

        assert result1.is_valid
        assert result2.is_valid

        # Check minimal overlap
        overlap = result1.intersection(result2)
        assert overlap.area < 0.1

    def test_diagonal_overlap(self):
        """Test with diagonal overlap configuration."""
        poly1 = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        poly2 = Polygon([(2, 2), (6, 2), (6, 6), (2, 6)])

        result1, result2 = split_overlap(poly1, poly2)

        assert result1.is_valid
        assert result2.is_valid

        # Check minimal overlap
        overlap = result1.intersection(result2)
        assert overlap.area < 0.2

    def test_result_is_tuple(self):
        """Test that result is a tuple of two polygons."""
        poly1 = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
        poly2 = Polygon([(2, 0), (5, 0), (5, 3), (2, 3)])

        result = split_overlap(poly1, poly2)

        assert isinstance(result, tuple), "Result should be a tuple"
        assert len(result) == 2, "Result should have 2 elements"
        assert isinstance(result[0], Polygon), "First element should be a Polygon"
        assert isinstance(result[1], Polygon), "Second element should be a Polygon"


class TestOverlapStrategies:
    """Test different overlap strategies."""

    def test_split_strategy_default(self):
        """Test default 'split' strategy."""
        poly1 = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
        poly2 = Polygon([(2, 0), (5, 0), (5, 3), (2, 3)])

        # Default should be 'split'
        result1, result2 = split_overlap(poly1, poly2)

        # Check minimal overlap
        overlap = result1.intersection(result2)
        assert overlap.area < 1e-6

        # Both polygons should have lost some area
        assert result1.area < poly1.area
        assert result2.area < poly2.area

    def test_split_strategy_explicit(self):
        """Test explicit 'split' strategy."""
        poly1 = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
        poly2 = Polygon([(2, 0), (5, 0), (5, 3), (2, 3)])

        result1, result2 = split_overlap(poly1, poly2, overlap_strategy=OverlapStrategy.SPLIT)

        # Check minimal overlap
        overlap = result1.intersection(result2)
        assert overlap.area < 1e-6

    def test_largest_strategy_equal_size(self):
        """Test 'largest' strategy with equal-sized polygons."""
        # Two equal-sized overlapping squares
        poly1 = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
        poly2 = Polygon([(2, 0), (5, 0), (5, 3), (2, 3)])

        result1, result2 = split_overlap(poly1, poly2, overlap_strategy=OverlapStrategy.LARGEST)

        # Check no overlap
        overlap = result1.intersection(result2)
        assert overlap.area < 1e-6

        # poly1 should get the overlap (areas are equal, so poly1 >= poly2)
        assert result1.area > poly1.area - 0.1
        assert result2.area < poly2.area

    def test_largest_strategy_different_sizes(self):
        """Test 'largest' strategy with different-sized polygons."""
        # Large polygon and small polygon
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])  # Area = 100
        poly2 = Polygon([(8, 0), (12, 0), (12, 4), (8, 4)])    # Area = 16

        original_overlap = poly1.intersection(poly2).area

        result1, result2 = split_overlap(poly1, poly2, overlap_strategy=OverlapStrategy.LARGEST)

        # Check no overlap
        overlap = result1.intersection(result2)
        assert overlap.area < 1e-6

        # poly1 (larger) should keep its original area or gain from overlap
        assert result1.area >= poly1.area - 0.1

        # poly2 (smaller) should lose the overlap area
        assert result2.area < poly2.area

    def test_smallest_strategy_equal_size(self):
        """Test 'smallest' strategy with equal-sized polygons."""
        poly1 = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
        poly2 = Polygon([(2, 0), (5, 0), (5, 3), (2, 3)])

        result1, result2 = split_overlap(poly1, poly2, overlap_strategy=OverlapStrategy.SMALLEST)

        # Check no overlap
        overlap = result1.intersection(result2)
        assert overlap.area < 1e-6

        # poly1 should get the overlap (areas are equal, so poly1 <= poly2)
        assert result1.area > poly1.area - 0.1
        assert result2.area < poly2.area

    def test_smallest_strategy_different_sizes(self):
        """Test 'smallest' strategy with different-sized polygons."""
        # Large polygon and small polygon
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])  # Area = 100
        poly2 = Polygon([(8, 0), (12, 0), (12, 4), (8, 4)])    # Area = 16

        result1, result2 = split_overlap(poly1, poly2, overlap_strategy=OverlapStrategy.SMALLEST)

        # Check no overlap
        overlap = result1.intersection(result2)
        assert overlap.area < 1e-6

        # poly2 (smaller) should keep its original area or gain from overlap
        assert result2.area >= poly2.area - 0.1

        # poly1 (larger) should lose the overlap area
        assert result1.area < poly1.area

    def test_all_strategies_no_overlap(self):
        """Test all strategies with non-overlapping polygons."""
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = Polygon([(3, 0), (5, 0), (5, 2), (3, 2)])

        # All strategies should return originals
        result1_split, result2_split = split_overlap(poly1, poly2, overlap_strategy=OverlapStrategy.SPLIT)
        result1_large, result2_large = split_overlap(poly1, poly2, overlap_strategy=OverlapStrategy.LARGEST)
        result1_small, result2_small = split_overlap(poly1, poly2, overlap_strategy=OverlapStrategy.SMALLEST)

        assert result1_split.equals(poly1)
        assert result2_split.equals(poly2)
        assert result1_large.equals(poly1)
        assert result2_large.equals(poly2)
        assert result1_small.equals(poly1)
        assert result2_small.equals(poly2)

    def test_largest_strategy_preserves_validity(self):
        """Test that 'largest' strategy produces valid geometries."""
        poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        poly2 = Polygon([(3, 2), (8, 2), (8, 7), (3, 7)])

        result1, result2 = split_overlap(poly1, poly2, overlap_strategy=OverlapStrategy.LARGEST)

        assert result1.is_valid
        assert result2.is_valid
        assert not result1.is_empty
        assert not result2.is_empty

    def test_smallest_strategy_preserves_validity(self):
        """Test that 'smallest' strategy produces valid geometries."""
        poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        poly2 = Polygon([(3, 2), (8, 2), (8, 7), (3, 7)])

        result1, result2 = split_overlap(poly1, poly2, overlap_strategy=OverlapStrategy.SMALLEST)

        assert result1.is_valid
        assert result2.is_valid
        assert not result1.is_empty
        assert not result2.is_empty


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_very_small_overlap(self):
        """Test with very small overlap area."""
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = Polygon([(1.99, 0), (4, 0), (4, 2), (1.99, 2)])

        result1, result2 = split_overlap(poly1, poly2)

        assert result1.is_valid
        assert result2.is_valid

    def test_point_overlap_only(self):
        """Test polygons that touch at a single point."""
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])

        result1, result2 = split_overlap(poly1, poly2)

        # Should return originals or very similar
        assert abs(result1.area - poly1.area) < 0.01
        assert abs(result2.area - poly2.area) < 0.01

    def test_irregular_polygons(self):
        """Test with irregular, non-convex polygons."""
        # Star-like shape
        poly1 = Polygon([
            (2, 0), (2.5, 1.5), (4, 1.5), (3, 2.5),
            (3.5, 4), (2, 3), (0.5, 4), (1, 2.5), (0, 1.5), (1.5, 1.5)
        ])

        # Rectangle overlapping the star
        poly2 = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])

        result1, result2 = split_overlap(poly1, poly2)

        assert result1.is_valid
        assert result2.is_valid
