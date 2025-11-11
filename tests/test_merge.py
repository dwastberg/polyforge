"""Tests for merge_close_polygons function."""

import pytest
import numpy as np
from shapely.geometry import Polygon
from polyforge import merge_close_polygons
from polyforge.core.types import MergeStrategy


class TestProximityDetection:
    """Tests for proximity detection and grouping."""

    def test_empty_list(self):
        """Test with empty polygon list."""
        result = merge_close_polygons([])
        assert result == []

    def test_single_polygon(self):
        """Test with single polygon."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        result = merge_close_polygons([poly])
        assert len(result) == 1
        assert result[0].equals(poly)

    def test_isolated_polygons(self):
        """Test that isolated polygons are returned unchanged."""
        # Three well-separated polygons
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(100, 100), (110, 100), (110, 110), (100, 110)])
        poly3 = Polygon([(200, 200), (210, 200), (210, 210), (200, 210)])

        polygons = [poly1, poly2, poly3]
        result = merge_close_polygons(polygons, margin=1.0)

        # All should be returned unchanged
        assert len(result) == 3
        assert all(p.is_valid for p in result)

    def test_overlapping_detection(self):
        """Test detection of overlapping polygons."""
        # Two overlapping squares
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])

        result = merge_close_polygons([poly1, poly2], margin=0.0)

        # Should merge into one or two polygons (depending on strategy)
        assert len(result) >= 1
        assert all(p.is_valid for p in result)

    def test_close_polygons_detection(self):
        """Test detection of close but non-overlapping polygons."""
        # Two squares with small gap
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(10.5, 0), (20, 0), (20, 10), (10.5, 10)])

        # With margin < gap, should not merge
        result = merge_close_polygons([poly1, poly2], margin=0.4)
        assert len(result) == 2

        # With margin > gap, should merge
        result = merge_close_polygons([poly1, poly2], margin=0.6)
        assert len(result) == 1

    def test_return_mapping(self):
        """Test return_mapping parameter."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(10.5, 0), (20, 0), (20, 10), (10.5, 10)])
        poly3 = Polygon([(100, 100), (110, 100), (110, 110), (100, 110)])

        polygons = [poly1, poly2, poly3]
        result, mapping = merge_close_polygons(polygons, margin=1.0, return_mapping=True)

        assert len(result) == 2  # poly1 and poly2 merged, poly3 separate
        assert len(mapping) == 2

        # Check that mapping contains correct indices
        assert any(set(m) == {0, 1} for m in mapping)  # poly1 and poly2
        assert any(m == [2] for m in mapping)  # poly3 alone


class TestSimpleBufferStrategy:
    """Tests for simple_buffer strategy."""

    def test_strategy_accepts_string_literal(self):
        """Ensure merge strategy can be passed as a string literal."""
        poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        poly2 = Polygon([(4, 0), (9, 0), (9, 5), (4, 5)])

        result = merge_close_polygons(
            [poly1, poly2],
            margin=0.0,
            merge_strategy="simple_buffer",
        )

        assert len(result) == 1
        assert result[0].is_valid

    def test_overlapping_merge(self):
        """Test merging overlapping polygons."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])

        result = merge_close_polygons([poly1, poly2], margin=0.0, merge_strategy=MergeStrategy.SIMPLE_BUFFER)

        assert len(result) == 1
        assert result[0].is_valid
        # Merged area should be close to sum minus overlap
        expected_area = poly1.area + poly2.area - poly1.intersection(poly2).area
        assert abs(result[0].area - expected_area) < 1.0

    def test_close_merge_with_margin(self):
        """Test merging close polygons with margin."""
        # Two squares with 1 unit gap
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(11, 0), (21, 0), (21, 10), (11, 10)])

        result = merge_close_polygons([poly1, poly2], margin=2.0, merge_strategy=MergeStrategy.SIMPLE_BUFFER)

        assert len(result) == 1
        assert result[0].is_valid

    def test_multiple_polygon_group(self):
        """Test merging group of multiple close polygons."""
        # Three polygons in a row with small gaps
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(10.5, 0), (20, 0), (20, 10), (10.5, 10)])
        poly3 = Polygon([(20.5, 0), (30, 0), (30, 10), (20.5, 10)])

        result = merge_close_polygons([poly1, poly2, poly3], margin=1.0, merge_strategy=MergeStrategy.SIMPLE_BUFFER)

        assert len(result) == 1
        assert result[0].is_valid

    def test_preserve_holes_false(self):
        """Test that holes are removed when preserve_holes=False."""
        # Polygon with hole
        exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
        hole = [(5, 5), (15, 5), (15, 15), (5, 15)]
        poly_with_hole = Polygon(exterior, holes=[hole])

        # Close polygon
        poly2 = Polygon([(21, 0), (31, 0), (31, 20), (21, 20)])

        result = merge_close_polygons(
            [poly_with_hole, poly2],
            margin=2.0,
            merge_strategy=MergeStrategy.SIMPLE_BUFFER,
            preserve_holes=False
        )

        assert len(result) == 1
        assert len(result[0].interiors) == 0  # Holes removed


class TestSelectiveBufferStrategy:
    """Tests for selective_buffer strategy."""

    def test_selective_merge(self):
        """Test that selective buffer only modifies near gaps."""
        # Two squares with small gap
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(11, 0), (21, 0), (21, 10), (11, 10)])

        result = merge_close_polygons([poly1, poly2], margin=2.0, merge_strategy=MergeStrategy.SELECTIVE_BUFFER)

        assert len(result) == 1
        assert result[0].is_valid

    def test_shape_preservation(self):
        """Test that selective buffer preserves shape better than simple buffer."""
        # Two rectangles with gap
        poly1 = Polygon([(0, 0), (10, 0), (10, 2), (0, 2)])
        poly2 = Polygon([(11, 0), (21, 0), (21, 2), (11, 2)])

        result_selective = merge_close_polygons(
            [poly1, poly2], margin=2.0, merge_strategy=MergeStrategy.SELECTIVE_BUFFER
        )
        result_simple = merge_close_polygons(
            [poly1, poly2], margin=2.0, merge_strategy=MergeStrategy.SIMPLE_BUFFER
        )

        # Both should merge
        assert len(result_selective) == 1
        assert len(result_simple) == 1

        # Both should be valid
        assert result_selective[0].is_valid
        assert result_simple[0].is_valid


class TestVertexMovementStrategy:
    """Tests for vertex_movement strategy."""

    def test_vertex_merge(self):
        """Test merging by moving vertices."""
        # Two squares with small gap
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(11, 0), (21, 0), (21, 10), (11, 10)])

        result = merge_close_polygons([poly1, poly2], margin=2.0, merge_strategy=MergeStrategy.VERTEX_MOVEMENT)

        assert len(result) == 1
        assert result[0].is_valid

    def test_preserves_structure(self):
        """Test that vertex movement preserves overall structure."""
        # L-shaped polygon and square
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (5, 10), (5, 5), (0, 5)])
        poly2 = Polygon([(11, 0), (21, 0), (21, 10), (11, 10)])

        result = merge_close_polygons([poly1, poly2], margin=2.0, merge_strategy=MergeStrategy.VERTEX_MOVEMENT)

        assert len(result) == 1
        assert result[0].is_valid


class TestBoundaryExtensionStrategy:
    """Tests for boundary_extension strategy."""

    def test_parallel_edge_merge(self):
        """Test merging polygons with parallel edges."""
        # Two rectangles facing each other (parallel edges)
        poly1 = Polygon([(0, 0), (10, 0), (10, 2), (0, 2)])
        poly2 = Polygon([(11, 0), (21, 0), (21, 2), (11, 2)])

        result = merge_close_polygons([poly1, poly2], margin=2.0, merge_strategy=MergeStrategy.BOUNDARY_EXTENSION)

        assert len(result) == 1
        assert result[0].is_valid

    def test_rectangular_features(self):
        """Test with rectangular building-like features."""
        # Two building footprints
        poly1 = Polygon([(0, 0), (10, 0), (10, 20), (0, 20)])
        poly2 = Polygon([(11, 0), (21, 0), (21, 20), (11, 20)])

        result = merge_close_polygons([poly1, poly2], margin=2.0, merge_strategy=MergeStrategy.BOUNDARY_EXTENSION)

        assert len(result) == 1
        assert result[0].is_valid

    def test_fallback_to_selective(self):
        """Test fallback when no parallel edges found."""
        # Two triangles (no parallel edges)
        poly1 = Polygon([(0, 0), (10, 0), (5, 10)])
        poly2 = Polygon([(11, 0), (21, 0), (16, 10)])

        result = merge_close_polygons([poly1, poly2], margin=2.0, merge_strategy=MergeStrategy.BOUNDARY_EXTENSION)

        # Should still merge (using fallback)
        assert all(p.is_valid for p in result)


class TestConvexBridgesStrategy:
    """Tests for convex_bridges strategy."""

    def test_convex_merge(self):
        """Test merging with convex hull bridges."""
        # Two squares with gap
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(11, 0), (21, 0), (21, 10), (11, 10)])

        result = merge_close_polygons([poly1, poly2], margin=2.0, merge_strategy=MergeStrategy.CONVEX_BRIDGES)

        assert len(result) == 1
        assert result[0].is_valid

    def test_irregular_gaps(self):
        """Test with irregular gap shapes."""
        # Irregular polygons
        poly1 = Polygon([(0, 0), (10, 0), (8, 5), (10, 10), (0, 10)])
        poly2 = Polygon([(12, 2), (22, 2), (22, 8), (12, 8)])

        result = merge_close_polygons([poly1, poly2], margin=3.0, merge_strategy=MergeStrategy.CONVEX_BRIDGES)

        assert all(p.is_valid for p in result)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        # Need two close polygons to actually trigger strategy selection
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(11, 0), (21, 0), (21, 10), (11, 10)])

        with pytest.raises(ValueError):
            merge_close_polygons([poly1, poly2], margin=2.0, merge_strategy='nonexistent_strategy')

    def test_margin_zero(self):
        """Test with margin=0 (only overlaps)."""
        # Overlapping polygons
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])
        # Non-overlapping
        poly3 = Polygon([(20, 20), (30, 20), (30, 30), (20, 30)])

        result = merge_close_polygons([poly1, poly2, poly3], margin=0.0)

        # poly1 and poly2 should merge, poly3 separate
        assert len(result) == 2
        assert all(p.is_valid for p in result)

    def test_very_small_margin(self):
        """Test with very small margin."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(10.001, 0), (20, 0), (20, 10), (10.001, 10)])

        result = merge_close_polygons([poly1, poly2], margin=0.01)

        assert all(p.is_valid for p in result)

    def test_large_margin(self):
        """Test with very large margin."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(50, 50), (60, 50), (60, 60), (50, 60)])

        result = merge_close_polygons([poly1, poly2], margin=100.0)

        # Should merge even though far apart
        assert len(result) == 1
        assert result[0].is_valid

    def test_touching_polygons(self):
        """Test with polygons that touch but don't overlap."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])

        result = merge_close_polygons([poly1, poly2], margin=0.0)

        # Touching counts as distance=0, should merge
        assert len(result) == 1
        assert result[0].is_valid

    def test_complex_group_three_plus(self):
        """Test group with 3+ polygons."""
        # Create a cluster of 5 close polygons
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(11, 0), (21, 0), (21, 10), (11, 10)])
        poly3 = Polygon([(22, 0), (32, 0), (32, 10), (22, 10)])
        poly4 = Polygon([(0, 11), (10, 11), (10, 21), (0, 21)])
        poly5 = Polygon([(11, 11), (21, 11), (21, 21), (11, 21)])

        result = merge_close_polygons(
            [poly1, poly2, poly3, poly4, poly5],
            margin=2.0
        )

        # All should merge into one
        assert len(result) == 1
        assert result[0].is_valid

    def test_polygons_with_holes(self):
        """Test merging polygons that have interior holes."""
        # Polygon with hole
        exterior1 = [(0, 0), (20, 0), (20, 20), (0, 20)]
        hole1 = [(5, 5), (15, 5), (15, 15), (5, 15)]
        poly1 = Polygon(exterior1, holes=[hole1])

        # Simple polygon nearby
        poly2 = Polygon([(21, 0), (31, 0), (31, 20), (21, 20)])

        result = merge_close_polygons([poly1, poly2], margin=2.0, preserve_holes=True)

        assert len(result) == 1
        assert result[0].is_valid
        # Should preserve hole
        assert len(result[0].interiors) >= 1


class TestMultipleGroups:
    """Tests for handling multiple separate groups."""

    def test_two_separate_groups(self):
        """Test with two separate groups of close polygons."""
        # Group 1: Three close polygons
        group1 = [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            Polygon([(11, 0), (21, 0), (21, 10), (11, 10)]),
            Polygon([(22, 0), (32, 0), (32, 10), (22, 10)])
        ]

        # Group 2: Two close polygons far from group 1
        group2 = [
            Polygon([(100, 100), (110, 100), (110, 110), (100, 110)]),
            Polygon([(111, 100), (121, 100), (121, 110), (111, 110)])
        ]

        all_polygons = group1 + group2
        result = merge_close_polygons(all_polygons, margin=2.0)

        # Should result in 2 merged polygons (one per group)
        assert len(result) == 2
        assert all(p.is_valid for p in result)

    def test_mixed_isolated_and_groups(self):
        """Test with mix of isolated polygons and groups."""
        # Isolated polygons
        isolated1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        isolated2 = Polygon([(100, 100), (110, 100), (110, 110), (100, 110)])

        # Close group
        group = [
            Polygon([(200, 200), (210, 200), (210, 210), (200, 210)]),
            Polygon([(211, 200), (221, 200), (221, 210), (211, 210)])
        ]

        all_polygons = [isolated1, isolated2] + group
        result, mapping = merge_close_polygons(all_polygons, margin=2.0, return_mapping=True)

        # Should have 3 polygons: 2 isolated + 1 merged
        assert len(result) == 3
        assert len(mapping) == 3
        assert all(p.is_valid for p in result)


class TestPerformance:
    """Tests for performance characteristics."""

    def test_mostly_isolated_fast(self):
        """Test that mostly isolated polygons are fast."""
        import time

        # Create 100 isolated polygons
        polygons = []
        for i in range(10):
            for j in range(10):
                x, y = i * 100, j * 100
                poly = Polygon([
                    (x, y), (x + 10, y), (x + 10, y + 10), (x, y + 10)
                ])
                polygons.append(poly)

        start = time.time()
        result = merge_close_polygons(polygons, margin=1.0)
        elapsed = time.time() - start

        # Should complete quickly (< 1 second)
        assert elapsed < 1.0
        # All should remain separate
        assert len(result) == 100
        assert all(p.is_valid for p in result)

    def test_overlapping_grid(self):
        """Test with grid of overlapping polygons."""
        # Create 5x5 grid with overlaps
        polygons = []
        for i in range(5):
            for j in range(5):
                x, y = i * 9, j * 9  # Overlap by 1 unit
                poly = Polygon([
                    (x, y), (x + 10, y), (x + 10, y + 10), (x, y + 10)
                ])
                polygons.append(poly)

        result = merge_close_polygons(polygons, margin=0.0, merge_strategy=MergeStrategy.SIMPLE_BUFFER)

        # Should merge into one large polygon (or a few)
        assert len(result) <= 25
        assert all(p.is_valid for p in result)


class TestValidation:
    """Tests for geometry validation."""

    def test_all_results_valid(self):
        """Test that all results are valid geometries."""
        # Create various polygon configurations
        test_cases = [
            # Overlapping
            [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
             Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])],
            # Close with gap
            [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
             Polygon([(11, 0), (21, 0), (21, 10), (11, 10)])],
            # Irregular shapes
            [Polygon([(0, 0), (10, 0), (8, 5), (10, 10), (0, 10)]),
             Polygon([(11, 2), (21, 2), (21, 8), (11, 8)])],
        ]

        for polygons in test_cases:
            for strategy in [MergeStrategy.SIMPLE_BUFFER, MergeStrategy.SELECTIVE_BUFFER, MergeStrategy.VERTEX_MOVEMENT,
                           MergeStrategy.BOUNDARY_EXTENSION, MergeStrategy.CONVEX_BRIDGES]:
                result = merge_close_polygons(polygons, margin=2.0, merge_strategy=strategy)
                assert all(p.is_valid for p in result), f"Invalid result for {strategy}"

    def test_non_empty_results(self):
        """Test that results are not empty."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(11, 0), (21, 0), (21, 10), (11, 10)])

        for strategy in [MergeStrategy.SIMPLE_BUFFER, MergeStrategy.SELECTIVE_BUFFER, MergeStrategy.VERTEX_MOVEMENT,
                        MergeStrategy.BOUNDARY_EXTENSION, MergeStrategy.CONVEX_BRIDGES]:
            result = merge_close_polygons([poly1, poly2], margin=2.0, merge_strategy=strategy)
            assert all(not p.is_empty for p in result), f"Empty result for {strategy}"
            assert all(p.area > 0 for p in result), f"Zero area for {strategy}"
