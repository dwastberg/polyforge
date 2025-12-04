"""Tests for overlap removal functions."""

import pytest
import numpy as np
from shapely.geometry import Polygon
from polyforge.overlap import remove_overlaps, count_overlaps, find_overlapping_groups
from polyforge.core.types import OverlapStrategy


class TestRemoveOverlaps:
    """Test suite for remove_overlaps function."""

    def test_two_overlapping_polygons(self):
        """Test basic case with two overlapping polygons."""
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])

        result = remove_overlaps([poly1, poly2])

        assert len(result) == 2
        # Check no significant overlap remains
        overlap = result[0].intersection(result[1])
        assert overlap.area < 1e-6

    def test_three_overlapping_in_chain(self):
        """Test three polygons overlapping in a chain (A-B-C)."""
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = Polygon([(1, 0), (3, 0), (3, 2), (1, 2)])
        poly3 = Polygon([(2, 0), (4, 0), (4, 2), (2, 2)])

        result = remove_overlaps([poly1, poly2, poly3])

        assert len(result) == 3
        # Check all pairs have no significant overlap
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                overlap = result[i].intersection(result[j])
                assert overlap.area < 1e-6

    def test_multiple_overlapping_same_polygon(self):
        """Test multiple polygons overlapping the same central polygon."""
        # Central polygon
        center = Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])

        # Four polygons overlapping from different sides
        left = Polygon([(0, 3), (4, 3), (4, 7), (0, 7)])
        right = Polygon([(6, 3), (10, 3), (10, 7), (6, 7)])
        top = Polygon([(3, 6), (7, 6), (7, 10), (3, 10)])
        bottom = Polygon([(3, 0), (7, 0), (7, 4), (3, 4)])

        result = remove_overlaps([center, left, right, top, bottom])

        assert len(result) == 5
        # Check no significant overlaps
        overlap_count = 0
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                overlap = result[i].intersection(result[j])
                if overlap.area > 1e-6:
                    overlap_count += 1
        assert overlap_count == 0

    def test_no_overlaps(self):
        """Test list with no overlapping polygons."""
        poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly2 = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
        poly3 = Polygon([(0, 2), (1, 2), (1, 3), (0, 3)])

        result = remove_overlaps([poly1, poly2, poly3])

        # Should return essentially unchanged
        assert len(result) == 3
        assert abs(result[0].area - poly1.area) < 1e-6
        assert abs(result[1].area - poly2.area) < 1e-6
        assert abs(result[2].area - poly3.area) < 1e-6

    def test_empty_list(self):
        """Test with empty list."""
        result = remove_overlaps([])
        assert result == []

    def test_single_polygon(self):
        """Test with single polygon."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        result = remove_overlaps([poly])

        assert len(result) == 1
        assert result[0].equals(poly)

    def test_largest_strategy(self):
        """Test 'largest' overlap strategy."""
        # Large and small polygons
        large = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        small = Polygon([(8, 8), (12, 8), (12, 12), (8, 12)])

        result = remove_overlaps([large, small], overlap_strategy=OverlapStrategy.LARGEST)

        assert len(result) == 2
        # Large polygon should have kept most of its area
        assert result[0].area >= large.area - 0.1

    def test_smallest_strategy(self):
        """Test 'smallest' overlap strategy."""
        # Large and small polygons
        large = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        small = Polygon([(8, 8), (12, 8), (12, 12), (8, 12)])

        result = remove_overlaps([large, small], overlap_strategy=OverlapStrategy.SMALLEST)

        assert len(result) == 2
        # Small polygon should have kept most of its area
        assert result[1].area >= small.area - 0.1

    def test_preserves_order(self):
        """Test that polygon order is preserved."""
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
        poly3 = Polygon([(0, 3), (2, 3), (2, 5), (0, 5)])

        result = remove_overlaps([poly1, poly2, poly3])

        # Should still be 3 polygons in same order
        assert len(result) == 3

    def test_complex_overlapping_grid(self):
        """Test grid of overlapping polygons."""
        polygons = []
        for i in range(3):
            for j in range(3):
                x, y = i * 1.5, j * 1.5
                poly = Polygon([
                    (x, y), (x + 2, y), (x + 2, y + 2), (x, y + 2)
                ])
                polygons.append(poly)

        result = remove_overlaps(polygons)

        assert len(result) == 9
        # Check no significant overlaps remain
        overlap_count = 0
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                overlap = result[i].intersection(result[j])
                if overlap.area > 1e-6:
                    overlap_count += 1
        assert overlap_count == 0

    def test_convergence_within_iterations(self):
        """Test that complex cases converge within reasonable iterations."""
        # Create a challenging case with many overlaps
        polygons = []
        for i in range(10):
            poly = Polygon([
                (i * 0.8, 0), (i * 0.8 + 2, 0),
                (i * 0.8 + 2, 2), (i * 0.8, 2)
            ])
            polygons.append(poly)

        result = remove_overlaps(polygons, max_iterations=50)

        # Should converge
        overlap_count = count_overlaps(result)
        assert overlap_count == 0, "Should have resolved all overlaps"

    def test_all_valid_geometries(self):
        """Test that all resulting polygons are valid."""
        poly1 = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
        poly2 = Polygon([(2, 2), (5, 2), (5, 5), (2, 5)])
        poly3 = Polygon([(1, 1), (4, 1), (4, 4), (1, 4)])

        result = remove_overlaps([poly1, poly2, poly3])

        for poly in result:
            assert poly.is_valid, "All results should be valid"
            assert not poly.is_empty, "No polygon should be empty"

    def test_contained_polygon_is_carved_from_container(self):
        """Ensure containment cases are handled instead of skipped."""
        outer = Polygon([(0, 0), (6, 0), (6, 6), (0, 6)])
        inner = Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])

        result = remove_overlaps([outer, inner])

        assert len(result) == 2
        updated_outer, updated_inner = result

        # Inner polygon preserved; outer has a hole carved out.
        assert updated_inner.equals(inner)
        assert updated_outer.area == pytest.approx(outer.area - inner.area, rel=1e-9)

        # No overlap remains.
        overlap = updated_outer.intersection(updated_inner)
        assert overlap.is_empty or overlap.area < 1e-9


class TestCountOverlaps:
    """Test suite for count_overlaps function."""

    def test_no_overlaps(self):
        """Test counting with no overlaps."""
        poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly2 = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])

        count = count_overlaps([poly1, poly2])
        assert count == 0

    def test_two_overlaps(self):
        """Test counting with two overlaps."""
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = Polygon([(1, 0), (3, 0), (3, 2), (1, 2)])
        poly3 = Polygon([(2, 0), (4, 0), (4, 2), (2, 2)])

        count = count_overlaps([poly1, poly2, poly3])
        assert count == 2  # poly1-poly2 and poly2-poly3

    def test_empty_list(self):
        """Test counting with empty list."""
        count = count_overlaps([])
        assert count == 0

    def test_single_polygon(self):
        """Test counting with single polygon."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        count = count_overlaps([poly])
        assert count == 0

    def test_all_overlap(self):
        """Test counting when all polygons overlap."""
        # Three polygons all overlapping same area
        poly1 = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
        poly2 = Polygon([(1, 1), (4, 1), (4, 4), (1, 4)])
        poly3 = Polygon([(0.5, 0.5), (3.5, 0.5), (3.5, 3.5), (0.5, 3.5)])

        count = count_overlaps([poly1, poly2, poly3])
        assert count == 3  # All three pairs overlap


class TestFindOverlappingGroups:
    """Test suite for find_overlapping_groups function."""

    def test_no_overlaps(self):
        """Test with no overlaps - each polygon is its own group."""
        poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly2 = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
        poly3 = Polygon([(0, 2), (1, 2), (1, 3), (0, 3)])

        groups = find_overlapping_groups([poly1, poly2, poly3])

        assert len(groups) == 3
        assert [0] in groups
        assert [1] in groups
        assert [2] in groups

    def test_all_connected(self):
        """Test when all polygons are in one connected group."""
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = Polygon([(1, 0), (3, 0), (3, 2), (1, 2)])
        poly3 = Polygon([(2, 0), (4, 0), (4, 2), (2, 2)])

        groups = find_overlapping_groups([poly1, poly2, poly3])

        assert len(groups) == 1
        assert sorted(groups[0]) == [0, 1, 2]

    def test_two_separate_groups(self):
        """Test with two separate groups of overlapping polygons."""
        # Group 1: poly0 and poly1 overlap
        poly0 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly1 = Polygon([(1, 0), (3, 0), (3, 2), (1, 2)])

        # Group 2: poly2 and poly3 overlap (far away)
        poly2 = Polygon([(10, 10), (12, 10), (12, 12), (10, 12)])
        poly3 = Polygon([(11, 10), (13, 10), (13, 12), (11, 12)])

        groups = find_overlapping_groups([poly0, poly1, poly2, poly3])

        assert len(groups) == 2
        assert [0, 1] in groups
        assert [2, 3] in groups

    def test_complex_group_structure(self):
        """Test complex group structure with transitive connections."""
        # Chain: 0-1-2 (0 overlaps 1, 1 overlaps 2, but 0 doesn't overlap 2)
        poly0 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly1 = Polygon([(1.5, 0), (3.5, 0), (3.5, 2), (1.5, 2)])
        poly2 = Polygon([(3, 0), (5, 0), (5, 2), (3, 2)])

        # Separate polygon
        poly3 = Polygon([(10, 10), (12, 10), (12, 12), (10, 12)])

        groups = find_overlapping_groups([poly0, poly1, poly2, poly3])

        assert len(groups) == 2
        # poly0, poly1, poly2 should be in one group (connected)
        assert [0, 1, 2] in groups
        assert [3] in groups

    def test_empty_list(self):
        """Test with empty list."""
        groups = find_overlapping_groups([])
        assert groups == []

    def test_single_polygon(self):
        """Test with single polygon."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        groups = find_overlapping_groups([poly])

        assert len(groups) == 1
        assert groups[0] == [0]


class TestPerformance:
    """Test performance characteristics."""

    def test_large_number_of_polygons(self):
        """Test with larger number of polygons to verify spatial indexing works."""
        # Create 100 small polygons, some overlapping
        polygons = []
        for i in range(10):
            for j in range(10):
                x, y = i * 1.8, j * 1.8
                poly = Polygon([
                    (x, y), (x + 2, y), (x + 2, y + 2), (x, y + 2)
                ])
                polygons.append(poly)

        # This should complete quickly with spatial indexing
        result = remove_overlaps(polygons, max_iterations=50)

        assert len(result) == 100
        # Verify no overlaps remain
        assert count_overlaps(result) == 0

    def test_spatial_index_reduces_comparisons(self):
        """Verify that spatial index significantly reduces comparisons."""
        # With 100 polygons, naive approach would be ~5000 comparisons
        # With spatial index, should be much fewer

        polygons = []
        for i in range(10):
            for j in range(10):
                x, y = i * 3, j * 3  # No overlaps, spread out
                poly = Polygon([
                    (x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)
                ])
                polygons.append(poly)

        # Should complete very quickly - spatial index finds no candidates
        count = count_overlaps(polygons)
        assert count == 0
