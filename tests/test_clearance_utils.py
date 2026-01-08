"""Tests for clearance utility functions.

Note: These tests cover internal utility functions from the ops layer.
They are not part of the public API but are tested for correctness.
"""

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from polyforge.ops.clearance.utils import (
    _find_nearest_vertex_index,
    _find_nearest_edge_index,
    _point_to_segment_distance,
    _get_vertex_neighborhood,
    _calculate_curvature_at_vertex,
    _remove_vertices_between
)


class TestFindNearestVertexIndex:
    """Tests for _find_nearest_vertex_index."""

    def test_find_nearest_with_point_object(self):
        """Test finding nearest vertex with Point object."""
        coords = np.array([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        point = Point(0.1, 0.1)

        idx = _find_nearest_vertex_index(coords, point)

        assert idx == 0  # Nearest to (0, 0)

    def test_find_nearest_with_tuple(self):
        """Test finding nearest vertex with tuple."""
        coords = np.array([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        point = (0.9, 0.1)

        idx = _find_nearest_vertex_index(coords, point)

        assert idx == 1  # Nearest to (1, 0)

    def test_find_nearest_with_array(self):
        """Test finding nearest vertex with numpy array."""
        coords = np.array([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        point = np.array([0.9, 0.9])

        idx = _find_nearest_vertex_index(coords, point)

        assert idx == 2  # Nearest to (1, 1)

    def test_excludes_closing_vertex(self):
        """Test that duplicate closing vertex is excluded."""
        coords = np.array([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        point = (0, 0)

        idx = _find_nearest_vertex_index(coords, point)

        # Should return 0 (first vertex), not 4 (duplicate)
        assert idx == 0


class TestFindNearestEdgeIndex:
    """Tests for _find_nearest_edge_index."""

    def test_find_nearest_edge(self):
        """Test finding nearest edge."""
        coords = np.array([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
        point = np.array([1, -0.5])  # Below bottom edge

        idx = _find_nearest_edge_index(coords, point)

        assert idx == 0  # Edge from (0,0) to (2,0)

    def test_find_nearest_edge_to_side(self):
        """Test finding edge on side."""
        coords = np.array([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
        point = np.array([2.5, 1])  # To the right of right edge

        idx = _find_nearest_edge_index(coords, point)

        assert idx == 1  # Edge from (2,0) to (2,2)


class TestPointToSegmentDistance:
    """Tests for _point_to_segment_distance."""

    def test_distance_to_horizontal_segment(self):
        """Test distance to horizontal segment."""
        point = np.array([1, 2])
        segment_start = np.array([0, 0])
        segment_end = np.array([2, 0])

        dist = _point_to_segment_distance(point, segment_start, segment_end)

        assert dist == pytest.approx(2.0)

    def test_distance_to_vertical_segment(self):
        """Test distance to vertical segment."""
        point = np.array([3, 1])
        segment_start = np.array([0, 0])
        segment_end = np.array([0, 2])

        dist = _point_to_segment_distance(point, segment_start, segment_end)

        assert dist == pytest.approx(3.0)

    def test_distance_projection_before_start(self):
        """Test when perpendicular projection is before segment start."""
        point = np.array([-1, 1])
        segment_start = np.array([0, 0])
        segment_end = np.array([2, 0])

        dist = _point_to_segment_distance(point, segment_start, segment_end)

        # Should be distance to start point
        assert dist == pytest.approx(np.sqrt(2))

    def test_distance_projection_after_end(self):
        """Test when perpendicular projection is after segment end."""
        point = np.array([3, 1])
        segment_start = np.array([0, 0])
        segment_end = np.array([2, 0])

        dist = _point_to_segment_distance(point, segment_start, segment_end)

        # Should be distance to end point
        assert dist == pytest.approx(np.sqrt(2))

    def test_degenerate_segment(self):
        """Test with degenerate segment (start == end)."""
        point = np.array([1, 1])
        segment_start = np.array([0, 0])
        segment_end = np.array([0, 0])

        dist = _point_to_segment_distance(point, segment_start, segment_end)

        assert dist == pytest.approx(np.sqrt(2))


class TestGetVertexNeighborhood:
    """Tests for _get_vertex_neighborhood."""

    def test_neighborhood_middle_vertex(self):
        """Test neighborhood of middle vertex."""
        coords = np.array([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 0)])

        indices = _get_vertex_neighborhood(2, coords, radius=1)

        assert indices == [1, 2, 3]

    def test_neighborhood_wraps_around(self):
        """Test neighborhood wrapping around array boundary."""
        coords = np.array([(0, 0), (1, 0), (2, 0), (3, 0), (0, 0)])

        indices = _get_vertex_neighborhood(0, coords, radius=1)

        # Should wrap: last vertex (3), vertex 0, vertex 1
        assert indices == [3, 0, 1]

    def test_neighborhood_larger_radius(self):
        """Test with larger radius."""
        coords = np.array([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (0, 0)])

        indices = _get_vertex_neighborhood(3, coords, radius=2)

        assert indices == [1, 2, 3, 4, 5]


class TestCalculateCurvatureAtVertex:
    """Tests for _calculate_curvature_at_vertex."""

    def test_straight_line_angle(self):
        """Test angle at vertex on straight line.

        On a straight line, the vectors to previous and next point
        are in opposite directions, giving 180 degrees.
        """
        coords = np.array([(0, 0), (1, 0), (2, 0), (0, 0)])

        angle = _calculate_curvature_at_vertex(coords, 1)

        # Straight line: vectors point in opposite directions = 180°
        assert angle == pytest.approx(180.0, abs=1)

    def test_right_angle(self):
        """Test 90 degree turn."""
        coords = np.array([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])

        angle = _calculate_curvature_at_vertex(coords, 1)

        assert angle == pytest.approx(90.0, abs=1)

    def test_sharp_turn(self):
        """Test sharp reversal (close to 180 degrees)."""
        # Create a sharp spike that nearly reverses direction
        coords = np.array([(0, 0), (1, 0), (1.01, 0), (0, 0)])

        angle = _calculate_curvature_at_vertex(coords, 1)

        # Nearly reversing direction
        assert angle > 170.0
        assert angle <= 180.0


class TestRemoveVerticesBetween:
    """Tests for _remove_vertices_between."""

    def test_remove_middle_vertices(self):
        """Test removing vertices in middle of array."""
        coords = np.array([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 0)])

        new_coords = _remove_vertices_between(coords, start_idx=1, end_idx=4)

        expected = np.array([(0, 0), (1, 0), (4, 0), (0, 0)])
        np.testing.assert_array_almost_equal(new_coords, expected)

    def test_remove_wrapped_vertices(self):
        """Test removing vertices when range wraps around."""
        coords = np.array([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)])

        # Remove from index 4 to index 1 (wrapping around)
        new_coords = _remove_vertices_between(coords, start_idx=4, end_idx=1)

        expected = np.array([(1, 0), (2, 0), (3, 0), (4, 0)])
        np.testing.assert_array_almost_equal(new_coords, expected)

    def test_remove_single_vertex(self):
        """Test removing single vertex."""
        coords = np.array([(0, 0), (1, 0), (2, 0), (3, 0), (0, 0)])

        new_coords = _remove_vertices_between(coords, start_idx=1, end_idx=2)

        expected = np.array([(0, 0), (1, 0), (2, 0), (3, 0), (0, 0)])
        np.testing.assert_array_almost_equal(new_coords, expected)

