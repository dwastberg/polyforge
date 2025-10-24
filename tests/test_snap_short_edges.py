import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from polyforge.simplify import snap_short_edges


class TestSnapShortEdges:
    """Tests for snap_short_edges function."""

    def test_linestring_with_short_edges(self):
        """Test snapping short edges in a LineString."""
        # LineString with edges: 0.001, 0.999, 0.002, 1.0
        line = LineString([(0, 0), (0.001, 0), (1, 0), (1, 0.002), (1, 1)])
        result = snap_short_edges(line, min_length=0.01)

        # Should remove the two short edges
        coords = np.array(result.coords)
        assert len(coords) == 3  # Down from 5 vertices

    def test_linestring_midpoint_mode(self):
        """Test that midpoint mode averages vertices correctly."""
        line = LineString([(0, 0), (0.005, 0), (1, 0)])
        result = snap_short_edges(line, min_length=0.01, snap_mode='midpoint')

        coords = np.array(result.coords)
        # First two vertices should be snapped to midpoint
        assert coords[0][0] == pytest.approx(0.0025)
        assert coords[0][1] == pytest.approx(0.0)

    def test_linestring_first_mode(self):
        """Test that 'first' mode keeps the first vertex."""
        line = LineString([(0, 0), (0.005, 0), (1, 0)])
        result = snap_short_edges(line, min_length=0.01, snap_mode='first')

        coords = np.array(result.coords)
        # Should keep first vertex
        assert coords[0][0] == pytest.approx(0.0)
        assert coords[0][1] == pytest.approx(0.0)

    def test_linestring_last_mode(self):
        """Test that 'last' mode keeps the last vertex."""
        line = LineString([(0, 0), (0.005, 0), (1, 0)])
        result = snap_short_edges(line, min_length=0.01, snap_mode='last')

        coords = np.array(result.coords)
        # Should keep last of the snapped pair
        assert coords[0][0] == pytest.approx(0.005)
        assert coords[0][1] == pytest.approx(0.0)

    def test_polygon_with_short_edges(self):
        """Test snapping short edges in a Polygon."""
        # Square with one short edge
        poly = Polygon([(0, 0), (0.002, 0.001), (1, 0), (1, 1), (0, 1)])
        result = snap_short_edges(poly, min_length=0.01)

        coords = np.array(result.exterior.coords)
        # Should have fewer vertices
        assert len(coords) < 6  # Original is 6 (including closing)

    def test_polygon_closed_ring_handling(self):
        """Test that polygon closing vertex is handled correctly."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0.001, 1)])
        result = snap_short_edges(poly, min_length=0.01)

        # Result should still be a valid closed polygon
        assert result.is_valid
        coords = np.array(result.exterior.coords)
        # First and last should be same (closed ring)
        np.testing.assert_array_almost_equal(coords[0], coords[-1])

    def test_polygon_wrap_around_edge(self):
        """Test snapping when the edge between last and first vertex is short."""
        # Create polygon where last vertex is very close to first
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 0.001)])
        result = snap_short_edges(poly, min_length=0.01, snap_mode='midpoint')

        coords = np.array(result.exterior.coords)
        # First and last should be snapped together
        assert result.is_valid
        np.testing.assert_array_almost_equal(coords[0], coords[-1])

    def test_multiple_consecutive_short_edges(self):
        """Test snapping multiple consecutive short edges."""
        line = LineString([
            (0, 0),
            (0.001, 0),
            (0.002, 0),
            (0.003, 0),
            (1, 0)
        ])
        result = snap_short_edges(line, min_length=0.01)

        coords = np.array(result.coords)
        # All short edges should collapse to one vertex
        assert len(coords) == 2

    def test_preserves_minimum_vertices(self):
        """Test that at least 2 vertices are preserved."""
        # All edges are short
        line = LineString([(0, 0), (0.001, 0), (0.002, 0)])
        result = snap_short_edges(line, min_length=1.0)

        coords = np.array(result.coords)
        assert len(coords) >= 2

    def test_no_short_edges(self):
        """Test geometry with no short edges remains unchanged."""
        line = LineString([(0, 0), (1, 0), (2, 0), (3, 0)])
        result = snap_short_edges(line, min_length=0.01)

        np.testing.assert_array_almost_equal(
            np.array(line.coords),
            np.array(result.coords)
        )

    def test_empty_geometry(self):
        """Test handling of empty geometry."""
        empty_line = LineString()
        result = snap_short_edges(empty_line, min_length=0.01)
        assert result.is_empty

    def test_single_point_linestring(self):
        """Test handling of degenerate LineString."""
        # LineString with only 2 identical points
        line = LineString([(0, 0), (0, 0)])
        result = snap_short_edges(line, min_length=0.01)
        assert len(result.coords) >= 2

    def test_3d_geometry_z_preserved(self):
        """Test that Z coordinates are preserved with snapping."""
        # Note: When snapping removes vertices, the Z values from removed vertices
        # are lost. This is expected behavior since we can't keep Z from deleted vertices.
        line = LineString([(0, 0, 10), (1, 0, 20), (2, 0, 30)])
        # All edges are long, no snapping should occur
        result = snap_short_edges(line, min_length=0.01)

        # Z should be preserved for all kept vertices
        assert result.has_z
        coords = np.array(result.coords)
        assert coords.shape[1] == 3
        # All vertices kept, Z values should match
        assert coords[0][2] == pytest.approx(10.0)
        assert coords[1][2] == pytest.approx(20.0)
        assert coords[2][2] == pytest.approx(30.0)
