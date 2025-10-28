import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, Point, MultiPolygon

from polyforge.simplify import (
    collapse_short_edges,
    deduplicate_vertices,
    simplify_rdp,
    simplify_vw,
    simplify_vwp,
    remove_small_holes
)
from polyforge.core.types import CollapseMode


class TestSnapShortEdges:
    """Tests for collapse_short_edges function."""

    def test_linestring_with_short_edges(self):
        """Test snapping short edges in a LineString."""
        # LineString with edges: 0.001, 0.999, 0.002, 1.0
        line = LineString([(0, 0), (0.001, 0), (1, 0), (1, 0.002), (1, 1)])
        result = collapse_short_edges(line, min_length=0.01)

        # Should remove the two short edges
        coords = np.array(result.coords)
        assert len(coords) == 3  # Down from 5 vertices

    def test_linestring_midpoint_mode(self):
        """Test that midpoint mode averages vertices correctly."""
        line = LineString([(0, 0), (0.005, 0), (1, 0)])
        result = collapse_short_edges(line, min_length=0.01, snap_mode=CollapseMode.MIDPOINT)

        coords = np.array(result.coords)
        # First two vertices should be snapped to midpoint
        assert coords[0][0] == pytest.approx(0.0025)
        assert coords[0][1] == pytest.approx(0.0)

    def test_linestring_first_mode(self):
        """Test that 'first' mode keeps the first vertex."""
        line = LineString([(0, 0), (0.005, 0), (1, 0)])
        result = collapse_short_edges(line, min_length=0.01, snap_mode=CollapseMode.FIRST)

        coords = np.array(result.coords)
        # Should keep first vertex
        assert coords[0][0] == pytest.approx(0.0)
        assert coords[0][1] == pytest.approx(0.0)

    def test_linestring_last_mode(self):
        """Test that 'last' mode keeps the last vertex."""
        line = LineString([(0, 0), (0.005, 0), (1, 0)])
        result = collapse_short_edges(line, min_length=0.01, snap_mode=CollapseMode.LAST)

        coords = np.array(result.coords)
        # Should keep last of the snapped pair
        assert coords[0][0] == pytest.approx(0.005)
        assert coords[0][1] == pytest.approx(0.0)

    def test_polygon_with_short_edges(self):
        """Test snapping short edges in a Polygon."""
        # Square with one short edge
        poly = Polygon([(0, 0), (0.002, 0.001), (1, 0), (1, 1), (0, 1)])
        result = collapse_short_edges(poly, min_length=0.01)

        coords = np.array(result.exterior.coords)
        # Should have fewer vertices
        assert len(coords) < 6  # Original is 6 (including closing)

    def test_polygon_closed_ring_handling(self):
        """Test that polygon closing vertex is handled correctly."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0.001, 1)])
        result = collapse_short_edges(poly, min_length=0.01)

        # Result should still be a valid closed polygon
        assert result.is_valid
        coords = np.array(result.exterior.coords)
        # First and last should be same (closed ring)
        np.testing.assert_array_almost_equal(coords[0], coords[-1])

    def test_polygon_wrap_around_edge(self):
        """Test snapping when the edge between last and first vertex is short."""
        # Create polygon where last vertex is very close to first
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 0.001)])
        result = collapse_short_edges(poly, min_length=0.01, snap_mode=CollapseMode.MIDPOINT)

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
        result = collapse_short_edges(line, min_length=0.01)

        coords = np.array(result.coords)
        # All short edges should collapse to one vertex
        assert len(coords) == 2

    def test_preserves_minimum_vertices(self):
        """Test that at least 2 vertices are preserved."""
        # All edges are short
        line = LineString([(0, 0), (0.001, 0), (0.002, 0)])
        result = collapse_short_edges(line, min_length=1.0)

        coords = np.array(result.coords)
        assert len(coords) >= 2

    def test_no_short_edges(self):
        """Test geometry with no short edges remains unchanged."""
        line = LineString([(0, 0), (1, 0), (2, 0), (3, 0)])
        result = collapse_short_edges(line, min_length=0.01)

        np.testing.assert_array_almost_equal(
            np.array(line.coords),
            np.array(result.coords)
        )

    def test_empty_geometry(self):
        """Test handling of empty geometry."""
        empty_line = LineString()
        result = collapse_short_edges(empty_line, min_length=0.01)
        assert result.is_empty

    def test_single_point_linestring(self):
        """Test handling of degenerate LineString."""
        # LineString with only 2 identical points
        line = LineString([(0, 0), (0, 0)])
        result = collapse_short_edges(line, min_length=0.01)
        assert len(result.coords) >= 2

    def test_3d_geometry_z_preserved(self):
        """Test that Z coordinates are preserved with snapping."""
        # Note: When snapping removes vertices, the Z values from removed vertices
        # are lost. This is expected behavior since we can't keep Z from deleted vertices.
        line = LineString([(0, 0, 10), (1, 0, 20), (2, 0, 30)])
        # All edges are long, no snapping should occur
        result = collapse_short_edges(line, min_length=0.01)

        # Z should be preserved for all kept vertices
        assert result.has_z
        coords = np.array(result.coords)
        assert coords.shape[1] == 3
        # All vertices kept, Z values should match
        assert coords[0][2] == pytest.approx(10.0)
        assert coords[1][2] == pytest.approx(20.0)
        assert coords[2][2] == pytest.approx(30.0)


class TestRemoveDuplicateVertices:
    """Tests for deduplicate_vertices function."""

    def test_remove_exact_duplicates(self):
        """Test removing exact duplicate vertices."""
        line = LineString([(0, 0), (0, 0), (1, 1), (1, 1), (2, 2)])
        result = deduplicate_vertices(line)

        coords = np.array(result.coords)
        assert len(coords) == 3  # (0,0), (1,1), (2,2)

    def test_remove_duplicates_with_tolerance(self):
        """Test removing duplicates within tolerance."""
        line = LineString([(0, 0), (1e-12, 1e-12), (1, 1)])
        result = deduplicate_vertices(line, tolerance=1e-10)

        coords = np.array(result.coords)
        assert len(coords) == 2

    def test_no_duplicates(self):
        """Test geometry without duplicates remains unchanged."""
        line = LineString([(0, 0), (1, 1), (2, 2)])
        result = deduplicate_vertices(line)

        np.testing.assert_array_almost_equal(
            np.array(line.coords),
            np.array(result.coords)
        )

    def test_polygon_closed_ring(self):
        """Test that closed ring is maintained."""
        poly = Polygon([(0, 0), (0, 0), (1, 0), (1, 0), (1, 1), (0, 1)])
        result = deduplicate_vertices(poly)

        coords = np.array(result.exterior.coords)
        # Should still be closed
        np.testing.assert_array_almost_equal(coords[0], coords[-1])




class TestSimplifyRDP:
    """Tests for RDP simplification."""

    def test_linestring_rdp_basic(self):
        """Test basic RDP simplification."""
        # Create a wavy line
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        line = LineString(zip(x, y))

        result = simplify_rdp(line, epsilon=0.1)

        # Should have significantly fewer vertices
        assert len(result.coords) < len(line.coords)
        assert len(result.coords) > 2  # But not too few

    def test_straight_line_rdp(self):
        """Test RDP on a straight line."""
        line = LineString([(0, 0), (1, 0), (2, 0), (3, 0)])
        result = simplify_rdp(line, epsilon=0.01)

        # Should collapse to endpoints
        assert len(result.coords) == 2

    def test_polygon_rdp(self):
        """Test RDP on a polygon."""
        # Square with extra points on edges
        poly = Polygon([
            (0, 0), (0.5, 0), (1, 0),
            (1, 0.5), (1, 1),
            (0.5, 1), (0, 1),
            (0, 0.5)
        ])
        result = simplify_rdp(poly, epsilon=0.1)

        # Should simplify to fewer vertices
        assert len(result.exterior.coords) < len(poly.exterior.coords)
        assert result.is_valid

    def test_empty_geometry_rdp(self):
        """Test RDP with empty geometry."""
        empty = LineString()
        result = simplify_rdp(empty, epsilon=1.0)
        assert result.is_empty

    def test_3d_geometry_rdp(self):
        """Test RDP with 3D geometry - Z coordinates cannot be fully preserved.

        Note: When simplification removes vertices, the Z values for those removed
        vertices are lost. This is expected behavior for simplification algorithms.
        """
        # Use a line where no vertices will be removed (straight line in 3D)
        line = LineString([(0, 0, 10), (1, 1, 20), (2, 2, 30), (3, 3, 40)])
        # Large epsilon won't simplify this varied line much
        result = simplify_rdp(line, epsilon=10.0)

        # The function works, output is valid
        assert isinstance(result, LineString)
        assert len(result.coords) >= 2


class TestSimplifyVW:
    """Tests for Visvalingam-Whyatt simplification."""

    def test_linestring_vw_basic(self):
        """Test basic VW simplification."""
        # Create a curved line
        t = np.linspace(0, 2*np.pi, 200)
        line = LineString(zip(np.cos(t), np.sin(t)))

        result = simplify_vw(line, threshold=0.001)

        # Should simplify significantly
        assert len(result.coords) < len(line.coords)
        assert len(result.coords) > 10

    def test_polygon_vw(self):
        """Test VW on a polygon."""
        # Create polygon with many vertices
        t = np.linspace(0, 2*np.pi, 50)
        r = 5 + 0.5 * np.sin(10 * t)
        exterior = list(zip(r * np.cos(t), r * np.sin(t)))
        poly = Polygon(exterior)

        result = simplify_vw(poly, threshold=0.1)

        assert len(result.exterior.coords) < len(poly.exterior.coords)

    def test_3d_geometry_vw(self):
        """Test VW preserves Z coordinates."""
        line = LineString([(i, np.sin(i), i*10) for i in np.linspace(0, 10, 100)])
        result = simplify_vw(line, threshold=0.1)

        assert result.has_z


class TestSimplifyVWP:
    """Tests for topology-preserving VW simplification."""

    def test_linestring_vwp_basic(self):
        """Test basic VWP simplification."""
        line = LineString([(i, i**2) for i in range(50)])
        result = simplify_vwp(line, threshold=10.0)

        assert len(result.coords) < len(line.coords)

    def test_polygon_vwp_valid(self):
        """Test that VWP maintains validity."""
        # Create a complex polygon
        t = np.linspace(0, 2*np.pi, 100)
        r = 5 + 2 * np.sin(5 * t)
        exterior = list(zip(r * np.cos(t), r * np.sin(t)))
        poly = Polygon(exterior)

        # Aggressive simplification
        result = simplify_vwp(poly, threshold=1.0)

        # Should still be valid
        assert result.is_valid
        assert len(result.exterior.coords) < len(poly.exterior.coords)

    def test_3d_geometry_vwp(self):
        """Test VWP preserves Z coordinates."""
        # Create a proper 3D polygon (not a degenerate line)
        # Use a star-shaped polygon with varying Z values
        t = np.linspace(0, 2*np.pi, 20, endpoint=False)
        r = 5 + 2 * np.sin(5 * t)
        x = r * np.cos(t)
        y = r * np.sin(t)
        z = 100 + 10 * np.sin(3 * t)  # Varying Z values
        poly = Polygon(zip(x, y, z))

        result = simplify_vwp(poly, threshold=0.5)

        assert result.has_z
        # Should still have some vertices after simplification
        assert len(result.exterior.coords) >= 4  # Minimum for a valid polygon


class TestRemoveSmallHoles:
    """Tests for remove_small_holes function."""

    def test_remove_single_small_hole(self):
        """Test removing a single small hole from a polygon."""
        # Outer ring: square
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        # Inner ring (hole): small square with area 1
        hole = [(4, 4), (5, 4), (5, 5), (4, 5)]

        poly = Polygon(exterior, holes=[hole])
        result = remove_small_holes(poly, min_area=2.0)

        # Hole should be removed
        assert len(result.interiors) == 0
        # Exterior should remain unchanged
        assert result.exterior.coords[:] == Polygon(exterior).exterior.coords[:]

    def test_keep_large_hole(self):
        """Test that large holes are preserved."""
        # Outer ring: square
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        # Inner ring (hole): larger square with area 16
        hole = [(2, 2), (6, 2), (6, 6), (2, 6)]

        poly = Polygon(exterior, holes=[hole])
        result = remove_small_holes(poly, min_area=2.0)

        # Hole should be kept
        assert len(result.interiors) == 1
        # Hole coordinates should match
        np.testing.assert_array_almost_equal(
            np.array(result.interiors[0].coords),
            np.array(hole + [hole[0]])  # Add closing coordinate
        )

    def test_remove_multiple_small_holes(self):
        """Test removing multiple small holes."""
        exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
        # Create three small holes
        hole1 = [(2, 2), (3, 2), (3, 3), (2, 3)]  # area = 1
        hole2 = [(7, 7), (8, 7), (8, 8), (7, 8)]  # area = 1
        hole3 = [(12, 12), (13, 12), (13, 13), (12, 13)]  # area = 1

        poly = Polygon(exterior, holes=[hole1, hole2, hole3])
        result = remove_small_holes(poly, min_area=2.0)

        # All three holes should be removed
        assert len(result.interiors) == 0

    def test_mixed_hole_sizes(self):
        """Test with both small and large holes."""
        exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
        small_hole = [(2, 2), (3, 2), (3, 3), (2, 3)]  # area = 1
        large_hole = [(7, 7), (13, 7), (13, 13), (7, 13)]  # area = 36

        poly = Polygon(exterior, holes=[small_hole, large_hole])
        result = remove_small_holes(poly, min_area=2.0)

        # Only large hole should remain
        assert len(result.interiors) == 1
        # Verify it's the large hole
        hole_area = Polygon(result.interiors[0]).area
        assert hole_area == pytest.approx(36.0)

    def test_no_holes(self):
        """Test polygon without any holes."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        poly = Polygon(exterior)

        result = remove_small_holes(poly, min_area=1.0)

        # Should return polygon unchanged
        assert len(result.interiors) == 0
        assert result.exterior.coords[:] == poly.exterior.coords[:]

    def test_multipolygon_with_small_holes(self):
        """Test removing small holes from MultiPolygon."""
        # First polygon with one small hole
        exterior1 = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole1 = [(4, 4), (5, 4), (5, 5), (4, 5)]  # area = 1
        poly1 = Polygon(exterior1, holes=[hole1])

        # Second polygon with one large hole
        exterior2 = [(15, 0), (25, 0), (25, 10), (15, 10)]
        hole2 = [(17, 2), (23, 2), (23, 8), (17, 8)]  # area = 36
        poly2 = Polygon(exterior2, holes=[hole2])

        multipoly = MultiPolygon([poly1, poly2])
        result = remove_small_holes(multipoly, min_area=2.0)

        # Result should be a MultiPolygon
        assert isinstance(result, MultiPolygon)
        assert len(result.geoms) == 2

        # First polygon should have no holes
        assert len(result.geoms[0].interiors) == 0

        # Second polygon should still have its large hole
        assert len(result.geoms[1].interiors) == 1

    def test_multipolygon_no_holes(self):
        """Test MultiPolygon with no holes."""
        poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        poly2 = Polygon([(10, 0), (15, 0), (15, 5), (10, 5)])

        multipoly = MultiPolygon([poly1, poly2])
        result = remove_small_holes(multipoly, min_area=1.0)

        assert isinstance(result, MultiPolygon)
        assert len(result.geoms) == 2
        assert len(result.geoms[0].interiors) == 0
        assert len(result.geoms[1].interiors) == 0

    def test_exact_area_threshold(self):
        """Test behavior when hole area exactly equals threshold."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        # Hole with area exactly 4.0
        hole = [(4, 4), (6, 4), (6, 6), (4, 6)]

        poly = Polygon(exterior, holes=[hole])

        # With threshold = 4.0, hole should be kept (area >= min_area)
        result_keep = remove_small_holes(poly, min_area=4.0)
        assert len(result_keep.interiors) == 1

        # With threshold = 4.1, hole should be removed (area < min_area)
        result_remove = remove_small_holes(poly, min_area=4.1)
        assert len(result_remove.interiors) == 0

    def test_invalid_geometry_type_raises_error(self):
        """Test that non-polygon geometries raise TypeError."""
        line = LineString([(0, 0), (1, 1)])

        with pytest.raises(TypeError, match="Input geometry must be a Polygon or MultiPolygon"):
            remove_small_holes(line, min_area=1.0)

        point = Point(0, 0)
        with pytest.raises(TypeError, match="Input geometry must be a Polygon or MultiPolygon"):
            remove_small_holes(point, min_area=1.0)

    def test_zero_area_threshold(self):
        """Test with zero area threshold - all holes should be kept."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(4, 4), (5, 4), (5, 5), (4, 5)]

        poly = Polygon(exterior, holes=[hole])
        result = remove_small_holes(poly, min_area=0.0)

        # Hole should be kept
        assert len(result.interiors) == 1

    def test_very_small_hole(self):
        """Test removing very small holes (numerical precision test)."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        # Very small hole
        hole = [(5, 5), (5.001, 5), (5.001, 5.001), (5, 5.001)]

        poly = Polygon(exterior, holes=[hole])
        result = remove_small_holes(poly, min_area=0.01)

        # Tiny hole should be removed
        assert len(result.interiors) == 0


