import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, Point, MultiPolygon

from polyforge.simplify import remove_small_holes


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

    def test_polygon_validity_preserved(self):
        """Test that resulting polygon is valid."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole1 = [(2, 2), (3, 2), (3, 3), (2, 3)]
        hole2 = [(6, 6), (8, 6), (8, 8), (6, 8)]

        poly = Polygon(exterior, holes=[hole1, hole2])
        result = remove_small_holes(poly, min_area=1.5)

        # Result should be valid
        assert result.is_valid
        # Area should increase when holes are filled
        assert result.area > poly.area  # Area increased due to filled holes
