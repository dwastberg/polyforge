import numpy as np
import pytest
from shapely.geometry import (
    Polygon, MultiPolygon, LinearRing, MultiLineString, LineString,
    Point, MultiPoint, GeometryCollection
)

from polyforge.process import process_geometry


# Helper functions for testing
def scale_vertices(vertices, scale_factor=1.0):
    """Scale vertices by a factor."""
    return vertices * scale_factor


def translate_vertices(vertices, dx=0.0, dy=0.0):
    """Translate vertices by dx and dy."""
    translation = np.array([dx, dy])
    return vertices + translation


def add_z_coordinate(vertices, z=0.0):
    """Add a z-coordinate to 2D vertices."""
    z_column = np.full((vertices.shape[0], 1), z)
    return np.hstack([vertices, z_column])


class TestProcessGeometryPoint:
    """Tests for Point geometry processing."""

    def test_point_scale(self):
        """Test scaling a Point geometry."""
        point = Point(2.0, 3.0)
        result = process_geometry(point, scale_vertices, scale_factor=2.0)

        assert isinstance(result, Point)
        assert result.x == pytest.approx(4.0)
        assert result.y == pytest.approx(6.0)

    def test_point_translate(self):
        """Test translating a Point geometry."""
        point = Point(1.0, 1.0)
        result = process_geometry(point, translate_vertices, dx=5.0, dy=3.0)

        assert isinstance(result, Point)
        assert result.x == pytest.approx(6.0)
        assert result.y == pytest.approx(4.0)

    def test_point_3d(self):
        """Test processing a 3D Point geometry - Z is preserved, not scaled."""
        point = Point(1.0, 2.0, 3.0)
        result = process_geometry(point, scale_vertices, scale_factor=2.0)

        assert isinstance(result, Point)
        assert result.x == pytest.approx(2.0)
        assert result.y == pytest.approx(4.0)
        assert result.z == pytest.approx(3.0)  # Z is preserved, not scaled

    def test_point_identity(self):
        """Test Point geometry with identity function."""
        point = Point(5.5, 7.3)
        result = process_geometry(point, lambda v: v)

        assert isinstance(result, Point)
        assert result.x == pytest.approx(5.5)
        assert result.y == pytest.approx(7.3)


class TestProcessGeometryLineString:
    """Tests for LineString geometry processing."""

    def test_linestring_scale(self):
        """Test scaling a LineString geometry."""
        line = LineString([(0, 0), (1, 0), (1, 1), (0, 1)])
        result = process_geometry(line, scale_vertices, scale_factor=2.0)

        assert isinstance(result, LineString)
        expected_coords = [(0, 0), (2, 0), (2, 2), (0, 2)]
        np.testing.assert_array_almost_equal(
            np.array(result.coords),
            np.array(expected_coords)
        )

    def test_linestring_translate(self):
        """Test translating a LineString geometry."""
        line = LineString([(0, 0), (1, 1), (2, 0)])
        result = process_geometry(line, translate_vertices, dx=10.0, dy=5.0)

        assert isinstance(result, LineString)
        expected_coords = [(10, 5), (11, 6), (12, 5)]
        np.testing.assert_array_almost_equal(
            np.array(result.coords),
            np.array(expected_coords)
        )

    def test_linestring_3d(self):
        """Test processing a 3D LineString geometry - Z is preserved, not scaled."""
        line = LineString([(0, 0, 0), (1, 1, 1), (2, 2, 2)])
        result = process_geometry(line, scale_vertices, scale_factor=3.0)

        assert isinstance(result, LineString)
        # XY scaled, Z preserved
        expected_coords = [(0, 0, 0), (3, 3, 1), (6, 6, 2)]
        np.testing.assert_array_almost_equal(
            np.array(result.coords),
            np.array(expected_coords)
        )

    def test_linestring_two_points(self):
        """Test LineString with minimum number of points."""
        line = LineString([(0, 0), (5, 5)])
        result = process_geometry(line, scale_vertices, scale_factor=0.5)

        assert isinstance(result, LineString)
        expected_coords = [(0, 0), (2.5, 2.5)]
        np.testing.assert_array_almost_equal(
            np.array(result.coords),
            np.array(expected_coords)
        )


class TestProcessGeometryLinearRing:
    """Tests for LinearRing geometry processing."""

    def test_linearring_scale(self):
        """Test scaling a LinearRing geometry."""
        ring = LinearRing([(0, 0), (1, 0), (1, 1), (0, 1)])
        result = process_geometry(ring, scale_vertices, scale_factor=3.0)

        assert isinstance(result, LinearRing)
        coords = np.array(result.coords)
        # LinearRing automatically closes, so first and last points should be equal
        assert coords[0][0] == pytest.approx(0.0)
        assert coords[0][1] == pytest.approx(0.0)
        assert coords[1][0] == pytest.approx(3.0)
        assert coords[1][1] == pytest.approx(0.0)

    def test_linearring_translate(self):
        """Test translating a LinearRing geometry."""
        ring = LinearRing([(0, 0), (2, 0), (2, 2), (0, 2)])
        result = process_geometry(ring, translate_vertices, dx=1.0, dy=1.0)

        assert isinstance(result, LinearRing)
        coords = np.array(result.coords)
        assert coords[0][0] == pytest.approx(1.0)
        assert coords[0][1] == pytest.approx(1.0)

    def test_linearring_closed(self):
        """Test that LinearRing remains closed after processing."""
        ring = LinearRing([(0, 0), (4, 0), (4, 4), (0, 4)])
        result = process_geometry(ring, scale_vertices, scale_factor=0.5)

        assert isinstance(result, LinearRing)
        assert result.is_closed
        coords = np.array(result.coords)
        np.testing.assert_array_almost_equal(coords[0], coords[-1])


class TestProcessGeometryPolygon:
    """Tests for Polygon geometry processing."""

    def test_polygon_no_holes_scale(self):
        """Test scaling a Polygon without holes."""
        polygon = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        result = process_geometry(polygon, scale_vertices, scale_factor=2.0)

        assert isinstance(result, Polygon)
        expected_coords = [(0, 0), (8, 0), (8, 8), (0, 8), (0, 0)]
        np.testing.assert_array_almost_equal(
            np.array(result.exterior.coords),
            np.array(expected_coords)
        )

    def test_polygon_no_holes_translate(self):
        """Test translating a Polygon without holes."""
        polygon = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        result = process_geometry(polygon, translate_vertices, dx=100.0, dy=50.0)

        assert isinstance(result, Polygon)
        coords = np.array(result.exterior.coords)
        assert coords[0][0] == pytest.approx(100.0)
        assert coords[0][1] == pytest.approx(50.0)

    def test_polygon_with_holes_scale(self):
        """Test scaling a Polygon with holes."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole1 = [(2, 2), (4, 2), (4, 4), (2, 4)]
        hole2 = [(6, 6), (8, 6), (8, 8), (6, 8)]
        polygon = Polygon(exterior, holes=[hole1, hole2])

        result = process_geometry(polygon, scale_vertices, scale_factor=2.0)

        assert isinstance(result, Polygon)
        assert len(result.interiors) == 2

        # Check exterior
        expected_exterior = [(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)]
        np.testing.assert_array_almost_equal(
            np.array(result.exterior.coords),
            np.array(expected_exterior)
        )

        # Check holes
        expected_hole1 = [(4, 4), (8, 4), (8, 8), (4, 8), (4, 4)]
        np.testing.assert_array_almost_equal(
            np.array(result.interiors[0].coords),
            np.array(expected_hole1)
        )

    def test_polygon_with_holes_translate(self):
        """Test translating a Polygon with holes."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(2, 2), (6, 2), (6, 6), (2, 6)]
        polygon = Polygon(exterior, holes=[hole])

        result = process_geometry(polygon, translate_vertices, dx=5.0, dy=5.0)

        assert isinstance(result, Polygon)
        assert len(result.interiors) == 1

        exterior_coords = np.array(result.exterior.coords)
        hole_coords = np.array(result.interiors[0].coords)

        assert exterior_coords[0][0] == pytest.approx(5.0)
        assert exterior_coords[0][1] == pytest.approx(5.0)
        assert hole_coords[0][0] == pytest.approx(7.0)
        assert hole_coords[0][1] == pytest.approx(7.0)

    def test_polygon_3d(self):
        """Test processing a 3D Polygon geometry - Z is preserved, not scaled."""
        polygon = Polygon([(0, 0, 1), (4, 0, 1), (4, 4, 1), (0, 4, 1)])
        result = process_geometry(polygon, scale_vertices, scale_factor=2.0)

        assert isinstance(result, Polygon)
        coords = np.array(result.exterior.coords)
        assert coords[0][2] == pytest.approx(1.0)  # Z preserved, not scaled


class TestProcessGeometryMultiLineString:
    """Tests for MultiLineString geometry processing."""

    def test_multilinestring_scale(self):
        """Test scaling a MultiLineString geometry."""
        line1 = [(0, 0), (1, 1)]
        line2 = [(2, 2), (3, 3)]
        multi_line = MultiLineString([line1, line2])

        result = process_geometry(multi_line, scale_vertices, scale_factor=2.0)

        assert isinstance(result, MultiLineString)
        assert len(result.geoms) == 2

        np.testing.assert_array_almost_equal(
            np.array(result.geoms[0].coords),
            np.array([(0, 0), (2, 2)])
        )
        np.testing.assert_array_almost_equal(
            np.array(result.geoms[1].coords),
            np.array([(4, 4), (6, 6)])
        )

    def test_multilinestring_translate(self):
        """Test translating a MultiLineString geometry."""
        line1 = [(0, 0), (1, 0)]
        line2 = [(0, 1), (1, 1)]
        line3 = [(0, 2), (1, 2)]
        multi_line = MultiLineString([line1, line2, line3])

        result = process_geometry(multi_line, translate_vertices, dx=10.0, dy=20.0)

        assert isinstance(result, MultiLineString)
        assert len(result.geoms) == 3

        coords0 = np.array(result.geoms[0].coords)
        assert coords0[0][0] == pytest.approx(10.0)
        assert coords0[0][1] == pytest.approx(20.0)

    def test_multilinestring_single_line(self):
        """Test MultiLineString with a single line."""
        multi_line = MultiLineString([[(0, 0), (5, 5), (10, 0)]])
        result = process_geometry(multi_line, scale_vertices, scale_factor=0.5)

        assert isinstance(result, MultiLineString)
        assert len(result.geoms) == 1


class TestProcessGeometryMultiPolygon:
    """Tests for MultiPolygon geometry processing."""

    def test_multipolygon_scale(self):
        """Test scaling a MultiPolygon geometry."""
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])
        multi_poly = MultiPolygon([poly1, poly2])

        result = process_geometry(multi_poly, scale_vertices, scale_factor=2.0)

        assert isinstance(result, MultiPolygon)
        assert len(result.geoms) == 2

        expected_coords1 = [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]
        np.testing.assert_array_almost_equal(
            np.array(result.geoms[0].exterior.coords),
            np.array(expected_coords1)
        )

    def test_multipolygon_with_holes(self):
        """Test MultiPolygon where polygons have holes."""
        exterior1 = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole1 = [(2, 2), (4, 2), (4, 4), (2, 4)]
        poly1 = Polygon(exterior1, holes=[hole1])

        exterior2 = [(20, 20), (30, 20), (30, 30), (20, 30)]
        poly2 = Polygon(exterior2)

        multi_poly = MultiPolygon([poly1, poly2])
        result = process_geometry(multi_poly, scale_vertices, scale_factor=0.5)

        assert isinstance(result, MultiPolygon)
        assert len(result.geoms) == 2
        assert len(result.geoms[0].interiors) == 1
        assert len(result.geoms[1].interiors) == 0

    def test_multipolygon_translate(self):
        """Test translating a MultiPolygon geometry."""
        poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
        multi_poly = MultiPolygon([poly1, poly2])

        result = process_geometry(multi_poly, translate_vertices, dx=100.0, dy=100.0)

        assert isinstance(result, MultiPolygon)
        coords0 = np.array(result.geoms[0].exterior.coords)
        assert coords0[0][0] == pytest.approx(100.0)
        assert coords0[0][1] == pytest.approx(100.0)


class TestProcessGeometryMultiPoint:
    """Tests for MultiPoint geometry processing."""

    def test_multipoint_not_supported(self):
        """Test that MultiPoint raises an error (not in supported list)."""
        multi_point = MultiPoint([(0, 0), (1, 1), (2, 2)])

        with pytest.raises(ValueError, match="Unsupported geometry type"):
            process_geometry(multi_point, scale_vertices, scale_factor=2.0)


class TestProcessGeometryCollection:
    """Tests for GeometryCollection processing."""

    def test_geometry_collection_mixed_types(self):
        """Test processing a GeometryCollection with mixed geometry types."""
        point = Point(0, 0)
        line = LineString([(1, 1), (2, 2)])
        poly = Polygon([(3, 3), (4, 3), (4, 4), (3, 4)])

        collection = GeometryCollection([point, line, poly])
        result = process_geometry(collection, scale_vertices, scale_factor=2.0)

        assert isinstance(result, GeometryCollection)
        assert len(result.geoms) == 3
        assert isinstance(result.geoms[0], Point)
        assert isinstance(result.geoms[1], LineString)
        assert isinstance(result.geoms[2], Polygon)

        # Check scaled values
        assert result.geoms[0].x == pytest.approx(0.0)
        assert result.geoms[0].y == pytest.approx(0.0)

        line_coords = np.array(result.geoms[1].coords)
        assert line_coords[0][0] == pytest.approx(2.0)

    def test_geometry_collection_nested(self):
        """Test processing a GeometryCollection with nested collections."""
        point = Point(1, 1)
        line = LineString([(0, 0), (1, 1)])
        inner_collection = GeometryCollection([point, line])

        outer_collection = GeometryCollection([
            inner_collection,
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
        ])

        result = process_geometry(outer_collection, scale_vertices, scale_factor=3.0)

        assert isinstance(result, GeometryCollection)
        assert len(result.geoms) == 2
        assert isinstance(result.geoms[0], GeometryCollection)
        assert isinstance(result.geoms[1], Polygon)

    def test_geometry_collection_translate(self):
        """Test translating a GeometryCollection."""
        point = Point(0, 0)
        line = LineString([(1, 1), (2, 2)])
        collection = GeometryCollection([point, line])

        result = process_geometry(collection, translate_vertices, dx=5.0, dy=10.0)

        assert isinstance(result, GeometryCollection)
        assert result.geoms[0].x == pytest.approx(5.0)
        assert result.geoms[0].y == pytest.approx(10.0)


class TestProcessGeometryEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_coordinates(self):
        """Test processing geometries with empty coordinates."""
        # Empty LineString
        empty_line = LineString()
        result = process_geometry(empty_line, scale_vertices, scale_factor=2.0)
        assert isinstance(result, LineString)
        assert result.is_empty

    def test_process_function_with_args(self):
        """Test passing positional arguments to process function."""
        def add_values(vertices, add_x, add_y):
            return vertices + np.array([add_x, add_y])

        point = Point(1, 2)
        result = process_geometry(point, add_values, 10, 20)

        assert result.x == pytest.approx(11.0)
        assert result.y == pytest.approx(22.0)

    def test_process_function_with_kwargs(self):
        """Test passing keyword arguments to process function."""
        def multiply_and_add(vertices, multiplier=1.0, offset=0.0):
            return vertices * multiplier + offset

        line = LineString([(1, 1), (2, 2)])
        result = process_geometry(line, multiply_and_add, multiplier=3.0, offset=5.0)

        coords = np.array(result.coords)
        assert coords[0][0] == pytest.approx(8.0)  # 1*3 + 5
        assert coords[0][1] == pytest.approx(8.0)

    def test_process_function_mixed_args_kwargs(self):
        """Test passing both args and kwargs to process function."""
        def complex_transform(vertices, scale, dx=0.0, dy=0.0):
            return vertices * scale + np.array([dx, dy])

        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        result = process_geometry(polygon, complex_transform, 2.0, dx=10.0, dy=5.0)

        coords = np.array(result.exterior.coords)
        assert coords[0][0] == pytest.approx(10.0)
        assert coords[0][1] == pytest.approx(5.0)

    def test_unsupported_geometry_type(self):
        """Test that unsupported geometry types raise ValueError."""
        multi_point = MultiPoint([(0, 0), (1, 1)])

        with pytest.raises(ValueError) as exc_info:
            process_geometry(multi_point, scale_vertices, scale_factor=2.0)

        assert "Unsupported geometry type" in str(exc_info.value)
        assert "MultiPoint" in str(exc_info.value)

    def test_identity_transformation(self):
        """Test that identity transformation preserves geometry."""
        polygon = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        result = process_geometry(polygon, lambda v: v)

        assert isinstance(result, Polygon)
        np.testing.assert_array_almost_equal(
            np.array(polygon.exterior.coords),
            np.array(result.exterior.coords)
        )

    def test_negative_scaling(self):
        """Test negative scaling (reflection)."""
        line = LineString([(1, 1), (2, 2), (3, 1)])
        result = process_geometry(line, scale_vertices, scale_factor=-1.0)

        expected_coords = [(-1, -1), (-2, -2), (-3, -1)]
        np.testing.assert_array_almost_equal(
            np.array(result.coords),
            np.array(expected_coords)
        )

    def test_zero_scaling(self):
        """Test zero scaling (collapse to origin)."""
        polygon = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
        result = process_geometry(polygon, scale_vertices, scale_factor=0.0)

        coords = np.array(result.exterior.coords)
        # All coordinates should be at origin
        assert np.allclose(coords, 0.0)

    def test_very_large_coordinates(self):
        """Test processing with very large coordinate values."""
        point = Point(1e10, 1e10)
        result = process_geometry(point, scale_vertices, scale_factor=2.0)

        assert result.x == pytest.approx(2e10)
        assert result.y == pytest.approx(2e10)

    def test_very_small_coordinates(self):
        """Test processing with very small coordinate values."""
        point = Point(1e-10, 1e-10)
        result = process_geometry(point, scale_vertices, scale_factor=2.0)

        assert result.x == pytest.approx(2e-10)
        assert result.y == pytest.approx(2e-10)


class TestProcessGeometryZComponents:
    """Tests for Z-component handling in all geometry types."""

    def test_point_3d_z_preserved(self):
        """Test that Point Z coordinate is preserved during processing."""
        point = Point(1.0, 2.0, 5.0)
        result = process_geometry(point, scale_vertices, scale_factor=2.0)

        assert isinstance(result, Point)
        assert result.x == pytest.approx(2.0)
        assert result.y == pytest.approx(4.0)
        assert result.z == pytest.approx(5.0)  # Z should be unchanged
        assert result.has_z

    def test_point_3d_translate_z_preserved(self):
        """Test Point 3D with translation preserves Z."""
        point = Point(0.0, 0.0, 100.0)
        result = process_geometry(point, translate_vertices, dx=10.0, dy=20.0)

        assert result.x == pytest.approx(10.0)
        assert result.y == pytest.approx(20.0)
        assert result.z == pytest.approx(100.0)
        assert result.has_z

    def test_linestring_3d_z_preserved(self):
        """Test that LineString Z coordinates are preserved."""
        line = LineString([(0, 0, 10), (1, 1, 20), (2, 2, 30)])
        result = process_geometry(line, scale_vertices, scale_factor=3.0)

        assert isinstance(result, LineString)
        assert result.has_z

        coords = np.array(result.coords)
        expected = np.array([(0, 0, 10), (3, 3, 20), (6, 6, 30)])
        np.testing.assert_array_almost_equal(coords, expected)

    def test_linestring_3d_different_z_values(self):
        """Test LineString with different Z values for each vertex."""
        line = LineString([(0, 0, 5), (1, 0, 10), (2, 0, 15), (3, 0, 20)])
        result = process_geometry(line, scale_vertices, scale_factor=2.0)

        coords = np.array(result.coords)
        # X values should be doubled, Y unchanged, Z preserved
        assert coords[0][0] == pytest.approx(0.0)
        assert coords[0][2] == pytest.approx(5.0)
        assert coords[1][0] == pytest.approx(2.0)
        assert coords[1][2] == pytest.approx(10.0)
        assert coords[2][0] == pytest.approx(4.0)
        assert coords[2][2] == pytest.approx(15.0)
        assert coords[3][0] == pytest.approx(6.0)
        assert coords[3][2] == pytest.approx(20.0)

    def test_linearring_3d_z_preserved(self):
        """Test that LinearRing Z coordinates are preserved."""
        ring = LinearRing([(0, 0, 50), (4, 0, 50), (4, 4, 50), (0, 4, 50)])
        result = process_geometry(ring, scale_vertices, scale_factor=2.0)

        assert isinstance(result, LinearRing)
        assert result.has_z

        coords = np.array(result.coords)
        # All Z values should remain 50
        assert np.all(coords[:, 2] == pytest.approx(50.0))

    def test_polygon_3d_no_holes_z_preserved(self):
        """Test Polygon without holes preserves Z coordinates."""
        polygon = Polygon([(0, 0, 100), (4, 0, 100), (4, 4, 100), (0, 4, 100)])
        result = process_geometry(polygon, scale_vertices, scale_factor=2.0)

        assert isinstance(result, Polygon)
        assert result.has_z

        coords = np.array(result.exterior.coords)
        # Check that XY are scaled but Z is preserved
        assert coords[0][0] == pytest.approx(0.0)
        assert coords[0][1] == pytest.approx(0.0)
        assert coords[0][2] == pytest.approx(100.0)
        assert coords[1][0] == pytest.approx(8.0)
        assert coords[1][2] == pytest.approx(100.0)

    def test_polygon_3d_with_holes_z_preserved(self):
        """Test Polygon with holes preserves Z coordinates in both exterior and holes."""
        exterior = [(0, 0, 50), (10, 0, 50), (10, 10, 50), (0, 10, 50)]
        hole1 = [(2, 2, 60), (4, 2, 60), (4, 4, 60), (2, 4, 60)]
        hole2 = [(6, 6, 70), (8, 6, 70), (8, 8, 70), (6, 8, 70)]
        polygon = Polygon(exterior, holes=[hole1, hole2])

        result = process_geometry(polygon, scale_vertices, scale_factor=0.5)

        assert isinstance(result, Polygon)
        assert result.has_z
        assert len(result.interiors) == 2

        # Check exterior Z values
        ext_coords = np.array(result.exterior.coords)
        assert np.all(ext_coords[:, 2] == pytest.approx(50.0))

        # Check hole Z values
        hole1_coords = np.array(result.interiors[0].coords)
        assert np.all(hole1_coords[:, 2] == pytest.approx(60.0))

        hole2_coords = np.array(result.interiors[1].coords)
        assert np.all(hole2_coords[:, 2] == pytest.approx(70.0))

        # Verify XY were actually scaled
        assert ext_coords[1][0] == pytest.approx(5.0)  # 10 * 0.5

    def test_polygon_3d_varying_z_in_exterior(self):
        """Test Polygon where exterior vertices have different Z values."""
        exterior = [(0, 0, 10), (4, 0, 20), (4, 4, 30), (0, 4, 40)]
        polygon = Polygon(exterior)

        result = process_geometry(polygon, translate_vertices, dx=1.0, dy=1.0)

        assert result.has_z
        coords = np.array(result.exterior.coords)

        # Check that each vertex kept its Z value
        assert coords[0][2] == pytest.approx(10.0)
        assert coords[1][2] == pytest.approx(20.0)
        assert coords[2][2] == pytest.approx(30.0)
        assert coords[3][2] == pytest.approx(40.0)

    def test_multilinestring_3d_z_preserved(self):
        """Test MultiLineString preserves Z coordinates."""
        line1 = [(0, 0, 5), (1, 1, 5)]
        line2 = [(2, 2, 10), (3, 3, 10)]
        line3 = [(4, 4, 15), (5, 5, 15)]
        multi_line = MultiLineString([line1, line2, line3])

        result = process_geometry(multi_line, scale_vertices, scale_factor=2.0)

        assert isinstance(result, MultiLineString)
        assert len(result.geoms) == 3

        # Check each line preserved its Z
        for i, (original_z, geom) in enumerate([(5, result.geoms[0]),
                                                  (10, result.geoms[1]),
                                                  (15, result.geoms[2])]):
            coords = np.array(geom.coords)
            assert geom.has_z
            assert np.all(coords[:, 2] == pytest.approx(original_z))

    def test_multipolygon_3d_z_preserved(self):
        """Test MultiPolygon preserves Z coordinates."""
        poly1 = Polygon([(0, 0, 100), (2, 0, 100), (2, 2, 100), (0, 2, 100)])
        poly2 = Polygon([(3, 3, 200), (5, 3, 200), (5, 5, 200), (3, 5, 200)])
        multi_poly = MultiPolygon([poly1, poly2])

        result = process_geometry(multi_poly, scale_vertices, scale_factor=3.0)

        assert isinstance(result, MultiPolygon)
        assert len(result.geoms) == 2

        # Check first polygon
        coords1 = np.array(result.geoms[0].exterior.coords)
        assert result.geoms[0].has_z
        assert np.all(coords1[:, 2] == pytest.approx(100.0))

        # Check second polygon
        coords2 = np.array(result.geoms[1].exterior.coords)
        assert result.geoms[1].has_z
        assert np.all(coords2[:, 2] == pytest.approx(200.0))

    def test_multipolygon_3d_with_holes_z_preserved(self):
        """Test MultiPolygon with holes preserves all Z coordinates."""
        exterior1 = [(0, 0, 50), (10, 0, 50), (10, 10, 50), (0, 10, 50)]
        hole1 = [(2, 2, 55), (4, 2, 55), (4, 4, 55), (2, 4, 55)]
        poly1 = Polygon(exterior1, holes=[hole1])

        poly2 = Polygon([(20, 20, 80), (30, 20, 80), (30, 30, 80), (20, 30, 80)])

        multi_poly = MultiPolygon([poly1, poly2])
        result = process_geometry(multi_poly, translate_vertices, dx=5.0, dy=5.0)

        assert isinstance(result, MultiPolygon)

        # Check poly1 exterior and hole
        ext1_coords = np.array(result.geoms[0].exterior.coords)
        assert np.all(ext1_coords[:, 2] == pytest.approx(50.0))

        hole1_coords = np.array(result.geoms[0].interiors[0].coords)
        assert np.all(hole1_coords[:, 2] == pytest.approx(55.0))

        # Check poly2
        ext2_coords = np.array(result.geoms[1].exterior.coords)
        assert np.all(ext2_coords[:, 2] == pytest.approx(80.0))

    def test_geometry_collection_3d_mixed_z_preserved(self):
        """Test GeometryCollection with 3D geometries preserves Z."""
        point = Point(0, 0, 10)
        line = LineString([(1, 1, 20), (2, 2, 30)])
        poly = Polygon([(3, 3, 40), (4, 3, 40), (4, 4, 40), (3, 4, 40)])

        collection = GeometryCollection([point, line, poly])
        result = process_geometry(collection, scale_vertices, scale_factor=2.0)

        assert isinstance(result, GeometryCollection)
        assert len(result.geoms) == 3

        # Check Point
        assert result.geoms[0].has_z
        assert result.geoms[0].z == pytest.approx(10.0)

        # Check LineString
        assert result.geoms[1].has_z
        line_coords = np.array(result.geoms[1].coords)
        assert line_coords[0][2] == pytest.approx(20.0)
        assert line_coords[1][2] == pytest.approx(30.0)

        # Check Polygon
        assert result.geoms[2].has_z
        poly_coords = np.array(result.geoms[2].exterior.coords)
        assert np.all(poly_coords[:, 2] == pytest.approx(40.0))

    def test_geometry_collection_3d_nested_z_preserved(self):
        """Test nested GeometryCollection preserves Z coordinates."""
        point = Point(1, 1, 100)
        line = LineString([(0, 0, 200), (1, 1, 200)])
        inner_collection = GeometryCollection([point, line])

        poly = Polygon([(2, 2, 300), (3, 2, 300), (3, 3, 300), (2, 3, 300)])
        outer_collection = GeometryCollection([inner_collection, poly])

        result = process_geometry(outer_collection, scale_vertices, scale_factor=0.5)

        assert isinstance(result, GeometryCollection)

        # Check nested collection
        inner = result.geoms[0]
        assert inner.geoms[0].z == pytest.approx(100.0)
        inner_line_coords = np.array(inner.geoms[1].coords)
        assert np.all(inner_line_coords[:, 2] == pytest.approx(200.0))

        # Check polygon
        poly_coords = np.array(result.geoms[1].exterior.coords)
        assert np.all(poly_coords[:, 2] == pytest.approx(300.0))

    def test_2d_geometry_unchanged_behavior(self):
        """Verify 2D geometries still work correctly (no Z added)."""
        point = Point(5.0, 10.0)
        result = process_geometry(point, scale_vertices, scale_factor=2.0)

        assert not result.has_z
        assert result.x == pytest.approx(10.0)
        assert result.y == pytest.approx(20.0)

    def test_process_function_only_receives_2d(self):
        """Verify process_function always receives Nx2 arrays even with 3D input."""
        received_shapes = []

        def capture_shape(vertices):
            received_shapes.append(vertices.shape)
            return vertices

        # Test with 3D geometries
        Point3D = Point(1, 2, 3)
        process_geometry(Point3D, capture_shape)

        line_3d = LineString([(0, 0, 10), (1, 1, 20)])
        process_geometry(line_3d, capture_shape)

        poly_3d = Polygon([(0, 0, 5), (1, 0, 5), (1, 1, 5), (0, 1, 5)])
        process_geometry(poly_3d, capture_shape)

        # All should be 2D (Nx2)
        for shape in received_shapes:
            assert shape[1] == 2, f"Expected Nx2 array, got {shape}"
