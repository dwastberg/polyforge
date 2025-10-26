"""Tests for vertex insertion optimization in merge_close_polygons.

Tests the insert_vertices parameter that intelligently adds vertices at
optimal connection points before merging.
"""

import pytest
import numpy as np
from shapely.geometry import Polygon
from polyforge import merge_close_polygons
from polyforge.merge import _insert_connection_vertices


class TestVertexInsertion:
    """Test the _insert_connection_vertices helper function."""

    def test_basic_vertex_insertion(self):
        """Test that vertices are inserted at closest points on edges."""
        # Two rectangles with gap
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(12, 0), (22, 0), (22, 10), (12, 10)])

        result = _insert_connection_vertices([poly1, poly2], margin=3.0)

        assert len(result) == 2

        # Check that new vertices were added
        # Original has 5 coords (4 vertices + closing), modified should have more
        assert len(result[0].exterior.coords) >= 5
        assert len(result[1].exterior.coords) >= 5

        # Verify polygons are still valid
        assert result[0].is_valid
        assert result[1].is_valid

        # Area should be unchanged (only added vertices, not changed shape)
        assert abs(result[0].area - poly1.area) < 1e-6
        assert abs(result[1].area - poly2.area) < 1e-6

    def test_skip_insertion_near_existing_vertex(self):
        """Test that vertices are not inserted if closest point is near existing vertex."""
        # Polygons where closest points are at corners
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(10.005, 0), (20, 0), (20, 10), (10.005, 10)])  # Very close to vertex

        result = _insert_connection_vertices([poly1, poly2], margin=1.0, tolerance=0.01)

        # Should not insert vertices (closest point is within 0.01 of existing vertex)
        assert len(result[0].exterior.coords) == len(poly1.exterior.coords)
        assert len(result[1].exterior.coords) == len(poly2.exterior.coords)

    def test_preserve_holes(self):
        """Test that interior holes are preserved during vertex insertion."""
        # Polygon with a hole
        exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
        hole = [(5, 5), (15, 5), (15, 15), (5, 15)]
        poly1 = Polygon(exterior, [hole])
        poly2 = Polygon([(22, 0), (32, 0), (32, 20), (22, 20)])

        result = _insert_connection_vertices([poly1, poly2], margin=3.0)

        assert len(result) == 2
        # First polygon should still have its hole
        assert len(list(result[0].interiors)) == 1
        assert result[0].is_valid

    def test_3d_coordinates_interpolation(self):
        """Test that Z-coordinates are properly interpolated."""
        # 3D polygons
        poly1 = Polygon([(0, 0, 0), (10, 0, 10), (10, 10, 10), (0, 10, 0)])
        poly2 = Polygon([(12, 0, 0), (22, 0, 10), (22, 10, 10), (12, 10, 0)])

        result = _insert_connection_vertices([poly1, poly2], margin=3.0)

        assert len(result) == 2

        # Check that result has 3D coordinates
        coords = list(result[0].exterior.coords)
        assert all(len(coord) == 3 for coord in coords)

        # Area should be unchanged (Z doesn't affect 2D area)
        assert abs(result[0].area - poly1.area) < 1e-6

    def test_no_modification_when_distant(self):
        """Test that distant polygons are not modified."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(50, 0), (60, 0), (60, 10), (50, 10)])

        result = _insert_connection_vertices([poly1, poly2], margin=5.0)

        # Polygons are too far apart, should be unchanged
        assert len(result[0].exterior.coords) == len(poly1.exterior.coords)
        assert len(result[1].exterior.coords) == len(poly2.exterior.coords)

    def test_single_polygon_unchanged(self):
        """Test that single polygon is returned unchanged."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        result = _insert_connection_vertices([poly], margin=5.0)

        assert len(result) == 1
        assert len(result[0].exterior.coords) == len(poly.exterior.coords)


class TestMergeWithVertexInsertion:
    """Test merge_close_polygons with insert_vertices=True."""

    def test_selective_buffer_with_insertion(self):
        """Test that selective_buffer produces better results with vertex insertion."""
        poly1 = Polygon([(0, 5), (10, 5), (10, 15), (0, 15)])
        poly2 = Polygon([(11, 0), (21, 0), (21, 10), (11, 10)])

        # Without vertex insertion
        result_no_insert = merge_close_polygons(
            [poly1, poly2], margin=2.0, strategy='selective_buffer', insert_vertices=False
        )

        # With vertex insertion
        result_with_insert = merge_close_polygons(
            [poly1, poly2], margin=2.0, strategy='selective_buffer', insert_vertices=True
        )

        assert len(result_no_insert) == 1
        assert len(result_with_insert) == 1

        # Both should be valid
        assert result_no_insert[0].is_valid
        assert result_with_insert[0].is_valid

        # With insertion should have equal or less area (more precise bridge)
        area_no_insert = result_no_insert[0].area
        area_with_insert = result_with_insert[0].area

        # Both should be reasonable (close to original 200 + bridge)
        assert 200 < area_no_insert < 220
        assert 200 < area_with_insert < 220

    def test_convex_bridges_with_insertion(self):
        """Test convex_bridges strategy with vertex insertion."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(12, 0), (22, 0), (22, 10), (12, 10)])

        result = merge_close_polygons(
            [poly1, poly2], margin=3.0, strategy='convex_bridges', insert_vertices=True
        )

        assert len(result) == 1
        assert result[0].is_valid
        # Should merge the two polygons
        assert result[0].area > poly1.area + poly2.area

    def test_vertex_movement_with_insertion(self):
        """Test vertex_movement strategy with vertex insertion."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(12, 0), (22, 0), (22, 10), (12, 10)])

        result = merge_close_polygons(
            [poly1, poly2], margin=3.0, strategy='vertex_movement', insert_vertices=True
        )

        assert len(result) == 1
        assert result[0].is_valid

    def test_boundary_extension_with_insertion(self):
        """Test boundary_extension strategy with vertex insertion."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(12, 0), (22, 0), (22, 10), (12, 10)])

        result = merge_close_polygons(
            [poly1, poly2], margin=3.0, strategy='boundary_extension', insert_vertices=True
        )

        assert len(result) == 1
        assert result[0].is_valid

    def test_all_strategies_with_insertion(self):
        """Test that all strategies work with insert_vertices=True."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(12, 0), (22, 0), (22, 10), (12, 10)])

        strategies = ['simple_buffer', 'selective_buffer', 'vertex_movement',
                     'boundary_extension', 'convex_bridges']

        for strategy in strategies:
            result = merge_close_polygons(
                [poly1, poly2], margin=3.0, strategy=strategy, insert_vertices=True
            )

            assert len(result) == 1, f"Strategy {strategy} failed"
            assert result[0].is_valid, f"Strategy {strategy} produced invalid geometry"
            assert result[0].area > poly1.area, f"Strategy {strategy} didn't merge"

    def test_mapping_preserved_with_insertion(self):
        """Test that return_mapping works correctly with vertex insertion."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(12, 0), (22, 0), (22, 10), (12, 10)])

        result, mapping = merge_close_polygons(
            [poly1, poly2], margin=3.0, strategy='selective_buffer',
            insert_vertices=True, return_mapping=True
        )

        assert len(result) == 1
        assert len(mapping) == 1
        assert set(mapping[0]) == {0, 1}

    def test_isolated_polygons_unchanged_with_insertion(self):
        """Test that isolated polygons are not affected by vertex insertion."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(50, 0), (60, 0), (60, 10), (50, 10)])  # Far away

        result = merge_close_polygons(
            [poly1, poly2], margin=3.0, strategy='selective_buffer', insert_vertices=True
        )

        # Should remain as two separate polygons
        assert len(result) == 2
        # Areas should match originals
        areas = sorted([p.area for p in result])
        assert abs(areas[0] - 100) < 1e-6
        assert abs(areas[1] - 100) < 1e-6

    def test_three_polygons_with_insertion(self):
        """Test vertex insertion with three polygons."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(12, 0), (22, 0), (22, 10), (12, 10)])
        poly3 = Polygon([(24, 0), (34, 0), (34, 10), (24, 10)])

        result = merge_close_polygons(
            [poly1, poly2, poly3], margin=3.0, strategy='selective_buffer', insert_vertices=True
        )

        assert len(result) == 1
        assert result[0].is_valid
        # Should merge all three
        assert result[0].area > poly1.area + poly2.area + poly3.area


class TestEdgeCases:
    """Test edge cases for vertex insertion."""

    def test_empty_list(self):
        """Test with empty polygon list."""
        result = merge_close_polygons([], margin=5.0, insert_vertices=True)
        assert result == []

    def test_single_polygon(self):
        """Test with single polygon."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        result = merge_close_polygons([poly], margin=5.0, insert_vertices=True)

        assert len(result) == 1
        assert result[0].area == poly.area

    def test_overlapping_polygons_with_insertion(self):
        """Test overlapping polygons with vertex insertion."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])

        result = merge_close_polygons(
            [poly1, poly2], margin=0.0, strategy='convex_bridges', insert_vertices=True
        )

        assert len(result) == 1
        assert result[0].is_valid

        # Area should be union area
        expected = poly1.area + poly2.area - poly1.intersection(poly2).area
        assert abs(result[0].area - expected) < 0.1
