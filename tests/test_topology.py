"""Tests for topology operations (align_boundaries)."""

import pytest
import numpy as np
from shapely.geometry import Polygon
from polyforge import align_boundaries


class TestConformBoundariesBasic:
    """Basic tests for boundary conforming."""

    def test_no_vertices_to_add(self):
        """Test when polygons already have conforming boundaries."""
        # Two squares sharing an edge, all vertices align
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = Polygon([(2, 0), (4, 0), (4, 2), (2, 2)])

        result1, result2 = align_boundaries(poly1, poly2)

        # Should be unchanged
        assert len(result1.exterior.coords) == len(poly1.exterior.coords)
        assert len(result2.exterior.coords) == len(poly2.exterior.coords)
        assert result1.is_valid
        assert result2.is_valid

    def test_single_vertex_on_edge(self):
        """Test adding single vertex to edge."""
        # poly2 has vertex at (2, 1) on poly1's right edge
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = Polygon([(2, 0), (4, 0), (4, 2), (2, 1), (2, 2)])

        result1, result2 = align_boundaries(poly1, poly2)

        # result1 should now have vertex at (2, 1)
        coords1 = list(result1.exterior.coords)
        assert len(coords1) == 6  # Original 5 + 1 new
        assert (2, 1) in coords1
        assert result1.is_valid

        # result2 should be unchanged
        assert len(result2.exterior.coords) == len(poly2.exterior.coords)

    def test_multiple_vertices_on_same_edge(self):
        """Test adding multiple vertices to the same edge."""
        # poly2 has two vertices on poly1's right edge
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = Polygon([(2, 0), (4, 0), (4, 2), (2, 1.5), (2, 0.5), (2, 2)])

        result1, result2 = align_boundaries(poly1, poly2)

        # result1 should have both vertices added
        coords1 = list(result1.exterior.coords)
        assert len(coords1) == 7  # Original 5 + 2 new
        assert (2, 0.5) in coords1
        assert (2, 1.5) in coords1

        # Check vertices are in correct order along edge
        coords1_list = [tuple(c) for c in coords1]
        idx_05 = coords1_list.index((2, 0.5))
        idx_15 = coords1_list.index((2, 1.5))
        # 0.5 should come before 1.5
        assert idx_05 < idx_15 or (idx_05 > idx_15 and idx_05 - idx_15 > 2)

        assert result1.is_valid

    def test_bidirectional_conforming(self):
        """Test when both polygons need vertices added."""
        # Both polygons have vertices on each other's edges
        # poly1 has vertex at (2, 1) on shared edge
        poly1 = Polygon([(0, 0), (2, 0), (2, 1), (2, 2), (0, 2)])
        # poly2 has vertex at (2, 0.5) on shared edge - vertices in correct order
        poly2 = Polygon([(2, 0), (2, 0.5), (2, 2), (4, 2), (4, 0)])

        result1, result2 = align_boundaries(poly1, poly2)

        # result1 gets vertex from poly2
        coords1 = list(result1.exterior.coords)
        assert (2, 0.5) in coords1

        # result2 gets vertex from poly1
        coords2 = list(result2.exterior.coords)
        assert (2, 1) in coords2

        assert result1.is_valid
        assert result2.is_valid


class TestConformBoundariesEdgeCases:
    """Tests for edge cases."""

    def test_vertex_at_edge_endpoint(self):
        """Test that vertices at endpoints are not duplicated."""
        # Vertex exactly at edge endpoint
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = Polygon([(2, 0), (4, 0), (4, 2), (2, 2)])

        result1, result2 = align_boundaries(poly1, poly2)

        # Should not add duplicate vertices
        assert len(result1.exterior.coords) == len(poly1.exterior.coords)
        assert result1.is_valid

    def test_vertex_very_close_to_edge(self):
        """Test vertex within tolerance of edge."""
        # Vertex at (2.0000001, 1) should be considered on edge
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = Polygon([(2, 0), (4, 0), (4, 2), (2.0000001, 1), (2, 2)])

        result1, result2 = align_boundaries(poly1, poly2, distance_tolerance=1e-5)

        # Should add the vertex (snapped to edge)
        assert len(result1.exterior.coords) > len(poly1.exterior.coords)
        assert result1.is_valid

    def test_non_touching_polygons(self):
        """Test with polygons that don't touch."""
        # Separated polygons
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = Polygon([(3, 0), (5, 0), (5, 2), (3, 2)])

        result1, result2 = align_boundaries(poly1, poly2)

        # Should be unchanged
        assert len(result1.exterior.coords) == len(poly1.exterior.coords)
        assert len(result2.exterior.coords) == len(poly2.exterior.coords)

    def test_diagonal_touching(self):
        """Test polygons touching at a corner."""
        # Touch at single point (2, 2)
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])

        result1, result2 = align_boundaries(poly1, poly2)

        # Should not add vertices (corner is already a vertex)
        assert result1.is_valid
        assert result2.is_valid


class TestConformBoundariesWithHoles:
    """Tests for polygons with holes."""

    def test_vertex_on_hole_boundary(self):
        """Test adding vertex to a hole boundary."""
        # Polygon with hole
        exterior1 = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole1 = [(2, 2), (8, 2), (8, 8), (2, 8)]
        poly1 = Polygon(exterior1, holes=[hole1])

        # Polygon with vertex on hole edge
        poly2 = Polygon([(1, 1), (3, 1), (3, 3), (2, 3), (1, 3)])

        result1, result2 = align_boundaries(poly1, poly2)

        # Check that poly1's hole got the vertex
        assert len(result1.interiors) == 1
        hole_coords = list(result1.interiors[0].coords)
        # Should have added vertex from poly2
        assert len(hole_coords) > 5  # Original 5 points

        assert result1.is_valid
        assert result2.is_valid

    def test_hole_vertex_on_exterior(self):
        """Test when a hole vertex lies on another polygon's exterior."""
        # Polygon with hole whose vertex lies on poly2's edge
        exterior1 = [(0, 0), (5, 0), (5, 5), (0, 5)]
        hole1 = [(1, 1), (4, 1), (4, 4), (1, 4)]
        poly1 = Polygon(exterior1, holes=[hole1])

        # Polygon touching the hole
        poly2 = Polygon([(4, 0), (6, 0), (6, 5), (4, 5)])

        result1, result2 = align_boundaries(poly1, poly2)

        assert result1.is_valid
        assert result2.is_valid


class TestConformBoundariesComplexCases:
    """Tests for complex polygon configurations."""

    def test_l_shaped_polygons(self):
        """Test with L-shaped polygons."""
        # L-shaped polygon
        poly1 = Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])

        # Rectangle touching multiple edges
        poly2 = Polygon([(1, 0), (3, 0), (3, 2), (1, 2)])

        result1, result2 = align_boundaries(poly1, poly2)

        assert result1.is_valid
        assert result2.is_valid

    def test_many_vertices_on_edge(self):
        """Test with many vertices on a single edge."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        # Polygon with many vertices along shared edge
        vertices = [(10, 0)]
        for i in range(1, 10):
            vertices.append((10, i))
        vertices.extend([(10, 10), (15, 10), (15, 0)])
        poly2 = Polygon(vertices)

        result1, result2 = align_boundaries(poly1, poly2)

        # result1 should have all vertices from poly2's left edge
        coords1 = list(result1.exterior.coords)
        assert len(coords1) >= 13  # Original 5 + 9 new vertices
        assert result1.is_valid

    def test_irregular_shapes(self):
        """Test with irregular polygon shapes."""
        # Triangle
        poly1 = Polygon([(0, 0), (4, 0), (2, 4)])

        # Pentagon touching triangle edge
        poly2 = Polygon([(2, 0), (6, 0), (7, 2), (5, 4), (2, 2)])

        result1, result2 = align_boundaries(poly1, poly2)

        assert result1.is_valid
        assert result2.is_valid


class TestVertexOrdering:
    """Tests to ensure vertices are inserted in correct order."""

    def test_vertex_order_preserved(self):
        """Test that vertices are inserted in correct order along edge."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        # Three vertices at 2, 5, 8 along right edge
        poly2 = Polygon([(10, 0), (15, 0), (15, 10), (10, 8), (10, 5), (10, 2), (10, 10)])

        result1, result2 = align_boundaries(poly1, poly2)

        # Get coordinates of result1
        coords = [tuple(c) for c in result1.exterior.coords]

        # Find positions of the three vertices
        idx_2 = coords.index((10, 2))
        idx_5 = coords.index((10, 5))
        idx_8 = coords.index((10, 8))

        # Check they're in order (accounting for ring wrapping)
        assert idx_2 < idx_5 < idx_8 or (idx_8 < idx_2 < idx_5)

    def test_winding_order_preserved(self):
        """Test that winding order (CCW/CW) is preserved."""
        # CCW polygon
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = Polygon([(2, 0), (4, 0), (4, 2), (2, 1), (2, 2)])

        result1, result2 = align_boundaries(poly1, poly2)

        # Check winding order is still CCW (positive area)
        assert result1.exterior.is_ccw == poly1.exterior.is_ccw
        assert result2.exterior.is_ccw == poly2.exterior.is_ccw


class TestPerformance:
    """Tests for performance with many vertices."""

    def test_many_vertices_performance(self):
        """Test performance doesn't degrade badly with many vertices."""
        import time

        # Create polygon with 100 vertices
        angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        coords1 = [(5 + 4 * np.cos(a), 5 + 4 * np.sin(a)) for a in angles]
        poly1 = Polygon(coords1)

        # Another polygon touching it
        coords2 = [(9, 0), (15, 0), (15, 10), (9, 10)]
        poly2 = Polygon(coords2)

        start = time.time()
        result1, result2 = align_boundaries(poly1, poly2)
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 1.0  # Less than 1 second
        assert result1.is_valid
        assert result2.is_valid


class TestResultValidity:
    """Tests to ensure results are always valid."""

    def test_all_results_valid(self):
        """Test that various configurations produce valid results."""
        test_cases = [
            # Simple squares
            (
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(2, 0), (2, 1), (2, 2), (4, 2), (4, 0)])
            ),
            # Triangles
            (
                Polygon([(0, 0), (4, 0), (2, 4)]),
                Polygon([(2, 0), (6, 0), (4, 4)])
            ),
            # L-shaped and rectangle
            (
                Polygon([(0, 0), (3, 0), (3, 2), (1, 2), (1, 3), (0, 3)]),
                Polygon([(3, 0), (3, 2), (3, 3), (5, 3), (5, 0)])
            ),
        ]

        for poly1, poly2 in test_cases:
            result1, result2 = align_boundaries(poly1, poly2)
            assert result1.is_valid, f"Invalid result1 for {poly1}"
            assert result2.is_valid, f"Invalid result2 for {poly2}"
            assert not result1.is_empty
            assert not result2.is_empty

    def test_area_preserved(self):
        """Test that polygon areas are preserved."""
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = Polygon([(2, 0), (4, 0), (4, 2), (2, 1), (2, 2)])

        original_area1 = poly1.area
        original_area2 = poly2.area

        result1, result2 = align_boundaries(poly1, poly2)

        # Areas should be identical (within floating point tolerance)
        assert abs(result1.area - original_area1) < 1e-10
        assert abs(result2.area - original_area2) < 1e-10
