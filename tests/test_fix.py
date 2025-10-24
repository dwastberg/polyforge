"""Tests for geometry fixing functions."""

import pytest
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.validation import explain_validity
from polyforge.fix import (
    fix_geometry,
    diagnose_geometry,
    batch_fix_geometries,
    GeometryFixError
)


class TestFixGeometry:
    """Test suite for fix_geometry function."""

    def test_valid_geometry_unchanged(self):
        """Test that valid geometries are returned unchanged."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        result = fix_geometry(poly)

        assert result.is_valid
        assert result.equals(poly)

    def test_bow_tie_polygon(self):
        """Test fixing bow-tie (self-intersecting) polygon."""
        # Bow-tie shape: crosses itself
        poly = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])

        assert not poly.is_valid

        result = fix_geometry(poly)

        assert result.is_valid
        assert not result.is_empty

    def test_duplicate_vertices(self):
        """Test fixing polygon with duplicate vertices."""
        # Polygon with duplicate consecutive vertices
        poly = Polygon([
            (0, 0), (1, 0), (1, 0), (1, 1), (1, 1), (0, 1)
        ])

        result = fix_geometry(poly, strategy='auto')

        assert result.is_valid


    def test_buffer_strategy(self):
        """Test explicit buffer strategy."""
        poly = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])

        assert not poly.is_valid

        result = fix_geometry(poly, strategy='buffer')

        assert result.is_valid

    def test_simplify_strategy(self):
        """Test explicit simplify strategy."""
        # Create invalid polygon
        poly = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])

        result = fix_geometry(poly, strategy='simplify')

        assert result.is_valid

    def test_reconstruct_strategy(self):
        """Test reconstruct strategy (uses convex hull)."""
        # Self-intersecting polygon
        poly = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])

        result = fix_geometry(poly, strategy='reconstruct')

        assert result.is_valid
        # Reconstruction uses convex hull, so should be convex
        assert result.equals(result.convex_hull)

    def test_strict_strategy_fails_on_complex_issues(self):
        """Test that strict strategy fails for complex issues."""
        # Bow-tie that can't be fixed with just coordinate cleaning
        poly = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])

        with pytest.raises(GeometryFixError):
            fix_geometry(poly, strategy='strict')

    def test_auto_strategy_tries_multiple_fixes(self):
        """Test that auto strategy tries multiple approaches."""
        poly = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])

        result = fix_geometry(poly, strategy='auto', verbose=False)

        assert result.is_valid

    def test_multipolygon_with_invalid_parts(self):
        """Test fixing MultiPolygon with some invalid parts."""
        valid_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        invalid_poly = Polygon([(2, 0), (4, 2), (4, 0), (2, 2)])

        multi = MultiPolygon([valid_poly, invalid_poly])

        result = fix_geometry(multi)

        assert result.is_valid

    def test_invalid_strategy_raises_error(self):
        """Test that invalid strategy raises ValueError."""
        # Use invalid polygon so it actually tries to fix it
        poly = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])

        with pytest.raises(ValueError, match="Unknown strategy"):
            fix_geometry(poly, strategy='nonexistent')

    def test_verbose_mode(self):
        """Test that verbose mode runs without errors."""
        poly = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])

        # Should not raise, and should print diagnostic info
        result = fix_geometry(poly, verbose=True)

        assert result.is_valid

    def test_custom_buffer_distance(self):
        """Test using custom buffer distance."""
        poly = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])

        result = fix_geometry(poly, strategy='buffer', buffer_distance=0.01)

        assert result.is_valid

    def test_custom_tolerance(self):
        """Test using custom tolerance."""
        # Polygon with very small duplicates
        poly = Polygon([
            (0, 0), (1, 0), (1.00001, 0.00001), (1, 1), (0, 1)
        ])

        result = fix_geometry(poly, tolerance=0.001)

        assert result.is_valid

    def test_complex_self_intersection(self):
        """Test fixing complex self-intersecting polygon."""
        # Star shape with self-intersections
        poly = Polygon([
            (0, 0), (2, 1), (4, 0), (3, 2), (4, 4),
            (2, 3), (0, 4), (1, 2)
        ])

        if not poly.is_valid:
            result = fix_geometry(poly)
            assert result.is_valid


class TestDiagnoseGeometry:
    """Test suite for diagnose_geometry function."""

    def test_diagnose_valid_geometry(self):
        """Test diagnosing valid geometry."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        diagnosis = diagnose_geometry(poly)

        assert diagnosis['is_valid'] is True
        assert diagnosis['geometry_type'] == 'Polygon'
        assert not diagnosis['is_empty']
        assert diagnosis['area'] == 1.0

    def test_diagnose_self_intersection(self):
        """Test diagnosing self-intersecting polygon."""
        poly = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])

        diagnosis = diagnose_geometry(poly)

        assert diagnosis['is_valid'] is False
        assert 'Self-intersection' in diagnosis['issues']
        assert any('buffer' in s.lower() for s in diagnosis['suggestions'])

    def test_diagnose_duplicate_vertices(self):
        """Test diagnosing duplicate vertices."""
        # Create polygon with duplicates
        poly = Polygon([
            (0, 0), (1, 0), (1, 0), (1, 1), (0, 1)
        ])

        diagnosis = diagnose_geometry(poly)

        # Check for duplicate-related issues
        assert 'Consecutive duplicate vertices' in diagnosis['issues']

    def test_diagnose_includes_all_fields(self):
        """Test that diagnosis includes all expected fields."""
        poly = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])

        diagnosis = diagnose_geometry(poly)

        assert 'is_valid' in diagnosis
        assert 'validity_message' in diagnosis
        assert 'issues' in diagnosis
        assert 'suggestions' in diagnosis
        assert 'geometry_type' in diagnosis
        assert 'is_empty' in diagnosis
        assert 'area' in diagnosis

    def test_diagnose_empty_geometry(self):
        """Test diagnosing empty geometry."""
        poly = Polygon()

        diagnosis = diagnose_geometry(poly)

        assert diagnosis['is_empty'] is True


class TestBatchFixGeometries:
    """Test suite for batch_fix_geometries function."""

    def test_batch_fix_all_valid(self):
        """Test batch fixing when all geometries are valid."""
        polys = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            Polygon([(0, 2), (1, 2), (1, 3), (0, 3)])
        ]

        fixed, failed = batch_fix_geometries(polys)

        assert len(fixed) == 3
        assert len(failed) == 0
        assert all(p.is_valid for p in fixed)

    def test_batch_fix_some_invalid(self):
        """Test batch fixing with some invalid geometries."""
        polys = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # Valid
            Polygon([(0, 0), (2, 2), (2, 0), (0, 2)]),  # Invalid (bow-tie)
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])   # Valid
        ]

        fixed, failed = batch_fix_geometries(polys)

        assert len(fixed) == 3
        assert len(failed) == 0
        assert all(p.is_valid for p in fixed)

    def test_batch_fix_skip_unfixable(self):
        """Test batch fixing with skip on error."""
        polys = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
        ]

        # Use strict strategy which might fail
        fixed, failed = batch_fix_geometries(
            polys,
            strategy='strict',
            on_error='skip'
        )

        # Should skip unfixable ones
        assert len(fixed) <= len(polys)

    def test_batch_fix_keep_on_error(self):
        """Test batch fixing with keep on error."""
        polys = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
        ]

        fixed, failed = batch_fix_geometries(
            polys,
            strategy='strict',
            on_error='keep'
        )

        # Should keep all, even if invalid
        assert len(fixed) == len(polys)

    def test_batch_fix_raise_on_error(self):
        """Test batch fixing with raise on error."""
        polys = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
        ]

        with pytest.raises(GeometryFixError):
            batch_fix_geometries(
                polys,
                strategy='strict',
                on_error='raise'
            )

    def test_batch_fix_empty_list(self):
        """Test batch fixing with empty list."""
        fixed, failed = batch_fix_geometries([])

        assert len(fixed) == 0
        assert len(failed) == 0

    def test_batch_fix_with_verbose(self):
        """Test batch fixing with verbose mode."""
        polys = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
        ]

        fixed, failed = batch_fix_geometries(polys, verbose=True)

        assert len(fixed) == 2

    def test_batch_fix_preserves_order(self):
        """Test that batch fixing preserves order."""
        polys = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            Polygon([(0, 2), (1, 2), (1, 3), (0, 3)])
        ]

        fixed, failed = batch_fix_geometries(polys)

        assert len(fixed) == 3
        # Check that they're in the same order
        for i, poly in enumerate(fixed):
            # Original and fixed should have same centroid (roughly)
            assert abs(poly.centroid.x - polys[i].centroid.x) < 1.0


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_fix_very_small_polygon(self):
        """Test fixing very small polygon."""
        poly = Polygon([
            (0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001)
        ])

        if poly.is_valid:
            result = fix_geometry(poly)
            assert result.is_valid
        else:
            result = fix_geometry(poly)
            assert result.is_valid or result.is_empty

    def test_fix_polygon_with_holes(self):
        """Test fixing polygon with holes."""
        exterior = [(0, 0), (4, 0), (4, 4), (0, 4)]
        hole = [(1, 1), (2, 1), (2, 2), (1, 2)]

        poly = Polygon(exterior, [hole])

        result = fix_geometry(poly)
        assert result.is_valid

    def test_fix_linestring(self):
        """Test fixing LineString geometry."""
        # LineString with duplicate points
        line = LineString([(0, 0), (1, 1), (1, 1), (2, 2)])

        result = fix_geometry(line)
        assert result.is_valid

    def test_fix_nearly_collinear_points(self):
        """Test fixing polygon with nearly collinear points."""
        poly = Polygon([
            (0, 0), (1, 0), (1.0001, 0.0001), (2, 0),
            (2, 2), (0, 2)
        ])

        result = fix_geometry(poly, tolerance=0.001)
        assert result.is_valid
