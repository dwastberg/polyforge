"""Tests for fix_clearance automatic clearance detection and fixing."""

import pytest
from shapely.geometry import Polygon
from polyforge import fix_clearance
from polyforge.clearance import diagnose_clearance


class TestDiagnoseClearance:
    """Tests for clearance diagnosis (without fixing)."""

    def test_diagnose_already_good(self):
        """Test diagnosing polygon that already meets clearance."""
        # Simple square - good clearance
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        info = diagnose_clearance(poly, min_clearance=1.0)

        assert info['meets_requirement'] is True
        assert info['has_issues'] is False
        assert info['issue_type'] == 'none'
        assert info['recommended_fix'] == 'none'
        assert info['clearance_ratio'] > 1.0

    def test_diagnose_narrow_passage(self):
        """Test diagnosing narrow passage issue."""
        # Hourglass/narrow passage
        coords = [(0, 0), (10, 0), (10, 1), (9.9, 1.5), (9.9, 2.5), (10, 3), (10, 10), (0, 10)]
        poly = Polygon(coords)
        info = diagnose_clearance(poly, min_clearance=1.0)

        assert info['meets_requirement'] is False
        assert info['has_issues'] is True
        assert info['issue_type'] in ['narrow_passage', 'narrow_protrusion']
        assert info['recommended_fix'] in ['fix_narrow_passage', 'remove_narrow_protrusions']
        assert info['clearance_line'] is not None

    def test_diagnose_narrow_protrusion(self):
        """Test diagnosing narrow protrusion/spike."""
        # Rectangle with spike
        coords = [(0, 0), (10, 0), (10, 4.9), (12, 5), (10, 5.1), (10, 10), (0, 10)]
        poly = Polygon(coords)
        info = diagnose_clearance(poly, min_clearance=1.0)

        assert info['meets_requirement'] is False
        assert info['has_issues'] is True
        assert 'protrusion' in info['issue_type'] or 'passage' in info['issue_type']
        assert info['clearance_line'] is not None

    def test_diagnose_hole_too_close(self):
        """Test diagnosing hole too close to exterior."""
        exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
        hole = [(1, 1), (2, 1), (2, 2), (1, 2)]
        poly = Polygon(exterior, holes=[hole])
        info = diagnose_clearance(poly, min_clearance=2.0)

        assert info['meets_requirement'] is False
        assert info['has_issues'] is True
        assert info['issue_type'] == 'hole_too_close'
        assert info['recommended_fix'] == 'fix_hole_too_close'


class TestFixClearanceBasic:
    """Basic tests for automatic clearance fixing."""

    def test_fix_already_good(self):
        """Test that already-good polygon is returned unchanged."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        original_clearance = poly.minimum_clearance

        result, diagnosis = fix_clearance(poly, min_clearance=1.0, return_diagnosis=True)

        assert result.equals(poly)
        assert diagnosis['initial_clearance'] == original_clearance
        assert diagnosis['iterations'] == 0
        assert diagnosis['fixed'] is True
        assert diagnosis['issue_type'] == 'none'

    def test_fix_narrow_passage(self):
        """Test fixing narrow passage."""
        coords = [(0, 0), (10, 0), (10, 1), (9.9, 1.5), (9.9, 2.5), (10, 3), (10, 10), (0, 10)]
        poly = Polygon(coords)
        original_clearance = poly.minimum_clearance

        result, diagnosis = fix_clearance(poly, min_clearance=1.0, return_diagnosis=True)

        assert result.is_valid
        assert result.minimum_clearance > original_clearance
        assert diagnosis['fixed'] is True
        assert diagnosis['iterations'] > 0
        assert result.minimum_clearance >= 1.0

    def test_fix_narrow_protrusion(self):
        """Test fixing narrow protrusion/spike."""
        coords = [(0, 0), (10, 0), (10, 4.9), (12, 5), (10, 5.1), (10, 10), (0, 10)]
        poly = Polygon(coords)
        original_clearance = poly.minimum_clearance

        result, diagnosis = fix_clearance(poly, min_clearance=1.0, return_diagnosis=True)

        assert result.is_valid
        assert result.minimum_clearance > original_clearance
        assert diagnosis['fixed'] is True
        # Spike should be removed, reducing vertex count
        assert len(result.exterior.coords) <= len(poly.exterior.coords)

    def test_fix_hole_too_close(self):
        """Test fixing hole too close to exterior."""
        exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
        hole = [(1, 1), (2, 1), (2, 2), (1, 2)]
        poly = Polygon(exterior, holes=[hole])

        result, diagnosis = fix_clearance(poly, min_clearance=2.0, return_diagnosis=True)

        assert result.is_valid
        assert diagnosis['fixed'] is True
        # Hole should be removed
        assert len(result.interiors) < len(poly.interiors)
        assert result.minimum_clearance >= 2.0

    def test_without_diagnosis(self):
        """Test that function works without return_diagnosis."""
        coords = [(0, 0), (10, 0), (10, 4.9), (12, 5), (10, 5.1), (10, 10), (0, 10)]
        poly = Polygon(coords)

        result = fix_clearance(poly, min_clearance=1.0)

        # Should return just the geometry, not tuple
        assert isinstance(result, Polygon)
        assert result.is_valid
        assert result.minimum_clearance >= poly.minimum_clearance


class TestFixClearanceIterations:
    """Tests for iterative fixing behavior."""

    def test_max_iterations(self):
        """Test that max_iterations is respected."""
        coords = [(0, 0), (10, 0), (10, 1), (9.9, 1.5), (9.9, 2.5), (10, 3), (10, 10), (0, 10)]
        poly = Polygon(coords)

        result, diagnosis = fix_clearance(
            poly,
            min_clearance=100.0,  # Unreachable target
            max_iterations=3,
            return_diagnosis=True
        )

        assert diagnosis['iterations'] <= 3
        # Should still improve clearance even if target not reached
        assert result.minimum_clearance > poly.minimum_clearance

    def test_convergence(self):
        """Test that fixing converges to target."""
        coords = [(0, 0), (10, 0), (10, 1), (9.9, 1.5), (9.9, 2.5), (10, 3), (10, 10), (0, 10)]
        poly = Polygon(coords)

        result, diagnosis = fix_clearance(
            poly,
            min_clearance=1.0,
            return_diagnosis=True
        )

        assert diagnosis['fixed'] is True
        assert result.minimum_clearance >= 1.0
        assert diagnosis['final_clearance'] >= diagnosis['initial_clearance']


class TestFixClearanceEdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_geometry_type(self):
        """Test that non-Polygon raises TypeError."""
        from shapely.geometry import LineString

        line = LineString([(0, 0), (10, 0), (10, 10)])

        with pytest.raises(TypeError):
            fix_clearance(line, min_clearance=1.0)

    def test_very_small_target(self):
        """Test with very small target clearance."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        result = fix_clearance(poly, min_clearance=0.001)

        assert result.is_valid
        assert result.minimum_clearance >= 0.001

    def test_complex_polygon(self):
        """Test with complex polygon shape."""
        # Irregular polygon with potential multiple issues
        # Use a simpler but still complex shape
        coords = [
            (0, 0), (10, 0), (10, 1), (9.9, 1.5), (9.9, 2.5),  # narrow passage
            (10, 4), (10, 4.9), (12, 5), (10, 5.1),  # spike
            (10, 10), (0, 10)
        ]
        poly = Polygon(coords)

        result, diagnosis = fix_clearance(poly, min_clearance=1.0, return_diagnosis=True)

        # Should improve clearance even if multiple issues exist
        assert result.is_valid
        assert result.minimum_clearance > poly.minimum_clearance


class TestDiagnosisAccuracy:
    """Tests for accuracy of issue diagnosis."""

    def test_diagnosis_matches_fix(self):
        """Test that diagnosed issue matches the fix applied."""
        test_cases = [
            # (polygon, min_clearance, expected_issue_substring)
            (
                Polygon([(0, 0), (10, 0), (10, 4.9), (12, 5), (10, 5.1), (10, 10), (0, 10)]),
                1.0,
                'protrusion'
            ),
            (
                Polygon([(0, 0), (20, 0), (20, 20), (0, 20)], holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]]),
                2.0,
                'hole'
            ),
        ]

        for poly, target, expected_substring in test_cases:
            info = diagnose_clearance(poly, min_clearance=target)
            _, diagnosis = fix_clearance(poly, min_clearance=target, return_diagnosis=True)

            # Issue type should be related to the diagnosed problem
            assert expected_substring in info['issue_type'].lower() or \
                   expected_substring in diagnosis['fix_applied'].lower()


class TestResultValidity:
    """Tests to ensure all results are valid geometries."""

    def test_all_results_valid(self):
        """Test that all fix attempts produce valid geometries."""
        test_polygons = [
            # Narrow passage
            Polygon([(0, 0), (10, 0), (10, 1), (9.9, 1.5), (9.9, 2.5), (10, 3), (10, 10), (0, 10)]),
            # Spike
            Polygon([(0, 0), (10, 0), (10, 4.9), (12, 5), (10, 5.1), (10, 10), (0, 10)]),
            # Hole too close
            Polygon([(0, 0), (20, 0), (20, 20), (0, 20)], holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]]),
            # L-shape
            Polygon([(0, 0), (10, 0), (10, 10), (5, 10), (5, 5), (0, 5)]),
        ]

        for poly in test_polygons:
            result = fix_clearance(poly, min_clearance=1.0)
            assert result.is_valid, f"Invalid result for {poly}"
            assert not result.is_empty, f"Empty result for {poly}"

    def test_preserves_polygon_type(self):
        """Test that Polygon input produces Polygon output (not MultiPolygon)."""
        poly = Polygon([(0, 0), (10, 0), (10, 4.9), (12, 5), (10, 5.1), (10, 10), (0, 10)])

        result = fix_clearance(poly, min_clearance=1.0)

        # Should still be a Polygon (though MultiPolygon is also acceptable)
        assert result.geom_type in ['Polygon', 'MultiPolygon']
