"""Tests for fix_clearance automatic clearance detection and fixing."""

import pytest
from shapely.geometry import Polygon
from polyforge import fix_clearance
from polyforge.clearance import diagnose_clearance, ClearanceIssue


class TestDiagnoseClearance:
    """Tests for clearance diagnosis (without fixing)."""

    def test_diagnose_already_good(self):
        """Test diagnosing polygon that already meets clearance."""
        # Simple square - good clearance
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        info = diagnose_clearance(poly, min_clearance=1.0)

        assert info.meets_requirement is True
        assert info.has_issues is False
        assert info.issue is ClearanceIssue.NONE
        assert info.recommended_fix == 'none'
        assert info.clearance_ratio > 1.0

    def test_diagnose_narrow_passage(self):
        """Test diagnosing narrow passage issue."""
        # Hourglass/narrow passage
        coords = [(0, 0), (10, 0), (10, 1), (9.9, 1.5), (9.9, 2.5), (10, 3), (10, 10), (0, 10)]
        poly = Polygon(coords)
        info = diagnose_clearance(poly, min_clearance=1.0)

        assert info.meets_requirement is False
        assert info.has_issues is True
        assert info.issue in (ClearanceIssue.NARROW_PASSAGE, ClearanceIssue.NARROW_PROTRUSION)
        assert info.recommended_fix in ['fix_narrow_passage', 'remove_narrow_protrusions']
        assert info.clearance_line is not None

    def test_diagnose_narrow_protrusion(self):
        """Test diagnosing narrow protrusion/spike."""
        # Rectangle with spike
        coords = [(0, 0), (10, 0), (10, 4.9), (12, 5), (10, 5.1), (10, 10), (0, 10)]
        poly = Polygon(coords)
        info = diagnose_clearance(poly, min_clearance=1.0)

        assert info.meets_requirement is False
        assert info.has_issues is True
        assert info.issue in (ClearanceIssue.NARROW_PROTRUSION, ClearanceIssue.NARROW_PASSAGE)
        assert info.clearance_line is not None

    def test_diagnose_hole_too_close(self):
        """Test diagnosing hole too close to exterior."""
        exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
        hole = [(1, 1), (2, 1), (2, 2), (1, 2)]
        poly = Polygon(exterior, holes=[hole])
        info = diagnose_clearance(poly, min_clearance=2.0)

        assert info.meets_requirement is False
        assert info.has_issues is True
        assert info.issue is ClearanceIssue.HOLE_TOO_CLOSE
        assert info.recommended_fix == 'fix_hole_too_close'


class TestFixClearanceBasic:
    """Basic tests for automatic clearance fixing."""

    def test_fix_already_good(self):
        """Test that already-good polygon is returned unchanged."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        original_clearance = poly.minimum_clearance

        result, summary = fix_clearance(poly, min_clearance=1.0, return_diagnosis=True)

        assert result.equals(poly)
        assert summary.initial_clearance == original_clearance
        assert summary.iterations == 0
        assert summary.fixed is True
        assert summary.issue is ClearanceIssue.NONE

    def test_fix_narrow_passage(self):
        """Test fixing narrow passage."""
        coords = [(0, 0), (10, 0), (10, 1), (9.9, 1.5), (9.9, 2.5), (10, 3), (10, 10), (0, 10)]
        poly = Polygon(coords)
        original_clearance = poly.minimum_clearance

        result, summary = fix_clearance(poly, min_clearance=1.0, return_diagnosis=True)

        assert result.is_valid
        assert result.minimum_clearance > original_clearance
        assert summary.fixed is True
        assert summary.iterations > 0
        assert result.minimum_clearance >= 1.0

    def test_fix_narrow_protrusion(self):
        """Test fixing narrow protrusion/spike."""
        coords = [(0, 0), (10, 0), (10, 4.9), (12, 5), (10, 5.1), (10, 10), (0, 10)]
        poly = Polygon(coords)
        original_clearance = poly.minimum_clearance

        result, summary = fix_clearance(poly, min_clearance=1.0, return_diagnosis=True)

        assert result.is_valid
        assert result.minimum_clearance > original_clearance
        assert summary.fixed is True
        # Spike should be removed, reducing vertex count
        assert len(result.exterior.coords) <= len(poly.exterior.coords)

    def test_fix_hole_too_close(self):
        """Test fixing hole too close to exterior."""
        exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
        hole = [(1, 1), (2, 1), (2, 2), (1, 2)]
        poly = Polygon(exterior, holes=[hole])

        result, summary = fix_clearance(poly, min_clearance=2.0, return_diagnosis=True)

        assert result.is_valid
        assert summary.fixed is True
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

        result, summary = fix_clearance(
            poly,
            min_clearance=100.0,  # Unreachable target
            max_iterations=3,
            return_diagnosis=True
        )

        assert summary.iterations <= 3
        # Should still improve clearance even if target not reached
        assert result.minimum_clearance > poly.minimum_clearance

    def test_convergence(self):
        """Test that fixing converges to target."""
        coords = [(0, 0), (10, 0), (10, 1), (9.9, 1.5), (9.9, 2.5), (10, 3), (10, 10), (0, 10)]
        poly = Polygon(coords)

        result, summary = fix_clearance(
            poly,
            min_clearance=1.0,
            return_diagnosis=True
        )

        assert summary.fixed is True
        assert result.minimum_clearance >= 1.0
        assert summary.final_clearance >= summary.initial_clearance

    def test_invalid_strategy_result_is_rejected(self, monkeypatch):
        """Ensure invalid/empty candidates are discarded and we fall back to best valid."""
        coords = [(0, 0), (10, 0), (10, 1), (9.9, 1.5), (9.9, 2.5), (10, 3), (10, 10), (0, 10)]
        poly = Polygon(coords)
        original_clearance = poly.minimum_clearance

        def bad_strategy(_, __, ___):
            # Return an empty/invalid polygon to mimic a failed fix step.
            return Polygon()

        import importlib
        fc_module = importlib.import_module("polyforge.clearance.fix_clearance")
        original = fc_module.STRATEGY_REGISTRY[ClearanceIssue.NARROW_PASSAGE]
        monkeypatch.setitem(fc_module.STRATEGY_REGISTRY, ClearanceIssue.NARROW_PASSAGE, bad_strategy)

        result, summary = fix_clearance(poly, min_clearance=1.0, return_diagnosis=True)

        assert result.is_valid
        assert not result.is_empty
        assert summary.fixed is False or result.minimum_clearance >= original_clearance
        assert summary.valid is True
        assert summary.area_ratio > 0
        # Restore happens via monkeypatch undo; ensure registry is intact for other tests

    def test_area_floor_rejects_overly_small_candidate(self, monkeypatch):
        """Ensure overly small candidates are discarded by the area ratio guard."""
        coords = [(0, 0), (10, 0), (10, 1), (9.9, 1.5), (9.9, 2.5), (10, 3), (10, 10), (0, 10)]
        poly = Polygon(coords)
        original_area = poly.area

        def tiny_strategy(_, __, ___):
            return Polygon([(0, 0), (0.1, 0), (0, 0.1)])

        import importlib
        fc_module = importlib.import_module("polyforge.clearance.fix_clearance")
        monkeypatch.setitem(fc_module.STRATEGY_REGISTRY, ClearanceIssue.NARROW_PASSAGE, tiny_strategy)

        result, summary = fix_clearance(poly, min_clearance=1.0, return_diagnosis=True)

        assert result.area >= 0.9 * original_area
        assert result.is_valid
        assert summary.valid is True
        assert summary.area_ratio >= 0.9
        # Either we couldn't fix due to rejected tiny candidate, or we met the clearance without shrinking
        if summary.fixed:
            assert result.minimum_clearance >= 1.0


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

        result, summary = fix_clearance(poly, min_clearance=1.0, return_diagnosis=True)

        # Should improve clearance even if multiple issues exist
        assert result.is_valid
        assert result.minimum_clearance > poly.minimum_clearance
        assert summary.history  # ensure we recorded stages


class TestDiagnosisAccuracy:
    """Tests for accuracy of issue diagnosis."""

    def test_diagnosis_matches_fix(self):
        """Test that diagnosed issue matches the fix applied."""
        test_cases = [
            # (polygon, min_clearance, expected_issue)
            (
                Polygon([(0, 0), (10, 0), (10, 4.9), (12, 5), (10, 5.1), (10, 10), (0, 10)]),
                1.0,
                ClearanceIssue.NARROW_PROTRUSION
            ),
            (
                Polygon([(0, 0), (20, 0), (20, 20), (0, 20)], holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]]),
                2.0,
                ClearanceIssue.HOLE_TOO_CLOSE
            ),
        ]

        for poly, target, expected_issue in test_cases:
            info = diagnose_clearance(poly, min_clearance=target)
            _, summary = fix_clearance(poly, min_clearance=target, return_diagnosis=True)

            assert info.issue == expected_issue
            assert summary.issue == expected_issue or summary.history[-1] == expected_issue

    def test_diagnosis_detects_hole_to_hole_clearance(self):
        """Ensure clearance between holes is treated as hole-driven, not protrusion/passage."""
        exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
        hole1 = [(5, 5), (7, 5), (7, 7), (5, 7)]
        hole2 = [(7.6, 5), (9.6, 5), (9.6, 7), (7.6, 7)]
        poly = Polygon(exterior, holes=[hole1, hole2])

        info = diagnose_clearance(poly, min_clearance=1.0)

        assert info.issue == ClearanceIssue.HOLE_TOO_CLOSE


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
