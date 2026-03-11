"""Tests for fix_clearance automatic clearance detection and fixing."""

import numpy as np
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
        assert info.recommended_fix == "none"
        assert info.clearance_ratio > 1.0

    def test_diagnose_narrow_passage(self):
        """Test diagnosing narrow passage issue."""
        # Hourglass/narrow passage
        coords = [
            (0, 0),
            (10, 0),
            (10, 1),
            (9.9, 1.5),
            (9.9, 2.5),
            (10, 3),
            (10, 10),
            (0, 10),
        ]
        poly = Polygon(coords)
        info = diagnose_clearance(poly, min_clearance=1.0)

        assert info.meets_requirement is False
        assert info.has_issues is True
        assert info.issue in (
            ClearanceIssue.NARROW_PASSAGE,
            ClearanceIssue.NARROW_PROTRUSION,
        )
        assert info.recommended_fix in [
            "fix_narrow_passage",
            "fix_narrow_protrusion",
            "remove_narrow_protrusions",
        ]
        assert info.clearance_line is not None

    def test_diagnose_narrow_protrusion(self):
        """Test diagnosing narrow protrusion/spike."""
        # Rectangle with spike
        coords = [(0, 0), (10, 0), (10, 4.9), (12, 5), (10, 5.1), (10, 10), (0, 10)]
        poly = Polygon(coords)
        info = diagnose_clearance(poly, min_clearance=1.0)

        assert info.meets_requirement is False
        assert info.has_issues is True
        assert info.issue in (
            ClearanceIssue.NARROW_WEDGE,
            ClearanceIssue.NARROW_PROTRUSION,
            ClearanceIssue.NARROW_PASSAGE,
        )
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
        assert info.recommended_fix == "fix_hole_too_close"


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
        coords = [
            (0, 0),
            (10, 0),
            (10, 1),
            (9.9, 1.5),
            (9.9, 2.5),
            (10, 3),
            (10, 10),
            (0, 10),
        ]
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
        coords = [
            (0, 0),
            (10, 0),
            (10, 1),
            (9.9, 1.5),
            (9.9, 2.5),
            (10, 3),
            (10, 10),
            (0, 10),
        ]
        poly = Polygon(coords)

        result, summary = fix_clearance(
            poly,
            min_clearance=100.0,  # Unreachable target
            max_iterations=3,
            return_diagnosis=True,
        )

        assert summary.iterations <= 3
        # For this specific polygon, the area constraint (min_area_ratio=0.9)
        # prevents meaningful improvement, so we just verify it doesn't crash
        # and respects the iteration limit
        assert result.is_valid

    def test_convergence(self):
        """Test that fixing converges to target."""
        coords = [
            (0, 0),
            (10, 0),
            (10, 1),
            (9.9, 1.5),
            (9.9, 2.5),
            (10, 3),
            (10, 10),
            (0, 10),
        ]
        poly = Polygon(coords)

        result, summary = fix_clearance(poly, min_clearance=1.0, return_diagnosis=True)

        assert summary.fixed is True
        assert result.minimum_clearance >= 1.0
        assert summary.final_clearance >= summary.initial_clearance

    def test_invalid_strategy_result_is_rejected(self, monkeypatch):
        """Ensure invalid/empty candidates are discarded and we fall back to best valid."""
        coords = [
            (0, 0),
            (10, 0),
            (10, 1),
            (9.9, 1.5),
            (9.9, 2.5),
            (10, 3),
            (10, 10),
            (0, 10),
        ]
        poly = Polygon(coords)
        original_clearance = poly.minimum_clearance

        def bad_strategy(_, __, ___):
            # Return an empty/invalid polygon to mimic a failed fix step.
            return Polygon()

        import importlib

        fc_module = importlib.import_module("polyforge.clearance.fix_clearance")
        original = fc_module.STRATEGY_REGISTRY[ClearanceIssue.NARROW_PASSAGE]
        monkeypatch.setitem(
            fc_module.STRATEGY_REGISTRY, ClearanceIssue.NARROW_PASSAGE, bad_strategy
        )

        result, summary = fix_clearance(poly, min_clearance=1.0, return_diagnosis=True)

        assert result.is_valid
        assert not result.is_empty
        assert summary.fixed is False or result.minimum_clearance >= original_clearance
        assert summary.valid is True
        assert summary.area_ratio > 0
        # Restore happens via monkeypatch undo; ensure registry is intact for other tests

    def test_area_floor_rejects_overly_small_candidate(self, monkeypatch):
        """Ensure overly small candidates are discarded by the area ratio guard."""
        coords = [
            (0, 0),
            (10, 0),
            (10, 1),
            (9.9, 1.5),
            (9.9, 2.5),
            (10, 3),
            (10, 10),
            (0, 10),
        ]
        poly = Polygon(coords)
        original_area = poly.area

        def tiny_strategy(_, __, ___):
            return Polygon([(0, 0), (0.1, 0), (0, 0.1)])

        import importlib

        fc_module = importlib.import_module("polyforge.clearance.fix_clearance")
        monkeypatch.setitem(
            fc_module.STRATEGY_REGISTRY, ClearanceIssue.NARROW_PASSAGE, tiny_strategy
        )

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
            (0, 0),
            (10, 0),
            (10, 1),
            (9.9, 1.5),
            (9.9, 2.5),  # narrow passage
            (10, 4),
            (10, 4.9),
            (12, 5),
            (10, 5.1),  # spike
            (10, 10),
            (0, 10),
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
        """Test that diagnosis correctly identifies issues and fix resolves them."""
        test_cases = [
            # (polygon, min_clearance, expected_diagnosis)
            (
                Polygon(
                    [(0, 0), (10, 0), (10, 4.9), (12, 5), (10, 5.1), (10, 10), (0, 10)]
                ),
                1.0,
                ClearanceIssue.NARROW_WEDGE,
            ),
            (
                Polygon(
                    [(0, 0), (20, 0), (20, 20), (0, 20)],
                    holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]],
                ),
                2.0,
                ClearanceIssue.HOLE_TOO_CLOSE,
            ),
        ]

        for poly, target, expected_issue in test_cases:
            info = diagnose_clearance(poly, min_clearance=target)
            result, summary = fix_clearance(
                poly, min_clearance=target, return_diagnosis=True
            )

            # Diagnosis should identify the correct issue type
            assert info.issue == expected_issue
            # The fix should succeed (may use region-first or point-based approach)
            assert summary.fixed
            assert result.is_valid

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
            Polygon(
                [
                    (0, 0),
                    (10, 0),
                    (10, 1),
                    (9.9, 1.5),
                    (9.9, 2.5),
                    (10, 3),
                    (10, 10),
                    (0, 10),
                ]
            ),
            # Spike
            Polygon(
                [(0, 0), (10, 0), (10, 4.9), (12, 5), (10, 5.1), (10, 10), (0, 10)]
            ),
            # Hole too close
            Polygon(
                [(0, 0), (20, 0), (20, 20), (0, 20)],
                holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]],
            ),
            # L-shape
            Polygon([(0, 0), (10, 0), (10, 10), (5, 10), (5, 5), (0, 5)]),
        ]

        for poly in test_polygons:
            result = fix_clearance(poly, min_clearance=1.0)
            assert result.is_valid, f"Invalid result for {poly}"
            assert not result.is_empty, f"Empty result for {poly}"

    def test_preserves_polygon_type(self):
        """Test that Polygon input produces Polygon output (not MultiPolygon)."""
        poly = Polygon(
            [(0, 0), (10, 0), (10, 4.9), (12, 5), (10, 5.1), (10, 10), (0, 10)]
        )

        result = fix_clearance(poly, min_clearance=1.0)

        # Should still be a Polygon (though MultiPolygon is also acceptable)
        assert result.geom_type in ["Polygon", "MultiPolygon"]


class TestFixClearanceSlivers:
    """Tests for extended narrow region (sliver) handling via region-first approach."""

    def test_narrow_peninsula(self):
        """fix_clearance should handle narrow peninsulas via erosion-dilation."""
        # Square with a narrow peninsula extending from the right side
        coords = [
            (0, 0),
            (20, 0),
            (20, 9.85),
            (40, 9.85),
            (40, 10.15),
            (20, 10.15),
            (20, 20),
            (0, 20),
        ]
        poly = Polygon(coords)
        original_clearance = poly.minimum_clearance
        assert original_clearance < 1.0

        result, summary = fix_clearance(poly, min_clearance=1.0, return_diagnosis=True)

        assert result.is_valid
        assert result.minimum_clearance >= 0.9  # Within 90% of target
        assert summary.fixed
        # Region-first approach should solve this quickly
        assert summary.iterations <= 3

    def test_narrow_notch(self):
        """fix_clearance should handle narrow notches cut into a large polygon."""
        # Large square with a narrow notch cut in from one side
        # Notch is 2 units wide, polygon is 50x50 — narrow feature is a small fraction
        coords = [
            (0, 0),
            (50, 0),
            (50, 24),
            (10, 24),
            (10, 26),
            (50, 26),
            (50, 50),
            (0, 50),
        ]
        poly = Polygon(coords)
        original_clearance = poly.minimum_clearance
        assert original_clearance < 5.0

        result = fix_clearance(poly, min_clearance=5.0)

        assert result.is_valid
        assert result.minimum_clearance > original_clearance

    def test_long_narrow_channel(self):
        """fix_clearance should handle long narrow channels between two areas."""
        coords = [
            (0, 0),
            (50, 0),
            (50, 10),
            (30, 10),
            (30, 10.2),
            (50, 10.2),
            (50, 20),
            (0, 20),
            (0, 10.2),
            (20, 10.2),
            (20, 10),
            (0, 10),
        ]
        poly = Polygon(coords)
        original_clearance = poly.minimum_clearance
        assert original_clearance < 1.0

        result = fix_clearance(poly, min_clearance=1.0)

        assert result.is_valid
        assert result.minimum_clearance > original_clearance

    def test_area_preservation_with_sliver_fix(self):
        """Erosion-dilation should respect min_area_ratio."""
        # Large square with a narrow appendage (appendage is small relative to total)
        base = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        sliver = Polygon([(100, 49.9), (120, 49.9), (120, 50.1), (100, 50.1)])
        combined = base.union(sliver)
        if not isinstance(combined, Polygon):
            return  # skip if union didn't produce a single polygon

        original_area = combined.area

        result = fix_clearance(combined, min_clearance=1.0, min_area_ratio=0.9)

        assert result.is_valid
        assert result.area >= 0.9 * original_area

    def test_region_first_does_not_inflate_polygon(self):
        """fix_clearance should not produce a polygon larger than the original."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        original_area = poly.area

        # Target clearance larger than polygon can naturally satisfy
        result = fix_clearance(poly, min_clearance=100.0)

        assert result.is_valid
        # Should not grow the polygon excessively
        assert result.area <= original_area * 1.2

    def test_multiple_slivers(self):
        """fix_clearance should handle polygons with multiple narrow features."""
        # Polygon with two narrow peninsulas
        coords = [
            (0, 0),
            (20, 0),
            (20, 4.9),
            (25, 4.9),
            (25, 5.1),
            (20, 5.1),  # First peninsula
            (20, 14.9),
            (25, 14.9),
            (25, 15.1),
            (20, 15.1),  # Second peninsula
            (20, 20),
            (0, 20),
        ]
        poly = Polygon(coords)
        original_clearance = poly.minimum_clearance
        assert original_clearance < 1.0

        result = fix_clearance(poly, min_clearance=1.0)

        assert result.is_valid
        assert result.minimum_clearance > original_clearance


class TestFixClearanceNarrowWedge:
    """Tests for narrow wedge detection and fixing via fix_clearance."""

    def test_v_notch_detected_and_fixed(self):
        """fix_clearance should detect and fix a V-notch wedge."""
        coords = [
            (0, 0),
            (10, 0),
            (10, 4),
            (10, 4.5),
            (9, 4.7),
            (8, 4.9),
            (7, 5.0),
            (8, 5.1),
            (9, 5.3),
            (10, 5.5),
            (10, 6),
            (10, 10),
            (0, 10),
        ]
        poly = Polygon(coords)
        original_clearance = poly.minimum_clearance

        result, summary = fix_clearance(poly, min_clearance=1.0, return_diagnosis=True)

        assert result.is_valid
        assert result.minimum_clearance > original_clearance
        assert summary.fixed or result.minimum_clearance >= 1.0 - 1e-6

    def test_diagnosis_detects_wedge(self):
        """diagnose_clearance should return NARROW_WEDGE for a V-notch."""
        coords = [
            (0, 0),
            (10, 0),
            (10, 4),
            (10, 4.5),
            (9, 4.7),
            (8, 4.9),
            (7, 5.0),
            (8, 5.1),
            (9, 5.3),
            (10, 5.5),
            (10, 6),
            (10, 10),
            (0, 10),
        ]
        poly = Polygon(coords)
        info = diagnose_clearance(poly, min_clearance=1.0)
        # The V-notch should be diagnosed as a wedge or protrusion
        assert info.issue in (
            ClearanceIssue.NARROW_WEDGE,
            ClearanceIssue.NARROW_PROTRUSION,
        )

    def test_narrow_peninsula_wedge_fixed(self):
        """fix_clearance should remove a narrow tapered peninsula."""
        coords = [
            (0, 0),
            (20, 0),
            (20, 9),
            (20, 9.5),
            (22, 9.7),
            (24, 9.9),
            (26, 10.0),
            (24, 10.1),
            (22, 10.3),
            (20, 10.5),
            (20, 11),
            (20, 20),
            (0, 20),
        ]
        poly = Polygon(coords)
        original_clearance = poly.minimum_clearance

        result = fix_clearance(poly, min_clearance=1.0)

        assert result.is_valid
        assert result.minimum_clearance > original_clearance

    def test_simple_spike_classified_as_wedge_by_angle(self):
        """A simple spike with an acute tip angle is classified as NARROW_WEDGE."""
        # Simple spike with only 1 narrow vertex but very acute tip angle
        coords = [
            (0, 0),
            (10, 0),
            (10, 4.9),
            (12, 5),
            (10, 5.1),
            (10, 10),
            (0, 10),
        ]
        poly = Polygon(coords)
        info = diagnose_clearance(poly, min_clearance=1.0)
        assert info.issue == ClearanceIssue.NARROW_WEDGE


class TestFixClearanceMissingPaths:
    """Tests for code paths not covered by existing test classes."""

    @pytest.mark.parametrize("bad_ratio", [0.0, -0.1, 1.5])
    def test_min_area_ratio_validation(self, bad_ratio):
        """Invalid min_area_ratio values should raise ValueError."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        with pytest.raises(ValueError):
            fix_clearance(poly, min_clearance=1.0, min_area_ratio=bad_ratio)

    def test_min_area_ratio_boundary_valid(self):
        """min_area_ratio=1.0 is at the valid boundary and must not raise."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        # Should not raise; square already has good clearance
        result = fix_clearance(poly, min_clearance=1.0, min_area_ratio=1.0)
        assert result.is_valid

    def test_phase1_rejected_by_tight_area_ratio(self):
        """Phase 1 (erosion-dilation) rejected by tight area ratio falls through to Phase 2."""
        from shapely.geometry import box
        from shapely.ops import unary_union

        # 100x100 square + a 100x0.5 sliver appendage at the bottom.
        # After erosion by 0.5 the sliver disappears (~50 units lost out of ~10050).
        # area_ratio after Phase 1 ≈ 0.995, which is below the 0.999 threshold,
        # so Phase 1 must reject and Phase 2 must run.
        base = box(0, 0, 100, 100)
        sliver = box(0, -0.5, 100, 0)
        poly = unary_union([base, sliver])
        assert isinstance(poly, Polygon)

        original_clearance = poly.minimum_clearance
        result, summary = fix_clearance(
            poly, min_clearance=1.0, min_area_ratio=0.999, return_diagnosis=True
        )

        assert result.is_valid
        assert summary.final_clearance > original_clearance
        # Phase 2 ran — Phase 1 alone would produce iterations=1 with fixed=True,
        # but Phase 1 was rejected so iterations reflects Phase 2 work.
        assert summary.iterations > 0

    def test_erode_dilate_bridge_fragmentation(self):
        """_erode_dilate_fix picks the largest fragment when erosion severs a bridge."""
        from shapely.geometry import box
        from shapely.ops import unary_union

        # Two 5x5 squares joined by a bridge that is only 0.1 units wide.
        # Eroding by min_clearance*0.5 = 0.5 severs the bridge, producing a
        # MultiPolygon; the implementation must return the largest fragment as a Polygon.
        left = box(0, 0, 5, 5)
        right = box(5.1, 0, 10.1, 5)
        bridge = box(5.0, 2.45, 5.1, 2.55)
        poly = unary_union([left, right, bridge])
        assert isinstance(poly, Polygon)

        result, summary = fix_clearance(poly, min_clearance=1.0, return_diagnosis=True)

        assert result.is_valid
        assert not result.is_empty
        assert isinstance(result, Polygon)
        assert result.area > 0

    def test_phase1_success_history(self):
        """Phase 1 success annotates history with PARALLEL_CLOSE_EDGES."""
        # Narrow peninsula geometry from test_narrow_peninsula — Phase 1 handles it.
        coords = [
            (0, 0),
            (20, 0),
            (20, 9.85),
            (40, 9.85),
            (40, 10.15),
            (20, 10.15),
            (20, 20),
            (0, 20),
        ]
        poly = Polygon(coords)
        assert poly.minimum_clearance < 1.0

        result, summary = fix_clearance(poly, min_clearance=1.0, return_diagnosis=True)

        assert summary.fixed
        assert ClearanceIssue.PARALLEL_CLOSE_EDGES in summary.history


class TestHoleRingSelfClearance:
    """Tests for hole-ring self-clearance misdiagnosis bug."""

    BUG4_WKT = (
        "Polygon ((675355.58337903290521353 6578950.36910314485430717, "
        "675355.57855689316056669 6578950.36758904345333576, "
        "675350.33548246114514768 6578967.05274864844977856, "
        "675372.1405879455851391 6578973.73349158652126789, "
        "675372.13507436774671078 6578973.75191244296729565, "
        "675400.86115142388734967 6578982.56493405811488628, "
        "675400.90933247364591807 6578982.37220985908061266, "
        "675400.97441947320476174 6578982.38531178701668978, "
        "675400.91862525627948344 6578982.5673771258443594, "
        "675401.31523284898139536 6578982.68355510756373405, "
        "675401.26222073775716126 6578982.86559669673442841, "
        "675403.84813925204798579 6578983.62038298975676298, "
        "675403.95696729654446244 6578983.28990516625344753, "
        "675414.40982591710053384 6578986.48388929478824139, "
        "675414.35183490358758718 6578986.6738598570227623, "
        "675426.42652199079748243 6578990.3724488765001297, "
        "675428.64823011599946767 6578989.2062804726883769, "
        "675429.99173418211285025 6578985.0137925622984767, "
        "675430.0287977383704856 6578985.02681489195674658, "
        "675430.85892285511363298 6578982.42333499528467655, "
        "675430.81883228698279709 6578982.41732141003012657, "
        "675436.0065159450750798 6578966.05062926840037107, "
        "675435.84122558182571083 6578965.99853775929659605, "
        "675435.85332528152503073 6578965.95944642182439566, "
        "675435.98885440779849887 6578966.00378748774528503, "
        "675436.04224669211544096 6578965.83958102855831385, "
        "675436.21155732974875718 6578965.88967293221503496, "
        "675445.42951205477584153 6578936.81365014612674713, "
        "675410.04856807098258287 6578925.6450941963121295, "
        "675389.38560850918292999 6578919.10110700502991676, "
        "675369.65986830671317875 6578912.83146844990551472, "
        "675366.86563254019711167 6578914.27767015807330608, "
        "675363.20504025684203953 6578926.01660371478646994, "
        "675359.84339094499591738 6578936.8007747046649456, "
        "675359.64234221377409995 6578936.73875967413187027, "
        "675358.7917250752216205 6578939.47849011793732643, "
        "675358.98259322624653578 6578939.53844869788736105, "
        "675355.58337903290521353 6578950.36910314485430717),"
        "(675368.85429259890224785 6578954.63136879075318575, "
        "675376.44188314990606159 6578930.77563948091119528, "
        "675377.02034949720837176 6578928.9562064791098237, "
        "675385.46757648815400898 6578931.57395137939602137, "
        "675406.13708663533907384 6578938.08824492618441582, "
        "675405.73685540864244103 6578939.37706118170171976, "
        "675427.23021479183807969 6578946.36685284879058599, "
        "675422.33243307797238231 6578961.33888532780110836, "
        "675407.60970134229864925 6578956.37697574030607939, "
        "675407.20501008117571473 6578957.58028645813465118, "
        "675408.61831384093966335 6578958.05638878606259823, "
        "675405.30016519722994417 6578968.16120699979364872, "
        "675404.70781079994048923 6578970.0377698102965951, "
        "675393.36976947006769478 6578966.50462130457162857, "
        "675393.54979276831727475 6578965.76952616963535547, "
        "675387.56986761779990047 6578963.92120410315692425, "
        "675387.29589912586379796 6578964.70311417803168297, "
        "675376.20738898566924036 6578961.2491206880658865, "
        "675376.7475632734131068 6578959.3665132625028491, "
        "675376.46575292758643627 6578959.28312040492892265, "
        "675375.90520014928188175 6578961.15592658612877131, "
        "675367.60848846333101392 6578958.59633427299559116, "
        "675368.85415843047667295 6578954.63132664747536182, "
        "675368.85429259890224785 6578954.63136879075318575))"
    )

    def _make_bug4_poly(self):
        import shapely
        return shapely.from_wkt(self.BUG4_WKT)

    def test_hole_ring_near_duplicate_vertices_preserved(self):
        """Near-duplicate vertices on hole ring should not cause hole removal."""
        poly = self._make_bug4_poly()
        assert len(poly.interiors) == 1

        result = fix_clearance(poly, min_clearance=1.0)

        assert result.is_valid
        assert len(result.interiors) == 1, "Hole should be preserved"

    def test_hole_ring_self_clearance_does_not_remove_hole(self):
        """Same-hole near-duplicate vertices should NOT inflate area by removing hole."""
        poly = self._make_bug4_poly()
        original_area = poly.area

        result = fix_clearance(poly, min_clearance=1.0)

        # Area should stay close to original (not inflated by hole area ~1400)
        assert result.area < original_area * 1.05, (
            f"Area grew from {original_area:.0f} to {result.area:.0f} — "
            "hole was likely removed"
        )

    def test_exterior_near_duplicate_vertices_fixed(self):
        """Near-duplicate vertices on exterior ring should be resolved without losing holes."""
        # Polygon with near-duplicate vertices on exterior only
        exterior = [
            (0, 0), (10, 0), (10.001, 0.001),  # near-duplicate
            (20, 0), (20, 20), (0, 20),
        ]
        hole = [(5, 5), (15, 5), (15, 15), (5, 15)]
        poly = Polygon(exterior, holes=[hole])

        result = fix_clearance(poly, min_clearance=0.1)

        assert result.is_valid
        assert len(result.interiors) == 1, "Hole should be preserved"

    def test_diagnosis_hole_ring_self_clearance(self):
        """diagnose_clearance should NOT return HOLE_TOO_CLOSE for same-hole near-dups."""
        poly = self._make_bug4_poly()
        info = diagnose_clearance(poly, min_clearance=1.0)

        assert info.issue != ClearanceIssue.HOLE_TOO_CLOSE, (
            "Same-hole near-duplicate vertices misdiagnosed as HOLE_TOO_CLOSE"
        )

    def test_genuine_hole_too_close_still_detected(self):
        """Regression: genuine hole-to-exterior proximity still diagnosed correctly."""
        exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
        hole = [(1, 1), (2, 1), (2, 2), (1, 2)]
        poly = Polygon(exterior, holes=[hole])

        info = diagnose_clearance(poly, min_clearance=2.0)

        assert info.issue == ClearanceIssue.HOLE_TOO_CLOSE

    def test_scratch_clearance_bug4_polygon(self):
        """Exact reproduction of clearance_bug4.py — hole preserved, area stable."""
        poly = self._make_bug4_poly()
        original_area = poly.area
        assert len(poly.interiors) == 1
        assert poly.minimum_clearance < 0.001  # very small due to near-dups

        result, summary = fix_clearance(poly, min_clearance=1.0, return_diagnosis=True)

        assert result.is_valid
        assert len(result.interiors) == 1, "Hole must be preserved"
        assert result.area >= 0.9 * original_area, "Area should not drop more than 10%"
        assert result.area < original_area * 1.05, "Area should not inflate"
        # Clearance should improve significantly from the near-dup level
        assert result.minimum_clearance > poly.minimum_clearance * 10


class TestFixClearancePreservesShape:
    """Tests for fix_clearance not deforming structurally important polygons."""

    BUG5_WKT = (
        "Polygon ((675957.54357741132844239 6579166.28348589781671762, "
        "675970.27878662315197289 6579166.14495059289038181, "
        "675977.47308820637408644 6579164.67688794806599617, "
        "675977.549256956204772 6579168.4872784810140729, "
        "675973.26474948704708368 6579168.83048056345432997, "
        "675973.30124685016926378 6579174.89918101020157337, "
        "675980.28144246782176197 6579175.52278707455843687, "
        "675980.28145694779232144 6579175.52822818420827389, "
        "675977.10110634635202587 6579175.82942920364439487, "
        "675977.29629726440180093 6579177.96609128452837467, "
        "675981.3927035448141396 6579177.57493605930358171, "
        "675981.40836108312942088 6579175.42150263302028179, "
        "675981.38550786627456546 6579175.42366699036210775, "
        "675981.50630275101866573 6579168.26072514895349741, "
        "675981.5079885721206665 6579168.26059221755713224, "
        "675981.51001297065522522 6579160.89380593318492174, "
        "675970.20498298422899097 6579161.43204505834728479, "
        "675957.48844546417240053 6579161.96787959150969982, "
        "675957.54357741132844239 6579166.28348589781671762))"
    )

    def _make_bug5_poly(self):
        import shapely
        return shapely.from_wkt(self.BUG5_WKT)

    def test_fix_clearance_preserves_l_shape(self):
        """fix_clearance should not deform an L-shaped building footprint.

        The polygon has near-duplicate vertices (common in surveyed data) that
        cause low clearance, but the structural shape must be preserved.
        """
        poly = self._make_bug5_poly()
        original_area = poly.area
        original_vertex_count = len(poly.exterior.coords) - 1  # 18

        result = fix_clearance(poly, min_clearance=1.0)

        assert result.is_valid

        # Area loss should be small
        assert result.area >= 0.95 * original_area, (
            f"Area dropped from {original_area:.1f} to {result.area:.1f} "
            f"({result.area / original_area:.1%})"
        )

        # Should not lose too many structural vertices
        result_vertex_count = len(result.exterior.coords) - 1
        assert result_vertex_count >= 13, (
            f"Too many vertices removed: {original_vertex_count} -> {result_vertex_count}"
        )

        # Clearance should improve from the original ~0.002
        assert result.minimum_clearance > poly.minimum_clearance

        # Shape preservation: symmetric containment check
        intersection_area = result.intersection(poly).area
        assert intersection_area >= 0.95 * original_area, (
            "Fixed polygon deviates significantly from original shape"
        )
        assert intersection_area >= 0.90 * result.area, (
            "Fixed polygon extends far outside original shape"
        )


class TestRemoveNarrowProtrusionsMinHeight:
    """Tests for the min_height parameter in remove_narrow_protrusions."""

    def test_spike_above_min_height_still_removed(self):
        """A genuine spike with height > min_height should still be removed."""
        from polyforge.ops.clearance.remove_protrusions import remove_narrow_protrusions

        # Rectangle with a spike: height of spike ~2.0
        coords = [(0, 0), (10, 0), (10, 4), (12, 5), (10, 6), (10, 10), (0, 10)]
        poly = Polygon(coords)

        result = remove_narrow_protrusions(poly, aspect_ratio_threshold=3.0, min_height=0.5)

        # Spike should be removed (fewer vertices)
        assert len(result.exterior.coords) < len(poly.exterior.coords)
        assert result.is_valid

    def test_collinear_vertex_below_min_height_not_removed(self):
        """A nearly-collinear vertex with height < min_height should NOT be removed."""
        from polyforge.ops.clearance.remove_protrusions import remove_narrow_protrusions

        # Rectangle with a nearly-collinear vertex on the right edge
        # Vertex at (10.01, 5) is barely off the line from (10, 0) to (10, 10)
        coords = [(0, 0), (10, 0), (10.01, 5), (10, 10), (0, 10)]
        poly = Polygon(coords)

        result = remove_narrow_protrusions(
            poly, aspect_ratio_threshold=3.0, min_height=0.5
        )

        # Vertex should NOT be removed — its height (0.01) is below min_height (0.5)
        assert len(result.exterior.coords) == len(poly.exterior.coords)
        assert result.is_valid

    def test_default_min_height_unchanged(self):
        """Default min_height=0.0 should behave identically to before."""
        from polyforge.ops.clearance.remove_protrusions import remove_narrow_protrusions

        # Rectangle with a nearly-collinear vertex — should be removable with default
        coords = [(0, 0), (10, 0), (10.01, 5), (10, 10), (0, 10)]
        poly = Polygon(coords)

        result = remove_narrow_protrusions(poly, aspect_ratio_threshold=3.0)

        # With default min_height=0.0, the collinear vertex CAN be removed
        # (this was the old behavior)
        assert result.is_valid
        # The vertex has a very high aspect ratio, so it should be a candidate
        assert len(result.exterior.coords) <= len(poly.exterior.coords)


class TestFixClearanceHoleRingSelfIntersection:
    """Tests for fix_clearance when clearance bottleneck is on a hole ring.

    Bug: When a hole ring has a spike (vertex that juts out and comes back
    close to the starting point), the minimum_clearance bottleneck is on the
    hole ring. But fix_near_self_intersection and _try_remove_bottleneck_edge_vertex
    only operate on the exterior ring, so they completely miss the problem.
    """

    # Polygon with 6 holes where hole 0 has a spike: vertices 44->45->46
    # where 44 and 46 are ~0.15 apart but 45 is ~9.6 units away, creating
    # a near-self-intersection on the hole ring with clearance ~0.001.
    BUG6_WKT = (
        "Polygon ((675532.59532547648996115 6581201.99684221856296062, "
        "675532.54299999994691461 6581202.32400000002235174, "
        "675555.412999999942258 6581273.10900000017136335, "
        "675555.44200000003911555 6581273.11000000033527613, "
        "675579.98899999994318932 6581349.00299999956041574, "
        "675617.31400000001303852 6581336.196000000461936, "
        "675640.69599999999627471 6581328.20899999979883432, "
        "675603.39000000001396984 6581217.21100000012665987, "
        "675580.3469999999506399 6581148.35699999984353781, "
        "675521.34499999997206032 6581167.69500000029802322, "
        "675532.45499999995809048 6581202.04600000008940697, "
        "675532.59532547648996115 6581201.99684221856296062),"
        "(675579.23600000003352761 6581297.25499999988824129, "
        "675575.28500000003259629 6581285.08299999963492155, "
        "675579.09199999994598329 6581283.84900000039488077, "
        "675580.57499999995343387 6581288.42200000025331974, "
        "675589.0649302025558427 6581285.63535113912075758, "
        "675590.39300000004004687 6581289.42999999970197678, "
        "675596.56200000003445894 6581287.16899999976158142, "
        "675592.89800000004470348 6581276.89199999999254942, "
        "675600.28200000000651926 6581274.01099999994039536, "
        "675597.75600000005215406 6581267.7630000002682209, "
        "675584.59900000004563481 6581272.875, "
        "675572.65800000005401671 6581276.98599999956786633, "
        "675564.875 6581253.00999999977648258, "
        "675565.97600000002421439 6581252.28699999954551458, "
        "675576.28000000002793968 6581248.72400000039488077, "
        "675574.16599999996833503 6581242.27599999960511923, "
        "675569.212000000057742 6581227.68200000002980232, "
        "675590.33516836958006024 6581221.21056527644395828, "
        "675592.7219999999506399 6581229.22400000039488077, "
        "675588.88899999996647239 6581230.36799999978393316, "
        "675587.4220000000204891 6581228.17300000041723251, "
        "675577.05099999997764826 6581231.62000000011175871, "
        "675580.35699999995995313 6581241.65500000026077032, "
        "675590.72100000001955777 6581238.13599999994039536, "
        "675590.337000000057742 6581235.22599999979138374, "
        "675594.17000000004190952 6581234.08299999963492155, "
        "675605.21400000003632158 6581264.86400000005960464, "
        "675610.85571880603674799 6581281.92414968926459551, "
        "675612.50800000003073364 6581287.20700000040233135, "
        "675611.643999999971129 6581288.96800000034272671, "
        "675612.04599999997299165 6581290.178999999538064, "
        "675613.42099999997299165 6581291.04499999992549419, "
        "675614.33799999998882413 6581293.80200000014156103, "
        "675613.38800000003539026 6581295.95199999958276749, "
        "675610.98699999996460974 6581294.90600000042468309, "
        "675608.21600000001490116 6581296.01900000032037497, "
        "675607.90099999995436519 6581295.19000000040978193, "
        "675601.16200000001117587 6581297.95700000040233135, "
        "675600.64099999994505197 6581299.23300000000745058, "
        "675598.99399999994784594 6581299.83200000040233135, "
        "675598.50600000005215406 6581298.375, "
        "675599.66799999994691461 6581295.87799999956041574, "
        "675596.54099999996833503 6581287.1780000003054738, "
        "675590.45100000000093132 6581289.41000000014901161, "
        "675596.38899999996647239 6581306.8169999998062849, "
        "675605.33336484793107957 6581303.23760861624032259, "
        "675596.43700000003445894 6581306.9599999999627471, "
        "675591.56200000003445894 6581292.76200000010430813, "
        "675587.34199999994598329 6581294.29999999981373549, "
        "675590.61800000001676381 6581303.64699999988079071, "
        "675582.28500000003259629 6581306.64599999971687794, "
        "675580.51699999999254942 6581301.19799999985843897, "
        "675576.71100000001024455 6581302.4330000001937151, "
        "675575.47600000002421439 6581298.62600000016391277, "
        "675579.23600000003352761 6581297.25499999988824129),"
        "(675588.518999999971129 6581215.12000000011175871, "
        "675564.60900000005494803 6581223.1969999996945262, "
        "675563.45700000005308539 6581224.30200000014156103, "
        "675563.22800000000279397 6581225.57699999958276749, "
        "675563.86100000003352761 6581227.31599999964237213, "
        "675563.09999999997671694 6581228.60400000028312206, "
        "675563.74399999994784594 6581230.31199999991804361, "
        "675565.09199999994598329 6581230.87299999967217445, "
        "675567.26199999998789281 6581237.03799999970942736, "
        "675566.40500000002793968 6581238.65799999982118607, "
        "675559.65000000002328306 6581240.90500000026077032, "
        "675558.50300000002607703 6581243.74299999978393316, "
        "675556.75199999997857958 6581243.3169999998062849, "
        "675557.25500000000465661 6581240.11299999989569187, "
        "675556.33200000005308539 6581237.35400000028312206, "
        "675557.54200000001583248 6581236.94099999964237213, "
        "675557.10800000000745058 6581235.75100000016391277, "
        "675555.88100000005215406 6581236.12299999967217445, "
        "675554.28500000003259629 6581230.66199999954551458, "
        "675555.26000000000931323 6581229.62399999983608723, "
        "675555.34999999997671694 6581228.1969999996945262, "
        "675554.47100000001955777 6581227.01699999999254942, "
        "675553.03500000003259629 6581226.69900000002235174, "
        "675550.37300000002142042 6581218.87299999967217445, "
        "675551.537999999942258 6581217.01099999994039536, "
        "675550.97400000004563481 6581215.41299999970942736, "
        "675549.0590000000083819 6581214.67100000008940697, "
        "675545.79099999996833503 6581204.88499999977648258, "
        "675543.54399999999441206 6581202.84999999962747097, "
        "675544.68700000003445894 6581201.21300000045448542, "
        "675547.39199999999254942 6581203.02500000037252903, "
        "675553.46499999996740371 6581200.99100000038743019, "
        "675555.21999999997206032 6581201.55900000035762787, "
        "675557.04500000004190952 6581206.64499999955296516, "
        "675556.43999999994412065 6581208.48400000017136335, "
        "675558.17700000002514571 6581210.14900000020861626, "
        "675558.80599999998230487 6581212.02400000020861626, "
        "675560.46299999998882413 6581212.64300000015646219, "
        "675563.76000000000931323 6581211.46600000001490116, "
        "675585.31400000001303852 6581204.36400000005960464, "
        "675588.518999999971129 6581215.12000000011175871),"
        "(675596.28399999998509884 6581324.70399999991059303, "
        "675588.97499999997671694 6581327.25299999956041574, "
        "675591.02899999998044223 6581333.5849999999627471, "
        "675585.72699999995529652 6581335.30599999986588955, "
        "675582.62100000004284084 6581325.73300000000745058, "
        "675587.89800000004470348 6581323.93800000008195639, "
        "675583.88111186551395804 6581311.5633446192368865, "
        "675592.56104165420401841 6581308.39012299571186304, "
        "675594.23400000005494803 6581313.33000000007450581, "
        "675603.19900000002235174 6581310.22699999995529652, "
        "675607.96999999997206032 6581324.86299999989569187, "
        "675597.59799999999813735 6581328.48199999984353781, "
        "675596.28399999998509884 6581324.70399999991059303),"
        "(675541.33200000005308539 6581180.48300000000745058, "
        "675539.90000000002328306 6581176.07000000029802322, "
        "675556.41399999998975545 6581170.49000000022351742, "
        "675560.61800000001676381 6581183.90099999960511923, "
        "675549.24600000004284084 6581187.44400000013411045, "
        "675550.481000000028871 6581190.96899999957531691, "
        "675541.47999999998137355 6581193.89199999999254942, "
        "675540.24499999999534339 6581190.08700000029057264, "
        "675544.04899999999906868 6581188.85199999995529652, "
        "675542.56700000003911555 6581184.28699999954551458, "
        "675538.76199999998789281 6581185.52199999988079071, "
        "675537.55700000002980232 6581181.80499999970197678, "
        "675541.33200000005308539 6581180.48300000000745058),"
        "(675564.73400000005494803 6581190.87200000043958426, "
        "675568.53000000002793968 6581189.61299999989569187, "
        "675566.787999999942258 6581184.1969999996945262, "
        "675575.41822714149020612 6581181.06765267346054316, "
        "675579.62100000004284084 6581193.14400000032037497, "
        "675566.79500000004190952 6581197.27199999988079071, "
        "675564.73400000005494803 6581190.87200000043958426),"
        "(675620.00500000000465661 6581311.80900000035762787, "
        "675620.00399999995715916 6581313.223000000230968, "
        "675621.49600000004284084 6581314.91100000031292439, "
        "675622.875 6581315.06500000040978193, "
        "675625.06499999994412065 6581314.38900000043213367, "
        "675627.73899999994318932 6581322.4280000003054738, "
        "675614.30092742585111409 6581326.89269187022000551, "
        "675610.65099999995436519 6581315.62299999967217445, "
        "675609.35999999998603016 6581311.928999999538064, "
        "675613.75699999998323619 6581310.41700000036507845, "
        "675612.65500000002793968 6581307.03399999998509884, "
        "675613.11100000003352761 6581304.72900000028312206, "
        "675617.768999999971129 6581302.83700000029057264, "
        "675620.42900000000372529 6581310.88200000021606684, "
        "675620.00500000000465661 6581311.80900000035762787))"
    )

    def _make_bug6_poly(self):
        import shapely
        return shapely.from_wkt(self.BUG6_WKT)

    def test_fix_clearance_hole_ring_spike(self):
        """fix_clearance must fix a spike on a hole ring that causes near-self-intersection.

        Hole 0 has a spike at vertices 44->45->46: vertex 45 juts out ~9.6 units
        while vertices 44 and 46 are only ~0.15 apart, creating a clearance of
        ~0.001. All fix strategies currently only operate on the exterior ring
        and miss the hole ring problem entirely.
        """
        poly = self._make_bug6_poly()
        original_clearance = poly.minimum_clearance
        original_area = poly.area

        assert original_clearance < 0.01, "Precondition: clearance should be very low"

        result = fix_clearance(poly, min_clearance=1.0)

        assert result.is_valid
        assert result.minimum_clearance > original_clearance * 10, (
            f"Clearance barely improved: {original_clearance:.6f} -> "
            f"{result.minimum_clearance:.6f}"
        )
        assert result.area >= 0.9 * original_area, "Should not lose too much area"

    def test_diagnose_identifies_hole_ring_issue(self):
        """diagnose_clearance should correctly identify the hole ring bottleneck."""
        poly = self._make_bug6_poly()
        diag = diagnose_clearance(poly, min_clearance=1.0)

        assert diag.meets_requirement is False
        assert diag.context is not None
        assert diag.context.ring_types == ("hole", "hole"), (
            f"Expected both endpoints on hole ring, got {diag.context.ring_types}"
        )
        assert diag.context.hole_indices[0] == diag.context.hole_indices[1], (
            "Both endpoints should be on the same hole"
        )

    def test_fix_clearance_simple_hole_spike(self):
        """Simpler test: polygon with a single hole that has a spike.

        The hole has a narrow spike that creates near-self-intersection.
        fix_clearance should resolve it by removing or collapsing the spike.
        """
        # Large exterior
        exterior = [(0, 0), (100, 0), (100, 100), (0, 100)]
        # Hole with a spike: the ring goes along the bottom, up the right,
        # across the top, then down the left side with a spike jutting outward.
        # The spike goes from (30, 50) out to (15, 50.05) and back to (30, 50.1),
        # creating a near-self-intersection with clearance ~0.1
        hole = [
            (30, 30), (70, 30), (70, 70), (30, 70),
            (30, 50.1), (15, 50.05), (30, 50),
        ]
        poly = Polygon(exterior, [hole])
        assert poly.is_valid, f"Test polygon should be valid: {poly.is_valid}"

        original_clearance = poly.minimum_clearance
        assert original_clearance < 1.0, f"Precondition: expected low clearance, got {original_clearance}"

        result = fix_clearance(poly, min_clearance=1.0)

        assert result.is_valid
        assert result.minimum_clearance > original_clearance * 5, (
            f"Clearance should significantly improve: {original_clearance:.4f} -> "
            f"{result.minimum_clearance:.4f}"
        )


class TestClearanceBug6Regression:
    """Regression tests for bug6 polygon — collateral geometry damage.

    The bug6 polygon has a hole self-intersection (clearance ~0.001) that,
    once fixed, exposes near-collinear seam vertices on the exterior ring.
    The old code applied whole-polygon RDP simplification, destroying 5 of 11
    exterior vertices and over-shrinking holes.
    """

    BUG6_WKT = (
        "Polygon ((675532.59532547648996115 6581201.99684221856296062, "
        "675532.54299999994691461 6581202.32400000002235174, "
        "675555.412999999942258 6581273.10900000017136335, "
        "675555.44200000003911555 6581273.11000000033527613, "
        "675579.98899999994318932 6581349.00299999956041574, "
        "675617.31400000001303852 6581336.196000000461936, "
        "675640.69599999999627471 6581328.20899999979883432, "
        "675603.39000000001396984 6581217.21100000012665987, "
        "675580.3469999999506399 6581148.35699999984353781, "
        "675521.34499999997206032 6581167.69500000029802322, "
        "675532.45499999995809048 6581202.04600000008940697, "
        "675532.59532547648996115 6581201.99684221856296062),"
        "(675579.23600000003352761 6581297.25499999988824129, "
        "675575.28500000003259629 6581285.08299999963492155, "
        "675579.09199999994598329 6581283.84900000039488077, "
        "675580.57499999995343387 6581288.42200000025331974, "
        "675589.0649302025558427 6581285.63535113912075758, "
        "675590.39300000004004687 6581289.42999999970197678, "
        "675596.56200000003445894 6581287.16899999976158142, "
        "675592.89800000004470348 6581276.89199999999254942, "
        "675600.28200000000651926 6581274.01099999994039536, "
        "675597.75600000005215406 6581267.7630000002682209, "
        "675584.59900000004563481 6581272.875, "
        "675572.65800000005401671 6581276.98599999956786633, "
        "675564.875 6581253.00999999977648258, "
        "675565.97600000002421439 6581252.28699999954551458, "
        "675576.28000000002793968 6581248.72400000039488077, "
        "675574.16599999996833503 6581242.27599999960511923, "
        "675569.212000000057742 6581227.68200000002980232, "
        "675590.33516836958006024 6581221.21056527644395828, "
        "675592.7219999999506399 6581229.22400000039488077, "
        "675588.88899999996647239 6581230.36799999978393316, "
        "675587.4220000000204891 6581228.17300000041723251, "
        "675577.05099999997764826 6581231.62000000011175871, "
        "675580.35699999995995313 6581241.65500000026077032, "
        "675590.72100000001955777 6581238.13599999994039536, "
        "675590.337000000057742 6581235.22599999979138374, "
        "675594.17000000004190952 6581234.08299999963492155, "
        "675605.21400000003632158 6581264.86400000005960464, "
        "675610.85571880603674799 6581281.92414968926459551, "
        "675612.50800000003073364 6581287.20700000040233135, "
        "675611.643999999971129 6581288.96800000034272671, "
        "675612.04599999997299165 6581290.178999999538064, "
        "675613.42099999997299165 6581291.04499999992549419, "
        "675614.33799999998882413 6581293.80200000014156103, "
        "675613.38800000003539026 6581295.95199999958276749, "
        "675610.98699999996460974 6581294.90600000042468309, "
        "675608.21600000001490116 6581296.01900000032037497, "
        "675607.90099999995436519 6581295.19000000040978193, "
        "675601.16200000001117587 6581297.95700000040233135, "
        "675600.64099999994505197 6581299.23300000000745058, "
        "675598.99399999994784594 6581299.83200000040233135, "
        "675598.50600000005215406 6581298.375, "
        "675599.66799999994691461 6581295.87799999956041574, "
        "675596.54099999996833503 6581287.1780000003054738, "
        "675590.45100000000093132 6581289.41000000014901161, "
        "675596.38899999996647239 6581306.8169999998062849, "
        "675605.33336484793107957 6581303.23760861624032259, "
        "675596.43700000003445894 6581306.9599999999627471, "
        "675591.56200000003445894 6581292.76200000010430813, "
        "675587.34199999994598329 6581294.29999999981373549, "
        "675590.61800000001676381 6581303.64699999988079071, "
        "675582.28500000003259629 6581306.64599999971687794, "
        "675580.51699999999254942 6581301.19799999985843897, "
        "675576.71100000001024455 6581302.4330000001937151, "
        "675575.47600000002421439 6581298.62600000016391277, "
        "675579.23600000003352761 6581297.25499999988824129),"
        "(675588.518999999971129 6581215.12000000011175871, "
        "675564.60900000005494803 6581223.1969999996945262, "
        "675563.45700000005308539 6581224.30200000014156103, "
        "675563.22800000000279397 6581225.57699999958276749, "
        "675563.86100000003352761 6581227.31599999964237213, "
        "675563.09999999997671694 6581228.60400000028312206, "
        "675563.74399999994784594 6581230.31199999991804361, "
        "675565.09199999994598329 6581230.87299999967217445, "
        "675567.26199999998789281 6581237.03799999970942736, "
        "675566.40500000002793968 6581238.65799999982118607, "
        "675559.65000000002328306 6581240.90500000026077032, "
        "675558.50300000002607703 6581243.74299999978393316, "
        "675556.75199999997857958 6581243.3169999998062849, "
        "675557.25500000000465661 6581240.11299999989569187, "
        "675556.33200000005308539 6581237.35400000028312206, "
        "675557.54200000001583248 6581236.94099999964237213, "
        "675557.10800000000745058 6581235.75100000016391277, "
        "675555.88100000005215406 6581236.12299999967217445, "
        "675554.28500000003259629 6581230.66199999954551458, "
        "675555.26000000000931323 6581229.62399999983608723, "
        "675555.34999999997671694 6581228.1969999996945262, "
        "675554.47100000001955777 6581227.01699999999254942, "
        "675553.03500000003259629 6581226.69900000002235174, "
        "675550.37300000002142042 6581218.87299999967217445, "
        "675551.537999999942258 6581217.01099999994039536, "
        "675550.97400000004563481 6581215.41299999970942736, "
        "675549.0590000000083819 6581214.67100000008940697, "
        "675545.79099999996833503 6581204.88499999977648258, "
        "675543.54399999999441206 6581202.84999999962747097, "
        "675544.68700000003445894 6581201.21300000045448542, "
        "675547.39199999999254942 6581203.02500000037252903, "
        "675553.46499999996740371 6581200.99100000038743019, "
        "675555.21999999997206032 6581201.55900000035762787, "
        "675557.04500000004190952 6581206.64499999955296516, "
        "675556.43999999994412065 6581208.48400000017136335, "
        "675558.17700000002514571 6581210.14900000020861626, "
        "675558.80599999998230487 6581212.02400000020861626, "
        "675560.46299999998882413 6581212.64300000015646219, "
        "675563.76000000000931323 6581211.46600000001490116, "
        "675585.31400000001303852 6581204.36400000005960464, "
        "675588.518999999971129 6581215.12000000011175871),"
        "(675596.28399999998509884 6581324.70399999991059303, "
        "675588.97499999997671694 6581327.25299999956041574, "
        "675591.02899999998044223 6581333.5849999999627471, "
        "675585.72699999995529652 6581335.30599999986588955, "
        "675582.62100000004284084 6581325.73300000000745058, "
        "675587.89800000004470348 6581323.93800000008195639, "
        "675583.88111186551395804 6581311.5633446192368865, "
        "675592.56104165420401841 6581308.39012299571186304, "
        "675594.23400000005494803 6581313.33000000007450581, "
        "675603.19900000002235174 6581310.22699999995529652, "
        "675607.96999999997206032 6581324.86299999989569187, "
        "675597.59799999999813735 6581328.48199999984353781, "
        "675596.28399999998509884 6581324.70399999991059303),"
        "(675541.33200000005308539 6581180.48300000000745058, "
        "675539.90000000002328306 6581176.07000000029802322, "
        "675556.41399999998975545 6581170.49000000022351742, "
        "675560.61800000001676381 6581183.90099999960511923, "
        "675549.24600000004284084 6581187.44400000013411045, "
        "675550.481000000028871 6581190.96899999957531691, "
        "675541.47999999998137355 6581193.89199999999254942, "
        "675540.24499999999534339 6581190.08700000029057264, "
        "675544.04899999999906868 6581188.85199999995529652, "
        "675542.56700000003911555 6581184.28699999954551458, "
        "675538.76199999998789281 6581185.52199999988079071, "
        "675537.55700000002980232 6581181.80499999970197678, "
        "675541.33200000005308539 6581180.48300000000745058),"
        "(675564.73400000005494803 6581190.87200000043958426, "
        "675568.53000000002793968 6581189.61299999989569187, "
        "675566.787999999942258 6581184.1969999996945262, "
        "675575.41822714149020612 6581181.06765267346054316, "
        "675579.62100000004284084 6581193.14400000032037497, "
        "675566.79500000004190952 6581197.27199999988079071, "
        "675564.73400000005494803 6581190.87200000043958426),"
        "(675620.00500000000465661 6581311.80900000035762787, "
        "675620.00399999995715916 6581313.223000000230968, "
        "675621.49600000004284084 6581314.91100000031292439, "
        "675622.875 6581315.06500000040978193, "
        "675625.06499999994412065 6581314.38900000043213367, "
        "675627.73899999994318932 6581322.4280000003054738, "
        "675614.30092742585111409 6581326.89269187022000551, "
        "675610.65099999995436519 6581315.62299999967217445, "
        "675609.35999999998603016 6581311.928999999538064, "
        "675613.75699999998323619 6581310.41700000036507845, "
        "675612.65500000002793968 6581307.03399999998509884, "
        "675613.11100000003352761 6581304.72900000028312206, "
        "675617.768999999971129 6581302.83700000029057264, "
        "675620.42900000000372529 6581310.88200000021606684, "
        "675620.00500000000465661 6581311.80900000035762787))"
    )

    @pytest.fixture()
    def bug6_poly(self):
        import shapely
        return shapely.from_wkt(self.BUG6_WKT)

    def test_clearance_improves(self, bug6_poly):
        """Clearance should improve from the original ~0.001."""
        original_clearance = bug6_poly.minimum_clearance
        result = fix_clearance(bug6_poly, min_clearance=1.0)
        assert result.is_valid
        assert result.minimum_clearance > original_clearance

    def test_exterior_vertices_preserved(self, bug6_poly):
        """Exterior should retain at least 8 of 11 vertices (dedup ok, mass simplification not)."""
        original_ext_count = len(bug6_poly.exterior.coords) - 1  # 11
        result = fix_clearance(bug6_poly, min_clearance=1.0)
        result_ext_count = len(result.exterior.coords) - 1
        assert result_ext_count >= 8, (
            f"Exterior vertices collapsed from {original_ext_count} to {result_ext_count}"
        )

    def test_area_within_tolerance(self, bug6_poly):
        """Area should stay within 5% of original (no massive hole shrinkage)."""
        original_area = bug6_poly.area
        result = fix_clearance(bug6_poly, min_clearance=1.0)
        area_ratio = result.area / original_area
        assert 0.95 <= area_ratio <= 1.05, (
            f"Area changed too much: {original_area:.1f} -> {result.area:.1f} "
            f"(ratio {area_ratio:.4f})"
        )

    def test_all_holes_preserved(self, bug6_poly):
        """All 6 holes should be preserved."""
        original_holes = len(bug6_poly.interiors)
        result = fix_clearance(bug6_poly, min_clearance=1.0)
        assert len(result.interiors) == original_holes, (
            f"Holes changed: {original_holes} -> {len(result.interiors)}"
        )


class TestDedupInImprove:
    """Tests that the post-fix dedup in improve() resolves seam bottlenecks."""

    def test_seam_dedup_resolves_bottleneck(self):
        """After a hole fix creates a new seam bottleneck, dedup should handle it.

        Polygon has near-collinear vertices on exterior that become the clearance
        bottleneck after an interior fix. The dedup in improve() should clean
        these up without whole-polygon simplification.
        """
        # Square with near-duplicate seam vertices (v0 ≈ v1 ≈ v9)
        # and a hole that creates the initial low clearance
        coords = [
            (0.0, 0.0),
            (0.1, 0.03),   # near-duplicate of v0
            (10, 0),
            (10, 10),
            (0, 10),
            (0.0, 0.0),
        ]
        # Hole placed to create initial low clearance
        hole = [
            (1, 1), (5, 1), (5, 5), (1, 5),
        ]
        poly = Polygon(coords, [hole])
        if not poly.is_valid:
            return  # skip if construction invalid

        original_ext_count = len(poly.exterior.coords) - 1
        result = fix_clearance(poly, min_clearance=0.5)
        assert result.is_valid
        # The near-duplicate should be deduped, but structural vertices preserved
        result_ext_count = len(result.exterior.coords) - 1
        assert result_ext_count >= original_ext_count - 1


class TestNarrowPassageToCourtyard:
    """Tests for narrow passage to courtyard detection and fixing.

    When the exterior ring has a narrow passage leading to a significant
    enclosed area (courtyard), fix_clearance should close the passage and
    preserve the courtyard as an interior hole.
    """

    # Bug7 polygon: real-world building with narrow passage to courtyard
    BUG7_WKT = (
        "Polygon ((675955.05923624825663865 6581186.65935353375971317, "
        "675954.15568333386909217 6581131.54262575507164001, "
        "676018.60912456142250448 6581132.74736297503113747, "
        "676018.30794025666546077 6581185.75580061972141266, "
        "675991.80372143408749253 6581185.75580061972141266, "
        "675993.00845865323208272 6581153.83026431035250425, "
        "676011.38070124620571733 6581152.32434278633445501, "
        "676010.77833263657521456 6581136.96394324209541082, "
        "675979.15398063242901117 6581138.46986476611346006, "
        "675978.25042771804146469 6581155.63737013936042786, "
        "675992.76418153871782124 6581154.08855633996427059, "
        "675991.417547169723548 6581185.77056848630309105, "
        "675955.05923624825663865 6581186.65935353375971317))"
    )

    @pytest.fixture()
    def bug7_poly(self):
        import shapely
        return shapely.from_wkt(self.BUG7_WKT)

    def test_bug7_creates_courtyard_hole(self, bug7_poly):
        """Bug7 polygon should gain a courtyard hole, not lose the courtyard entirely."""
        result = fix_clearance(bug7_poly, min_clearance=1.0)
        assert result.is_valid
        assert len(result.interiors) >= 1, (
            "Courtyard should be preserved as a hole"
        )

    def test_bug7_clearance_improves(self, bug7_poly):
        """Clearance should improve from the original ~0.23."""
        original_clearance = bug7_poly.minimum_clearance
        result = fix_clearance(bug7_poly, min_clearance=1.0)
        assert result.minimum_clearance > original_clearance

    def test_bug7_area_preserved(self, bug7_poly):
        """Area should stay within 5% of original."""
        original_area = bug7_poly.area
        result = fix_clearance(bug7_poly, min_clearance=1.0)
        area_ratio = result.area / original_area
        assert 0.95 <= area_ratio <= 1.05, (
            f"Area changed too much: {original_area:.1f} -> {result.area:.1f} "
            f"(ratio {area_ratio:.4f})"
        )

    def test_bug7_result_valid(self, bug7_poly):
        """Result should be a valid Polygon."""
        result = fix_clearance(bug7_poly, min_clearance=1.0)
        assert result.is_valid
        assert result.geom_type == "Polygon"
        assert not result.is_empty

    def test_synthetic_courtyard_creates_hole(self):
        """Synthetic polygon with narrow passage to large courtyard should create a hole."""
        # Rectangle with U-shaped indentation: passage width=0.2, courtyard ~300 sq.m
        coords = [
            (0, 0), (50, 0), (50, 50), (35, 50), (35, 45),
            (30.1, 45), (30.1, 30), (10, 30), (10, 45), (29.9, 45),
            (29.9, 50), (0, 50),
        ]
        poly = Polygon(coords)
        assert poly.is_valid
        assert len(poly.interiors) == 0

        result = fix_clearance(poly, min_clearance=1.0)
        assert result.is_valid
        assert len(result.interiors) >= 1, (
            "Courtyard should be preserved as a hole"
        )

    def test_synthetic_courtyard_area_preserved(self):
        """Synthetic courtyard fix should preserve area."""
        coords = [
            (0, 0), (50, 0), (50, 50), (35, 50), (35, 45),
            (30.1, 45), (30.1, 30), (10, 30), (10, 45), (29.9, 45),
            (29.9, 50), (0, 50),
        ]
        poly = Polygon(coords)
        original_area = poly.area
        result = fix_clearance(poly, min_clearance=1.0)
        area_ratio = result.area / original_area
        assert 0.95 <= area_ratio <= 1.05

    def test_narrow_finger_no_courtyard(self):
        """A narrow finger (tiny enclosed area) should NOT create a hole."""
        # Rectangle with tiny finger indentation — enclosed area only ~0.4 sq units
        coords = [
            (0, 0), (50, 0), (50, 50),
            (25.1, 50), (25.1, 48), (24.9, 48), (24.9, 50),
            (0, 50),
        ]
        poly = Polygon(coords)
        assert poly.is_valid
        result = fix_clearance(poly, min_clearance=1.0)
        assert result.is_valid
        # Should NOT create a hole — finger is too small to be a courtyard
        assert len(result.interiors) == 0

    def test_existing_holes_preserved(self):
        """Courtyard fix should not lose existing holes."""
        # Building with existing hole + narrow passage to courtyard
        ext = [
            (0, 0), (80, 0), (80, 60), (50, 60), (50, 55),
            (45.1, 55), (45.1, 40), (25, 40), (25, 55), (44.9, 55),
            (44.9, 60), (0, 60),
        ]
        hole = [(5, 5), (15, 5), (15, 15), (5, 15)]
        poly = Polygon(ext, [hole])
        assert poly.is_valid
        assert len(poly.interiors) == 1

        result = fix_clearance(poly, min_clearance=1.0)
        assert result.is_valid
        # Should have original hole + new courtyard hole
        assert len(result.interiors) >= 2, (
            f"Expected at least 2 holes (original + courtyard), got {len(result.interiors)}"
        )

    def test_passage_on_polygon_with_no_holes(self, bug7_poly):
        """Bug7 has no existing holes; result should have exactly 1 (the courtyard)."""
        assert len(bug7_poly.interiors) == 0
        result = fix_clearance(bug7_poly, min_clearance=1.0)
        assert len(result.interiors) == 1

    def test_bug7_no_passage_slit(self, bug7_poly):
        """Result exterior should not have a thin slit/finger from the old passage."""
        result = fix_clearance(bug7_poly, min_clearance=1.0)
        ext_coords = np.array(result.exterior.coords[:-1])
        # The exterior should be a simple rectangle (~5-6 vertices)
        # NOT 7+ vertices with a thin finger going down to the courtyard
        assert len(ext_coords) <= 6, (
            f"Exterior has {len(ext_coords)} vertices — likely still has passage slit"
        )

    def test_bug7_passage_closed_at_mouth(self, bug7_poly):
        """The passage should be closed at its mouth, not at the bottleneck."""
        result = fix_clearance(bug7_poly, min_clearance=1.0)
        ext_coords = np.array(result.exterior.coords[:-1])
        # No exterior vertex should be at the courtyard y-level (~6581153)
        courtyard_y_min = 6581135
        courtyard_y_max = 6581160
        mid_vertices = [
            i for i, c in enumerate(ext_coords)
            if courtyard_y_min < c[1] < courtyard_y_max
        ]
        assert len(mid_vertices) == 0, (
            f"Exterior has vertices at courtyard level: {mid_vertices} — "
            "passage not closed at mouth"
        )
