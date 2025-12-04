"""Tests for robust constraint-aware geometry fixing."""

import pytest
from shapely.geometry import Polygon, MultiPolygon, Point

from polyforge.repair import robust as robust_mod

from polyforge import (
    robust_fix_geometry,
    robust_fix_batch,
    repair_geometry,
)
from polyforge.core import (
    GeometryConstraints,
    ConstraintStatus,
    ConstraintType,
    ConstraintViolation,
    FixWarning,
)


def _spike_polygon() -> Polygon:
    return Polygon([
        (0, 0),
        (10, 0),
        (10, 10),
        (5.001, 10),
        (5.001, 10.002),
        (4.999, 10.002),
        (4.999, 10),
        (0, 10),
        (0, 0),
    ])


def _sensitive_polygon() -> Polygon:
    return Polygon([
        (0, 0),
        (1, 0),
        (1, 1),
        (0.55, 1),
        (0.55, 1.1),
        (0.45, 1.1),
        (0.45, 1),
        (0, 1),
        (0, 0),
    ])


def _polygon_with_sliver() -> Polygon:
    """Create a polygon with a thin sliver extending from the main body."""
    # Main body is a large ellipse
    # Add a thin sliver extending from the right side
    # The sliver has width ~1.2 and extends horizontally
    coords = [
        # Main ellipse body (approximate)
        (250, 400),
        (300, 390),
        (350, 395),
        (400, 405),
        (450, 420),
        (500, 440),
        (550, 465),
        (600, 490),
        (650, 510),
        (700, 520),
        (750, 520),
        # Sliver extends here - thin horizontal protrusion
        (800, 520),  # Tip of sliver
        (875, 520.6),  # Sliver top edge (width = 1.2)
        (800, 521.2),  # Sliver returns
        (750, 521.2),
        # Bottom half of ellipse
        (700, 521),
        (650, 531),
        (600, 550),
        (550, 575),
        (500, 600),
        (450, 620),
        (400, 635),
        (350, 645),
        (300, 650),
        (250, 640),
        (200, 620),
        (150, 590),
        (100, 550),
        (60, 500),
        (40, 450),
        (200, 430),
        (250, 400),
    ]
    return Polygon(coords)


class TestCleanupHelpers:
    """Unit coverage for the new cleanup helpers."""

    def test_smooth_low_clearance_improves_clearance(self):
        spike = _spike_polygon()
        baseline = spike.minimum_clearance

        # With new signature: pass min_clearance target
        # The function will buffer by min_clearance * 0.5
        improved = robust_mod._smooth_low_clearance(
            spike,
            min_clearance=0.02,  # Will buffer by 0.01
        )

        assert improved.minimum_clearance > baseline
        assert improved.is_valid

    def test_smooth_low_clearance_respects_area_limit(self):
        sensitive = _sensitive_polygon()

        # With new signature: use large min_clearance to trigger area loss rejection
        # The function will attempt to buffer by min_clearance * 0.5 = 0.5
        # This should exceed the automatic area loss tolerance for this small polygon
        strict = robust_mod._smooth_low_clearance(
            sensitive,
            min_clearance=1.0,  # Will attempt to buffer by 0.5
        )

        # Should return original due to excessive area loss
        assert strict.equals(sensitive)

    def test_cleanup_geometry_noop_for_non_polygon(self):
        point = Point(0, 0)
        constraints = GeometryConstraints(must_be_valid=True)

        result = robust_mod._cleanup_geometry(point, constraints)
        assert result.equals(point)


class TestGeometryConstraints:
    """Test constraint checking and validation."""

    def test_all_constraints_satisfied(self):
        """Test checking when all constraints are met."""
        # Create a simple valid square
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        constraints = GeometryConstraints(
            min_clearance=2.0,
            min_area_ratio=0.8,
            must_be_valid=True
        )

        status = constraints.check(poly, poly)

        assert status.all_satisfied()
        assert status.validity
        assert len(status.violations) == 0

    def test_validity_constraint_violated(self):
        """Test detection of invalid geometry."""
        # Create invalid self-intersecting polygon (bow-tie)
        poly = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])

        constraints = GeometryConstraints(must_be_valid=True)
        status = constraints.check(poly, poly)

        assert not status.all_satisfied()
        assert not status.validity
        assert len(status.violations) > 0
        assert any(v.constraint_type == ConstraintType.VALIDITY for v in status.violations)

    def test_clearance_constraint_violated(self):
        """Test detection of low clearance."""
        # Create polygon with very low clearance (thin rectangle)
        poly = Polygon([(0, 0), (100, 0), (100, 0.5), (0, 0.5)])

        constraints = GeometryConstraints(min_clearance=2.0)
        status = constraints.check(poly, poly)

        assert not status.all_satisfied()
        assert status.clearance is not None
        assert status.clearance < 2.0
        clearance_violations = status.get_violations_by_type(ConstraintType.CLEARANCE)
        assert len(clearance_violations) > 0

    def test_area_preservation_constraint(self):
        """Test detection of excessive area loss."""
        original = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        # Small polygon (50% area loss)
        small = Polygon([(0, 0), (5, 0), (5, 10), (0, 10)])

        constraints = GeometryConstraints(min_area_ratio=0.9)
        status = constraints.check(small, original)

        assert not status.all_satisfied()
        assert status.area_ratio < 0.9
        area_violations = status.get_violations_by_type(ConstraintType.AREA_PRESERVATION)
        assert len(area_violations) > 0

    def test_empty_geometry(self):
        """Test handling of empty geometry."""
        from shapely.geometry import Polygon as EmptyPoly
        empty = EmptyPoly()
        original = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        constraints = GeometryConstraints(must_be_valid=True)
        status = constraints.check(empty, original)

        assert not status.all_satisfied()
        assert len(status.violations) > 0

    def test_overlap_constraint_violation(self):
        """Overlap area exceeding threshold triggers violation."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        constraints = GeometryConstraints(max_overlap_area=1.0)
        status = constraints.check(poly, poly, overlap_area=5.0)

        assert not status.all_satisfied()
        overlap_violations = status.get_violations_by_type(ConstraintType.OVERLAP)
        assert overlap_violations
        assert overlap_violations[0].actual_value == pytest.approx(5.0)

    def test_disallow_multipolygon_constraint(self):
        """Disallowing multipolygons adds validity violation."""
        multi = MultiPolygon([
            Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
            Polygon([(10, 0), (15, 0), (15, 5), (10, 5)]),
        ])
        constraints = GeometryConstraints(allow_multipolygon=False)
        status = constraints.check(multi, multi)

        assert not status.all_satisfied()
        validity = status.get_violations_by_type(ConstraintType.VALIDITY)
        assert any("multipolygon" in v.message for v in validity)


class TestConstraintStatus:
    """Test constraint status comparison and analysis."""

    def test_is_better_or_equal_fewer_violations(self):
        """Test that status with fewer violations is better."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        # Create two statuses with different violation counts
        status1 = ConstraintStatus(
            geometry=poly,
            violations=[
                ConstraintViolation(ConstraintType.CLEARANCE, 5.0, "Low clearance"),
                ConstraintViolation(ConstraintType.AREA_PRESERVATION, 3.0, "Area loss")
            ]
        )

        status2 = ConstraintStatus(
            geometry=poly,
            violations=[
                ConstraintViolation(ConstraintType.CLEARANCE, 5.0, "Low clearance")
            ]
        )

        # status2 has fewer violations, so it's better
        assert status2.is_better_or_equal(status1)
        assert not status1.is_better_or_equal(status2)

    def test_improved_checks_strict_improvement(self):
        """Test that improved requires strict improvement."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        status1 = ConstraintStatus(
            geometry=poly,
            violations=[
                ConstraintViolation(ConstraintType.CLEARANCE, 5.0, "Low clearance")
            ]
        )

        status2 = ConstraintStatus(
            geometry=poly,
            violations=[
                ConstraintViolation(ConstraintType.CLEARANCE, 3.0, "Low clearance")
            ]
        )

        # status2 has lower severity, so it's improved
        assert status2.improved(status1)
        assert not status1.improved(status2)

    def test_worst_violation(self):
        """Test finding the most severe violation."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        status = ConstraintStatus(
            geometry=poly,
            violations=[
                ConstraintViolation(ConstraintType.CLEARANCE, 5.0, "Low clearance"),
                ConstraintViolation(ConstraintType.AREA_PRESERVATION, 15.0, "Large area loss"),
                ConstraintViolation(ConstraintType.VALIDITY, 3.0, "Minor issue")
            ]
        )

        worst = status.worst_violation()
        assert worst is not None
        assert worst.constraint_type == ConstraintType.AREA_PRESERVATION
        assert worst.severity == 15.0


class TestRobustFixGeometry:
    """Test robust_fix_geometry function."""

    def test_fix_invalid_geometry(self):
        """Test fixing invalid geometry while preserving constraints."""
        # Create invalid bow-tie polygon
        invalid = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])

        constraints = GeometryConstraints(
            must_be_valid=True,
            min_area_ratio=0.5  # Allow significant area change for repair
        )

        fixed, warning = robust_fix_geometry(invalid, constraints)

        # Should be fixed
        assert fixed.is_valid
        assert warning is None or not warning.status.get_violations_by_type(ConstraintType.VALIDITY)

    def test_fix_low_clearance(self):
        """Test fixing low clearance geometry."""
        # Create very thin rectangle with low clearance
        thin = Polygon([(0, 0), (100, 0), (100, 0.1), (0, 0.1)])

        constraints = GeometryConstraints(
            min_clearance=1.0,
            min_area_ratio=0.0,  # Allow any area change
            must_be_valid=True
        )

        with pytest.warns(UserWarning):
            fixed, warning = robust_fix_geometry(thin, constraints, verbose=False)

        # Clearance fixing may not always improve thin geometries
        # But should at least return valid geometry without making things worse
        assert fixed.is_valid
        if fixed.is_valid and not fixed.is_empty:
            fixed_clearance = fixed.minimum_clearance
            original_clearance = thin.minimum_clearance
            # Should not make clearance worse
            assert fixed_clearance >= original_clearance

    def test_remove_thin_sliver_with_buffer_approach(self):
        """Test that thin slivers are removed when they violate min_clearance."""
        # Create polygon with thin sliver (width ~1.2)
        poly_with_sliver = _polygon_with_sliver()

        # Verify the polygon has low clearance initially
        original_clearance = poly_with_sliver.minimum_clearance
        assert original_clearance < 2.0, "Test polygon should have clearance < 2.0"

        constraints = GeometryConstraints(
            min_clearance=2.0,
            min_area_ratio=0.5,  # Allow some area loss to remove sliver
            must_be_valid=True
        )

        fixed, warning = robust_fix_geometry(poly_with_sliver, constraints)

        # Should be valid
        assert fixed.is_valid
        assert not fixed.is_empty

        # Clearance should be improved (sliver removed by buffer operation)
        fixed_clearance = fixed.minimum_clearance
        assert fixed_clearance > original_clearance, "Clearance should improve after fixing"

        # Should satisfy the constraint (or at least get close)
        # The buffer-based approach should remove features narrower than min_clearance/2 = 1.0
        # Since the sliver has width 1.2, it should be removed or significantly reduced
        if warning and warning.unmet_constraints:
            # If still not meeting constraint, should at least be much closer
            assert fixed_clearance >= 0.9 * constraints.min_clearance, \
                f"Fixed clearance {fixed_clearance} should be close to target {constraints.min_clearance}"

    def test_returns_warning_when_constraints_not_met(self):
        """Test that warning is returned when constraints cannot be satisfied."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        # Create impossible constraints
        constraints = GeometryConstraints(
            min_clearance=1000.0,  # Impossible for this geometry
            must_be_valid=True
        )

        with pytest.warns(UserWarning):
            fixed, warning = robust_fix_geometry(poly, constraints, raise_on_failure=False)

        assert warning is not None
        assert isinstance(warning, FixWarning)
        assert len(warning.unmet_constraints) > 0
        assert 'CLEARANCE' in warning.unmet_constraints

    def test_allow_multipolygon_false_returns_single_polygon(self):
        """robust_fix_geometry collapses multipolygons when disallowed."""
        multi = MultiPolygon([
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            Polygon([(5, 0), (7, 0), (7, 2), (5, 2)]),
        ])
        constraints = GeometryConstraints(
            allow_multipolygon=False,
            must_be_valid=True,
        )

        fixed, warning = robust_fix_geometry(multi, constraints)

        assert isinstance(fixed, Polygon)
        assert fixed.area == pytest.approx(4.0)
        assert warning is None

    def test_raises_on_failure_when_requested(self):
        """Test that FixWarning is raised when raise_on_failure=True."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        constraints = GeometryConstraints(
            min_clearance=1000.0,
            must_be_valid=True
        )

        with pytest.raises(FixWarning) as exc_info:
            robust_fix_geometry(poly, constraints, raise_on_failure=True)

        assert len(exc_info.value.unmet_constraints) > 0

    def test_prevents_fix_interference(self):
        """Test that fixes don't undo previous fixes."""
        # Create geometry with multiple issues
        # First create valid base
        base = Polygon([(0, 0), (20, 0), (20, 0.1), (0, 0.1)])  # Low clearance

        constraints = GeometryConstraints(
            min_clearance=0.5,
            min_area_ratio=0.3,
            must_be_valid=True
        )

        with pytest.warns(UserWarning):
            fixed, warning = robust_fix_geometry(base, constraints, verbose=False)

        # Check that it's still valid after all fixes
        assert fixed.is_valid

        # The key test: we should not make things worse
        # Check that violations didn't increase
        initial_status = constraints.check(base, base)
        if warning:
            # Should have same or fewer violations (not more)
            assert len(warning.status.violations) <= len(initial_status.violations)
            # Should not have worse total severity
            initial_severity = sum(v.severity for v in initial_status.violations)
            final_severity = sum(v.severity for v in warning.status.violations)
            assert final_severity <= initial_severity

    def test_max_iterations_limit(self):
        """Test that max_iterations prevents infinite loops."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        constraints = GeometryConstraints(
            min_clearance=1000.0  # Impossible
        )

        # Should complete without hanging
        with pytest.warns(UserWarning):
            fixed, warning = robust_fix_geometry(
                poly,
                constraints,
                max_iterations=3,
                raise_on_failure=False
            )

        assert warning is not None  # Couldn't satisfy constraints
        assert fixed.is_valid  # But returned valid geometry

    def test_no_changes_when_already_satisfied(self):
        """Test that geometry is not modified when constraints already met."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        constraints = GeometryConstraints(
            min_clearance=2.0,
            min_area_ratio=0.9,
            must_be_valid=True
        )

        fixed, warning = robust_fix_geometry(poly, constraints)

        # Should succeed without warning
        assert warning is None
        assert fixed.is_valid

    def test_area_ratio_compares_against_raw_input(self):
        """Area ratio should be measured against caller-provided geometry, not pre-cleaned version."""
        # Overlapping MultiPolygon: raw area sums components (8); union after cleanup is 6.
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = Polygon([(1, 0), (3, 0), (3, 2), (1, 2)])
        multi = MultiPolygon([poly1, poly2])

        constraints = GeometryConstraints(
            min_area_ratio=0.9,
            must_be_valid=True,
        )

        with pytest.warns(UserWarning):
            fixed, warning = robust_fix_geometry(multi, constraints)

        assert warning is not None
        area_violations = warning.status.get_violations_by_type(ConstraintType.AREA_PRESERVATION)
        assert area_violations, "Expected area preservation violation relative to raw input"
        # Raw area = 8, cleaned/unioned area = 6
        assert warning.status.area_ratio == pytest.approx(0.75, rel=1e-9)
        assert fixed.is_valid


class TestRobustFixBatch:
    """Test robust_fix_batch for multiple geometries."""

    def test_fix_multiple_geometries(self):
        """Test fixing multiple geometries."""
        poly1 = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])  # Invalid bow-tie
        poly2 = Polygon([(20, 0), (30, 0), (30, 10), (20, 10)])  # Valid
        poly3 = Polygon([(40, 0), (140, 0), (140, 0.1), (40, 0.1)])  # Low clearance

        geometries = [poly1, poly2, poly3]

        constraints = GeometryConstraints(
            must_be_valid=True,
            min_clearance=0.5,
            min_area_ratio=0.3
        )

        with pytest.warns(UserWarning):
            fixed_list, warnings, _ = robust_fix_batch(
                geometries,
                constraints,
                handle_overlaps=False,
                verbose=False
            )

        assert len(fixed_list) == 3
        assert len(warnings) == 3

        # All should be valid
        assert all(geom.is_valid for geom in fixed_list)

    def test_handle_overlaps_in_batch(self):
        """Test overlap resolution in batch mode."""
        # Create overlapping polygons
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(5, 0), (15, 0), (15, 10), (5, 10)])  # Overlaps with poly1

        geometries = [poly1, poly2]

        constraints = GeometryConstraints(
            must_be_valid=True,
            max_overlap_area=0.0  # No overlaps allowed
        )

        fixed_list, warnings, _ = robust_fix_batch(
            geometries,
            constraints,
            handle_overlaps=True,
            verbose=False
        )

        assert len(fixed_list) == 2

        # Check no overlaps
        if fixed_list[0].is_valid and fixed_list[1].is_valid:
            intersection = fixed_list[0].intersection(fixed_list[1])
            # Should be minimal or no overlap (may have tiny numerical artifacts)
            assert intersection.area < 0.001

    def test_batch_preserves_order(self):
        """Test that batch processing preserves geometry order."""
        geometries = [
            Polygon([(i*20, 0), (i*20+10, 0), (i*20+10, 10), (i*20, 10)])
            for i in range(5)
        ]

        constraints = GeometryConstraints(must_be_valid=True)

        fixed_list, warnings, _ = robust_fix_batch(
            geometries,
            constraints,
            handle_overlaps=False
        )

        assert len(fixed_list) == len(geometries)

        # Check areas are roughly preserved (order maintained)
        for original, fixed in zip(geometries, fixed_list):
            assert abs(fixed.area - original.area) < 1.0

    def test_overlap_violation_reported_without_resolution(self):
        """Overlap violations are surfaced when not resolving overlaps."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(5, 0), (15, 0), (15, 10), (5, 10)])

        constraints = GeometryConstraints(
            must_be_valid=True,
            max_overlap_area=1.0,
        )

        with pytest.warns(UserWarning):
            _, warnings, _ = robust_fix_batch(
                [poly1, poly2],
                constraints,
                handle_overlaps=False,
                verbose=False,
            )

        assert any(warnings)
        first_warning = next(w for w in warnings if w is not None)
        assert first_warning is not None
        overlap_violations = first_warning.status.get_violations_by_type(ConstraintType.OVERLAP)
        assert overlap_violations


class TestIntegration:
    """Integration tests for real-world scenarios."""

    def test_complex_fix_scenario(self):
        """Test a complex scenario with multiple issues."""
        # Create a challenging geometry:
        # - Valid but with holes
        # - Low clearance
        # - Complex shape

        exterior = [(0, 0), (50, 0), (50, 50), (0, 50)]
        hole = [(10, 10), (40, 10), (40, 40), (10, 40)]

        poly_with_hole = Polygon(exterior, [hole])

        # Make it have low clearance by adding narrow protrusion
        coords = list(poly_with_hole.exterior.coords)
        coords.insert(1, (50, 0.1))  # Very narrow protrusion
        narrow_poly = Polygon(coords, poly_with_hole.interiors)

        constraints = GeometryConstraints(
            must_be_valid=True,
            min_clearance=1.0,
            min_area_ratio=0.7,
            max_holes=1
        )

        fixed, warning = robust_fix_geometry(
            narrow_poly,
            constraints,
            max_iterations=20,
            verbose=False
        )

        # Should at least be valid
        assert fixed.is_valid

        # Should have made some improvement
        if warning:
            # Check we didn't make things worse
            initial_violations = len(constraints.check(narrow_poly, narrow_poly).violations)
            final_violations = len(warning.status.violations)
            assert final_violations <= initial_violations

    def test_constraint_validation_comprehensive(self):
        """Test that all constraint types are properly validated."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        # Test each constraint type
        constraints_list = [
            (GeometryConstraints(must_be_valid=True), "validity"),
            (GeometryConstraints(min_clearance=0.5), "clearance"),
            (GeometryConstraints(min_area_ratio=0.9), "area"),
            (GeometryConstraints(max_holes=0), "holes"),
        ]

        for constraints, name in constraints_list:
            status = constraints.check(poly, poly)
            # All should be satisfied for this simple square
            assert status.all_satisfied(), f"Failed for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
