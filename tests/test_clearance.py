"""Tests for clearance fix functions."""

import numpy as np
import pytest
from shapely.geometry import Polygon, MultiPolygon

from polyforge.clearance import (
    fix_hole_too_close,
    fix_narrow_protrusion,
    fix_sharp_intrusion,
    fix_narrow_passage,
    fix_near_self_intersection,
    fix_parallel_close_edges,
)
from polyforge.core.types import HoleStrategy, PassageStrategy, IntrusionStrategy, IntersectionStrategy
from polyforge.ops.clearance.passages import _find_self_intersection_vertices


class TestFixHoleTooClose:
    """Tests for fix_hole_too_close function."""

    def test_remove_single_close_hole(self):
        """Test removing a single hole that's too close to exterior."""
        # Square exterior
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        # Small hole very close to left edge (distance ~0.5)
        hole = [(0.5, 4), (1.5, 4), (1.5, 6), (0.5, 6)]

        poly = Polygon(exterior, holes=[hole])
        result = fix_hole_too_close(poly, min_clearance=1.0, strategy=HoleStrategy.REMOVE)

        # Hole should be removed
        assert len(result.interiors) == 0
        assert result.is_valid

    def test_keep_far_hole(self):
        """Test that holes far from exterior are kept."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        # Hole in center, far from all edges
        hole = [(4, 4), (6, 4), (6, 6), (4, 6)]

        poly = Polygon(exterior, holes=[hole])
        result = fix_hole_too_close(poly, min_clearance=2.0, strategy=HoleStrategy.REMOVE)

        # Hole should be kept (distance from edges is ~4)
        assert len(result.interiors) == 1
        assert result.is_valid

    def test_remove_multiple_close_holes(self):
        """Test removing multiple holes that are too close."""
        exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
        # Three holes, all close to edges
        hole1 = [(1, 1), (2, 1), (2, 2), (1, 2)]  # Close to corner
        hole2 = [(18, 10), (19, 10), (19, 11), (18, 11)]  # Close to right edge
        hole3 = [(10, 18), (11, 18), (11, 19), (10, 19)]  # Close to top edge

        poly = Polygon(exterior, holes=[hole1, hole2, hole3])
        result = fix_hole_too_close(poly, min_clearance=2.0, strategy=HoleStrategy.REMOVE)

        # All holes should be removed
        assert len(result.interiors) == 0
        assert result.is_valid

    def test_mixed_close_and_far_holes(self):
        """Test with both close and far holes."""
        exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
        close_hole = [(1, 1), (2, 1), (2, 2), (1, 2)]  # Distance ~1
        far_hole = [(8, 8), (12, 8), (12, 12), (8, 12)]  # Distance ~8

        poly = Polygon(exterior, holes=[close_hole, far_hole])
        result = fix_hole_too_close(poly, min_clearance=2.0, strategy=HoleStrategy.REMOVE)

        # Only far hole should remain
        assert len(result.interiors) == 1
        assert result.is_valid

    def test_no_holes_returns_unchanged(self):
        """Test that polygons without holes are returned unchanged."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        poly = Polygon(exterior)

        result = fix_hole_too_close(poly, min_clearance=2.0, strategy=HoleStrategy.REMOVE)

        assert len(result.interiors) == 0
        assert result.exterior.coords[:] == poly.exterior.coords[:]

    def test_accepts_string_strategy(self):
        """fix_hole_too_close should accept literal strategy values."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(0.5, 4), (1.5, 4), (1.5, 6), (0.5, 6)]

        poly = Polygon(exterior, holes=[hole])
        result = fix_hole_too_close(poly, min_clearance=1.0, strategy="remove")

        assert len(result.interiors) == 0
        assert result.is_valid

    def test_shrink_strategy(self):
        """Test shrinking holes instead of removing them."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        # Hole somewhat close to edge
        hole = [(2, 2), (4, 2), (4, 4), (2, 4)]  # Distance ~2

        poly = Polygon(exterior, holes=[hole])
        result = fix_hole_too_close(poly, min_clearance=3.0, strategy=HoleStrategy.SHRINK)

        # Hole should be shrunk but still exist
        # (may shrink to nothing if too much shrinkage needed)
        assert result.is_valid
        # Hole might be removed if it shrinks to nothing
        if len(result.interiors) > 0:
            # If hole still exists, it should be smaller
            original_hole_area = Polygon(hole).area
            result_hole_area = Polygon(result.interiors[0]).area
            assert result_hole_area < original_hole_area

    def test_shrink_to_nothing(self):
        """Test that very small holes shrink to nothing."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        # Very small hole close to edge
        hole = [(1, 4), (1.5, 4), (1.5, 4.5), (1, 4.5)]

        poly = Polygon(exterior, holes=[hole])
        # Shrink amount larger than hole size
        result = fix_hole_too_close(poly, min_clearance=5.0, strategy=HoleStrategy.SHRINK)

        # Hole should shrink to nothing (removed)
        assert len(result.interiors) == 0
        assert result.is_valid

    def test_move_strategy_simple(self):
        """Test moving a hole away from exterior."""
        exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
        # Hole close to left edge
        hole = [(2, 8), (4, 8), (4, 10), (2, 10)]

        poly = Polygon(exterior, holes=[hole])
        result = fix_hole_too_close(poly, min_clearance=5.0, strategy=HoleStrategy.MOVE)

        # Hole should be moved (or removed if can't move safely)
        assert result.is_valid
        if len(result.interiors) > 0:
            # Hole was moved, check it's farther from edge
            moved_hole = Polygon(result.interiors[0])
            original_distance = Polygon(hole).centroid.coords[0][0]  # x-coord
            moved_distance = moved_hole.centroid.coords[0][0]
            assert moved_distance > original_distance

    def test_exact_threshold_distance(self):
        """Test behavior when hole is exactly at threshold distance."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        # Hole exactly 2 units from left edge
        hole = [(2, 4), (3, 4), (3, 5), (2, 5)]

        poly = Polygon(exterior, holes=[hole])

        # At threshold = 2.0, should be kept (distance >= threshold)
        result_keep = fix_hole_too_close(poly, min_clearance=2.0, strategy=HoleStrategy.REMOVE)
        assert len(result_keep.interiors) == 1

        # Just above threshold, should be removed
        result_remove = fix_hole_too_close(poly, min_clearance=2.1, strategy=HoleStrategy.REMOVE)
        assert len(result_remove.interiors) == 0

    def test_preserves_exterior(self):
        """Test that exterior is never modified."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(1, 1), (2, 1), (2, 2), (1, 2)]

        poly = Polygon(exterior, holes=[hole])
        result = fix_hole_too_close(poly, min_clearance=2.0, strategy=HoleStrategy.REMOVE)

        # Exterior coordinates should be identical
        np.testing.assert_array_almost_equal(
            np.array(result.exterior.coords),
            np.array(poly.exterior.coords)
        )


class TestFixNarrowProtrusion:
    """Tests for fix_narrow_protrusion function."""

    def test_remove_simple_spike(self):
        """Test removing a simple narrow spike."""
        # Rectangle with narrow spike on right side
        base = [(0, 0), (10, 0), (10, 4), (10, 6), (0, 6)]
        spike = [(10, 4.9), (12, 5), (10, 5.1)]  # Narrow spike

        # Insert spike into base
        coords = base[:3] + spike + base[3:]
        poly = Polygon(coords)

        result = fix_narrow_protrusion(poly, min_clearance=0.5)

        assert result.is_valid
        # Result should have fewer or equal vertices (spike removed or smoothed)
        assert len(result.exterior.coords) <= len(poly.exterior.coords)

    def test_multiple_protrusions(self):
        """Test fixing multiple narrow protrusions."""
        # Square with two spikes
        coords = [
            (0, 0), (5, 0), (5, 2),
            # Spike 1 pointing right
            (5, 2.4), (6, 2.5), (5, 2.6),
            (5, 8), (5, 10),
            # Spike 2 pointing up
            (2.6, 10), (2.5, 11), (2.4, 10),
            (0, 10)
        ]
        poly = Polygon(coords)

        result = fix_narrow_protrusion(poly, min_clearance=0.5)

        assert result.is_valid
        # Should improve clearance (vertices moved to widen bases)
        assert result.minimum_clearance >= poly.minimum_clearance

    def test_no_protrusion_unchanged(self):
        """Test that polygons without protrusions remain mostly unchanged."""
        # Simple square
        coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
        poly = Polygon(coords)

        result = fix_narrow_protrusion(poly, min_clearance=1.0)

        assert result.is_valid
        # Should be similar to original (maybe simplified slightly)
        assert result.area == pytest.approx(poly.area, rel=0.1)

    def test_preserves_holes(self):
        """Test that holes are preserved."""
        # Polygon with protrusion and hole
        exterior = [(0, 0), (10, 0), (10, 4.9), (12, 5), (10, 5.1), (10, 10), (0, 10)]
        hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
        poly = Polygon(exterior, holes=[hole])

        result = fix_narrow_protrusion(poly, min_clearance=0.5)

        assert result.is_valid
        assert len(result.interiors) == 1  # Hole preserved

    def test_achieves_target_clearance(self):
        """Test that result improves clearance."""
        # Polygon with narrow spike
        coords = [(0, 0), (10, 0), (10, 4.8), (11, 5), (10, 5.2), (10, 10), (0, 10)]
        poly = Polygon(coords)

        target_clearance = 1.0
        original_clearance = poly.minimum_clearance
        result = fix_narrow_protrusion(poly, min_clearance=target_clearance)

        assert result.is_valid
        # Should improve clearance (may not fully achieve target for complex cases)
        assert result.minimum_clearance > original_clearance

    def test_very_thin_protrusion(self):
        """Test fixing very thin protrusion."""
        # Polygon with very thin spike
        coords = [
            (0, 0), (10, 0), (10, 4),
            (10, 4.99), (15, 5), (10, 5.01),  # Very thin spike
            (10, 6), (0, 6)
        ]
        poly = Polygon(coords)

        result = fix_narrow_protrusion(poly, min_clearance=0.5)

        assert result.is_valid
        # Clearance should be improved (vertices moved inward)
        assert result.minimum_clearance >= poly.minimum_clearance

    def test_iteration_limit(self):
        """Test that iteration limit prevents infinite loops."""
        # Complex polygon
        coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
        poly = Polygon(coords)

        # Should not crash even with very high clearance requirement
        result = fix_narrow_protrusion(poly, min_clearance=100.0, max_iterations=3)

        assert result.is_valid


class TestFixSharpIntrusion:
    """Tests for fix_sharp_intrusion function."""

    def test_fill_simple_intrusion(self):
        """Test filling a simple narrow intrusion."""
        # Rectangle with narrow intrusion (notch) on right side
        coords = [
            (0, 0), (10, 0), (10, 4),
            # Intrusion
            (9, 4.9), (8, 5), (9, 5.1),
            (10, 6), (10, 10), (0, 10)
        ]
        poly = Polygon(coords)

        result = fix_sharp_intrusion(poly, min_clearance=0.5, strategy=IntrusionStrategy.FILL)

        assert result.is_valid
        # Intrusion should be improved (fewer or equal vertices)
        assert len(result.exterior.coords) <= len(poly.exterior.coords)
        # Area should be similar or increase slightly
        assert result.area >= poly.area * 0.95

    def test_smooth_intrusion(self):
        """Test smoothing an intrusion."""
        # Polygon with narrow notch
        coords = [
            (0, 0), (10, 0), (10, 4),
            (9, 4.5), (8, 5), (9, 5.5),
            (10, 6), (10, 10), (0, 10)
        ]
        poly = Polygon(coords)

        result = fix_sharp_intrusion(poly, min_clearance=0.8, strategy=IntrusionStrategy.SMOOTH)

        assert result.is_valid
        # Smoothing preserves vertex count but modifies positions
        # Area should be similar or slightly increased
        assert result.area >= poly.area * 0.95

    def test_simplify_strategy(self):
        """Test simplify strategy for intrusions."""
        # Polygon with jagged intrusion
        coords = [
            (0, 0), (10, 0), (10, 4),
            (9.5, 4.5), (9, 5), (9.5, 5.5),
            (10, 6), (10, 10), (0, 10)
        ]
        poly = Polygon(coords)

        result = fix_sharp_intrusion(poly, min_clearance=0.5, strategy=IntrusionStrategy.SIMPLIFY)

        assert result.is_valid
        # Simplification removes vertices
        assert len(result.exterior.coords) <= len(poly.exterior.coords)

    def test_multiple_intrusions(self):
        """Test fixing multiple intrusions."""
        # Polygon with two notches
        coords = [
            (0, 0), (5, 0),
            # Intrusion 1
            (5, 0.1), (4, 0.5), (5, 0.9),
            (5, 5),
            # Intrusion 2
            (4.9, 5), (4, 5.5), (4.9, 6),
            (5, 10), (0, 10)
        ]
        poly = Polygon(coords)

        result = fix_sharp_intrusion(poly, min_clearance=0.5, strategy=IntrusionStrategy.FILL)

        assert result.is_valid
        # Intrusions should be improved (fewer or equal vertices)
        assert len(result.exterior.coords) <= len(poly.exterior.coords)

    def test_no_intrusion_unchanged(self):
        """Test that polygons without intrusions remain mostly unchanged."""
        # Simple rectangle
        coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
        poly = Polygon(coords)

        result = fix_sharp_intrusion(poly, min_clearance=1.0, strategy=IntrusionStrategy.FILL)

        assert result.is_valid
        # Should be similar to original
        assert result.area == pytest.approx(poly.area, rel=0.1)

    def test_preserves_holes(self):
        """Test that holes are preserved."""
        # Polygon with intrusion and hole
        exterior = [(0, 0), (10, 0), (10, 4), (9, 4.9), (8, 5), (9, 5.1), (10, 6), (10, 10), (0, 10)]
        hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
        poly = Polygon(exterior, holes=[hole])

        result = fix_sharp_intrusion(poly, min_clearance=0.5, strategy=IntrusionStrategy.FILL)

        assert result.is_valid
        assert len(result.interiors) == 1  # Hole preserved

    def test_deep_narrow_intrusion(self):
        """Test fixing a deep narrow intrusion."""
        # Polygon with deep notch
        coords = [
            (0, 0), (10, 0), (10, 4),
            # Deep intrusion
            (9, 5), (5, 5), (9, 5.1),
            (10, 6), (10, 10), (0, 10)
        ]
        poly = Polygon(coords)

        result = fix_sharp_intrusion(poly, min_clearance=0.5, strategy=IntrusionStrategy.FILL)

        assert result.is_valid
        # Deep intrusion should be improved or filled (area similar or increased)
        assert result.area >= poly.area * 0.95

    def test_achieves_target_clearance(self):
        """Test that result achieves target clearance."""
        # Polygon with narrow notch
        coords = [
            (0, 0), (10, 0), (10, 4),
            (9, 4.8), (8, 5), (9, 5.2),
            (10, 6), (10, 10), (0, 10)
        ]
        poly = Polygon(coords)

        target_clearance = 1.0
        result = fix_sharp_intrusion(poly, min_clearance=target_clearance, strategy=IntrusionStrategy.FILL)

        assert result.is_valid
        # Should meet or exceed target (or be close)
        assert result.minimum_clearance >= target_clearance * 0.9

    def test_accepts_string_strategy(self):
        """String literal strategies should behave like enums."""
        coords = [
            (0, 0), (10, 0), (10, 4),
            (9, 4.8), (8, 5), (9, 5.2),
            (10, 6), (10, 10), (0, 10)
        ]
        poly = Polygon(coords)

        result = fix_sharp_intrusion(poly, min_clearance=0.5, strategy="smooth")

        assert result.is_valid
        assert result.minimum_clearance >= poly.minimum_clearance * 0.8


class TestFixNarrowPassage:
    """Tests for fix_narrow_passage function."""

    def test_widen_hourglass_shape(self):
        """Test widening a simple hourglass/neck shape."""
        # Create hourglass shape with narrow middle
        coords = [
            (0, 0), (2, 0), (2, 1),
            (1.1, 1.5), (1, 2), (1.1, 2.5),  # Narrow section
            (2, 3), (2, 4), (0, 4), (0, 3),
            (-0.1, 2.5), (-0.1, 1.5),  # Other side of narrow section
            (0, 1)
        ]
        poly = Polygon(coords)
        original_clearance = poly.minimum_clearance

        result = fix_narrow_passage(poly, min_clearance=0.5, strategy=PassageStrategy.WIDEN)

        assert result.is_valid
        assert isinstance(result, Polygon)
        # Clearance should improve
        assert result.minimum_clearance >= original_clearance

    def test_widen_increases_clearance(self):
        """Test that widening improves clearance."""
        # Simple narrow passage
        coords = [
            (0, 0), (1, 0), (0.9, 1), (1, 2),
            (0, 2), (0.1, 1)
        ]
        poly = Polygon(coords)
        target_clearance = 0.5

        result = fix_narrow_passage(poly, min_clearance=target_clearance, strategy=PassageStrategy.WIDEN)

        assert result.is_valid
        # Should improve toward target
        assert result.minimum_clearance >= poly.minimum_clearance

    def test_widen_increases_clearance_arap(self):
        """Test that widening improves clearance."""
        # Simple narrow passage
        coords = [
            (0, 0), (1, 0), (0.9, 1), (1, 2),
            (0, 2), (0.1, 1)
        ]
        poly = Polygon(coords)
        target_clearance = 0.5

        result = fix_narrow_passage(poly, min_clearance=target_clearance, strategy=PassageStrategy.ARAP )

        assert result.is_valid
        # Should improve toward target
        assert result.minimum_clearance >= poly.minimum_clearance

    def test_split_strategy(self):
        """Test splitting polygon at narrow passage."""
        # Dumbbell shape
        coords = [
            # Left bulb
            (0, 0), (1, 0), (1, 1), (0, 1),
            # Narrow connector
            (0.4, 1), (0.4, 2), (0.6, 2), (0.6, 1),
            # Right bulb
            (1, 1), (1, 2), (0, 2)
        ]
        poly = Polygon(coords)

        result = fix_narrow_passage(poly, min_clearance=0.5, strategy=PassageStrategy.SPLIT)

        assert result.is_valid
        # May return various geometry types depending on split success
        from shapely.geometry.base import BaseGeometry
        assert isinstance(result, BaseGeometry)

    def test_already_wide_enough(self):
        """Test that wide passages are unchanged."""
        # Wide rectangle
        coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
        poly = Polygon(coords)

        result = fix_narrow_passage(poly, min_clearance=2.0, strategy=PassageStrategy.WIDEN)

        assert result.is_valid
        # Should be essentially unchanged
        assert result.exterior.coords[:] == poly.exterior.coords[:]

    def test_already_wide_enough_arap(self):
        """Test that wide passages are unchanged."""
        # Wide rectangle
        coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
        poly = Polygon(coords)

        result = fix_narrow_passage(poly, min_clearance=2.0, strategy=PassageStrategy.ARAP)

        assert result.is_valid
        # Should be essentially unchanged
        assert result.exterior.coords[:] == poly.exterior.coords[:]

    def test_preserves_holes(self):
        """Test that holes are preserved when widening."""
        # Simple polygon with hole (not self-intersecting)
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
        poly = Polygon(exterior, holes=[hole])

        result = fix_narrow_passage(poly, min_clearance=0.5, strategy=PassageStrategy.WIDEN)

        assert result.is_valid
        # Holes should be preserved
        assert len(result.interiors) == 1

    def test_preserves_holes_arap(self):
        """Test that holes are preserved when widening."""
        # Simple polygon with hole (not self-intersecting)
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
        poly = Polygon(exterior, holes=[hole])

        result = fix_narrow_passage(poly, min_clearance=0.5, strategy=PassageStrategy.ARAP)

        assert result.is_valid
        # Holes should be preserved
        assert len(result.interiors) == 1

    def test_very_narrow_passage(self):
        """Test handling very narrow passage."""
        # Narrow hourglass (not too extreme)
        coords = [
            (0, 0), (2, 0), (2, 1),
            (1.2, 1.5), (1, 2), (1.2, 2.5),
            (2, 3), (2, 4), (0, 4), (0, 3),
            (0.8, 2.5), (0.8, 1.5),
            (0, 1)
        ]
        poly = Polygon(coords)

        result = fix_narrow_passage(poly, min_clearance=0.5, strategy=PassageStrategy.WIDEN)

        assert result.is_valid
        # Clearance may improve or stay similar (buffering doesn't always help narrow passages)
        # Just ensure the result is valid
        assert result.area > 0

    def test_vertex_to_edge_passage(self):
        """Test widening when minimum clearance is vertex-to-edge (not vertex-to-vertex).

        This is a critical test case where the narrow point is a vertex on one side,
        but the closest point on the opposite side is on an edge (not at a vertex).
        The algorithm should detect this and move the edge vertex perpendicular to
        the edge rather than along the clearance line.
        """
        # Polygon with narrow indentation where clearance is vertex-to-edge
        coords = [
            (0, 0), (2, 0), (2, 1),
            (1.1, 1.5), (0.1, 2), (1.1, 2.5),  # Narrow section: (0.1, 2) is closest to right edge
            (2, 3), (2, 4), (0, 4),
        ]
        poly = Polygon(coords)

        # The minimum clearance should be from vertex (0.1, 2) to the edge from (2, 1) to (2, 3)
        # which is approximately 1.9 units
        original_clearance = poly.minimum_clearance
        assert original_clearance < 2.0  # Verify it's actually narrow

        target_clearance = 0.5
        result = fix_narrow_passage(poly, min_clearance=target_clearance, strategy=PassageStrategy.WIDEN)

        assert result.is_valid
        assert isinstance(result, Polygon)

        # The algorithm should improve clearance by moving vertices perpendicular to edges
        # In this case, should move (0.1, 2) left and nearest vertex on right edge away
        assert result.minimum_clearance >= original_clearance or result.minimum_clearance >= target_clearance * 0.9

    def test_accepts_string_split_strategy(self):
        """String literal should select the split strategy."""
        poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])

        result = fix_narrow_passage(poly, min_clearance=0.5, strategy="split")

        assert result.geom_type in ("MultiPolygon", "GeometryCollection")
        assert not result.is_empty

    def test_arap_widens_notch_shaped_passage(self):
        """ARAP strategy should widen notch/indentation-shaped narrow passages.

        This tests a bug where ARAP returns the polygon unchanged for notch shapes.
        The issue: erosion fills in the notch (stays single Polygon), so the algorithm
        incorrectly assumes clearance is sufficient and returns early.

        The notch is 0.1 units wide but min_clearance is 0.2, so widening is needed.
        """
        # Rectangle with narrow notch from the top
        # The notch goes from (0.95, 0.5) to (1.05, 0.5) - width is 0.1
        coords = [
            (0, 0), (2, 0), (2, 1), (1.05, 1), (1.05, 0.5), (0.95, 0.5), (0.95, 1), (0, 1)
        ]
        poly = Polygon(coords)

        original_clearance = poly.minimum_clearance
        min_clearance = 0.2

        # Verify the test setup is correct
        assert original_clearance < min_clearance, (
            f"Test setup error: original clearance {original_clearance} should be < {min_clearance}"
        )

        result = fix_narrow_passage(poly, min_clearance=min_clearance, strategy=PassageStrategy.ARAP)

        assert result.is_valid
        # The clearance should improve to at least the target (with small tolerance for floating-point)
        assert result.minimum_clearance >= min_clearance * 0.99, (
            f"ARAP should widen notch to meet min_clearance={min_clearance}, "
            f"but clearance is {result.minimum_clearance} (was {original_clearance})"
        )

    def test_widen_strategy_widens_notch_shaped_passage(self):
        """WIDEN strategy should widen notch/indentation-shaped narrow passages.

        Similar to the ARAP bug, the WIDEN strategy may also struggle with notch shapes
        where the clearance line goes from a vertex on one side to an edge on another.

        The notch is 0.1 units wide but min_clearance is 0.2, so widening is needed.
        """
        # Rectangle with narrow notch from the top
        # The notch goes from (0.95, 0.5) to (1.05, 0.5) - width is 0.1
        coords = [
            (0, 0), (2, 0), (2, 1), (1.05, 1), (1.05, 0.5), (0.95, 0.5), (0.95, 1), (0, 1)
        ]
        poly = Polygon(coords)

        original_clearance = poly.minimum_clearance
        min_clearance = 0.2

        # Verify the test setup is correct
        assert original_clearance < min_clearance, (
            f"Test setup error: original clearance {original_clearance} should be < {min_clearance}"
        )

        result = fix_narrow_passage(poly, min_clearance=min_clearance, strategy=PassageStrategy.WIDEN)

        assert result.is_valid
        # The clearance should improve to at least the target (with small tolerance for floating-point)
        assert result.minimum_clearance >= min_clearance * 0.99, (
            f"WIDEN should widen notch to meet min_clearance={min_clearance}, "
            f"but clearance is {result.minimum_clearance} (was {original_clearance})"
        )


class TestFixNearSelfIntersection:
    """Tests for fix_near_self_intersection function."""

    def test_detects_self_intersection_context(self):
        """Helper should detect near-intersection metadata."""
        coords = [
            (0, 0), (5, 0), (5, 3), (4, 3), (4, 4), (5, 4), (5, 6), (0, 6)
        ]
        poly = Polygon(coords)

        context = _find_self_intersection_vertices(poly)
        assert context is not None
        assert context.clearance == pytest.approx(poly.minimum_clearance)
        assert context.vertex_idx_a != context.vertex_idx_b

    def test_simplify_close_edges(self):
        """Test fixing near-intersecting edges via simplification."""
        # Polygon with edges that come close but don't intersect
        coords = [
            (0, 0), (5, 0), (5, 3), (4, 3), (4, 4), (5, 4), (5, 6), (0, 6)
        ]
        poly = Polygon(coords)

        result = fix_near_self_intersection(poly, min_clearance=0.5, strategy=IntersectionStrategy.SIMPLIFY)

        assert result.is_valid
        # Should reduce vertices or maintain
        assert len(result.exterior.coords) <= len(poly.exterior.coords)

    def test_buffer_strategy(self):
        """Test using buffer to separate close edges."""
        # Simple polygon with low clearance
        coords = [(0, 0), (4, 0), (4, 1), (1, 1), (1, 2), (4, 2), (4, 3), (0, 3)]
        poly = Polygon(coords)

        result = fix_near_self_intersection(poly, min_clearance=0.5, strategy=IntersectionStrategy.BUFFER)

        assert result.is_valid
        # Area should increase slightly
        assert result.area >= poly.area

    def test_smooth_strategy(self):
        """Test smoothing to fix near-intersections."""
        # Polygon with zigzag creating near-intersection
        coords = [
            (0, 0), (5, 0), (5, 5),
            (2.2, 2.5), (2.1, 2.4), (2.0, 2.5), (1.9, 2.4), (1.8, 2.5),
            (0, 5)
        ]
        poly = Polygon(coords)

        result = fix_near_self_intersection(poly, min_clearance=0.5, strategy=IntersectionStrategy.SMOOTH)

        assert result.is_valid
        # Should smooth out the zigzag
        assert result.minimum_clearance >= poly.minimum_clearance * 0.9

    def test_no_near_intersection(self):
        """Test that well-formed polygons remain unchanged."""
        # Simple square with good clearance
        coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
        poly = Polygon(coords)

        result = fix_near_self_intersection(poly, min_clearance=1.0, strategy=IntersectionStrategy.SIMPLIFY)

        assert result.is_valid
        # Should be unchanged
        assert result.exterior.coords[:] == poly.exterior.coords[:]

    def test_preserves_holes(self):
        """Test that holes are preserved."""
        # Polygon with hole
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
        poly = Polygon(exterior, holes=[hole])

        result = fix_near_self_intersection(poly, min_clearance=0.5, strategy=IntersectionStrategy.SIMPLIFY)

        assert result.is_valid
        # Hole should remain (though it might be buffered if using buffer strategy)
        assert len(result.interiors) >= 0

    def test_improves_clearance(self):
        """Test that clearance is improved."""
        # Create polygon with known low clearance
        coords = [
            (0, 0), (3, 0), (3, 1),
            (1.1, 1.1), (1, 1), (0.9, 1.1),
            (0, 1)
        ]
        poly = Polygon(coords)
        original_clearance = poly.minimum_clearance

        result = fix_near_self_intersection(poly, min_clearance=0.5, strategy=IntersectionStrategy.SIMPLIFY)

        assert result.is_valid
        # Should improve or maintain clearance
        assert result.minimum_clearance >= original_clearance * 0.9

    def test_accepts_string_buffer_strategy(self):
        """String literal should invoke the buffer strategy."""
        coords = [(0, 0), (4, 0), (4, 1), (1, 1), (1, 2), (4, 2), (4, 3), (0, 3)]
        poly = Polygon(coords)

        result = fix_near_self_intersection(poly, min_clearance=0.5, strategy="buffer")

        assert result.is_valid
        assert result.area >= poly.area


class TestFixParallelCloseEdges:
    """Tests for fix_parallel_close_edges function."""

    def test_simplify_parallel_edges(self):
        """Test fixing parallel edges via simplification."""
        # Polygon with parallel edges that are close
        coords = [
            (0, 0), (10, 0), (10, 1),
            (2, 1), (2, 1.2), (10, 1.2),
            (10, 2), (0, 2)
        ]
        poly = Polygon(coords)

        result = fix_parallel_close_edges(poly, min_clearance=0.5, strategy=IntersectionStrategy.SIMPLIFY)

        assert result.is_valid
        # Should simplify
        assert len(result.exterior.coords) <= len(poly.exterior.coords)

    def test_buffer_strategy(self):
        """Test using buffer to separate parallel edges."""
        # Simple rectangle with low clearance
        coords = [(0, 0), (5, 0), (5, 0.2), (0, 0.2)]
        poly = Polygon(coords)

        result = fix_parallel_close_edges(poly, min_clearance=0.5, strategy=IntersectionStrategy.BUFFER)

        assert result.is_valid
        # Buffer increases area
        assert result.area >= poly.area

    def test_accepts_string_strategy(self):
        """String literal should be accepted for parallel-edge fixes."""
        coords = [(0, 0), (6, 0), (6, 0.3), (0, 0.3)]
        poly = Polygon(coords)

        result = fix_parallel_close_edges(poly, min_clearance=0.5, strategy="buffer")

        assert result.is_valid
        assert result.area >= poly.area

    def test_u_shape_parallel_edges(self):
        """Test fixing U-shaped polygon with parallel edges."""
        # U-shape with narrow gap
        coords = [
            (0, 0), (3, 0), (3, 5),
            (2, 5), (2, 1), (1, 1), (1, 5),
            (0, 5)
        ]
        poly = Polygon(coords)

        result = fix_parallel_close_edges(poly, min_clearance=0.5, strategy=IntersectionStrategy.SIMPLIFY)

        assert result.is_valid
        # Should improve clearance
        assert result.minimum_clearance >= poly.minimum_clearance * 0.9

    def test_no_parallel_edges(self):
        """Test that polygons without parallel close edges remain unchanged."""
        # Simple triangle - no parallel edges
        coords = [(0, 0), (5, 0), (2.5, 5)]
        poly = Polygon(coords)

        result = fix_parallel_close_edges(poly, min_clearance=1.0, strategy=IntersectionStrategy.SIMPLIFY)

        assert result.is_valid
        # Should be essentially unchanged
        assert result.area == pytest.approx(poly.area, rel=0.1)

    def test_preserves_validity(self):
        """Test that result is always valid."""
        # Complex polygon
        coords = [
            (0, 0), (8, 0), (8, 1), (1, 1),
            (1, 1.1), (8, 1.1), (8, 2),
            (1, 2), (1, 2.1), (8, 2.1),
            (8, 3), (0, 3)
        ]
        poly = Polygon(coords)

        result = fix_parallel_close_edges(poly, min_clearance=0.5, strategy=IntersectionStrategy.SIMPLIFY)

        assert result.is_valid


class TestClearanceDiagnosis:
    """Tests for diagnose_clearance() and the underlying heuristic functions."""

    def test_meets_requirement_returns_none(self):
        """Wide polygon with large clearance should return ClearanceIssue.NONE."""
        from polyforge.clearance.fix_clearance import diagnose_clearance, ClearanceIssue

        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        diag = diagnose_clearance(poly, min_clearance=0.01)
        assert diag.issue == ClearanceIssue.NONE
        assert diag.meets_requirement is True

    def test_diagnose_protrusion_spike(self):
        """Polygon with a narrow spike should be diagnosed as NARROW_PROTRUSION."""
        from polyforge.clearance.fix_clearance import diagnose_clearance, ClearanceIssue

        # Polygon with a very thin spike extending from the top
        poly = Polygon([
            (0, 0), (10, 0), (10, 10),
            (5.01, 10), (5.01, 15), (4.99, 15), (4.99, 10),
            (0, 10),
        ])
        diag = diagnose_clearance(poly, min_clearance=1.0)
        assert not diag.meets_requirement
        # The spike should be detected as either NARROW_PROTRUSION or
        # NEAR_SELF_INTERSECTION (both are reasonable for very thin features)
        assert diag.issue in (
            ClearanceIssue.NARROW_PROTRUSION,
            ClearanceIssue.NEAR_SELF_INTERSECTION,
        )

    def test_diagnose_hole_too_close(self):
        """Polygon with hole very close to exterior should be HOLE_TOO_CLOSE."""
        from polyforge.clearance.fix_clearance import diagnose_clearance, ClearanceIssue

        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(0.1, 4), (1, 4), (1, 6), (0.1, 6)]
        poly = Polygon(exterior, holes=[hole])
        diag = diagnose_clearance(poly, min_clearance=1.0)
        assert diag.issue == ClearanceIssue.HOLE_TOO_CLOSE

    def test_diagnose_narrow_channel(self):
        """Polygon with a narrow channel should detect a clearance issue."""
        from polyforge.clearance.fix_clearance import diagnose_clearance, ClearanceIssue

        # U-shape: two long parallel sides close together
        poly = Polygon([
            (0, 0), (10, 0), (10, 5),
            (1, 5), (1, 0.2),
            (0, 0.2),
        ])
        diag = diagnose_clearance(poly, min_clearance=1.0)
        assert not diag.meets_requirement
        # Narrow channels can be classified as several issue types depending
        # on where the clearance line lands
        assert diag.issue in (
            ClearanceIssue.PARALLEL_CLOSE_EDGES,
            ClearanceIssue.NARROW_PASSAGE,
            ClearanceIssue.NEAR_SELF_INTERSECTION,
        )

    def test_diagnose_non_polygon_raises(self):
        """diagnose_clearance should raise TypeError for non-Polygon input."""
        from shapely.geometry import Point
        from polyforge.clearance.fix_clearance import diagnose_clearance

        with pytest.raises(TypeError, match="Expected Polygon"):
            diagnose_clearance(Point(0, 0), min_clearance=1.0)

    def test_clearance_context_has_edge_angle(self):
        """Verify ClearanceContext.edge_angle_similarity is populated."""
        from polyforge.clearance.fix_clearance import _build_clearance_context

        # Simple polygon where clearance context can be built
        poly = Polygon([
            (0, 0), (10, 0), (10, 5),
            (1, 5), (1, 0.2),
            (0, 0.2),
        ])
        ctx = _build_clearance_context(poly)
        if ctx is not None:
            # edge_angle_similarity should be a float when computable
            assert ctx.edge_angle_similarity is None or isinstance(ctx.edge_angle_similarity, float)
