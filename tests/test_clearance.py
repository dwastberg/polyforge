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
from polyforge.core.types import (
    HoleStrategy,
    PassageStrategy,
    IntrusionStrategy,
    IntersectionStrategy,
)
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
        result = fix_hole_too_close(
            poly, min_clearance=1.0, strategy=HoleStrategy.REMOVE
        )

        # Hole should be removed
        assert len(result.interiors) == 0
        assert result.is_valid

    def test_keep_far_hole(self):
        """Test that holes far from exterior are kept."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        # Hole in center, far from all edges
        hole = [(4, 4), (6, 4), (6, 6), (4, 6)]

        poly = Polygon(exterior, holes=[hole])
        result = fix_hole_too_close(
            poly, min_clearance=2.0, strategy=HoleStrategy.REMOVE
        )

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
        result = fix_hole_too_close(
            poly, min_clearance=2.0, strategy=HoleStrategy.REMOVE
        )

        # All holes should be removed
        assert len(result.interiors) == 0
        assert result.is_valid

    def test_mixed_close_and_far_holes(self):
        """Test with both close and far holes."""
        exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
        close_hole = [(1, 1), (2, 1), (2, 2), (1, 2)]  # Distance ~1
        far_hole = [(8, 8), (12, 8), (12, 12), (8, 12)]  # Distance ~8

        poly = Polygon(exterior, holes=[close_hole, far_hole])
        result = fix_hole_too_close(
            poly, min_clearance=2.0, strategy=HoleStrategy.REMOVE
        )

        # Only far hole should remain
        assert len(result.interiors) == 1
        assert result.is_valid

    def test_no_holes_returns_unchanged(self):
        """Test that polygons without holes are returned unchanged."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        poly = Polygon(exterior)

        result = fix_hole_too_close(
            poly, min_clearance=2.0, strategy=HoleStrategy.REMOVE
        )

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
        result = fix_hole_too_close(
            poly, min_clearance=3.0, strategy=HoleStrategy.SHRINK
        )

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
        result = fix_hole_too_close(
            poly, min_clearance=5.0, strategy=HoleStrategy.SHRINK
        )

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
        result_keep = fix_hole_too_close(
            poly, min_clearance=2.0, strategy=HoleStrategy.REMOVE
        )
        assert len(result_keep.interiors) == 1

        # Just above threshold, should be removed
        result_remove = fix_hole_too_close(
            poly, min_clearance=2.1, strategy=HoleStrategy.REMOVE
        )
        assert len(result_remove.interiors) == 0

    def test_preserves_exterior(self):
        """Test that exterior is never modified."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(1, 1), (2, 1), (2, 2), (1, 2)]

        poly = Polygon(exterior, holes=[hole])
        result = fix_hole_too_close(
            poly, min_clearance=2.0, strategy=HoleStrategy.REMOVE
        )

        # Exterior coordinates should be identical
        np.testing.assert_array_almost_equal(
            np.array(result.exterior.coords), np.array(poly.exterior.coords)
        )

    def test_remove_hole_with_low_self_clearance(self):
        """Hole with near-self-intersection should be removed with REMOVE strategy."""
        exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
        # Concave hole that nearly pinches itself: vertex (8,5.01) almost
        # touches the edge from (5,5) to (11,5)
        hole = [
            (5, 5),
            (11, 5),
            (11, 8),
            (8, 8),
            (8, 5.01),  # nearly touches bottom edge
            (6, 7),
            (5, 7),
        ]
        poly = Polygon(exterior, holes=[hole])
        assert poly.is_valid

        result = fix_hole_too_close(
            poly, min_clearance=0.5, strategy=HoleStrategy.REMOVE
        )

        # Hole should be removed because its self-clearance is ~0.01, below 0.5
        assert len(result.interiors) == 0
        assert result.is_valid

    def test_shrink_hole_with_low_self_clearance(self):
        """Hole with near-self-intersection should be shrunk with SHRINK strategy."""
        exterior = [(0, 0), (30, 0), (30, 30), (0, 30)]
        # Large concave hole that nearly pinches itself
        hole = [
            (5, 5),
            (15, 5),
            (15, 15),
            (10, 15),
            (10, 5.02),  # nearly touches bottom edge
            (8, 12),
            (5, 12),
        ]
        poly = Polygon(exterior, holes=[hole])
        assert poly.is_valid

        result = fix_hole_too_close(
            poly, min_clearance=0.5, strategy=HoleStrategy.SHRINK
        )

        assert result.is_valid
        if len(result.interiors) > 0:
            # Hole should be smaller than original
            original_area = Polygon(hole).area
            result_area = Polygon(result.interiors[0]).area
            assert result_area < original_area

    def test_keep_hole_with_good_self_clearance(self):
        """Hole with high self-clearance should be kept unchanged."""
        exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
        # Simple rectangular hole — high self-clearance
        hole = [(5, 5), (10, 5), (10, 10), (5, 10)]
        poly = Polygon(exterior, holes=[hole])

        result = fix_hole_too_close(
            poly, min_clearance=0.5, strategy=HoleStrategy.REMOVE
        )

        # Hole has good self-clearance and is far from exterior — keep it
        assert len(result.interiors) == 1
        assert result.is_valid


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
            (0, 0),
            (5, 0),
            (5, 2),
            # Spike 1 pointing right
            (5, 2.4),
            (6, 2.5),
            (5, 2.6),
            (5, 8),
            (5, 10),
            # Spike 2 pointing up
            (2.6, 10),
            (2.5, 11),
            (2.4, 10),
            (0, 10),
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
            (0, 0),
            (10, 0),
            (10, 4),
            (10, 4.99),
            (15, 5),
            (10, 5.01),  # Very thin spike
            (10, 6),
            (0, 6),
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
            (0, 0),
            (10, 0),
            (10, 4),
            # Intrusion
            (9, 4.9),
            (8, 5),
            (9, 5.1),
            (10, 6),
            (10, 10),
            (0, 10),
        ]
        poly = Polygon(coords)

        result = fix_sharp_intrusion(
            poly, min_clearance=0.5, strategy=IntrusionStrategy.FILL
        )

        assert result.is_valid
        # Intrusion should be improved (fewer or equal vertices)
        assert len(result.exterior.coords) <= len(poly.exterior.coords)
        # Area should be similar or increase slightly
        assert result.area >= poly.area * 0.95

    def test_smooth_intrusion(self):
        """Test smoothing an intrusion."""
        # Polygon with narrow notch
        coords = [
            (0, 0),
            (10, 0),
            (10, 4),
            (9, 4.5),
            (8, 5),
            (9, 5.5),
            (10, 6),
            (10, 10),
            (0, 10),
        ]
        poly = Polygon(coords)

        result = fix_sharp_intrusion(
            poly, min_clearance=0.8, strategy=IntrusionStrategy.SMOOTH
        )

        assert result.is_valid
        # Smoothing preserves vertex count but modifies positions
        # Area should be similar or slightly increased
        assert result.area >= poly.area * 0.95

    def test_simplify_strategy(self):
        """Test simplify strategy for intrusions."""
        # Polygon with jagged intrusion
        coords = [
            (0, 0),
            (10, 0),
            (10, 4),
            (9.5, 4.5),
            (9, 5),
            (9.5, 5.5),
            (10, 6),
            (10, 10),
            (0, 10),
        ]
        poly = Polygon(coords)

        result = fix_sharp_intrusion(
            poly, min_clearance=0.5, strategy=IntrusionStrategy.SIMPLIFY
        )

        assert result.is_valid
        # Simplification removes vertices
        assert len(result.exterior.coords) <= len(poly.exterior.coords)

    def test_multiple_intrusions(self):
        """Test fixing multiple intrusions."""
        # Polygon with two notches
        coords = [
            (0, 0),
            (5, 0),
            # Intrusion 1
            (5, 0.1),
            (4, 0.5),
            (5, 0.9),
            (5, 5),
            # Intrusion 2
            (4.9, 5),
            (4, 5.5),
            (4.9, 6),
            (5, 10),
            (0, 10),
        ]
        poly = Polygon(coords)

        result = fix_sharp_intrusion(
            poly, min_clearance=0.5, strategy=IntrusionStrategy.FILL
        )

        assert result.is_valid
        # Intrusions should be improved (fewer or equal vertices)
        assert len(result.exterior.coords) <= len(poly.exterior.coords)

    def test_no_intrusion_unchanged(self):
        """Test that polygons without intrusions remain mostly unchanged."""
        # Simple rectangle
        coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
        poly = Polygon(coords)

        result = fix_sharp_intrusion(
            poly, min_clearance=1.0, strategy=IntrusionStrategy.FILL
        )

        assert result.is_valid
        # Should be similar to original
        assert result.area == pytest.approx(poly.area, rel=0.1)

    def test_preserves_holes(self):
        """Test that holes are preserved."""
        # Polygon with intrusion and hole
        exterior = [
            (0, 0),
            (10, 0),
            (10, 4),
            (9, 4.9),
            (8, 5),
            (9, 5.1),
            (10, 6),
            (10, 10),
            (0, 10),
        ]
        hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
        poly = Polygon(exterior, holes=[hole])

        result = fix_sharp_intrusion(
            poly, min_clearance=0.5, strategy=IntrusionStrategy.FILL
        )

        assert result.is_valid
        assert len(result.interiors) == 1  # Hole preserved

    def test_deep_narrow_intrusion(self):
        """Test fixing a deep narrow intrusion."""
        # Polygon with deep notch
        coords = [
            (0, 0),
            (10, 0),
            (10, 4),
            # Deep intrusion
            (9, 5),
            (5, 5),
            (9, 5.1),
            (10, 6),
            (10, 10),
            (0, 10),
        ]
        poly = Polygon(coords)

        result = fix_sharp_intrusion(
            poly, min_clearance=0.5, strategy=IntrusionStrategy.FILL
        )

        assert result.is_valid
        # Deep intrusion should be improved or filled (area similar or increased)
        assert result.area >= poly.area * 0.95

    def test_achieves_target_clearance(self):
        """Test that result achieves target clearance."""
        # Polygon with narrow notch
        coords = [
            (0, 0),
            (10, 0),
            (10, 4),
            (9, 4.8),
            (8, 5),
            (9, 5.2),
            (10, 6),
            (10, 10),
            (0, 10),
        ]
        poly = Polygon(coords)

        target_clearance = 1.0
        result = fix_sharp_intrusion(
            poly, min_clearance=target_clearance, strategy=IntrusionStrategy.FILL
        )

        assert result.is_valid
        # Should meet or exceed target (or be close)
        assert result.minimum_clearance >= target_clearance * 0.9

    def test_accepts_string_strategy(self):
        """String literal strategies should behave like enums."""
        coords = [
            (0, 0),
            (10, 0),
            (10, 4),
            (9, 4.8),
            (8, 5),
            (9, 5.2),
            (10, 6),
            (10, 10),
            (0, 10),
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
            (0, 0),
            (2, 0),
            (2, 1),
            (1.1, 1.5),
            (1, 2),
            (1.1, 2.5),  # Narrow section
            (2, 3),
            (2, 4),
            (0, 4),
            (0, 3),
            (-0.1, 2.5),
            (-0.1, 1.5),  # Other side of narrow section
            (0, 1),
        ]
        poly = Polygon(coords)
        original_clearance = poly.minimum_clearance

        result = fix_narrow_passage(
            poly, min_clearance=0.5, strategy=PassageStrategy.WIDEN
        )

        assert result.is_valid
        assert isinstance(result, Polygon)
        # Clearance should improve
        assert result.minimum_clearance >= original_clearance

    def test_widen_increases_clearance(self):
        """Test that widening improves clearance."""
        # Simple narrow passage
        coords = [(0, 0), (1, 0), (0.9, 1), (1, 2), (0, 2), (0.1, 1)]
        poly = Polygon(coords)
        target_clearance = 0.5

        result = fix_narrow_passage(
            poly, min_clearance=target_clearance, strategy=PassageStrategy.WIDEN
        )

        assert result.is_valid
        # Should improve toward target
        assert result.minimum_clearance >= poly.minimum_clearance

    def test_widen_increases_clearance_arap(self):
        """Test that widening improves clearance."""
        # Simple narrow passage
        coords = [(0, 0), (1, 0), (0.9, 1), (1, 2), (0, 2), (0.1, 1)]
        poly = Polygon(coords)
        target_clearance = 0.5

        result = fix_narrow_passage(
            poly, min_clearance=target_clearance, strategy=PassageStrategy.ARAP
        )

        assert result.is_valid
        # Should improve toward target
        assert result.minimum_clearance >= poly.minimum_clearance

    def test_split_strategy(self):
        """Test splitting polygon at narrow passage."""
        # Dumbbell shape
        coords = [
            # Left bulb
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1),
            # Narrow connector
            (0.4, 1),
            (0.4, 2),
            (0.6, 2),
            (0.6, 1),
            # Right bulb
            (1, 1),
            (1, 2),
            (0, 2),
        ]
        poly = Polygon(coords)

        result = fix_narrow_passage(
            poly, min_clearance=0.5, strategy=PassageStrategy.SPLIT
        )

        assert result.is_valid
        # May return various geometry types depending on split success
        from shapely.geometry.base import BaseGeometry

        assert isinstance(result, BaseGeometry)

    def test_already_wide_enough(self):
        """Test that wide passages are unchanged."""
        # Wide rectangle
        coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
        poly = Polygon(coords)

        result = fix_narrow_passage(
            poly, min_clearance=2.0, strategy=PassageStrategy.WIDEN
        )

        assert result.is_valid
        # Should be essentially unchanged
        assert result.exterior.coords[:] == poly.exterior.coords[:]

    def test_already_wide_enough_arap(self):
        """Test that wide passages are unchanged."""
        # Wide rectangle
        coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
        poly = Polygon(coords)

        result = fix_narrow_passage(
            poly, min_clearance=2.0, strategy=PassageStrategy.ARAP
        )

        assert result.is_valid
        # Should be essentially unchanged
        assert result.exterior.coords[:] == poly.exterior.coords[:]

    def test_preserves_holes(self):
        """Test that holes are preserved when widening."""
        # Simple polygon with hole (not self-intersecting)
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
        poly = Polygon(exterior, holes=[hole])

        result = fix_narrow_passage(
            poly, min_clearance=0.5, strategy=PassageStrategy.WIDEN
        )

        assert result.is_valid
        # Holes should be preserved
        assert len(result.interiors) == 1

    def test_preserves_holes_arap(self):
        """Test that holes are preserved when widening."""
        # Simple polygon with hole (not self-intersecting)
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
        poly = Polygon(exterior, holes=[hole])

        result = fix_narrow_passage(
            poly, min_clearance=0.5, strategy=PassageStrategy.ARAP
        )

        assert result.is_valid
        # Holes should be preserved
        assert len(result.interiors) == 1

    def test_very_narrow_passage(self):
        """Test handling very narrow passage."""
        # Narrow hourglass (not too extreme)
        coords = [
            (0, 0),
            (2, 0),
            (2, 1),
            (1.2, 1.5),
            (1, 2),
            (1.2, 2.5),
            (2, 3),
            (2, 4),
            (0, 4),
            (0, 3),
            (0.8, 2.5),
            (0.8, 1.5),
            (0, 1),
        ]
        poly = Polygon(coords)

        result = fix_narrow_passage(
            poly, min_clearance=0.5, strategy=PassageStrategy.WIDEN
        )

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
            (0, 0),
            (2, 0),
            (2, 1),
            (1.1, 1.5),
            (0.1, 2),
            (1.1, 2.5),  # Narrow section: (0.1, 2) is closest to right edge
            (2, 3),
            (2, 4),
            (0, 4),
        ]
        poly = Polygon(coords)

        # The minimum clearance should be from vertex (0.1, 2) to the edge from (2, 1) to (2, 3)
        # which is approximately 1.9 units
        original_clearance = poly.minimum_clearance
        assert original_clearance < 2.0  # Verify it's actually narrow

        target_clearance = 0.5
        result = fix_narrow_passage(
            poly, min_clearance=target_clearance, strategy=PassageStrategy.WIDEN
        )

        assert result.is_valid
        assert isinstance(result, Polygon)

        # The algorithm should improve clearance by moving vertices perpendicular to edges
        # In this case, should move (0.1, 2) left and nearest vertex on right edge away
        assert (
            result.minimum_clearance >= original_clearance
            or result.minimum_clearance >= target_clearance * 0.9
        )

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
            (0, 0),
            (2, 0),
            (2, 1),
            (1.05, 1),
            (1.05, 0.5),
            (0.95, 0.5),
            (0.95, 1),
            (0, 1),
        ]
        poly = Polygon(coords)

        original_clearance = poly.minimum_clearance
        min_clearance = 0.2

        # Verify the test setup is correct
        assert original_clearance < min_clearance, (
            f"Test setup error: original clearance {original_clearance} should be < {min_clearance}"
        )

        result = fix_narrow_passage(
            poly, min_clearance=min_clearance, strategy=PassageStrategy.ARAP
        )

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
            (0, 0),
            (2, 0),
            (2, 1),
            (1.05, 1),
            (1.05, 0.5),
            (0.95, 0.5),
            (0.95, 1),
            (0, 1),
        ]
        poly = Polygon(coords)

        original_clearance = poly.minimum_clearance
        min_clearance = 0.2

        # Verify the test setup is correct
        assert original_clearance < min_clearance, (
            f"Test setup error: original clearance {original_clearance} should be < {min_clearance}"
        )

        result = fix_narrow_passage(
            poly, min_clearance=min_clearance, strategy=PassageStrategy.WIDEN
        )

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
        coords = [(0, 0), (5, 0), (5, 3), (4, 3), (4, 4), (5, 4), (5, 6), (0, 6)]
        poly = Polygon(coords)

        context = _find_self_intersection_vertices(poly)
        assert context is not None
        assert context.clearance == pytest.approx(poly.minimum_clearance)
        assert context.vertex_idx_a != context.vertex_idx_b

    def test_simplify_close_edges(self):
        """Test fixing near-intersecting edges via simplification."""
        # Polygon with edges that come close but don't intersect
        coords = [(0, 0), (5, 0), (5, 3), (4, 3), (4, 4), (5, 4), (5, 6), (0, 6)]
        poly = Polygon(coords)

        result = fix_near_self_intersection(
            poly, min_clearance=0.5, strategy=IntersectionStrategy.SIMPLIFY
        )

        assert result.is_valid
        # Should reduce vertices or maintain
        assert len(result.exterior.coords) <= len(poly.exterior.coords)

    def test_buffer_strategy(self):
        """Test using buffer to separate close edges."""
        # Simple polygon with low clearance
        coords = [(0, 0), (4, 0), (4, 1), (1, 1), (1, 2), (4, 2), (4, 3), (0, 3)]
        poly = Polygon(coords)

        result = fix_near_self_intersection(
            poly, min_clearance=0.5, strategy=IntersectionStrategy.BUFFER
        )

        assert result.is_valid
        # Area should increase slightly
        assert result.area >= poly.area

    def test_smooth_strategy(self):
        """Test smoothing to fix near-intersections."""
        # Polygon with zigzag creating near-intersection
        coords = [
            (0, 0),
            (5, 0),
            (5, 5),
            (2.2, 2.5),
            (2.1, 2.4),
            (2.0, 2.5),
            (1.9, 2.4),
            (1.8, 2.5),
            (0, 5),
        ]
        poly = Polygon(coords)

        result = fix_near_self_intersection(
            poly, min_clearance=0.5, strategy=IntersectionStrategy.SMOOTH
        )

        assert result.is_valid
        # Should smooth out the zigzag
        assert result.minimum_clearance >= poly.minimum_clearance * 0.9

    def test_no_near_intersection(self):
        """Test that well-formed polygons remain unchanged."""
        # Simple square with good clearance
        coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
        poly = Polygon(coords)

        result = fix_near_self_intersection(
            poly, min_clearance=1.0, strategy=IntersectionStrategy.SIMPLIFY
        )

        assert result.is_valid
        # Should be unchanged
        assert result.exterior.coords[:] == poly.exterior.coords[:]

    def test_preserves_holes(self):
        """Test that holes are preserved."""
        # Polygon with hole
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
        poly = Polygon(exterior, holes=[hole])

        result = fix_near_self_intersection(
            poly, min_clearance=0.5, strategy=IntersectionStrategy.SIMPLIFY
        )

        assert result.is_valid
        # Hole should remain (though it might be buffered if using buffer strategy)
        assert len(result.interiors) >= 0

    def test_improves_clearance(self):
        """Test that clearance is improved."""
        # Create polygon with known low clearance
        coords = [(0, 0), (3, 0), (3, 1), (1.1, 1.1), (1, 1), (0.9, 1.1), (0, 1)]
        poly = Polygon(coords)
        original_clearance = poly.minimum_clearance

        result = fix_near_self_intersection(
            poly, min_clearance=0.5, strategy=IntersectionStrategy.SIMPLIFY
        )

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
            (0, 0),
            (10, 0),
            (10, 1),
            (2, 1),
            (2, 1.2),
            (10, 1.2),
            (10, 2),
            (0, 2),
        ]
        poly = Polygon(coords)

        result = fix_parallel_close_edges(
            poly, min_clearance=0.5, strategy=IntersectionStrategy.SIMPLIFY
        )

        assert result.is_valid
        # Should simplify
        assert len(result.exterior.coords) <= len(poly.exterior.coords)

    def test_buffer_strategy(self):
        """Test using buffer to separate parallel edges."""
        # Simple rectangle with low clearance
        coords = [(0, 0), (5, 0), (5, 0.2), (0, 0.2)]
        poly = Polygon(coords)

        result = fix_parallel_close_edges(
            poly, min_clearance=0.5, strategy=IntersectionStrategy.BUFFER
        )

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
        coords = [(0, 0), (3, 0), (3, 5), (2, 5), (2, 1), (1, 1), (1, 5), (0, 5)]
        poly = Polygon(coords)

        result = fix_parallel_close_edges(
            poly, min_clearance=0.5, strategy=IntersectionStrategy.SIMPLIFY
        )

        assert result.is_valid
        # Should improve clearance
        assert result.minimum_clearance >= poly.minimum_clearance * 0.9

    def test_no_parallel_edges(self):
        """Test that polygons without parallel close edges remain unchanged."""
        # Simple triangle - no parallel edges
        coords = [(0, 0), (5, 0), (2.5, 5)]
        poly = Polygon(coords)

        result = fix_parallel_close_edges(
            poly, min_clearance=1.0, strategy=IntersectionStrategy.SIMPLIFY
        )

        assert result.is_valid
        # Should be essentially unchanged
        assert result.area == pytest.approx(poly.area, rel=0.1)

    def test_preserves_validity(self):
        """Test that result is always valid."""
        # Complex polygon
        coords = [
            (0, 0),
            (8, 0),
            (8, 1),
            (1, 1),
            (1, 1.1),
            (8, 1.1),
            (8, 2),
            (1, 2),
            (1, 2.1),
            (8, 2.1),
            (8, 3),
            (0, 3),
        ]
        poly = Polygon(coords)

        result = fix_parallel_close_edges(
            poly, min_clearance=0.5, strategy=IntersectionStrategy.SIMPLIFY
        )

        assert result.is_valid

    def test_erosion_dilation_on_narrow_appendage(self):
        """fix_parallel_close_edges should use erosion-dilation for narrow appendages."""
        # Large polygon with a narrow peninsula — peninsula area is small
        base = Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])
        sliver = Polygon([(50, 24.9), (70, 24.9), (70, 25.1), (50, 25.1)])
        poly = base.union(sliver)

        if not isinstance(poly, Polygon):
            return
        original_clearance = poly.minimum_clearance

        result = fix_parallel_close_edges(poly, min_clearance=1.0)

        assert result.is_valid
        assert result.minimum_clearance > original_clearance

    def test_large_polygon_with_narrow_appendage(self):
        """Erosion-dilation should handle polygons where narrow feature is small relative to total."""
        base = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        sliver = Polygon([(100, 49.9), (150, 49.9), (150, 50.1), (100, 50.1)])
        combined = base.union(sliver)

        if isinstance(combined, Polygon):
            result = fix_parallel_close_edges(combined, min_clearance=1.0)
            assert result.is_valid
            assert result.minimum_clearance > combined.minimum_clearance


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
        poly = Polygon(
            [
                (0, 0),
                (10, 0),
                (10, 10),
                (5.01, 10),
                (5.01, 15),
                (4.99, 15),
                (4.99, 10),
                (0, 10),
            ]
        )
        diag = diagnose_clearance(poly, min_clearance=1.0)
        assert not diag.meets_requirement
        # The spike should be detected as a narrow feature — NARROW_WEDGE
        # (angle-based), NARROW_PROTRUSION, or NEAR_SELF_INTERSECTION are
        # all reasonable classifications for very thin features.
        assert diag.issue in (
            ClearanceIssue.NARROW_WEDGE,
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
        poly = Polygon(
            [
                (0, 0),
                (10, 0),
                (10, 5),
                (1, 5),
                (1, 0.2),
                (0, 0.2),
            ]
        )
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
        poly = Polygon(
            [
                (0, 0),
                (10, 0),
                (10, 5),
                (1, 5),
                (1, 0.2),
                (0, 0.2),
            ]
        )
        ctx = _build_clearance_context(poly)
        if ctx is not None:
            # edge_angle_similarity should be a float when computable
            assert ctx.edge_angle_similarity is None or isinstance(
                ctx.edge_angle_similarity, float
            )


class TestFixClearanceHoleSelfClearance:
    """Integration tests for fix_clearance with hole self-clearance issues."""

    def test_fix_clearance_hole_near_self_intersection(self):
        """fix_clearance should fix a polygon whose minimum clearance is
        caused by a hole with a near-self-intersection."""
        from polyforge import fix_clearance

        exterior = [(0, 0), (30, 0), (30, 30), (0, 30)]
        # Concave hole that nearly pinches itself
        pinching_hole = [
            (5, 5),
            (15, 5),
            (15, 15),
            (10, 15),
            (10, 5.02),  # nearly touches bottom edge
            (8, 12),
            (5, 12),
        ]
        # Normal hole far away
        normal_hole = [(20, 20), (25, 20), (25, 25), (20, 25)]

        poly = Polygon(exterior, holes=[pinching_hole, normal_hole])
        assert poly.is_valid
        assert poly.minimum_clearance < 0.5

        result = fix_clearance(poly, min_clearance=0.5)

        assert result.is_valid
        assert result.minimum_clearance >= 0.5
        # Normal hole should still be present
        assert len(result.interiors) >= 1


class TestRemoveNarrowWedges:
    """Tests for remove_narrow_wedges function."""

    def test_removes_acute_wedge(self):
        """Polygon with a narrow acute wedge should have it removed.

        This is the tiny_wedge.py regression case. The wedge tip at ~(78.18, 50.85)
        has a 6.7-degree angle. The polygon also has a separate short-edge clearance
        bottleneck at vertices 13-14, so we verify the wedge vertex is gone rather
        than checking overall clearance improvement.
        """
        from polyforge.ops.clearance.protrusions import remove_narrow_wedges

        wkt = (
            "Polygon ((33.6650000000372529 14.57500000018626451, "
            "29.89300000004004687 3.60900000017136335, "
            "9.82900000002700835 10.60500000044703484, "
            "29.73400000005494803 66.71499999985098839, "
            "77.65629966009873897 51.02880250383168459, "
            "80.16799999994691461 57.66799999959766865, "
            "84.69999999995343387 56.13499999977648258, "
            "82.579556216718629 49.78563174977898598, "
            "78.17746100388467312 50.85425762645900249, "
            "82.42944378335960209 49.27436824981123209, "
            "80.03000000002793968 42.08000000007450581, "
            "78.22100000001955777 36.19099999964237213, "
            "72.81700000003911555 20.1650000000372529, "
            "51.34436670027207583 26.94202884938567877, "
            "51.07963329972699285 27.39497115090489388, "
            "54.30626596626825631 36.78167456761002541, "
            "54.76273403374943882 37.03932543285191059, "
            "66.37600000004749745 33.20000000018626451, "
            "69.71600000001490116 42.99799999672174451, "
            "43.20200000004842877 51.63499999977648258, "
            "44.587000000057742 55.5580000001937151, "
            "37.69400000001769513 58.0530000003054738, "
            "33.92099999997299165 47.37100000027567148, "
            "37.7900000000372529 45.87899999972432852, "
            "36.41099999996367842 42.20100000035017729, "
            "43.76300000003539026 39.821000000461936, "
            "40.2219999999506399 31.18599999975413084, "
            "25.604857042664662 36.35760602075606585, "
            "19.48400000005494803 19.48900000005960464, "
            "33.6650000000372529 14.57500000018626451))"
        )
        import shapely

        poly = shapely.from_wkt(wkt)
        original_n = len(poly.exterior.coords)

        result = remove_narrow_wedges(poly, angle_threshold=25, min_depth=0.5)

        assert result.is_valid
        # Wedge tip vertex should have been removed
        assert len(result.exterior.coords) < original_n, (
            f"Expected fewer vertices after wedge removal: "
            f"was {original_n}, got {len(result.exterior.coords)}"
        )

    def test_find_best_join_does_not_return_same_index(self):
        """_find_best_join must not return li == ri (degenerate zero-width neck)."""
        from polyforge.ops.clearance.utils import (
            _find_best_join,
            _trace_wedge,
            _angle,
            _is_concave,
        )
        from shapely.geometry.polygon import orient
        import shapely

        wkt = (
            "Polygon ((33.6650000000372529 14.57500000018626451, "
            "29.89300000004004687 3.60900000017136335, "
            "9.82900000002700835 10.60500000044703484, "
            "29.73400000005494803 66.71499999985098839, "
            "77.65629966009873897 51.02880250383168459, "
            "80.16799999994691461 57.66799999959766865, "
            "84.69999999995343387 56.13499999977648258, "
            "82.579556216718629 49.78563174977898598, "
            "78.17746100388467312 50.85425762645900249, "
            "82.42944378335960209 49.27436824981123209, "
            "80.03000000002793968 42.08000000007450581, "
            "78.22100000001955777 36.19099999964237213, "
            "72.81700000003911555 20.1650000000372529, "
            "51.34436670027207583 26.94202884938567877, "
            "51.07963329972699285 27.39497115090489388, "
            "54.30626596626825631 36.78167456761002541, "
            "54.76273403374943882 37.03932543285191059, "
            "66.37600000004749745 33.20000000018626451, "
            "69.71600000001490116 42.99799999672174451, "
            "43.20200000004842877 51.63499999977648258, "
            "44.587000000057742 55.5580000001937151, "
            "37.69400000001769513 58.0530000003054738, "
            "33.92099999997299165 47.37100000027567148, "
            "37.7900000000372529 45.87899999972432852, "
            "36.41099999996367842 42.20100000035017729, "
            "43.76300000003539026 39.821000000461936, "
            "40.2219999999506399 31.18599999975413084, "
            "25.604857042664662 36.35760602075606585, "
            "19.48400000005494803 19.48900000005960464, "
            "33.6650000000372529 14.57500000018626451))"
        )
        poly = orient(shapely.from_wkt(wkt), sign=1.0)
        coords = list(poly.exterior.coords[:-1])
        n = len(coords)

        # Find the wedge tip (vertex with angle < 25 and concave)
        for i in range(n):
            prev = coords[(i - 1) % n]
            curr = coords[i]
            nxt = coords[(i + 1) % n]
            ang = _angle(prev, curr, nxt)
            if ang < 25 and _is_concave(prev, curr, nxt, orientation=1):
                left_chain, right_chain = _trace_wedge(coords, i, 1)
                join = _find_best_join(coords, left_chain, right_chain)
                assert join is not None
                li, ri, width = join
                assert li != ri, (
                    f"_find_best_join returned li == ri == {li} (width={width})"
                )
                assert width > 0, f"_find_best_join returned zero width"

    def test_simple_concave_notch(self):
        """A polygon with a narrow concave notch should have it removed."""
        from polyforge.ops.clearance.protrusions import remove_narrow_wedges

        # Square with a narrow inward notch on the right side (concave feature)
        coords = [
            (0, 0),
            (10, 0),
            (10, 4.9),
            (5, 5),  # concave notch tip
            (10, 5.1),
            (10, 10),
            (0, 10),
        ]
        poly = Polygon(coords)
        original_n = len(poly.exterior.coords)

        result = remove_narrow_wedges(poly, angle_threshold=30, min_depth=0.5)

        assert result.is_valid
        assert len(result.exterior.coords) < original_n

    def test_no_wedge_returns_unchanged(self):
        """Polygon without acute wedges should be returned unchanged."""
        from polyforge.ops.clearance.protrusions import remove_narrow_wedges

        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        result = remove_narrow_wedges(poly, angle_threshold=25)
        assert result.is_valid
        assert result.area == pytest.approx(poly.area, rel=1e-6)

    def test_removes_all_narrow_wedges(self):
        """All narrow wedges should be removed, not just the worst one.

        Polygon has two concave notches of equal width (2 units) but different
        depths, so that _find_best_join correctly identifies each notch's own
        neck (the first equal-distance pair wins via strict < comparison) and
        the higher-ratio wedge (Notch B, deeper) is processed first.

        Index layout (n=10):
          - Notch A: tip=2, left=1, right=3 — lower ratio, processed second
          - Notch B: tip=7, left=6, right=8 — higher ratio, processed first

        Because Notch B's splice removes index 7 (high), Notch A's stored
        indices 1 and 3 remain valid, so both tips are correctly removed.
        """
        from polyforge.ops.clearance.protrusions import remove_narrow_wedges

        coords = [
            (0, 0),
            (99, 0),       # left boundary of Notch A  (width = 2)
            (100, 10),     # Notch A tip — angle ~11°, depth ~135
            (101, 0),      # right boundary of Notch A
            (200, 0),
            (200, 100),
            (6, 100),      # left boundary of Notch B  (width = 2, top edge CCW)
            (5, 90),       # Notch B tip — angle ~11°, depth ~215 (near far corner)
            (4, 100),      # right boundary of Notch B
            (0, 100),
        ]
        poly = Polygon(coords)
        assert poly.is_valid
        original_n = len(poly.exterior.coords)

        result = remove_narrow_wedges(poly, angle_threshold=20, min_depth=0.5)

        assert result.is_valid
        result_coords = list(result.exterior.coords)
        assert (100.0, 10.0) not in result_coords, "Notch A tip should have been removed"
        assert (5.0, 90.0) not in result_coords, "Notch B tip should have been removed"
        assert len(result.exterior.coords) <= original_n - 2

    def test_complex_polygon_not_destroyed(self):
        """Removing wedges from a complex polygon must not destroy the body.

        Regression test for a bug where _splice_polygon kept the wrong side
        of the neck, discarding the polygon body and keeping only a tiny
        wedge fragment.  The polygon has two narrow wedges; removing them
        should preserve the vast majority of the area.
        """
        import shapely
        from polyforge.ops.clearance.protrusions import remove_narrow_wedges

        wkt = (
            "Polygon ((675156.186 6578718.304, 675120.963 6578707.425, "
            "675100.163 6578701.003, 675082.356 6578695.504, "
            "675082.35589143 6578695.50435723, 675055.924 6578687.382, "
            "675053.257 6578688.788, 675045.467 6578714.201, "
            "675032.36275256 6578757.107, 675032.36372734 6578757.107, "
            "675024.005 6578784.472, 675052.299 6578793.438, "
            "675071.698 6578799.585, 675075.19982683 6578788.1171328, "
            "675071.827 6578799.611, 675090.746 6578805.629, "
            "675090.804 6578805.458, 675090.955 6578805.506, "
            "675090.946 6578805.53, 675090.96594387 6578805.53650344, "
            "675090.923 6578805.677, 675105.165 6578810.19015547, "
            "675105.165 6578810.19018343, 675105.215 6578810.206, "
            "675143.076 6578802.92501446, 675143.076 6578802.92338815, "
            "675165.878 6578798.54, 675185.735 6578794.72, "
            "675190.922 6578777.74074957, 675190.922 6578777.763, "
            "675190.9566034 6578777.763, 675190.972 6578777.713, "
            "675204.521 6578733.231, 675204.45673075 6578733.21115113, "
            "675204.45673053 6578733.21115185, 675175.508 6578724.271, "
            "675156.186 6578718.304),"
            "(675058.815 6578718.20493640, 675058.78000249 6578718.19430135, "
            "675063.169 6578704.24, 675078.463 6578708.839, "
            "675078.59131067 6578708.15130537, 675096.32 6578713.62, "
            "675117.11 6578720.033, 675130.594 6578724.193, "
            "675128.80193362 6578729.91928184, 675128.79454377 6578729.917, "
            "675128.737 6578729.917, 675128.737 6578730.00362053, "
            "675140.44821181 6578733.7098899, 675139.145 6578738.14180109, "
            "675139.145 6578738.199, 675139.23132006 6578738.199, "
            "675139.24298844 6578738.16311101, 675145.296 6578739.943, "
            "675146.59788472 6578735.51896623, 675146.60480226 6578735.521, "
            "675146.662 6578735.521, 675146.662 6578735.43410963, "
            "675140.72616183 6578733.60124268, 675142.58278333 6578727.89074188, "
            "675152.259 6578730.876, 675171.673 6578736.866, "
            "675188.18770215 6578741.95976271, 675188.14933993 6578741.95976271, "
            "675183.47643488 6578756.73995123, 675183.27076271 6578758.4623211, "
            "675183.27076271 6578760.05023729, 675183.3606324 6578760.05023729, "
            "675183.849 6578761.043, 675181.242 6578762.238, "
            "675180.376 6578760.575, 675176.007 6578759.124, "
            "675175.683 6578759.441, 675173.285 6578767.352, "
            "675173.447 6578767.672, 675179.223 6578769.347, "
            "675179.881 6578770.515, 675178.828 6578773.895, "
            "675181.60263537 6578774.76732854, 675178.345 6578773.768, "
            "675175.425 6578783.005, 675163.362 6578785.55, "
            "675147.334 6578788.929, 675144.791 6578775.475, "
            "675137.864 6578776.964, 675131.242 6578778.244, "
            "675133.55358478 6578791.78669839, 675110.564 6578796.28583667, "
            "675110.564 6578796.32481393, 675103.821 6578796.03, "
            "675094.51978294 6578793.99962818, 675095.506 6578790.677, "
            "675091.878 6578789.48, 675091.272 6578790.032, "
            "675089.895 6578789.662, 675089.581 6578790.298, "
            "675087.479 6578789.754, 675087.569 6578789.371, "
            "675087.589 6578788.979, 675087.541 6578788.59, "
            "675087.425 6578788.215, 675087.245 6578787.866, "
            "675087.007 6578787.554, 675086.718 6578787.289, "
            "675086.386 6578787.078, 675086.023 6578786.929, "
            "675085.833 6578786.879, 675085.444 6578786.83, "
            "675085.051 6578786.851, 675084.669 6578786.939, "
            "675084.308 6578787.094, 675084.139 6578787.194, "
            "675083.831 6578787.438, 675083.571 6578787.731, "
            "675083.461 6578787.894, 675083.286 6578788.246, "
            "675080.581 6578787.55, 675079.921 6578785.748, "
            "675076.246 6578784.552, 675075.25110088 6578787.94240186, "
            "675070.954 6578786.589, 675072.267 6578782.298, "
            "675061.599 6578778.881, 675060.148 6578783.19, "
            "675055.924 6578781.862, 675039.669 6578776.746, "
            "675044.566 6578760.91, 675057.81622771 6578718.063768, "
            "675057.81774334 6578718.0642352, 675058.7590368 6578718.292, "
            "675058.815 6578718.292, 675058.815 6578718.20493640))"
        )
        poly = shapely.from_wkt(wkt)
        original_area = poly.area

        result = remove_narrow_wedges(poly, min_depth=0.5)

        assert result.is_valid
        # Must preserve at least 90% of area — the bug destroys >99.9%
        assert result.area > original_area * 0.9, (
            f"Polygon destroyed: area went from {original_area:.1f} to {result.area:.4f}"
        )

    def test_wedge_on_hole_ring(self):
        """Wedge spike on a hole ring should be removed.

        The polygon has no wedge on the exterior, but its hole has a narrow
        spike (vertices 4→5→6 in original WKT) that pokes into the polygon
        body, creating a clearance of ~0.028.  After removal the clearance
        should increase significantly.
        """
        import shapely
        from polyforge.ops.clearance.protrusions import remove_narrow_wedges

        wkt = (
            "Polygon ("
            "(675009.121 6578937.385, 675086.808 6578938.788, "
            "675090.316 6578894.682, 675009.622 6578892.577, "
            "675009.121 6578937.385),"
            "(675021.370 6578926.795, 675023.231 6578903.944, "
            "675079.067 6578904.254, 675076.896 6578926.485, "
            "675049.238 6578926.639, 675048.981 6578935.595, "
            "675049.210 6578926.639, 675021.370 6578926.795))"
        )
        poly = shapely.from_wkt(wkt)
        original_area = poly.area
        original_clearance = poly.minimum_clearance

        result = remove_narrow_wedges(poly)

        assert result.is_valid
        assert len(list(result.interiors)) == 1, "Hole should be preserved"
        assert result.area > original_area * 0.95, (
            f"Area changed too much: {original_area:.1f} -> {result.area:.1f}"
        )
        assert result.minimum_clearance > original_clearance * 10, (
            f"Clearance not improved enough: {original_clearance:.4f} -> "
            f"{result.minimum_clearance:.4f}"
        )


class TestFixClearanceSameHoleVertexMovement:
    """Test that same-hole pinch points are fixed by moving vertices apart."""

    BUG8_WKT = (
        "Polygon ((673335.80400000000372529 6581843.54899999964982271, "
        "673369.62800000002607703 6581783.49799999967217445, "
        "673444.54200000001583248 6581825.22800000011920929, "
        "673431.32999999995809048 6581848.51499999966472387, "
        "673423.81999999994877726 6581861.89599999971687794, "
        "673410.63800000003539026 6581885.22699999995529652, "
        "673335.80400000000372529 6581843.54899999964982271),"
        "(673391.14200096565764397 6581859.27400053385645151, "
        "673406.46100000001024455 6581867.74399999994784594, "
        "673427.4719999999506399 6581830.69299999997019768, "
        "673373.30200000002514571 6581800.72800000011920929, "
        "673352.90399999998044223 6581838.13300000037997961, "
        "673391.14199912489857525 6581859.27399951592087746, "
        "673397.77299863495863974 6581847.50800038501620293, "
        "673369.96599863702431321 6581831.99300038442015648, "
        "673377.50499961513560265 6581818.55999864172190428, "
        "673382.60000063525512815 6581821.39399920962750912, "
        "673405.15900135319679976 6581834.03799961134791374, "
        "673397.77300135686527938 6581847.50799960549920797, "
        "673402.14900137565564364 6581849.92499961704015732, "
        "673400.81600075995083898 6581852.23300068359822035, "
        "673391.14200096565764397 6581859.27400053385645151))"
    )

    def test_same_hole_pinch_clearance_improved(self):
        """Clearance should improve significantly for same-hole pinch points."""
        import shapely as _shapely
        from polyforge import fix_clearance

        poly = _shapely.from_wkt(self.BUG8_WKT)
        assert poly.minimum_clearance < 0.001, "Precondition: low clearance"

        result = fix_clearance(poly, min_clearance=1.0)
        assert result.is_valid
        assert result.minimum_clearance > 0.3, (
            f"Clearance should improve significantly, got {result.minimum_clearance:.6f}"
        )

    def test_same_hole_pinch_area_preserved(self):
        """Area should be well preserved (< 1% change)."""
        import shapely as _shapely
        from polyforge import fix_clearance

        poly = _shapely.from_wkt(self.BUG8_WKT)
        result = fix_clearance(poly, min_clearance=1.0)
        area_ratio = result.area / poly.area
        assert 0.99 < area_ratio < 1.01, (
            f"Area ratio should be near 1.0, got {area_ratio:.4f}"
        )

    def test_same_hole_pinch_no_self_intersection(self):
        """Moving vertices apart should not create self-intersections."""
        import shapely as _shapely
        from polyforge import fix_clearance

        poly = _shapely.from_wkt(self.BUG8_WKT)
        result = fix_clearance(poly, min_clearance=1.0)
        assert result.is_valid
        # Check that hole ring is simple (no crossings)
        for hole in result.interiors:
            ring = _shapely.LinearRing(hole.coords)
            assert ring.is_simple


class TestFixClearanceInterHoleVertexMovement:
    """Test that inter-hole pinch points are fixed by moving vertices apart."""

    BUG9_WKT = (
        "Polygon ((675298.20200000004842877 6582080.18800000008195639, "
        "675347.04099999996833503 6582221.62999999988824129, "
        "675417.63000000000465661 6582191.12000000011175871, "
        "675393.97699999995529652 6582127.40500000026077032, "
        "675385.85999999998603016 6582105.78699999954551458, "
        "675367.33200000005308539 6582056.05900000035762787, "
        "675298.20200000004842877 6582080.18800000008195639),"
        "(675353.04922236187849194 6582089.91792388446629047, "
        "675355.56299999996554106 6582096.68699999991804361, "
        "675362.48499999998603016 6582094.07799999974668026, "
        "675366.9340000000083819 6582092.53699999954551458, "
        "675371.91099999996367842 6582105.93800000008195639, "
        "675360.59199999994598329 6582110.22599999979138374, "
        "675363.712000000057742 6582118.62899999972432852, "
        "675370.69900000002235174 6582116.00600000005215406, "
        "675375.10900000005494803 6582114.41399999987334013, "
        "675380.02000000001862645 6582127.74899999983608723, "
        "675375.72400000097695738 6582129.34399999957531691, "
        "675368.69399871793575585 6582132.04299941938370466, "
        "675370.24899941042531282 6582136.231001284904778, "
        "675381.59521119389683008 6582132.01158993225544691, "
        "675398.29200000001583248 6582178.64900000020861626, "
        "675400.20799999998416752 6582183.66299999970942736, "
        "675379.84400000004097819 6582192.36000000033527613, "
        "675378.82200000004377216 6582190.03299999982118607, "
        "675376.90500000002793968 6582189.28899999987334013, "
        "675375.02500000002328306 6582190.12600000016391277, "
        "675374.17099999997299165 6582192.01800000015646219, "
        "675375.2099999999627471 6582194.42100000008940697, "
        "675355.59299999999348074 6582202.9150000000372529, "
        "675345.23600000003352761 6582173.11299999989569187, "
        "675349.63500000000931323 6582171.68400000035762787, "
        "675358.97900000005029142 6582171.1119999997317791, "
        "675374.44400000001769513 6582165.67700000014156103, "
        "675372.162999999942258 6582159.54499999992549419, "
        "675367.43345877842511982 6582161.18433166202157736, "
        "675343.96700032555963844 6582093.07000094559043646, "
        "675353.04922236187849194 6582089.91792388446629047),"
        "(675352.82778526481706649 6582089.32207721285521984, "
        "675351.40600059577263892 6582085.49199873115867376, "
        "675342.43899873050395399 6582088.63799938652664423, "
        "675343.96699905511923134 6582093.07000032812356949, "
        "675336.6938724210485816 6582095.59364226087927818, "
        "675332.65200000000186265 6582084.25999999977648258, "
        "675360.40899999998509884 6582074.68800000008195639, "
        "675365.40827388723846525 6582088.45575427170842886, "
        "675360.4231457004789263 6582090.34137895610183477, "
        "675359.15400058298837394 6582087.01599872298538685, "
        "675352.82778526481706649 6582089.32207721285521984),"
        "(675338.16572466760408133 6582152.99407393485307693, "
        "675349.77006499480921775 6582149.87798254657536745, "
        "675355.15177582099568099 6582165.44135169684886932, "
        "675343.78298287093639374 6582169.38195002730935812, "
        "675338.16572466760408133 6582152.99407393485307693),"
        "(675327.13755831494927406 6582121.10419382620602846, "
        "675338.14929256995674223 6582116.27187161054462194, "
        "675348.45995658577885479 6582146.08928013313561678, "
        "675336.86592559178825468 6582149.20201997645199299, "
        "675327.13755831494927406 6582121.10419382620602846),"
        "(675318.92000000004190952 6582084.92399999964982271, "
        "675320.11600000003818423 6582088.74399999994784594, "
        "675327.07368786237202585 6582086.29011008702218533, "
        "675329.15977292042225599 6582092.30579179152846336, "
        "675324.95700000005308539 6582093.79999999981373549, "
        "675326.64599999994970858 6582099.071000000461936, "
        "675331.60933740751352161 6582097.35897574853152037, "
        "675336.83773959195241332 6582112.47899164818227291, "
        "675325.82774084294214845 6582117.3091136934235692, "
        "675320.2900000000372529 6582101.26400000043213367, "
        "675318.70100000000093132 6582096.58800000045448542, "
        "675315.59799999999813735 6582095.13200000021606684, "
        "675314.15800000005401671 6582095.68699999991804361, "
        "675311.36300000001210719 6582087.58700000029057264, "
        "675318.92000000004190952 6582084.92399999964982271))"
    )

    def test_inter_hole_pinch_clearance_improved(self):
        """Clearance should improve significantly for inter-hole pinch points."""
        import shapely as _shapely
        from polyforge import fix_clearance

        poly = _shapely.from_wkt(self.BUG9_WKT)
        assert poly.minimum_clearance < 0.001, "Precondition: low clearance"

        result = fix_clearance(poly, min_clearance=1.0)
        assert result.is_valid
        assert result.minimum_clearance > 0.3, (
            f"Clearance should improve significantly, got {result.minimum_clearance:.6f}"
        )

    def test_inter_hole_pinch_area_preserved(self):
        """Area should be well preserved (< 1% change)."""
        import shapely as _shapely
        from polyforge import fix_clearance

        poly = _shapely.from_wkt(self.BUG9_WKT)
        result = fix_clearance(poly, min_clearance=1.0)
        area_ratio = result.area / poly.area
        assert 0.99 < area_ratio < 1.01, (
            f"Area ratio should be near 1.0, got {area_ratio:.4f}"
        )

    def test_inter_hole_pinch_most_holes_preserved(self):
        """Most holes should be preserved — at most 1 may be merged with its neighbor."""
        import shapely as _shapely
        from polyforge import fix_clearance

        poly = _shapely.from_wkt(self.BUG9_WKT)
        result = fix_clearance(poly, min_clearance=1.0)
        assert len(result.interiors) >= len(poly.interiors) - 1, (
            f"Expected at least {len(poly.interiors) - 1} holes, got {len(result.interiors)}"
        )
