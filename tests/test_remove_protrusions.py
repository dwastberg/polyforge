"""Tests for remove_narrow_protrusions function."""

import pytest
import numpy as np
from shapely.geometry import Polygon
from polyforge import remove_narrow_protrusions
from polyforge.ops.clearance.remove_protrusions import _collect_protrusion_candidate


class TestRemoveNarrowProtrusions:
    """Tests for basic protrusion removal functionality."""

    def test_collect_candidate_helper(self):
        coords = np.array([
            (0, 0),
            (5, 0),
            (5, 2),
            (5.2, 2.2),
            (5, 4),
            (0, 4),
            (0, 0),
        ])

        candidate = _collect_protrusion_candidate(coords, threshold=3.0)
        assert candidate is not None
        idx, aspect_ratio = candidate
        assert idx == 2  # central spike vertex after removal
        assert aspect_ratio > 3.0

    def test_horizontal_spike(self):
        """Test removing a simple horizontal spike."""
        # Rectangle with narrow horizontal spike - modified to avoid sharp corners
        coords = [
            (0, 0), (10, 0), (10, 3),
            (10, 4.9), (12, 5), (10, 5.1),  # Narrow spike
            (10, 7), (0, 7)
        ]
        poly = Polygon(coords)

        # With corrected distance calculation, use appropriate threshold
        result = remove_narrow_protrusions(poly, aspect_ratio_threshold=5.0)

        assert result.is_valid
        # Spike tip should be removed
        assert len(result.exterior.coords) < len(poly.exterior.coords)
        # Spike tip (12, 5) should not be in result
        result_coords = list(result.exterior.coords)
        assert (12, 5) not in result_coords

    def test_vertical_spike(self):
        """Test removing a vertical spike."""
        coords = [
            (0, 0), (10, 0), (10, 10),
            (5.1, 10), (5, 12), (4.9, 10),  # Vertical spike on top edge
            (0, 10)
        ]
        poly = Polygon(coords)

        result = remove_narrow_protrusions(poly, aspect_ratio_threshold=5.0)

        assert result.is_valid
        # Spike should be removed
        assert len(result.exterior.coords) < len(poly.exterior.coords)

    def test_no_protrusions(self):
        """Test that polygons without narrow protrusions are unchanged."""
        # Simple rectangle
        coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
        poly = Polygon(coords)

        result = remove_narrow_protrusions(poly, aspect_ratio_threshold=5.0)

        assert result.is_valid
        assert len(result.exterior.coords) == len(poly.exterior.coords)
        # Should be essentially the same polygon
        assert abs(result.area - poly.area) < 0.01

    def test_aspect_ratio_threshold(self):
        """Test that aspect ratio threshold controls what gets removed."""
        coords = [
            (0, 0), (10, 0), (10, 4),
            (10, 4.9), (12, 5), (10, 5.1),
            (10, 6), (0, 6)
        ]
        poly = Polygon(coords)

        # Low threshold - should remove spike
        result_low = remove_narrow_protrusions(poly, aspect_ratio_threshold=3.0)
        assert len(result_low.exterior.coords) < len(poly.exterior.coords)

        # High threshold - should NOT remove spike
        result_high = remove_narrow_protrusions(poly, aspect_ratio_threshold=20.0)
        assert len(result_high.exterior.coords) == len(poly.exterior.coords)

    def test_multiple_spikes(self):
        """Test removing multiple narrow spikes."""
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

        result = remove_narrow_protrusions(poly, aspect_ratio_threshold=5.0)

        assert result.is_valid
        # Both spikes should be removed
        assert len(result.exterior.coords) < len(poly.exterior.coords) - 1


class TestProtrusionPreservation:
    """Tests for preserving polygon features."""

    def test_preserves_holes(self):
        """Test that interior holes are preserved."""
        # Polygon with spike and hole
        exterior = [
            (0, 0), (10, 0), (10, 4),
            (10, 4.9), (12, 5), (10, 5.1),
            (10, 10), (0, 10)
        ]
        hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
        poly = Polygon(exterior, holes=[hole])

        result = remove_narrow_protrusions(poly, aspect_ratio_threshold=5.0)

        assert result.is_valid
        assert len(result.interiors) == 1  # Hole preserved

    def test_preserves_validity(self):
        """Test that result is always valid."""
        # Various polygon shapes
        test_cases = [
            # Triangle
            [(0, 0), (10, 0), (5, 10)],
            # Pentagon
            [(0, 0), (10, 0), (12, 5), (5, 10), (0, 5)],
            # Complex shape
            [(0, 0), (10, 0), (10, 5), (15, 5), (15, 10), (0, 10)]
        ]

        for coords in test_cases:
            poly = Polygon(coords)
            result = remove_narrow_protrusions(poly)
            assert result.is_valid

    def test_minimum_vertices(self):
        """Test that polygon maintains minimum vertex count."""
        # Triangle (minimum valid polygon)
        coords = [(0, 0), (10, 0), (5, 10)]
        poly = Polygon(coords)

        result = remove_narrow_protrusions(poly, aspect_ratio_threshold=5.0)

        assert result.is_valid
        # Should still be a valid polygon (at least 4 coords including closing)
        assert len(result.exterior.coords) >= 4


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_narrow_spike(self):
        """Test with extremely narrow spike."""
        coords = [
            (0, 0), (10, 0), (10, 5),
            (10, 4.99), (15, 5), (10, 5.01),  # Very narrow
            (10, 10), (0, 10)
        ]
        poly = Polygon(coords)

        result = remove_narrow_protrusions(poly, aspect_ratio_threshold=5.0)

        assert result.is_valid
        # Very narrow spike should definitely be removed
        assert len(result.exterior.coords) < len(poly.exterior.coords)

    def test_wide_protrusion(self):
        """Test that wide protrusions are NOT removed."""
        coords = [
            (0, 0), (10, 0), (10, 3),
            (10, 2), (12, 5), (10, 8),  # Wide protrusion (low aspect ratio)
            (10, 10), (0, 10)
        ]
        poly = Polygon(coords)

        result = remove_narrow_protrusions(poly, aspect_ratio_threshold=5.0)

        # Wide protrusion should remain
        assert len(result.exterior.coords) == len(poly.exterior.coords)

    def test_iteration_limit(self):
        """Test that max_iterations prevents infinite loops."""
        coords = [
            (0, 0), (10, 0), (10, 10),
            (5.1, 10), (5, 11), (4.9, 10),
            (0, 10)
        ]
        poly = Polygon(coords)

        # Should work with limited iterations
        result = remove_narrow_protrusions(poly, max_iterations=1)
        assert result.is_valid

    def test_invalid_geometry_type(self):
        """Test that non-Polygon input raises TypeError."""
        from shapely.geometry import LineString

        line = LineString([(0, 0), (10, 0), (10, 10)])

        with pytest.raises(TypeError):
            remove_narrow_protrusions(line)

    def test_empty_result_handling(self):
        """Test graceful handling when operation would create invalid geometry."""
        # Very small polygon that might become invalid
        coords = [(0, 0), (0.1, 0), (0.05, 0.1)]
        poly = Polygon(coords)

        # Should return valid polygon even if no changes made
        result = remove_narrow_protrusions(poly)
        assert result.is_valid


class TestAspectRatioCalculation:
    """Tests for aspect ratio calculation logic."""

    def test_equilateral_triangle(self):
        """Test that equilateral triangle has low aspect ratio."""
        # Equilateral triangle - all sides equal
        coords = [(0, 0), (10, 0), (5, 8.66), (5, 0)]
        poly = Polygon(coords)

        # Equilateral has aspect ratio ~ 1.15, should not be removed
        result = remove_narrow_protrusions(poly, aspect_ratio_threshold=2.0)
        assert len(result.exterior.coords) == len(poly.exterior.coords)

    def test_right_triangle(self):
        """Test aspect ratio calculation for right triangles."""
        # Right triangle with known dimensions
        coords = [(0, 0), (10, 0), (10, 10), (0, 0)]
        poly = Polygon(coords)

        # Right triangle aspect ratio ~ 1.41, should not be removed
        result = remove_narrow_protrusions(poly, aspect_ratio_threshold=3.0)
        assert len(result.exterior.coords) == len(poly.exterior.coords)

    def test_very_thin_triangle(self):
        """Test that very thin triangles have high aspect ratio."""
        # Very thin triangle - narrow spike on right edge
        coords = [
            (0, 0), (10, 0), (10, 2.45),
            (10.05, 2.5),  # Very narrow spike tip, only 0.05 units out
            (10, 2.55), (10, 5), (0, 5)
        ]
        poly = Polygon(coords)

        # Should have very high aspect ratio and be removed
        result = remove_narrow_protrusions(poly, aspect_ratio_threshold=3.0)
        assert len(result.exterior.coords) < len(poly.exterior.coords)
