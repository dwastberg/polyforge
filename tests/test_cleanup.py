"""Tests for shared geometry cleanup utilities."""

import pytest
from shapely.geometry import Polygon, MultiPolygon

from polyforge.core.cleanup import CleanupConfig, cleanup_polygon, remove_small_holes, remove_narrow_holes


class TestCleanupConfig:
    """Test CleanupConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        cfg = CleanupConfig()
        assert cfg.min_zero_area == 1e-10
        assert cfg.hole_area_threshold is None
        assert cfg.hole_aspect_ratio is None
        assert cfg.hole_min_width is None
        assert cfg.preserve_holes is True

    def test_custom_values(self):
        """Test custom configuration values."""
        cfg = CleanupConfig(
            min_zero_area=1e-6,
            hole_area_threshold=5.0,
            hole_aspect_ratio=10.0,
            hole_min_width=0.5,
            preserve_holes=False,
        )
        assert cfg.min_zero_area == 1e-6
        assert cfg.hole_area_threshold == 5.0
        assert cfg.hole_aspect_ratio == 10.0
        assert cfg.hole_min_width == 0.5
        assert cfg.preserve_holes is False


class TestRemoveSmallHoles:
    """Tests for remove_small_holes function."""

    def test_removes_holes_below_threshold(self):
        """Test that holes below the threshold are removed."""
        # Polygon with two holes: one small (area=1), one large (area=4)
        poly = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            holes=[
                [(1, 1), (2, 1), (2, 2), (1, 2)],  # area = 1
                [(5, 5), (7, 5), (7, 7), (5, 7)],  # area = 4
            ],
        )
        result = remove_small_holes(poly, min_area=2.0)
        assert len(result.interiors) == 1
        # The remaining hole should be the larger one
        remaining_hole = Polygon(result.interiors[0])
        assert remaining_hole.area >= 2.0

    def test_keeps_holes_above_threshold(self):
        """Test that holes above the threshold are kept."""
        poly = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            holes=[
                [(1, 1), (3, 1), (3, 3), (1, 3)],  # area = 4
                [(5, 5), (8, 5), (8, 8), (5, 8)],  # area = 9
            ],
        )
        result = remove_small_holes(poly, min_area=2.0)
        assert len(result.interiors) == 2

    def test_polygon_without_holes(self):
        """Test polygon with no holes remains unchanged."""
        poly = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        result = remove_small_holes(poly, min_area=1.0)
        assert result.equals(poly)
        assert len(result.interiors) == 0

    def test_zero_min_area_keeps_all_holes(self):
        """Test that zero min_area keeps all holes."""
        poly = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            holes=[
                [(1, 1), (1.1, 1), (1.1, 1.1), (1, 1.1)],  # tiny hole
            ],
        )
        result = remove_small_holes(poly, min_area=0.0)
        assert len(result.interiors) == 1

    def test_negative_min_area_keeps_all_holes(self):
        """Test that negative min_area keeps all holes."""
        poly = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]],
        )
        result = remove_small_holes(poly, min_area=-1.0)
        assert len(result.interiors) == 1

    def test_multipolygon_with_holes(self):
        """Test MultiPolygon with holes in each part."""
        poly1 = Polygon(
            [(0, 0), (6, 0), (6, 6), (0, 6)],
            holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]],  # area = 1
        )
        poly2 = Polygon(
            [(10, 0), (20, 0), (20, 10), (10, 10)],
            holes=[[(11, 1), (15, 1), (15, 5), (11, 5)]],  # area = 16
        )
        combined = MultiPolygon([poly1, poly2])
        result = remove_small_holes(combined, min_area=3.0)

        assert isinstance(result, MultiPolygon)
        # First polygon should have no holes (area < 3)
        assert len(result.geoms[0].interiors) == 0
        # Second polygon should keep its hole (area > 3)
        assert len(result.geoms[1].interiors) == 1

    def test_removes_all_small_holes(self):
        """Test removal of all holes when all are small."""
        poly = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            holes=[
                [(1, 1), (2, 1), (2, 2), (1, 2)],  # area = 1
                [(4, 4), (5, 4), (5, 5), (4, 5)],  # area = 1
            ],
        )
        result = remove_small_holes(poly, min_area=5.0)
        assert len(result.interiors) == 0

    def test_hole_exactly_at_threshold(self):
        """Test hole with area exactly at threshold is kept."""
        # Hole with area exactly 4.0
        poly = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]],  # area = 4
        )
        result = remove_small_holes(poly, min_area=4.0)
        # Should be kept since area >= min_area
        assert len(result.interiors) == 1


class TestRemoveNarrowHoles:
    """Tests for remove_narrow_holes function."""

    def test_removes_high_aspect_ratio_holes(self):
        """Test removal of narrow (high aspect ratio) holes."""
        # Narrow hole: 0.2 width, 4 length -> aspect ratio = 20
        narrow_hole = [(1, 1), (5, 1), (5, 1.2), (1, 1.2)]
        poly = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            holes=[narrow_hole],
        )
        result = remove_narrow_holes(poly, max_aspect_ratio=5.0)
        assert len(result.interiors) == 0

    def test_keeps_low_aspect_ratio_holes(self):
        """Test that square-ish holes are kept."""
        # Square hole: aspect ratio = 1
        square_hole = [(1, 1), (3, 1), (3, 3), (1, 3)]
        poly = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            holes=[square_hole],
        )
        result = remove_narrow_holes(poly, max_aspect_ratio=5.0)
        assert len(result.interiors) == 1

    def test_min_width_filtering(self):
        """Test filtering by minimum width."""
        # Narrow hole with small width
        narrow_hole = [(1, 1), (5, 1), (5, 1.3), (1, 1.3)]  # width ~0.3
        poly = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            holes=[narrow_hole],
        )
        result = remove_narrow_holes(poly, max_aspect_ratio=100.0, min_width=0.5)
        assert len(result.interiors) == 0

    def test_min_width_keeps_wide_holes(self):
        """Test that holes wider than min_width are kept."""
        # Wide hole: 2x2 square
        wide_hole = [(1, 1), (3, 1), (3, 3), (1, 3)]
        poly = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            holes=[wide_hole],
        )
        result = remove_narrow_holes(poly, max_aspect_ratio=100.0, min_width=0.5)
        assert len(result.interiors) == 1

    def test_combined_aspect_ratio_and_width(self):
        """Test combined filtering by aspect ratio and width."""
        poly = Polygon(
            [(0, 0), (20, 0), (20, 20), (0, 20)],
            holes=[
                [(1, 1), (3, 1), (3, 3), (1, 3)],        # square, width=2, keep
                [(5, 1), (9, 1), (9, 1.3), (5, 1.3)],    # narrow, width=0.3, remove
                [(11, 1), (13, 1), (13, 1.5), (11, 1.5)], # moderate aspect, narrow, depends
            ],
        )
        # Only remove very narrow holes
        result = remove_narrow_holes(poly, max_aspect_ratio=20.0, min_width=0.4)
        # Square hole should be kept, very narrow should be removed
        assert len(result.interiors) <= 2

    def test_multipolygon_narrow_holes(self):
        """Test MultiPolygon with narrow holes."""
        poly1 = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            holes=[[(1, 1), (5, 1), (5, 1.2), (1, 1.2)]],  # narrow
        )
        poly2 = Polygon(
            [(20, 0), (30, 0), (30, 10), (20, 10)],
            holes=[[(21, 1), (23, 1), (23, 3), (21, 3)]],  # square
        )
        combined = MultiPolygon([poly1, poly2])
        result = remove_narrow_holes(combined, max_aspect_ratio=5.0)

        assert isinstance(result, MultiPolygon)
        assert len(result.geoms[0].interiors) == 0  # narrow hole removed
        assert len(result.geoms[1].interiors) == 1  # square hole kept

    def test_default_max_aspect_ratio(self):
        """Test that default max_aspect_ratio (50) is applied."""
        # Very narrow hole with aspect ratio > 50
        very_narrow = [(1, 1), (11, 1), (11, 1.1), (1, 1.1)]  # 10 x 0.1, aspect ~100
        poly = Polygon(
            [(0, 0), (15, 0), (15, 15), (0, 15)],
            holes=[very_narrow],
        )
        # Using default aspect ratio of 50
        result = remove_narrow_holes(poly)
        assert len(result.interiors) == 0


class TestCleanupPolygon:
    """Tests for cleanup_polygon function."""

    def test_removes_zero_area_holes(self):
        """Test removal of effectively zero-area holes."""
        poly = Polygon(
            [(0, 0), (5, 0), (5, 5), (0, 5)],
            holes=[[(1, 1), (1.00001, 1), (1.00001, 1.00001), (1, 1.00001)]],
        )
        cfg = CleanupConfig(min_zero_area=1e-6)
        cleaned = cleanup_polygon(poly, cfg)
        assert len(cleaned.interiors) == 0

    def test_hole_area_threshold(self):
        """Test hole_area_threshold configuration."""
        poly = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            holes=[
                [(1, 1), (2, 1), (2, 2), (1, 2)],  # area = 1
                [(5, 5), (8, 5), (8, 8), (5, 8)],  # area = 9
            ],
        )
        cfg = CleanupConfig(hole_area_threshold=5.0)
        cleaned = cleanup_polygon(poly, cfg)
        assert len(cleaned.interiors) == 1

    def test_hole_aspect_ratio_config(self):
        """Test hole_aspect_ratio configuration."""
        poly = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            holes=[
                [(1, 1), (5, 1), (5, 1.2), (1, 1.2)],  # narrow
                [(6, 6), (8, 6), (8, 8), (6, 8)],       # square
            ],
        )
        cfg = CleanupConfig(hole_aspect_ratio=5.0)
        cleaned = cleanup_polygon(poly, cfg)
        assert len(cleaned.interiors) == 1

    def test_hole_min_width_config(self):
        """Test hole_min_width configuration."""
        poly = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            holes=[
                [(1, 1), (3, 1), (3, 1.3), (1, 1.3)],  # width ~0.3
                [(5, 5), (7, 5), (7, 7), (5, 7)],       # width = 2
            ],
        )
        cfg = CleanupConfig(hole_min_width=0.5)
        cleaned = cleanup_polygon(poly, cfg)
        assert len(cleaned.interiors) == 1

    def test_preserve_holes_false(self):
        """Test preserve_holes=False removes all holes."""
        poly = Polygon(
            [(0, 0), (5, 0), (5, 5), (0, 5)],
            holes=[[(1, 1), (4, 1), (4, 4), (1, 4)]],
        )
        cfg = CleanupConfig(preserve_holes=False)
        cleaned = cleanup_polygon(poly, cfg)
        assert len(cleaned.interiors) == 0

    def test_combined_hole_filters(self):
        """Test multiple hole filters applied together."""
        poly = Polygon(
            [(0, 0), (20, 0), (20, 20), (0, 20)],
            holes=[
                [(1, 1), (1.5, 1), (1.5, 1.5), (1, 1.5)],     # tiny area
                [(3, 3), (7, 3), (7, 3.2), (3, 3.2)],         # narrow
                [(10, 10), (14, 10), (14, 14), (10, 14)],     # good hole, keep
            ],
        )
        cfg = CleanupConfig(
            hole_area_threshold=1.0,
            hole_aspect_ratio=5.0,
        )
        cleaned = cleanup_polygon(poly, cfg)
        assert len(cleaned.interiors) == 1

    def test_multipolygon_cleanup(self):
        """Test cleanup on MultiPolygon."""
        poly1 = Polygon(
            [(0, 0), (5, 0), (5, 5), (0, 5)],
            holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]],
        )
        poly2 = Polygon(
            [(10, 0), (15, 0), (15, 5), (10, 5)],
            holes=[[(11, 1), (14, 1), (14, 4), (11, 4)]],
        )
        combined = MultiPolygon([poly1, poly2])
        cfg = CleanupConfig(hole_area_threshold=5.0)
        cleaned = cleanup_polygon(combined, cfg)

        assert isinstance(cleaned, MultiPolygon)
        # First hole (area=1) removed, second hole (area=9) kept
        assert len(cleaned.geoms[0].interiors) == 0
        assert len(cleaned.geoms[1].interiors) == 1

    def test_default_config(self):
        """Test cleanup with default configuration."""
        poly = Polygon(
            [(0, 0), (5, 0), (5, 5), (0, 5)],
            holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]],  # normal hole
        )
        cfg = CleanupConfig()
        cleaned = cleanup_polygon(poly, cfg)
        # Default config should preserve normal holes
        assert len(cleaned.interiors) == 1


class TestCleanupHelpers:
    """Ensure helper functions remain functional (original tests)."""

    def test_remove_small_holes_multipolygon(self):
        poly = Polygon([(0, 0), (6, 0), (6, 6), (0, 6)], holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]])
        combined = MultiPolygon([poly, Polygon([(10, 0), (12, 0), (12, 2), (10, 2)])])
        result = remove_small_holes(combined, min_area=3.0)
        assert isinstance(result, MultiPolygon)
        assert all(len(p.interiors) == 0 for p in result.geoms)

    def test_remove_narrow_holes_filters(self):
        hole = [(1, 1), (4, 1), (4, 1.2), (1, 1.2)]
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], holes=[hole])
        result = remove_narrow_holes(poly, max_aspect_ratio=5.0)
        assert len(result.interiors) == 0
