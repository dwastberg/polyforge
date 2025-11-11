"""Tests for shared geometry cleanup utilities."""

from shapely.geometry import Polygon, MultiPolygon

from polyforge.core.cleanup import CleanupConfig, cleanup_polygon, remove_small_holes, remove_narrow_holes


class TestCleanupPolygon:
    """Validate cleanup_polygon behavior."""

    def test_removes_zero_area_holes(self):
        poly = Polygon(
            [(0, 0), (5, 0), (5, 5), (0, 5)],
            holes=[[(1, 1), (1.00001, 1), (1.00001, 1.00001), (1, 1.00001)]],
        )
        cfg = CleanupConfig(min_zero_area=1e-6)
        cleaned = cleanup_polygon(poly, cfg)
        assert len(cleaned.interiors) == 0

    def test_hole_threshold_and_shape(self):
        poly = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            holes=[
                [(1, 1), (3, 1), (3, 3), (1, 3)],  # keep
                [(5, 1), (5.5, 1), (5.5, 5), (5, 5)],  # narrow width
            ],
        )
        cfg = CleanupConfig(hole_area_threshold=3.0, hole_min_width=0.7)
        cleaned = cleanup_polygon(poly, cfg)
        assert len(cleaned.interiors) == 1

    def test_preserve_holes_false(self):
        poly = Polygon(
            [(0, 0), (5, 0), (5, 5), (0, 5)],
            holes=[[(1, 1), (4, 1), (4, 4), (1, 4)]],
        )
        cfg = CleanupConfig(preserve_holes=False)
        cleaned = cleanup_polygon(poly, cfg)
        assert len(cleaned.interiors) == 0


class TestCleanupHelpers:
    """Ensure helper functions remain functional."""

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
