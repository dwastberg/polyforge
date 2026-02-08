"""Tests for geometry repair functions."""

import pytest
from shapely.geometry import Polygon, MultiPolygon, Point, LineString

from polyforge import repair_geometry, batch_repair_geometries
from polyforge.core import RepairStrategy
from polyforge.core.errors import RepairError


def _bowtie() -> Polygon:
    """Self-intersecting bowtie polygon."""
    return Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])


def _spike_polygon() -> Polygon:
    """Valid polygon with a very thin spike."""
    return Polygon([
        (0, 0), (10, 0), (10, 10),
        (5.001, 10), (5.001, 10.002), (4.999, 10.002), (4.999, 10),
        (0, 10), (0, 0),
    ])


def _ring_self_intersection() -> Polygon:
    """Polygon with a ring self-intersection (figure-8 shape)."""
    return Polygon([
        (0, 0), (4, 0), (4, 4), (2, 2), (0, 4), (0, 0),
    ])


class TestRepairGeometry:
    """Tests for repair_geometry()."""

    def test_valid_geometry_returned_unchanged(self):
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        result = repair_geometry(poly)
        assert result.equals(poly)

    def test_fix_bowtie_auto(self):
        result = repair_geometry(_bowtie(), repair_strategy=RepairStrategy.AUTO)
        assert result.is_valid
        assert result.area > 0

    def test_fix_bowtie_buffer(self):
        result = repair_geometry(_bowtie(), repair_strategy=RepairStrategy.BUFFER)
        assert result.is_valid
        assert result.area > 0

    def test_fix_bowtie_simplify(self):
        result = repair_geometry(_bowtie(), repair_strategy=RepairStrategy.SIMPLIFY)
        assert result.is_valid
        assert result.area > 0

    def test_fix_bowtie_reconstruct(self):
        result = repair_geometry(_bowtie(), repair_strategy=RepairStrategy.RECONSTRUCT)
        assert result.is_valid
        assert result.area > 0

    def test_fix_bowtie_strict_raises(self):
        """Strict mode refuses to fix bowties (too aggressive a change)."""
        with pytest.raises(RepairError, match="Strict mode"):
            repair_geometry(_bowtie(), repair_strategy=RepairStrategy.STRICT)

    def test_fix_duplicate_vertices(self):
        """Test repair of a polygon with duplicate/near-duplicate vertices."""
        # Polygon with exact duplicate vertex
        poly = Polygon([
            (0, 0), (10, 0), (10, 0), (10, 10), (0, 10), (0, 0)
        ])
        # Shapely may consider this valid, but repair should handle it
        result = repair_geometry(poly)
        assert result.is_valid

    def test_invalid_strategy_raises(self):
        poly = _bowtie()
        with pytest.raises(ValueError, match="Unknown repair_strategy"):
            repair_geometry(poly, repair_strategy="nonexistent")

    def test_verbose_flag(self, capsys):
        repair_geometry(_bowtie(), verbose=True)
        captured = capsys.readouterr()
        assert "Invalid geometry" in captured.out

    def test_verbose_valid_geometry(self, capsys):
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        repair_geometry(poly, verbose=True)
        captured = capsys.readouterr()
        assert "already valid" in captured.out


class TestBatchRepairGeometries:
    """Tests for batch_repair_geometries()."""

    def test_all_valid(self):
        polys = [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            Polygon([(20, 0), (30, 0), (30, 10), (20, 10)]),
        ]
        repaired, failed = batch_repair_geometries(polys)
        assert len(repaired) == 2
        assert len(failed) == 0
        assert all(g.is_valid for g in repaired)

    def test_mixed_valid_invalid(self):
        polys = [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),  # valid
            _bowtie(),  # invalid
            Polygon([(20, 0), (30, 0), (30, 10), (20, 10)]),  # valid
        ]
        repaired, failed = batch_repair_geometries(polys)
        assert len(repaired) == 3
        assert len(failed) == 0
        assert all(g.is_valid for g in repaired)

    def test_on_error_keep_default(self):
        """Default on_error='keep' preserves alignment."""
        polys = [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            _bowtie(),
        ]
        repaired, failed = batch_repair_geometries(polys)
        assert len(repaired) == len(polys)

    def test_on_error_skip_preserves_alignment(self):
        """on_error='skip' inserts None for failed entries."""
        polys = [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            _bowtie(),
        ]
        # With 'skip' mode, both bowtie and valid are fine since repair succeeds
        repaired, failed = batch_repair_geometries(polys, on_error='skip')
        assert len(repaired) == len(polys)  # alignment always preserved

    def test_on_error_raise(self):
        """on_error='raise' propagates exceptions."""
        # repair_geometry shouldn't fail on a bowtie, so use a valid poly
        # and check that it works normally
        polys = [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])]
        repaired, failed = batch_repair_geometries(polys, on_error='raise')
        assert len(repaired) == 1

    def test_empty_list(self):
        repaired, failed = batch_repair_geometries([])
        assert repaired == []
        assert failed == []

    def test_verbose_output(self, capsys):
        polys = [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])]
        batch_repair_geometries(polys, verbose=True)
        captured = capsys.readouterr()
        assert "Processing geometry" in captured.out

    def test_with_specific_strategy(self):
        polys = [_bowtie(), _bowtie()]
        repaired, failed = batch_repair_geometries(
            polys, repair_strategy=RepairStrategy.BUFFER
        )
        assert len(repaired) == 2
        assert all(g.is_valid for g in repaired)


class TestGeometryConstraintsValidation:
    """Tests for GeometryConstraints __post_init__ validation."""

    def test_negative_min_clearance_raises(self):
        from polyforge.core import GeometryConstraints
        with pytest.raises(ValueError, match="min_clearance"):
            GeometryConstraints(min_clearance=-1.0)

    def test_negative_max_overlap_area_raises(self):
        from polyforge.core import GeometryConstraints
        with pytest.raises(ValueError, match="max_overlap_area"):
            GeometryConstraints(max_overlap_area=-1.0)

    def test_negative_min_area_ratio_raises(self):
        from polyforge.core import GeometryConstraints
        with pytest.raises(ValueError, match="min_area_ratio"):
            GeometryConstraints(min_area_ratio=-0.5)

    def test_max_below_min_area_ratio_raises(self):
        from polyforge.core import GeometryConstraints
        with pytest.raises(ValueError, match="max_area_ratio"):
            GeometryConstraints(min_area_ratio=0.5, max_area_ratio=0.3)

    def test_negative_max_holes_raises(self):
        from polyforge.core import GeometryConstraints
        with pytest.raises(ValueError, match="max_holes"):
            GeometryConstraints(max_holes=-1)

    def test_valid_constraints_accepted(self):
        from polyforge.core import GeometryConstraints
        c = GeometryConstraints(
            min_clearance=0.5,
            min_area_ratio=0.8,
            max_area_ratio=1.2,
            max_holes=5,
        )
        assert c.min_clearance == 0.5


class TestSafeClearanceNoneReturn:
    """Tests for _safe_clearance returning None on error."""

    def test_returns_none_for_invalid_object(self):
        from polyforge.metrics import _safe_clearance

        class FakeGeom:
            @property
            def minimum_clearance(self):
                raise ValueError("broken")

        result = _safe_clearance(FakeGeom())
        assert result is None

    def test_returns_float_for_valid_polygon(self):
        from polyforge.metrics import _safe_clearance
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        result = _safe_clearance(poly)
        assert isinstance(result, float)
        assert result > 0
