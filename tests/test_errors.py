"""Tests for polyforge error hierarchy."""

import pytest
from shapely.geometry import Polygon

from polyforge.core.errors import (
    PolyforgeError,
    ValidationError,
    RepairError,
    OverlapResolutionError,
    MergeError,
    ClearanceError,
    ConfigurationError,
)


class TestErrorHierarchy:
    """Test exception class hierarchy."""

    def test_all_errors_inherit_from_polyforge_error(self):
        """All custom errors should inherit from PolyforgeError."""
        assert issubclass(ValidationError, PolyforgeError)
        assert issubclass(RepairError, PolyforgeError)
        assert issubclass(OverlapResolutionError, PolyforgeError)
        assert issubclass(MergeError, PolyforgeError)
        assert issubclass(ClearanceError, PolyforgeError)
        assert issubclass(ConfigurationError, PolyforgeError)

    def test_polyforge_error_inherits_from_exception(self):
        """PolyforgeError should inherit from Exception."""
        assert issubclass(PolyforgeError, Exception)

    def test_catch_all_polyforge_errors(self):
        """Should be able to catch all polyforge errors with PolyforgeError."""
        with pytest.raises(PolyforgeError):
            raise ValidationError("test")

        with pytest.raises(PolyforgeError):
            raise RepairError("test")

        with pytest.raises(PolyforgeError):
            raise ConfigurationError("test")


class TestValidationError:
    """Test ValidationError functionality."""

    def test_basic_validation_error(self):
        """ValidationError can be raised with just a message."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Invalid geometry")

        assert str(exc_info.value) == "Invalid geometry"

    def test_validation_error_with_issues(self):
        """ValidationError can carry issues list."""
        error = ValidationError(
            "Invalid geometry",
            issues=['self-intersection', 'duplicate vertices']
        )

        assert error.issues == ['self-intersection', 'duplicate vertices']
        assert 'self-intersection' in str(error)

    def test_validation_error_with_geometry(self):
        """ValidationError can carry geometry object."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        error = ValidationError("Invalid", geometry=poly)

        assert error.geometry is poly

    def test_validation_error_repr(self):
        """ValidationError has useful repr."""
        error = ValidationError("test", issues=['issue1'])
        repr_str = repr(error)

        assert 'ValidationError' in repr_str
        assert 'issue1' in repr_str


class TestRepairError:
    """Test RepairError functionality."""

    def test_basic_repair_error(self):
        """RepairError can be raised with just a message."""
        with pytest.raises(RepairError) as exc_info:
            raise RepairError("Could not repair geometry")

        assert "Could not repair" in str(exc_info.value)

    def test_repair_error_with_strategies(self):
        """RepairError can carry strategies_tried list."""
        error = RepairError(
            "Repair failed",
            strategies_tried=['buffer', 'simplify', 'reconstruct']
        )

        assert error.strategies_tried == ['buffer', 'simplify', 'reconstruct']

    def test_repair_error_with_last_error(self):
        """RepairError can carry the last error encountered."""
        last_err = ValueError("Something went wrong")
        error = RepairError("Repair failed", last_error=last_err)

        assert error.last_error is last_err

    def test_repair_error_repr(self):
        """RepairError has useful repr."""
        error = RepairError("test", strategies_tried=['buffer', 'simplify'])
        repr_str = repr(error)

        assert 'RepairError' in repr_str
        assert 'buffer' in repr_str


class TestOverlapResolutionError:
    """Test OverlapResolutionError functionality."""

    def test_basic_overlap_error(self):
        """OverlapResolutionError can be raised with just a message."""
        with pytest.raises(OverlapResolutionError):
            raise OverlapResolutionError("Failed to resolve overlaps")

    def test_overlap_error_with_metadata(self):
        """OverlapResolutionError can carry iteration metadata."""
        error = OverlapResolutionError(
            "Max iterations reached",
            iterations=100,
            remaining_overlaps=5
        )

        assert error.iterations == 100
        assert error.remaining_overlaps == 5

    def test_overlap_error_repr(self):
        """OverlapResolutionError has useful repr."""
        error = OverlapResolutionError("test", iterations=10, remaining_overlaps=3)
        repr_str = repr(error)

        assert 'OverlapResolutionError' in repr_str
        assert '10' in repr_str
        assert '3' in repr_str


class TestMergeError:
    """Test MergeError functionality."""

    def test_basic_merge_error(self):
        """MergeError can be raised with just a message."""
        with pytest.raises(MergeError):
            raise MergeError("Merge operation failed")

    def test_merge_error_with_strategy(self):
        """MergeError can carry strategy information."""
        error = MergeError(
            "Merge failed",
            strategy='selective_buffer',
            group_indices=[0, 1, 2]
        )

        assert error.strategy == 'selective_buffer'
        assert error.group_indices == [0, 1, 2]

    def test_merge_error_repr(self):
        """MergeError has useful repr."""
        error = MergeError("test", strategy='convex_bridges')
        repr_str = repr(error)

        assert 'MergeError' in repr_str
        assert 'convex_bridges' in repr_str


class TestClearanceError:
    """Test ClearanceError functionality."""

    def test_basic_clearance_error(self):
        """ClearanceError can be raised with just a message."""
        with pytest.raises(ClearanceError):
            raise ClearanceError("Failed to achieve target clearance")

    def test_clearance_error_with_metadata(self):
        """ClearanceError can carry clearance metadata."""
        error = ClearanceError(
            "Clearance not achieved",
            target_clearance=1.0,
            achieved_clearance=0.5,
            issue_type='narrow_passage'
        )

        assert error.target_clearance == 1.0
        assert error.achieved_clearance == 0.5
        assert error.issue_type == 'narrow_passage'

    def test_clearance_error_repr(self):
        """ClearanceError has useful repr."""
        error = ClearanceError("test", target_clearance=1.0, achieved_clearance=0.5)
        repr_str = repr(error)

        assert 'ClearanceError' in repr_str
        assert '1.0' in repr_str
        assert '0.5' in repr_str


class TestConfigurationError:
    """Test ConfigurationError functionality."""

    def test_basic_configuration_error(self):
        """ConfigurationError can be raised with just a message."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Invalid configuration parameter")


class TestErrorMessages:
    """Test error message formatting."""

    def test_validation_error_includes_issues_in_message(self):
        """ValidationError should include issues in string representation."""
        error = ValidationError(
            "Invalid geometry",
            issues=['self-intersection', 'duplicate vertices']
        )

        error_str = str(error)
        assert 'Invalid geometry' in error_str
        assert 'self-intersection' in error_str
        assert 'duplicate vertices' in error_str

    def test_errors_have_meaningful_messages(self):
        """All errors should have meaningful string representations."""
        errors = [
            ValidationError("validation failed", issues=['issue1']),
            RepairError("repair failed", strategies_tried=['buffer']),
            OverlapResolutionError("overlap failed", iterations=10),
            MergeError("merge failed", strategy='test'),
            ClearanceError("clearance failed", target_clearance=1.0),
            ConfigurationError("config failed"),
        ]

        for error in errors:
            assert len(str(error)) > 0
            assert len(repr(error)) > 0


class TestErrorImports:
    """Test that errors can be imported from expected locations."""

    def test_import_from_core_errors(self):
        """Should be able to import from core.errors module."""
        from polyforge.core.errors import (
            PolyforgeError,
            ValidationError,
            RepairError,
        )
        assert PolyforgeError is not None
        assert ValidationError is not None
        assert RepairError is not None

    def test_import_from_core(self):
        """Should be able to import from core module."""
        from polyforge.core import (
            PolyforgeError,
            ValidationError,
            RepairError,
        )
        assert PolyforgeError is not None
        assert ValidationError is not None
        assert RepairError is not None

    def test_import_from_main_package(self):
        """Should be able to import from main polyforge package."""
        from polyforge import (
            PolyforgeError,
            ValidationError,
            RepairError,
        )
        assert PolyforgeError is not None
        assert ValidationError is not None
        assert RepairError is not None
