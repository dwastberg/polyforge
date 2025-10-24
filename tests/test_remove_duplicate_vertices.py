import numpy as np
from shapely.geometry import LineString, Polygon

from polyforge.simplify import remove_duplicate_vertices


class TestRemoveDuplicateVertices:
    """Tests for remove_duplicate_vertices function."""

    def test_remove_exact_duplicates(self):
        """Test removing exact duplicate vertices."""
        line = LineString([(0, 0), (0, 0), (1, 1), (1, 1), (2, 2)])
        result = remove_duplicate_vertices(line)

        coords = np.array(result.coords)
        assert len(coords) == 3  # (0,0), (1,1), (2,2)

    def test_remove_duplicates_with_tolerance(self):
        """Test removing duplicates within tolerance."""
        line = LineString([(0, 0), (1e-12, 1e-12), (1, 1)])
        result = remove_duplicate_vertices(line, tolerance=1e-10)

        coords = np.array(result.coords)
        assert len(coords) == 2

    def test_no_duplicates(self):
        """Test geometry without duplicates remains unchanged."""
        line = LineString([(0, 0), (1, 1), (2, 2)])
        result = remove_duplicate_vertices(line)

        np.testing.assert_array_almost_equal(
            np.array(line.coords),
            np.array(result.coords)
        )

    def test_polygon_closed_ring(self):
        """Test that closed ring is maintained."""
        poly = Polygon([(0, 0), (0, 0), (1, 0), (1, 0), (1, 1), (0, 1)])
        result = remove_duplicate_vertices(poly)

        coords = np.array(result.exterior.coords)
        # Should still be closed
        np.testing.assert_array_almost_equal(coords[0], coords[-1])
