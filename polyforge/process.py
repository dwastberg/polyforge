import numpy as np
import shapely
from shapely.geometry import (
    Polygon, MultiPolygon, LinearRing, MultiLineString, LineString,
    Point, GeometryCollection
)
from shapely.geometry.base import BaseGeometry


def _process_coords(coords, process_function, *args, **kwargs):
    """Process coordinates, handling Z components if present.

    Args:
        coords: Numpy array of coordinates (Nx2 or Nx3)
        process_function: Function that processes 2D coordinates (Nx2)
        *args, **kwargs: Arguments to pass to process_function

    Returns:
        Processed coordinates with Z component preserved if it was present.
        For 3D coordinates, Z values are interpolated along the simplified line
        based on the cumulative distance along the 2D path.
    """
    coords_array = np.array(coords)
    has_z = coords_array.shape[1] == 3

    if has_z:
        # Split into XY and Z components
        coords_2d = coords_array[:, :2]
        z_values = coords_array[:, 2]

        # Process only the 2D coordinates
        processed_2d = process_function(coords_2d, *args, **kwargs)

        # If the number of vertices hasn't changed, just recombine
        if len(processed_2d) == len(coords_2d):
            return np.column_stack([processed_2d, z_values])

        # Otherwise, interpolate Z values based on cumulative distance along the line
        # Calculate cumulative distances along the original 2D path
        orig_dists = np.zeros(len(coords_2d))
        for i in range(1, len(coords_2d)):
            orig_dists[i] = orig_dists[i-1] + np.linalg.norm(coords_2d[i] - coords_2d[i-1])

        # Calculate cumulative distances along the processed 2D path
        proc_dists = np.zeros(len(processed_2d))
        for i in range(1, len(processed_2d)):
            proc_dists[i] = proc_dists[i-1] + np.linalg.norm(processed_2d[i] - processed_2d[i-1])

        # Interpolate Z values at the processed vertex positions
        # Use the cumulative distance along the path as the interpolation parameter
        interpolated_z = np.interp(proc_dists, orig_dists, z_values)

        # Recombine with interpolated Z values
        return np.column_stack([processed_2d, interpolated_z])
    else:
        # No Z component, process as-is
        return process_function(coords_array, *args, **kwargs)


def process_geometry(geometry: BaseGeometry, process_function, *args, **kwargs) -> BaseGeometry:
    """Process a shapely geometry by applying a function to its vertices.

    This function takes any shapely geometry and applies a processing function to its
    coordinate arrays, returning a new geometry of the same type with processed coordinates.
    If the geometry has Z coordinates, they are automatically stripped before processing
    and then restored after processing.

    Args:
        geometry (BaseGeometry): The shapely geometry to process.
        process_function (callable): A function that takes a numpy array of 2D vertices (Nx2)
            and optional args/kwargs, and returns a processed array of the same shape.
        *args: Additional positional arguments to pass to the processing function.
        **kwargs: Additional keyword arguments to pass to the processing function.

    Returns:
        BaseGeometry: A new geometry of the same type with processed vertices.
            Z coordinates are preserved if present in the input geometry.
    """
    geom_type = geometry.geom_type

    if geom_type == 'Point':
        coords = np.array(geometry.coords)
        processed_coords = _process_coords(coords, process_function, *args, **kwargs)
        return Point(processed_coords[0])

    elif geom_type == 'LineString':
        coords = np.array(geometry.coords)
        processed_coords = _process_coords(coords, process_function, *args, **kwargs)
        return LineString(processed_coords)

    elif geom_type == 'LinearRing':
        coords = np.array(geometry.coords)
        processed_coords = _process_coords(coords, process_function, *args, **kwargs)
        return LinearRing(processed_coords)

    elif geom_type == 'Polygon':
        # Process exterior ring
        exterior_coords = np.array(geometry.exterior.coords)
        processed_exterior = _process_coords(exterior_coords, process_function, *args, **kwargs)

        # Process interior rings (holes)
        processed_interiors = []
        for interior in geometry.interiors:
            interior_coords = np.array(interior.coords)
            processed_interior = _process_coords(interior_coords, process_function, *args, **kwargs)
            processed_interiors.append(processed_interior)

        return Polygon(processed_exterior, holes=processed_interiors)

    elif geom_type == 'MultiLineString':
        processed_lines = []
        for line in geometry.geoms:
            coords = np.array(line.coords)
            processed_coords = _process_coords(coords, process_function, *args, **kwargs)
            processed_lines.append(LineString(processed_coords))
        return MultiLineString(processed_lines)

    elif geom_type == 'MultiPolygon':
        processed_polygons = [
            process_geometry(polygon, process_function, *args, **kwargs)
            for polygon in geometry.geoms
        ]
        return MultiPolygon(processed_polygons)

    elif geom_type == 'GeometryCollection':
        processed_geoms = [
            process_geometry(geom, process_function, *args, **kwargs)
            for geom in geometry.geoms
        ]
        return GeometryCollection(processed_geoms)

    else:
        raise ValueError(f"Unsupported geometry type: {geom_type}")
