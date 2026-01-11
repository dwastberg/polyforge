"""Simple geometry visualization helpers for debugging."""

import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.geometry.base import BaseGeometry


def plot_comparison(original: BaseGeometry, modified: BaseGeometry, title: str = "Geometry Comparison"):
    """Plot original and modified geometries side by side.

    Args:
        original: Original geometry
        modified: Modified geometry after operation
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot original
    _plot_geometry(ax1, original, color='red', alpha=0.5)
    ax1.set_title("Original")
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Plot modified
    _plot_geometry(ax2, modified, color='blue', alpha=0.5)
    ax2.set_title("Modified")
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def _plot_geometry(ax, geom: BaseGeometry, color='blue', alpha=0.5):
    """Plot a geometry on the given axes.

    Args:
        ax: Matplotlib axes
        geom: Geometry to plot
        color: Fill color
        alpha: Transparency
    """
    if isinstance(geom, Polygon):
        _plot_polygon(ax, geom, color=color, alpha=alpha)
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            _plot_polygon(ax, poly, color=color, alpha=alpha)
    elif isinstance(geom, LineString):
        x, y = geom.xy
        ax.plot(x, y, color=color, linewidth=2)
    else:
        # Try to plot as generic geometry
        try:
            x, y = geom.exterior.xy
            ax.fill(x, y, color=color, alpha=alpha, edgecolor='black', linewidth=1)
        except:
            pass


def _plot_polygon(ax, poly: Polygon, color='blue', alpha=0.5):
    """Plot a single polygon with holes.

    Args:
        ax: Matplotlib axes
        poly: Polygon to plot
        color: Fill color
        alpha: Transparency
    """
    # Plot exterior
    x, y = poly.exterior.xy
    ax.fill(x, y, color=color, alpha=alpha, edgecolor='black', linewidth=1.5)

    # Plot holes (as white)
    for interior in poly.interiors:
        x, y = interior.xy
        ax.fill(x, y, color='white', edgecolor='black', linewidth=1)
