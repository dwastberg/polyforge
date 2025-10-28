import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import shapely

import polyforge
from shapely.geometry import Polygon

from plot_geometry import plot_comparison

poly1 = Polygon([(0, 5), (10, 5), (10, 15), (0, 15)])
poly2 = Polygon([(11, 0), (21, 0), (21, 10), (11, 10)])

strategies = ['simple_buffer','selective_buffer','vertex_movement', 'boundary_extension', 'convex_bridges']

for s in strategies:
    print(f"Strategy: {s}")
    result = polyforge.merge_close_polygons([poly1, poly2], margin=2.0, strategy=s, insert_vertices=True )
    result = result[0]
    result = polyforge.simplify_vwp(result, threshold=1.0)
    plot_comparison(poly1.union(poly2), result, mode='overlay', title=f"Merging with strategy: {s}")

