
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


import polyforge
from shapely.geometry import Polygon

from plot_geometry import plot_comparison

base = [(0, 0), (10, 0), (10, 4), (10, 6), (0, 6)]
spike = [(10, 4.9), (12, 5), (10, 5.1)]  # Narrow spike

# Insert spike into base
coords = base[:3] + spike + base[3:]
poly = Polygon(coords)
fixed = polyforge.fix_narrow_protrusion(poly, min_clearance=1.0)
plot_comparison(poly, fixed, )