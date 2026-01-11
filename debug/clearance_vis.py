
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import shapely

import polyforge
from shapely.geometry import Polygon

from plot_geometry import plot_comparison
from polyforge.core.types import PassageStrategy

# base = [(0, 0), (10, 0), (10, 4), (10, 6), (0, 6)]
# spike = [(10, 4.9), (12, 5), (10, 5.1)]  # Narrow spike
#
# # Insert spike into base
# coords = base[:3] + spike + base[3:]
# poly = Polygon(coords)
# print("Original clearance", shapely.minimum_clearance(poly))
# fixed = (polyforge.fix_narrow_protrusion(poly, min_clearance=0.5))
# print("Fixed clearance", shapely.minimum_clearance(fixed))
# plot_comparison(poly, fixed, )

# coords = [
#             (0, 0), (2, 0), (2, 1),
#             (1.1, 1.5), (0.1, 2), (1.1, 2.5),  # Narrow section
#             (2, 3), (2, 4), (0, 4),
#         ]

coords = [
            (0, 0), (2, 0), (2, 1), (1.05,1), (1.05,0.5), (0.95,0.5), (0.95,1), (0,1)


        ]
poly = Polygon(coords)
print ("Original clearance", shapely.minimum_clearance(poly))
result = polyforge.fix_narrow_passage(poly, min_clearance=0.2, strategy=PassageStrategy.ARAP)
print ("Fixed clearance", shapely.minimum_clearance(result))
plot_comparison(poly, result)