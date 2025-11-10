# Polyforge

Polyforge is a focused toolkit for cleaning, simplifying, repairing, and merging planar geometries built on top of Shapely.  It exposes a small set of high-level functions that combine fast NumPy-based vertex processing with STRtree-powered spatial indexing, so you can run the same code on a handful of polygons or on thousands of building footprints.

## Installation
```bash
pip install polyforge
```
Python 3.10+ with Shapely ≥ 2.1 is required.

## What You Get
| Area | Highlights |
| --- | --- |
| **Simplify & Clean** | `simplify_rdp`, `simplify_vw`, `collapse_short_edges`, `remove_small_holes`, `remove_narrow_holes` |
| **Clearance Fixing** | `fix_clearance` auto-detects low-clearance issues (holes too close, spikes, passages) and applies the right fix. |
| **Overlap & Merge** | `split_overlap` for pairs, `remove_overlaps` for batches, `merge_close_polygons` with 5 strategies (simple/selective buffer, vertex movement, boundary extension, convex bridges). |
| **Repair & QA** | `repair_geometry`, `analyze_geometry`, robust batch fixing with `robust_fix_geometry` + constraint validation. |
| **Core Types** | Strategy enums (MergeStrategy, RepairStrategy, OverlapStrategy, …), `GeometryConstraints`, and shared cleanup/spatial utilities. |

## Quick Examples

### Simplify & Clean
```python
from shapely.geometry import Polygon
from polyforge import simplify_vwp, remove_small_holes

poly = Polygon([(0, 0), (5, 0.1), (10, 0), (10, 10), (0, 10)])
poly = simplify_vwp(poly, threshold=0.2)
poly = remove_small_holes(poly, min_area=1.0)
```

### Fix Narrow Passages Automatically
```python
from polyforge import fix_clearance

improved, info = fix_clearance(complex_poly, min_clearance=1.5, return_diagnosis=True)
print(info.issue, info.fixed)
```

### Merge Buildings into Blocks
```python
from polyforge import merge_close_polygons
from polyforge.core import MergeStrategy

merged = merge_close_polygons(buildings, margin=2.0,
                              merge_strategy=MergeStrategy.BOUNDARY_EXTENSION,
                              insert_vertices=True)
```

### Remove Overlaps at Scale
```python
from polyforge import remove_overlaps
clean = remove_overlaps(parcel_list, overlap_strategy='split')
```

### Constrain Repairs
```python
from polyforge import robust_fix_geometry
from polyforge.core import GeometryConstraints

constraints = GeometryConstraints(min_clearance=1.0, min_area_ratio=0.9)
fixed, warn = robust_fix_geometry(polygon, constraints)
```

## Design Notes
- **process_geometry() everywhere** – every simplification/cleanup call is just a NumPy function applied to each vertex array, so Z values are preserved automatically.
- **STRtree first** – overlap removal, merges, and vertex insertion all walk spatial indexes, keeping runtime roughly O(n log n) even for thousands of polygons.
- **Composable stages** – robust fixing uses small “stages” (validity repair, clearance fix, merge, cleanup) so you can reason about the pipeline and extend it.

## Project Layout (high level)
```
polyforge/
  simplify.py        # simplification + hole helpers
  clearance/         # clearance diagnosis + strategies
  overlap/           # overlap engine + batch helpers
  merge/             # merge orchestrator and strategy modules
  repair/            # repair stages, transaction logic, batch fix
  core/              # enums, constraints, cleanup, spatial utils
```

## Running Tests
```bash
python -m pytest tests -q
```

That’s it—import what you need from `polyforge` and combine the high-level functions to build your own geometry pipelines.
