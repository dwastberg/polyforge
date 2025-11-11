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
| **Clearance Fixing** | `fix_clearance` auto-detects issues (holes too close, spikes, passages) and routes to ops helpers. Individual functions (`fix_hole_too_close`, `fix_narrow_passage`, …) now accept either enums or plain strings (e.g. `strategy="split"`). |
| **Overlap & Merge** | `split_overlap` for pairs, `remove_overlaps` for batches, `merge_close_polygons` with ops-based strategies (`"simple_buffer"`, `"selective_buffer"`, `"vertex_movement"`, `"boundary_extension"`, `"convex_bridges"`). |
| **Repair & QA** | `repair_geometry`, `analyze_geometry`, plus the pipeline-driven `robust_fix_geometry` / `robust_fix_batch` that iterate validity → clearance → merge → cleanup steps. |
| **Core Types** | Strategy enums (MergeStrategy, RepairStrategy, OverlapStrategy, …) remain available, but everything also accepts literal strings; `GeometryConstraints` + shared ops utilities live in `polyforge.ops.*`. |

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

merged = merge_close_polygons(
    buildings,
    margin=2.0,
    merge_strategy="boundary_extension",
    insert_vertices=True,
)
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
- **Ops-first architecture** – low-level helpers live in `polyforge/ops/…` (simplify, cleanup, clearance, merge). Public modules are thin wrappers that call these ops so behaviour stays consistent across the library.
- **Literal-friendly configuration** – strategy parameters accept enums or plain strings (`"selective_buffer"`, `"smooth"`, etc.), making it easy to drive Polyforge from config files or CLI flags.
- **Pipeline-driven repair** – `robust_fix_geometry` now runs a simple step list (validity → clearance → merge → cleanup) via `polyforge.pipeline.run_steps`, replacing the old transactional stage system.
- **process_geometry() everywhere** – every simplification/cleanup call is just a NumPy function applied to each vertex array, so Z values are preserved automatically.
- **STRtree first** – overlap removal, merges, and vertex insertion all walk spatial indexes, keeping runtime roughly O(n log n) even for thousands of polygons.

## Project Layout (high level)
```
polyforge/
  ops/
    simplify_ops.py        # coordinate-level simplify helpers
    cleanup_ops.py         # hole removal, sliver cleanup
    clearance/             # clearance fix primitives
    merge/ + merge_*       # merge strategies & utilities
  simplify.py              # thin wrappers over ops.simplify
  clearance/               # fix_clearance + public wrappers
  overlap.py               # overlap engine + batch helpers
  merge/                   # merge orchestrator (calls ops)
  repair/                  # classic repair + pipeline-based robust fixes
  core/                    # enums, constraints, geometry/spatial utils
  pipeline.py              # FixConfig + run_steps helper
```

### Advanced: Using the ops layer directly
Most users should stick to the high-level API (`polyforge.simplify`, `polyforge.clearance`, `polyforge.merge`, `polyforge.repair`). If you need to compose Polyforge primitives yourself (custom pipelines, batch transforms, etc.), import the ops helpers:

```python
from shapely.geometry import Polygon
from polyforge.process import process_geometry
from polyforge.ops.simplify_ops import snap_short_edges
from polyforge.ops.clearance import fix_hole_too_close
from polyforge.ops.merge import merge_selective_buffer

poly = Polygon([(0, 0), (1, 0.01), (2, 0), (2, 2), (0, 2)])

# Run a raw NumPy transform via process_geometry
collapsed = process_geometry(poly, snap_short_edges, min_length=0.1, snap_mode="midpoint")

# Call clearance helpers directly (accept enums or strings)
fixed = fix_hole_too_close(collapsed, min_clearance=1.0, strategy="shrink")

# Use merge strategies without going through merge_close_polygons
merged = merge_selective_buffer([fixed, fixed.translate(1.5, 0)], margin=0.5, preserve_holes=True)
```

These ops modules expose the same functions the public API uses internally, so they’re ideal for custom workflows or experimentation.

### Using the ops modules directly
The public API (`polyforge.simplify`, `polyforge.clearance`, `polyforge.merge`) covers most workflows, but the ops layer is available if you need fine-grained control (e.g., custom pipelines, bulk transformations). A few quick examples:

```python
from shapely.geometry import Polygon
import numpy as np
from polyforge.process import process_geometry
from polyforge.ops.simplify_ops import snap_short_edges

poly = Polygon([(0, 0), (1, 0.01), (2, 0), (2, 2), (0, 2)])
# Call the raw NumPy helper via process_geometry
short_edges_collapsed = process_geometry(
    poly,
    snap_short_edges,
    min_length=0.1,
    snap_mode="midpoint",
)
```

```python
from polyforge.ops.clearance import fix_hole_too_close

result = fix_hole_too_close(poly, min_clearance=1.0, strategy="shrink")
```

```python
from polyforge.ops.merge import merge_selective_buffer

merged = merge_selective_buffer([poly1, poly2], margin=1.0, preserve_holes=True)
```

Use these helpers when you need to compose Polyforge primitives in ways the high-level wrappers don’t cover. For everything else, prefer the public API (`polyforge.simplify`, `polyforge.clearance`, `polyforge.merge`, `polyforge.repair`).

## Running Tests
```bash
python -m pytest -q
```
The suite asserts all expected warnings, so any output indicates a regression.

That’s it—import what you need from `polyforge` and combine the high-level functions to build your own geometry pipelines.
