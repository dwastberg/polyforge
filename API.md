# Polyforge API Reference

Comprehensive documentation for all functions, enums, and exceptions in the polyforge library.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [API Functions](#api-functions)
   - [Simplification](#simplification-functions)
   - [Clearance Fixing](#clearance-fixing-functions)
   - [Overlap Handling](#overlap-handling-functions)
   - [Merge Operations](#merge-operations)
   - [Topology](#topology-operations)
   - [Geometry Repair](#geometry-repair-functions)
4. [Strategy Enums](#strategy-enums)
5. [Exceptions](#exceptions)
6. [Strategy Selection Guide](#strategy-selection-guide)
7. [Performance Characteristics](#performance-characteristics)
8. [Error Handling](#error-handling)
9. [Examples](#examples)

---

## Installation

```bash
pip install polyforge
```

**Requirements:**
- Python >= 3.10
- numpy >= 2.0.1
- scipy >= 1.15.3
- shapely >= 2.1.0
- simplification >= 0.7.14

---

## Quick Start

```python
from polyforge import (
    simplify_rdp,
    split_overlap,
    collapse_short_edges,
    merge_close_polygons,
    repair_geometry
)
from polyforge.core import MergeStrategy, RepairStrategy
from shapely.geometry import Polygon

# Simplify a polygon
poly = Polygon([(0, 0), (1, 0), (1.1, 0.1), (2, 0), (2, 2), (0, 2)])
simplified = simplify_rdp(poly, epsilon=0.2)

# Split overlapping polygons
poly1 = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
poly2 = Polygon([(2, 0), (5, 0), (5, 3), (2, 3)])
result1, result2 = split_overlap(poly1, poly2)

# Merge close polygons
buildings = [building1, building2, building3]
merged = merge_close_polygons(buildings, margin=2.0, merge_strategy=MergeStrategy.BOUNDARY_EXTENSION)

# Repair invalid geometry
fixed = repair_geometry(invalid_poly, repair_strategy=RepairStrategy.AUTO)
```

---

## API Functions

### Simplification Functions

#### `simplify_rdp(geometry, epsilon)`

Simplify geometry using the Ramer-Douglas-Peucker algorithm.

- **Args:**
  - `geometry` (BaseGeometry): Shapely geometry to simplify
  - `epsilon` (float): Tolerance value (larger = more simplification)

- **Returns:** Simplified geometry with fewer vertices

- **Algorithm:** RDP iteratively removes vertices that deviate less than epsilon from the line segment between neighboring vertices.

- **Performance:** O(n log n) - Fast, suitable for most use cases

**Example:**
```python
from polyforge import simplify_rdp
from shapely.geometry import Polygon

# Create a noisy polygon
poly = Polygon([
    (0, 0), (1, 0.01), (2, 0), (2, 1), (1.99, 2),
    (2, 3), (0, 3), (0.01, 2), (0, 1)
])

# Simplify with epsilon=0.1
simplified = simplify_rdp(poly, epsilon=0.1)
print(f"Vertices reduced: {len(poly.exterior.coords)} → {len(simplified.exterior.coords)}")
```

---

#### `simplify_vw(geometry, threshold)`

Simplify geometry using the Visvalingam-Whyatt algorithm.

- **Args:**
  - `geometry` (BaseGeometry): Shapely geometry to simplify
  - `threshold` (float): Area threshold (larger = more simplification)

- **Returns:** Simplified geometry with fewer vertices

- **Algorithm:** V-W removes vertices based on the area of triangles formed with neighboring vertices. Better visual results than RDP.

- **Performance:** O(n log n) - Slower than RDP but better shape preservation

**Example:**
```python
from polyforge import simplify_vw

# Simplify based on triangle area
simplified = simplify_vw(poly, threshold=0.5)
```

---

#### `simplify_vwp(geometry, threshold)`

Simplify geometry using topology-preserving Visvalingam-Whyatt.

- **Args:**
  - `geometry` (BaseGeometry): Shapely geometry to simplify
  - `threshold` (float): Area threshold (larger = more simplification)

- **Returns:** Simplified geometry, guaranteed to be topologically valid

- **Algorithm:** V-W with topology preservation - will not create self-intersections or invalid geometries.

- **Performance:** O(n log n) - Slowest but guarantees validity

**Example:**
```python
from polyforge import simplify_vwp

# Simplify with topology preservation guarantee
simplified = simplify_vwp(poly, threshold=0.5)
assert simplified.is_valid  # Always True
```

---

#### `collapse_short_edges(geometry, min_length, snap_mode='midpoint')`

Collapse edges shorter than min_length by snapping vertices together.

- **Args:**
  - `geometry` (BaseGeometry): Shapely geometry to process
  - `min_length` (float): Minimum edge length threshold
  - `snap_mode` (str or CollapseMode): How to snap vertices:
    - `'midpoint'`: Snap to midpoint of edge (default)
    - `'first'`: Snap to first vertex
    - `'last'`: Snap to second vertex

- **Returns:** Geometry with short edges collapsed

**Example:**
```python
from polyforge import collapse_short_edges
from polyforge.core import CollapseMode

# Remove edges shorter than 0.5 units
cleaned = collapse_short_edges(poly, min_length=0.5, snap_mode=CollapseMode.MIDPOINT)
```

---

#### `deduplicate_vertices(geometry, tolerance=1e-10)`

Remove consecutive duplicate vertices within tolerance.

- **Args:**
  - `geometry` (BaseGeometry): Shapely geometry to process
  - `tolerance` (float): Distance tolerance for considering vertices duplicates (default: 1e-10)

- **Returns:** Geometry with consecutive duplicates removed

**Example:**
```python
from polyforge import deduplicate_vertices

# Remove duplicate vertices
cleaned = deduplicate_vertices(poly, tolerance=1e-8)
```

---

#### `remove_small_holes(geometry, min_area)`

Remove holes (interior rings) smaller than min_area from Polygon geometries.

- **Args:**
  - `geometry` (Polygon or MultiPolygon): Polygon with holes
  - `min_area` (float): Minimum area threshold for holes

- **Returns:** Geometry with small holes removed

**Example:**
```python
from polyforge import remove_small_holes

# Remove holes smaller than 1.0 square units
cleaned = remove_small_holes(poly_with_holes, min_area=1.0)
```

---

### Clearance Fixing Functions

"Clearance" refers to the minimum distance a vertex can move before creating an invalid geometry. These functions fix geometries with low minimum clearance.

#### `fix_clearance(geometry, min_clearance, max_iterations=10)`

Automatically detect and fix clearance issues using multi-strategy approach.

- **Args:**
  - `geometry` (Polygon): Input polygon
  - `min_clearance` (float): Target minimum clearance
  - `max_iterations` (int): Maximum fix iterations (default: 10)

- **Returns:** Fixed polygon with clearance >= min_clearance

- **Strategy:** Auto-detects issue type (protrusion, hole, passage, intersection) and applies appropriate fix

**Example:**
```python
from polyforge import fix_clearance

# Auto-detect and fix all clearance issues
fixed = fix_clearance(problematic_poly, min_clearance=1.0, max_iterations=10)
```

---

#### `fix_hole_too_close(geometry, min_clearance, strategy='remove')`

Fix holes that are too close to the polygon exterior.

- **Args:**
  - `geometry` (Polygon): Polygon with holes
  - `min_clearance` (float): Target minimum clearance
  - `strategy` (str or HoleStrategy): How to handle close holes:
    - `'remove'`: Remove the hole entirely (default)
    - `'shrink'`: Shrink the hole to maintain clearance
    - `'move'`: Move the hole away from exterior

- **Returns:** Polygon with holes fixed

**Example:**
```python
from polyforge import fix_hole_too_close
from polyforge.core import HoleStrategy

exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
hole = [(1, 1), (2, 1), (2, 2), (1, 2)]  # Too close to edge
poly_with_hole = Polygon(exterior, [hole])

# Remove close holes
fixed = fix_hole_too_close(poly_with_hole, min_clearance=2.0, strategy=HoleStrategy.REMOVE)
```

---

#### `fix_narrow_protrusion(geometry, min_clearance, max_iterations=10)`

Remove narrow protrusions (spikes) from polygons.

- **Args:**
  - `geometry` (Polygon): Input polygon
  - `min_clearance` (float): Target minimum clearance
  - `max_iterations` (int): Maximum iterations (default: 10)

- **Returns:** Polygon with narrow protrusions removed

**Example:**
```python
from polyforge import fix_narrow_protrusion

# Fix a polygon with a narrow spike
spiky_poly = Polygon([
    (0, 0), (10, 0), (10, 10), (5, 15),  # Spike extends upward
    (5, 10.5), (5, 10), (0, 10)
])
fixed = fix_narrow_protrusion(spiky_poly, min_clearance=1.0)
```

---

#### `remove_narrow_protrusions(geometry, min_clearance, width_ratio=0.3)`

Detect and remove narrow protrusions in batch.

- **Args:**
  - `geometry` (Polygon): Input polygon
  - `min_clearance` (float): Minimum clearance threshold
  - `width_ratio` (float): Width-to-length ratio for protrusion detection (default: 0.3)

- **Returns:** Polygon with protrusions removed

**Example:**
```python
from polyforge import remove_narrow_protrusions

# Detect and remove all narrow protrusions
fixed = remove_narrow_protrusions(spiky_poly, min_clearance=1.0, width_ratio=0.3)
```

---

#### `fix_sharp_intrusion(geometry, min_clearance, strategy='fill')`

Fix sharp narrow intrusions (indentations) by filling or smoothing.

- **Args:**
  - `geometry` (Polygon): Input polygon
  - `min_clearance` (float): Target minimum clearance
  - `strategy` (str or IntrusionStrategy): How to fix:
    - `'fill'`: Fill the intrusion (default)
    - `'smooth'`: Smooth the intrusion
    - `'simplify'`: Simplify the region

- **Returns:** Polygon with intrusion fixed

**Example:**
```python
from polyforge import fix_sharp_intrusion
from polyforge.core import IntrusionStrategy

# Fill sharp intrusions
fixed = fix_sharp_intrusion(poly, min_clearance=1.0, strategy=IntrusionStrategy.FILL)
```

---

#### `fix_narrow_passage(geometry, min_clearance, strategy='widen')`

Fix narrow passages (hourglass shapes) by widening or splitting.

- **Args:**
  - `geometry` (Polygon): Input polygon
  - `min_clearance` (float): Target minimum clearance
  - `strategy` (str or PassageStrategy): How to handle:
    - `'widen'`: Widen the passage (default)
    - `'split'`: Split into separate polygons

- **Returns:** Fixed geometry (Polygon if widened, MultiPolygon if split)

**Example:**
```python
from polyforge import fix_narrow_passage
from polyforge.core import PassageStrategy

# Widen narrow passages
fixed = fix_narrow_passage(hourglass_poly, min_clearance=2.0, strategy=PassageStrategy.WIDEN)

# Or split at narrow points
split_result = fix_narrow_passage(hourglass_poly, min_clearance=2.0, strategy=PassageStrategy.SPLIT)
```

---

#### `fix_near_self_intersection(geometry, min_clearance, strategy='simplify')`

Fix near self-intersections by separating close edges.

- **Args:**
  - `geometry` (Polygon): Input polygon
  - `min_clearance` (float): Target minimum clearance
  - `strategy` (str or IntersectionStrategy): How to separate:
    - `'simplify'`: Simplify the region (default)
    - `'buffer'`: Use buffering
    - `'smooth'`: Smooth the edges

- **Returns:** Polygon with edges separated

**Example:**
```python
from polyforge import fix_near_self_intersection

# Fix nearly self-intersecting edges
fixed = fix_near_self_intersection(twisted_poly, min_clearance=1.0, strategy='simplify')
```

---

#### `fix_parallel_close_edges(geometry, min_clearance, strategy='simplify')`

Fix parallel edges that are too close together.

- **Args:**
  - `geometry` (Polygon): Input polygon
  - `min_clearance` (float): Target minimum clearance
  - `strategy` (str or EdgeStrategy): How to separate:
    - `'simplify'`: Simplify the region (default)
    - `'buffer'`: Use buffering

- **Returns:** Polygon with edges separated

**Example:**
```python
from polyforge import fix_parallel_close_edges

# Separate parallel edges
fixed = fix_parallel_close_edges(narrow_poly, min_clearance=2.0, strategy='simplify')
```

---

### Overlap Handling Functions

#### `split_overlap(poly1, poly2, overlap_strategy=OverlapStrategy.SPLIT)`

Split or assign the overlapping area between two polygons.

- **Args:**
  - `poly1` (Polygon): First polygon
  - `poly2` (Polygon): Second polygon
  - `overlap_strategy` (OverlapStrategy): How to handle the overlap:
    - `OverlapStrategy.SPLIT`: Split overlap 50/50 (default)
    - `OverlapStrategy.LARGEST`: Assign entire overlap to larger polygon
    - `OverlapStrategy.SMALLEST`: Assign entire overlap to smaller polygon

- **Returns:** Tuple of (modified_poly1, modified_poly2) that touch but don't overlap

- **Behavior:**
  - If polygons don't overlap, returns originals unchanged
  - If one contains the other, returns originals unchanged
  - Otherwise, resolves overlap according to strategy

**Example:**
```python
from polyforge import split_overlap
from polyforge.core import OverlapStrategy

# Two overlapping squares
poly1 = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
poly2 = Polygon([(2, 0), (5, 0), (5, 3), (2, 3)])

# Split overlap 50/50
result1, result2 = split_overlap(poly1, poly2, overlap_strategy=OverlapStrategy.SPLIT)
print(f"Overlap area: {result1.intersection(result2).area}")  # ~0.0
print(f"Polygons touch: {result1.intersects(result2)}")  # True

# Assign to largest
result1, result2 = split_overlap(poly1, poly2, overlap_strategy=OverlapStrategy.LARGEST)
```

---

#### `remove_overlaps(polygons, overlap_strategy=OverlapStrategy.SPLIT, max_iterations=10)`

Batch overlap removal for many polygons using spatial indexing.

- **Args:**
  - `polygons` (List[Polygon]): List of polygons to process
  - `overlap_strategy` (OverlapStrategy): How to handle overlaps (default: SPLIT)
  - `max_iterations` (int): Maximum iterations for resolution (default: 10)

- **Returns:** List of polygons with overlaps resolved

- **Algorithm:**
  1. Build spatial index (STRtree) - O(n log n)
  2. Find overlapping pairs using index
  3. Select independent pairs (no polygon appears twice)
  4. Resolve pairs in parallel
  5. Repeat until no overlaps or max_iterations reached

- **Performance:** Typical convergence in 1-5 iterations. Handles 1000+ polygons efficiently.

**Example:**
```python
from polyforge import remove_overlaps
from polyforge.core import OverlapStrategy

# Large dataset with many overlaps
many_polygons = [...]  # List of 1000 polygons

# Efficiently remove all overlaps
fixed = remove_overlaps(
    many_polygons,
    overlap_strategy=OverlapStrategy.SPLIT,
    max_iterations=10
)

print(f"Input: {len(many_polygons)} polygons")
print(f"Output: {len(fixed)} polygons")
print(f"All valid: {all(p.is_valid for p in fixed)}")
```

---

#### `count_overlaps(polygons, min_area_threshold=1e-10) -> int`

Count the number of overlapping pairs in a polygon dataset.

- **Args:**
  - `polygons` (List[Polygon]): List of polygons
  - `min_area_threshold` (float): Minimum overlap area to count (default: 1e-10)

- **Returns:** Integer count of overlapping pairs

- **Performance:** Uses spatial indexing - O(n log n)

**Example:**
```python
from polyforge import count_overlaps

overlap_count = count_overlaps(polygons, min_area_threshold=0.01)
print(f"Found {overlap_count} overlapping pairs")
```

---

#### `find_overlapping_groups(polygons, min_area_threshold=1e-10) -> List[List[int]]`

Find groups of mutually overlapping polygons.

- **Args:**
  - `polygons` (List[Polygon]): List of polygons
  - `min_area_threshold` (float): Minimum overlap area to consider (default: 1e-10)

- **Returns:** List of groups, where each group is a list of polygon indices

**Example:**
```python
from polyforge import find_overlapping_groups

# Find which polygons overlap
groups = find_overlapping_groups(polygons, min_area_threshold=0.01)

for i, group in enumerate(groups):
    if len(group) > 1:
        print(f"Group {i}: Polygons {group} overlap with each other")
    else:
        print(f"Polygon {group[0]} is isolated")
```

---

### Merge Operations

#### `merge_close_polygons(polygons, margin=0.0, merge_strategy=MergeStrategy.SELECTIVE_BUFFER, preserve_holes=True, return_mapping=False, insert_vertices=False)`

Merge polygons that overlap or are within margin distance.

- **Args:**
  - `polygons` (List[Polygon]): List of input polygons
  - `margin` (float): Maximum distance for merging (0.0 = only overlapping)
  - `merge_strategy` (MergeStrategy): Merging strategy (see strategies below)
  - `preserve_holes` (bool): Whether to preserve interior holes (default: True)
  - `return_mapping` (bool): If True, return (merged_polygons, groups) tuple (default: False)
  - `insert_vertices` (bool): Insert vertices at optimal connection points (default: False)

- **Returns:**
  - If `return_mapping=False`: List[Polygon]
  - If `return_mapping=True`: Tuple[List[Polygon], List[List[int]]] where groups[i] contains indices of original polygons merged

- **Performance:**
  - Uses spatial indexing (STRtree) for O(n log n) performance
  - Most isolated polygons returned unchanged (fast path)
  - Only processes close polygon groups

**Merge Strategies:**

1. **`MergeStrategy.SIMPLE_BUFFER`**: Fast expand-contract (changes shape)
2. **`MergeStrategy.SELECTIVE_BUFFER`**: Only buffer near gaps (good balance, default)
3. **`MergeStrategy.VERTEX_MOVEMENT`**: Move vertices toward each other (precise)
4. **`MergeStrategy.BOUNDARY_EXTENSION`**: Extend parallel edges (best for buildings)
5. **`MergeStrategy.CONVEX_BRIDGES`**: Use convex hull bridges (smooth connections)

**Example:**
```python
from polyforge import merge_close_polygons
from polyforge.core import MergeStrategy

# Merge overlapping building footprints
buildings = [building1, building2, building3]

# Use boundary extension for architectural features
merged = merge_close_polygons(
    buildings,
    margin=2.0,
    merge_strategy=MergeStrategy.BOUNDARY_EXTENSION,
    preserve_holes=True
)

# Track which polygons were merged
merged, groups = merge_close_polygons(
    buildings,
    margin=2.0,
    return_mapping=True
)

for i, (poly, source_indices) in enumerate(zip(merged, groups)):
    if len(source_indices) > 1:
        print(f"Polygon {i} was merged from {len(source_indices)} original polygons")
```

---

### Topology Operations

#### `align_boundaries(polygons, distance_tolerance=1e-10) -> List[Polygon]`

Create conforming boundaries between adjacent polygons.

- **Args:**
  - `polygons` (List[Polygon]): List of adjacent polygons
  - `distance_tolerance` (float): Distance tolerance for snapping (default: 1e-10)

- **Returns:** List of polygons with aligned boundaries

- **Purpose:** Ensures adjacent polygons share exact boundary vertices, eliminating slivers and gaps in polygon networks.

**Example:**
```python
from polyforge import align_boundaries

# Create conforming boundaries for adjacent parcels
parcels = [parcel1, parcel2, parcel3]
aligned = align_boundaries(parcels, distance_tolerance=1e-8)

# Boundaries now match exactly
assert aligned[0].boundary.intersects(aligned[1].boundary)
```

---

### Geometry Repair Functions

#### `repair_geometry(geometry, repair_strategy=RepairStrategy.AUTO, min_area_threshold=0.0, tolerance=1e-10, verbose=False)`

Validate and repair invalid geometries using multi-strategy approach.

- **Args:**
  - `geometry` (BaseGeometry): Input geometry to repair
  - `repair_strategy` (RepairStrategy): Repair strategy to use
  - `min_area_threshold` (float): Minimum area for valid result (default: 0.0)
  - `tolerance` (float): Tolerance for coordinate cleaning (default: 1e-10)
  - `verbose` (bool): Print diagnostic information (default: False)

- **Returns:** Repaired geometry (same type as input)

- **Raises:** `RepairError` if repair fails

**Repair Strategies:**

1. **`RepairStrategy.AUTO`**: Try multiple strategies automatically (default)
2. **`RepairStrategy.BUFFER`**: Use buffer(0) trick to fix topology
3. **`RepairStrategy.SIMPLIFY`**: Simplify geometry to remove issues
4. **`RepairStrategy.RECONSTRUCT`**: Rebuild geometry from scratch
5. **`RepairStrategy.STRICT`**: Only apply conservative fixes

**Example:**
```python
from polyforge import repair_geometry
from polyforge.core import RepairStrategy, RepairError

# Auto-detect and repair
try:
    fixed = repair_geometry(
        invalid_poly,
        repair_strategy=RepairStrategy.AUTO,
        min_area_threshold=1.0,
        verbose=True
    )
    print(f"Repair successful: {fixed.is_valid}")
except RepairError as e:
    print(f"Repair failed: {e}")
    print(f"Suggested strategy: {e.suggested_strategy}")
```

---

#### `analyze_geometry(geometry) -> dict`

Perform diagnostic analysis of geometry to identify issues.

- **Args:**
  - `geometry` (BaseGeometry): Geometry to analyze

- **Returns:** Dictionary with diagnostic information:
  - `'is_valid'`: bool
  - `'issues'`: List[str] of detected issues
  - `'geometry_type'`: str
  - `'area'`: float (if applicable)
  - `'minimum_clearance'`: float (if applicable)
  - `'suggested_strategy'`: RepairStrategy or None

**Example:**
```python
from polyforge import analyze_geometry

# Analyze before repairing
report = analyze_geometry(problematic_poly)

print(f"Valid: {report['is_valid']}")
print(f"Issues: {report['issues']}")
print(f"Minimum clearance: {report['minimum_clearance']}")
print(f"Suggested fix: {report['suggested_strategy']}")

# Use suggested strategy
fixed = repair_geometry(problematic_poly, repair_strategy=report['suggested_strategy'])
```

---

#### `batch_repair_geometries(geometries, repair_strategy=RepairStrategy.AUTO, min_area_threshold=0.0, skip_on_error=True) -> Tuple[List[BaseGeometry], List[Exception]]`

Batch repair multiple geometries with error handling.

- **Args:**
  - `geometries` (List[BaseGeometry]): List of geometries to repair
  - `repair_strategy` (RepairStrategy): Strategy to use (default: AUTO)
  - `min_area_threshold` (float): Minimum area threshold (default: 0.0)
  - `skip_on_error` (bool): If True, skip failed repairs; if False, raise on first error (default: True)

- **Returns:** Tuple of (repaired_geometries, errors) where:
  - `repaired_geometries`: List of repaired geometries (or originals if skip_on_error=True)
  - `errors`: List of exceptions (empty if all succeeded)

**Example:**
```python
from polyforge import batch_repair_geometries
from polyforge.core import RepairStrategy

# Repair many geometries
geometries = [poly1, poly2, poly3, ..., poly1000]

repaired, errors = batch_repair_geometries(
    geometries,
    repair_strategy=RepairStrategy.AUTO,
    skip_on_error=True
)

print(f"Repaired: {len(repaired)} geometries")
print(f"Errors: {len(errors)} failures")

for i, error in enumerate(errors):
    print(f"Geometry {i} failed: {error}")
```

---

## Strategy Enums

### OverlapStrategy

How to handle overlapping area between two polygons.

```python
from polyforge.core import OverlapStrategy

class OverlapStrategy(Enum):
    SPLIT = "split"       # Split overlap 50/50 between polygons
    LARGEST = "largest"   # Assign entire overlap to larger polygon
    SMALLEST = "smallest" # Assign entire overlap to smaller polygon
```

**When to use:**
- `SPLIT`: Fair division, maintain relative sizes
- `LARGEST`: Preserve dominant features, absorb small overlaps
- `SMALLEST`: Grow smaller features, fill gaps

---

### MergeStrategy

Algorithm for merging close or overlapping polygons.

```python
from polyforge.core import MergeStrategy

class MergeStrategy(Enum):
    SIMPLE_BUFFER = "simple_buffer"           # Fast expand-contract
    SELECTIVE_BUFFER = "selective_buffer"     # Only buffer near gaps (default)
    VERTEX_MOVEMENT = "vertex_movement"       # Move vertices toward each other
    BOUNDARY_EXTENSION = "boundary_extension" # Extend parallel edges
    CONVEX_BRIDGES = "convex_bridges"         # Convex hull bridges
```

**When to use:**
- `SIMPLE_BUFFER`: Fastest, acceptable shape distortion
- `SELECTIVE_BUFFER`: Good balance of speed and quality (default)
- `VERTEX_MOVEMENT`: Precise shape preservation, slower
- `BOUNDARY_EXTENSION`: Architectural features (buildings, parcels)
- `CONVEX_BRIDGES`: Smooth connections, irregular gaps

---

### RepairStrategy

Strategy for repairing invalid geometries.

```python
from polyforge.core import RepairStrategy

class RepairStrategy(Enum):
    AUTO = "auto"               # Try multiple strategies automatically
    BUFFER = "buffer"           # Use buffer(0) trick
    SIMPLIFY = "simplify"       # Simplify to remove issues
    RECONSTRUCT = "reconstruct" # Rebuild from scratch
    STRICT = "strict"           # Only conservative fixes
```

**When to use:**
- `AUTO`: Default, tries strategies in order until success
- `BUFFER`: Fast, fixes most topology issues
- `SIMPLIFY`: When buffer fails, reduces complexity
- `RECONSTRUCT`: Severe issues, last resort
- `STRICT`: Critical data, minimal modification

---

### SimplifyAlgorithm

Simplification algorithm selection.

```python
from polyforge.core import SimplifyAlgorithm

class SimplifyAlgorithm(Enum):
    RDP = "rdp"   # Ramer-Douglas-Peucker (fast)
    VW = "vw"     # Visvalingam-Whyatt (better visual)
    VWP = "vwp"   # V-W topology-preserving (guaranteed valid)
```

---

### CollapseMode

How to collapse short edges.

```python
from polyforge.core import CollapseMode

class CollapseMode(Enum):
    MIDPOINT = "midpoint"  # Snap to edge midpoint
    FIRST = "first"        # Snap to first vertex
    LAST = "last"          # Snap to second vertex
```

---

### Clearance Strategy Enums

Five enums for different clearance fixing operations:

```python
from polyforge.core import (
    HoleStrategy,
    PassageStrategy,
    IntrusionStrategy,
    IntersectionStrategy,
    EdgeStrategy
)

class HoleStrategy(Enum):
    REMOVE = "remove"  # Remove close holes
    SHRINK = "shrink"  # Shrink holes to maintain clearance
    MOVE = "move"      # Move holes away from exterior

class PassageStrategy(Enum):
    WIDEN = "widen"  # Widen narrow passages
    SPLIT = "split"  # Split at narrow points

class IntrusionStrategy(Enum):
    FILL = "fill"           # Fill intrusions
    SMOOTH = "smooth"       # Smooth intrusions
    SIMPLIFY = "simplify"   # Simplify region

class IntersectionStrategy(Enum):
    SIMPLIFY = "simplify"  # Simplify region
    BUFFER = "buffer"      # Use buffering
    SMOOTH = "smooth"      # Smooth edges

class EdgeStrategy(Enum):
    SIMPLIFY = "simplify"  # Simplify region
    BUFFER = "buffer"      # Use buffering
```

---

## Exceptions

Polyforge provides a hierarchy of custom exceptions for precise error handling.

### PolyforgeError (Base Exception)

Base exception for all polyforge errors.

```python
from polyforge.core import PolyforgeError

try:
    result = some_polyforge_function(geometry)
except PolyforgeError as e:
    print(f"Polyforge error: {e}")
```

---

### ValidationError

Raised when input geometry is invalid or doesn't meet requirements.

```python
from polyforge.core import ValidationError

class ValidationError(PolyforgeError):
    """Input geometry validation failed."""
```

**Attributes:**
- `geometry`: The invalid geometry
- `reason`: String describing validation failure

**Example:**
```python
try:
    result = repair_geometry(geometry)
except ValidationError as e:
    print(f"Validation failed: {e.reason}")
    print(f"Invalid geometry type: {e.geometry.geom_type}")
```

---

### RepairError

Raised when geometry repair fails.

```python
from polyforge.core import RepairError

class RepairError(PolyforgeError):
    """Geometry repair operation failed."""
```

**Attributes:**
- `geometry`: The geometry that couldn't be repaired
- `strategy`: The repair strategy that failed
- `suggested_strategy`: Alternative strategy to try

**Example:**
```python
try:
    fixed = repair_geometry(poly, repair_strategy=RepairStrategy.BUFFER)
except RepairError as e:
    print(f"Buffer repair failed, trying: {e.suggested_strategy}")
    fixed = repair_geometry(poly, repair_strategy=e.suggested_strategy)
```

---

### OverlapResolutionError

Raised when overlap resolution fails.

```python
from polyforge.core import OverlapResolutionError

class OverlapResolutionError(PolyforgeError):
    """Overlap resolution operation failed."""
```

**Attributes:**
- `polygon_indices`: Indices of polygons involved in failed resolution
- `iteration`: Iteration number where failure occurred

---

### MergeError

Raised when polygon merge fails.

```python
from polyforge.core import MergeError

class MergeError(PolyforgeError):
    """Polygon merge operation failed."""
```

**Attributes:**
- `polygon_count`: Number of polygons involved in merge
- `strategy`: Merge strategy that failed

---

### ClearanceError

Raised when clearance fixing fails.

```python
from polyforge.core import ClearanceError

class ClearanceError(PolyforgeError):
    """Clearance fixing operation failed."""
```

**Attributes:**
- `geometry`: The geometry with clearance issues
- `min_clearance`: Target clearance value
- `actual_clearance`: Actual clearance achieved

---

### ConfigurationError

Raised when function parameters are invalid.

```python
from polyforge.core import ConfigurationError

class ConfigurationError(PolyforgeError):
    """Invalid configuration or parameters."""
```

**Example:**
```python
try:
    result = merge_close_polygons(polygons, margin=-1.0)  # Invalid
except ConfigurationError as e:
    print(f"Invalid parameter: {e}")
```

---

## Strategy Selection Guide

### Choosing a Merge Strategy

| Use Case | Recommended Strategy | Reason |
|----------|---------------------|--------|
| Building footprints | `BOUNDARY_EXTENSION` | Preserves straight edges and right angles |
| Natural features | `CONVEX_BRIDGES` | Smooth, organic connections |
| Fast processing | `SIMPLE_BUFFER` | Fastest, acceptable distortion |
| General purpose | `SELECTIVE_BUFFER` | Good balance (default) |
| Precise control | `VERTEX_MOVEMENT` | Minimal shape change |

**Example Decision Tree:**
```python
from polyforge import merge_close_polygons
from polyforge.core import MergeStrategy

# Choose strategy based on feature type
if feature_type == "building":
    merge_strategy = MergeStrategy.BOUNDARY_EXTENSION
elif feature_type == "natural":
    merge_strategy = MergeStrategy.CONVEX_BRIDGES
elif performance_critical:
    merge_strategy = MergeStrategy.SIMPLE_BUFFER
else:
    merge_strategy = MergeStrategy.SELECTIVE_BUFFER  # Default

merged = merge_close_polygons(polygons, margin=2.0, merge_strategy=merge_strategy)
```

---

### Choosing a Repair Strategy

| Problem Type | Recommended Strategy | Notes |
|-------------|---------------------|-------|
| Unknown issues | `AUTO` | Tries strategies automatically |
| Self-intersection | `BUFFER` | Fast, effective for topology |
| Too many vertices | `SIMPLIFY` | Reduces complexity |
| Severe corruption | `RECONSTRUCT` | Last resort |
| Critical data | `STRICT` | Minimal modification |

**Example Workflow:**
```python
from polyforge import analyze_geometry, repair_geometry
from polyforge.core import RepairStrategy

# 1. Analyze first
report = analyze_geometry(geometry)

# 2. Use suggested strategy
if report['suggested_strategy']:
    fixed = repair_geometry(geometry, repair_strategy=report['suggested_strategy'])
else:
    # 3. Try AUTO if no specific suggestion
    fixed = repair_geometry(geometry, repair_strategy=RepairStrategy.AUTO)
```

---

### Choosing an Overlap Strategy

| Goal | Recommended Strategy | Use Case |
|------|---------------------|----------|
| Fair division | `SPLIT` | Equal importance, maintain proportions |
| Preserve large features | `LARGEST` | Cadastral, absorb measurement errors |
| Grow small features | `SMALLEST` | Fill gaps, expand patches |

**Example:**
```python
from polyforge import split_overlap, remove_overlaps
from polyforge.core import OverlapStrategy

# Cadastral: large parcels absorb overlaps from measurement errors
fixed_parcels = remove_overlaps(parcels, overlap_strategy=OverlapStrategy.LARGEST)

# Fair division: split contested areas 50/50
poly1, poly2 = split_overlap(poly1, poly2, overlap_strategy=OverlapStrategy.SPLIT)
```

---

## Performance Characteristics

### Spatial Indexing

Operations using spatial indexing (STRtree):
- **`remove_overlaps()`**: O(n log n) vs O(n²) naive
- **`merge_close_polygons()`**: O(n log n) for grouping
- **`find_overlapping_groups()`**: O(n log n)

**Typical Performance:**
- 100 polygons: ~5ms
- 1,000 polygons: ~50ms
- 5,000 polygons: ~200ms
- 10,000 polygons: ~500ms

---

### Merge Strategy Performance

| Strategy | Speed | Quality | Best For |
|----------|-------|---------|----------|
| SIMPLE_BUFFER | ⚡⚡⚡ | ⭐⭐ | Speed-critical |
| SELECTIVE_BUFFER | ⚡⚡ | ⭐⭐⭐ | General use (default) |
| VERTEX_MOVEMENT | ⚡ | ⭐⭐⭐⭐ | Precision-critical |
| BOUNDARY_EXTENSION | ⚡⚡ | ⭐⭐⭐⭐ | Architectural |
| CONVEX_BRIDGES | ⚡⚡ | ⭐⭐⭐⭐ | Natural features |

---

### Memory Considerations

- **Spatial indexing**: ~50-100 bytes per polygon overhead
- **3D coordinates**: ~20% memory increase vs 2D
- **Batch operations**: Process in chunks for datasets > 50,000 polygons

---

## Error Handling

### Exception Handling Patterns

```python
from polyforge import repair_geometry
from polyforge.core import ValidationError, RepairError, PolyforgeError

# Specific exception handling
try:
    fixed = repair_geometry(geometry)
except ValidationError as e:
    print(f"Invalid input: {e.reason}")
except RepairError as e:
    print(f"Repair failed, try: {e.suggested_strategy}")
except PolyforgeError as e:
    print(f"General error: {e}")

# Batch operations with error collection
from polyforge import batch_repair_geometries

repaired, errors = batch_repair_geometries(
    geometries,
    skip_on_error=True  # Collect errors instead of raising
)

# Process errors
for i, error in enumerate(errors):
    if isinstance(error, RepairError):
        print(f"Geometry {i}: {error.suggested_strategy}")
```

---

### Silent Failures vs Exceptions

**Functions that return originals on failure (no exception):**
- `split_overlap()`: Returns originals if no overlap or containment
- Most clearance functions: Return original if fix fails

**Functions that raise exceptions:**
- `repair_geometry()`: Raises `RepairError` if repair fails
- `merge_close_polygons()`: Raises `MergeError` on critical failures
- Validation functions: Raise `ValidationError` for invalid inputs

---

## Examples

### Example 1: Complete Geometry Cleaning Pipeline

```python
from polyforge import (
    deduplicate_vertices,
    collapse_short_edges,
    simplify_rdp,
    repair_geometry,
    fix_clearance
)
from polyforge.core import RepairStrategy

# Step 1: Clean duplicates
poly = deduplicate_vertices(raw_poly, tolerance=1e-8)

# Step 2: Remove short edges
poly = collapse_short_edges(poly, min_length=0.1)

# Step 3: Simplify
poly = simplify_rdp(poly, epsilon=0.5)

# Step 4: Repair if invalid
if not poly.is_valid:
    poly = repair_geometry(poly, repair_strategy=RepairStrategy.AUTO)

# Step 5: Fix clearance issues
poly = fix_clearance(poly, min_clearance=1.0)

print(f"Final: {poly.is_valid}, area={poly.area:.2f}")
```

---

### Example 2: Batch Processing with Error Handling

```python
from polyforge import batch_repair_geometries, remove_overlaps
from polyforge.core import RepairStrategy

# Step 1: Repair all geometries
geometries = [poly1, poly2, ..., poly1000]
repaired, errors = batch_repair_geometries(
    geometries,
    repair_strategy=RepairStrategy.AUTO,
    skip_on_error=True
)

print(f"Repaired: {len(repaired)}, Errors: {len(errors)}")

# Step 2: Remove overlaps from valid geometries
valid = [g for g in repaired if g.is_valid]
fixed = remove_overlaps(valid)

print(f"Final: {len(fixed)} non-overlapping polygons")
```

---

### Example 3: Merge Buildings by Type

```python
from polyforge import merge_close_polygons
from polyforge.core import MergeStrategy

# Separate by building type
residential = [...]
commercial = [...]

# Merge each type with appropriate strategy
merged_residential = merge_close_polygons(
    residential,
    margin=3.0,
    merge_strategy=MergeStrategy.BOUNDARY_EXTENSION,  # Preserve straight edges
    preserve_holes=True
)

merged_commercial = merge_close_polygons(
    commercial,
    margin=5.0,
    merge_strategy=MergeStrategy.CONVEX_BRIDGES,  # Smooth complex shapes
    preserve_holes=True
)

all_buildings = merged_residential + merged_commercial
```

---

### Example 4: Progressive Simplification

```python
from polyforge import simplify_rdp, simplify_vw, simplify_vwp

# Try progressively aggressive simplification
original_vertices = len(poly.exterior.coords)

# Conservative: topology-preserving
result = simplify_vwp(poly, threshold=0.5)
if len(result.exterior.coords) > original_vertices * 0.5:
    # More aggressive: visual quality
    result = simplify_vw(poly, threshold=1.0)

if len(result.exterior.coords) > original_vertices * 0.3:
    # Most aggressive: speed
    result = simplify_rdp(poly, epsilon=2.0)

print(f"Reduced: {original_vertices} → {len(result.exterior.coords)} vertices")
```

---

### Example 5: Clearance Fixing Workflow

```python
from polyforge import (
    analyze_geometry,
    fix_narrow_protrusion,
    fix_hole_too_close,
    fix_narrow_passage
)

# Analyze issues
report = analyze_geometry(poly)
print(f"Minimum clearance: {report['minimum_clearance']}")
print(f"Issues: {report['issues']}")

# Fix specific issues
if "narrow protrusion" in report['issues']:
    poly = fix_narrow_protrusion(poly, min_clearance=1.0)

if "hole too close" in report['issues']:
    poly = fix_hole_too_close(poly, min_clearance=1.0)

if "narrow passage" in report['issues']:
    poly = fix_narrow_passage(poly, min_clearance=1.0)

# Verify
final_report = analyze_geometry(poly)
print(f"Final clearance: {final_report['minimum_clearance']}")
```

---

## Complete Function List

```python
from polyforge import (
    # Simplification (6)
    simplify_rdp,
    simplify_vw,
    simplify_vwp,
    collapse_short_edges,
    deduplicate_vertices,
    remove_small_holes,

    # Clearance Fixing (8)
    fix_clearance,
    fix_hole_too_close,
    fix_narrow_protrusion,
    remove_narrow_protrusions,
    fix_sharp_intrusion,
    fix_narrow_passage,
    fix_near_self_intersection,
    fix_parallel_close_edges,

    # Overlap Handling (4)
    split_overlap,
    remove_overlaps,
    count_overlaps,
    find_overlapping_groups,

    # Merge (1)
    merge_close_polygons,

    # Topology (1)
    align_boundaries,

    # Repair (3)
    repair_geometry,
    analyze_geometry,
    batch_repair_geometries,
)

from polyforge.core import (
    # Strategy Enums (10)
    OverlapStrategy,
    MergeStrategy,
    RepairStrategy,
    SimplifyAlgorithm,
    CollapseMode,
    HoleStrategy,
    PassageStrategy,
    IntrusionStrategy,
    IntersectionStrategy,
    EdgeStrategy,

    # Exceptions (7)
    PolyforgeError,
    ValidationError,
    RepairError,
    OverlapResolutionError,
    MergeError,
    ClearanceError,
    ConfigurationError,
)
```

---

## Requirements

- Python >= 3.10
- numpy >= 2.0.1
- scipy >= 1.15.3
- shapely >= 2.1.0
- simplification >= 0.7.14

---

## See Also

- **[README.md](README.md)**: Project overview and quick start
- **[examples/](examples/)**: Comprehensive usage examples
- **[tests/](tests/)**: Test suite for all functions

---

**Last Updated:** January 2025 • **Version:** 0.1.0
