# Debug/Demonstration Scripts

This directory contains visualization scripts that demonstrate the various geometry operations available in Polyforge.

## Running Scripts

Each script can be run directly:
```bash
python debug/simplify_vis.py
python debug/overlap_vis.py
# etc.
```

Scripts will open matplotlib windows showing before/after comparisons of the operations.

## Available Demonstrations

### Core Operations

- **`simplify_vis.py`** - Simplification algorithms
  - Ramer-Douglas-Peucker (RDP)
  - Visvalingam-Whyatt (VW)
  - Visvalingam-Whyatt Preserve Topology (VWP)

- **`overlap_vis.py`** - Overlap resolution
  - Different resolution strategies (split, largest, smallest)
  - Batch overlap removal for multiple polygons

- **`merge_vis.py`** - Polygon merging
  - All 5 merge strategies (simple_buffer, selective_buffer, vertex_movement, boundary_extension, convex_bridges)

- **`repair_vis.py`** - Geometry repair
  - Different repair strategies (auto, buffer, simplify, reconstruct)
  - Invalid geometry fixing (self-intersections, bowtie polygons)

- **`clearance_vis.py`** - Clearance fixing
  - Narrow passage fixing
  - Low-clearance geometry repair

### Advanced Operations

- **`robust_fix_vis.py`** - Constraint-driven fixing
  - Quality requirement enforcement
  - Iterative fixing with constraints
  - Hole cleanup with quality thresholds

- **`hole_cleanup_vis.py`** - Hole removal
  - Remove small holes by area
  - Remove narrow holes by aspect ratio
  - Remove narrow holes by width
  - Combined filtering

- **`topology_vis.py`** - Topology operations
  - Boundary alignment for adjacent polygons
  - Mesh alignment for multiple touching polygons

- **`vertex_ops_vis.py`** - Vertex-level operations
  - Deduplicate vertices
  - Collapse short edges (3 modes: midpoint, first, last)
  - Combined operations

## Utility Module

- **`plot_geometry.py`** - Plotting utilities
  - `plot_comparison()` - Side-by-side before/after visualization
  - Handles Polygon, MultiPolygon, and holes

## Creating New Demonstrations

To create a new demonstration script:

1. Copy the template structure from any existing script
2. Import necessary modules and set up path
3. Create example geometry that highlights the operation
4. Apply the operation and print diagnostics
5. Use `plot_comparison()` for visualization

Example template:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import polyforge
from shapely.geometry import Polygon
from plot_geometry import plot_comparison

# Create geometry
poly = Polygon([...])

# Apply operation
result = polyforge.some_operation(poly, param=value)

# Print diagnostics
print(f"Original: {poly.area:.2f}")
print(f"Result: {result.area:.2f}")

# Visualize
plot_comparison(poly, result, title="Operation Name")
```
