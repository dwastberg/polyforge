# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Polyforge is a polygon processing and manipulation library built on top of Shapely. It provides utilities for:
- Geometry simplification (RDP, VW, VWP algorithms)
- Overlap detection and resolution at scale (using spatial indexing)
- Polygon merging with 5 strategies
- Clearance fixing for low-clearance geometries
- Geometry validation and repair with 5 strategies
- Vertex processing (collapse short edges, remove duplicates)
- Topology operations (boundary alignment)
- Batch processing with error handling

**Key Dependencies:** shapely, numpy, scipy, simplification library

**Important Design Principles:**
- **No backward compatibility code** - Library hasn't been released yet (v0.1.0)
- **Only enums for strategy parameters** - No string alternatives
- **Comprehensive error hierarchy** - All exceptions inherit from PolyforgeError
- **Explicit exports** - All modules define `__all__`
- **DRY architecture** - Shared utilities in core/ to eliminate duplication

## Development Commands

### Testing
```bash
# Run all tests (371 tests)
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_split.py -v

# Run specific test class or method
python -m pytest tests/test_overlap.py::TestRemoveOverlaps::test_batch_fix_all_valid -v

# Run with short traceback
python -m pytest tests/ --tb=short

# Quick run (quiet mode)
python -m pytest tests/ -q
```

### Running Examples
```bash
# Examples require PYTHONPATH set
PYTHONPATH=/Users/dwastberg/repos/polyforge:$PYTHONPATH python examples/remove_overlaps_demo.py
PYTHONPATH=/Users/dwastberg/repos/polyforge:$PYTHONPATH python examples/fix_geometry_demo.py
PYTHONPATH=/Users/dwastberg/repos/polyforge:$PYTHONPATH python examples/merge_close_polygons_demo.py
```

### Python Environment
- Requires Python >= 3.10
- Uses uv for dependency management (see uv.lock)

## Architecture

### Module Organization

The codebase follows a clean, hierarchical structure with shared utilities:

```
polyforge/
├── __init__.py                 # Public API exports (24 functions, 10 enums, 7 exceptions)
├── core/                       # Core types, errors, and shared utilities
│   ├── __init__.py
│   ├── types.py                # Enum definitions (10 strategy enums)
│   ├── errors.py               # Exception hierarchy (7 exception classes)
│   ├── geometry_utils.py       # Shared geometry operations (to_single_polygon, remove_holes, calculate_internal_angles, etc.)
│   ├── validation_utils.py     # Shared validation patterns
│   ├── spatial_utils.py        # Spatial indexing utilities (STRtree operations, adjacency graphs)
│   └── iterative_utils.py      # Iterative improvement patterns
├── process.py                  # Core: process_geometry() - applies functions to geometry vertices
├── simplify.py                 # Simplification algorithms (6 functions)
├── split.py                    # Pairwise overlap splitting (split_overlap)
├── overlap.py                  # Batch overlap removal (remove_overlaps, count_overlaps, find_overlapping_groups)
├── topology.py                 # Boundary alignment operations (align_boundaries)
├── tile.py                     # Polygon tiling functions
├── merge/                      # Polygon merging subsystem
│   ├── __init__.py
│   ├── core.py                 # Orchestration: merge_close_polygons()
│   ├── strategies/             # 5 merge strategy implementations
│   │   ├── simple_buffer.py
│   │   ├── selective_buffer.py
│   │   ├── vertex_movement.py
│   │   ├── boundary_extension.py
│   │   └── convex_bridges.py
│   └── utils/                  # Merge-specific utilities
│       ├── boundary_analysis.py
│       ├── edge_detection.py
│       └── vertex_insertion.py
├── repair/                     # Geometry repair subsystem
│   ├── __init__.py
│   ├── core.py                 # Orchestration: repair_geometry(), batch_repair_geometries()
│   ├── analysis.py             # Diagnostic: analyze_geometry()
│   ├── strategies/             # 5 repair strategy implementations
│   │   ├── auto.py
│   │   ├── buffer.py
│   │   ├── simplify.py
│   │   ├── reconstruct.py
│   │   └── strict.py
│   └── utils/                  # Repair-specific utilities
└── clearance/                  # Clearance fixing subsystem (8 functions)
    ├── __init__.py
    ├── fix_clearance.py        # Auto-detection: fix_clearance()
    ├── utils.py                # Shared clearance utilities
    ├── holes.py                # fix_hole_too_close
    ├── protrusions.py          # fix_narrow_protrusion, fix_sharp_intrusion
    ├── remove_protrusions.py   # remove_narrow_protrusions
    └── passages.py             # fix_narrow_passage, fix_near_self_intersection, fix_parallel_close_edges
```

### Key Architectural Patterns

**1. Shared Utility Modules (DRY Architecture)**
- **Created to eliminate 200-250 lines of duplicated code**
- Located in `core/` directory for clear dependency hierarchy
- Four utility modules:
  - `geometry_utils.py`: Common geometry operations (to_single_polygon, remove_holes, validate_and_fix, calculate_internal_angles)
  - `validation_utils.py`: Validation patterns (is_valid_polygon, is_ring_closed)
  - `spatial_utils.py`: Spatial indexing (find_polygon_pairs, build_adjacency_graph, find_connected_components)
  - `iterative_utils.py`: Iterative improvement frameworks

**2. Enum-Based Strategy Parameters**
- All strategy parameters use enum types from `core/types.py`
- **NEVER use strings** - only enums (library not released, no backward compatibility)
- **Consistent naming**: Domain-specific parameter names (`merge_strategy`, `overlap_strategy`, `repair_strategy`)
- Example:
```python
from polyforge import merge_close_polygons
from polyforge.core import MergeStrategy

# Correct - use enums with domain-specific parameter name
result = merge_close_polygons(polys, margin=2.0, merge_strategy=MergeStrategy.SELECTIVE_BUFFER)

# Wrong - do NOT use strings or generic 'strategy' parameter
result = merge_close_polygons(polys, margin=2.0, strategy='selective_buffer')  # Will raise error!
```

**3. process_geometry() Pattern**
- Core abstraction in `process.py`
- Higher-order function that applies vertex-processing functions to any Shapely geometry
- Automatically handles 3D coordinates (preserves/interpolates Z values)
- Recursively processes MultiPolygon, GeometryCollection, etc.
- Used by all simplification functions

Example internal pattern:
```python
def some_public_function(geometry, param):
    return process_geometry(geometry, _internal_function, param=param)

def _internal_function(vertices: np.ndarray, param) -> np.ndarray:
    # Works on numpy arrays, returns numpy arrays
    # process_geometry handles Shapely conversion
```

**4. Spatial Indexing for Performance**
- `overlap.py` and `merge/core.py` use Shapely's STRtree for O(n log n) performance
- Shared implementation in `core/spatial_utils.py`
- Critical for handling 1000+ polygons efficiently
- Reduces candidate comparisons by 90-99% in typical cases
- Pattern: build index → query candidates → validate actual overlaps

**5. Strategy Pattern with Enums**
- Multiple modules use enum strategy parameters with **consistent naming**:
  - `split_overlap(..., overlap_strategy=OverlapStrategy.SPLIT)`
  - `repair_geometry(..., repair_strategy=RepairStrategy.AUTO)`
  - `merge_close_polygons(..., merge_strategy=MergeStrategy.SELECTIVE_BUFFER)`  ⚠️ Note: `merge_strategy` not `strategy`
  - `collapse_short_edges(..., snap_mode=CollapseMode.MIDPOINT)`

**6. Iterative Resolution**
- `remove_overlaps()` uses iterative algorithm:
  1. Build spatial index
  2. Find overlapping pairs
  3. Select independent pairs (no polygon appears twice)
  4. Resolve pairs in parallel
  5. Repeat until no overlaps or max_iterations
- Most cases converge in 1-5 iterations
- Default max_iterations: 100 for batch operations, 10 for single-geometry operations

**7. Exception Hierarchy**
- All custom exceptions inherit from `PolyforgeError`
- Specific exceptions for different error types:
  - `RepairError` - geometry repair failures
  - `ValidationError` - validation failures
  - `OverlapResolutionError` - overlap resolution failures
  - `MergeError` - merge operation failures
  - `ClearanceError` - clearance fixing failures
  - `ConfigurationError` - invalid parameters
- Exceptions carry metadata (geometry, strategies_tried, suggested_strategy, etc.)

### Important Implementation Details

**Coordinate Handling:**
- All internal processing functions work on numpy arrays (Nx2 for 2D, Nx3 for 3D)
- `process_geometry()` handles the Shapely ↔ numpy conversion
- 3D geometries: Z values preserved via interpolation based on 2D path distance
- Closed rings: First and last vertices must be identical

**Overlap Resolution:**
- `split_overlap()` handles pairwise overlaps
- `remove_overlaps()` handles many-to-many overlaps using spatial indexing
- Returns originals unchanged for: no overlap, containment, or touching-only

**Clearance Fixing:**
- "Clearance" = minimum distance a vertex can move before creating invalid geometry (Shapely's `minimum_clearance`)
- Each clearance module targets specific issue types (protrusions, holes, passages)
- Uses minimal geometric modifications to achieve target clearance
- Default max_iterations: 10

**Geometry Repair:**
- `repair_geometry()` tries multiple strategies in order: clean coords → buffer(0) → simplify → reconstruct
- Buffer(0) trick: often fixes self-intersections and topology errors
- Strict mode: only applies conservative fixes that preserve geometric intent
- Raises `RepairError` if geometry cannot be repaired
- `batch_repair_geometries()` processes many geometries with error collection

**Polygon Merging:**
- `merge_close_polygons()` merges polygons within specified margin distance
- Five strategies available: SIMPLE_BUFFER, SELECTIVE_BUFFER, VERTEX_MOVEMENT, BOUNDARY_EXTENSION, CONVEX_BRIDGES
- Uses spatial indexing (via `core/spatial_utils.py`) for efficient group detection
- Supports vertex insertion for improved merge quality
- Parameter name: `merge_strategy` (not `strategy`)

**Tolerance Parameters:**
- **Disambiguated for clarity:**
  - `min_area_threshold`: Minimum overlap area to consider (used in `count_overlaps`, `find_overlapping_groups`)
  - `distance_tolerance`: Distance tolerance for snapping (used in `align_boundaries`)
  - `tolerance`: Coordinate comparison tolerance (used in `repair_geometry`, `deduplicate_vertices`)

## Testing Conventions

### Test Organization
- One test file per module: `test_split.py`, `test_overlap.py`, `test_repair.py`, etc.
- Tests organized into classes by functionality
- Pattern: `TestFunctionName` for main tests, `TestEdgeCases` for edge cases
- **Total: 371 tests** (all passing as of latest API consistency update)

### Test Patterns Used
```python
# Standard pattern for geometry tests
def test_something():
    # Create geometry
    poly = Polygon([...])

    # Apply function
    result = some_function(poly)

    # Assertions
    assert result.is_valid
    assert result.area > 0
    assert abs(result.area - expected) < tolerance
```

### Running Subset of Tests
```bash
# Run all tests in a class
pytest tests/test_overlap.py::TestRemoveOverlaps -v

# Run all performance tests
pytest tests/test_overlap.py::TestPerformance -v

# Run merge strategy tests
pytest tests/test_merge.py::TestSelectiveBufferStrategy -v
```

## Code Patterns to Follow

### When Adding New Functions with Strategy Parameters
1. **Always use enum types** for strategy parameters
2. Define enum in `core/types.py` if it doesn't exist
3. **Use domain-specific parameter names**: `merge_strategy`, `overlap_strategy`, `repair_strategy` (not generic `strategy`)
4. Function signature pattern:
```python
from .core.types import MyStrategy

def my_function(geometry, my_strategy: MyStrategy = MyStrategy.DEFAULT):
    if my_strategy == MyStrategy.OPTION1:
        # handle option 1
    elif my_strategy == MyStrategy.OPTION2:
        # handle option 2
```
5. **Never accept `Union[str, EnumType]`** - only the enum type
6. Add to module's `__all__` list

### When Adding New Simplification Functions
1. Create private `_function_name(vertices: np.ndarray, ...) -> np.ndarray`
2. Create public wrapper using `process_geometry()`
3. Handle 2D coordinate arrays only (process_geometry handles 3D)
4. Export via `__all__` in module and `__init__.py`

### When Adding New Geometry Fixes
1. Add to appropriate clearance submodule or repair strategy
2. Follow minimal modification principle
3. Validate result before returning
4. Raise appropriate exception type (RepairError, ClearanceError, etc.)
5. Add diagnostic information for failures
6. Use standard `max_iterations=10` for single-geometry operations

### When Adding New Exception Types
1. Always inherit from `PolyforgeError` or one of its subclasses
2. Add exception to `core/errors.py`
3. Include relevant metadata as attributes (geometry, strategies, suggested_strategy, etc.)
4. Export via `core/__init__.py` and main `__init__.py`

### When Working with Large Datasets
- Always use spatial indexing (STRtree) for polygon-to-polygon operations
- Use shared utilities from `core/spatial_utils.py`:
  - `build_adjacency_graph()` for finding close polygon pairs
  - `find_connected_components()` for graph-based grouping
- Avoid O(n²) nested loops over polygons
- See `overlap.py` and `merge/core.py` for reference implementations

### When Adding Shared Utilities
- Place in appropriate `core/` module:
  - Geometry operations → `core/geometry_utils.py`
  - Validation → `core/validation_utils.py`
  - Spatial operations → `core/spatial_utils.py`
  - Iterative patterns → `core/iterative_utils.py`
- Prevents code duplication across modules
- Add to `__all__` for proper exports

## Performance Considerations

- Spatial indexing in `overlap.py` and `merge/core.py` handles 1000+ polygons efficiently
- Typical performance (spatial indexing):
  - 100 polygons: ~5ms
  - 1,000 polygons: ~50ms
  - 5,000 polygons: ~200ms
- For batch operations, use provided batch functions: `batch_repair_geometries()`, `remove_overlaps()`, `merge_close_polygons()`
- Z-coordinate interpolation adds ~10-20% overhead; use 2D when possible
- Simplification algorithms have different performance profiles:
  - RDP: Fast, good for most cases
  - VW: Slower, better visual results
  - VWP: Slowest, topology-preserving

## API Design Principles

### Parameter Naming Consistency
- **Strategy parameters use domain-specific names**:
  - `merge_strategy` (not `strategy`) for merge operations
  - `overlap_strategy` for overlap operations
  - `repair_strategy` for repair operations
- **Tolerance parameters are semantically clear**:
  - `min_area_threshold` for minimum overlap areas
  - `distance_tolerance` for distance-based snapping
  - `tolerance` for coordinate comparisons
- **Iteration parameters standardized**:
  - Single-geometry: `max_iterations=10`
  - Batch operations: `max_iterations=100`

### No Backward Compatibility
- Library has **not been released** yet (v0.1.0)
- No backward compatibility code exists
- Only one canonical way to use each feature
- Breaking changes are acceptable during development

### Enum Usage
- All strategy parameters use enum types
- Enums are defined in `core/types.py`:
  - `OverlapStrategy` - SPLIT, LARGEST, SMALLEST
  - `MergeStrategy` - SIMPLE_BUFFER, SELECTIVE_BUFFER, VERTEX_MOVEMENT, BOUNDARY_EXTENSION, CONVEX_BRIDGES
  - `RepairStrategy` - AUTO, BUFFER, SIMPLIFY, RECONSTRUCT, STRICT
  - `SimplifyAlgorithm` - RDP, VW, VWP
  - `CollapseMode` - MIDPOINT, FIRST, LAST
  - Plus 5 clearance-specific enums: `HoleStrategy`, `PassageStrategy`, `IntrusionStrategy`, `IntersectionStrategy`, `EdgeStrategy`
- Enums do NOT inherit from `str` (pure Enum types)
- Functions only accept enum types, not strings

### Error Handling
- Comprehensive exception hierarchy in `core/errors.py`
- All exceptions inherit from `PolyforgeError`
- Exceptions carry metadata for debugging
- Consistent error handling across all modules
- Some functions return originals on failure (e.g., `split_overlap`), others raise exceptions (e.g., `repair_geometry`)

## Common Gotchas

1. **Use Enums, Not Strings:** All strategy parameters require enum types from `core/types.py`, not strings
2. **Parameter Names Matter:** Use `merge_strategy` not `strategy`, `min_area_threshold` not `tolerance` (context-dependent)
3. **Closed Rings:** Polygon exterior/interior rings must have first == last vertex
4. **Empty Geometries:** Always check `geom.is_empty` before accessing properties
5. **MultiPolygon Results:** Many operations can return MultiPolygon even when input was Polygon
6. **Spatial Index Queries:** `tree.query()` returns candidates, must validate actual intersection
7. **Buffer Distance:** `buffer(0)` is not the same as `buffer(0.0)` in some Shapely versions
8. **Function Names:** Use `repair_geometry` not `fix_geometry`, `collapse_short_edges` not `snap_short_edges`
9. **Exception Names:** Use `RepairError` not `GeometryRepairError` or `GeometryFixError`
10. **Recent API Changes:** If you see old code using `strategy=`, `tolerance=` in merge/overlap/topology functions, update to new parameter names

## Documentation

### Comprehensive Documentation Available
- **README.md**: Project overview, quick start, architecture guide, performance notes
- **API.md**: Complete API reference for all 24 functions, 10 enums, 7 exceptions
  - Includes strategy selection guide
  - Performance characteristics
  - Error handling patterns
  - Complete examples
- **examples/**: Three comprehensive demo scripts
  - `fix_geometry_demo.py`
  - `remove_overlaps_demo.py`
  - `merge_close_polygons_demo.py`

### API Reference Quick Links
- All functions documented in API.md with:
  - Complete parameter descriptions
  - Return types
  - Usage examples
  - Strategy comparisons
  - Performance notes

## Quick Reference: Common Imports

```python
# Main functions
from polyforge import (
    # Simplification (6 functions)
    simplify_rdp, simplify_vw, simplify_vwp,
    collapse_short_edges, deduplicate_vertices, remove_small_holes,

    # Overlap handling (4 functions)
    split_overlap, remove_overlaps, count_overlaps, find_overlapping_groups,

    # Merging (1 function, 5 strategies)
    merge_close_polygons,

    # Repair (3 functions)
    repair_geometry, analyze_geometry, batch_repair_geometries,

    # Topology (1 function)
    align_boundaries,

    # Clearance fixing (8 functions)
    fix_clearance,
    fix_hole_too_close,
    fix_narrow_protrusion, remove_narrow_protrusions,
    fix_sharp_intrusion,
    fix_narrow_passage,
    fix_near_self_intersection,
    fix_parallel_close_edges,
)

# Strategy enums (10 enums)
from polyforge.core import (
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
)

# Exceptions (7 exceptions)
from polyforge.core import (
    PolyforgeError,
    ValidationError,
    RepairError,
    OverlapResolutionError,
    MergeError,
    ClearanceError,
    ConfigurationError,
)
```

## Recent Major Changes (for context)

### API Consistency Update (Breaking Changes)
Five breaking changes were made to improve API consistency:
1. `merge_close_polygons(..., strategy=)` → `merge_strategy=`
2. `count_overlaps(..., tolerance=)` → `min_area_threshold=`
3. `find_overlapping_groups(..., tolerance=)` → `min_area_threshold=`
4. `align_boundaries(..., tolerance=)` → `distance_tolerance=`
5. `fix_sharp_intrusion(..., max_iterations=5)` → default now 10

**All 371 tests updated and passing** after these changes.

### DRY Refactoring
Created four shared utility modules in `core/` to eliminate 200-250 lines of duplicated code:
- `geometry_utils.py` - Common geometry operations
- `validation_utils.py` - Validation patterns
- `spatial_utils.py` - Spatial indexing utilities
- `iterative_utils.py` - Iterative improvement frameworks

Key additions:
- `to_single_polygon()` - Convert MultiPolygon/GeometryCollection to largest Polygon
- `remove_holes()` - Unified hole preservation/removal
- `calculate_internal_angles()` - Calculate vertex angles for Polygon or LineString
- `build_adjacency_graph()` - Spatial indexing for polygon grouping
- `find_connected_components()` - DFS-based component finding

### Module Restructuring
- `merge.py` split into `merge/` package with strategies/ and utils/ subdirectories
- `fix.py` renamed to `repair/` package with strategies/ and utils/ subdirectories
- Improved separation of concerns and code organization
