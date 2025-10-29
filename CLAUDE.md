# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Polyforge is a polygon processing and manipulation library built on top of Shapely. It provides utilities for:
- Geometry simplification (RDP, VW, VWP algorithms)
- Overlap detection and resolution at scale (using spatial indexing)
- Polygon merging with multiple strategies
- Clearance fixing for low-clearance geometries
- Geometry validation and repair
- Vertex processing (snap short edges, remove duplicates)
- Topology operations (boundary alignment)
- Polygon tiling

**Key Dependencies:** shapely, numpy, scipy, simplification library

**Important Design Principles:**
- **No backward compatibility code** - Library hasn't been released yet
- **Only enums for strategy parameters** - No string alternatives
- **Comprehensive error hierarchy** - All exceptions inherit from PolyforgeError
- **Explicit exports** - All modules define `__all__`

## Development Commands

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_split.py -v

# Run specific test class or method
python -m pytest tests/test_overlap.py::TestRemoveOverlaps::test_batch_fix_all_valid -v

# Run with short traceback
python -m pytest tests/ --tb=short
```

### Running Examples
```bash
# Examples require PYTHONPATH set
PYTHONPATH=/Users/dwastberg/repos/polyforge:$PYTHONPATH python examples/remove_overlaps_demo.py
PYTHONPATH=/Users/dwastberg/repos/polyforge:$PYTHONPATH python examples/fix_geometry_demo.py
```

### Python Environment
- Requires Python >= 3.10
- Uses uv for dependency management (see uv.lock)

## Architecture

### Module Organization

The codebase is organized into focused modules, each handling a specific aspect of polygon processing:

```
polyforge/
├── __init__.py          # Public API exports
├── core/                # Core types and exceptions
│   ├── __init__.py
│   ├── types.py         # Enum definitions (OverlapStrategy, MergeStrategy, RepairStrategy, etc.)
│   └── errors.py        # Exception hierarchy (PolyforgeError, RepairError, etc.)
├── process.py           # Core: process_geometry() - applies functions to geometry vertices
├── simplify.py          # Simplification algorithms (RDP, VW, VWP, collapse_short_edges)
├── split.py             # Pairwise overlap splitting (split_overlap)
├── overlap.py           # Batch overlap removal using spatial indexing
├── merge.py             # Polygon merging with multiple strategies
├── fix.py               # Geometry validation and repair (repair_geometry)
├── topology.py          # Boundary alignment operations
├── tile.py              # Polygon tiling functions
└── clearance/           # Clearance fixing (low minimum_clearance issues)
    ├── __init__.py
    ├── utils.py         # Shared utilities for clearance operations
    ├── holes.py         # fix_hole_too_close
    ├── protrusions.py   # fix_narrow_protrusion, fix_sharp_intrusion
    ├── remove_protrusions.py  # remove_narrow_protrusions
    ├── passages.py      # fix_narrow_passage, fix_near_self_intersection, fix_parallel_close_edges
    └── fix_clearance.py # fix_clearance (auto-detection), diagnose_clearance
```

### Key Architectural Patterns

**1. Enum-Based Strategy Parameters**
- All strategy parameters use enum types from `core/types.py`
- **NEVER use strings** - only enums (library not released, no backward compatibility)
- Example:
```python
from polyforge import repair_geometry
from polyforge.core.types import RepairStrategy

# Correct - use enums
result = repair_geometry(poly, repair_strategy=RepairStrategy.BUFFER)

# Wrong - do NOT use strings
result = repair_geometry(poly, repair_strategy='buffer')  # Will raise error!
```

**2. process_geometry() Pattern**
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

**3. Spatial Indexing for Performance**
- `overlap.py` and `merge.py` use Shapely's STRtree for O(n log n) performance
- Critical for handling 1000+ polygons efficiently
- Reduces candidate comparisons by 90-99% in typical cases
- Pattern: build index → query candidates → validate actual overlaps

**4. Strategy Pattern with Enums**
- Multiple modules use enum strategy parameters:
  - `split_overlap(..., overlap_strategy=OverlapStrategy.SPLIT)`
  - `repair_geometry(..., repair_strategy=RepairStrategy.AUTO)`
  - `merge_close_polygons(..., strategy=MergeStrategy.SELECTIVE_BUFFER)`
  - `collapse_short_edges(..., snap_mode=CollapseMode.MIDPOINT)`

**5. Iterative Resolution**
- `remove_overlaps()` uses iterative algorithm:
  1. Build spatial index
  2. Find overlapping pairs
  3. Select independent pairs (no polygon appears twice)
  4. Resolve pairs in parallel
  5. Repeat until no overlaps or max_iterations
- Most cases converge in 1-5 iterations

**6. Exception Hierarchy**
- All custom exceptions inherit from `PolyforgeError`
- Specific exceptions for different error types:
  - `RepairError` - geometry repair failures
  - `ValidationError` - validation failures
  - `OverlapResolutionError` - overlap resolution failures
  - `MergeError` - merge operation failures
  - `ClearanceError` - clearance fixing failures
  - `ConfigurationError` - invalid parameters
- Exceptions carry metadata (geometry, strategies_tried, etc.)

### Important Implementation Details

**Coordinate Handling:**
- All internal processing functions work on numpy arrays (Nx2 for 2D)
- `process_geometry()` handles the Shapely ↔ numpy conversion
- 3D geometries: Z values preserved via interpolation based on 2D path distance
- Closed rings: First and last vertices must be identical

**Overlap Resolution:**
- `split_overlap()` handles pairwise overlaps
- `remove_overlaps()` handles many-to-many overlaps using spatial indexing
- Returns originals unchanged for: no overlap, containment, or touching-only

**Clearance Fixing:**
- "Clearance" = minimum distance a vertex can move before creating invalid geometry
- Each clearance module targets specific issue types (protrusions, holes, passages)
- Uses minimal geometric modifications to achieve target clearance

**Geometry Fixing:**
- `repair_geometry()` tries multiple strategies in order: clean coords → buffer(0) → simplify → reconstruct
- Buffer(0) trick: often fixes self-intersections and topology errors
- Strict mode: only applies conservative fixes that preserve geometric intent
- Raises `RepairError` if geometry cannot be repaired

**Polygon Merging:**
- `merge_close_polygons()` merges polygons within specified margin distance
- Five strategies available: simple_buffer, selective_buffer, vertex_movement, boundary_extension, convex_bridges
- Uses spatial indexing for efficient group detection
- Supports vertex insertion for improved merge quality

## Testing Conventions

### Test Organization
- One test file per module: `test_split.py`, `test_overlap.py`, `test_fix.py`, etc.
- Tests organized into classes by functionality
- Pattern: `TestFunctionName` for main tests, `TestEdgeCases` for edge cases

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
```

## Code Patterns to Follow

### When Adding New Functions with Strategy Parameters
1. **Always use enum types** for strategy parameters
2. Define enum in `core/types.py` if it doesn't exist
3. Function signature pattern:
```python
from .core.types import MyStrategy

def my_function(geometry, strategy: MyStrategy = MyStrategy.DEFAULT):
    if strategy == MyStrategy.OPTION1:
        # handle option 1
    elif strategy == MyStrategy.OPTION2:
        # handle option 2
```
4. **Never accept `Union[str, EnumType]`** - only the enum type
5. Add to module's `__all__` list

### When Adding New Simplification Functions
1. Create private `_function_name(vertices: np.ndarray, ...) -> np.ndarray`
2. Create public wrapper using `process_geometry()`
3. Handle 2D coordinate arrays only (process_geometry handles 3D)
4. Export via `__all__` in module and `__init__.py`

### When Adding New Geometry Fixes
1. Add to appropriate clearance submodule or create new one
2. Follow minimal modification principle
3. Validate result before returning
4. Raise appropriate exception type (RepairError, ClearanceError, etc.)
5. Add diagnostic information for failures

### When Adding New Exception Types
1. Always inherit from `PolyforgeError` or one of its subclasses
2. Add exception to `core/errors.py`
3. Include relevant metadata as attributes (geometry, strategies, etc.)
4. Export via `core/__init__.py` and main `__init__.py`

### When Working with Large Datasets
- Always use spatial indexing (STRtree) for polygon-to-polygon operations
- Avoid O(n²) nested loops over polygons
- See `overlap.py` and `merge.py` for reference implementations

## Performance Considerations

- Spatial indexing in `overlap.py` and `merge.py` handles 1000+ polygons efficiently
- For batch operations, use provided batch functions: `batch_repair_geometries()`, `remove_overlaps()`, `merge_close_polygons()`
- Z-coordinate interpolation adds overhead; use 2D when possible
- Simplification algorithms have different performance profiles:
  - RDP: Fast, good for most cases
  - VW: Slower, better visual results
  - VWP: Slowest, topology-preserving

## API Design Principles

### No Backward Compatibility
- Library has **not been released** yet
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
- Enums do NOT inherit from `str` (pure Enum types)
- Functions only accept enum types, not strings

### Error Handling
- Comprehensive exception hierarchy in `core/errors.py`
- All exceptions inherit from `PolyforgeError`
- Exceptions carry metadata for debugging
- Consistent error handling across all modules

## Common Gotchas

1. **Use Enums, Not Strings:** All strategy parameters require enum types from `core/types.py`, not strings
2. **Closed Rings:** Polygon exterior/interior rings must have first == last vertex
3. **Empty Geometries:** Always check `geom.is_empty` before accessing properties
4. **MultiPolygon Results:** Many operations can return MultiPolygon even when input was Polygon
5. **Spatial Index Queries:** `tree.query()` returns candidates, must validate actual intersection
6. **Buffer Distance:** `buffer(0)` is not the same as `buffer(0.0)` in some Shapely versions
7. **Function Names:** Use `repair_geometry` not `fix_geometry`, `collapse_short_edges` not `snap_short_edges`
8. **Exception Names:** Use `RepairError` not `GeometryRepairError` or `GeometryFixError`

## Quick Reference: Common Imports

```python
# Main functions
from polyforge import (
    # Simplification
    simplify_rdp, simplify_vw, simplify_vwp,
    collapse_short_edges, deduplicate_vertices,

    # Overlap handling
    split_overlap, remove_overlaps,

    # Merging
    merge_close_polygons,

    # Repair
    repair_geometry, analyze_geometry,

    # Clearance fixing
    fix_clearance, diagnose_clearance,
)

# Strategy enums
from polyforge.core.types import (
    OverlapStrategy,
    MergeStrategy,
    RepairStrategy,
    SimplifyAlgorithm,
    CollapseMode,
)

# Exceptions
from polyforge.core.errors import (
    PolyforgeError,
    RepairError,
    ValidationError,
    OverlapResolutionError,
    MergeError,
    ClearanceError,
)
```
