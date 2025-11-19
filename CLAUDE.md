# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Polyforge is a polygon processing and manipulation library built on top of Shapely. It provides utilities for:
- Geometry simplification (RDP, VW, VWP algorithms)
- Overlap detection and resolution at scale (using spatial indexing)
- Polygon merging with 5 strategies
- Clearance fixing for low-clearance geometries
- Geometry validation and repair with 5 strategies
- **Robust fixing with constraint validation** - ensures quality requirements are met
- Vertex processing (collapse short edges, remove duplicates)
- **Hole cleanup** - remove small, narrow, or degenerate holes
- Topology operations (boundary alignment)
- Batch processing with error handling and automatic cleanup

**Key Dependencies:** shapely, numpy, scipy, simplification library

**Important Design Principles:**
- **No backward compatibility code** - Library hasn't been released yet (v0.1.0)
- **Enum-based strategy parameters** - Enums accepted at public API, converted to strings internally
- **Comprehensive error hierarchy** - All exceptions inherit from PolyforgeError
- **Explicit exports** - All modules define `__all__`
- **Functional core, object shell** - Public API uses dataclasses/enums, internal ops use functions
- **Separation of concerns** - Pure operations in `ops/`, orchestration in top-level modules

## Development Commands

### Testing
```bash
# Run all tests (427 tests)
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_overlap.py -v

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
PYTHONPATH=/Users/dwastberg/repos/polyforge:$PYTHONPATH python examples/generators/simplification_examples.py
PYTHONPATH=/Users/dwastberg/repos/polyforge:$PYTHONPATH python examples/generators/overlap_examples.py
```

### Python Environment
- Requires Python >= 3.10
- Uses uv for dependency management (see uv.lock)

## Architecture

### Module Organization

The codebase follows a **layered architecture** with clear separation between public API and internal operations:

```
polyforge/
├── __init__.py                 # Public API exports (~30 functions, 10 enums, 8 exceptions)
├── core/                       # Core types, errors, constraints, and shared utilities
│   ├── __init__.py
│   ├── types.py                # Enum definitions (10 strategy enums + coerce_enum helper)
│   ├── errors.py               # Exception hierarchy (PolyforgeError + 7 specific exceptions)
│   ├── constraints.py          # Constraint validation (GeometryConstraints, MergeConstraints, ConstraintStatus)
│   ├── cleanup.py              # Backward-compatible imports from ops/cleanup_ops.py
│   ├── geometry_utils.py       # Shared geometry operations (to_single_polygon, etc.)
│   ├── spatial_utils.py        # Spatial indexing utilities (STRtree operations)
│   └── iterative_utils.py      # Iterative improvement patterns
├── ops/                        # Low-level pure operations (no orchestration)
│   ├── __init__.py
│   ├── simplify_ops.py         # Coordinate-level simplification (RDP, VW, VWP, snap, dedupe)
│   ├── cleanup_ops.py          # Hole removal, cleanup logic (CleanupConfig, remove_small_holes, etc.)
│   ├── merge_ops.py            # Merge orchestration helpers
│   ├── merge_simple_buffer.py  # Simple buffer merge strategy
│   ├── merge_selective_buffer.py # Selective buffer merge strategy
│   ├── merge_vertex_movement.py # Vertex movement merge strategy
│   ├── merge_boundary_extension.py # Boundary extension merge strategy
│   ├── merge_convex_bridges.py # Convex bridges merge strategy
│   ├── merge_edge_detection.py # Edge detection utilities for merge
│   ├── merge/                  # Package for merge utilities
│   │   └── __init__.py
│   └── clearance/              # Clearance fixing operations
│       ├── __init__.py
│       ├── utils.py            # Clearance utilities (distance calculations, vertex finding)
│       ├── holes.py            # fix_hole_too_close
│       ├── protrusions.py      # fix_narrow_protrusion, fix_sharp_intrusion
│       ├── remove_protrusions.py # remove_narrow_protrusions
│       └── passages.py         # fix_narrow_passage, fix_near_self_intersection, fix_parallel_close_edges
├── process.py                  # Core: process_geometry() - applies functions to geometry vertices
├── simplify.py                 # High-level simplification API (wraps ops/simplify_ops.py)
├── overlap.py                  # Overlap resolution (consolidates old overlap/ package)
├── topology.py                 # Boundary alignment operations
├── tile.py                     # Polygon tiling functions
├── pipeline.py                 # Minimal pipeline runner (FixConfig, PipelineContext, run_steps)
├── metrics.py                  # Measurement utilities (measure_geometry, total_overlap_area)
├── merge/                      # Polygon merging public API
│   ├── __init__.py
│   └── core.py                 # merge_close_polygons() orchestration
├── repair/                     # Geometry repair public API
│   ├── __init__.py
│   ├── core.py                 # repair_geometry(), batch_repair_geometries()
│   ├── analysis.py             # analyze_geometry()
│   ├── robust.py               # robust_fix_geometry(), robust_fix_batch()
│   ├── utils.py                # Repair utilities
│   └── strategies/             # 5 repair strategy implementations
│       ├── __init__.py
│       ├── auto.py
│       ├── buffer.py
│       ├── simplify.py
│       ├── reconstruct.py
│       └── strict.py
└── clearance/                  # Clearance fixing public API
    ├── __init__.py             # Re-exports from ops/clearance/
    └── fix_clearance.py        # Auto-detection: fix_clearance(), diagnose_clearance()
```

### Key Architectural Patterns

**1. Functional Core, Object Shell**
- **Public API layer**: Uses dataclasses (GeometryConstraints, ConstraintStatus) and enums (MergeStrategy, RepairStrategy) for type safety and clarity
- **Operations layer (`ops/`)**: Pure functions working on Shapely geometries and numpy arrays
- **Conversion**: `coerce_enum()` helper converts enums to strings at API boundary
- Benefits:
  - Type-safe public API with IDE autocomplete
  - Testable, composable pure functions internally
  - Clear separation of concerns

**2. Layered Architecture**
Three distinct layers:
1. **Public API** (`polyforge/__init__.py`, top-level modules): User-facing functions with stable signatures
2. **Orchestration** (`merge/core.py`, `repair/core.py`, `clearance/fix_clearance.py`): Workflow coordination, strategy selection
3. **Operations** (`ops/*`): Pure geometric transformations, no business logic

**3. Pipeline System (New)**
- `pipeline.py` provides minimal orchestration framework
- `PipelineContext`: Shares state across pipeline steps (original geometry, constraints, config)
- `FixConfig`: Lightweight configuration extracted from GeometryConstraints
- `run_steps()`: Executes steps iteratively until constraints satisfied or progress stalls
- Used by robust fixing to avoid heavy transaction/stage system

**4. Metrics-First Validation**
- `metrics.py` provides measurement functions:
  - `measure_geometry()`: Returns dict with is_valid, clearance, area, area_ratio
  - `total_overlap_area()`: Computes overlap within a collection of geometries
- Constraint validation uses these metrics to check quality requirements
- Avoids coupling constraint checking to specific data structures

**5. Enum-Based Strategy Parameters (Public API)**
- All strategy parameters accept enum types from `core/types.py`
- Enums provide type safety, IDE autocomplete, and documentation
- **Internal conversion**: `coerce_enum()` accepts both Enum and string values
  - This allows testing with string literals if needed
  - Public API enforces enums for safety
- **Consistent naming**: Domain-specific parameter names (`merge_strategy`, `overlap_strategy`, `repair_strategy`)

Example:
```python
from polyforge import merge_close_polygons
from polyforge.core import MergeStrategy

# Correct - use enums with domain-specific parameter name
result = merge_close_polygons(polys, margin=2.0, merge_strategy=MergeStrategy.SELECTIVE_BUFFER)

# Also works (coerce_enum accepts strings internally)
from polyforge.core import coerce_enum
strategy = coerce_enum('selective_buffer', MergeStrategy)  # Returns MergeStrategy.SELECTIVE_BUFFER
```

**6. process_geometry() Pattern**
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

**7. Spatial Indexing for Performance**
- `overlap.py` and `merge/core.py` use Shapely's STRtree for O(n log n) performance
- Shared implementation in `core/spatial_utils.py`
- Critical for handling 1000+ polygons efficiently
- Reduces candidate comparisons by 90-99% in typical cases
- Pattern: build index → query candidates → validate actual overlaps

**8. Iterative Resolution**
- `remove_overlaps()` uses iterative algorithm:
  1. Build spatial index
  2. Find overlapping pairs
  3. Select independent pairs (no polygon appears twice)
  4. Resolve pairs in parallel
  5. Repeat until no overlaps or max_iterations
- Most cases converge in 1-5 iterations
- Default max_iterations: 100 for batch operations, 10 for single-geometry operations

**9. Exception Hierarchy**
- All custom exceptions inherit from `PolyforgeError`
- Specific exceptions for different error types:
  - `RepairError` - geometry repair failures
  - `ValidationError` - validation failures
  - `OverlapResolutionError` - overlap resolution failures
  - `MergeError` - merge operation failures
  - `ClearanceError` - clearance fixing failures
  - `ConfigurationError` - invalid parameters
  - `FixWarning` - warnings about partial fixes
- Exceptions carry metadata (geometry, strategies_tried, suggested_strategy, etc.)

**10. Constraint Validation System**
- `GeometryConstraints` defines quality requirements that must be met
- Constraints are validated using `constraints.check()` which returns `ConstraintStatus`
- Supported constraints:
  - `min_clearance`: Minimum clearance (Shapely's minimum_clearance property)
  - `min_area_ratio` / `max_area_ratio`: Area preservation bounds
  - `must_be_valid`: Topological validity requirement
  - `allow_multipolygon`: Whether MultiPolygon results are acceptable
  - `max_holes`: Maximum number of interior holes
  - `min_hole_area`: Minimum hole area (smaller holes flagged as violations)
  - `max_hole_aspect_ratio`: Maximum hole aspect ratio (using OBB)
  - `min_hole_width`: Minimum hole width (using OBB shorter dimension)
- Constraint validation happens automatically during iterative fixing

**11. Robust Fixing Pipeline (robust_fix_batch)**
The batch fixing pipeline runs in 3 phases:
- **Phase 1: Individual Fixes**
  - Applies `robust_fix_geometry()` to each geometry
  - Iteratively attempts fixes until constraints satisfied
  - Uses `pipeline.run_steps()` for orchestration
- **Phase 2: Overlap Resolution**
  - Uses `remove_overlaps()` to resolve remaining overlaps
  - Validates that overlap fixes don't violate other constraints
- **Phase 3: Final Cleanup**
  - Removes zero-area holes
  - Removes small holes (min_hole_area)
  - Removes narrow holes (max_hole_aspect_ratio, min_hole_width)
  - Removes degenerate exterior features using erosion-dilation
  - Critical for cleaning up artifacts from overlap resolution

### Important Implementation Details

**Coordinate Handling:**
- All internal processing functions work on numpy arrays (Nx2 for 2D, Nx3 for 3D)
- `process_geometry()` handles the Shapely ↔ numpy conversion
- 3D geometries: Z values preserved via interpolation based on 2D path distance
- Closed rings: First and last vertices must be identical

**Overlap Resolution:**
- `split_overlap()` is now an alias for `resolve_overlap_pair()` in `overlap.py`
- `remove_overlaps()` handles many-to-many overlaps using spatial indexing
- Returns originals unchanged for: no overlap, containment, or touching-only

**Clearance Fixing:**
- "Clearance" = minimum distance a vertex can move before creating invalid geometry (Shapely's `minimum_clearance`)
- Public API in `clearance/` delegates to implementations in `ops/clearance/`
- Each clearance module targets specific issue types (protrusions, holes, passages)
- Uses minimal geometric modifications to achieve target clearance
- Default max_iterations: 10

**Geometry Repair:**
- `repair_geometry()` tries multiple strategies in order based on RepairStrategy enum
- Strategies implemented in `repair/strategies/`: auto, buffer, simplify, reconstruct, strict
- Buffer(0) trick: often fixes self-intersections and topology errors
- Strict mode: only applies conservative fixes that preserve geometric intent
- Raises `RepairError` if geometry cannot be repaired
- `batch_repair_geometries()` processes many geometries with error collection

**Robust Fixing (Advanced):**
- `robust_fix_geometry()` iteratively fixes geometries until all constraints satisfied
- Uses new `pipeline.run_steps()` for orchestration (replaces old transaction/stage system)
- `robust_fix_batch()` processes multiple geometries with 3-phase pipeline
- Constraint-driven: uses `GeometryConstraints` to define quality requirements
- Automatic hole cleanup: removes small, narrow, and degenerate holes
- Degenerate feature removal: eliminates zero-width slivers using erosion-dilation
- Essential for production use where quality guarantees are required

**Polygon Merging:**
- `merge_close_polygons()` merges polygons within specified margin distance
- Five strategies available: SIMPLE_BUFFER, SELECTIVE_BUFFER, VERTEX_MOVEMENT, BOUNDARY_EXTENSION, CONVEX_BRIDGES
- Strategy implementations in `ops/merge_*.py` files
- Uses spatial indexing (via `core/spatial_utils.py`) for efficient group detection
- Parameter name: `merge_strategy` (not `strategy`)

**Tolerance Parameters:**
- **Disambiguated for clarity:**
  - `min_area_threshold`: Minimum overlap area to consider (used in `count_overlaps`, `find_overlapping_groups`)
  - `distance_tolerance`: Distance tolerance for snapping (used in `align_boundaries`)
  - `tolerance`: Coordinate comparison tolerance (used in `repair_geometry`, `deduplicate_vertices`)

## Testing Conventions

### Test Organization
- One test file per module: `test_overlap.py`, `test_repair.py`, `test_robust_fix.py`, etc.
- Tests organized into classes by functionality
- Pattern: `TestFunctionName` for main tests, `TestEdgeCases` for edge cases
- **Total: 427 tests** (all passing as of latest refactoring)

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

# Run all merge tests
pytest tests/test_merge.py -v

# Run merge strategy tests
pytest tests/test_merge.py::TestSelectiveBufferStrategy -v
```

## Code Patterns to Follow

### When Adding New Functions with Strategy Parameters
1. **Use enum types** for strategy parameters in public API
2. Define enum in `core/types.py` if it doesn't exist
3. **Use domain-specific parameter names**: `merge_strategy`, `overlap_strategy`, `repair_strategy` (not generic `strategy`)
4. Accept both enum and string values using `coerce_enum()`
5. Function signature pattern:
```python
from .core.types import MyStrategy, coerce_enum

def my_function(geometry, my_strategy: Union[MyStrategy, str] = MyStrategy.DEFAULT):
    my_strategy = coerce_enum(my_strategy, MyStrategy)

    if my_strategy == MyStrategy.OPTION1:
        # handle option 1
    elif my_strategy == MyStrategy.OPTION2:
        # handle option 2
```
6. Add to module's `__all__` list

### When Adding New Simplification Functions
1. Create private `_function_name(vertices: np.ndarray, ...) -> np.ndarray` in `ops/simplify_ops.py`
2. Create public wrapper in `simplify.py` using `process_geometry()`
3. Handle 2D coordinate arrays only (process_geometry handles 3D)
4. Export via `__all__` in module and `__init__.py`

### When Adding New Geometry Operations
1. Add pure operation function to appropriate `ops/` module
2. Create public API wrapper in top-level module (if needed)
3. Follow pattern: geometry in → geometry out, no side effects
4. Use existing utilities from `core/` where applicable
5. Validate result before returning

### When Adding New Exception Types
1. Always inherit from `PolyforgeError` or one of its subclasses
2. Add exception to `core/errors.py`
3. Include relevant metadata as attributes (geometry, strategies, suggested_strategy, etc.)
4. Export via `core/__init__.py` and main `__init__.py`

### When Working with Large Datasets
- Always use spatial indexing (STRtree) for polygon-to-polygon operations
- Use shared utilities from `core/spatial_utils.py`
- Avoid O(n²) nested loops over polygons
- See `overlap.py` and `merge/core.py` for reference implementations

### When Adding Shared Utilities
- Place in appropriate `core/` module:
  - Geometry operations → `core/geometry_utils.py`
  - Spatial operations → `core/spatial_utils.py`
  - Iterative patterns → `core/iterative_utils.py`
- If it's a pure low-level operation, consider `ops/` instead
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
- All strategy parameters accept enum types (and strings via coerce_enum)
- Enums are defined in `core/types.py`:
  - `OverlapStrategy` - SPLIT, LARGEST, SMALLEST
  - `MergeStrategy` - SIMPLE_BUFFER, SELECTIVE_BUFFER, VERTEX_MOVEMENT, BOUNDARY_EXTENSION, CONVEX_BRIDGES
  - `RepairStrategy` - AUTO, BUFFER, SIMPLIFY, RECONSTRUCT, STRICT
  - `SimplifyAlgorithm` - RDP, VW, VWP
  - `CollapseMode` - MIDPOINT, FIRST, LAST
  - Plus 5 clearance-specific enums: `HoleStrategy`, `PassageStrategy`, `IntrusionStrategy`, `IntersectionStrategy`, `EdgeStrategy`
- Enums do NOT inherit from `str` (pure Enum types)
- `coerce_enum()` helper allows both enum and string values internally

### Error Handling
- Comprehensive exception hierarchy in `core/errors.py`
- All exceptions inherit from `PolyforgeError`
- Exceptions carry metadata for debugging
- Consistent error handling across all modules
- Some functions return originals on failure (e.g., `resolve_overlap_pair`), others raise exceptions (e.g., `repair_geometry`)

## Common Gotchas

1. **Enum/String Coercion:** Public API uses enums, but `coerce_enum()` accepts strings too. Tests can use either.
2. **Parameter Names Matter:** Use `merge_strategy` not `strategy`, `min_area_threshold` not `tolerance` (context-dependent)
3. **Closed Rings:** Polygon exterior/interior rings must have first == last vertex
4. **Empty Geometries:** Always check `geom.is_empty` before accessing properties
5. **MultiPolygon Results:** Many operations can return MultiPolygon even when input was Polygon
6. **Spatial Index Queries:** `tree.query()` returns candidates, must validate actual intersection
7. **Buffer Distance:** `buffer(0)` is not the same as `buffer(0.0)` in some Shapely versions
8. **Function Names:** Use `repair_geometry` not `fix_geometry`, `collapse_short_edges` not `snap_short_edges`
9. **Exception Names:** Use `RepairError` not `GeometryRepairError` or `GeometryFixError`
10. **split_overlap is now an alias:** `split_overlap()` directly calls `resolve_overlap_pair()` in `overlap.py`
11. **ops/ is internal:** Don't import from `polyforge.ops.*` - use public API from top-level modules

## Documentation

### Comprehensive Documentation Available
- **README.md**: Project overview, quick start, architecture guide, performance notes
- **API.md**: Complete API reference for all functions, enums, exceptions (if exists)
- **SIMPLIFY_CODE_DESIGN.md**: Detailed analysis of refactoring decisions
- **SIMPLIFY_DESIGN.md**: Functional architecture proposal and rationale
- **examples/**: Example scripts using framework
  - `examples/framework/`: Test data generators and plotting utilities
  - `examples/generators/`: Specific example generators (simplification, overlap)

## Quick Reference: Common Imports

```python
# Main functions
from polyforge import (
    # Simplification (7 functions)
    simplify_rdp, simplify_vw, simplify_vwp,
    collapse_short_edges, deduplicate_vertices,
    remove_small_holes, remove_narrow_holes,

    # Overlap handling (5 functions)
    split_overlap, resolve_overlap_pair, remove_overlaps,
    count_overlaps, find_overlapping_groups,

    # Merging (1 function, 5 strategies)
    merge_close_polygons,

    # Repair (3 functions)
    repair_geometry, analyze_geometry, batch_repair_geometries,

    # Robust fixing (2 functions) - Advanced constraint-driven fixing
    robust_fix_geometry, robust_fix_batch,

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

# Constraint system
from polyforge.core import (
    GeometryConstraints,  # Define quality requirements
    MergeConstraints,     # Configure merging behavior
    ConstraintStatus,     # Validation results
    ConstraintViolation,  # Individual violations
    ConstraintType,       # Enum of constraint types
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
    coerce_enum,  # Helper to convert string → enum
)

# Exceptions (8 exceptions)
from polyforge.core import (
    PolyforgeError,
    ValidationError,
    RepairError,
    OverlapResolutionError,
    MergeError,
    ClearanceError,
    ConfigurationError,
    FixWarning,
)
```

## Recent Major Changes (for context)

### Architectural Refactoring (Latest - Major Simplification)
**Major restructuring of the codebase to separate concerns:**

**Changes Made:**

1. **Created `ops/` Layer**:
   - Moved all low-level geometry operations to `polyforge/ops/`
   - Pure functions with no orchestration logic
   - Modules: `simplify_ops.py`, `cleanup_ops.py`, `merge_*.py`, `clearance/*`
   - Clear separation from public API and business logic

2. **Added Pipeline System**:
   - New `pipeline.py` module for minimal orchestration
   - `FixConfig`: Lightweight configuration from constraints
   - `PipelineContext`: Shares state across steps
   - `run_steps()`: Iterative execution until constraints satisfied
   - Replaces heavy transaction/stage system

3. **Added Metrics Module**:
   - New `metrics.py` for measurement functions
   - `measure_geometry()`: Returns validation metrics as dict
   - `total_overlap_area()`: Computes overlap in collections
   - Decouples measurement from constraint validation

4. **Consolidated Overlap Module**:
   - `overlap/` package → `overlap.py` single file
   - `split.py` removed - `split_overlap()` now alias in `overlap.py`
   - All overlap logic in one place: resolution, batch removal, counting, grouping

5. **Clearance Delegation**:
   - `clearance/` now delegates to `ops/clearance/`
   - Public API stable, implementations moved to ops layer
   - Clear separation of interface and implementation

6. **Enum Coercion**:
   - Added `coerce_enum()` helper in `core/types.py`
   - Public API enforces enums for type safety
   - Internal code can accept both enum and string values
   - Simplifies testing while maintaining API guarantees

**Impact:**
- Clearer separation of concerns (API / orchestration / operations)
- More testable pure functions in `ops/` layer
- Lighter orchestration without transaction/stage overhead
- **All 427 tests pass**

**Philosophy:**
- **Functional core, object shell**: Pure operations internally, typed API externally
- **Measure, don't model**: Use simple metrics over complex constraint objects where possible
- **Explicit over implicit**: Clear module boundaries, obvious data flow

### Constraint Validation & Phase 3 Cleanup (Previous Major Update)
**Enhanced constraint validation and final cleanup in batch fixing:**

1. **Enhanced Constraint Validation** (`core/constraints.py`):
   - `GeometryConstraints.check()` validates hole properties using OBB
   - Checks: `min_hole_area`, `max_hole_aspect_ratio`, `min_hole_width`
   - Holes violating constraints flagged as violations

2. **Added Phase 3 Final Cleanup** (`repair/robust.py`):
   - `robust_fix_batch()` includes Phase 3 after overlap resolution
   - Removes: zero-area holes, small holes, narrow holes, degenerate exterior features
   - Degenerate feature removal uses erosion-dilation technique

3. **New Function** (`simplify.py`):
   - `remove_narrow_holes()` filters by aspect ratio and/or width
   - Uses oriented bounding box (OBB) for accurate detection

## Architecture Decision Records

### Why ops/ Layer?
**Decision:** Separate pure operations from orchestration and public API

**Rationale:**
- **Testability**: Pure functions easier to test in isolation
- **Reusability**: Operations can be composed in different ways
- **Clarity**: Clear boundary between "what" (API) and "how" (ops)
- **Maintainability**: Changes to operations don't affect public API

**Tradeoffs:**
- ✅ Better separation of concerns
- ✅ More testable code
- ❌ Extra directory level
- ❌ Potential import confusion (use public API, not ops directly)

### Why Pipeline System Instead of Transactions/Stages?
**Decision:** Replace transaction/stage system with lightweight pipeline

**Rationale:**
- **Simpler**: Transactions add complexity for no benefit (geometry ops are cheap)
- **Clearer**: Explicit step execution vs. hidden transaction tracking
- **Flexible**: Easy to compose custom pipelines
- **Lighter**: No snapshot history, rollback infrastructure

**Tradeoffs:**
- ✅ 80% less code for same functionality
- ✅ Easier to understand and debug
- ❌ Less "enterprise-y" (but we don't need that)

### Why Enum Coercion Instead of Pure Strings?
**Decision:** Accept enums at API, convert internally with coerce_enum()

**Rationale:**
- **Type Safety**: Enums catch errors at write-time (IDE) and runtime (mypy)
- **Discoverability**: IDE autocomplete shows all valid options
- **Testing Flexibility**: Tests can use string literals if convenient
- **Best of Both**: Safety at API boundary, flexibility internally

**Tradeoffs:**
- ✅ Type-safe public API
- ✅ Flexible testing
- ❌ Slight conversion overhead (negligible)

---

**Document Version**: 2.0
**Date**: 2025-11-11
**Last Updated**: 2025-11-11 (Complete architectural refactoring update)
**Author**: Updated to reflect ops/ layer, pipeline system, and current codebase structure
