# Polyforge API

High-level reference for the public surface of the `polyforge` package. The library builds on Shapely and NumPy; every entrypoint consumes and returns Shapely geometries unless noted.

## Conventions & Data Model
- Functions never mutate inputs. Expect new Shapely objects (Polygon, MultiPolygon, LineString, etc.) on return.
- Most strategy/config parameters accept either enums from `polyforge.core.types` or matching lowercase strings (`"selective_buffer"`, `"buffer"`, etc.). Use `coerce_enum(value, EnumClass)` if you need to normalize user input yourself.
- Z coordinates are preserved automatically when you go through `process_geometry` or any high-level wrapper that calls it.
- Spatial indexing (STRtree) is used internally for `overlap`, `merge`, and some clearance routines; you do not have to manage the index.

## Package Map

| Module | Purpose |
| --- | --- |
| `polyforge.simplify` | Vertex-level simplification, deduplication, and hole cleanup helpers. |
| `polyforge.clearance` | Diagnosis + fixers for minimum-clearance problems (holes too close, passages, spikes, parallel edges). |
| `polyforge.overlap` | Resolve overlaps between polygons (pairwise or batch) with configurable strategies. |
| `polyforge.merge` | Merge polygons that touch or fall within a distance margin using several geometry-aware strategies. |
| `polyforge.repair` | “Classic” validity repair plus geometry analysis utilities. |
| `polyforge.repair.robust` | Constraint-driven repair pipeline that iterates validity → clearance → merge → cleanup. |
| `polyforge.core` | Shared enums, constraint dataclasses, errors, and lower-level geometry/spatial utilities. |
| `polyforge.pipeline` | Lightweight step runner (`run_steps`) used by the robust repair pipeline; reusable for custom workflows. |
| `polyforge.metrics` | Measurements (validity flags, clearance, overlap area) consumed by constraints/pipelines. |
| `polyforge.process`, `polyforge.topology`, `polyforge.tile` | Standalone helpers for applying NumPy ops, aligning shared boundaries, and tiling geometries. |
| `polyforge.ops.*` | Low-level NumPy/Shapely primitives backing the public API; import directly when you need granular control. |

---

## Simplify & Clean (`polyforge.simplify`)
- `simplify_rdp(geometry, epsilon)` – Ramer–Douglas–Peucker simplifier for any Shapely geometry. Removes vertices within `epsilon` of their neighbors.
- `simplify_vw(geometry, threshold)` – Visvalingam–Whyatt simplifier; progressively drops points that contribute less than `threshold` area for smoother visual results.
- `simplify_vwp(geometry, threshold)` – Topology-preserving VW variant; slower but guarantees a valid result.
- `collapse_short_edges(geometry, min_length, snap_mode=CollapseMode.MIDPOINT)` – Collapses edges shorter than `min_length` by snapping vertices using midpoint/first/last rules.
- `deduplicate_vertices(geometry, tolerance=1e-10)` – Removes consecutive duplicate vertices without shrinking near-duplicates.
- `remove_small_holes(geometry, min_area)` – Strips holes whose area falls below `min_area`. Accepts `Polygon` or `MultiPolygon`.
- `remove_narrow_holes(geometry, max_aspect_ratio=50, min_width=None)` – Deletes holes that exceed an aspect-ratio threshold or are thinner than `min_width`.

All of the above are thin wrappers over `process_geometry`, so they support Multi* geometries and preserve Z coordinates.

## Clearance Toolkit (`polyforge.clearance`)
### Automatic diagnosis & fixing
- `fix_clearance(polygon, min_clearance, max_iterations=10, return_diagnosis=False)` – Detects the dominant clearance issue (holes too close, protrusions, passages, near self-intersections, parallel edges) and iteratively applies targeted fixes until `minimum_clearance >= min_clearance` or no progress. Returns either a fixed `Polygon` or `(polygon, ClearanceFixSummary)` when `return_diagnosis=True`.
  - `ClearanceFixSummary` captures `initial_clearance`, `final_clearance`, `iterations`, `issue` (last detected `ClearanceIssue`), `fixed` flag, and `history` of issue types encountered.
- `diagnose_clearance(polygon, min_clearance)` – Runs the same heuristics without modifying the geometry. Returns a `ClearanceDiagnosis` dataclass with fields: `issue`, `meets_requirement`, `current_clearance`, `clearance_ratio`, `clearance_line` (Shapely `LineString` or `None`), `recommended_fix`, and helper property `has_issues`.
- `ClearanceIssue` enum values: `NONE`, `HOLE_TOO_CLOSE`, `NARROW_PROTRUSION`, `NARROW_PASSAGE`, `NEAR_SELF_INTERSECTION`, `PARALLEL_CLOSE_EDGES`, `UNKNOWN`.

### Targeted fixers (all accept enums or strings for strategy parameters)
- `fix_hole_too_close(polygon, min_clearance, strategy=HoleStrategy.REMOVE)` – Remove, shrink, or translate interior rings that come too close to the exterior boundary.
- `fix_narrow_protrusion(polygon, min_clearance, max_iterations=10)` – Uses `minimum_clearance_line` to identify spikes and moves the offending vertices apart.
- `remove_narrow_protrusions(polygon, aspect_ratio_threshold=5, min_iterations=1, max_iterations=10)` – Pure geometric cleanup that chops off high-aspect-ratio spikes by deleting vertices.
- `fix_sharp_intrusion(polygon, min_clearance, strategy=IntrusionStrategy.FILL, max_iterations=10)` – Fills or smooths deep indentations until clearance improves.
- `fix_narrow_passage(polygon, min_clearance, strategy=PassageStrategy.WIDEN)` – Either widens the neck by relocating vertices or splits the polygon in two (`PassageStrategy.SPLIT`).
- `fix_near_self_intersection(polygon, min_clearance, strategy=IntersectionStrategy.SIMPLIFY)` – Simplifies or buffers edges that almost cross to restore clearance.
- `fix_parallel_close_edges(polygon, min_clearance, strategy=IntersectionStrategy.SIMPLIFY)` – Alias that routes through the near-self-intersection logic for parallel edges.

Underlying utilities such as `_find_nearest_vertex_index`, `_point_to_segment_distance`, and curvature helpers are exposed from `polyforge.ops.clearance` for advanced workflows and testing.

## Overlap Management (`polyforge.overlap`)
- `split_overlap(poly1, poly2, overlap_strategy=OverlapStrategy.SPLIT)` – Convenience alias for `resolve_overlap_pair`.
- `resolve_overlap_pair(poly1, poly2, overlap_strategy=OverlapStrategy.SPLIT)` – Resolves the shared area between two polygons. Strategies: `SPLIT` (cut overlap roughly in half along a centroid-informed line), `LARGEST` (assign entire overlap to larger polygon), `SMALLEST` (assign to smaller).
- `remove_overlaps(polygons, overlap_strategy=OverlapStrategy.SPLIT, max_iterations=100)` – Iteratively resolves overlaps across a list using STRtree acceleration and pairwise resolution.
- `count_overlaps(polygons, min_area_threshold=1e-10)` – Returns the number of polygon pairs with overlap area above the threshold.
- `find_overlapping_groups(polygons, min_area_threshold=1e-10)` – Returns connected components (lists of indices) representing overlap groups, useful before merging or repair passes.

## Merge Operations (`polyforge.merge`)
- `merge_close_polygons(polygons, margin=0.0, merge_strategy=MergeStrategy.SELECTIVE_BUFFER, preserve_holes=True, return_mapping=False, insert_vertices=False)`  
  Efficiently merges polygons that either touch or sit within `margin` distance. Isolated polygons bypass heavy processing. Strategy choices:
  - `SIMPLE_BUFFER` – Expand by `margin/2`, union, contract; fastest but softens corners.
  - `SELECTIVE_BUFFER` – Buffer only near gaps, then union; good balance of fidelity vs. coverage.
  - `VERTEX_MOVEMENT` – Slides vertices toward nearby polygons to create contact before unioning.
  - `BOUNDARY_EXTENSION` – Detects parallel edges and spans bridges between them; ideal for rectilinear footprints.
  - `CONVEX_BRIDGES` – Samples nearby boundaries, builds convex hull bridges, and unions for smooth organic connections.
  Options: `preserve_holes` keeps interior rings, `insert_vertices=True` adds explicit connection points before merging, and `return_mapping=True` yields `(result, groups)` where `groups[i]` lists source indices that produced `result[i]`.
- `find_close_polygon_groups(polygons, margin)` – Returns `(isolated_indices, merge_groups)` using STRtree proximity queries, so you can inspect candidate clusters before calling `merge_close_polygons`.

Advanced strategy helpers live under `polyforge.ops.merge`: `merge_simple_buffer`, `merge_selective_buffer`, `merge_vertex_movement`, `merge_boundary_extension`, `merge_convex_bridges`, `insert_connection_vertices`, `find_parallel_close_edges`, etc.

## Geometry Repair & QA
### Classic repair (`polyforge.repair`)
- `repair_geometry(geometry, repair_strategy=RepairStrategy.AUTO, buffer_distance=0.0, tolerance=1e-10, verbose=False)` – Routes invalid geometries through one of five strategies (`AUTO`, `BUFFER`, `SIMPLIFY`, `RECONSTRUCT`, `STRICT`). Returns the first valid fix or raises `RepairError`.
- `batch_repair_geometries(geometries, repair_strategy=RepairStrategy.AUTO, on_error="skip", verbose=False)` – Applies `repair_geometry` across a list. `on_error` controls failure handling (`"skip"`, `"keep"`, `"raise"`). Returns `(repaired_list, failed_indices)`.
- `analyze_geometry(geometry)` – Produces a diagnostic dict with metrics (`is_valid`, `validity_message`, `geometry_type`, `is_empty`, `area`) plus `issues`/`suggestions` derived from messaging/coordinate checks.

### Constraint-driven pipeline (`polyforge.repair.robust`)
- `robust_fix_geometry(geometry, constraints: GeometryConstraints, max_iterations=20, raise_on_failure=False, merge_constraints: MergeConstraints | None = None, verbose=False)` – Runs a deterministic step list (validity repair → clearance improvement via `fix_clearance` → optional merging per `merge_constraints` → cleanup) using the pipeline runner. Returns `(best_geometry, warning)` where `warning` is a `FixWarning` or `None`. Set `raise_on_failure=True` to raise when constraints remain unmet.
- `robust_fix_batch(geometries, constraints, max_iterations=20, handle_overlaps=True, merge_constraints=None, properties=None, verbose=False)` – Applies the same pipeline to many geometries. When `properties` is provided (list of metadata dicts), the method returns `(fixed_geoms, warnings, carried_properties)` so you can keep the metadata aligned. If `handle_overlaps=True`, post-processes results with `remove_overlaps`.

Internally these use `polyforge.pipeline.run_steps`, so you can assemble your own `PipelineStep` list if you need bespoke behavior.

## Constraints & Pipeline Primitives (`polyforge.core`, `polyforge.pipeline`)
- `GeometryConstraints` – Dataclass describing the repair targets: `min_clearance`, `max_overlap_area`, `min_area_ratio`, `max_area_ratio`, `must_be_valid`, `allow_multipolygon`, `max_holes`, `min_hole_area`, `max_hole_aspect_ratio`, `min_hole_width`. Call `constraints.check(geometry, original, overlap_area=None, metrics=None)` to obtain a `ConstraintStatus`.
- `ConstraintStatus` – Holds `geometry`, `violations`, cached `metrics`, and `overlap_area`. Helper methods: `all_satisfied()`, `is_better_or_equal(other)`, `improved(other)`, `get_violations_by_type`, `worst_violation`.
- `ConstraintViolation` – Records `constraint_type` (a `ConstraintType` enum: `VALIDITY`, `CLEARANCE`, `OVERLAP`, `AREA_PRESERVATION`, `HOLE_VALIDITY`), `severity`, message, and optional `actual_value`/`required_value`.
- `MergeConstraints` – Enables merge-aware passes inside the robust pipeline. Fields: `enabled`, `margin`, `merge_strategy`, `preserve_holes`, `insert_vertices`.
- `FixConfig`, `PipelineContext`, `PipelineStep`, `StepResult`, `config_from_constraints`, `run_steps` (`polyforge.pipeline`) – Building blocks for custom pipelines. `PipelineContext.get_metrics()` caches measurements, `run_steps(initial_geometry, steps, context, max_passes=10)` keeps looping until `context.constraints.check(...)` passes or no step makes progress.

## Metrics & Measurement (`polyforge.metrics`)
- `measure_geometry(geometry, original=None, skip_clearance=False)` – Returns a dict with `is_valid`, `is_empty`, `clearance` (unless skipped), `area`, and `area_ratio` vs. `original`.
- `total_overlap_area(geometries)` – Sum of overlapping areas across the list (0 for <2 geometries).
- `overlap_area_by_geometry(geometries, min_area_threshold=1e-10)` – Returns a list where each entry is the overlap area attributed to that geometry (0 if none).

## Spatial Utilities
- `process_geometry(geometry, process_function, *args, **kwargs)` (`polyforge.process`) – Applies a NumPy-level function to the coordinates of any geometry type (Point, LineString, Polygon, Multi*, GeometryCollection). Handles 3D coordinates by interpolating Z values after simplification.
- `align_boundaries(poly1, poly2, distance_tolerance=1e-10)` (`polyforge.topology`) – Inserts missing vertices so two touching polygons share conforming boundaries (holes included), preventing T-junctions.
- `tile_polygon(polygon, tile_count=None, tile_size=None, axis_oriented=False)` (`polyforge.tile`) – Intersects a polygon with a generated grid (axis-aligned or oriented bounding box). Accept either a square count/size or `(cols, rows)` / `(width, height)`.
- `polyforge.core.geometry_utils` – Frequently used helpers: `to_single_polygon`, `remove_holes`, `validate_and_fix`, `safe_buffer_fix`, `update_coord_preserve_z`, `create_polygon_with_z_preserved`, `hole_shape_metrics`, etc.
- `polyforge.core.spatial_utils` – STRtree-powered helpers for graph construction and proximity: `find_polygon_pairs`, `build_adjacency_graph`, `find_nearest_boundary_point`, `point_to_segment_projection`, `build_segment_index`, `iterate_unique_pairs`.
- `polyforge.core.iterative_utils` – Generic iteration helpers: `iterative_improve`, `progressive_simplify`, `iterative_clearance_fix`.
- `polyforge.core.cleanup` – Backwards-compatible access to `CleanupConfig`, `cleanup_polygon`, `remove_small_holes`, `remove_narrow_holes`.

## Configuration Enums (`polyforge.core.types`)
- `SimplifyAlgorithm`: `RDP`, `VW`, `VWP`.
- `OverlapStrategy`: `split`, `largest`, `smallest`.
- `MergeStrategy`: `simple_buffer`, `selective_buffer`, `vertex_movement`, `boundary_extension`, `convex_bridges`.
- `RepairStrategy`: `auto`, `buffer`, `simplify`, `reconstruct`, `strict`.
- `CollapseMode`: `midpoint`, `first`, `last` — governs how short edges collapse.
- Clearance helpers: `HoleStrategy` (`remove`, `shrink`, `move`), `PassageStrategy` (`widen`, `split`), `IntrusionStrategy` (`fill`, `smooth`, `simplify`), `IntersectionStrategy` (`simplify`, `buffer`, `smooth`), `EdgeStrategy` (`simplify`, `buffer`).
- `coerce_enum(value, EnumClass)` – Convert a string or enum member to the desired enum, raising if unsupported.

## Error Types (`polyforge.core.errors`)
- `PolyforgeError` – Base class for all library-specific exceptions.
- `ValidationError` – Input geometry or parameters failed validation; exposes `.geometry` and `.issues`.
- `RepairError` – Entire repair pipeline failed (fields: `.geometry`, `.strategies_tried`, `.last_error`).
- `OverlapResolutionError` – Overlap removal exceeded iterations or produced invalid output; includes `.iterations`, `.remaining_overlaps`.
- `MergeError` – Merge strategy failure; includes `.strategy`, `.group_indices`.
- `ClearanceError` – Clearance routines could not meet the requested threshold; carries `.target_clearance`, `.achieved_clearance`, `.issue_type`.
- `ConfigurationError` – Invalid configuration values.
- `FixWarning` – Non-fatal warning signalling that `robust_fix_*` returned the best geometry found but constraints remain violated. Contains `.geometry`, `.status`, `.unmet_constraints`, `.history`.

## Advanced Ops Layer (`polyforge.ops`)
Use these when you need the raw building blocks without the orchestration:
- `polyforge.ops.simplify_ops` – NumPy coordinate functions (`simplify_rdp_coords`, `simplify_vw_coords`, `snap_short_edges`, `remove_duplicate_vertices`).
- `polyforge.ops.cleanup_ops` – `CleanupConfig`, `cleanup_polygon`, `remove_small_holes`, `remove_narrow_holes`.
- `polyforge.ops.clearance.*` – Low-level clearance primitives described earlier (holes, protrusions, passages, geometry utilities).
- `polyforge.ops.merge` & companions – Strategy implementations plus helpers like `find_close_boundary_pairs`, `get_boundary_points_near`, `insert_connection_vertices`, and `find_parallel_close_edges`.

These modules are pure functions with minimal dependencies, making them ideal for custom pipelines, tests, or experimentation beyond the default API.

---

## Putting It Together (Example)
```python
from shapely.geometry import Polygon
from polyforge import (
    simplify_vwp,
    fix_clearance,
    merge_close_polygons,
    robust_fix_geometry,
    GeometryConstraints,
    MergeStrategy,
)

# Clean and simplify
poly = simplify_vwp(raw_poly, threshold=0.05)
poly = fix_clearance(poly, min_clearance=1.0)

# Merge with nearby neighbors
merged = merge_close_polygons([poly] + nearby, margin=1.5, merge_strategy=MergeStrategy.BOUNDARY_EXTENSION)

# Enforce repair constraints
constraints = GeometryConstraints(min_clearance=1.0, min_area_ratio=0.95)
fixed, warn = robust_fix_geometry(merged[0], constraints)
if warn:
    print("Best-effort fix:", warn.unmet_constraints)
```

This flow hits the most common parts of the API: simplification, clearance repair, merging, and constraint-aware fixing. Mix and match modules as needed—each function is independent and works with standard Shapely geometries.
