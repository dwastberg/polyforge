# Simplify Design

## Architecture Snapshot
- Runtime code fans out into `polyforge/simplify.py`, `polyforge/clearance/`, `polyforge/merge/`, `polyforge/overlap/`, `polyforge/repair/`, and `polyforge/core/`. Each subpackage wraps small Shapely helpers in dedicated modules (e.g., ten enums inside `polyforge/core/types.py:9`, staged pipelines inside `polyforge/repair/stages.py:25`, and STRtree helpers inside `polyforge/core/spatial_utils.py:8`).
- High-level APIs (exported from `polyforge/__init__.py:6`) largely forward to those submodules without sharing state, yet an extensive layer of classes (`GeometryConstraints`, `FixTransaction`, `StageContext`, etc.) orchestrates the order of operations.
- Tests interact with the public functions, so the shape of the API matters more than the exact module layout; however, the current implementation forces any change through multiple abstraction layers.

## Overengineering Symptoms

### 1. Transactional repair pipeline
- `polyforge/repair/stages.py:25` defines `StageResult`, `StageContext`, and `FixStage` dataclasses plus a registry of stage factories. Each stage delegates back into `FixTransaction.try_fix`, so the indirection only forwards parameters.
- `polyforge/repair/transaction.py:16` keeps full snapshot history, severity comparisons, and rollback helpers, even though the pipeline always runs deterministically from the current geometry and keeps the best result separately in `robust_fix_geometry`.
- Combining the two means every fix call flows through: user -> `robust_fix_geometry` -> `FixTransaction` -> `FixStage` -> real shapely operation. None of the intermediate classes add geometry knowledge; they only repeat bookkeeping already available from simpler metrics.

### 2. Constraint & enum explosion
- `polyforge/core/types.py:9` exports ten different enums for strategy choices, many of which only toggle between two obvious options (e.g., `EdgeStrategy` with `SIMPLIFY` vs `BUFFER`). This spreads a single conceptual toggle across multiple files, adds typing friction, and forces users to import enums for everyday use.
- The constraint stack (`polyforge/core/constraints.py:23`) defines `ConstraintType`, `ConstraintViolation`, `ConstraintStatus`, `GeometryConstraints`, and `MergeConstraints`. Much of the code simply packages floats into objects and compares counts/severities. A functional approach could measure the few metrics we care about (validity, clearance, area ratio, overlap) and return plain dictionaries.

### 3. Clearance micro-framework
- `polyforge/clearance/fix_clearance.py:24` introduces another enum (`ClearanceIssue`), diagnostic dataclasses, a registry decorator, and iterative helpers, even though the function ultimately calls one of a handful of fixers (`fix_hole_too_close`, `remove_narrow_protrusions`, etc.). The registry and dataclasses add indirection without enabling extensibility—every strategy is defined in the same file.
- Many of those helper functions already depend on the generic enums defined elsewhere, so we end up hopping between three layers of configuration just to remove a spike from a polygon.

### 4. Merge and overlap layering
- `polyforge/merge/core.py:12` funnels into five strategy modules and several utilities packages (`polyforge/merge/utils/`, `polyforge/core/spatial_utils.py:250`). The same pattern (STRtree, nearest-points, simple buffer) repeats across modules, but each is wrapped behind its own API, making it hard to trace data flow.
- `remove_overlaps` (`polyforge/overlap/__init__.py:9`) wraps `resolve_overlap_pair`, which itself lives in `polyforge/overlap/engine.py` with another dataclass and helper methods. Each layer just forwards polygons while reformatting them.

### 5. Core module sprawl & duplication
- Cleanup utilities exist both in `polyforge/core/cleanup.py:11` and as public helpers in `polyforge/simplify.py:200`, forcing the same function signature to be maintained twice.
- Supporting utilities such as `iterative_improve` (`polyforge/core/iterative_utils.py:10`) and `geometry_utils` (`polyforge/core/geometry_utils.py:9`) sit in separate modules even though they are thin wrappers. Pulling them together would reduce import churn and clarify the flow of data.

## Simplification Principles
1. **Functional pipelines** – model every operation as `Geometry -> Geometry` functions with explicit arguments; collect logging/metrics through return values rather than hidden state.
2. **Single configuration surface** – replace scattered enums/dataclasses with one lightweight config object (e.g., `FixConfig` holding `min_clearance`, `merge_margin`, `cleanup` toggles). Strategy selection becomes a dictionary lookup keyed by simple literals.
   - **Enum boundary only** – keep the existing Enums at the public API edge so callers retain IDE/autocomplete benefits, but immediately convert them to their string values before passing data into the ops layer.
3. **Measure, don’t model** – compute the few metrics we need (`is_valid`, `minimum_clearance`, `area_ratio`, `overlap_area`) via plain functions. Let callers compose validation rules by checking those numbers instead of instantiating `ConstraintStatus`.
4. **Lean module layout** – collapse micro-modules into topical files: `ops/simplify.py`, `ops/cleanup.py`, `ops/clearance.py`, `ops/merge.py`, and shared `metrics.py`. Keep `pipeline.py` as the sole orchestrator.
5. **Keep high-level API stable** – continue exporting `simplify_vwp`, `fix_clearance`, etc., but implement them via the functional core so callers see the same names without the abstract machinery.

## Proposed Functional Architecture

```
polyforge/
  ops/
    simplify_ops.py      # collapse process_geometry + simplification helpers
    cleanup_ops.py       # hole removal, dedupe, sliver trimming
    clearance_ops.py     # direct fixes, no registries
    merge_ops.py         # shared STRtree helpers + strategy functions
  metrics.py             # measure validity, clearance, area, overlaps
  pipeline.py            # run_steps(geometry, steps, config)
  config.py              # FixConfig + helper constructors
```

- **Step functions** – standardized signature `Step = Callable[[Geometry, Context], StepResult]` where `StepResult` is a small dataclass `(geometry, changed: bool, note: str)`. No transactions; any step simply returns the new geometry plus metadata.
- **Pipeline runner** – `run_pipeline(geometry, steps, config, max_passes)` loops over steps, recomputes metrics via `metrics.measure(geometry, config)`, and short-circuits when all requested metrics pass. Because steps are plain functions, custom workflows become trivial lists like `steps = [repair_validity, widen_clearance, cleanup, merge_components]`.
- **Config + strategies** – replace enums with simple dictionaries:
  ```python
  CLEARANCE_FIXES = {
      "hole": fix_hole_too_close,
      "protrusion": remove_narrow_protrusions,
  }
  ```
  The config chooses strings, the pipeline looks them up, and unit tests can patch dictionaries without importing enums.
- **Metrics-first validation** – `metrics.measure(geometry, original)` returns a dict such as `{"is_valid": True, "clearance": 1.7, "area_ratio": 0.94}`. The pipeline compares against thresholds declared inside `FixConfig`. No `ConstraintStatus` objects or severity calculations are required.
- **Shared STRtree helpers** – move `find_polygon_pairs`, `build_segment_index`, etc., into `ops/merge_ops.py` next to the strategies so callers can see the full data flow inside one module.

## Comparison with SIMPLIFY_CODE_DESIGN

- **Where we agree** – Both approaches delete the transactional repair pipeline, flatten stage orchestration into plain functions, and keep the public API untouched. The other document’s quantified file/line savings reinforce the urgency of this cleanup and the focus on consolidating the `repair`, `merge`, and `clearance` packages.
- **Where we diverge** – Their plan keeps most enums and `GeometryConstraints`, merely shrinking them, whereas this proposal replaces them with literal-driven configs and direct metric checks. They also prefer collapsing code into a handful of large modules (e.g., `repair.py`, `merge.py`), while this design favors an `ops/` namespace that groups purely functional helpers by concern.
- **Adopted adjustments** – Two concrete ideas strengthen this plan:
  1. Drop unnecessary wrapper files such as `split.py` and re-export `resolve_overlap_pair` directly; this aligns with the other document’s “delete thin wrappers” recommendation.
  2. Collapse the overlap helpers back into a single `overlap.py` module (still called through the ops layer) so the resolver and batch removal logic live side by side, mirroring their observation that the current package layering obscures control flow.

These tweaks fold naturally into the functional architecture (the `overlap` ops module becomes that single file, and the public API now references it directly) without sacrificing the broader goals around metrics-first validation and lightweight configs.

## Refactor Plan

1. **Document operations (Week 1)**  
   - Extract the pure geometry helpers from `polyforge/simplify.py:200`, `polyforge/core/cleanup.py:11`, `polyforge/clearance/*.py`, and `polyforge/merge/*.py` into `polyforge/ops/*` modules without changing behavior.  
   - In the same sweep, delete `split.py` and point the public API at the consolidated overlap module so there are no redundant wrappers.
   - Add `metrics.measure_geometry` that wraps `is_valid`, `minimum_clearance`, `area_ratio`, and `overlap_area`.

2. **Introduce the functional pipeline (Week 2)**  
   - Implement `pipeline.run_steps` plus a minimal `FixConfig`/`StepResult`.  
   - Rewire `robust_fix_geometry` (`polyforge/repair/robust.py:20`) to call `run_pipeline` with a list of step functions instead of `FixTransaction`/`FixStage`. Keep old classes temporarily but mark them deprecated.

3. **Flatten configuration (Week 3)**  
   - Replace enum parameters in public APIs with `Literal` strings; keep enum aliases for backward compatibility but have them return `.value` to the new functional layer.  
   - Remove `GeometryConstraints`/`ConstraintStatus` in favor of helper functions inside `config.py` and `metrics.py`. Update high-level helpers (`merge_close_polygons`, `fix_clearance`) to accept the new config dicts.

4. **Delete unused abstractions (Week 4)**  
   - Remove `FixTransaction`, `FixStage`, enum modules that now only forward to strings, and the clearance strategy registry.  
   - Collapse `polyforge/core` into `ops/` + `metrics.py`, keeping only custom exceptions in `core/errors.py` if they are still valuable.

Throughout the refactor, keep the public API surface identical (function names + keyword args) and rely on the test suite to confirm behavior. Internally, the code path becomes `public wrapper -> ops function -> shapely`, without orchestration classes.

## Open Questions / Follow-ups
- How much of the existing warning/error hierarchy (`polyforge/core/errors.py:1`) still matters once constraint objects disappear? A lighter-weight exception set (`PolyforgeError`, `FixWarning`) may be enough.
- Should we keep the current overlap/merge strategy diversity, or collapse to a single pragmatic approach? After porting to `ops/merge.py`, we can measure test coverage and decide whether rarely used strategies are worth maintaining.
- The new pipeline will recompute metrics after every step; if this becomes a bottleneck on large batches we can memoize measurements inside the context, but only if profiling shows a real hit.
