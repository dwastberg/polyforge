"""
Robust constraint-aware geometry fixing with transaction support.

This module provides a unified fixing approach that validates all constraints
after each fix and rolls back changes that cause regressions.
"""

from typing import Optional, Tuple, List, Dict, Any
import warnings

from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ..core.constraints import GeometryConstraints, ConstraintStatus, ConstraintType, MergeConstraints
from ..core.errors import FixWarning
from ..core.types import OverlapStrategy, RepairStrategy
from .core import repair_geometry
from .transaction import FixTransaction
from ..clearance.fix_clearance import fix_clearance
from ..overlap import remove_overlaps
from ..merge import merge_close_polygons
from ..simplify import remove_small_holes, remove_narrow_holes


def _merge_properties(
    source_geometries: List[BaseGeometry],
    source_properties: List[Dict[str, Any]],
    merged_geometry: BaseGeometry
) -> Dict[str, Any]:
    """
    Merge properties from source polygons when creating a merged polygon.

    Strategy: Keep properties from the largest source polygon.

    Args:
        source_geometries: Original geometries that were merged
        source_properties: Properties associated with each source geometry
        merged_geometry: The resulting merged geometry

    Returns:
        Properties dict for the merged polygon
    """
    if not source_geometries or not source_properties:
        return {}

    # Find the largest source polygon
    largest_idx = max(range(len(source_geometries)), key=lambda i: source_geometries[i].area)

    # Copy properties from largest
    merged_props = source_properties[largest_idx].copy() if source_properties[largest_idx] else {}

    return merged_props


def _validate_merge_group(
    merged_geometry: BaseGeometry,
    source_geometries: List[BaseGeometry],
    source_originals: List[BaseGeometry],
    constraints: GeometryConstraints,
    verbose: bool = False
) -> ConstraintStatus:
    """
    Validate a merged polygon against the union of its source polygons.

    For area preservation, we compare the merged result against the union
    of the original source polygons (before any fixes).

    Args:
        merged_geometry: The merged result
        source_geometries: Fixed source geometries that were merged
        source_originals: Original source geometries (before fixes)
        constraints: Constraints to validate
        verbose: Print diagnostic info

    Returns:
        ConstraintStatus for the merged geometry
    """
    # Create union of original source polygons for area comparison
    original_union = unary_union(source_originals)

    # Validate against constraints
    status = constraints.check(merged_geometry, original_union)

    if verbose and not status.all_satisfied():
        print(f"  Merge validation failed: {len(status.violations)} violation(s)")
        for v in status.violations:
            print(f"    - {v}")

    return status


def robust_fix_geometry(
    geometry: BaseGeometry,
    constraints: GeometryConstraints,
    max_iterations: int = 20,
    raise_on_failure: bool = False,
    merge_constraints: Optional[MergeConstraints] = None,
    verbose: bool = False
) -> Tuple[BaseGeometry, Optional[FixWarning]]:
    """
    Apply geometry fixes while ensuring no constraint regressions occur.

    This function uses a transactional approach to apply multiple fixes
    (repair, clearance, overlaps) while validating that all specified
    constraints are maintained after each fix. If a fix would cause any
    constraint to regress, it is automatically rolled back.

    The function applies fixes in priority order:
    1. Validity repair (if invalid)
    2. Clearance improvement (if below minimum)
    3. Overlap removal (if overlaps exist)
    4. Hole fixing (if invalid holes)
    5. MultiPolygon component merging (if merge_constraints provided)
    6. Final cleanup: Remove small holes (always removes zero-area holes)

    After each fix, ALL constraints are validated. If constraints cannot
    all be satisfied, the best result found is returned with a FixWarning.

    Args:
        geometry: The geometry to fix
        constraints: Constraints that must be satisfied (or best-effort if not possible)
        max_iterations: Maximum total fix iterations across all fix types (default: 20)
        raise_on_failure: If True, raise FixWarning when constraints not met.
                          If False, return best result with warning (default: False)
        merge_constraints: Optional configuration for merging MultiPolygon components
        verbose: Print detailed progress information (default: False)

    Returns:
        Tuple of (fixed_geometry, warning):
        - fixed_geometry: Best geometry found (may not satisfy all constraints)
        - warning: FixWarning if constraints not met, None if all satisfied

    Raises:
        FixWarning: Only if raise_on_failure=True and constraints not satisfied

    Example:
        ```python
        from polyforge import robust_fix_geometry
        from polyforge.core import GeometryConstraints, MergeConstraints, MergeStrategy

        # Define quality requirements
        constraints = GeometryConstraints(
            min_clearance=2.0,
            min_area_ratio=0.9,
            must_be_valid=True
        )

        # Optionally enable component merging for MultiPolygons
        merge_config = MergeConstraints(
            enabled=True,
            margin=0.5,
            merge_strategy=MergeStrategy.SELECTIVE_BUFFER
        )

        # Apply fixes with constraint validation
        fixed, warning = robust_fix_geometry(
            polygon,
            constraints,
            merge_constraints=merge_config
        )

        if warning:
            print(f"Warning: {warning}")
            print(f"Unmet constraints: {warning.unmet_constraints}")
        else:
            print("All constraints satisfied!")

        # Use the best result regardless
        use_polygon(fixed)
        ```

    See Also:
        - repair_geometry: Simple validity repair
        - fix_clearance: Clearance-specific fixing
        - remove_overlaps: Overlap resolution
        - GeometryConstraints: Constraint specification
    """
    original = geometry

    # Create transaction to manage fixes with rollback
    transaction = FixTransaction(geometry, original, constraints)

    if verbose:
        print(f"Starting robust fix - Initial status: {transaction.current_status}")
        print(f"Initial violations: {len(transaction.current_status.violations)}")

    # Track iterations
    iteration = 0
    made_progress = True

    # Keep trying to fix issues until:
    # 1. All constraints satisfied, OR
    # 2. No more progress being made, OR
    # 3. Max iterations reached
    while not transaction.current_status.all_satisfied() and made_progress and iteration < max_iterations:
        iteration += 1
        iteration_start_status = transaction.current_status

        if verbose:
            print(f"\n--- Iteration {iteration} ---")

        # Priority 1: Fix validity issues (highest priority)
        if constraints.must_be_valid and not transaction.current_status.validity:
            if verbose:
                print("Attempting validity repair...")

            success = transaction.try_fix(
                repair_geometry,
                fix_name="repair_validity",
                repair_strategy=RepairStrategy.AUTO
            )

            if verbose:
                if success:
                    print(f"  ✓ Validity repair succeeded")
                else:
                    print(f"  ✗ Validity repair caused regression, rolled back")

        # Priority 2: Fix clearance issues
        if constraints.min_clearance is not None:
            current_clearance = transaction.current_status.clearance

            # Only try to fix if we can measure clearance and it's too low
            if current_clearance is not None and current_clearance < constraints.min_clearance:
                if verbose:
                    print(f"Attempting clearance fix (current: {current_clearance:.4f}, target: {constraints.min_clearance:.4f})...")

                success = transaction.try_fix(
                    fix_clearance,
                    fix_name="fix_clearance",
                    min_clearance=constraints.min_clearance,
                    max_iterations=5  # Limit clearance iterations within each transaction attempt
                )

                if verbose:
                    if success:
                        new_clearance = transaction.current_status.clearance
                        print(f"  ✓ Clearance fix succeeded (new: {new_clearance:.4f})")
                    else:
                        print(f"  ✗ Clearance fix caused regression, rolled back")

        # Priority 3: Fix overlaps (if geometry is part of a batch - we handle single geometry here)
        # Note: For now, we skip overlap fixing for single geometries as it requires multiple polygons
        # This can be extended to handle MultiPolygon or batch scenarios
        if constraints.max_overlap_area > 0:
            # TODO: Handle overlap detection/fixing for single geometry (e.g., MultiPolygon with internal overlaps)
            if verbose:
                print("Overlap fixing not yet implemented for single geometry")

        # Check if we made progress this iteration
        if not transaction.current_status.improved(iteration_start_status):
            # No improvement - stop trying
            made_progress = False
            if verbose:
                print("\nNo progress made this iteration, stopping")

    # Priority 4: MultiPolygon component merging (if requested and applicable)
    if merge_constraints and merge_constraints.enabled:
        current_geom = transaction.current_geometry

        if isinstance(current_geom, MultiPolygon) and len(current_geom.geoms) > 1:
            if verbose:
                print(f"\n--- MultiPolygon Component Merging ---")
                print(f"Attempting to merge {len(current_geom.geoms)} components (margin={merge_constraints.margin})")

            try:
                # Extract components as list
                components = list(current_geom.geoms)

                # Try to merge components
                merged_components = merge_close_polygons(
                    components,
                    margin=merge_constraints.margin,
                    merge_strategy=merge_constraints.merge_strategy,
                    preserve_holes=merge_constraints.preserve_holes,
                    insert_vertices=merge_constraints.insert_vertices
                )

                if verbose:
                    print(f"  Merged {len(components)} → {len(merged_components)} component(s)")

                # Convert back to appropriate geometry type
                if len(merged_components) == 1:
                    merged_geom = merged_components[0]
                elif len(merged_components) > 1:
                    merged_geom = MultiPolygon(merged_components)
                else:
                    # Empty result - shouldn't happen, but handle gracefully
                    merged_geom = current_geom

                # Validate merged result
                if merge_constraints.validate_after_merge:
                    merged_status = constraints.check(merged_geom, original)

                    if merged_status.is_better_or_equal(transaction.current_status):
                        # Merge improved or maintained constraints
                        transaction.current_geometry = merged_geom
                        transaction.current_status = merged_status
                        if verbose:
                            print(f"  ✓ Component merge accepted (better or equal constraints)")
                    else:
                        if verbose:
                            print(f"  ✗ Component merge rejected (constraint regression)")
                else:
                    # No validation - accept merge
                    transaction.current_geometry = merged_geom
                    transaction.current_status = constraints.check(merged_geom, original)
                    if verbose:
                        print(f"  ✓ Component merge accepted (validation disabled)")

            except Exception as e:
                if verbose:
                    print(f"  Component merge failed: {e}")
                # Keep current geometry

    # Get the best result achieved
    best_geometry, best_status = transaction.get_best_result()

    # Final cleanup: Remove small holes
    # Always remove zero-area holes, optionally remove holes below min_hole_area
    if isinstance(best_geometry, (Polygon, MultiPolygon)):
        # First, always remove zero-area holes (using small epsilon)
        try:
            best_geometry = remove_small_holes(best_geometry, min_area=1e-10)
            if verbose:
                print(f"\n--- Final Cleanup: Zero-area holes removed ---")
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to remove zero-area holes: {e}")

        # Then, if min_hole_area is specified, remove small holes
        if constraints.min_hole_area is not None and constraints.min_hole_area > 0:
            try:
                before_count = 0
                if isinstance(best_geometry, Polygon):
                    before_count = len(best_geometry.interiors)
                elif isinstance(best_geometry, MultiPolygon):
                    before_count = sum(len(p.interiors) for p in best_geometry.geoms)

                best_geometry = remove_small_holes(best_geometry, min_area=constraints.min_hole_area)

                after_count = 0
                if isinstance(best_geometry, Polygon):
                    after_count = len(best_geometry.interiors)
                elif isinstance(best_geometry, MultiPolygon):
                    after_count = sum(len(p.interiors) for p in best_geometry.geoms)

                if verbose and before_count > after_count:
                    print(f"--- Final Cleanup: Removed {before_count - after_count} small holes (< {constraints.min_hole_area}) ---")

                # Re-validate after hole removal
                best_status = constraints.check(best_geometry, original)
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to remove small holes: {e}")

        # Finally, if max_hole_aspect_ratio or min_hole_width is specified, remove narrow holes
        if ((constraints.max_hole_aspect_ratio is not None and constraints.max_hole_aspect_ratio > 0) or
            (constraints.min_hole_width is not None and constraints.min_hole_width > 0)):
            try:
                before_count = 0
                if isinstance(best_geometry, Polygon):
                    before_count = len(best_geometry.interiors)
                elif isinstance(best_geometry, MultiPolygon):
                    before_count = sum(len(p.interiors) for p in best_geometry.geoms)

                # Use default aspect ratio if not specified
                max_aspect = constraints.max_hole_aspect_ratio if constraints.max_hole_aspect_ratio is not None else 50.0

                best_geometry = remove_narrow_holes(
                    best_geometry,
                    max_aspect_ratio=max_aspect,
                    min_width=constraints.min_hole_width
                )

                after_count = 0
                if isinstance(best_geometry, Polygon):
                    after_count = len(best_geometry.interiors)
                elif isinstance(best_geometry, MultiPolygon):
                    after_count = sum(len(p.interiors) for p in best_geometry.geoms)

                if verbose and before_count > after_count:
                    filters = []
                    if constraints.max_hole_aspect_ratio is not None:
                        filters.append(f"aspect > {constraints.max_hole_aspect_ratio}")
                    if constraints.min_hole_width is not None:
                        filters.append(f"width < {constraints.min_hole_width}")
                    filter_str = " or ".join(filters)
                    print(f"--- Final Cleanup: Removed {before_count - after_count} narrow holes ({filter_str}) ---")

                # Re-validate after hole removal
                best_status = constraints.check(best_geometry, original)
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to remove narrow holes: {e}")

    if verbose:
        print(f"\n=== Final Result ===")
        print(f"Total iterations: {iteration}")
        print(f"Best status: {best_status}")
        print(f"Violations: {len(best_status.violations)}")
        if best_status.violations:
            for v in best_status.violations:
                print(f"  - {v}")

    # Check if all constraints satisfied
    if best_status.all_satisfied():
        # Success!
        return best_geometry, None

    # Not all constraints satisfied - create warning
    unmet_constraint_types = [
        v.constraint_type.name for v in best_status.violations
    ]

    history = transaction.get_history_summary()

    warning = FixWarning(
        message=f"Could not satisfy all constraints after {iteration} iterations. "
                f"Returning best result found with {len(best_status.violations)} unmet constraint(s).",
        geometry=best_geometry,
        status=best_status,
        unmet_constraints=unmet_constraint_types,
        history=history
    )

    # Either raise or return with warning based on flag
    if raise_on_failure:
        raise warning
    else:
        # Issue a Python warning so it shows up in warnings filters
        warnings.warn(str(warning), UserWarning, stacklevel=2)
        return best_geometry, warning


def robust_fix_batch(
    geometries: List[BaseGeometry],
    constraints: GeometryConstraints,
    max_iterations: int = 20,
    handle_overlaps: bool = True,
    merge_constraints: Optional[MergeConstraints] = None,
    properties: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = False
) -> Tuple[List[BaseGeometry], List[Optional[FixWarning]], Optional[List[Dict[str, Any]]]]:
    """
    Apply robust fixing to multiple geometries, optionally handling inter-geometry overlaps and merging.

    This function is similar to robust_fix_geometry but handles multiple geometries
    at once, allowing for overlap detection and resolution between geometries, and
    optionally merging close polygons.

    The pipeline consists of:
    - Phase 0.5: Polygon merging (if merge_constraints.enabled=True) - FIRST!
    - Phase 1: Individual geometry fixes (validity, clearance, holes)
    - Phase 2: Overlap resolution (if handle_overlaps=True)
    - Phase 3: Final constraint validation

    Note: Merging happens BEFORE fixes to prevent individual fixes from pushing
    polygons apart and making merging impossible.

    Args:
        geometries: List of geometries to fix
        constraints: Constraints that must be satisfied for each geometry
        max_iterations: Maximum fix iterations per geometry (default: 20)
        handle_overlaps: If True, detect and resolve overlaps between geometries (default: True)
        merge_constraints: Configuration for polygon merging (default: None/disabled)
        properties: Optional list of property dicts (one per geometry). If provided and
                    merging is enabled, properties are merged using largest-polygon strategy.
        verbose: Print detailed progress information (default: False)

    Returns:
        Tuple of (fixed_geometries, warnings, merged_properties):
        - fixed_geometries: List of best geometries found (may be fewer if merged)
        - warnings: List of FixWarning (None if constraints met for that geometry)
        - merged_properties: List of property dicts if properties were provided, else None

    Example:
        ```python
        from polyforge import robust_fix_batch
        from polyforge.core import GeometryConstraints, MergeConstraints, MergeStrategy

        constraints = GeometryConstraints(
            min_clearance=2.0,
            min_area_ratio=0.9,
            must_be_valid=True,
            max_overlap_area=0.0  # No overlaps allowed
        )

        merge_config = MergeConstraints(
            enabled=True,
            margin=0.5,
            merge_strategy=MergeStrategy.SELECTIVE_BUFFER,
            validate_after_merge=True,
            fix_violations=True
        )

        fixed, warnings, props = robust_fix_batch(
            polygons,
            constraints,
            merge_constraints=merge_config,
            properties=polygon_properties
        )

        for i, (poly, warning) in enumerate(zip(fixed, warnings)):
            if warning:
                print(f"Polygon {i}: {warning}")
        ```
    """
    if not geometries:
        return [], [], None if properties is None else []

    originals = geometries.copy()

    # Initialize properties tracking if not provided
    if properties is None:
        props = [{}] * len(geometries)
    else:
        props = [p.copy() if p else {} for p in properties]

    # Phase 0.5: Polygon merging FIRST (before individual fixes!)
    # This prevents individual fixes from pushing polygons apart
    working_geometries = geometries.copy()
    working_props = props.copy()
    working_originals = originals.copy()

    if merge_constraints and merge_constraints.enabled:
        if verbose:
            print(f"\n=== Phase 0.5: Polygon Merging (BEFORE fixes, margin={merge_constraints.margin}) ===")

        # Convert to Polygons for merging
        polygons = [g for g in working_geometries if isinstance(g, (Polygon, MultiPolygon))]

        if len(polygons) == len(working_geometries):
            # Extract just Polygons (expand MultiPolygons if needed)
            polygon_list = []
            for p in polygons:
                if isinstance(p, Polygon):
                    polygon_list.append(p)
                elif isinstance(p, MultiPolygon):
                    # For MultiPolygon, take largest component
                    polygon_list.append(max(p.geoms, key=lambda g: g.area))

            try:
                # Apply merging using merge_close_polygons
                merged_result = merge_close_polygons(
                    polygon_list,
                    margin=merge_constraints.margin,
                    merge_strategy=merge_constraints.merge_strategy,
                    preserve_holes=merge_constraints.preserve_holes,
                    insert_vertices=merge_constraints.insert_vertices
                )

                if verbose:
                    print(f"  Merged {len(polygon_list)} → {len(merged_result)} polygons")

                # Track mapping: which merged polygon came from which sources
                from shapely.strtree import STRtree

                # Build spatial index of source polygons
                source_tree = STRtree(polygon_list)

                # Map merged polygons back to their sources
                new_geometries = []
                new_props = []
                new_originals = []

                for merged_geom in merged_result:
                    # Find sources that contributed to this merged polygon
                    candidate_indices = source_tree.query(merged_geom)
                    source_indices = []

                    for idx in candidate_indices:
                        source = polygon_list[idx]
                        # Check if this source is contained or significantly overlaps
                        if merged_geom.contains(source) or merged_geom.overlaps(source) or source.contains(merged_geom):
                            source_indices.append(idx)

                    if not source_indices:
                        # Shouldn't happen, but handle gracefully - keep as-is
                        new_geometries.append(merged_geom)
                        new_props.append({})
                        new_originals.append(merged_geom)
                        continue

                    # Get source geometries and properties
                    source_geoms = [polygon_list[i] for i in source_indices]
                    source_props_list = [working_props[i] for i in source_indices]
                    source_origs = [working_originals[i] for i in source_indices]

                    # Merge properties from source polygons (use largest)
                    merged_props = _merge_properties(source_geoms, source_props_list, merged_geom)

                    # Use union of original sources as the "original" for this merged polygon
                    merged_original = unary_union(source_origs)

                    new_geometries.append(merged_geom)
                    new_props.append(merged_props)
                    new_originals.append(merged_original)

                # Update working sets with merged results
                working_geometries = new_geometries
                working_props = new_props
                working_originals = new_originals

                if verbose:
                    print(f"  Result: {len(working_geometries)} polygons after merging")

            except Exception as e:
                if verbose:
                    print(f"  Polygon merging failed: {e}")
                    import traceback
                    traceback.print_exc()
                # Keep original geometries

    # Phase 1: Fix individual geometries (validity, clearance, holes)
    # Note: These may already be merged from Phase 0.5
    fixed = []
    warnings_list = []

    if verbose:
        print(f"\n=== Phase 1: Individual Fixes ({len(working_geometries)} geometries) ===")

    for i, (geom, original) in enumerate(zip(working_geometries, working_originals)):
        if verbose:
            print(f"\nGeometry {i+1}/{len(working_geometries)}")

        # Create constraints without overlap checking for individual fix phase
        individual_constraints = GeometryConstraints(
            min_clearance=constraints.min_clearance,
            min_area_ratio=constraints.min_area_ratio,
            max_area_ratio=constraints.max_area_ratio,
            must_be_valid=constraints.must_be_valid,
            allow_multipolygon=constraints.allow_multipolygon,
            max_holes=constraints.max_holes,
            max_overlap_area=0.0  # Don't check overlaps in individual phase
        )

        fixed_geom, warning = robust_fix_geometry(
            geom,
            individual_constraints,
            max_iterations=max_iterations,
            raise_on_failure=False,
            verbose=verbose
        )

        fixed.append(fixed_geom)
        warnings_list.append(warning)

    # Phase 2: Handle inter-geometry overlaps if requested
    if handle_overlaps and constraints.max_overlap_area == 0.0:
        if verbose:
            print(f"\n=== Phase 2: Overlap Resolution ===")

        # Convert to Polygons for overlap handling
        polygons = [g for g in fixed if isinstance(g, (Polygon, MultiPolygon))]

        if len(polygons) == len(fixed):
            # Extract just Polygons (expand MultiPolygons if needed)
            polygon_list = []
            for p in polygons:
                if isinstance(p, Polygon):
                    polygon_list.append(p)
                elif isinstance(p, MultiPolygon):
                    # For MultiPolygon, take largest component
                    polygon_list.append(max(p.geoms, key=lambda g: g.area))

            try:
                # Use remove_overlaps to handle overlaps
                no_overlaps = remove_overlaps(
                    polygon_list,
                    overlap_strategy=OverlapStrategy.SPLIT,
                    max_iterations=max_iterations
                )

                # Validate that overlap removal didn't violate other constraints
                for i, (new_geom, original, old_warning) in enumerate(zip(no_overlaps, working_originals, warnings_list)):
                    # Check if this geometry still satisfies constraints
                    status = constraints.check(new_geom, original)

                    if status.all_satisfied():
                        # Overlap removal succeeded and constraints maintained
                        fixed[i] = new_geom
                        if old_warning and old_warning.status.get_violations_by_type(ConstraintType.OVERLAP):
                            # Overlap was the only issue - clear the warning
                            warnings_list[i] = None
                    else:
                        # Overlap removal caused other constraints to fail
                        # Keep the geometry from Phase 1
                        if verbose:
                            print(f"  Geometry {i}: overlap removal caused regression, keeping Phase 1 result")

            except Exception as e:
                if verbose:
                    print(f"Overlap removal failed: {e}")
                # Keep Phase 1 results

    # Phase 3: Final cleanup after overlap resolution
    # Overlap resolution can create thin holes and degenerate exterior features
    if verbose:
        print(f"\n=== Phase 3: Final Cleanup ===")

    for i, geom in enumerate(fixed):
        if not isinstance(geom, (Polygon, MultiPolygon)):
            continue

        original_geom = geom
        cleaned = geom

        # Step 1: Remove zero-area holes
        try:
            cleaned = remove_small_holes(cleaned, min_area=1e-10)
        except Exception as e:
            if verbose:
                print(f"  Geometry {i}: Could not remove zero-area holes: {e}")

        # Step 2: Remove small holes
        if constraints.min_hole_area is not None and constraints.min_hole_area > 0:
            try:
                cleaned = remove_small_holes(cleaned, min_area=constraints.min_hole_area)
            except Exception as e:
                if verbose:
                    print(f"  Geometry {i}: Could not remove small holes: {e}")

        # Step 3: Remove narrow holes
        if ((constraints.max_hole_aspect_ratio is not None and constraints.max_hole_aspect_ratio > 0) or
            (constraints.min_hole_width is not None and constraints.min_hole_width > 0)):
            try:
                max_aspect = constraints.max_hole_aspect_ratio if constraints.max_hole_aspect_ratio is not None else 50.0
                cleaned = remove_narrow_holes(
                    cleaned,
                    max_aspect_ratio=max_aspect,
                    min_width=constraints.min_hole_width
                )
            except Exception as e:
                if verbose:
                    print(f"  Geometry {i}: Could not remove narrow holes: {e}")

        # Step 4: Remove degenerate exterior features (zero-width slivers)
        # These are created by overlap resolution and have clearance near zero
        if cleaned.is_valid and not cleaned.is_empty:
            try:
                # Check if geometry has very low clearance (< 0.01)
                clearance = cleaned.minimum_clearance
                if clearance < 0.01:
                    # Apply erosion-dilation to remove zero-width features
                    # Use a very small buffer to only affect degenerate parts
                    eroded = cleaned.buffer(-0.01, join_style=2)  # MITRE join
                    if eroded.is_valid and not eroded.is_empty and eroded.area > 0:
                        dilated = eroded.buffer(0.01, join_style=2)
                        if dilated.is_valid and not dilated.is_empty:
                            # Check if area loss is acceptable (< 1%)
                            area_loss = (cleaned.area - dilated.area) / cleaned.area
                            if area_loss < 0.01:
                                cleaned = dilated
                                if verbose:
                                    print(f"  Geometry {i}: Removed degenerate features (clearance: {clearance:.4f} -> {dilated.minimum_clearance:.4f})")
                else:
                    # Still try buffer(0) for other topological cleanup
                    buffered = cleaned.buffer(0)
                    if buffered.is_valid and not buffered.is_empty and abs(buffered.area - cleaned.area) < 0.1:
                        cleaned = buffered
            except Exception as e:
                if verbose:
                    print(f"  Geometry {i}: Degenerate feature removal failed: {e}")
                # Try buffer(0) as fallback
                try:
                    buffered = cleaned.buffer(0)
                    if buffered.is_valid and not buffered.is_empty:
                        cleaned = buffered
                except:
                    pass

        # Re-validate after cleanup
        if i < len(working_originals):
            status = constraints.check(cleaned, working_originals[i])
            if status.all_satisfied() or status.is_better_or_equal(constraints.check(original_geom, working_originals[i])):
                fixed[i] = cleaned
            else:
                if verbose:
                    print(f"  Geometry {i}: Cleanup caused regression, keeping original")

    # Return results with properties if they were provided
    return_props = working_props if properties is not None else None
    return fixed, warnings_list, return_props


__all__ = [
    'robust_fix_geometry',
    'robust_fix_batch',
]
