from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity


def analyze_geometry(geometry: BaseGeometry) -> dict:
    """Analyze geometry validity issues.

    Args:
        geometry: Shapely geometry to analyze.

    Returns:
        Dict with keys: is_valid, validity_message, geometry_type, is_empty, area,
        issues (list of issue descriptions), suggestions (list of fix suggestions).
    """
    metrics = _collect_geometry_metrics(geometry)
    issues, suggestions = _categorize_issues(
        metrics["is_valid"], metrics["validity_message"]
    )
    extra_issues, extra_suggestions = _collect_coordinate_issues(geometry)
    issues.extend(extra_issues)
    suggestions.extend(extra_suggestions)

    return _format_analysis(metrics, issues, suggestions)


def _collect_geometry_metrics(geometry: BaseGeometry) -> dict[str, object]:
    return {
        "is_valid": geometry.is_valid,
        "validity_message": explain_validity(geometry),
        "geometry_type": geometry.geom_type,
        "is_empty": geometry.is_empty,
        "area": geometry.area if hasattr(geometry, "area") else None,
    }


def _categorize_issues(
    is_valid: bool, validity_message: str
) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    suggestions: list[str] = []

    if not is_valid:
        msg = validity_message.lower()
        rules = [
            (
                ("self-intersection", "self intersection"),
                False,
                "Self-intersection",
                "Try buffer(0) or simplification",
            ),
            (("duplicate",), False, "Duplicate vertices", "Clean coordinates"),
            (
                ("not closed", "unclosed"),
                False,
                "Unclosed ring",
                "Close coordinate rings",
            ),
            (("ring", "invalid"), True, "Invalid ring", "Reconstruct ring geometry"),
            (("hole",), False, "Invalid hole", "Remove or fix interior rings"),
            (
                ("spike", "collapse"),
                False,
                "Collapsed/spike geometry",
                "Simplification or buffer",
            ),
        ]
        for keywords, match_all, issue, suggestion in rules:
            match_func = all if match_all else any
            if match_func(keyword in msg for keyword in keywords):
                issues.append(issue)
                suggestions.append(suggestion)

        if not issues:
            issues.append("Unknown validity issue")
            suggestions.append("Try auto-fix strategy")

    return issues, suggestions


def _collect_coordinate_issues(geometry: BaseGeometry) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    suggestions: list[str] = []

    if hasattr(geometry, "exterior"):
        coords = list(geometry.exterior.coords)
        if len(coords) < 4:
            issues.append("Too few vertices")
            suggestions.append("Geometry may be degenerate")

        for i in range(len(coords) - 1):
            if coords[i] == coords[i + 1]:
                issues.append("Consecutive duplicate vertices")
                suggestions.append("Clean coordinates")
                break

    return issues, suggestions


def _format_analysis(
    metrics: dict[str, object], issues: list[str], suggestions: list[str]
) -> dict[str, object]:
    return {
        **metrics,
        "issues": issues,
        "suggestions": suggestions,
    }


__all__ = ["analyze_geometry"]
