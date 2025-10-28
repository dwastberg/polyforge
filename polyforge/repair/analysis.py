"""Geometry analysis and diagnostic functions."""

from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity


def analyze_geometry(geometry: BaseGeometry) -> dict:
    """Analyze geometry validity issues.

    Returns a dictionary with diagnostic information about the geometry.

    Args:
        geometry: Geometry to analyze

    Returns:
        Dictionary with keys:
            - 'is_valid': bool
            - 'validity_message': str (from Shapely)
            - 'issues': list of detected issues
            - 'suggestions': list of suggested repairs

    Examples:
        >>> poly = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
        >>> analysis = analyze_geometry(poly)
        >>> analysis['is_valid']
        False
        >>> 'Self-intersection' in analysis['issues']
        True
    """
    issues = []
    suggestions = []

    # Check validity
    is_valid = geometry.is_valid
    validity_msg = explain_validity(geometry)

    if not is_valid:
        # Analyze validity message
        msg_lower = validity_msg.lower()

        if 'self-intersection' in msg_lower or 'self intersection' in msg_lower:
            issues.append('Self-intersection')
            suggestions.append('Try buffer(0) or simplification')

        if 'duplicate' in msg_lower:
            issues.append('Duplicate vertices')
            suggestions.append('Clean coordinates')

        if 'not closed' in msg_lower or 'unclosed' in msg_lower:
            issues.append('Unclosed ring')
            suggestions.append('Close coordinate rings')

        if 'ring' in msg_lower and 'invalid' in msg_lower:
            issues.append('Invalid ring')
            suggestions.append('Reconstruct ring geometry')

        if 'hole' in msg_lower:
            issues.append('Invalid hole')
            suggestions.append('Remove or fix interior rings')

        if 'spike' in msg_lower or 'collapse' in msg_lower:
            issues.append('Collapsed/spike geometry')
            suggestions.append('Simplification or buffer')

        if not issues:
            issues.append('Unknown validity issue')
            suggestions.append('Try auto-fix strategy')

    # Check for other potential issues
    if hasattr(geometry, 'exterior'):
        exterior_coords = list(geometry.exterior.coords)
        if len(exterior_coords) < 4:
            issues.append('Too few vertices')
            suggestions.append('Geometry may be degenerate')

        # Check for duplicate consecutive vertices
        for i in range(len(exterior_coords) - 1):
            if exterior_coords[i] == exterior_coords[i + 1]:
                issues.append('Consecutive duplicate vertices')
                suggestions.append('Clean coordinates')
                break

    return {
        'is_valid': is_valid,
        'validity_message': validity_msg,
        'issues': issues,
        'suggestions': suggestions,
        'geometry_type': geometry.geom_type,
        'is_empty': geometry.is_empty,
        'area': geometry.area if hasattr(geometry, 'area') else None,
    }


__all__ = ['analyze_geometry']
