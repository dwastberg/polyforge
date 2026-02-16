from .holes import fix_hole_too_close
from .protrusions import (
    fix_narrow_protrusion,
    fix_sharp_intrusion,
    remove_narrow_wedges,
)
from .remove_protrusions import remove_narrow_protrusions
from .passages import (
    fix_narrow_passage,
    fix_near_self_intersection,
    fix_parallel_close_edges,
)

__all__ = [
    "fix_hole_too_close",
    "fix_narrow_protrusion",
    "fix_sharp_intrusion",
    "remove_narrow_protrusions",
    "fix_narrow_passage",
    "fix_near_self_intersection",
    "fix_parallel_close_edges",
    "remove_narrow_wedges",
]
