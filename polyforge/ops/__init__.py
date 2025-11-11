"""Low-level geometry operations used by the high-level polyforge API.

The ops package groups pure functions that transform coordinate arrays or
geometries without depending on the public orchestration layers. Keeping these
helpers centralized makes it easier to reuse them across simplify/merge/repair
features while avoiding the web of cross-package imports that previously lived
under ``polyforge.core``.
"""

__all__: list[str] = []
