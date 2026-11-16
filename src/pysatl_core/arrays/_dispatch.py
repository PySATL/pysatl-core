"""Internal operation dispatch."""

from __future__ import annotations

from typing import Any

from pysatl_core.arrays.state import resolve_array_backend


def dispatch(operation: str, /, *args: Any, **kwargs: Any) -> Any:
    """Call one operation from the active array registry."""
    return resolve_array_backend().call(operation, *args, **kwargs)


__all__: list[str] = []
