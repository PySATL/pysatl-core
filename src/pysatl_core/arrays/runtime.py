"""Root numerical backend composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pysatl_core.arrays.backend import ArrayBackend


@dataclass(frozen=True, slots=True)
class Backend:
    """Complete numerical backend selected for the current process."""

    identity: str
    arrays: ArrayBackend


__all__ = ["Backend"]
