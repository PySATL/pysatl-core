"""Backend-independent structural types for native values."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class NativeArray(Protocol):
    """Structural interface implemented by native backend arrays."""

    @property
    def dtype(self) -> object: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def ndim(self) -> int: ...


__all__ = ["NativeArray"]
