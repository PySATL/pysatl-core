"""Process-wide numerical backend state."""

from __future__ import annotations

from threading import Lock

from pysatl_core.arrays.backend import ArrayBackend
from pysatl_core.arrays.context import ArrayContext
from pysatl_core.arrays.runtime import Backend

_active_context: ArrayContext | None = None
_context_lock = Lock()


class BackendAlreadyInitializedError(RuntimeError):
    """Raised when code attempts to replace the active numerical backend."""


def _prepare_context(backend: Backend, device: str) -> ArrayContext:
    context = ArrayContext(backend=backend, device=device)
    context.arrays.native_dtype(context.default_dtype)
    context.arrays.freeze()
    return context


def initialize_backend(backend: Backend, *, device: str = "cpu") -> None:
    """Set the process-wide numerical backend exactly once."""
    global _active_context
    with _context_lock:
        if _active_context is not None:
            raise BackendAlreadyInitializedError("The numerical backend is already initialized.")
        _active_context = _prepare_context(backend, device)


def resolve_array_context() -> ArrayContext:
    """Return the active context, initializing the NumPy backend by default."""
    global _active_context
    if _active_context is not None:
        return _active_context

    with _context_lock:
        if _active_context is None:
            from pysatl_core.arrays.numpy import numpy_backend

            _active_context = _prepare_context(numpy_backend, "cpu")
        return _active_context


def resolve_backend() -> Backend:
    """Return the active root backend."""
    return resolve_array_context().backend


def resolve_array_backend() -> ArrayBackend:
    """Return the array registry of the active root backend."""
    return resolve_array_context().arrays


def _reset_backend() -> None:
    """Reset process state for tests."""
    global _active_context
    with _context_lock:
        _active_context = None


__all__ = [
    "BackendAlreadyInitializedError",
    "initialize_backend",
    "resolve_array_backend",
    "resolve_array_context",
    "resolve_backend",
]
