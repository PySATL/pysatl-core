"""Registry used by array backends."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from pysatl_core.arrays.types import _Parameters, _ReturnType

type ArrayOperation = Callable[..., Any]


class UnsupportedArrayOperationError(LookupError):
    """Raised when an operation is absent from an array backend."""


class UnsupportedDTypeError(TypeError):
    """Raised when an array backend cannot represent a data type."""


class UnsupportedDeviceError(ValueError):
    """Raised when an array backend cannot use the requested device."""


class ArrayBackendFrozenError(RuntimeError):
    """Raised when code tries to mutate an initialized backend."""


class ArrayBackend:
    """
    Registry of array operations.

    Parameters
    ----------
    name : str
        Backend identifier.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._operations: dict[str, ArrayOperation] = {}
        self._frozen = False

    @property
    def operation_names(self) -> frozenset[str]:
        """Return registered operation names."""
        return frozenset(self._operations)

    def register(
        self, operation: Callable[_Parameters, _ReturnType], *, name: str | None = None
    ) -> Callable[_Parameters, _ReturnType]:
        """Register one operation and return it unchanged."""
        self._ensure_mutable()

        operation_name = name or operation.__name__
        if operation_name in self._operations:
            raise ValueError(f"Array operation '{operation_name}' is already registered.")

        self._operations[operation_name] = operation
        return operation

    def operation(self, name: str) -> ArrayOperation:
        """Return a registered operation."""
        try:
            return self._operations[name]
        except KeyError as exc:
            raise UnsupportedArrayOperationError(
                f"Array backend '{self.name}' does not support operation '{name}'."
            ) from exc

    def call(self, name: str, /, *args: Any, **kwargs: Any) -> Any:
        """Call a registered operation."""
        return self.operation(name)(*args, **kwargs)

    def missing_operations(self, required: Iterable[str]) -> frozenset[str]:
        """Return required operation names absent from this backend."""
        return frozenset(required) - self.operation_names

    def supports(self, required: Iterable[str]) -> bool:
        """Return whether all required operations are registered."""
        return not self.missing_operations(required)

    def freeze(self) -> None:
        """Prevent further operation registration."""
        self._frozen = True

    def _ensure_mutable(self) -> None:
        if self._frozen:
            raise ArrayBackendFrozenError(
                f"Array backend '{self.name}' is already initialized and cannot be changed."
            )


def array_operation(
    backend: ArrayBackend, name: str | None = None
) -> Callable[[Callable[_Parameters, _ReturnType]], Callable[_Parameters, _ReturnType]]:
    """Decorate a function and register it in ``backend``."""

    def decorator(
        operation: Callable[_Parameters, _ReturnType],
    ) -> Callable[_Parameters, _ReturnType]:
        return backend.register(operation, name=name)

    return decorator


__all__ = [
    "ArrayBackend",
    "ArrayBackendFrozenError",
    "ArrayOperation",
    "UnsupportedArrayOperationError",
    "UnsupportedDTypeError",
    "UnsupportedDeviceError",
    "array_operation",
]
