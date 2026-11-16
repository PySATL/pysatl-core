"""Registry used by array backends."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from typing import Any, ParamSpec, TypeVar

from pysatl_core.arrays.dtypes import DType

type ArrayOperation = Callable[..., Any]
type _DTypeNormalizer = Callable[[object], Hashable]

_Parameters = ParamSpec("_Parameters")
_ReturnType = TypeVar("_ReturnType")


class UnsupportedArrayOperationError(LookupError):
    """Raised when an operation is absent from an array backend."""


class UnsupportedDTypeError(TypeError):
    """Raised when an array backend cannot represent a data type."""


class ArrayBackendFrozenError(RuntimeError):
    """Raised when code tries to mutate an initialized backend."""


class ArrayBackend:
    """
    Registry of array operations and native data types.

    Parameters
    ----------
    name : str
        Backend identifier.
    dtype_normalizer : Callable[[object], Hashable]
        Function that converts a native data type to a stable mapping key.
    """

    def __init__(
        self,
        name: str,
        *,
        dtype_normalizer: _DTypeNormalizer,
    ) -> None:
        self.name = name
        self._dtype_normalizer = dtype_normalizer
        self._operations: dict[str, ArrayOperation] = {}
        self._native_dtypes: dict[DType, Hashable] = {}
        self._dtypes: dict[Hashable, DType] = {}
        self._frozen = False

    @property
    def supported_dtypes(self) -> frozenset[DType]:
        """Return backend-independent data types registered by this backend."""
        return frozenset(self._native_dtypes)

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

    def register_dtype(self, dtype: DType, native_dtype: object) -> None:
        """Bind a backend-independent data type to its native representation."""
        self._ensure_mutable()

        normalized = self._normalize_native_dtype(native_dtype)
        if dtype in self._native_dtypes and self._native_dtypes[dtype] != normalized:
            raise ValueError(f"DType '{dtype}' is already mapped to another native dtype.")

        if normalized in self._dtypes and self._dtypes[normalized] != dtype:
            raise ValueError(f"Native dtype {native_dtype!r} is already mapped to another DType.")

        self._native_dtypes[dtype] = normalized
        self._dtypes[normalized] = dtype

    def normalize_dtype(self, dtype: DType | object) -> DType:
        """Convert a native data type such as ``numpy.int64`` to :class:`DType`."""
        if isinstance(dtype, DType):
            if dtype in self._native_dtypes:
                return dtype

            raise UnsupportedDTypeError(
                f"Array backend '{self.name}' does not support dtype {dtype!r}."
            )

        normalized = self._normalize_native_dtype(dtype)

        try:
            return self._dtypes[normalized]
        except KeyError as exc:
            raise UnsupportedDTypeError(
                f"Array backend '{self.name}' does not support dtype {dtype!r}."
            ) from exc

    def native_dtype(self, dtype: DType | object) -> Hashable:
        """Convert a shared or native data type to the backend representation."""
        return self._native_dtypes[self.normalize_dtype(dtype)]

    def result_type(self, *dtypes: DType | object) -> DType:
        """Resolve a common data type through the backend promotion operation."""
        native_dtypes = tuple(self.native_dtype(dtype) for dtype in dtypes)
        return self.normalize_dtype(self.call("result_type", *native_dtypes))

    def _normalize_native_dtype(self, dtype: object) -> Hashable:
        try:
            return self._dtype_normalizer(dtype)
        except (TypeError, ValueError) as exc:
            raise UnsupportedDTypeError(
                f"Array backend '{self.name}' cannot interpret dtype {dtype!r}."
            ) from exc

    def freeze(self) -> None:
        """Prevent further operation and data type registration."""
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
    "array_operation",
]
