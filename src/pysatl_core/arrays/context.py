"""Process-wide settings, data types and capabilities for array computations."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from pysatl_core.arrays.backend import (
    UnsupportedArrayOperationError,
    UnsupportedDTypeError,
)
from pysatl_core.arrays.devices import Device
from pysatl_core.arrays.dtypes import DType

if TYPE_CHECKING:
    from pysatl_core.arrays.backend import ArrayBackend
    from pysatl_core.arrays.runtime import Backend


@dataclass(frozen=True, slots=True)
class ArrayContext:
    """Process settings and an immutable mapping between shared and native types."""

    backend: Backend
    device: Device = Device.CPU
    native_dtypes: Mapping[DType, Hashable] = field(default_factory=dict, repr=False, hash=False)
    default_dtype: DType = field(default=DType.FLOAT64, init=False)
    _dtypes: Mapping[Hashable, DType] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.device, Device):
            raise TypeError("device must be a Device enum member.")

        native_dtypes: dict[DType, Hashable] = {}
        dtypes: dict[Hashable, DType] = {}

        for dtype, native_dtype in self.native_dtypes.items():
            normalized = self._normalize_native_dtype(native_dtype)
            if normalized in dtypes and dtypes[normalized] != dtype:
                raise ValueError(
                    f"Native dtype {native_dtype!r} is already mapped to another DType."
                )

            native_dtypes[dtype] = normalized
            dtypes[normalized] = dtype

        object.__setattr__(self, "native_dtypes", MappingProxyType(native_dtypes))
        object.__setattr__(self, "_dtypes", MappingProxyType(dtypes))

    @property
    def arrays(self) -> ArrayBackend:
        """Return the active array operation registry."""
        return self.backend.arrays

    @property
    def supported_dtypes(self) -> frozenset[DType]:
        """Return the shared data types described by this context."""
        return frozenset(self.native_dtypes)

    def normalize_dtype(self, dtype: object) -> DType:
        """Convert a shared or native data type to a supported shared type."""
        if isinstance(dtype, DType):
            if dtype in self.native_dtypes:
                return dtype

            raise UnsupportedDTypeError(
                f"Array backend '{self.backend.identity}' does not support dtype {dtype!r}."
            )
        normalized = self._normalize_native_dtype(dtype)

        try:
            return self._dtypes[normalized]
        except KeyError as exc:
            raise UnsupportedDTypeError(
                f"Array backend '{self.backend.identity}' does not support dtype {dtype!r}."
            ) from exc

    def native_dtype(self, dtype: object) -> Hashable:
        """Return the native representation of a supported data type."""
        return self.native_dtypes[self.normalize_dtype(dtype)]

    def _normalize_native_dtype(self, dtype: object) -> Hashable:
        try:
            normalized = self.arrays.call("normalize_native_dtype", dtype)
            if not isinstance(normalized, Hashable):
                raise TypeError("The normalized data type must be hashable.")

            return normalized
        except (TypeError, ValueError) as exc:
            raise UnsupportedDTypeError(
                f"Array backend '{self.backend.identity}' cannot interpret dtype {dtype!r}."
            ) from exc

    def result_type(self, *dtypes: object) -> DType:
        """Resolve a common data type using native backend promotion rules."""
        if not dtypes:
            return self.default_dtype

        native_dtypes = tuple(self.native_dtype(dtype) for dtype in dtypes)
        return self.normalize_dtype(self.arrays.call("result_type", *native_dtypes))

    def supports(
        self,
        operations: Iterable[str] = (),
        dtypes: Iterable[object] = (),
    ) -> bool:
        """Return whether all requested operations and data types are supported."""
        if not self.arrays.supports(operations):
            return False

        try:
            for dtype in dtypes:
                self.native_dtype(dtype)
        except UnsupportedDTypeError:
            return False

        return True

    def require(
        self,
        operations: Iterable[str] = (),
        dtypes: Iterable[object] = (),
    ) -> None:
        """Raise when an operation or data type is unavailable."""
        missing = self.arrays.missing_operations(operations)
        if missing:
            raise UnsupportedArrayOperationError(
                f"Array backend '{self.backend.identity}' does not support operations "
                f"{sorted(missing)}."
            )

        for dtype in dtypes:
            self.native_dtype(dtype)


__all__ = ["ArrayContext"]
