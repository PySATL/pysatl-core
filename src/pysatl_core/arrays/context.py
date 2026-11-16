"""Process-wide settings and capabilities for array computations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pysatl_core.arrays.backend import (
    UnsupportedArrayOperationError,
    UnsupportedDTypeError,
)
from pysatl_core.arrays.dtypes import DType

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pysatl_core.arrays.backend import ArrayBackend
    from pysatl_core.arrays.runtime import Backend


@dataclass(frozen=True, slots=True)
class ArrayContext:
    """Settings and capabilities shared by all array operations in the process."""

    backend: Backend
    device: str = "cpu"
    default_dtype: DType = field(default=DType.FLOAT64, init=False)

    @property
    def arrays(self) -> ArrayBackend:
        """Return the active array operation registry."""
        return self.backend.arrays

    @property
    def supported_dtypes(self) -> frozenset[DType]:
        """Return data types supported by the active array backend."""
        return self.arrays.supported_dtypes

    def result_type(self, *dtypes: DType | object) -> DType:
        """Resolve a common data type using native backend promotion rules."""
        if not dtypes:
            return self.default_dtype
        return self.arrays.result_type(*dtypes)

    def supports(
        self,
        operations: Iterable[str] = (),
        dtypes: Iterable[DType | object] = (),
    ) -> bool:
        """Return whether all requested operations and data types are supported."""
        if not self.arrays.supports(operations):
            return False
        try:
            for dtype in dtypes:
                self.arrays.native_dtype(dtype)
        except UnsupportedDTypeError:
            return False
        return True

    def require(
        self,
        operations: Iterable[str] = (),
        dtypes: Iterable[DType | object] = (),
    ) -> None:
        """Raise when an operation or data type is unavailable."""
        missing = self.arrays.missing_operations(operations)
        if missing:
            raise UnsupportedArrayOperationError(
                f"Array backend '{self.backend.identity}' does not support operations "
                f"{sorted(missing)}."
            )
        for dtype in dtypes:
            self.arrays.native_dtype(dtype)


__all__ = ["ArrayContext"]
