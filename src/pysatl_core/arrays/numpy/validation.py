"""NumPy native value inspection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pysatl_core.arrays.backend import array_operation
from pysatl_core.arrays.devices import Device
from pysatl_core.arrays.numpy._backend import numpy_array_backend
from pysatl_core.arrays.profiles import ValueKind, ValueProfile

if TYPE_CHECKING:
    from pysatl_core.arrays.context import ArrayContext


@array_operation(numpy_array_backend, "supports_device")
def _supports_device(device: Device) -> bool:
    return device is Device.CPU


@array_operation(numpy_array_backend, "is_native")
def _is_native(value: object) -> bool:
    return isinstance(value, np.ndarray | np.generic)


@array_operation(numpy_array_backend, "inspect")
def _inspect(value: object, *, context: ArrayContext) -> ValueProfile:
    if isinstance(value, np.ndarray):
        return ValueProfile(
            kind=ValueKind.SCALAR if value.ndim == 0 else ValueKind.ARRAY,
            dtype=context.normalize_dtype(value.dtype),
            device=Device.CPU,
            shape=tuple(value.shape),
            ndim=value.ndim,
            size=value.size,
        )
    if isinstance(value, np.generic):
        return ValueProfile(
            kind=ValueKind.SCALAR,
            dtype=context.normalize_dtype(value.dtype),
            device=Device.CPU,
            shape=(),
            ndim=0,
            size=1,
        )
    raise TypeError(f"Unsupported NumPy value type: {type(value).__name__!r}.")


__all__: list[str] = []
