"""NumPy creation operations."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from pysatl_core.arrays.backend import array_operation
from pysatl_core.arrays.dtypes import DType
from pysatl_core.arrays.numpy._backend import numpy_array_backend
from pysatl_core.arrays.state import resolve_array_context


def _dtype(dtype: DType | object | None, *, default: bool = False) -> Any:
    if dtype is None:
        dtype = resolve_array_context().default_dtype if default else None

    if dtype is None:
        return None

    return resolve_array_context().native_dtype(dtype)


def _default_dtype() -> DType:
    return resolve_array_context().default_dtype


@array_operation(numpy_array_backend)
def scalar(value: object, dtype: DType | object | None = None) -> np.generic:
    """Create a NumPy scalar."""
    result = np.asarray(value, dtype=_dtype(dtype))

    if result.ndim != 0:
        raise ValueError("scalar() requires a zero-dimensional value.")

    return cast(np.generic, result[()])


@array_operation(numpy_array_backend)
def asarray(value: object, dtype: DType | object | None = None) -> np.ndarray[Any, Any]:
    """Create a NumPy array."""
    return np.asarray(value, dtype=_dtype(dtype))


@array_operation(numpy_array_backend)
def astype(value: np.ndarray[Any, Any], dtype: DType | object) -> np.ndarray[Any, Any]:
    """Convert an array to another data type."""
    return value.astype(_dtype(dtype))


@array_operation(numpy_array_backend)
def copy(value: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Copy an array."""
    return value.copy()


@array_operation(numpy_array_backend)
def empty(shape: Any, dtype: DType | object | None = None) -> np.ndarray[Any, Any]:
    """Create an uninitialized array."""
    return np.empty(shape, dtype=_dtype(dtype, default=True))


@array_operation(numpy_array_backend)
def zeros(shape: Any, dtype: DType | object | None = None) -> np.ndarray[Any, Any]:
    """Create an array filled with zeros."""
    return np.zeros(shape, dtype=_dtype(dtype, default=True))


@array_operation(numpy_array_backend)
def empty_like(
    value: np.ndarray[Any, Any], dtype: DType | object | None = None
) -> np.ndarray[Any, Any]:
    """Create an uninitialized array matching another array."""
    return np.empty_like(value, dtype=_dtype(dtype))


@array_operation(numpy_array_backend)
def zeros_like(
    value: np.ndarray[Any, Any], dtype: DType | object | None = None
) -> np.ndarray[Any, Any]:
    """Create a zero array matching another array."""
    return np.zeros_like(value, dtype=_dtype(dtype))


@array_operation(numpy_array_backend)
def full_like(
    value: np.ndarray[Any, Any], fill_value: object, dtype: DType | object | None = None
) -> np.ndarray[Any, Any]:
    """Create a filled array matching another array."""
    return np.full_like(value, fill_value, dtype=_dtype(dtype))


@array_operation(numpy_array_backend)
def arange(*args: Any, dtype: DType | object | None = None) -> np.ndarray[Any, Any]:
    """Create evenly spaced values in an interval."""
    return np.arange(*args, dtype=_dtype(dtype))


@array_operation(numpy_array_backend)
def linspace(
    start: Any,
    stop: Any,
    num: int = 50,
    *,
    dtype: DType | object | None = None,
) -> np.ndarray[Any, Any]:
    """Create evenly spaced values over a closed interval."""
    return np.linspace(start, stop, num=num, dtype=_dtype(dtype))


@array_operation(numpy_array_backend)
def pi(dtype: DType | object | None = None) -> np.generic:
    """Return pi in a backend data type."""
    return scalar(np.pi, dtype or _default_dtype())


@array_operation(numpy_array_backend)
def nan(dtype: DType | object | None = None) -> np.generic:
    """Return NaN in a backend data type."""
    return scalar(np.nan, dtype or _default_dtype())


@array_operation(numpy_array_backend)
def inf(dtype: DType | object | None = None) -> np.generic:
    """Return infinity in a backend data type."""
    return scalar(np.inf, dtype or _default_dtype())


__all__: list[str] = []
