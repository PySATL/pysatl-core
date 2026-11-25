"""Backend-independent array creation operations."""

from __future__ import annotations

from typing import Any, cast

from pysatl_core.arrays._dispatch import dispatch
from pysatl_core.arrays.dtypes import DType
from pysatl_core.arrays.types import NativeArray


def scalar(value: object, dtype: DType | object | None = None) -> Any:
    """Create a native scalar."""
    return dispatch("scalar", value, dtype=dtype)


def asarray(value: object, dtype: DType | object | None = None) -> NativeArray:
    """Create a native array."""
    return cast(NativeArray, dispatch("asarray", value, dtype=dtype))


def astype(value: object, dtype: DType | object) -> NativeArray:
    """Convert an array to another data type."""
    return cast(NativeArray, dispatch("astype", value, dtype))


def copy(value: object) -> NativeArray:
    """Copy an array."""
    return cast(NativeArray, dispatch("copy", value))


def empty(shape: object, dtype: DType | object | None = None) -> NativeArray:
    """Create an uninitialized array."""
    return cast(NativeArray, dispatch("empty", shape, dtype=dtype))


def zeros(shape: object, dtype: DType | object | None = None) -> NativeArray:
    """Create an array filled with zeros."""
    return cast(NativeArray, dispatch("zeros", shape, dtype=dtype))


def empty_like(value: object, dtype: DType | object | None = None) -> NativeArray:
    """Create an uninitialized array matching another value."""
    return cast(NativeArray, dispatch("empty_like", value, dtype=dtype))


def zeros_like(value: object, dtype: DType | object | None = None) -> NativeArray:
    """Create a zero array matching another value."""
    return cast(NativeArray, dispatch("zeros_like", value, dtype=dtype))


def full_like(
    value: object,
    fill_value: object,
    dtype: DType | object | None = None,
) -> NativeArray:
    """Create a filled array matching another value."""
    return cast(NativeArray, dispatch("full_like", value, fill_value, dtype=dtype))


def arange(*args: object, dtype: DType | object | None = None) -> NativeArray:
    """Create evenly spaced values in an interval."""
    return cast(NativeArray, dispatch("arange", *args, dtype=dtype))


def linspace(
    start: object,
    stop: object,
    num: int = 50,
    *,
    dtype: DType | object | None = None,
) -> NativeArray:
    """Create evenly spaced values over a closed interval."""
    return cast(NativeArray, dispatch("linspace", start, stop, num, dtype=dtype))


def pi(dtype: DType | object | None = None) -> Any:
    """Return pi as a native scalar."""
    return dispatch("pi", dtype=dtype)


def nan(dtype: DType | object | None = None) -> Any:
    """Return NaN as a native scalar."""
    return dispatch("nan", dtype=dtype)


def inf(dtype: DType | object | None = None) -> Any:
    """Return infinity as a native scalar."""
    return dispatch("inf", dtype=dtype)


__all__ = [
    "arange",
    "asarray",
    "astype",
    "copy",
    "empty",
    "empty_like",
    "full_like",
    "inf",
    "linspace",
    "nan",
    "pi",
    "scalar",
    "zeros",
    "zeros_like",
]
