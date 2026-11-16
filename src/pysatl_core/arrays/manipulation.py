"""Backend-independent shape and indexing operations."""

from __future__ import annotations

from typing import Any, cast

from pysatl_core.arrays._dispatch import dispatch
from pysatl_core.arrays.types import NativeArray


def atleast_1d(value: object) -> NativeArray:
    return cast(NativeArray, dispatch("atleast_1d", value))


def reshape(value: object, shape: object) -> NativeArray:
    return cast(NativeArray, dispatch("reshape", value, shape))


def flatten(value: object) -> NativeArray:
    return cast(NativeArray, dispatch("flatten", value))


def expand_dims(value: object, axis: int) -> NativeArray:
    return cast(NativeArray, dispatch("expand_dims", value, axis))


def stack(values: object, axis: int = 0) -> NativeArray:
    return cast(NativeArray, dispatch("stack", values, axis=axis))


def concatenate(values: object, axis: int = 0) -> NativeArray:
    return cast(NativeArray, dispatch("concatenate", values, axis=axis))


def getitem(value: object, item: object) -> Any:
    return dispatch("getitem", value, item)


def setitem(value: object, item: object, update: object) -> NativeArray:
    return cast(NativeArray, dispatch("setitem", value, item, update))


def take(value: object, indices: object, axis: int | None = None) -> Any:
    return dispatch("take", value, indices, axis=axis)


def nonzero(value: object) -> Any:
    return dispatch("nonzero", value)


def unique(value: object) -> NativeArray:
    return cast(NativeArray, dispatch("unique", value))


def scatter_add(value: object, indices: object, updates: object) -> NativeArray:
    return cast(NativeArray, dispatch("scatter_add", value, indices, updates))


def argsort(value: object, axis: int = -1) -> NativeArray:
    return cast(NativeArray, dispatch("argsort", value, axis=axis))


def searchsorted(value: object, query: object, side: str = "left") -> Any:
    return dispatch("searchsorted", value, query, side=side)


__all__ = [
    "argsort",
    "atleast_1d",
    "concatenate",
    "expand_dims",
    "flatten",
    "getitem",
    "nonzero",
    "reshape",
    "scatter_add",
    "searchsorted",
    "setitem",
    "stack",
    "take",
    "unique",
]
