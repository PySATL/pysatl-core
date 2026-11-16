"""Backend-independent reduction operations."""

from __future__ import annotations

from typing import Any

from pysatl_core.arrays._dispatch import dispatch


def any(value: object, axis: object = None, keepdims: bool = False) -> Any:
    return dispatch("any", value, axis=axis, keepdims=keepdims)


def all(value: object, axis: object = None, keepdims: bool = False) -> Any:
    return dispatch("all", value, axis=axis, keepdims=keepdims)


def sum(value: object, axis: object = None, keepdims: bool = False) -> Any:
    return dispatch("sum", value, axis=axis, keepdims=keepdims)


def min(value: object, axis: object = None, keepdims: bool = False) -> Any:
    return dispatch("min", value, axis=axis, keepdims=keepdims)


def max(value: object, axis: object = None, keepdims: bool = False) -> Any:
    return dispatch("max", value, axis=axis, keepdims=keepdims)


def argmax(value: object, axis: int | None = None) -> Any:
    return dispatch("argmax", value, axis=axis)


def cumsum(value: object, axis: int | None = None) -> Any:
    return dispatch("cumsum", value, axis=axis)


def cummax(value: object, axis: int = 0) -> Any:
    return dispatch("cummax", value, axis=axis)


def diff(value: object, n: int = 1, axis: int = -1) -> Any:
    return dispatch("diff", value, n=n, axis=axis)


__all__ = [
    "all",
    "any",
    "argmax",
    "cummax",
    "cumsum",
    "diff",
    "max",
    "min",
    "sum",
]
