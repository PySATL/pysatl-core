"""Backend-independent element-wise operations."""

from __future__ import annotations

from typing import Any

from pysatl_core.arrays._dispatch import dispatch


def add(left: object, right: object) -> Any:
    return dispatch("add", left, right)


def subtract(left: object, right: object) -> Any:
    return dispatch("subtract", left, right)


def multiply(left: object, right: object) -> Any:
    return dispatch("multiply", left, right)


def divide(left: object, right: object) -> Any:
    return dispatch("divide", left, right)


def floor_divide(left: object, right: object) -> Any:
    return dispatch("floor_divide", left, right)


def remainder(left: object, right: object) -> Any:
    return dispatch("remainder", left, right)


def power(left: object, right: object) -> Any:
    return dispatch("power", left, right)


def negative(value: object) -> Any:
    return dispatch("negative", value)


def abs(value: object) -> Any:
    return dispatch("abs", value)


def exp(value: object) -> Any:
    return dispatch("exp", value)


def log(value: object) -> Any:
    return dispatch("log", value)


def log1p(value: object) -> Any:
    return dispatch("log1p", value)


def sqrt(value: object) -> Any:
    return dispatch("sqrt", value)


def sinc(value: object) -> Any:
    return dispatch("sinc", value)


def floor(value: object) -> Any:
    return dispatch("floor", value)


def round(value: object, decimals: int = 0) -> Any:
    return dispatch("round", value, decimals)


def nextafter(value: object, target: object) -> Any:
    return dispatch("nextafter", value, target)


def maximum(left: object, right: object) -> Any:
    return dispatch("maximum", left, right)


def minimum(left: object, right: object) -> Any:
    return dispatch("minimum", left, right)


def clip(value: object, minimum: object, maximum: object) -> Any:
    return dispatch("clip", value, minimum, maximum)


def where(condition: object, left: object, right: object) -> Any:
    return dispatch("where", condition, left, right)


def equal(left: object, right: object) -> Any:
    return dispatch("equal", left, right)


def not_equal(left: object, right: object) -> Any:
    return dispatch("not_equal", left, right)


def less(left: object, right: object) -> Any:
    return dispatch("less", left, right)


def less_equal(left: object, right: object) -> Any:
    return dispatch("less_equal", left, right)


def greater(left: object, right: object) -> Any:
    return dispatch("greater", left, right)


def greater_equal(left: object, right: object) -> Any:
    return dispatch("greater_equal", left, right)


def logical_and(left: object, right: object) -> Any:
    return dispatch("logical_and", left, right)


def logical_or(left: object, right: object) -> Any:
    return dispatch("logical_or", left, right)


def logical_not(value: object) -> Any:
    return dispatch("logical_not", value)


def isfinite(value: object) -> Any:
    return dispatch("isfinite", value)


def isnan(value: object) -> Any:
    return dispatch("isnan", value)


def isinf(value: object) -> Any:
    return dispatch("isinf", value)


def isclose(
    left: object,
    right: object,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> Any:
    return dispatch(
        "isclose",
        left,
        right,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    )


def nan_to_num(
    value: object,
    *,
    nan: object = 0.0,
    posinf: object | None = None,
    neginf: object | None = None,
) -> Any:
    return dispatch("nan_to_num", value, nan=nan, posinf=posinf, neginf=neginf)


def real(value: object) -> Any:
    return dispatch("real", value)


def imag(value: object) -> Any:
    return dispatch("imag", value)


def conjugate(value: object) -> Any:
    return dispatch("conjugate", value)


__all__ = [
    "abs",
    "add",
    "clip",
    "conjugate",
    "divide",
    "equal",
    "exp",
    "floor",
    "floor_divide",
    "greater",
    "greater_equal",
    "imag",
    "isclose",
    "isfinite",
    "isinf",
    "isnan",
    "less",
    "less_equal",
    "log",
    "log1p",
    "logical_and",
    "logical_not",
    "logical_or",
    "maximum",
    "minimum",
    "multiply",
    "nan_to_num",
    "negative",
    "nextafter",
    "not_equal",
    "power",
    "real",
    "remainder",
    "round",
    "sinc",
    "sqrt",
    "subtract",
    "where",
]
