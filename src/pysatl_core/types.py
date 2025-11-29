"""
Core Types
==========

Lightweight enums and dataclasses used across the distributions core.
"""

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from math import inf
from typing import Any, cast, overload

import numpy as np
from numpy.typing import NDArray


class Kind(StrEnum):
    """Distribution kind."""

    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class DistributionType:
    """
    Base class for distribution type descriptors.

    Besides acting as a marker, this class provides a small
    feature interface used by the characteristic registry.
    """

    __slots__ = ()

    @property
    def registry_features(self) -> Mapping[str, Any]:
        """
        Return a mapping of features consulted by the characteristic registry.

        Default implementation exposes public dataclass fields (if any)
        plus simple attributes.

        Subclasses may override this to provide derived or computed features.
        """
        data: dict[str, Any] = {}

        fields = getattr(self, "__dataclass_fields__", None)
        if fields is not None:
            for name in fields:
                data[name] = getattr(self, name)

        return data


@dataclass(frozen=True, slots=True)
class EuclideanDistributionType(DistributionType):
    """
    Euclidean distribution type.

    Parameters
    ----------
    kind : Kind
        Distribution kind (discrete or continuous).
    dimension : int
        Spatial dimension (e.g., 1 for univariate).
    """

    kind: Kind
    dimension: int


UnivariateContinuous = EuclideanDistributionType(kind=Kind.CONTINUOUS, dimension=1)
UnivariateDiscrete = EuclideanDistributionType(kind=Kind.DISCRETE, dimension=1)

NumPyNumber = np.floating[Any] | np.integer[Any]
Number = NumPyNumber | int | float
NumericArray = NDArray[np.floating[Any] | np.integer[Any]]
BoolArray = NDArray[np.bool_]


class ContinuousSupportShape1D(Enum):
    REAL_LINE = auto()
    RAY_LEFT = auto()
    RAY_RIGHT = auto()
    BOUNDED_INTERVAL = auto()
    EMPTY = auto()
    SINGLE_POINT = auto()


@dataclass(frozen=True, slots=True)
class Interval1D:
    left: float = -inf
    right: float = inf
    left_closed: bool = True  # Ignored if left == -inf
    right_closed: bool = True  # Ignored if right == inf

    def __post_init__(self) -> None:
        if self.left == -inf and self.left_closed:
            object.__setattr__(self, "left_closed", False)
        if self.right == inf and self.right_closed:
            object.__setattr__(self, "right_closed", False)

    @overload
    def contains(self, x: Number) -> bool: ...
    @overload
    def contains(self, x: NumericArray) -> BoolArray: ...

    def contains(self, x: Number | NumericArray) -> bool | BoolArray:
        arr = np.asarray(x)

        left_ok = (arr > self.left) | (self.left_closed & (arr >= self.left))
        right_ok = (arr < self.right) | (self.right_closed & (arr <= self.right))
        result = left_ok & right_ok

        if np.ndim(arr) == 0:
            return bool(result)

        return cast(BoolArray, result)

    def __contains__(self, x: object) -> bool:
        return bool(self.contains(cast(Number, x)))

    @property
    def is_empty(self) -> bool:
        if self.left > self.right:
            return True

        return bool(self.left == self.right and not (self.left_closed and self.right_closed))

    @property
    def shape(self) -> ContinuousSupportShape1D:
        if self.is_empty:
            return ContinuousSupportShape1D.EMPTY

        if self.left == self.right and self.left_closed and self.right_closed:
            return ContinuousSupportShape1D.SINGLE_POINT

        if self.left == -inf and self.right == inf:
            return ContinuousSupportShape1D.REAL_LINE
        if self.left == -inf and self.right < inf:
            return ContinuousSupportShape1D.RAY_LEFT
        if self.left > -inf and self.right == inf:
            return ContinuousSupportShape1D.RAY_RIGHT
        return ContinuousSupportShape1D.BOUNDED_INTERVAL


type GenericCharacteristicName = str
type ParametrizationName = str
ScalarFunc = Callable[[float], float]


__all__ = [
    "Kind",
    "EuclideanDistributionType",
    "UnivariateContinuous",
    "UnivariateDiscrete",
    "GenericCharacteristicName",
    "ParametrizationName",
    "DistributionType",
    "ScalarFunc",
    "Interval1D",
    "ContinuousSupportShape1D",
    "BoolArray",
    "NumPyNumber",
    "Number",
    "NumericArray",
]
