"""
Core Type Definitions
=====================

Fundamental types and data structures used throughout the PySATL core.
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
    """
    Enumeration of distribution kinds.

    Attributes
    ----------
    DISCRETE : str
        Discrete probability distribution.
    CONTINUOUS : str
        Continuous probability distribution.
    """

    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class DistributionType:
    """
    Base class for distribution type descriptors.

    Provides a feature interface used by the characteristic registry
    to query distribution properties.
    """

    __slots__ = ()

    @property
    def registry_features(self) -> Mapping[str, Any]:
        """
        Get features used by the characteristic registry.

        Returns
        -------
        Mapping[str, Any]
            Dictionary of feature names to values.

        Notes
        -----
        Default implementation exposes public dataclass fields and
        simple attributes. Subclasses may override to provide derived
        or computed features.
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
    Distribution type for Euclidean space distributions.

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
"""Type for univariate continuous distributions."""

UnivariateDiscrete = EuclideanDistributionType(kind=Kind.DISCRETE, dimension=1)
"""Type for univariate discrete distributions."""

NumPyNumber = np.floating[Any] | np.integer[Any]
"""Type alias for NumPy numeric types."""

Number = NumPyNumber | int | float
"""Type alias for all numeric types."""

NumericArray = NDArray[NumPyNumber]
"""Type alias for numeric arrays."""

ComplexArray = NDArray[np.complexfloating[Any]]
"""Type alias for complex arrays."""

BoolArray = NDArray[np.bool_]
"""Type alias for boolean arrays."""


class ContinuousSupportShape1D(Enum):
    """
    Enumeration of 1D continuous support shapes.

    Attributes
    ----------
    REAL_LINE
        Entire real line (-∞, ∞).
    RAY_LEFT
        Left-bounded ray [a, ∞) or (a, ∞).
    RAY_RIGHT
        Right-bounded ray (-∞, b] or (-∞, b).
    BOUNDED_INTERVAL
        Bounded interval [a, b], (a, b], [a, b), or (a, b).
    EMPTY
        Empty support.
    SINGLE_POINT
        Single point {a}.
    """

    REAL_LINE = auto()
    RAY_LEFT = auto()
    RAY_RIGHT = auto()
    BOUNDED_INTERVAL = auto()
    EMPTY = auto()
    SINGLE_POINT = auto()


@dataclass(frozen=True, slots=True)
class Interval1D:
    """
    1D interval with configurable closure.

    Parameters
    ----------
    left : float, default=-inf
        Left endpoint of the interval.
    right : float, default=inf
        Right endpoint of the interval.
    left_closed : bool, default=True
        Whether left endpoint is included (ignored if left = -inf).
    right_closed : bool, default=True
        Whether right endpoint is included (ignored if right = inf).
    """

    left: float = -inf
    right: float = inf
    left_closed: bool = True
    right_closed: bool = True

    def __post_init__(self) -> None:
        """Adjust closure for infinite endpoints."""
        if self.left == -inf and self.left_closed:
            object.__setattr__(self, "left_closed", False)
        if self.right == inf and self.right_closed:
            object.__setattr__(self, "right_closed", False)

    @overload
    def contains(self, x: Number) -> bool: ...
    @overload
    def contains(self, x: NumericArray) -> BoolArray: ...

    def contains(self, x: Number | NumericArray) -> bool | BoolArray:
        """
        Check if point(s) are contained in the interval.

        Parameters
        ----------
        x : Number or NumericArray
            Point(s) to check.

        Returns
        -------
        bool or BoolArray
            True for points within the interval, False otherwise.
        """
        arr = np.asarray(x)

        left_ok = (arr > self.left) | (self.left_closed & (arr >= self.left))
        right_ok = (arr < self.right) | (self.right_closed & (arr <= self.right))
        result = left_ok & right_ok

        if np.ndim(arr) == 0:
            return bool(result)

        return cast(BoolArray, result)

    def __contains__(self, x: object) -> bool:
        """Check if a single point is in the interval."""
        return bool(self.contains(cast(Number, x)))

    @property
    def is_empty(self) -> bool:
        """Check if the interval is empty."""
        if self.left > self.right:
            return True

        return bool(self.left == self.right and not (self.left_closed and self.right_closed))

    @property
    def shape(self) -> ContinuousSupportShape1D:
        """
        Get the topological shape of the interval.

        Returns
        -------
        ContinuousSupportShape1D
            Classification of the interval's shape.
        """
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
"""Type alias for characteristic names (e.g., 'pdf', 'cdf')."""

type ParametrizationName = str
"""Type alias for parametrization names."""

ScalarFunc = Callable[[float], float]
"""Type alias for scalar functions (float -> float)."""


class CharacteristicName(StrEnum):
    """
    Enumeration of statistical distribution characteristics.

    This enumeration defines standard names for distribution functions,
    moments, and other statistical characteristics used throughout the
    PySATL-core library.

    Note
    ----------
    It also means that these characteristics are accessible via registry graph.
    It doesn't mean that user can't add their characteristics to the graph.
    """

    PDF = "pdf"
    CDF = "cdf"
    PPF = "ppf"
    PMF = "pmf"
    CF = "cf"  # unimplemented in graph yet
    SF = "sf"  # unimplemented in graph yet
    MEAN = "mean"  # unimplemented in graph yet
    VAR = "var"  # unimplemented in graph yet
    SKEW = "skewness"  # unimplemented in graph yet
    KURT = "kurtosis"  # unimplemented in graph yet
    MOMENT = "moment"  # unimplemented in graph yet
    CENTRAL_MOMENT = "central_moment"  # unimplemented in graph yet
    STANDARD_MOMENT = "standardized_moment"  # unimplemented in graph yet


class FamilyName(StrEnum):
    NORMAL = "Normal"
    CONTINUOUS_UNIFORM = "ContinuousUniform"


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
    "CharacteristicName",
    "FamilyName",
]
