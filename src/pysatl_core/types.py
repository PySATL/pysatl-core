"""
Core Types
==========

Lightweight enums and dataclasses used across the distributions core.
"""

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


class Kind(StrEnum):
    """Distribution kind."""

    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class DistributionType:
    """Marker base for distribution type descriptors."""

    __slots__ = ()


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
]
