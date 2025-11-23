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
from enum import StrEnum
from typing import Any


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
