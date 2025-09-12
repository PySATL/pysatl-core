from dataclasses import dataclass
from enum import StrEnum
from typing import NewType


class Kind(StrEnum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class DistributionType:
    __slots__ = ()


@dataclass(frozen=True, slots=True)
class EuclidianDistributionType(DistributionType):
    kind: Kind
    dimension: int


GenericCharacteristicName = NewType("GenericCharacteristicName", str)


__all__ = ["Kind", "EuclidianDistributionType", "GenericCharacteristicName", "DistributionType"]
