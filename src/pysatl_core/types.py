from enum import StrEnum
from typing import NewType


class Kind(StrEnum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


GenericCharacteristicName = NewType("GenericCharacteristicName", str)
# Возможно тут надо делить не по числу, а на одномерные/многомерные, но непонятно как сэмплить
type Dimension = int

__all__ = ["Kind", "Dimension", "GenericCharacteristicName"]
