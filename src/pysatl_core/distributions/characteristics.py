from typing import Any, Protocol

from pysatl_core.types import (
    GenericCharacteristicName,
)

from .distribution import Distribution


class GenericCharacteristic[In, Out](Protocol):
    name: GenericCharacteristicName

    def __call__(self, distribution: Distribution, data: In, **options: Any) -> Out: ...
