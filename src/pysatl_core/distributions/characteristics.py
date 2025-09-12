from typing import TYPE_CHECKING, Any, Protocol, cast

from pysatl_core.distributions.strategies import Method
from pysatl_core.types import (
    GenericCharacteristicName,
)

if TYPE_CHECKING:
    from pysatl_core.distributions.distribution import Distribution


class GenericCharacteristic[In, Out](Protocol):
    name: GenericCharacteristicName

    # NOTICE: options контролирует математическую формулу характеристики
    def __call__(self, distribution: "Distribution", data: In, **options: Any) -> Out: ...


class pdf[In, Out](GenericCharacteristic[In, Out]):
    name: GenericCharacteristicName = GenericCharacteristicName("pdf")

    def __call__(self, distribution: "Distribution", data: In, **options: Any) -> Out:
        method = cast(
            Method[In, Out],
            distribution.computation_strategy.query_method(self.name, distribution),
        )
        return method(data)


class cdf[In, Out](GenericCharacteristic[In, Out]):
    name: GenericCharacteristicName = GenericCharacteristicName("cdf")

    def __call__(self, distribution: "Distribution", data: In, **options: Any) -> Out:
        method = cast(
            Method[In, Out],
            distribution.computation_strategy.query_method(self.name, distribution),
        )
        return method(data)


class ppf[In, Out](GenericCharacteristic[In, Out]):
    name: GenericCharacteristicName = GenericCharacteristicName("ppf")

    def __call__(self, distribution: "Distribution", data: In, **options: Any) -> Out:
        method = cast(
            Method[In, Out],
            distribution.computation_strategy.query_method(self.name, distribution),
        )
        return method(data)
