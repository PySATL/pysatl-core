from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pysatl_core.distributions.distribution import Distribution
from pysatl_core.types import (
    GenericCharacteristicName,
)


@runtime_checkable
class Computation[In, Out](Protocol):
    @property
    def target(self) -> GenericCharacteristicName: ...
    def __call__(self, data: In) -> Out: ...


@runtime_checkable
class FittedComputationMethodProtocol[In, Out](Protocol):
    @property
    def target(self) -> GenericCharacteristicName: ...
    @property
    def sources(self) -> Sequence[GenericCharacteristicName]: ...
    def __call__(self, data: In) -> Out: ...


@runtime_checkable
class ComputationMethodProtocol[In, Out](Protocol):
    @property
    def target(self) -> GenericCharacteristicName: ...
    @property
    def sources(self) -> Sequence[GenericCharacteristicName]: ...
    def fit(self, distribution: Distribution) -> FittedComputationMethodProtocol[In, Out]: ...


@dataclass(frozen=True, slots=True)
class AnalyticalComputation[In, Out]:
    target: GenericCharacteristicName
    func: Callable[[In], Out]

    def __call__(self, data: In) -> Out:
        return self.func(data)


@dataclass(frozen=True, slots=True)
class FittedComputationMethod[In, Out]:
    target: GenericCharacteristicName
    sources: Sequence[GenericCharacteristicName]
    func: Callable[[In], Out]

    def __call__(self, data: In) -> Out:
        return self.func(data)


@dataclass(frozen=True, slots=True)
class ComputationMethod[In, Out]:
    target: GenericCharacteristicName
    sources: Sequence[GenericCharacteristicName]
    fitter: Callable[[Distribution], FittedComputationMethod[In, Out]]

    def fit(self, distribution: Distribution) -> FittedComputationMethod[In, Out]:
        return self.fitter(distribution)
