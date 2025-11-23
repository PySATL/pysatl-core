from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from pysatl_core.distributions import (
    AnalyticalComputation,
    ArraySample,
    ComputationStrategy,
    DefaultComputationStrategy,
    DefaultSamplingUnivariateStrategy,
    Distribution,
    Sample,
    SamplingStrategy,
)
from pysatl_core.types import EuclideanDistributionType, GenericCharacteristicName, Kind


class MockSamplingStrategy(SamplingStrategy):
    def sample(self, n: int, distr: Distribution, **options: Any) -> Sample:
        return ArraySample(np.random.random((n, 1)))


@dataclass(slots=True)
class StandaloneEuclideanUnivariateDistribution(Distribution):
    """
    Minimal standalone univariate Euclidean distribution.

    Notes
    -----
    - Dimension is fixed to 1.
    - Default strategies are attached: computation and univariate sampling.
    """

    _distribution_type: EuclideanDistributionType
    _analytical: dict[GenericCharacteristicName, AnalyticalComputation[Any, Any]]

    def __init__(
        self,
        kind: Kind,
        analytical_computations: (
            Iterable[AnalyticalComputation[Any, Any]]
            | Mapping[GenericCharacteristicName, AnalyticalComputation[Any, Any]]
        ) = (),
    ) -> None:
        self._distribution_type = EuclideanDistributionType(kind, 1)
        if isinstance(analytical_computations, Mapping):
            self._analytical = dict(analytical_computations)
        else:
            self._analytical = {ac.target: ac for ac in analytical_computations}

    @property
    def distribution_type(self) -> EuclideanDistributionType:
        """Distribution type descriptor (kind and dimension)."""
        return self._distribution_type

    @property
    def analytical_computations(
        self,
    ) -> Mapping[GenericCharacteristicName, AnalyticalComputation[Any, Any]]:
        """Mapping from characteristic name to analytical callable."""
        return self._analytical

    @property
    def sampling_strategy(self) -> SamplingStrategy:
        """Sampling strategy instance."""
        return DefaultSamplingUnivariateStrategy()

    @property
    def computation_strategy(self) -> ComputationStrategy[Any, Any]:
        """Computation strategy instance."""
        return DefaultComputationStrategy()

    @property
    def support(self):
        return None


# ---------------------------------------------------------------------------
# Optional discrete specialization with explicit support
# ---------------------------------------------------------------------------


class StandaloneDiscreteUnivariateDistribution(StandaloneEuclideanUnivariateDistribution):
    """Discrete standalone distribution with optional `_support` for tests."""

    _support: DiscreteSupport | None

    def __init__(
        self,
        analytical_computations: (
            Iterable[AnalyticalComputation[Any, Any]]
            | Mapping[GenericCharacteristicName, AnalyticalComputation[Any, Any]]
        ) = (),
    ) -> None:
        super().__init__(kind=Kind.DISCRETE, analytical_computations=analytical_computations)
        self._support = None


# --- Discrete support helpers for tests --------------------------------------


class DiscreteSupport:
    """
    Simple discrete support to aid tests of discrete fitters.

    - Iterable over support points.
    - iter_leq(x): iterate values <= x.
    - prev(x): immediate predecessor of x (or None).
    """

    _values: list[float]

    def __init__(self, values: Iterable[float]) -> None:
        xs = sorted(float(v) for v in values)
        self._values = []
        last: float | None = None
        for v in xs:
            if last is None or v != last:
                self._values.append(v)
                last = v

    def __iter__(self) -> Iterator[float]:
        return iter(self._values)

    def iter_leq(self, x: float) -> Iterator[float]:
        for v in self._values:
            if v <= x:
                yield v

    def prev(self, x: float) -> float | None:
        p: float | None = None
        for v in self._values:
            if v < x + 1e-15:
                p = v
            else:
                break
        return p
