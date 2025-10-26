"""
Distribution Interfaces and a Standalone Univariate Implementation
==================================================================

This module defines the public :class:`Distribution` protocol and a concrete
standalone univariate Euclidean distribution:

- :class:`Distribution` protocol – abstract interface used throughout the core.
- :class:`StandaloneEuclideanUnivariateDistribution` – a minimal implementation
  that plugs default computation and sampling strategies.

Notes
-----
- The univariate sampling strategy draws from the distribution's ``ppf``.
- Log-likelihood is computed element-wise using ``pdf`` (continuous) or ``pmf``
  (discrete), assuming scalar characteristic functions (``float -> float``).
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail, Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from pysatl_core.distributions.computation import AnalyticalComputation
from pysatl_core.distributions.sampling import Sample
from pysatl_core.distributions.strategies import (
    ComputationStrategy,
    DefaultComputationStrategy,
    DefaultSamplingUnivariateStrategy,
    Method,
    SamplingStrategy,
)
from pysatl_core.types import (
    DistributionType,
    EuclideanDistributionType,
    GenericCharacteristicName,
    Kind,
)


@runtime_checkable
class Distribution(Protocol):
    """Public distribution interface used by strategies and fitters."""

    @property
    def distribution_type(self) -> DistributionType: ...

    @property
    def analytical_computations(
        self,
    ) -> Mapping[GenericCharacteristicName, AnalyticalComputation[Any, Any]]: ...

    @property
    def sampling_strategy(self) -> SamplingStrategy: ...
    @property
    def computation_strategy(self) -> ComputationStrategy[Any, Any]: ...

    def query_method(
        self, characteristic_name: GenericCharacteristicName, **options: Any
    ) -> Method[Any, Any]:
        return self.computation_strategy.query_method(characteristic_name, self, **options)

    def calculate_characteristic(
        self, characteristic_name: GenericCharacteristicName, value: Any, **options: Any
    ) -> Any:
        return self.query_method(characteristic_name, **options)(value)

    def sample(self, n: int, **options: Any) -> Sample:
        return self.sampling_strategy.sample(n, distr=self, **options)

    def log_likelihood(self, batch: Sample) -> float: ...


@dataclass(slots=True)
class StandaloneEuclideanUnivariateDistribution(Distribution):
    """
    Minimal standalone univariate Euclidean distribution.

    Parameters
    ----------
    kind : Kind
        ``Kind.CONTINUOUS`` or ``Kind.DISCRETE``.
    analytical_computations : \
        Iterable[AnalyticalComputation] or Mapping[str, AnalyticalComputation], optional
        Analytical characteristics provided by the distribution. If an iterable
        is given, items are keyed by their ``target`` attribute.

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
    ):
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

    def log_likelihood(self, batch: Sample) -> float:
        """
        Compute the log-likelihood of the given batch.

        Parameters
        ----------
        batch : Sample
            2D sample of shape ``(n, 1)``.

        Returns
        -------
        float
            Sum of ``log(pdf(x_i))`` for continuous distributions or
            ``log(pmf(x_i))`` for discrete ones.

        Notes
        -----
        Characteristic functions are assumed to be scalar (``float -> float``);
        hence values are computed element-wise.
        """
        name = "pdf" if self.distribution_type.kind == "continuous" else "pmf"
        method = self.query_method(name)
        xs = np.asarray(batch.array, dtype=np.float64).ravel()
        vals = np.fromiter((float(method(float(x))) for x in xs), dtype=np.float64, count=xs.size)
        if np.any(vals <= 0.0):
            return float("-inf")
        return float(np.sum(np.log(vals)))
