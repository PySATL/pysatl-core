"""
Chebyshev-based approximation for transformed distributions.

This module provides a simple example approximator that materializes chosen
univariate continuous characteristics with Chebyshev polynomials.
The implementation intentionally focuses on the architectural integration
rather than on exhaustive numerical guarantees.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
from numpy.polynomial import Chebyshev

from pysatl_core.distributions.computation import AnalyticalComputation
from pysatl_core.distributions.support import ContinuousSupport
from pysatl_core.transformations.approximations.approximation import DistributionApproximator
from pysatl_core.transformations.distribution import ApproximatedDistribution, DerivedDistribution
from pysatl_core.types import (
    ApproximationName,
    CharacteristicName,
    GenericCharacteristicName,
    Kind,
    NumericArray,
)


class ChebyshevApproximator(DistributionApproximator):
    """
    Approximate selected characteristics with Chebyshev polynomials.

    Parameters
    ----------
    degree : int, default=32
        Polynomial degree used by the approximation.
    sample_size : int, optional
        Number of grid points used to fit the polynomial. If omitted, a value
        derived from ``degree`` is used.
    characteristics : Sequence[GenericCharacteristicName] or None, optional
        Characteristics to materialize. If omitted, all direct analytical
        computations of the source derived distribution are approximated.
    domains : Mapping[GenericCharacteristicName, tuple[float, float]] or None, optional
        Explicit approximation domains for selected characteristics. If omitted,
        the domain is inferred from the transformed distribution support for
        ``pdf`` and ``cdf`` and from ``[0, 1]`` for ``ppf``.

    Notes
    -----
    This implementation currently targets univariate continuous
    characteristics. It is meant as the first example approximator and may be
    extended later with stricter error control and domain management.
    """

    def __init__(
        self,
        *,
        degree: int = 32,
        sample_size: int | None = None,
        characteristics: Sequence[GenericCharacteristicName] | None = None,
        domains: Mapping[GenericCharacteristicName, tuple[float, float]] | None = None,
    ) -> None:
        if degree < 0:
            raise ValueError("degree must be non-negative.")

        self._degree = degree
        self._sample_size = sample_size
        self._characteristics = tuple(characteristics) if characteristics is not None else None
        self._domains = dict(domains) if domains is not None else {}

    def approximate(
        self,
        distribution: DerivedDistribution,
        **options: Any,
    ) -> ApproximatedDistribution:
        """
        Build an approximated distribution using Chebyshev polynomials.

        Parameters
        ----------
        distribution : DerivedDistribution
            Distribution to approximate.
        **options : Any
            Extra options forwarded to queried source characteristics.

        Returns
        -------
        ApproximatedDistribution
            Distribution whose analytical computations are represented by
            Chebyshev approximations.
        """
        self._validate_distribution(distribution)

        characteristics = self._select_characteristics(distribution)
        analytical_computations: dict[
            GenericCharacteristicName, AnalyticalComputation[Any, Any]
        ] = {}

        for characteristic_name in characteristics:
            domain = self._resolve_domain(distribution, characteristic_name)
            analytical_computations[characteristic_name] = self._build_computation(
                distribution=distribution,
                characteristic_name=characteristic_name,
                domain=domain,
                **options,
            )

        return ApproximatedDistribution(
            source_distribution=distribution,
            approximation_name=ApproximationName.CHEBYSHEV,
            distribution_type=distribution.distribution_type,
            bases={},
            analytical_computations=analytical_computations,
            support=distribution.support,
            sampling_strategy=distribution.sampling_strategy,
            computation_strategy=distribution.computation_strategy,
        )

    def _validate_distribution(self, distribution: DerivedDistribution) -> None:
        """
        Validate that the distribution is supported by the approximator.
        """
        distribution_type = distribution.distribution_type
        kind = getattr(distribution_type, "kind", None)
        dimension = getattr(distribution_type, "dimension", None)

        if kind != Kind.CONTINUOUS or dimension != 1:
            raise TypeError(
                "ChebyshevApproximator currently supports only univariate continuous distributions."
            )

    def _select_characteristics(
        self,
        distribution: DerivedDistribution,
    ) -> tuple[GenericCharacteristicName, ...]:
        """
        Determine which characteristics should be approximated.
        """
        if self._characteristics is not None:
            selected = self._characteristics
        else:
            selected = tuple(distribution.analytical_computations)

        if not selected:
            raise ValueError(
                "No characteristics were selected for approximation. "
                "Pass them explicitly or provide direct analytical computations."
            )

        return tuple(selected)

    def _resolve_domain(
        self,
        distribution: DerivedDistribution,
        characteristic_name: GenericCharacteristicName,
    ) -> tuple[float, float]:
        """
        Resolve the approximation domain for a characteristic.
        """
        if characteristic_name in self._domains:
            return self._domains[characteristic_name]

        if characteristic_name == CharacteristicName.PPF:
            return 0.0, 1.0

        support = distribution.support
        if (
            isinstance(support, ContinuousSupport)
            and np.isfinite(support.left)
            and np.isfinite(support.right)
        ):
            return float(support.left), float(support.right)

        raise ValueError(
            "Could not infer a finite approximation domain for the requested characteristic."
        )

    def _build_computation(
        self,
        *,
        distribution: DerivedDistribution,
        characteristic_name: GenericCharacteristicName,
        domain: tuple[float, float],
        **options: Any,
    ) -> AnalyticalComputation[Any, Any]:
        """
        Approximate a single characteristic with a Chebyshev polynomial.
        """
        left, right = domain
        if left >= right:
            raise ValueError("Approximation domain must satisfy left < right.")

        sample_size = self._sample_size or max(2 * self._degree + 1, 33)
        method = distribution.query_method(characteristic_name, **options)

        grid = np.linspace(left, right, sample_size, dtype=float)
        values = np.asarray(method(grid), dtype=float)
        if values.shape != grid.shape:
            raise ValueError(
                "ChebyshevApproximator expects one-dimensional NumPy semantics "
                "for approximated methods."
            )

        polynomial = Chebyshev.fit(grid, values, deg=self._degree, domain=[left, right])

        def _func(data: Any, /, **_: Any) -> Any:
            array = np.asarray(data, dtype=float)
            approximated = np.asarray(polynomial(array), dtype=float)

            if np.ndim(array) == 0:
                return float(approximated)

            return cast(NumericArray, approximated)

        return AnalyticalComputation(target=characteristic_name, func=_func)


__all__ = [
    "ChebyshevApproximator",
]
