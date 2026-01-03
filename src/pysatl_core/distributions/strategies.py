"""
Computation and Sampling Strategies

This module defines strategies for computing distribution characteristics
and generating random samples.
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, Protocol

import numpy as np

from pysatl_core.distributions.registry import characteristic_registry
from pysatl_core.distributions.sampling import ArraySample

if TYPE_CHECKING:
    from typing import Any

    from pysatl_core.distributions.computation import AnalyticalComputation, FittedComputationMethod
    from pysatl_core.distributions.distribution import Distribution
    from pysatl_core.distributions.sampling import Sample
    from pysatl_core.types import GenericCharacteristicName

type Method[In, Out] = AnalyticalComputation[In, Out] | FittedComputationMethod[In, Out]


class ComputationStrategy[In, Out](Protocol):
    """
    Protocol for strategies that resolve computation methods for characteristics.

    Attributes
    ----------
    enable_caching : bool
        Whether to cache fitted computation methods.
    """

    enable_caching: bool

    def query_method(
        self, state: GenericCharacteristicName, distr: Distribution, **options: Any
    ) -> Method[In, Out]: ...


class DefaultComputationStrategy[In, Out]:
    """
    Default strategy for resolving characteristic computation methods.

    This strategy first checks for analytical implementations provided by
    the distribution. If none exists, it walks the characteristic graph
    to find a conversion path from an analytical characteristic to the
    target characteristic.

    Parameters
    ----------
    enable_caching : bool, default=False
        If True, cache fitted conversions to avoid repeated fitting.

    Attributes
    ----------
    enable_caching : bool
        Whether caching is enabled.
    _cache : dict[str, FittedComputationMethod]
        Cache of fitted computation methods.
    _resolving : dict[int, set[str]]
        Tracking of currently resolving characteristics to detect cycles.
    """

    def __init__(self, enable_caching: bool = False) -> None:
        self.enable_caching = enable_caching
        self._cache: dict[GenericCharacteristicName, FittedComputationMethod[In, Out]] = {}
        self._resolving: dict[int, set[GenericCharacteristicName]] = {}

    def _push_guard(self, distr: Distribution, state: GenericCharacteristicName) -> None:
        """
        Push a characteristic onto the resolution stack to detect cycles.

        Raises
        ------
        RuntimeError
            If a cycle is detected during resolution.
        """
        key = id(distr)
        seen = self._resolving.setdefault(key, set())
        if state in seen:
            raise RuntimeError(
                f"Cycle detected while resolving '{state}'. "
                "Provide at least one analytical base characteristic in the distribution."
            )
        seen.add(state)

    def _pop_guard(self, distr: Distribution, state: GenericCharacteristicName) -> None:
        """Pop a characteristic from the resolution stack."""
        key = id(distr)
        seen = self._resolving.get(key)
        if seen is not None:
            seen.discard(state)
            if not seen:
                self._resolving.pop(key, None)

    def query_method(
        self, state: GenericCharacteristicName, distr: Distribution, **options: Any
    ) -> Method[In, Out]:
        """
        Resolve a computation method for the target characteristic.

        Resolution order:
        1. Analytical implementation from the distribution
        2. Cached fitted method (if caching enabled)
        3. Conversion path from an analytical characteristic via the graph

        Parameters
        ----------
        state : str
            Target characteristic name (e.g., "pdf", "cdf").
        distr : Distribution
            Distribution to compute the characteristic for.
        **options : Any
            Additional options passed to fitters.

        Returns
        -------
        Method
            Callable that computes the characteristic.

        Raises
        ------
        RuntimeError
            If no analytical base exists, no conversion path is found,
            or a cycle is detected.
        """
        # 1. Check for analytical implementation
        if state in distr.analytical_computations:
            return distr.analytical_computations[state]

        # 2. Check cache if enabled
        if self.enable_caching:
            cached = self._cache.get(state)
            if cached is not None:
                return cached

        # 3. Require at least one analytical characteristic
        if not distr.analytical_computations:
            raise RuntimeError(
                "Distribution provides no analytical computations to ground conversions."
            )

        # 4. Get filtered graph view for this distribution
        reg = characteristic_registry().view(distr)

        self._push_guard(distr, state)
        try:
            # 5. Try each analytical characteristic as a source
            for src in distr.analytical_computations:
                if src == state:
                    return distr.analytical_computations[src]

                # Find conversion path in the graph
                path = reg.find_path(src, state)
                if not path:
                    continue

                # Fit each edge along the path
                last_fitted: FittedComputationMethod[In, Out] | None = None
                for edge in path:
                    fitted = edge.fit(distr, **options)
                    if self.enable_caching:
                        self._cache[edge.target] = fitted
                    last_fitted = fitted

                if last_fitted is None:
                    raise RuntimeError(f"Empty path when resolving '{state}' from '{src}'.")
                return last_fitted

            raise RuntimeError(
                f"No conversion path from any analytical characteristic to '{state}'."
            )
        finally:
            self._pop_guard(distr, state)


class SamplingStrategy(Protocol):
    """Protocol for strategies that generate samples from distributions."""

    def sample(self, n: int, distr: Distribution, **options: Any) -> Sample: ...


class DefaultSamplingUnivariateStrategy(SamplingStrategy):
    """
    Default univariate sampler using inverse transform sampling.

    This strategy generates samples by applying the PPF (inverse CDF)
    to uniform random variables.

    Notes
    -----
    - Requires the distribution to provide a PPF computation method.
    - Returns samples as a 2D array of shape (n, 1).
    """

    def sample(self, n: int, distr: Distribution, **options: Any) -> ArraySample:
        """
        Generate n samples from the distribution.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        distr : Distribution
            Distribution to sample from.
        **options : Any
            Additional options passed to the PPF computation.

        Returns
        -------
        ArraySample
            Samples as a 2D array of shape (n, 1).
        """
        ppf = distr.query_method("ppf", **options)
        rng = np.random.default_rng()
        U = rng.random(n)
        vals = np.array([ppf(Ui) for Ui in U], dtype=np.float64).reshape(n, 1)
        return ArraySample(vals)
