"""
Computation and Sampling Strategies
===================================

This module defines the pluggable strategy interfaces and default implementations:

- :class:`ComputationStrategy` — resolves characteristic methods.
- :class:`DefaultComputationStrategy` — resolves analyticals, caches fitted
  conversions (optional), and walks the characteristic graph on demand.
- :class:`SamplingStrategy` — draws samples from a distribution.
- :class:`DefaultSamplingUnivariateStrategy` — draws ``(n, 1)`` samples using
  ``ppf`` and i.i.d. uniform variates.

Notes
-----
- Strategies are intentionally lightweight and stateless by default.
- The default computation strategy can optionally cache fitted conversions.
"""

__author__ = "Leonid Elkin, Mikhail, Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from pysatl_core.distributions.computation import (
    AnalyticalComputation,
    FittedComputationMethod,
)
from pysatl_core.types import (
    GenericCharacteristicName,
)

from .registry import distribution_type_register
from .sampling import ArraySample, Sample

if TYPE_CHECKING:
    from .distribution import Distribution

type Method[In, Out] = AnalyticalComputation[In, Out] | FittedComputationMethod[In, Out]


class ComputationStrategy[In, Out](Protocol):
    """Protocol for characteristic resolution strategies."""

    enable_caching: bool

    def query_method(
        self, state: GenericCharacteristicName, distr: "Distribution", **options: Any
    ) -> Method[In, Out]: ...


class DefaultComputationStrategy[In, Out]:
    """
    Default characteristic resolver.

    Resolution order
    ----------------
    1. If the distribution provides an analytical implementation, return it.
    2. Else, if caching is enabled and the method is cached, return it.
    3. Else:
       a) Get the graph for the distribution type,
       b) choose any analytical characteristic as a source,
       c) find a path from the source to the target,
       d) fit the edges along the path (the fitter may recursively resolve
          dependencies via the strategy).

    Parameters
    ----------
    enable_caching : bool, default False
        If ``True``, cache fitted conversions keyed by target characteristic.

    Raises
    ------
    RuntimeError
        If the distribution has no analytical base, or no conversion path exists,
        or a cycle is detected during resolution.
    """

    def __init__(self, enable_caching: bool = False) -> None:
        self.enable_caching = enable_caching
        self._cache: dict[GenericCharacteristicName, FittedComputationMethod[In, Out]] = {}
        self._resolving: dict[int, set[GenericCharacteristicName]] = {}

    def _push_guard(self, distr: "Distribution", state: GenericCharacteristicName) -> None:
        key = id(distr)
        seen = self._resolving.setdefault(key, set())
        if state in seen:
            raise RuntimeError(
                f"Cycle detected while resolving '{state}'. "
                "Provide at least one analytical base characteristic in the distribution."
            )
        seen.add(state)

    def _pop_guard(self, distr: "Distribution", state: GenericCharacteristicName) -> None:
        key = id(distr)
        seen = self._resolving.get(key)
        if seen is not None:
            seen.discard(state)
            if not seen:
                self._resolving.pop(key, None)

    def query_method(
        self, state: GenericCharacteristicName, distr: "Distribution", **options: Any
    ) -> Method[In, Out]:
        """
        Resolve an analytical or fitted method for ``state``.

        Parameters
        ----------
        state : str
            Target characteristic name to resolve.
        distr : Distribution
            The distribution providing the analytical base and type.
        **options
            Passed to the fitter(s) when conversions are required.

        Returns
        -------
        Method
            Analytical or fitted callable implementing ``state``.
        """
        if state in distr.analytical_computations:
            return distr.analytical_computations[state]

        if self.enable_caching:
            cached = self._cache.get(state)
            if cached is not None:
                return cached

        if not distr.analytical_computations:
            raise RuntimeError(
                "Distribution provides no analytical computations to ground conversions."
            )

        reg = distribution_type_register().get(distr.distribution_type)

        self._push_guard(distr, state)
        try:
            for src in distr.analytical_computations:
                if src == state:
                    return distr.analytical_computations[src]  # на всякий случай

                path = reg.find_path(src, state)
                if not path:
                    continue

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
    """Protocol for sampling strategies (return a :class:`Sample`)."""

    def sample(self, n: int, distr: "Distribution", **options: Any) -> Sample: ...


class DefaultSamplingUnivariateStrategy(SamplingStrategy):
    """
    Default univariate sampler using inverse transform sampling.

    The strategy resolves the distribution's ``ppf`` and applies it to i.i.d.
    uniforms ``U ~ U(0, 1)``.

    Returns
    -------
    ArraySample
        A 2D sample of shape ``(n, 1)``.
    """

    def sample(self, n: int, distr: "Distribution", **options: Any) -> ArraySample:
        ppf = distr.query_method("ppf", **options)
        rng = np.random.default_rng()
        U = rng.random(n)
        vals = np.array([ppf(Ui) for Ui in U], dtype=np.float64).reshape(n, 1)
        return ArraySample(vals)
