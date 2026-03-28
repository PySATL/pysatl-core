"""
Computation and Sampling Strategies

This module defines strategies for computing distribution characteristics
and generating random samples.
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from pysatl_core.distributions.registry import characteristic_registry
from pysatl_core.types import CharacteristicName, Method, NumericArray

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pysatl_core.distributions.computation import (
        AnalyticalComputation,
        FittedComputationMethod,
    )
    from pysatl_core.distributions.distribution import Distribution
    from pysatl_core.distributions.registry.graph import RegistryView
    from pysatl_core.types import GenericCharacteristicName, LabelName


class ComputationStrategy(Protocol):
    """
    Protocol for strategies that resolve computation methods for characteristics.

    Attributes
    ----------
    enable_caching : bool
        Whether to cache fitted computation methods.
    """

    def query_method(
        self, state: GenericCharacteristicName, distr: Distribution, **options: Any
    ) -> Method[Any, Any]: ...


class DefaultComputationStrategy:
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
    _enable_caching : bool
        Whether caching is enabled.
    _cache : dict[str, FittedComputationMethod]
        Cache of fitted computation methods.
    _resolving : dict[int, set[str]]
        Tracking of currently resolving characteristics to detect cycles.
    """

    def __init__(self, enable_caching: bool = False) -> None:
        self._enable_caching = enable_caching
        self._cache: dict[GenericCharacteristicName, FittedComputationMethod[Any, Any]] = {}
        self._resolving: dict[int, set[GenericCharacteristicName]] = {}

    @property
    def is_caching_enabled(self) -> bool:
        return self._enable_caching

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

    @staticmethod
    def _pick_analytical_method(
        state: GenericCharacteristicName,
        methods: Mapping[LabelName, AnalyticalComputation[Any, Any]],
    ) -> AnalyticalComputation[Any, Any]:
        """
        Pick the first available analytical method for a characteristic.

        Raises
        ------
        RuntimeError
            If no labeled analytical methods are available for the characteristic.
        """
        try:
            return next(iter(methods.values()))
        except StopIteration as exc:
            raise RuntimeError(
                f"Characteristic '{state}' provides no labeled analytical computations."
            ) from exc

    @staticmethod
    def _pick_loop_method(
        state: GenericCharacteristicName,
        view: RegistryView,
    ) -> Method[Any, Any] | None:
        """
        Pick the first available self-loop method for a characteristic in a view.
        """
        loops = view.variants(state, state)
        if not loops:
            return None
        return cast(Method[Any, Any], next(iter(loops.values())).method)

    def query_method(
        self, state: GenericCharacteristicName, distr: Distribution, **options: Any
    ) -> Method[Any, Any]:
        """
        Resolve a computation method for the target characteristic.

        Resolution order:
        1. Cached fitted method (if caching enabled)
        2. Analytical implementation for non-registry characteristics
        3. First self-loop from the registry view
        4. Conversion path from loop characteristics via the graph

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
        # 1. Check cache if enabled
        if self._enable_caching:
            cached = self._cache.get(state)
            if cached is not None:
                return cached

        # 2. Require at least one analytical characteristic
        if not distr.analytical_computations:
            raise RuntimeError(
                "Distribution provides no analytical computations to ground conversions."
            )

        # 3. Non-registry characteristics are resolved directly.
        # It covers the situation where user is providing their analytical computation which isn't
        # in the graph
        registry = characteristic_registry()
        if state not in registry.declared_characteristics:
            if state in distr.analytical_computations:
                return self._pick_analytical_method(state, distr.analytical_computations[state])
            raise RuntimeError(
                f"Characteristic '{state}' is not declared in the registry and has no "
                "analytical implementation in the distribution."
            )

        # 4. Get filtered graph view for this distribution.
        view = registry.view(distr)

        self._push_guard(distr, state)
        try:
            loop_method = self._pick_loop_method(state, view)
            if loop_method is not None:
                return loop_method

            # 5. Try each loop characteristic as a source
            for src in distr.analytical_computations:
                if not view.variants(src, src):
                    continue

                # Find conversion path in the graph
                path = view.find_path(src, state)
                if not path:
                    continue

                # Fit each edge along the path
                last_fitted: FittedComputationMethod[Any, Any] | None = None
                for edge in path:
                    fitted = edge.prepare(distr, **options)
                    if self._enable_caching and edge.cacheable:
                        self._cache[edge.target] = fitted
                    last_fitted = fitted

                if last_fitted is None:
                    raise RuntimeError(f"Empty path when resolving '{state}' from '{src}'.")
                return last_fitted

            raise RuntimeError(
                "No conversion path from any characteristic in "
                f"analytical_computations to '{state}'."
            )
        finally:
            self._pop_guard(distr, state)


class SamplingStrategy(Protocol):
    """Protocol for strategies that generate samples from distributions."""

    def sample(self, n: int, distr: Distribution, **options: Any) -> NumericArray: ...


class DefaultSamplingUnivariateStrategy(SamplingStrategy):
    """
    Default univariate sampler based on inverse transform sampling.

    This strategy generates samples by applying the PPF (inverse CDF)
    to uniformly distributed random variables.

    Notes
    -----
    - Requires the distribution to provide a PPF computation method.
    - Assumes that the PPF follows NumPy semantics (vectorized evaluation).
    - Returns a NumPy array containing the generated samples.
    """

    def sample(self, n: int, distr: Distribution, **options: Any) -> NumericArray:
        """
        Generate samples from the distribution.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        distr : Distribution
            Distribution to sample from.
        **options : Any
            Additional options forwarded to the PPF computation.

        Returns
        -------
        NumericArray
            NumPy array containing ``n`` generated samples.
            The exact array shape depends on the distribution and sampling strategy.
        """
        ppf = distr.query_method(CharacteristicName.PPF, **options)
        rng = np.random.default_rng()
        U = rng.random(n)
        samples = np.asarray(ppf(U), dtype=float)
        if samples.shape != U.shape:
            raise RuntimeError(
                "PPF must preserve NumPy input shape for inverse-transform sampling."
            )
        return cast(NumericArray, samples)
