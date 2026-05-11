"""
Empirical Distribution

Wraps a density estimator (e.g. KDE) built from observed data into a
``Distribution`` that integrates with the characteristic graph.
Providing only a PDF analytical computation is enough — the graph
derives CDF and PPF automatically via numerical fitters.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from pysatl_core.distributions.computations.computation import AnalyticalComputation
from pysatl_core.distributions.distribution import _KEEP, Distribution
from pysatl_core.distributions.empirical.strategy import EmpiricalComputationStrategy
from pysatl_core.distributions.strategies import ComputationStrategy, SamplingStrategy
from pysatl_core.sampling.unuran.core.unuran_sampling_strategy import DefaultUnuranSamplingStrategy
from pysatl_core.types import CharacteristicName, ComputationFunc, UnivariateContinuous

if TYPE_CHECKING:
    from pysatl_core.distributions.support import Support


class FittedEmpirical(Protocol):
    """A fitted empirical density estimator that can evaluate PDF and CDF."""

    def pdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the probability density function at points *x*."""
        ...

    def cdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the cumulative distribution function at points *x*."""
        ...


class EmpiricalMethod(Protocol):
    """Strategy for constructing a :class:`FittedEmpirical` from a data sample."""

    def fit(self, sample: NDArray[np.float64]) -> FittedEmpirical:
        """Fit the estimator to *sample* and return an evaluable estimator."""
        ...


@dataclass(frozen=True)
class ScipyGaussianKde:
    """
    Gaussian KDE via :func:`scipy.stats.gaussian_kde`.

    Parameters
    ----------
    bandwidth : float or {"scott", "silverman"}, default "scott"
        Bandwidth selection method or explicit scalar value.
    """

    bandwidth: float | Literal["scott", "silverman"] = "scott"

    def fit(self, sample: NDArray[np.float64]) -> FittedEmpirical:
        from scipy.stats import gaussian_kde

        return _ScipyFittedKde(gaussian_kde(sample, bw_method=self.bandwidth))


class _ScipyFittedKde:
    """Thin wrapper that adapts ``scipy.stats.gaussian_kde`` to ``FittedEmpirical``."""

    def __init__(self, kde: Any) -> None:
        self._kde = kde

    def pdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        scalar_input = np.ndim(x) == 0
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        finite = np.isfinite(x_arr)
        result = np.zeros_like(x_arr)
        if finite.any():
            result[finite] = self._kde.pdf(x_arr[finite])
        return result[0] if scalar_input else result

    def cdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        scalar_input = np.ndim(x) == 0
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        result = np.where(x_arr == np.inf, 1.0, np.where(x_arr == -np.inf, 0.0, np.nan))
        finite = np.isfinite(x_arr)
        if finite.any():
            result[finite] = np.array(
                [self._kde.integrate_box_1d(-np.inf, xi) for xi in x_arr[finite]],
                dtype=float,
            )
        return result[0] if scalar_input else result


class EmpiricalDistribution(Distribution):
    """
    A continuous univariate distribution built from an empirical density
    estimator (KDE by default) fitted to observed data.

    The PDF is provided analytically via the estimator; CDF and PPF
    are derived automatically by the characteristic graph (numerical
    integration and root-finding respectively).

    Parameters
    ----------
    sample : NDArray[np.float64]
        One-dimensional array of observed values used to fit the estimator.
    method : EmpiricalMethod, default ScipyGaussianKde()
        Strategy used to construct the empirical density estimator.
    support : Support or None, default None
        Explicit support for the distribution.  ``None`` leaves the support
        unrestricted, which is the natural choice for Gaussian KDE.
    sampling_strategy : SamplingStrategy or None, default None
        Overrides the default inverse-transform sampling strategy.
    computation_strategy : ComputationStrategy or None, default None
        Overrides the default graph-based computation strategy.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> sample = rng.normal(0, 1, 500)
    >>> distr = EmpiricalDistribution(sample)
    >>> distr.calculate_characteristic("pdf", np.array([0.0]))
    array([0.39...])
    """

    def __init__(
        self,
        sample: NDArray[np.float64],
        method: EmpiricalMethod = ScipyGaussianKde(),
        support: Support | None = None,
        sampling_strategy: SamplingStrategy | None = None,
        computation_strategy: ComputationStrategy | None = None,
    ) -> None:
        self._sample = np.asarray(sample, dtype=float)
        self._method = method
        self._estimator = method.fit(self._sample)

        # TODO: DefaultUnuranSamplingStrategy fallback calls DefaultSamplingUnivariateStrategy
        #  which passes the full U array to graph-derived PPF (scalar-only) and will fail.
        #  Fix by making the fallback use a scalar loop instead of ppf(U).
        super().__init__(
            distribution_type=UnivariateContinuous,
            analytical_computations=self._build_analytical_computations(),
            support=support,
            sampling_strategy=sampling_strategy or DefaultUnuranSamplingStrategy(),
            computation_strategy=computation_strategy or EmpiricalComputationStrategy(),
        )

    def _pdf(self, x: NDArray[np.float64], **_options: Any) -> NDArray[np.float64]:
        """Indirection: always reads the current ``_estimator``.

        Used as the analytical PDF func so swapping ``_estimator`` (via
        :meth:`set_method`) takes effect without rebuilding analytical
        computations or graph loop edges.
        """
        return self._estimator.pdf(x)

    def _cdf(self, x: NDArray[np.float64], **_options: Any) -> NDArray[np.float64]:
        """Indirection: always reads the current ``_estimator``. See :meth:`_pdf`."""
        return self._estimator.cdf(x)

    def _build_analytical_computations(
        self,
    ) -> dict[str, AnalyticalComputation[Any, Any]]:
        """Single source of truth for the analytical PDF/CDF entries."""
        result: dict[str, AnalyticalComputation[Any, Any]] = {
            CharacteristicName.PDF: AnalyticalComputation(
                target=CharacteristicName.PDF,
                func=cast(ComputationFunc[Any, Any], self._pdf),
            ),
            CharacteristicName.CDF: AnalyticalComputation(
                target=CharacteristicName.CDF,
                func=cast(ComputationFunc[Any, Any], self._cdf),
            ),
        }
        return result

    @property
    def data(self) -> NDArray[np.float64]:
        """The original data sample used to fit this distribution."""
        return self._sample

    @property
    def method(self) -> EmpiricalMethod:
        """The empirical method currently configured on this distribution."""
        return self._method

    def with_method(self, method: EmpiricalMethod) -> EmpiricalDistribution:
        """
        Return a clone of this distribution with a different empirical method.

        The clone refits the new method on the same sample (shared array, no copy).
        Strategies are deep-copied like in any other ``with_*`` clone, so the new
        instance starts with empty caches and a fresh sampler — independent of
        anything the original may have memoised.

        Use this in preference to :meth:`set_method` when you want to compare
        methods side-by-side or keep the original distribution intact.
        """
        return self._clone_with_strategies(method=method)

    def set_method(self, method: EmpiricalMethod) -> None:
        """
        Replace the empirical method in place.

        Refits ``method`` on the original sample and rebinds the underlying
        estimator. The graph's analytical loops do not need to be rebuilt:
        :meth:`_pdf` and :meth:`_cdf` always read the current estimator,
        and :class:`EmpiricalComputationStrategy` clears its fitted-method
        cache automatically when it notices the estimator object has changed.

        Sampling strategies that hold cached state (notably
        :class:`DefaultUnuranSamplingStrategy`, whose generator is built once
        on the PDF at the time of first sample) are reset via their
        ``invalidate()`` method when present. Strategies without
        ``invalidate`` are left untouched — the assumption is that they hold
        no per-distribution state.

        Notes
        -----
        Any external code that holds a direct reference to internal sampler
        state (e.g. a value previously read from
        ``distr.sampling_strategy._sampler``) keeps that reference alive and
        will continue to sample from the *previous* distribution. Re-acquire
        such references after calling :meth:`set_method`.

        For side-by-side comparison of methods, prefer :meth:`with_method`.
        """
        self._method = method
        self._estimator = method.fit(self._sample)
        # Computation cache: auto-invalidated on next query via estimator-id
        # tracking in EmpiricalComputationStrategy. Custom strategies that
        # implement an invalidate() hook get a chance to reset, too.
        getattr(self._computation_strategy, "invalidate", lambda: None)()
        # Sampling cache: must be reset explicitly — UNURAN's C-side init
        # captured the old PDF and cannot be patched in place.
        getattr(self._sampling_strategy, "invalidate", lambda: None)()

    def _clone_with_strategies(
        self,
        *,
        sampling_strategy: SamplingStrategy | None | object = _KEEP,
        computation_strategy: ComputationStrategy | None | object = _KEEP,
        method: EmpiricalMethod | object = _KEEP,
    ) -> EmpiricalDistribution:
        clone = object.__new__(EmpiricalDistribution)
        clone._sample = self._sample
        if method is _KEEP:
            clone._method = self._method
            clone._estimator = self._estimator
        else:
            new_method = cast(EmpiricalMethod, method)
            clone._method = new_method
            clone._estimator = new_method.fit(self._sample)
        Distribution.__init__(
            clone,
            distribution_type=UnivariateContinuous,
            analytical_computations=clone._build_analytical_computations(),
            support=self.support,
            sampling_strategy=self._new_sampling_strategy(sampling_strategy=sampling_strategy),
            computation_strategy=self._new_computation_strategy(
                computation_strategy=computation_strategy
            ),
        )
        return clone


__all__ = [
    "EmpiricalDistribution",
    "EmpiricalMethod",
    "FittedEmpirical",
    "ScipyGaussianKde",
]
