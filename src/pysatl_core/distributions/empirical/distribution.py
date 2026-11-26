"""
Empirical Distribution

Fits an empirical estimator to observed data and passes its available
characteristics to ``Distribution``. The built-in estimators are a continuous
Gaussian KDE and a discrete empirical CDF.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from pysatl_core.distributions.computations.computation import AnalyticalComputation
from pysatl_core.distributions.distribution import _KEEP, Distribution
from pysatl_core.distributions.empirical.estimators import (
    EmpiricalCdf,
    EmpiricalDistributionEstimator,
    ScipyGaussianKde,
)
from pysatl_core.distributions.strategies import (
    ComputationStrategy,
    DefaultComputationStrategy,
    SamplingStrategy,
)
from pysatl_core.sampling.unuran.core.unuran_sampling_strategy import DefaultUnuranSamplingStrategy
from pysatl_core.types import (
    EuclideanDistributionType,
    GenericCharacteristicName,
    LabelName,
)

if TYPE_CHECKING:
    from pysatl_core.distributions.support import Support


class EmpiricalDistribution(Distribution):
    """
    A univariate distribution fitted to observations by an empirical estimator.

    The estimator supplies the estimate's direct characteristics, distribution type,
    and support. Gaussian KDE is the default continuous estimator; :class:`EmpiricalCdf`
    produces a discrete distribution with CDF and PMF. Other characteristics
    are resolved by the normal characteristic graph.

    Parameters
    ----------
    sample : NDArray[np.float64]
        One-dimensional array of observed values used to fit the estimator.
    estimator : EmpiricalDistributionEstimator, default ScipyGaussianKde()
        Strategy used to fit the empirical distribution.
    support : Support or None, default None
        Optional explicit support, validated by the selected estimator. The KDE
        requires an unbounded support; the ECDF derives its finite support
        from the observations when this is omitted.
    sampling_strategy : SamplingStrategy or None, default None
        Overrides the default UNU.RAN sampling strategy.
    computation_strategy : ComputationStrategy or None, default None
        Overrides the default graph-based computation strategy.

    Raises
    ------
    ValueError
        If *sample* is not a nonempty 1-D finite array, or the estimator cannot
        fit it. KDE additionally needs two distinct observations.
    NotImplementedError
        If the KDE is given a bounded support without boundary correction.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> sample = rng.normal(0, 1, 500)
    >>> distr = EmpiricalDistribution(sample)
    >>> distr.calculate_characteristic("pdf", np.array([0.0]))  # doctest: +ELLIPSIS
    array([0.365...])
    """

    def __init__(
        self,
        sample: NDArray[np.float64],
        estimator: EmpiricalDistributionEstimator = ScipyGaussianKde(),
        support: Support | None = None,
        sampling_strategy: SamplingStrategy | None = None,
        computation_strategy: ComputationStrategy | None = None,
    ) -> None:
        validated = self.validate_sample(sample)
        distribution_type = self._estimator_distribution_type(estimator)
        # Own the observations before fitting. A bytes-backed array cannot be
        # made writable through the public data property or a view of it.
        self._sample = np.frombuffer(validated.tobytes(), dtype=validated.dtype)
        self._requested_support = support
        resolved_support = estimator.resolve_support(self._sample, support)
        computations = estimator.fit(self._sample)
        self._estimator = estimator

        super().__init__(
            distribution_type=distribution_type,
            analytical_computations=computations,
            support=resolved_support,
            sampling_strategy=sampling_strategy or DefaultUnuranSamplingStrategy(),
            computation_strategy=computation_strategy
            or DefaultComputationStrategy(enable_caching=True),
        )

    @staticmethod
    def _estimator_distribution_type(
        estimator: EmpiricalDistributionEstimator,
    ) -> EuclideanDistributionType:
        distr_type = estimator.distribution_type
        if not isinstance(distr_type, EuclideanDistributionType) or distr_type.dimension != 1:
            raise ValueError("EmpiricalDistribution requires a univariate Euclidean estimator.")
        return distr_type

    @staticmethod
    def validate_sample(sample: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Check common univariate observation requirements and normalise the data.

        Parameters
        ----------
        sample : array_like
            Observed values, coerced to a float array.

        Returns
        -------
        NDArray[np.float64]
            The sample as a 1-D float array.

        Raises
        ------
        ValueError
            If *sample* is empty, non-finite, or not one-dimensional.
        """
        sample = np.asarray(sample, dtype=float)

        if sample.ndim != 1:
            raise ValueError(
                f"EmpiricalDistribution is univariate and requires a 1-D sample, "
                f"got an array of shape {sample.shape}."
            )
        if sample.size == 0:
            raise ValueError("EmpiricalDistribution requires at least 1 observation, got 0.")
        if not np.all(np.isfinite(sample)):
            non_finite = int((~np.isfinite(sample)).sum())
            positions = np.flatnonzero(~np.isfinite(sample))
            shown = ", ".join(str(int(i)) for i in positions[:5])
            if positions.size > 5:
                shown += ", ..."
            raise ValueError(
                f"Sample must be finite: {non_finite} of {sample.size} values are "
                f"NaN or infinite (at index {shown})."
            )
        return sample

    @property
    def data(self) -> NDArray[np.float64]:
        """Immutable snapshot of the observations used to fit this distribution."""
        return self._sample

    @property
    def estimator(self) -> EmpiricalDistributionEstimator:
        """The empirical estimator currently configured on this distribution."""
        return self._estimator

    def with_estimator(self, estimator: EmpiricalDistributionEstimator) -> EmpiricalDistribution:
        """
        Return a clone of this distribution with a different empirical estimator.

        The clone refits the new estimator on the same immutable sample snapshot.
        Strategies are deep-copied like in any other ``with_*`` clone, so the
        clone is independent of whatever the original memoises later.  The
        sampler starts empty (see ``DefaultUnuranSamplingStrategy.__deepcopy__``).
        The clone has a fresh analytical mapping, so the computation strategy
        drops copied plans on its first query.

        Use this in preference to :meth:`set_estimator` when you want to compare
        estimators side-by-side or keep the original distribution intact.
        """
        return self._clone_with_strategies(estimator=estimator)

    def set_estimator(self, estimator: EmpiricalDistributionEstimator) -> None:
        """
        Replace the empirical estimator in place.

        Refits ``estimator`` on the original sample and replaces the distribution
        type, support, and analytical computations together. The computation
        strategy drops cached methods and plans after the replacement.

        Sampling strategies that hold cached state (notably
        :class:`DefaultUnuranSamplingStrategy`, whose generator is built once
        on the PDF at the time of first sample) are reset via their
        ``invalidate()`` method when present. Strategies without
        ``invalidate`` are left untouched — the assumption is that they hold
        no per-distribution state.

        If fitting or validation fails, the current distribution remains unchanged.

        Notes
        -----
        Any external code that holds a direct reference to internal sampler
        state (e.g. a value previously read from
        ``distr.sampling_strategy._sampler``) keeps that reference alive and
        will continue to sample from the *previous* distribution. Re-acquire
        such references after calling :meth:`set_estimator`.

        For side-by-side comparison of estimators, prefer :meth:`with_estimator`.
        """
        # Prepare everything that can fail before mutating the live distribution.
        distribution_type = self._estimator_distribution_type(estimator)
        resolved_support = estimator.resolve_support(self._sample, self._requested_support)
        computations = self._normalize_analytical_computations(estimator.fit(self._sample))
        self._estimator = estimator
        self._distribution_type = distribution_type
        self._support = resolved_support
        self._analytical_computations = computations
        # Clear fitted methods and plans derived from the previous mapping.
        # Custom strategies can opt into this invalidate() hook.
        getattr(self._computation_strategy, "invalidate", lambda: None)()
        # Sampling cache: must be reset explicitly — UNURAN's C-side init
        # captured the old characteristics and cannot be patched in place.
        getattr(self._sampling_strategy, "invalidate", lambda: None)()

    def _clone_with_strategies(
        self,
        *,
        sampling_strategy: SamplingStrategy | None | object = _KEEP,
        computation_strategy: ComputationStrategy | None | object = _KEEP,
        estimator: EmpiricalDistributionEstimator | object = _KEEP,
    ) -> EmpiricalDistribution:
        clone = object.__new__(EmpiricalDistribution)
        clone._sample = self._sample
        clone._requested_support = self._requested_support
        if estimator is _KEEP:
            clone._estimator = self._estimator
            computations: Mapping[
                GenericCharacteristicName,
                AnalyticalComputation[Any, Any]
                | Mapping[LabelName, AnalyticalComputation[Any, Any]],
            ] = self.analytical_computations
            resolved_support = self.support
        else:
            new_estimator = cast(EmpiricalDistributionEstimator, estimator)
            clone._estimator = new_estimator
            resolved_support = new_estimator.resolve_support(self._sample, self._requested_support)
            computations = new_estimator.fit(self._sample)
        Distribution.__init__(
            clone,
            distribution_type=self._estimator_distribution_type(clone._estimator),
            analytical_computations=computations,
            support=resolved_support,
            sampling_strategy=self._new_sampling_strategy(sampling_strategy=sampling_strategy),
            computation_strategy=self._new_computation_strategy(
                computation_strategy=computation_strategy
            ),
        )
        return clone


__all__ = [
    "EmpiricalCdf",
    "EmpiricalDistribution",
    "EmpiricalDistributionEstimator",
    "ScipyGaussianKde",
]
