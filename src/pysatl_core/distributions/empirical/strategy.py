"""
Computation strategy specialised for :class:`EmpiricalDistribution`.

Adds estimator-identity tracking on top of :class:`DefaultComputationStrategy`:
fitted methods derived from a previous underlying estimator (e.g. KDE fit on
the sample) are dropped automatically when the empirical method is swapped,
so the strategy never returns a CDF/PPF derived from a stale PDF.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from pysatl_core.distributions.computations.computation import FittedComputationMethod
from pysatl_core.distributions.strategies import DefaultComputationStrategy
from pysatl_core.types import CharacteristicName, ComputationFunc, Method

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pysatl_core.distributions.computations.options import StepOptions
    from pysatl_core.distributions.distribution import Distribution
    from pysatl_core.types import GenericCharacteristicName


_PPF_GRID_SIZE = 1025
_PPF_TAIL_MARGIN_STD = 6.0
_PPF_QUANTILE_CLIP_EPS = 1e-6


class EmpiricalComputationStrategy(DefaultComputationStrategy):
    """
    Computation strategy used by :class:`EmpiricalDistribution`.

    Behaves like :class:`DefaultComputationStrategy`, except the fitted-method
    cache is implicitly keyed by the identity of the distribution's underlying
    estimator (``distr._estimator``). Whenever the strategy notices that the
    estimator has changed since the last query, it clears the cache so that
    any fitted CDF/PPF/etc. previously derived from the old estimator is
    rebuilt against the new one.

    This means callers do **not** need to call an explicit ``invalidate()``
    after swapping the empirical method on the distribution — the strategy
    detects the swap on the next characteristic query.

    The strategy also works for distributions without an ``_estimator``
    attribute (it then degenerates to plain Default behaviour).
    """

    def __init__(self, enable_caching: bool = False) -> None:
        super().__init__(enable_caching=enable_caching)
        # We hold a strong reference to the last seen estimator. Two reasons:
        # (1) `is`-comparison is unambiguous (id() can be recycled by GC);
        # (2) the small memory cost of keeping one extra reference is
        # negligible compared to the cost of silently caching a stale fit.
        self._tracked_estimator: object | None = None
        # Separate cache for the vectorised PPF built by this strategy.
        # Keyed by id(distr) so entries expire automatically when the
        # distribution object is replaced; cleared explicitly in
        # _maybe_invalidate when the estimator changes.
        self._ppf_cache: dict[int, FittedComputationMethod[Any, Any]] = {}

    def _maybe_invalidate(self, distr: Distribution) -> None:
        """
        Drop the cache if the distribution's estimator changed since last query.

        Distributions without an ``_estimator`` attribute (or with ``None``) are
        treated as "no estimator"; the cache is reset only on transitions between
        distinct estimator objects, not on every query.
        """
        current = getattr(distr, "_estimator", None)
        if current is not self._tracked_estimator:
            self._cache.clear()
            self._ppf_cache.clear()
            self._tracked_estimator = current

    def query_method(
        self,
        state: GenericCharacteristicName,
        distr: Distribution,
        options: StepOptions | None = None,
        *,
        characteristic_options: Mapping[str, Any] | None = None,
        computation_defaults: Mapping[str, Any] | None = None,
    ) -> Method[Any, Any]:
        self._maybe_invalidate(distr)

        if state == CharacteristicName.PPF and self._can_build_vectorized_ppf(distr):
            distr_id = id(distr)
            cached = self._ppf_cache.get(distr_id)
            if cached is not None:
                return cached
            fitted = self._build_vectorized_ppf(
                distr,
                options,
                characteristic_options=characteristic_options,
                computation_defaults=computation_defaults,
            )
            if self._enable_caching:
                self._ppf_cache[distr_id] = fitted
            return fitted

        return super().query_method(
            state,
            distr,
            options,
            characteristic_options=characteristic_options,
            computation_defaults=computation_defaults,
        )

    @staticmethod
    def _can_build_vectorized_ppf(distr: Distribution) -> bool:
        """The vectorised PPF needs a 1-D sample and a way to evaluate the CDF."""
        sample = getattr(distr, "_sample", None)
        if sample is None:
            return False
        sample_arr = np.asarray(sample, dtype=float)
        if sample_arr.ndim != 1 or sample_arr.size == 0:
            return False
        return CharacteristicName.CDF in distr.analytical_computations or (
            CharacteristicName.PDF in distr.analytical_computations
        )

    def _build_vectorized_ppf(
        self,
        distr: Distribution,
        options: StepOptions | None = None,
        *,
        characteristic_options: Mapping[str, Any] | None = None,
        computation_defaults: Mapping[str, Any] | None = None,
    ) -> FittedComputationMethod[Any, Any]:
        """
        Tabulate the CDF on a sample-aware grid and invert via PCHIP.

        Cheaper than per-point root-finding when the caller passes an array of
        quantiles, and naturally vectorised. PCHIP preserves the monotonicity
        of the input CDF so the inverse is well-defined as long as the
        tabulated CDF is strictly increasing, which we enforce by clipping
        numerical noise.
        """
        from scipy.interpolate import PchipInterpolator

        sample: NDArray[np.float64] = np.asarray(distr._sample, dtype=np.float64)  # type: ignore[attr-defined]
        sample_min = float(sample.min())
        sample_max = float(sample.max())
        # `std=0` only happens for a degenerate constant sample; in that case
        # the KDE collapses to a point mass and PPF is the constant itself.
        std = float(sample.std(ddof=0)) or 1.0
        margin = _PPF_TAIL_MARGIN_STD * std
        x_grid = np.linspace(sample_min - margin, sample_max + margin, _PPF_GRID_SIZE)

        cdf_method = super().query_method(
            CharacteristicName.CDF,
            distr,
            options,
            characteristic_options=characteristic_options,
            computation_defaults=computation_defaults,
        )
        cdf_values = np.asarray(cdf_method(x_grid), dtype=float)

        cdf_values = np.maximum.accumulate(cdf_values)
        cdf_values = np.clip(cdf_values, _PPF_QUANTILE_CLIP_EPS, 1.0 - _PPF_QUANTILE_CLIP_EPS)
        eps = np.finfo(float).eps
        for i in range(1, cdf_values.size):
            if cdf_values[i] <= cdf_values[i - 1]:
                cdf_values[i] = cdf_values[i - 1] + eps

        interpolator = PchipInterpolator(cdf_values, x_grid, extrapolate=False)
        cdf_lo, cdf_hi = float(cdf_values[0]), float(cdf_values[-1])
        x_lo, x_hi = float(x_grid[0]), float(x_grid[-1])

        def _ppf(q: Any, **_options: Any) -> NDArray[np.float64]:
            q_arr = np.asarray(q, dtype=float)
            q_clipped = np.clip(q_arr, cdf_lo, cdf_hi)
            result: NDArray[np.float64] = interpolator(q_clipped)
            # PchipInterpolator with extrapolate=False returns NaN outside the
            # support; the explicit clip above prevents this, but keep a safety
            # net for callers that pass q exactly on the grid endpoints.
            return np.where(np.isnan(result), np.where(q_arr <= cdf_lo, x_lo, x_hi), result)

        return FittedComputationMethod[Any, Any](
            target=CharacteristicName.PPF,
            sources=(CharacteristicName.CDF,),
            func=cast(ComputationFunc[Any, Any], _ppf),
        )


__all__ = ["EmpiricalComputationStrategy"]
