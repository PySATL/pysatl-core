"""Gaussian kernel density estimator for empirical distributions."""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from pysatl_core.distributions.computations.computation import AnalyticalComputation
from pysatl_core.distributions.support import ContinuousSupport
from pysatl_core.types import (
    CharacteristicName,
    ComputationFunc,
    DistributionType,
    GenericCharacteristicName,
    UnivariateContinuous,
)

if TYPE_CHECKING:
    from pysatl_core.distributions.support import Support


_CDF_CHUNK_ELEMENTS = 4_000_000
"""
Upper bound on the ``len(x) x len(sample)`` temporary built by the KDE CDF.

Evaluating the CDF in closed form materialises one kernel value per
(query point, observation) pair; at float64 this cap keeps the temporary
around 32 MB, and query points are processed in chunks that respect it.
"""


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

    @property
    def distribution_type(self) -> DistributionType:
        return UnivariateContinuous

    def resolve_support(
        self, sample: NDArray[np.float64], support: Support | None
    ) -> Support | None:
        _ = sample
        self._reject_bounded_support(support)
        return support

    @staticmethod
    def _reject_bounded_support(support: Support | None) -> None:
        """Reject finite bounds that an uncorrected Gaussian kernel cannot honour."""
        if support is None:
            return
        if not isinstance(support, ContinuousSupport):
            raise TypeError("ScipyGaussianKde requires a continuous support.")

        left = float(getattr(support, "left", -np.inf))
        right = float(getattr(support, "right", np.inf))
        if np.isfinite(left) or np.isfinite(right):
            raise NotImplementedError(
                f"EmpiricalDistribution cannot honour the bounded support "
                f"[{left}, {right}]: kernel estimators leak probability past a "
                f"boundary, and ScipyGaussianKde has no boundary correction. "
                f"Omit 'support' (the unbounded default is the honest choice "
                f"for a Gaussian kernel), or pre-transform the sample onto the "
                f"whole line."
            )

    @staticmethod
    def validate_sample(sample: NDArray[np.float64]) -> None:
        """Check that the observations can produce a nondegenerate Gaussian KDE."""
        if sample.size < 2:
            raise ValueError(
                f"Fitting a density estimator requires at least 2 observations, got {sample.size}."
            )
        if sample.std(ddof=0) == 0.0:
            raise ValueError(
                f"Sample is constant (all {sample.size} values equal {float(sample[0])!r}); "
                "a density estimate degenerates to a point mass and cannot be "
                "evaluated numerically."
            )

    def fit(
        self, sample: NDArray[np.float64]
    ) -> Mapping[GenericCharacteristicName, AnalyticalComputation[Any, Any]]:
        self.validate_sample(sample)
        from scipy.stats import gaussian_kde

        estimate = _ScipyKdeEstimate(gaussian_kde(sample, bw_method=self.bandwidth))
        # CDF first: the current resolver uses insertion order when choosing a
        # starting characteristic for a conversion to PPF.
        return {
            CharacteristicName.CDF: AnalyticalComputation(
                CharacteristicName.CDF, cast(ComputationFunc[Any, Any], estimate.cdf)
            ),
            CharacteristicName.PDF: AnalyticalComputation(
                CharacteristicName.PDF, cast(ComputationFunc[Any, Any], estimate.pdf)
            ),
        }


class _ScipyKdeEstimate:
    """Hold the fitted Gaussian kernel and its CDF parameters."""

    def __init__(self, kde: Any) -> None:
        self._kde = kde
        # Kernel parameters read once, because cdf() needs them on every call.
        # Nothing here refits, and the supported way to change the bandwidth is
        # a fresh fit through EmpiricalDistribution.set_estimator.  Note that
        # gaussian_kde is not frozen: its public set_bandwidth() rewrites
        # `covariance`, which would desync this cache from pdf() -- that one
        # reads the live object -- and leave the two describing different
        # distributions.  Do not call it on `self._kde`.
        self._bandwidth = float(np.sqrt(kde.covariance[0, 0]))
        self._centers = np.asarray(kde.dataset[0], dtype=float)
        self._weights = np.asarray(kde.weights, dtype=float)

    def pdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        scalar_input = np.ndim(x) == 0
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        finite = np.isfinite(x_arr)
        result = np.zeros_like(x_arr)
        if finite.any():
            result[finite] = self._kde.pdf(x_arr[finite])
        # Infinite inputs have zero Gaussian density; NaN must remain visible
        # rather than being mistaken for a valid zero-density observation.
        result[np.isnan(x_arr)] = np.nan
        return result[0] if scalar_input else result

    def cdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate the KDE's CDF at *x* in closed form.

        ``scipy.stats.gaussian_kde`` exposes no vectorised CDF, but a Gaussian
        kernel mixture has one: ``F(x) = sum_i w_i * Phi((x - d_i) / h)`` over
        the observations ``d_i`` with kernel bandwidth ``h``.  Evaluating that
        directly replaces the obvious loop over
        ``gaussian_kde.integrate_box_1d`` and, as a bonus, needs no special
        casing for non-finite input: ``Phi`` maps ``+-inf`` to ``1``/``0`` and
        propagates ``NaN`` on its own.

        Notes
        -----
        The cost is ``O(len(x) * len(sample))`` kernel evaluations, which is
        inherent to the closed form rather than to Python-level overhead —
        vectorising the loop buys ~1.4x, not an order of magnitude. PPF
        queries invert this CDF through vectorised bisection in the
        characteristic graph.

        If grid evaluation becomes the bottleneck, the way out is
        algorithmic, not micro-optimisation: on a *uniform* grid the sum above
        is a convolution, so linear binning plus an FFT computes it in
        ``O(m log m)`` independently of the sample size, at the price of a
        binning error. That path needs a uniform grid, so it does not fit this
        "evaluate at arbitrary points" signature; it would go behind an
        optional grid-evaluation hook on a future empirical estimator.
        """
        from scipy.special import ndtr

        scalar_input = np.ndim(x) == 0
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        flat = x_arr.ravel()

        centers = self._centers
        weights = self._weights
        flat_result = np.empty(flat.shape, dtype=float)

        chunk = max(1, _CDF_CHUNK_ELEMENTS // centers.size)
        for start in range(0, flat.size, chunk):
            block = flat[start : start + chunk]
            z = (block[:, None] - centers[None, :]) / self._bandwidth
            flat_result[start : start + chunk] = ndtr(z) @ weights

        result: NDArray[np.float64] = flat_result.reshape(x_arr.shape)
        return result[0] if scalar_input else result
