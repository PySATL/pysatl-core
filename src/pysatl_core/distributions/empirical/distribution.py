"""
Empirical Distribution

Wraps a density estimator (e.g. KDE) built from observed data into a
``Distribution`` that integrates with the characteristic graph.
PDF and CDF come straight from the estimator; PPF is derived by the
graph, which inverts the tabulated CDF (``cdf_to_ppf_tabulated_1C``).
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


_TABULATION_TAIL_STD = 6.0
"""Tail allowance of the tabulation domain, in sample standard deviations."""

_CDF_CHUNK_ELEMENTS = 4_000_000
"""
Upper bound on the ``len(x) x len(sample)`` temporary built by the KDE CDF.

Evaluating the CDF in closed form materialises one kernel value per
(query point, observation) pair; at float64 this cap keeps the temporary
around 32 MB, and query points are processed in chunks that respect it.
"""


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
        # Kernel parameters read once, because cdf() needs them on every call.
        # Nothing here refits, and the supported way to change the bandwidth is
        # a fresh fit through EmpiricalDistribution.set_method.  Note that
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
        vectorising the loop buys ~1.4x, not an order of magnitude.  The bulk
        of it is paid once per fit, when
        :func:`~pysatl_core.distributions.computations.continuous._fit_cdf_to_ppf_tabulated_1C`
        tabulates the CDF on its grid (measured: ~120 ms for a 1025-point grid
        over a 10k sample, ~99% of the PPF fit time), after which PPF calls are
        served from the cached interpolant in ~0.1 ms.

        If that fit time ever becomes the bottleneck, the way out is
        algorithmic, not micro-optimisation: on a *uniform* grid the sum above
        is a convolution, so linear binning plus an FFT computes it in
        ``O(m log m)`` independently of the sample size (measured: 0.5 ms for
        the same grid, i.e. 40-340x, at the price of a binning error of order
        1e-5 that shrinks quadratically with grid size).  That path needs a
        uniform grid, so it does not fit this "evaluate at arbitrary points"
        signature; it would go behind an optional ``cdf_on_grid(lo, hi, n)``
        hook on :class:`FittedEmpirical` that the tabulating fitter prefers
        when the estimator provides it.
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


class EmpiricalDistribution(Distribution):
    """
    A continuous univariate distribution built from an empirical density
    estimator (KDE by default) fitted to observed data.

    PDF and CDF are provided analytically by the estimator.  PPF is
    derived by the characteristic graph, which tabulates the CDF over
    :attr:`tabulation_domain` and inverts it monotonically, solving
    quantiles outside that range exactly.

    Parameters
    ----------
    sample : NDArray[np.float64]
        One-dimensional array of observed values used to fit the estimator.
    method : EmpiricalMethod, default ScipyGaussianKde()
        Strategy used to construct the empirical density estimator.
    support : Support or None, default None
        Explicit support for the distribution.  Must be unbounded: a kernel
        estimator cannot honour a finite bound without boundary correction,
        which is not implemented — see :meth:`_reject_bounded_support`.
        ``None`` leaves the support unrestricted, the natural choice for a
        Gaussian kernel.
    sampling_strategy : SamplingStrategy or None, default None
        Overrides the default inverse-transform sampling strategy.
    computation_strategy : ComputationStrategy or None, default None
        Overrides the default graph-based computation strategy.

    Raises
    ------
    ValueError
        If *sample* is not a 1-D array, holds fewer than two observations,
        contains non-finite values, or is constant.  See
        :meth:`validate_sample`.
    NotImplementedError
        If *support* declares a finite bound.  See
        :meth:`_reject_bounded_support`.

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
        method: EmpiricalMethod = ScipyGaussianKde(),
        support: Support | None = None,
        sampling_strategy: SamplingStrategy | None = None,
        computation_strategy: ComputationStrategy | None = None,
    ) -> None:
        self._sample = self.validate_sample(sample)
        self._reject_bounded_support(support)
        self._method = method
        self._estimator = method.fit(self._sample)

        super().__init__(
            distribution_type=UnivariateContinuous,
            analytical_computations=self._build_analytical_computations(),
            support=support,
            sampling_strategy=sampling_strategy or DefaultUnuranSamplingStrategy(),
            computation_strategy=computation_strategy or EmpiricalComputationStrategy(),
        )
        self._tabulation_domain = self._compute_tabulation_domain()

    @staticmethod
    def _reject_bounded_support(support: Support | None) -> None:
        """
        Refuse a support the estimator cannot actually honour.

        A kernel estimator places a kernel on every observation, and a kernel
        with unbounded tails (the Gaussian default) spreads mass across the
        whole line.  Declaring a finite bound does not stop that: on a sample
        of positive values with ``support=[0, inf)`` a few percent of the
        fitted mass ends up below zero -- measured on 1000 draws with Scott's
        bandwidth: 2% for gamma(2), 4% for a half-normal, 8% for an
        exponential, the leak growing with the density at the boundary -- so
        ``pdf`` is non-zero and ``cdf(0)`` is non-zero where both must vanish.

        Honouring a bounded support needs a boundary-corrected estimator
        (reflection, boundary kernels, a transform to the unbounded line),
        which no :class:`EmpiricalMethod` here implements.  Until one does,
        accepting the argument would hand back a distribution that quietly
        assigns probability to impossible values, so it is refused outright.

        Unbounded supports are accepted: they impose nothing, and an explicit
        ``(-inf, inf)`` is a legitimate way to state that.

        Raises
        ------
        NotImplementedError
            If *support* declares a finite lower or upper bound.
        """
        if support is None:
            return

        left = float(getattr(support, "left", -np.inf))
        right = float(getattr(support, "right", np.inf))
        if np.isfinite(left) or np.isfinite(right):
            raise NotImplementedError(
                f"EmpiricalDistribution cannot honour the bounded support "
                f"[{left}, {right}]: kernel estimators leak probability past a "
                f"boundary, and no boundary-corrected EmpiricalMethod is available "
                f"yet. Omit 'support' (the unbounded default is the honest choice "
                f"for a Gaussian kernel), or pre-transform the sample onto the "
                f"whole line."
            )

    @staticmethod
    def validate_sample(sample: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Check that *sample* can carry a density estimate, and normalise it.

        Runs before the estimator is fitted so that an unusable sample is
        rejected by this class with a message naming the real cause, rather
        than surfacing later as an error from the estimator's internals — or,
        worse, not surfacing at all.

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
            If *sample* is not one-dimensional; holds fewer than two
            observations; contains ``NaN`` or infinities; or is constant.

        Notes
        -----
        The constant-sample check is what makes this more than a convenience.
        Identical observations have zero variance, so a kernel estimator has no
        scale to work from and the density degenerates to a point mass.  The
        built-in method does fail on its own — ``gaussian_kde`` raises
        ``LinAlgError`` on the singular covariance matrix — but it fails in the
        vocabulary of its own internals, reporting data "in a lower-dimensional
        subspace" and advising principal component analysis, which is not
        actionable advice for a one-dimensional sample.  Checking here turns
        that into a ``ValueError`` naming the actual cause, and does so
        uniformly for every :class:`EmpiricalMethod` — including ones that
        would return unusable numbers rather than raise.

        The check catches exact degeneracy only.  A sample whose spread is
        merely tiny relative to its scale (identical values plus 1e-9 noise,
        say) is degenerate in the same way but passes; guarding that needs a
        scale-relative criterion and is left open deliberately.
        """
        sample = np.asarray(sample, dtype=float)

        if sample.ndim != 1:
            raise ValueError(
                f"EmpiricalDistribution is univariate and requires a 1-D sample, "
                f"got an array of shape {sample.shape}."
            )
        if sample.size < 2:
            raise ValueError(
                f"Fitting a density estimator requires at least 2 observations, got {sample.size}."
            )
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
        if sample.std(ddof=0) == 0.0:
            raise ValueError(
                f"Sample is constant (all {sample.size} values equal {float(sample[0])!r}); "
                f"a density estimate degenerates to a point mass and cannot be "
                f"evaluated numerically."
            )

        return sample

    def _pdf(self, x: NDArray[np.float64], **_options: Any) -> NDArray[np.float64]:
        """Indirection: always reads the current :attr:`estimator`.

        Used as the analytical PDF func so swapping the estimator (via
        :meth:`set_method`) takes effect without rebuilding analytical
        computations or graph loop edges.
        """
        return self._estimator.pdf(x)

    def _cdf(self, x: NDArray[np.float64], **_options: Any) -> NDArray[np.float64]:
        """Indirection: always reads the current :attr:`estimator`. See :meth:`_pdf`."""
        return self._estimator.cdf(x)

    def _build_analytical_computations(
        self,
    ) -> dict[str, AnalyticalComputation[Any, Any]]:
        """
        Single source of truth for the analytical PDF/CDF entries.

        .. warning::
            **The key order is load-bearing — keep CDF first.**

            When a target has no analytical self-loop of its own (PPF here),
            :meth:`DefaultComputationStrategy._build_plan` walks
            ``analytical_computations`` and takes the *first* key from which
            a path to the target exists — insertion order, not the shortest
            path.  With PDF first the resolver settles on
            ``pdf -> cdf -> ppf`` and rebuilds the CDF by numerical quadrature
            (~1000 quad calls per fit) even though this distribution already
            exposes an exact CDF from the estimator.  With CDF first it picks
            ``cdf -> ppf`` and uses that exact CDF directly.

            Measured on a 1000-point Gaussian KDE: 987 ms vs 24 ms to fit the
            PPF, for quantiles that agree to ~1e-14 (tens of ULP) — the extra
            work buys nothing.  The ordering is pinned by
            ``test_ppf_plan_starts_from_the_analytical_cdf``.

            This is a local workaround for a general gap in the resolver,
            which ought to prefer the shortest path over insertion order;
            once that is fixed upstream this ordering stops mattering.
        """
        result: dict[str, AnalyticalComputation[Any, Any]] = {
            CharacteristicName.CDF: AnalyticalComputation(
                target=CharacteristicName.CDF,
                func=cast(ComputationFunc[Any, Any], self._cdf),
            ),
            CharacteristicName.PDF: AnalyticalComputation(
                target=CharacteristicName.PDF,
                func=cast(ComputationFunc[Any, Any], self._pdf),
            ),
        }
        return result

    def _compute_tabulation_domain(self) -> tuple[float, float] | None:
        """
        Derive the interval on which tabulating the CDF is worthwhile.

        Covers the observed data plus a tail allowance of a few standard
        deviations — the region where a Gaussian kernel still carries mass.

        Returns ``None`` when no usable interval comes out.  A ``None`` domain
        makes the tabulated ``cdf -> ppf`` graph edge inapplicable, so the
        characteristic graph falls back to the general-purpose bisection
        fitter instead.

        No clamping to the support happens here: :meth:`_reject_bounded_support`
        has already ruled out finite bounds, so there is nothing to clamp
        against.  Reinstate it alongside any boundary-corrected estimator that
        lifts that restriction.  :meth:`validate_sample` likewise guarantees a
        1-D, finite, non-constant sample, so the spread below is positive.
        """
        sample = self._sample
        margin = _TABULATION_TAIL_STD * float(sample.std(ddof=0))
        lo = float(sample.min()) - margin
        hi = float(sample.max()) + margin

        return (lo, hi) if hi > lo else None

    @property
    def tabulation_domain(self) -> tuple[float, float] | None:
        """
        Interval on which the CDF may be tabulated to build a fast PPF.

        Read by the characteristic graph: the tabulated ``cdf -> ppf`` edge
        is only applicable to distributions that declare a finite domain.
        ``None`` means "no useful domain", which routes PPF resolution to
        the general-purpose bisection fitter.
        """
        return self._tabulation_domain

    @property
    def data(self) -> NDArray[np.float64]:
        """The original data sample used to fit this distribution."""
        return self._sample

    @property
    def method(self) -> EmpiricalMethod:
        """The empirical method currently configured on this distribution."""
        return self._method

    @property
    def estimator(self) -> FittedEmpirical:
        """
        The estimator currently fitted to :attr:`data`.

        Exposed as a public, read-only handle for two reasons.  It lets callers
        inspect what the method actually produced (a fitted
        ``scipy.stats.gaussian_kde``, for the built-in method), and — the part
        the library itself relies on — it gives cache-invalidating collaborators
        a supported way to tell one fit from another:
        :class:`~pysatl_core.distributions.empirical.strategy.EmpiricalComputationStrategy`
        compares this value by identity (``is``) to notice that
        :meth:`set_method` swapped the fit underneath it.

        The object is replaced, never mutated in place, so an ``is``-comparison
        is a reliable staleness signal.  Rebinding it is deliberately not
        offered: :meth:`set_method` also has to reset the sampler, which holds
        state derived from the old estimator, so a bare setter here would leave
        the distribution half-updated.
        """
        return self._estimator

    def with_method(self, method: EmpiricalMethod) -> EmpiricalDistribution:
        """
        Return a clone of this distribution with a different empirical method.

        The clone refits the new method on the same sample (shared array, no copy).
        Strategies are deep-copied like in any other ``with_*`` clone, so the
        clone is independent of whatever the original memoises later.  The
        sampler does start empty (see
        ``DefaultUnuranSamplingStrategy.__deepcopy__``); the computation caches
        are *copied* and then dropped on the clone's first query, when
        :class:`EmpiricalComputationStrategy` notices the refitted estimator is
        a different object.  Either way, no fit made for the old method is ever
        served for the new one.

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
        cache automatically when it notices that :attr:`estimator` no longer
        returns the object it last saw.

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
        # Depends only on the sample, carried over unchanged, so the parent's
        # value is reused rather than recomputed.
        clone._tabulation_domain = self._tabulation_domain
        return clone


__all__ = [
    "EmpiricalDistribution",
    "EmpiricalMethod",
    "FittedEmpirical",
    "ScipyGaussianKde",
]
