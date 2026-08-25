"""
Continuous-distribution fitters (1C).

Provides fitter functions and their descriptors for converting between
continuous distribution characteristics (PDF, CDF, PPF).

Option taxonomy used here
-------------------------
``CharacteristicOption``
    Parameters that are intrinsic to the *characteristic* being computed and
    therefore shared between the fitter and any evaluator for the same
    characteristic.  They affect the *meaning* of the result and must be
    encoded into the cache key.

    * ``_fit_cdf_to_ppf_1C``: ``eps``, ``x0``  — define the effective support
      bounds used when inverting the CDF; different values yield a different
      PPF.
    * ``_fit_cdf_to_ppf_tabulated_1C``: ``tail_margin`` — pads the tabulation
      domain and thus moves the boundary between the interpolated and the
      exactly solved region.
    * ``_fit_ppf_to_cdf_1C``: ``q_lowest``, ``q_highest`` — bracket for the
      root search; they define the domain of the resulting CDF approximation.

``ComputationOption``
    Parameters that control the *numerical algorithm* only.  They affect
    speed / accuracy but not the semantic meaning of the result.

    * ``_fit_pdf_to_cdf_1C``: ``limit`` — max ``quad`` subdivisions.
    * ``_fit_cdf_to_pdf_1C``: ``h`` — finite-difference step.
    * ``_fit_cdf_to_ppf_1C``: ``max_iter``, ``x_tol`` — bisection parameters.
    * ``_fit_cdf_to_ppf_tabulated_1C``: ``grid_size``, ``max_iter``, ``x_tol``
      — tabulation density and out-of-range solver parameters.
    * ``_fit_ppf_to_cdf_1C``: ``max_iter`` — brentq iterations.
"""

from __future__ import annotations

__author__ = "Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import (
    integrate as _sp_integrate,
    interpolate as _sp_interpolate,
    optimize as _sp_optimize,
)

from pysatl_core.distributions.computations._utils import (
    estimate_support_bounds,
    resolve,
)
from pysatl_core.distributions.computations.computation import FittedComputationMethod
from pysatl_core.distributions.computations.descriptors import FitterDescriptor
from pysatl_core.distributions.computations.options import (
    CharacteristicOption,
    ComputationOption,
)
from pysatl_core.types import CharacteristicName, NumericArray

if TYPE_CHECKING:
    from pysatl_core.distributions.distribution import Distribution


_MIN_TABULATED_CDF_STEP = 1e-12
"""
Smallest CDF increment treated as signal when inverting a tabulated CDF.

Matches the tail resolution the other 1C fitters work at (see the
``q_lowest`` / ``q_highest`` defaults of :func:`_fit_ppf_to_cdf_1C`).
Increments below it come from floating-point noise in a numerically
integrated CDF rather than from probability mass.
"""


def _fit_pdf_to_cdf_1C(
    distribution: Distribution,
    /,
    limit: int = 200,
) -> FittedComputationMethod[NumericArray, NumericArray]:
    """
    Fit a ``pdf -> cdf`` conversion via segment-wise numerical integration.

    Parameters
    ----------
    distribution : Distribution
        Must expose a ``pdf`` characteristic.
    limit : int, default 200
        *(Computation option)* Maximum number of ``quad`` subdivisions per
        integral.

    Returns
    -------
    FittedComputationMethod[NumericArray, NumericArray]
        Array-semantic ``cdf`` callable.
    """
    pdf_func = resolve(distribution, CharacteristicName.PDF)

    def _cdf(x: NumericArray, **options: Any) -> NumericArray:
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))

        if x_arr.size == 0:
            return x_arr.copy()

        def _pdf_scalar(t: float) -> float:
            return float(np.asarray(pdf_func(np.array([t]), **options), dtype=float).flat[0])

        order = np.argsort(x_arr)
        x_sorted = x_arr[order]

        base_val, _ = _sp_integrate.quad(
            _pdf_scalar,
            float("-inf"),
            float(x_sorted[0]),
            limit=limit,
        )

        n = x_sorted.size
        segments = np.empty(n, dtype=float)
        segments[0] = base_val
        for i in range(1, n):
            a, b = float(x_sorted[i - 1]), float(x_sorted[i])
            if a == b:
                segments[i] = 0.0
            else:
                seg_val, _ = _sp_integrate.quad(_pdf_scalar, a, b, limit=limit)
                segments[i] = seg_val

        cdf_sorted = np.clip(np.cumsum(segments), 0.0, 1.0)

        result = np.empty_like(x_arr)
        result[order] = cdf_sorted

        return result

    return FittedComputationMethod[NumericArray, NumericArray](
        target=CharacteristicName.CDF,
        sources=[CharacteristicName.PDF],
        func=_cdf,  # type: ignore[arg-type]
    )


def _build_pdf_to_cdf_1C() -> FitterDescriptor:
    return FitterDescriptor(
        name="pdf_to_cdf_1C",
        target=CharacteristicName.CDF,
        sources=[CharacteristicName.PDF],
        fitter=_fit_pdf_to_cdf_1C,
        characteristic_options=(),
        computation_options=(
            ComputationOption(
                name="limit",
                type=int,
                default=200,
                description="Maximum number of quad subdivisions per integral.",
                validate=lambda v: v > 0,
            ),
        ),
        constraint_tags=frozenset({"continuous", "univariate"}),
        description="PDF -> CDF via segment-wise scipy.integrate.quad with cumsum.",
    )


def _fit_cdf_to_pdf_1C(
    distribution: Distribution,
    /,
    h: float = 1e-5,
) -> FittedComputationMethod[NumericArray, NumericArray]:
    """
    Fit a ``cdf -> pdf`` conversion via five-point central finite difference.

    Parameters
    ----------
    distribution : Distribution
        Must expose an array-semantic ``cdf``.
    h : float, default 1e-5
        *(Computation option)* Finite-difference step size.

    Returns
    -------
    FittedComputationMethod[NumericArray, NumericArray]
        Array-semantic ``pdf`` callable.
    """
    cdf_func = resolve(distribution, CharacteristicName.CDF)

    def _pdf(x: NumericArray, **options: Any) -> NumericArray:
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))

        cdf_ph1 = np.asarray(cdf_func(x_arr + h, **options), dtype=float)
        cdf_mh1 = np.asarray(cdf_func(x_arr - h, **options), dtype=float)
        cdf_ph2 = np.asarray(cdf_func(x_arr + 2.0 * h, **options), dtype=float)
        cdf_mh2 = np.asarray(cdf_func(x_arr - 2.0 * h, **options), dtype=float)

        derivative = (-cdf_ph2 + 8.0 * cdf_ph1 - 8.0 * cdf_mh1 + cdf_mh2) / (12.0 * h)
        result: NumericArray = np.clip(derivative, 0.0, None)
        return result

    return FittedComputationMethod[NumericArray, NumericArray](
        target=CharacteristicName.PDF,
        sources=[CharacteristicName.CDF],
        func=_pdf,  # type: ignore[arg-type]
    )


def _build_cdf_to_pdf_1C() -> FitterDescriptor:
    return FitterDescriptor(
        name="cdf_to_pdf_1C",
        target=CharacteristicName.PDF,
        sources=[CharacteristicName.CDF],
        fitter=_fit_cdf_to_pdf_1C,
        characteristic_options=(),
        computation_options=(
            ComputationOption(
                name="h",
                type=float,
                default=1e-5,
                description=(
                    "Finite-difference step size.  Smaller values improve accuracy "
                    "for smooth CDFs but increase sensitivity to floating-point noise."
                ),
                validate=lambda v: v > 0,
            ),
        ),
        constraint_tags=frozenset({"continuous", "univariate"}),
        description="CDF -> PDF via five-point central finite difference.",
    )


def _fit_cdf_to_ppf_1C(
    distribution: Distribution,
    /,
    max_iter: int = 60,
    x_tol: float = 1e-10,
    eps: float = 1e-6,
    x0: float = 0.0,
) -> FittedComputationMethod[NumericArray, NumericArray]:
    """
    Fit a ``cdf -> ppf`` conversion via vectorised bisection.

    Parameters
    ----------
    distribution : Distribution
        Must expose an array-semantic ``cdf``.
    max_iter : int, default 60
        *(Computation option)* Maximum bisection iterations.
    x_tol : float, default 1e-10
        *(Computation option)* Early-stop tolerance on bracket width.
    eps : float, default 1e-6
        *(Characteristic option)* Tail probability threshold for bound
        estimation.  Different values yield a different effective support and
        therefore a different PPF.
    x0 : float, default 0.0
        *(Characteristic option)* Starting point for bound search.  Affects
        which support bounds are discovered.

    Returns
    -------
    FittedComputationMethod[NumericArray, NumericArray]
        Array-semantic ``ppf`` callable.
    """
    cdf_func = resolve(distribution, CharacteristicName.CDF)

    x_lowest, x_highest = estimate_support_bounds(cdf_func, eps=eps, x0=x0)

    def _ppf(q: NumericArray, **options: Any) -> NumericArray:
        q_arr = np.atleast_1d(np.asarray(q, dtype=float))
        result = np.full_like(q_arr, np.nan)

        result[q_arr <= 0.0] = -np.inf
        result[q_arr >= 1.0] = np.inf
        interior = (q_arr > 0.0) & (q_arr < 1.0)

        if np.any(interior):
            q_in = q_arr[interior]
            lo = np.full_like(q_in, x_lowest)
            hi = np.full_like(q_in, x_highest)

            for _ in range(max_iter):
                if float((hi - lo).max()) < x_tol:
                    break
                mid = 0.5 * (lo + hi)
                cdf_mid = np.asarray(cdf_func(mid, **options), dtype=float)
                below = cdf_mid < q_in
                lo = np.where(below, mid, lo)
                hi = np.where(below, hi, mid)

            result[interior] = 0.5 * (lo + hi)

        return result

    return FittedComputationMethod[NumericArray, NumericArray](
        target=CharacteristicName.PPF,
        sources=[CharacteristicName.CDF],
        func=_ppf,  # type: ignore[arg-type]
    )


def _build_cdf_to_ppf_1C() -> FitterDescriptor:
    return FitterDescriptor(
        name="cdf_to_ppf_1C",
        target=CharacteristicName.PPF,
        sources=[CharacteristicName.CDF],
        fitter=_fit_cdf_to_ppf_1C,
        characteristic_options=(
            CharacteristicOption(
                name="eps",
                type=float,
                default=1e-6,
                description=(
                    "Tail probability threshold for support bound estimation.  "
                    "Affects the effective domain of the PPF — different values "
                    "yield a different result."
                ),
                validate=lambda v: 0 < v < 0.5,
            ),
            CharacteristicOption(
                name="x0",
                type=float,
                default=0.0,
                description=(
                    "Starting point for exponential bound search.  "
                    "Affects which support bounds are discovered."
                ),
            ),
        ),
        computation_options=(
            ComputationOption(
                name="max_iter",
                type=int,
                default=60,
                description="Maximum bisection iterations.",
                validate=lambda v: v > 0,
            ),
            ComputationOption(
                name="x_tol",
                type=float,
                default=1e-10,
                description="Early-stop tolerance on bracket width.",
                validate=lambda v: v > 0,
            ),
        ),
        constraint_tags=frozenset({"continuous", "univariate"}),
        description="CDF -> PPF via vectorised bisection with exponential bound search.",
    )


def _fit_cdf_to_ppf_tabulated_1C(
    distribution: Distribution,
    /,
    tail_margin: float = 0.0,
    grid_size: int = 1025,
    max_iter: int = 60,
    x_tol: float = 1e-10,
) -> FittedComputationMethod[NumericArray, NumericArray]:
    """
    Fit a ``cdf -> ppf`` conversion by tabulating the CDF and inverting it.

    The CDF is evaluated once on a uniform grid spanning the distribution's
    ``tabulation_domain`` and inverted with monotone (PCHIP) interpolation.
    Because the CDF is sampled in a single vectorised call, evaluating the
    resulting PPF on an array of quantiles costs one interpolation instead
    of one root search per point -- a large win for distributions whose CDF
    is expensive (numerical integration, kernel estimators).

    Quantiles falling outside the tabulated probability range are **not**
    approximated by the grid endpoints: they are solved exactly against the
    real CDF by bracket expansion plus bisection.  The fast path therefore
    covers the bulk of the distribution while the tails are solved exactly.

    .. warning::
        One known gap in that split: the leading grid knot is always kept (it
        has no predecessor to test the increment against), so when the knots
        right after it are dropped as degenerate, the first interpolation
        interval spans the whole filtered-out run.  ``q_lowest`` then claims
        coverage the interpolant cannot deliver, and quantiles inside that
        first interval are interpolated instead of solved.  Measured on a
        Gaussian KDE over 2000 points: quantiles between 3e-169 and 4e-12 come
        back with a CDF off by up to 150 orders of magnitude.  Quantiles above
        the first genuinely usable knot are unaffected.

    Parameters
    ----------
    distribution : Distribution
        Must expose an array-semantic ``cdf`` and a ``tabulation_domain``
        attribute holding a finite ``(lo, hi)`` pair that brackets the
        region of interest.
    tail_margin : float, default 0.0
        *(Characteristic option)* Extra padding added to each side of
        ``tabulation_domain``, in units of its width.  Widening the domain
        moves the boundary between the interpolated and the exactly solved
        region, so different values yield a different PPF approximation.
    grid_size : int, default 1025
        *(Computation option)* Number of CDF tabulation points.
    max_iter : int, default 60
        *(Computation option)* Maximum bracket-expansion and bisection
        iterations used for quantiles outside the tabulated range.
    x_tol : float, default 1e-10
        *(Computation option)* Early-stop tolerance on bracket width for the
        out-of-range solver.

    Returns
    -------
    FittedComputationMethod[NumericArray, NumericArray]
        Array-semantic ``ppf`` callable.

    Raises
    ------
    RuntimeError
        If the distribution does not declare a ``tabulation_domain``.
    ValueError
        If the declared domain is not a finite, non-degenerate interval.
    """
    domain = getattr(distribution, "tabulation_domain", None)
    if domain is None:
        raise RuntimeError(
            "Tabulated CDF -> PPF requires the distribution to expose a "
            "'tabulation_domain' attribute; got None."
        )

    lo, hi = (float(bound) for bound in domain)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        raise ValueError(
            f"tabulation_domain must be a finite interval with hi > lo, got ({lo!r}, {hi!r})."
        )

    padding = tail_margin * (hi - lo)
    lo -= padding
    hi += padding

    cdf_func = resolve(distribution, CharacteristicName.CDF)

    x_grid = np.linspace(lo, hi, grid_size)
    cdf_grid = np.asarray(cdf_func(x_grid), dtype=float)

    # PCHIP needs abscissae that are not merely increasing but separated by a
    # meaningful amount: in the saturated tails consecutive CDF samples can
    # differ by a single ULP while x moves a full grid step, which sends the
    # interpolated slope to infinity.  Such knots are dropped rather than
    # nudged apart by an epsilon — a fabricated slope would map a whole range
    # of quantiles onto essentially arbitrary points.  Quantiles left uncovered
    # are solved exactly below, so dropping costs accuracy nowhere -- with the
    # one exception noted in the warning above: `usable[0]` is unconditionally
    # True, so a dropped run directly after the leading knot is bridged by
    # interpolation instead of being left to the exact solver.
    cdf_grid = np.maximum.accumulate(cdf_grid)
    usable = np.ones(cdf_grid.size, dtype=bool)
    usable[1:] = np.diff(cdf_grid) > _MIN_TABULATED_CDF_STEP
    q_knots = cdf_grid[usable]
    x_knots = x_grid[usable]

    if q_knots.size >= 2:
        interpolator = _sp_interpolate.PchipInterpolator(q_knots, x_knots, extrapolate=False)
        q_lowest, q_highest = float(q_knots[0]), float(q_knots[-1])
    else:
        # Degenerate tabulation (e.g. a CDF that is constant on the whole
        # domain): keep no fast path, every quantile is solved exactly.
        interpolator = None
        q_lowest, q_highest = np.inf, -np.inf

    span = hi - lo

    # Never search outside a declared support: an estimator whose kernel leaks
    # past the boundary (a Gaussian KDE on non-negative data, say) reports a
    # non-zero CDF there, and an unclamped search would happily return a
    # quantile the distribution cannot take.
    support = distribution.support
    support_lo = float(getattr(support, "left", -np.inf))
    support_hi = float(getattr(support, "right", np.inf))

    def _solve_exactly(q_out: NumericArray, **options: Any) -> NumericArray:
        """Bracket-expand around the domain, then bisect against the true CDF."""
        lo_b = np.full_like(q_out, lo)
        hi_b = np.full_like(q_out, hi)

        step = span
        for _ in range(max_iter):
            too_high = np.asarray(cdf_func(lo_b, **options), dtype=float) > q_out
            expandable = too_high & (lo_b > support_lo)
            if not bool(expandable.any()):
                break
            lo_b = np.where(expandable, np.maximum(lo_b - step, support_lo), lo_b)
            step *= 2.0

        step = span
        for _ in range(max_iter):
            too_low = np.asarray(cdf_func(hi_b, **options), dtype=float) < q_out
            expandable = too_low & (hi_b < support_hi)
            if not bool(expandable.any()):
                break
            hi_b = np.where(expandable, np.minimum(hi_b + step, support_hi), hi_b)
            step *= 2.0

        for _ in range(max_iter):
            if float((hi_b - lo_b).max()) < x_tol:
                break
            mid = 0.5 * (lo_b + hi_b)
            below = np.asarray(cdf_func(mid, **options), dtype=float) < q_out
            lo_b = np.where(below, mid, lo_b)
            hi_b = np.where(below, hi_b, mid)

        return 0.5 * (lo_b + hi_b)

    def _ppf(q: NumericArray, **options: Any) -> NumericArray:
        q_arr = np.atleast_1d(np.asarray(q, dtype=float))
        result = np.full_like(q_arr, np.nan)

        result[q_arr <= 0.0] = -np.inf
        result[q_arr >= 1.0] = np.inf
        interior = (q_arr > 0.0) & (q_arr < 1.0)

        interpolated = interior & (q_arr >= q_lowest) & (q_arr <= q_highest)
        if interpolator is not None and np.any(interpolated):
            clipped = np.clip(q_arr[interpolated], q_lowest, q_highest)
            result[interpolated] = interpolator(clipped)

        exact = interior & ~interpolated
        if np.any(exact):
            result[exact] = _solve_exactly(q_arr[exact], **options)

        return result

    return FittedComputationMethod[NumericArray, NumericArray](
        target=CharacteristicName.PPF,
        sources=[CharacteristicName.CDF],
        func=_ppf,  # type: ignore[arg-type]
    )


def _build_cdf_to_ppf_tabulated_1C() -> FitterDescriptor:
    return FitterDescriptor(
        name="cdf_to_ppf_tabulated_1C",
        target=CharacteristicName.PPF,
        sources=[CharacteristicName.CDF],
        fitter=_fit_cdf_to_ppf_tabulated_1C,
        characteristic_options=(
            CharacteristicOption(
                name="tail_margin",
                type=float,
                default=0.0,
                description=(
                    "Extra padding on each side of the declared tabulation "
                    "domain, in units of its width.  Moves the boundary between "
                    "the interpolated and the exactly solved region and therefore "
                    "changes the resulting PPF approximation."
                ),
                validate=lambda v: v >= 0.0,
            ),
        ),
        computation_options=(
            ComputationOption(
                name="grid_size",
                type=int,
                default=1025,
                description=(
                    "Number of CDF tabulation points.  More points improve "
                    "interpolation accuracy at a linear fit-time cost."
                ),
                validate=lambda v: v >= 2,
            ),
            ComputationOption(
                name="max_iter",
                type=int,
                default=60,
                description=(
                    "Maximum bracket-expansion and bisection iterations for "
                    "quantiles outside the tabulated range."
                ),
                validate=lambda v: v > 0,
            ),
            ComputationOption(
                name="x_tol",
                type=float,
                default=1e-10,
                description="Early-stop tolerance on bracket width for the out-of-range solver.",
                validate=lambda v: v > 0,
            ),
        ),
        constraint_tags=frozenset({"continuous", "univariate", "tabulated"}),
        description=(
            "CDF -> PPF via CDF tabulation on a declared domain and monotone "
            "(PCHIP) inversion, with exact bisection outside the tabulated range."
        ),
    )


def _fit_ppf_to_cdf_1C(
    distribution: Distribution,
    /,
    q_lowest: float = 1e-12,
    q_highest: float = 1.0 - 1e-12,
    max_iter: int = 256,
) -> FittedComputationMethod[NumericArray, NumericArray]:
    """
    Fit a ``ppf -> cdf`` conversion via root inversion
    (``scipy.optimize.brentq``).

    Parameters
    ----------
    distribution : Distribution
        Must expose an array-semantic ``ppf``.
    q_lowest : float, default 1e-12
        *(Characteristic option)* Left bracket for root search.  Defines the
        lower bound of the CDF domain approximation.
    q_highest : float, default 1 - 1e-12
        *(Characteristic option)* Right bracket for root search.  Defines the
        upper bound of the CDF domain approximation.  Must be strictly greater
        than *q_lowest*.
    max_iter : int, default 256
        *(Computation option)* Maximum brentq iterations per point.

    Returns
    -------
    FittedComputationMethod[NumericArray, NumericArray]
        Array-semantic ``cdf`` callable.

    Raises
    ------
    ValueError
        If ``q_highest <= q_lowest``.
    """
    if q_highest <= q_lowest:
        raise ValueError(
            f"q_highest must be greater than q_lowest, got q_lowest={q_lowest!r}, "
            f"q_highest={q_highest!r}."
        )
    ppf_func = resolve(distribution, CharacteristicName.PPF)

    def _cdf(x: NumericArray, **options: Any) -> NumericArray:
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        result = np.empty_like(x_arr)

        interior = np.isfinite(x_arr)
        result[~interior] = np.where(x_arr[~interior] < 0, 0.0, 1.0)

        if np.any(interior):
            x_in = x_arr[interior]

            def _single(xi: float) -> float:
                def f(q: float) -> float:
                    return (
                        float(np.asarray(ppf_func(np.array([q]), **options), dtype=float).flat[0])
                        - xi
                    )

                try:
                    return float(
                        _sp_optimize.brentq(f, q_lowest, q_highest, maxiter=max_iter)  # type: ignore[arg-type, unused-ignore]
                    )
                except ValueError:
                    left_bound = float(
                        np.asarray(ppf_func(np.array([q_lowest]), **options), dtype=float).flat[0]
                    )
                    return 0.0 if xi <= left_bound else 1.0

            solved = np.fromiter((_single(float(xi)) for xi in x_in), dtype=float, count=x_in.size)
            result[interior] = np.clip(solved, 0.0, 1.0)

        return result

    return FittedComputationMethod[NumericArray, NumericArray](
        target=CharacteristicName.CDF,
        sources=[CharacteristicName.PPF],
        func=_cdf,  # type: ignore[arg-type]
    )


def _build_ppf_to_cdf_1C() -> FitterDescriptor:
    return FitterDescriptor(
        name="ppf_to_cdf_1C",
        target=CharacteristicName.CDF,
        sources=[CharacteristicName.PPF],
        fitter=_fit_ppf_to_cdf_1C,
        characteristic_options=(
            CharacteristicOption(
                name="q_lowest",
                type=float,
                default=1e-12,
                description=(
                    "Left bracket for root search.  Defines the lower bound of "
                    "the CDF domain approximation — different values yield a "
                    "different result near the left tail."
                ),
                validate=lambda v: 0 < v < 1,
            ),
            CharacteristicOption(
                name="q_highest",
                type=float,
                default=1.0 - 1e-12,
                description=(
                    "Right bracket for root search.  Defines the upper bound of "
                    "the CDF domain approximation — different values yield a "
                    "different result near the right tail."
                ),
                validate=lambda v: 0 < v < 1,
            ),
        ),
        computation_options=(
            ComputationOption(
                name="max_iter",
                type=int,
                default=256,
                description="Maximum brentq iterations per point.",
                validate=lambda v: v > 0,
            ),
        ),
        constraint_tags=frozenset({"continuous", "univariate"}),
        description="PPF -> CDF via root inversion (scipy.optimize.brentq).",
    )


def _build_continuous_descriptors() -> list[FitterDescriptor]:
    """
    Build and return all continuous 1D fitter descriptors (lazy factory).

    Order matters: :meth:`FitterRegistry.find` returns the first descriptor
    whose ``constraint_tags`` are a superset of the requested ones, and the
    tabulated ``cdf -> ppf`` variant carries a superset of the plain
    ``{"continuous", "univariate"}`` tags.  Listing it after
    :func:`_build_cdf_to_ppf_1C` keeps the plain lookup resolving to the
    general-purpose bisection fitter; the tabulated one is reached only by
    asking for the ``"tabulated"`` tag explicitly.
    """
    return [
        _build_pdf_to_cdf_1C(),
        _build_cdf_to_pdf_1C(),
        _build_cdf_to_ppf_1C(),
        _build_ppf_to_cdf_1C(),
        _build_cdf_to_ppf_tabulated_1C(),
    ]


__all__: list[str] = []
