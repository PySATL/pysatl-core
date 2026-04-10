from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov, Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import integrate as _sp_integrate, optimize as _sp_optimize

from pysatl_core.distributions.computation import FittedComputationMethod
from pysatl_core.distributions.fitters.base import FitterDescriptor, FitterOption
from pysatl_core.distributions.fitters.helpers import (
    estimate_support_bounds,
    maybe_unwrap_scalar,
    resolve,
)
from pysatl_core.types import CharacteristicName, NumericArray

if TYPE_CHECKING:
    from pysatl_core.distributions.distribution import Distribution


def fit_pdf_to_cdf_1C(
    distribution: Distribution,
    /,
    **kwargs: Any,
) -> FittedComputationMethod[NumericArray, NumericArray]:
    """
    Fit a ``pdf -> cdf`` conversion via segment-wise numerical integration.

    Parameters
    ----------
    distribution : Distribution
        Must expose a ``pdf`` characteristic.
    **kwargs : Any
        ``limit`` (int, default 200) — max ``quad`` subdivisions.

    Returns
    -------
    FittedComputationMethod[NumericArray, NumericArray]
        Array-semantic ``cdf`` callable.
    """
    opts = FITTER_PDF_TO_CDF_1C.resolve_options(kwargs)
    limit: int = opts["limit"]

    pdf_func = resolve(distribution, CharacteristicName.PDF)

    def _cdf(x: NumericArray, **options: Any) -> NumericArray:
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))

        if x_arr.size == 0:
            return maybe_unwrap_scalar(x_arr.copy())

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

        return maybe_unwrap_scalar(result)

    return FittedComputationMethod[NumericArray, NumericArray](
        target=CharacteristicName.CDF,
        sources=[CharacteristicName.PDF],
        func=_cdf,  # type: ignore[arg-type]
    )


FITTER_PDF_TO_CDF_1C = FitterDescriptor(
    name="pdf_to_cdf_1C",
    target=CharacteristicName.CDF,
    sources=[CharacteristicName.PDF],
    fitter=fit_pdf_to_cdf_1C,
    options=(
        FitterOption(
            name="limit",
            type=int,
            default=200,
            description="Maximum number of quad subdivisions per integral.",
            validate=lambda v: v > 0,
        ),
    ),
    tags=frozenset({"continuous", "univariate"}),
    priority=0,
    description="PDF -> CDF via segment-wise scipy.integrate.quad with cumsum.",
)


def fit_cdf_to_pdf_1C(
    distribution: Distribution,
    /,
    **kwargs: Any,
) -> FittedComputationMethod[NumericArray, NumericArray]:
    """
    Fit a ``cdf -> pdf`` conversion via five-point central finite difference.

    Parameters
    ----------
    distribution : Distribution
        Must expose an array-semantic ``cdf``.
    **kwargs : Any
        ``h`` (float, default 1e-5) — finite-difference step size.

    Returns
    -------
    FittedComputationMethod[NumericArray, NumericArray]
        Array-semantic ``pdf`` callable.
    """
    opts = FITTER_CDF_TO_PDF_1C.resolve_options(kwargs)
    h: float = opts["h"]

    cdf_func = resolve(distribution, CharacteristicName.CDF)

    def _pdf(x: NumericArray, **options: Any) -> NumericArray:
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))

        cdf_ph1 = np.asarray(cdf_func(x_arr + h, **options), dtype=float)
        cdf_mh1 = np.asarray(cdf_func(x_arr - h, **options), dtype=float)
        cdf_ph2 = np.asarray(cdf_func(x_arr + 2.0 * h, **options), dtype=float)
        cdf_mh2 = np.asarray(cdf_func(x_arr - 2.0 * h, **options), dtype=float)

        derivative = (-cdf_ph2 + 8.0 * cdf_ph1 - 8.0 * cdf_mh1 + cdf_mh2) / (12.0 * h)
        result = np.clip(derivative, 0.0, None)
        return maybe_unwrap_scalar(result)

    return FittedComputationMethod[NumericArray, NumericArray](
        target=CharacteristicName.PDF,
        sources=[CharacteristicName.CDF],
        func=_pdf,  # type: ignore[arg-type]
    )


FITTER_CDF_TO_PDF_1C = FitterDescriptor(
    name="cdf_to_pdf_1C",
    target=CharacteristicName.PDF,
    sources=[CharacteristicName.CDF],
    fitter=fit_cdf_to_pdf_1C,
    options=(
        FitterOption(
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
    tags=frozenset({"continuous", "univariate"}),
    priority=0,
    description="CDF -> PDF via five-point central finite difference.",
)


def fit_cdf_to_ppf_1C(
    distribution: Distribution,
    /,
    **kwargs: Any,
) -> FittedComputationMethod[NumericArray, NumericArray]:
    """
    Fit a ``cdf -> ppf`` conversion via vectorised bisection.

    Parameters
    ----------
    distribution : Distribution
        Must expose an array-semantic ``cdf``.
    **kwargs : Any
        ``max_iter`` (int, 60), ``x_tol`` (float, 1e-10),
        ``eps`` (float, 1e-6), ``x0`` (float, 0.0).

    Returns
    -------
    FittedComputationMethod[NumericArray, NumericArray]
        Array-semantic ``ppf`` callable.
    """
    opts = FITTER_CDF_TO_PPF_1C.resolve_options(kwargs)
    max_iter: int = opts["max_iter"]
    x_tol: float = opts["x_tol"]
    eps: float = opts["eps"]
    x0: float = opts["x0"]

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

        return maybe_unwrap_scalar(result)

    return FittedComputationMethod[NumericArray, NumericArray](
        target=CharacteristicName.PPF,
        sources=[CharacteristicName.CDF],
        func=_ppf,  # type: ignore[arg-type]
    )


FITTER_CDF_TO_PPF_1C = FitterDescriptor(
    name="cdf_to_ppf_1C",
    target=CharacteristicName.PPF,
    sources=[CharacteristicName.CDF],
    fitter=fit_cdf_to_ppf_1C,
    options=(
        FitterOption(
            name="max_iter",
            type=int,
            default=60,
            description="Maximum bisection iterations.",
            validate=lambda v: v > 0,
        ),
        FitterOption(
            name="x_tol",
            type=float,
            default=1e-10,
            description="Early-stop tolerance on bracket width.",
            validate=lambda v: v > 0,
        ),
        FitterOption(
            name="eps",
            type=float,
            default=1e-6,
            description="Tail probability threshold for bound estimation.",
            validate=lambda v: 0 < v < 0.5,
        ),
        FitterOption(
            name="x0", type=float, default=0.0, description="Starting point for bound search."
        ),
    ),
    tags=frozenset({"continuous", "univariate"}),
    priority=0,
    description="CDF -> PPF via vectorised bisection with exponential bound search.",
)


def fit_ppf_to_cdf_1C(
    distribution: Distribution,
    /,
    **kwargs: Any,
) -> FittedComputationMethod[NumericArray, NumericArray]:
    """
    Fit a ``ppf -> cdf`` conversion via root inversion (``scipy.optimize.brentq``).

    Parameters
    ----------
    distribution : Distribution
        Must expose an array-semantic ``ppf``.
    **kwargs : Any
        ``q_lowest`` (float, 1e-12), ``q_highest`` (float, 1 - 1e-12),
        ``max_iter`` (int, 256).

    Returns
    -------
    FittedComputationMethod[NumericArray, NumericArray]
        Array-semantic ``cdf`` callable.
    """
    opts = FITTER_PPF_TO_CDF_1C.resolve_options(kwargs)
    q_lowest: float = opts["q_lowest"]
    q_highest: float = opts["q_highest"]
    max_iter: int = opts["max_iter"]

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
                        _sp_optimize.brentq(f, q_lowest, q_highest, maxiter=max_iter)  # type: ignore[arg-type]
                    )
                except ValueError:
                    return float("nan")

            result[interior] = np.clip(np.frompyfunc(_single, 1, 1)(x_in).astype(float), 0.0, 1.0)

        return maybe_unwrap_scalar(result)

    return FittedComputationMethod[NumericArray, NumericArray](
        target=CharacteristicName.CDF,
        sources=[CharacteristicName.PPF],
        func=_cdf,  # type: ignore[arg-type]
    )


FITTER_PPF_TO_CDF_1C = FitterDescriptor(
    name="ppf_to_cdf_1C",
    target=CharacteristicName.CDF,
    sources=[CharacteristicName.PPF],
    fitter=fit_ppf_to_cdf_1C,
    options=(
        FitterOption(
            name="q_lowest",
            type=float,
            default=1e-12,
            description="Left bracket for root search.",
            validate=lambda v: 0 < v < 1,
        ),
        FitterOption(
            name="q_highest",
            type=float,
            default=1.0 - 1e-12,
            description="Right bracket for root search.",
            validate=lambda v: 0 < v < 1,
        ),
        FitterOption(
            name="max_iter",
            type=int,
            default=256,
            description="Maximum brentq iterations per point.",
            validate=lambda v: v > 0,
        ),
    ),
    tags=frozenset({"continuous", "univariate"}),
    priority=0,
    description="PPF -> CDF via root inversion (scipy.optimize.brentq).",
)
