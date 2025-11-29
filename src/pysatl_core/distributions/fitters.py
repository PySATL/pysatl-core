"""
Fitters for Conversions Between PDF, CDF and PPF
================================================

Numerical fitters that construct scalar conversions between univariate
continuous characteristics:

- ``pdf -> cdf`` via numerical integration;
- ``cdf -> pdf`` via numerical differentiation;
- ``cdf -> ppf`` via a bracketing + bisection-like root finder;
- ``ppf -> cdf`` via numerical inversion using a root solver.

The returned objects are :class:`~pysatl_core.distributions.computation.FittedComputationMethod`
instances with scalar callables (``float -> float``).

Notes
-----
- SciPy is used for integration (``quad``) and root finding (``brentq``).
- Small numerical artefacts (e.g., tiny negative derivatives) are clipped to
  non-negative values where appropriate.
"""

__author__ = "Leonid Elkin, Mikhail, Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from math import isfinite
from typing import TYPE_CHECKING, Any

import numpy as np

# SciPy numerical routines (types ignored on import)
from scipy import (
    integrate as _sp_integrate,
    optimize as _sp_optimize,
)

from pysatl_core.distributions.computation import FittedComputationMethod
from pysatl_core.types import GenericCharacteristicName, ScalarFunc

if TYPE_CHECKING:
    from pysatl_core.distributions.distribution import Distribution

PDF = "pdf"
CDF = "cdf"
PPF = "ppf"


def _resolve(distribution: "Distribution", name: GenericCharacteristicName) -> ScalarFunc:
    """
    Resolve a scalar characteristic from the distribution.

    Parameters
    ----------
    distribution : Distribution
        Source distribution that provides the computation strategy.
    name : str
        Characteristic name to resolve (e.g., ``"cdf"``).

    Returns
    -------
    Callable[[float], float]
        Scalar callable for the requested characteristic.

    Raises
    ------
    RuntimeError
        If the distribution does not provide a suitable computation strategy.
    """
    try:
        fn = distribution.computation_strategy.query_method(name, distribution)
    except AttributeError as e:
        raise RuntimeError(
            "Distribution must provide computation_strategy.querry_method(name, distribution)."
        ) from e

    def _wrap(x: float) -> float:
        return float(fn(x))

    return _wrap


def _ppf_brentq_from_cdf(
    cdf: ScalarFunc,
    *,
    most_left: bool = False,
    x0: float = 0.0,
    init_step: float = 1.0,
    expand_factor: float = 2.0,
    max_expand: int = 60,
    x_tol: float = 1e-12,
    y_tol: float = 0.0,
    max_iter: int = 200,
) -> ScalarFunc:
    """
    Build a scalar ``ppf`` from a scalar ``cdf`` using bracket expansion
    and a bisection-like search.

    Parameters
    ----------
    cdf : Callable[[float], float]
        Monotone CDF in ``[-inf, +inf] -> [0, 1]``.
    most_left : bool, default False
        If ``True``, return the leftmost quantile for flat CDF plateaus.
    x0 : float, default 0.0
        Initial bracket center.
    init_step : float, default 1.0
        Initial half-width for the bracket.
    expand_factor : float, default 2.0
        Multiplicative factor for exponential bracket growth.
    max_expand : int, default 60
        Maximum expansions while searching for a valid bracket.
    x_tol : float, default 1e-12
        Absolute tolerance in ``x`` for stopping criterion.
    y_tol : float, default 0.0
        Optional tolerance in CDF values to stop early when the bracket is flat.
    max_iter : int, default 200
        Maximum iterations for the bisection-like refinement.

    Returns
    -------
    Callable[[float], float]
        Scalar ``ppf`` such that ``cdf(ppf(q)) ≈ q``.

    Notes
    -----
    This helper clamps the extreme tail queries: ``q <= 0`` maps to ``-inf``,
    ``q >= 1`` maps to ``+inf``.
    """

    def _expand_bracket(q: float) -> tuple[float, float, float, float]:
        if q <= 0.0:
            return float("-inf"), float("-inf"), 0.0, 0.0
        if q >= 1.0:
            return float("inf"), float("inf"), 1.0, 1.0

        step = init_step
        L = x0 - step
        R = x0 + step
        FL = float(cdf(L))
        FR = float(cdf(R))

        def left_ok(FL: float, FR: float) -> bool:
            return (q > FL) and (q <= FR)

        def right_ok(FL: float, FR: float) -> bool:
            return (q >= FL) and (q < FR)

        def ok(FL: float, FR: float) -> bool:
            return left_ok(FL, FR) if most_left else right_ok(FL, FR)

        for _ in range(max_expand):
            if ok(FL, FR):
                return L, R, FL, FR
            grow_left = not ((q > FL) if most_left else (q >= FL))
            grow_right = not ((q <= FR) if most_left else (q < FR))

            if grow_left:
                step *= expand_factor
                L -= step
                FL = float(cdf(L))
            if grow_right:
                step *= expand_factor
                R += step
                FR = float(cdf(R))

            if y_tol > 0.0:
                if most_left and (q - y_tol >= FL) and (q - y_tol <= FR):
                    return L, R, FL, FR
                if not most_left and (q + y_tol >= FL) and (q + y_tol <= FR):
                    return L, R, FL, FR

        return L, R, FL, FR

    def _ppf(q: float) -> float:
        if q <= 0.0:
            return float("-inf")
        if q >= 1.0:
            return float("inf")

        L, R, FL, FR = _expand_bracket(q)

        if not (isfinite(L) and isfinite(R)):
            return L if q <= 0.0 else R

        it = 0
        while it < max_iter and x_tol * (1.0 + max(abs(L), abs(R))) < (R - L):
            M = 0.5 * (L + R)
            FM = float(cdf(M))

            if most_left:
                if q <= FM:
                    R, FR = M, FM
                else:
                    L, FL = M, FM
            else:
                if q < FM:
                    R, FR = M, FM
                else:
                    L, FL = M, FM

            if y_tol > 0.0 and abs(FR - FL) <= y_tol:
                break

            it += 1

        return R if most_left else L

    return _ppf


def _num_derivative(f: ScalarFunc, x: float, h: float = 1e-5) -> float:
    """
    5-point central numerical derivative used for ``cdf -> pdf``.

    Parameters
    ----------
    f : Callable[[float], float]
        Scalar function.
    x : float
        Evaluation point.
    h : float, default 1e-5
        Step for the stencil.

    Returns
    -------
    float
        Approximated derivative ``f'(x)``.
    """
    if not isfinite(x):
        return float("nan")
    f1 = float(f(x + h))
    f_1 = float(f(x - h))
    f2 = float(f(x + 2 * h))
    f_2 = float(f(x - 2 * h))
    return float((-f2 + 8 * f1 - 8 * f_1 + f_2) / (12.0 * h))


def fit_pdf_to_cdf_1C(
    distribution: "Distribution", /, **_: Any
) -> FittedComputationMethod[float, float]:
    """
    Fit ``cdf`` from an analytical or resolvable ``pdf`` via numerical integration.

    Parameters
    ----------
    distribution : Distribution

    Returns
    -------
    FittedComputationMethod[float, float]
        Fitted ``pdf -> cdf`` conversion.
    """
    pdf_func = _resolve(distribution, PDF)

    def _cdf(x: float) -> float:
        val, _ = _sp_integrate.quad(lambda t: float(pdf_func(t)), float("-inf"), x, limit=200)
        return float(np.clip(val, 0.0, 1.0))

    return FittedComputationMethod[float, float](target=CDF, sources=[PDF], func=_cdf)


def fit_cdf_to_pdf_1C(
    distribution: "Distribution", /, **_: Any
) -> FittedComputationMethod[float, float]:
    """
    Fit ``pdf`` as a clipped numerical derivative of ``cdf``.

    Parameters
    ----------
    distribution : Distribution

    Returns
    -------
    FittedComputationMethod[float, float]
        Fitted ``cdf -> pdf`` conversion.
    """
    cdf_func = _resolve(distribution, CDF)

    def _pdf(x: float) -> float:
        d = _num_derivative(cdf_func, x, h=1e-5)
        return float(max(d, 0.0))

    return FittedComputationMethod[float, float](target=PDF, sources=[CDF], func=_pdf)


def fit_cdf_to_ppf_1C(
    distribution: "Distribution", /, **options: Any
) -> FittedComputationMethod[float, float]:
    """
    Fit ``ppf`` from a resolvable ``cdf`` using a robust bracketing procedure.

    Parameters
    ----------
    distribution : Distribution

    Returns
    -------
    FittedComputationMethod[float, float]
        Fitted ``cdf -> ppf`` conversion.
    """
    cdf_func = _resolve(distribution, CDF)
    ppf_func = _ppf_brentq_from_cdf(cdf_func, **options)
    return FittedComputationMethod[float, float](target=PPF, sources=[CDF], func=ppf_func)


def fit_ppf_to_cdf_1C(
    distribution: "Distribution", /, **_: Any
) -> FittedComputationMethod[float, float]:
    """
    Fit ``cdf`` by numerically inverting a resolvable ``ppf`` with a root solver.

    Parameters
    ----------
    distribution : Distribution

    Returns
    -------
    FittedComputationMethod[float, float]
        Fitted ``ppf -> cdf`` conversion.
    """
    ppf_func = _resolve(distribution, PPF)

    def _cdf(x: float) -> float:
        if not isfinite(x):
            return 0.0 if x == float("-inf") else 1.0

        def f(q: float) -> float:
            return float(ppf_func(q) - x)

        lo, hi = 1e-12, 1.0 - 1e-12
        flo, fhi = f(lo), f(hi)
        if flo > 0.0:
            return 0.0
        if fhi < 0.0:
            return 1.0
        # Anyway will be refactored. If it'll be a need to remove "ignore",
        # use unusual *_args: Any, **_kwargs: Any and change return type to
        # float | np.floating[Any] | np.integer[Any] | np.bool_
        q = float(_sp_optimize.brentq(f, lo, hi, maxiter=256))  # type: ignore[arg-type]
        return float(np.clip(q, 0.0, 1.0))

    return FittedComputationMethod[float, float](target=CDF, sources=[PPF], func=_cdf)
