from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable
from math import isfinite
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from mypy_extensions import KwArg
from scipy import (
    integrate as _sp_integrate,
    optimize as _sp_optimize,
)

from pysatl_core.distributions.computation import FittedComputationMethod
from pysatl_core.distributions.support import (
    DiscreteSupport,
    ExplicitTableDiscreteSupport,
    IntegerLatticeDiscreteSupport,
)
from pysatl_core.types import CharacteristicName

if TYPE_CHECKING:
    from typing import Any

    from pysatl_core.distributions.distribution import Distribution
    from pysatl_core.types import GenericCharacteristicName, ScalarFunc


def _resolve(distribution: Distribution, name: GenericCharacteristicName) -> ScalarFunc:
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
        fn = distribution.query_method(name)
    except AttributeError as e:
        raise RuntimeError(
            "Distribution must provide computation_strategy.querry_method(name, distribution)."
        ) from e

    def _wrap(x: float, **kwargs: Any) -> float:
        return float(fn(x, **kwargs))

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
        Monotone Characteristic.CDF in ``[-inf, +inf] -> [0, 1]``.
    most_left : bool, default False
        If ``True``, return the leftmost quantile for flat Characteristic.CDF plateaus.
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
        Optional tolerance in Characteristic.CDF values to stop early when the bracket is flat.
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

    def _ppf(q: float, **kwargs: Any) -> float:
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
    distribution: Distribution, /, **kwargs: Any
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
    pdf_func = _resolve(distribution, CharacteristicName.PDF)

    def _cdf(x: float, **options: Any) -> float:
        val, _ = _sp_integrate.quad(
            lambda t: float(pdf_func(t, **options)), float("-inf"), x, limit=200
        )
        return float(np.clip(val, 0.0, 1.0))

    cdf_func = cast(Callable[[float, KwArg(Any)], float], _cdf)
    return FittedComputationMethod[float, float](
        target=CharacteristicName.CDF, sources=[CharacteristicName.PDF], func=cdf_func
    )


def fit_cdf_to_pdf_1C(
    distribution: Distribution, /, **kwargs: Any
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
    cdf_func = _resolve(distribution, CharacteristicName.CDF)

    def _pdf(x: float, **options: Any) -> float:
        def wrapped_cdf(t: float) -> float:
            return cdf_func(t, **options)

        d = _num_derivative(wrapped_cdf, x, h=1e-5)
        return float(max(d, 0.0))

    pdf_func = cast(Callable[[float, KwArg(Any)], float], _pdf)
    return FittedComputationMethod[float, float](
        target=CharacteristicName.PDF, sources=[CharacteristicName.CDF], func=pdf_func
    )


def fit_cdf_to_ppf_1C(
    distribution: Distribution, /, **options: Any
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
    cdf_func = _resolve(distribution, CharacteristicName.CDF)

    def cdf_with_options(x: float) -> float:
        return cdf_func(x, **options)

    ppf_func = _ppf_brentq_from_cdf(cdf_with_options, **options)

    def _ppf(q: float, **kwargs: Any) -> float:
        return ppf_func(q)

    ppf_cast = cast(Callable[[float, KwArg(Any)], float], _ppf)
    return FittedComputationMethod[float, float](
        target=CharacteristicName.PPF, sources=[CharacteristicName.CDF], func=ppf_cast
    )


def fit_ppf_to_cdf_1C(
    distribution: Distribution, /, **_: Any
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
    ppf_func = _resolve(distribution, CharacteristicName.PPF)

    def _cdf(x: float, **options: Any) -> float:
        if not isfinite(x):
            return 0.0 if x == float("-inf") else 1.0

        def f(q: float) -> float:
            return float(ppf_func(q, **options) - x)

        lo, hi = 1e-12, 1.0 - 1e-12
        flo, fhi = f(lo), f(hi)
        if flo > 0.0:
            return 0.0
        if fhi < 0.0:
            return 1.0
        q = float(_sp_optimize.brentq(f, lo, hi, maxiter=256))  # type: ignore[arg-type]
        return float(np.clip(q, 0.0, 1.0))

    cdf_func = cast(Callable[[float, KwArg(Any)], float], _cdf)
    return FittedComputationMethod[float, float](
        target=CharacteristicName.CDF, sources=[CharacteristicName.PPF], func=cdf_func
    )


# --- Discrete fitters: pmf <-> cdf (1D) --------------------------------------


def fit_pmf_to_cdf_1D(
    distribution: Distribution, /, **_: Any
) -> FittedComputationMethod[float, float]:
    """
    Build Characteristic.CDF from Characteristic.PMF on a discrete support by partial summation.

    The behaviour depends on the kind of discrete support:

    * For table-like supports and left-bounded integer lattices, the Characteristic.CDF is
      constructed as a prefix sum over all support points ``k <= x``.
    * For right-bounded integer lattices (support extends to ``-inf``), the Characteristic.CDF
      is computed via a *tail* sum:

          Characteristic.CDF(x) = 1 - sum_{k > x} pmf(k),

      which only involves finitely many points.
    * Two-sided infinite integer lattices are not supported by this fitter —
      a numerically truncated algorithm would require additional configuration
      and is left for future work.
    """
    support = distribution.support
    if support is None or not isinstance(support, DiscreteSupport):
        raise RuntimeError("Discrete support is required for pmf->cdf.")

    pmf_func = _resolve(distribution, CharacteristicName.PMF)

    # Special case: right-bounded integer lattice
    if isinstance(support, IntegerLatticeDiscreteSupport):
        # Two-sided infinite lattice: exact pmf->cdf is not feasible without
        # additional truncation policy.
        if not support.is_left_bounded and not support.is_right_bounded:
            raise RuntimeError(
                "pmf->cdf for a two-sided infinite integer lattice is not supported "
                "by the generic fitter. Provide an analytical Characteristic.CDF or a "
                "custom fitter."
            )

        # Right-bounded, left-unbounded: use tail summation.
        if not support.is_left_bounded and support.max_k is not None:
            max_k = support.max_k

            def _cdf(x: float, **kwargs: Any) -> float:
                # Everything to the right of the upper bound has Characteristic.CDF == 1.
                if x >= max_k:
                    return 1.0

                import math

                # First integer strictly greater than x
                threshold = int(math.floor(float(x)))
                k = threshold + 1
                if k > max_k:
                    return 1.0

                # Align to the lattice: smallest k >= candidate with k ≡ residue (mod modulus)
                offset = (k - support.residue) % support.modulus
                if offset != 0:
                    k += support.modulus - offset
                if k > max_k:
                    return 1.0

                tail = 0.0
                cur = k
                while cur <= max_k:
                    tail += float(pmf_func(float(cur), **kwargs))
                    cur += support.modulus

                return float(np.clip(1.0 - tail, 0.0, 1.0))

            _cdf_func = cast(Callable[[float, KwArg(Any)], float], _cdf)

            return FittedComputationMethod[float, float](
                target=CharacteristicName.CDF, sources=[CharacteristicName.PMF], func=_cdf_func
            )

    def _cdf_prefix(x: float, **kwargs: Any) -> float:
        s = 0.0
        for k in support.iter_leq(x):
            s += float(pmf_func(float(k), **kwargs))
        return float(np.clip(s, 0.0, 1.0))

    _cdf_func = cast(Callable[[float, KwArg(Any)], float], _cdf_prefix)

    return FittedComputationMethod[float, float](
        target=CharacteristicName.CDF, sources=[CharacteristicName.PMF], func=_cdf_func
    )


def fit_cdf_to_pmf_1D(
    distribution: Distribution, /, **_: Any
) -> FittedComputationMethod[float, float]:
    """
    Extract Characteristic.PMF from Characteristic.CDF on a discrete support as jump sizes.

    Parameters
    ----------
    distribution : Distribution
        Distribution exposing a discrete support on ``.support`` and a scalar
        ``cdf`` via the computation strategy.

    Returns
    -------
    FittedComputationMethod[float, float]
        Fitted ``cdf -> pmf`` conversion.

    Raises
    ------
    RuntimeError
        If the distribution does not expose a discrete support.

    Notes
    -----
    ``pmf(x) = cdf(x) - cdf(prev(x))``, where ``prev(x)`` is the predecessor on
    the support (with ``cdf(prev) := 0`` if no predecessor exists).
    """
    support = distribution.support
    if support is None or not isinstance(support, DiscreteSupport):
        raise RuntimeError("Discrete support is required for cdf->pmf.")
    cdf_func = _resolve(distribution, CharacteristicName.CDF)

    def _pmf(x: float, **kwargs: Any) -> float:
        p = support.prev(x)
        left = 0.0 if p is None else float(cdf_func(float(p), **kwargs))
        right = float(cdf_func(x))
        mass = max(right - left, 0.0)
        return float(np.clip(mass, 0.0, 1.0))

    _pmf_func = cast(Callable[[float, KwArg(Any)], float], _pmf)

    return FittedComputationMethod[float, float](
        target=CharacteristicName.PMF, sources=[CharacteristicName.CDF], func=_pmf_func
    )


# --- DISCRETE (1D): Characteristic.CDF <-> Characteristic.PPF ---------------


def _collect_support_values(support: Any) -> np.ndarray:
    """
    Try to extract a sorted array of support values from a discrete support object.

    For built-in discrete supports:

    * ExplicitTableDiscreteSupport -> uses ``.points`` (already sorted and unique).
    * IntegerLatticeDiscreteSupport with finite bounds -> explicit ``arange`` over the grid.

    For user-defined supports, the following shapes are auto-detected in order:

    * iterable support: ``for x in support``
    * ``support.values()`` / ``support.to_list()``
    * cursor API: ``support.first()`` / ``support.next(x)``

    Returns
    -------
    np.ndarray
        1D float array of sorted support points.

    Raises
    ------
    RuntimeError
        If the support cannot be iterated by any of the strategies or if it
        corresponds to an unbounded integer lattice that cannot be enumerated.
    """
    # 0) Built-in explicit table support: finite, sorted, unique.
    if isinstance(support, ExplicitTableDiscreteSupport):
        xs = np.asarray(support.points, dtype=float)
        return xs

    # 0.1) Built-in integer lattice with finite bounds: finite grid.
    if isinstance(support, IntegerLatticeDiscreteSupport):
        if support.min_k is not None and support.max_k is not None:
            first = support.first()
            if first is None:
                return np.asarray([], dtype=float)
            xs = np.arange(first, support.max_k + 1, support.modulus, dtype=float)
            return xs

        raise RuntimeError(
            "Cannot collect all support values for an unbounded IntegerLatticeDiscreteSupport. "
            "Use lattice-aware fitters instead of _collect_support_values."
        )

    xs_list: list[float] = []

    # 1) Direct iteration
    try:
        it = iter(support)  # may raise TypeError
        xs_list = [float(v) for v in it]
        if xs_list:
            return np.asarray(sorted(xs_list), dtype=float)
    except Exception:
        pass

    # 2) Common containers: values() / to_list()
    for name in ("values", "to_list"):
        if hasattr(support, name):
            try:
                seq = getattr(support, name)()
                xs_list = [float(v) for v in seq]
                if xs_list:
                    return np.asarray(sorted(xs_list), dtype=float)
            except Exception:
                pass

    # 3) Cursor-like API: first(), next(x)
    if hasattr(support, "first") and hasattr(support, "next"):
        try:
            cur = support.first()
            seen: set[Any] = set()
            while cur is not None and cur not in seen:
                seen.add(cur)
                xs_list.append(float(cur))
                cur = support.next(cur)
            if xs_list:
                return np.asarray(sorted(xs_list), dtype=float)
        except Exception:
            pass

    raise RuntimeError("Discrete support must be iterable or expose first()/next().")


def fit_cdf_to_ppf_1D(
    distribution: Distribution, /, **options: Any
) -> FittedComputationMethod[float, float]:
    """
    Fit **discrete** Characteristic.PPF from a resolvable Characteristic.CDF
    and explicit discrete support.

    Semantics
    ---------
    For a given ``q ∈ [0, 1]`` returns the **leftmost** support point ``x`` such that
    ``Characteristic.CDF(x) ≥ q`` (step-quantile).

    Requires
    --------
    distribution.support : discrete support container (iterable or cursor-like).

    Parameters
    ----------
    distribution : Distribution
    **options : Any
        Unused (kept for a uniform API with continuous fitters).

    Returns
    -------
    FittedComputationMethod[float, float]
        Fitted ``cdf -> ppf`` conversion for discrete 1D distributions.
    """
    support = distribution.support
    if support is None or not isinstance(support, DiscreteSupport):
        raise RuntimeError("Discrete support is required for cdf->ppf.")

    cdf_func = _resolve(distribution, CharacteristicName.CDF)

    xs = _collect_support_values(support)  # sorted float array
    if xs.size == 0:
        raise RuntimeError("Discrete support is empty.")

    # Pre-compute Characteristic.CDF on support and enforce monotonicity (safety against FP noise)
    cdf_vals = np.asarray([float(cdf_func(float(x))) for x in xs], dtype=float)
    cdf_vals = np.clip(np.maximum.accumulate(cdf_vals), 0.0, 1.0)

    def _ppf(q: float, **kwargs: Any) -> float:
        if not isfinite(q):
            return float("nan")
        q = float(q)
        if q <= 0.0:
            return float(xs[0])
        if q >= 1.0:
            return float(xs[-1])
        idx = int(np.searchsorted(cdf_vals, q, side="left"))
        if idx >= xs.size:
            idx = xs.size - 1
        return float(xs[idx])

    _ppf_func = cast(Callable[[float, KwArg(Any)], float], _ppf)

    return FittedComputationMethod[float, float](
        target=CharacteristicName.PPF, sources=[CharacteristicName.CDF], func=_ppf_func
    )


def fit_ppf_to_cdf_1D(
    distribution: Distribution, /, **options: Any
) -> FittedComputationMethod[float, float]:
    """
    Fit **discrete** Characteristic.CDF using only a resolvable Characteristic.PPF
    via bisection on ``q``.

    Semantics
    ---------
    ``Characteristic.CDF(x) = sup { q ∈ [0,1] : Characteristic.PPF(q) ≤ x }``

    We implement this as a monotone predicate on ``q``:
      ``f(q) := (Characteristic.PPF(q) ≤ x)``, and find the largest ``q`` with ``f(q) = True``.

    Parameters
    ----------
    distribution : Distribution
    **options : Any
        Optional tuning:
        - q_tol : float, default 1e-12
        - max_iter : int, default 100

    Returns
    -------
    FittedComputationMethod[float, float]
        Fitted ``ppf -> cdf`` conversion for discrete 1D distributions.
    """
    ppf_func = _resolve(distribution, CharacteristicName.PPF)
    q_tol: float = float(options.get("q_tol", 1e-12))
    max_iter: int = int(options.get("max_iter", 100))

    # Quick edge probes (robust to weird Characteristic.PPF endpoints)
    try:
        p0 = float(ppf_func(0.0))
    except Exception:
        p0 = float("-inf")
    try:
        p1 = float(ppf_func(1.0 - 1e-15))
    except Exception:
        p1 = float("inf")

    def _cdf(x: float, **kwargs: Any) -> float:
        if not isfinite(x):
            return float("nan")
        # Hard clamps from endpoint probes
        if x < p0:
            return 0.0
        if x >= p1:
            return 1.0

        lo, hi = 0.0, 1.0
        it = 0
        while hi - lo > q_tol and it < max_iter:
            it += 1
            mid = 0.5 * (lo + hi)
            try:
                y = float(ppf_func(mid, **kwargs))
            except Exception:
                # If Characteristic.PPF fails at mid, shrink conservatively towards lo
                hi = mid
                continue
            if y <= x:
                lo = mid  # still True region
            else:
                hi = mid  # crossed threshold
        return float(np.clip(lo, 0.0, 1.0))

    _cdf_func = cast(Callable[[float, KwArg(Any)], float], _cdf)

    return FittedComputationMethod[float, float](
        target=CharacteristicName.CDF, sources=[CharacteristicName.PPF], func=_cdf_func
    )
