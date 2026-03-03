from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov, Irina Sergeeva"
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
from pysatl_core.types import CharacteristicName, NumericArray

if TYPE_CHECKING:
    from typing import Any

    from pysatl_core.distributions.distribution import Distribution
    from pysatl_core.types import GenericCharacteristicName, ScalarFunc


def _resolve(distribution: Distribution, name: GenericCharacteristicName):
    """
    Parameters
    ----------
    distribution : Distribution
        Source distribution that provides the computation strategy.
    name : GenericCharacteristicName
        Characteristic name to resolve (e.g., ``"pdf"``, ``"cdf"``).

    Returns
    -------
    Callable[[NumericArray, ...], NumericArray]
        Array-semantic callable for the requested characteristic.

    Raises
    ------
    RuntimeError
        If the distribution does not provide a suitable computation strategy.
    """
    try:
        fn = distribution.query_method(name)
    except AttributeError as e:
        raise RuntimeError(
            "Distribution must provide computation_strategy.query_method(name, distribution)."
        ) from e

    _probe = np.array([0.0])
    try:
        result = fn(_probe)
        if np.ndim(result) == 0:
            raise TypeError
    except (TypeError, ValueError): ## TODO: delete wrapper
        _fn_scalar = fn
        def fn(x: NumericArray, **kwargs: Any) -> NumericArray:
            return np.vectorize(lambda xi: _fn_scalar(float(xi), **kwargs))(x)

    return fn


def _collect_support(support: DiscreteSupport) -> np.ndarray:
    """
    Materialise a discrete support into a sorted float64 array.

    For built-in support types the array is constructed directly with NumPy
    (no Python-level iteration).  For user-defined supports a one-time Python
    iteration is unavoidable, but it occurs only at fit-time — never at
    call-time.

    Parameters
    ----------
    support : DiscreteSupport
        A discrete support object.  Recognised types are
        ``ExplicitTableDiscreteSupport``, ``IntegerLatticeDiscreteSupport``
        (finite-bounds only), and any user-defined support that is iterable
        or exposes ``values()`` / ``to_list()`` / ``first()`` + ``next()``.

    Returns
    -------
    np.ndarray
        Sorted 1-D float64 array of all support points.

    Raises
    ------
    RuntimeError
        If the support is a left-unbounded or fully-unbounded
        ``IntegerLatticeDiscreteSupport``, or if the support cannot be
        iterated by any recognised strategy.
    """
    # --- Built-in: explicit table -------------------------------------------
    if isinstance(support, ExplicitTableDiscreteSupport):
        return np.asarray(support.points, dtype=float)

    # --- Built-in: integer lattice ------------------------------------------
    if isinstance(support, IntegerLatticeDiscreteSupport):
        if support.min_k is not None and support.max_k is not None:
            first = support.first()
            if first is None:
                return np.empty(0, dtype=float)
            return np.arange(first, support.max_k + 1, support.modulus, dtype=float)

        # Right-bounded only: we DON'T materialise the infinite left tail.
        # Callers that need tail summation handle this case separately.
        if support.min_k is None and support.max_k is not None:
            raise RuntimeError(
                "Left-unbounded IntegerLatticeDiscreteSupport cannot be fully "
                "materialised. Use _build_tail_table for pmf->cdf tail summation."
            )

        raise RuntimeError(
            "Cannot materialise an unbounded IntegerLatticeDiscreteSupport. "
            "Provide at least one bound."
        )

    xs: list[float] = []

    # 1. Direct iteration
    try:
        xs = [float(v) for v in support]
        if xs:
            return np.array(sorted(xs), dtype=float)
    except Exception:
        pass

    # 2. values() / to_list()
    for attr in ("values", "to_list"):
        if hasattr(support, attr):
            try:
                xs = [float(v) for v in getattr(support, attr)()]
                if xs:
                    return np.array(sorted(xs), dtype=float)
            except Exception:
                pass

    # 3. Cursor API: first() / next(x)
    if hasattr(support, "first") and hasattr(support, "next"):
        try:
            cur = support.first()
            seen: set[Any] = set()
            while cur is not None and cur not in seen:
                seen.add(cur)
                xs.append(float(cur))
                cur = support.next(cur)
            if xs:
                return np.array(sorted(xs), dtype=float)
        except Exception:
            pass

    raise RuntimeError("Discrete support must be iterable or expose first()/next().")

def _build_tail_table(
    support: IntegerLatticeDiscreteSupport,
    pmf_func,
    **options: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a tail-probability table for a right-bounded, left-unbounded integer lattice.

    Evaluates ``pmf`` on the finite grid ``[residue, residue + modulus, ..., max_k]``
    in a single vectorised call, then computes the reverse cumulative sum to obtain
    the survival function at each grid point.

    Parameters
    ----------
    support : IntegerLatticeDiscreteSupport
        Must satisfy ``support.max_k is not None``.
    pmf_func : Callable[[NumericArray, ...], NumericArray]
        Array-semantic PMF callable.
    **options : Any
        Keyword arguments forwarded to ``pmf_func``.

    Returns
    -------
    xs : np.ndarray
        Lattice points from ``residue`` up to ``max_k``, shape ``(m,)``.
    tail_from : np.ndarray
        Tail probability array of shape ``(m + 1,)``.
        ``tail_from[i] = P(X >= xs[i])``, with ``tail_from[0] = 1.0``
        and ``tail_from[m] = 0.0``.  Normalised so that ``tail_from[0] == 1.0``
        to correct for any truncation of the finite grid.
    """
    max_k = support.max_k
    residue = support.residue
    modulus = support.modulus

    xs = np.arange(residue, max_k + 1, modulus, dtype=float)
    if xs.size == 0:
        return np.empty(0, dtype=float), np.array([1.0, 0.0])

    pmf_vals = np.asarray(pmf_func(xs, **options), dtype=float)
    pmf_vals = np.clip(pmf_vals, 0.0, None)

    tail_cumsum = np.concatenate(([0.0], np.cumsum(pmf_vals[::-1])))[::-1]
    total = float(tail_cumsum[0])
    if total > 0.0:
        tail_cumsum = tail_cumsum / total

    return xs, np.clip(tail_cumsum, 0.0, 1.0)


def _estimate_support_bounds(
    cdf_func,
    *,
    eps: float = 1e-6,
    x0: float = 0.0,
    max_steps: int = 100,
) -> tuple[float, float]:
    """
    Estimate the effective support bounds ``[x_lowest, x_highest]`` from a CDF.

    Expands exponentially left and right from ``x0`` until
    ``cdf(x_lowest) <= eps`` and ``cdf(x_highest) >= 1 - eps``.

    Parameters
    ----------
    cdf_func : Callable[[NumericArray], NumericArray]
        Array-semantic CDF callable.
    eps : float, default 1e-6
        Tail probability threshold.
    x0 : float, default 0.0
        Starting point for the search.
    max_steps : int, default 100
        Maximum expansion steps in each direction.

    Returns
    -------
    x_lowest : float
        Left bound where ``cdf(x_lowest) <= eps``.
    x_highest : float
        Right bound where ``cdf(x_highest) >= 1 - eps``.
    """
    def _eval(x: float) -> float:
        return float(np.asarray(cdf_func(np.array([x])), dtype=float).flat[0])

    x_lowest = x0
    step = 1.0
    for _ in range(max_steps):
        if _eval(x_lowest) <= eps:
            break
        x_lowest -= step
        step *= 2.0

    x_highest = x0
    step = 1.0
    for _ in range(max_steps):
        if _eval(x_highest) >= 1.0 - eps:
            break
        x_highest += step
        step *= 2.0

    return x_lowest, x_highest


# --- Continuous fitters (1C) --------------------------------------


def fit_pdf_to_cdf_1C(
    distribution: Distribution, /, **kwargs: Any
) -> FittedComputationMethod[NumericArray, NumericArray]:
    """
    Parameters
    ----------
    distribution : Distribution
        Must expose a ``pdf`` via the computation strategy.  Array semantics
        are not required of the source PDF: ``quad`` calls it with scalar inputs.
    limit : int, keyword, default 200
        Maximum number of ``quad`` subdivisions per integral.

    Returns
    -------
    FittedComputationMethod[NumericArray, NumericArray]
        Fitted ``pdf → cdf`` conversion.  Accepts array inputs; each element
        is integrated independently.
    """
    pdf_func = _resolve(distribution, CharacteristicName.PDF)
    limit = int(kwargs.get("limit", 200))

    def _cdf(x: NumericArray, **options: Any) -> NumericArray:
        x_array = np.atleast_1d(np.asarray(x, dtype=float))
        def _single(xi: float) -> float:
            val, _ = _sp_integrate.quad(
                lambda t: float(pdf_func(np.array([t]), **options).flat[0]),
                float("-inf"), xi,
                limit=limit,
            )
            return float(np.clip(val, 0.0, 1.0))
        result = np.frompyfunc(_single, 1, 1)(x_array).astype(float)
        return cast(NumericArray, result[0] if result.shape == (1,) else result)

    cdf_func = cast(Callable[[NumericArray, KwArg(Any)], NumericArray], _cdf)
    return FittedComputationMethod[NumericArray, NumericArray](
        target=CharacteristicName.CDF,
        sources=[CharacteristicName.PDF],
        func=cdf_func
    )


def fit_cdf_to_pdf_1C(
    distribution: Distribution, /, **kwargs: Any
) -> FittedComputationMethod[NumericArray, NumericArray]:
    """
    Parameters
    ----------
    distribution : Distribution
        Must expose an array-semantic ``cdf``.
    h : float, keyword, default 1e-5
        Finite-difference step size.  Smaller values improve accuracy for
        smooth CDFs but increase sensitivity to floating-point noise.

    Returns
    -------
    FittedComputationMethod[NumericArray, NumericArray]
        Fitted ``cdf → pdf`` conversion with full array semantics.
    """
    cdf_func = _resolve(distribution, CharacteristicName.CDF)
    h = float(kwargs.get("h", 1e-5))
    def _pdf(x: NumericArray, **options: Any) -> NumericArray:
        x_array = np.atleast_1d(np.asarray(x, dtype=float))
        cdf_ph1 = np.asarray(cdf_func(x_array + h, **options), dtype=float)
        cdf_mh1 = np.asarray(cdf_func(x_array - h, **options), dtype=float)
        cdf_ph2 = np.asarray(cdf_func(x_array + 2 * h, **options), dtype=float)
        cdf_mh2 = np.asarray(cdf_func(x_array - 2 * h, **options), dtype=float)

        derivative = (-cdf_ph2 + 8.0 * cdf_ph1 - 8.0 * cdf_mh1 + cdf_mh2) / (12.0 * h)
        result = np.clip(derivative, 0.0, None)
        return cast(NumericArray, result[0] if result.shape == (1,) else result)

    pdf_func = cast(Callable[[NumericArray, KwArg(Any)], NumericArray], _pdf)
    return FittedComputationMethod[NumericArray, NumericArray](
        target=CharacteristicName.PDF,
        sources=[CharacteristicName.CDF],
        func=pdf_func
    )


def fit_cdf_to_ppf_1C(
    distribution: Distribution, /, **kwargs: Any,
) -> FittedComputationMethod[NumericArray, NumericArray]:
    """
    Parameters
    ----------
    distribution : Distribution
        Must expose an array-semantic ``cdf``.
    n_grid : int, keyword, default 1024
        Reserved for future PDF-based bound estimation (currently unused).
    max_iter : int, keyword, default 60
        Maximum bisection iterations.
    x_tol : float, keyword, default 1e-10
        Early-stop tolerance on the bracket width.
    eps : float, keyword, default 1e-6
        Tail probability threshold for support bound estimation.
    x0 : float, keyword, default 0.0
        Starting point for the bound search.

    Returns
    -------
    FittedComputationMethod[NumericArray, NumericArray]
        Fitted ``cdf → ppf`` conversion with full array semantics.
    """
    n_grid = int(kwargs.get("n_grid", 1024))
    max_iter = int(kwargs.get("max_iter", 60))
    x_tol = float(kwargs.get("x_tol", 1e-10))
    eps = float(kwargs.get("eps", 1e-6))
    x0 = float(kwargs.get("x0", 0.0))

    cdf_func = _resolve(distribution, CharacteristicName.CDF)
    x_lowest, x_highest = _estimate_support_bounds(cdf_func, eps=eps, x0=x0)

    def _ppf(q: NumericArray, **options: Any) -> NumericArray:
        q_array = np.atleast_1d(np.asarray(q, dtype=float))
        result = np.full_like(q_array, np.nan)
        result[q_array <= 0.0] = -np.inf
        result[q_array >= 1.0] = np.inf
        interior = (q_array > 0.0) & (q_array < 1.0)
        if np.any(interior):
            q_in = q_array[interior]
            lowest = np.full_like(q_in, x_lowest)
            highest = np.full_like(q_in, x_highest)

            for _ in range(max_iter):
                if (highest - lowest).max() < x_tol:
                    break
                mid = 0.5 * (lowest + highest)
                cdf_mid = np.asarray(cdf_func(mid, **options), dtype=float)
                cond = cdf_mid < q_in
                lowest = np.where(cond, mid, lowest)
                highest = np.where(cond, highest, mid)

            result[interior] = 0.5 * (lowest + highest)

        return cast(NumericArray, result[0] if result.shape == (1,) else result)

    ppf_cast = cast(Callable[[NumericArray, KwArg(Any)], NumericArray], _ppf)
    return FittedComputationMethod[NumericArray, NumericArray](
        target=CharacteristicName.PPF,
        sources=[CharacteristicName.CDF],
        func=ppf_cast,
    )


def fit_ppf_to_cdf_1C(
    distribution: Distribution, /, **kwargs: Any,
) -> FittedComputationMethod[NumericArray, NumericArray]:
    """
    Parameters
    ----------
    distribution : Distribution
        Must expose an array-semantic ``ppf`` via the computation strategy.
    q_lowest : float, keyword, default 1e-12
        Left bracket for the root search.
    q_highest : float, keyword, default 1 - 1e-12
        Right bracket for the root search.
    max_iter : int, keyword, default 256
        Maximum iterations for ``brentq`` per query point.

    Returns
    -------
    FittedComputationMethod[NumericArray, NumericArray]
        Fitted ``ppf → cdf`` conversion.  Accepts array inputs; each element
        is inverted independently via ``brentq``.

    """
    q_lowest = float(kwargs.get("q_lowest", 1e-12))
    q_highest = float(kwargs.get("q_highest", 1.0 - 1e-12))
    max_iter = float(kwargs.get("max_iter", 256))
    ppf_func = _resolve(distribution, CharacteristicName.PPF)

    x_lowest, x_highest = -np.inf, np.inf
    def _cdf(x: NumericArray, **options: Any) -> NumericArray:
        x_array = np.atleast_1d(np.asarray(x, dtype=float))

        result = np.empty_like(x_array)
        result[x_array <= x_lowest] = 0.0
        result[x_array >= x_highest] = 1.0
        interior = (x_array > x_lowest) & (x_array < x_highest)

        if np.any(interior):
            x_in = x_array[interior]

            def _single(xi: float) -> float:
                def f(q: float) -> float:
                    return float(np.asarray(ppf_func(np.array([q]), **options)).flat[0]) - xi
                try:
                    return float(_sp_optimize.brentq(f, q_lowest, q_highest, maxiter=max_iter))
                except ValueError:
                    return float("nan")

            result[interior] = np.clip(np.frompyfunc(_single, 1, 1)(x_in).astype(float), 0.0, 1.0)

        return cast(NumericArray, result[0] if result.shape == (1,) else result)

    cdf_func = cast(Callable[[NumericArray, KwArg(Any)], NumericArray], _cdf)
    return FittedComputationMethod[NumericArray, NumericArray](
        target=CharacteristicName.CDF,
        sources=[CharacteristicName.PPF],
        func=cdf_func
    )


# --- Discrete fitters (1D) --------------------------------------

def fit_pmf_to_cdf_1D(
    distribution: Distribution, /, **kwargs: Any,
) -> FittedComputationMethod[NumericArray, NumericArray]:
    """
    Parameters
    ----------
    distribution : Distribution
        Must expose a discrete support on ``.support`` and an array-semantic
        ``pmf`` via the computation strategy.
    **kwargs : Any
        Forwarded to ``pmf`` at fit-time (e.g. distribution parameters).

    Returns
    -------
    FittedComputationMethod[NumericArray, NumericArray]
        Fitted ``pmf → cdf`` conversion with full array semantics.

    Raises
    ------
    RuntimeError
        If the distribution does not expose a discrete support, if the
        support is empty, or if the support is a two-sided infinite integer
        lattice.

    Notes
    -----
    For right-bounded, left-unbounded ``IntegerLatticeDiscreteSupport`` the
    CDF is computed as ``1 - P(X > x)`` using tail summation.
    For all other finite supports the CDF is the prefix sum of PMF values.
    """
    support = distribution.support
    if support is None or not isinstance(support, DiscreteSupport):
        raise RuntimeError("Discrete support is required for pmf->cdf.")

    pmf_func = _resolve(distribution, CharacteristicName.PMF)

    # --- Right-bounded, left-unbounded lattice: tail summation --------------
    if (
        isinstance(support, IntegerLatticeDiscreteSupport)
        and not support.is_left_bounded
        and support.is_right_bounded
    ):
        xs, tail_from = _build_tail_table(support, pmf_func, **kwargs)
        max_k = float(support.max_k)

        def _cdf_tail(x: NumericArray, **options: Any) -> NumericArray:
            x_array = np.atleast_1d(np.asarray(x, dtype=float))

            idx = np.searchsorted(xs, x_array, side="right")
            result = np.clip(1.0 - tail_from[idx], 0.0, 1.0)
            result[x_array >= max_k] = 1.0

            return cast(NumericArray, result[0] if result.shape == (1,) else result)

        return FittedComputationMethod[NumericArray, NumericArray](
            target=CharacteristicName.CDF,
            sources=[CharacteristicName.PMF],
            func=cast(Callable[[NumericArray, KwArg(Any)], NumericArray], _cdf_tail),
        )

    # --- Two-sided infinite lattice: not supported --------------------------
    if (
        isinstance(support, IntegerLatticeDiscreteSupport)
        and not support.is_left_bounded
        and not support.is_right_bounded
    ):
        raise RuntimeError(
            "pmf->cdf for a two-sided infinite integer lattice is not supported "
            "by the generic fitter. Provide an analytical CDF or a custom fitter."
        )

    # --- General case: finite support ---------------------------------------
    xs = _collect_support(support)
    if xs.size == 0:
        raise RuntimeError("Discrete support is empty.")

    pmf_vals = np.clip(np.asarray(pmf_func(xs, **kwargs), dtype=float), 0.0, None)
    cdf_vals = np.clip(np.maximum.accumulate(np.cumsum(pmf_vals)), 0.0, 1.0)

    def _cdf(x: NumericArray, **options: Any) -> NumericArray:
        x_array = np.atleast_1d(np.asarray(x, dtype=float))

        idx = np.searchsorted(xs, x_array, side="right") - 1
        result = np.where(idx < 0, 0.0, cdf_vals[np.clip(idx, 0, cdf_vals.size - 1)])

        return cast(NumericArray, result[0] if result.shape == (1,) else result)

    return FittedComputationMethod[NumericArray, NumericArray](
        target=CharacteristicName.CDF,
        sources=[CharacteristicName.PMF],
        func=cast(Callable[[NumericArray, KwArg(Any)], NumericArray], _cdf),
    )


def fit_cdf_to_pmf_1D(
    distribution: Distribution, /, **kwargs: Any,
) -> FittedComputationMethod[NumericArray, NumericArray]:
    """
    Parameters
    ----------
    distribution : Distribution
        Must expose a discrete support on ``.support`` and an array-semantic
        ``cdf`` via the computation strategy.
    **kwargs : Any
        Forwarded to ``cdf`` at fit-time.

    Returns
    -------
    FittedComputationMethod[NumericArray, NumericArray]
        Fitted ``cdf → pmf`` conversion with full array semantics.

    Raises
    ------
    RuntimeError
        If the distribution does not expose a discrete support or if the
        support is empty.

    Notes
    -----
    Points that do not belong to the discrete support always return ``0.0``,
    consistent with PMF semantics.
    """
    support = distribution.support
    if support is None or not isinstance(support, DiscreteSupport):
        raise RuntimeError("Discrete support is required for cdf->pmf.")

    cdf_func = _resolve(distribution, CharacteristicName.CDF)

    xs = _collect_support(support)
    if xs.size == 0:
        raise RuntimeError("Discrete support is empty.")

    cdf_vals = np.asarray(cdf_func(xs, **kwargs), dtype=float)
    cdf_vals = np.clip(np.maximum.accumulate(cdf_vals), 0.0, 1.0)

    pmf_vals = np.empty_like(cdf_vals)
    pmf_vals[0] = cdf_vals[0]
    pmf_vals[1:] = np.diff(cdf_vals)
    pmf_vals = np.clip(pmf_vals, 0.0, 1.0)

    def _pmf(x: NumericArray, **options: Any) -> NumericArray:
        x_array = np.atleast_1d(np.asarray(x, dtype=float))
        result = np.zeros_like(x_array)

        idx = np.searchsorted(xs, x_array, side="left")
        in_bounds = (idx >= 0) & (idx < xs.size)

        on_support = in_bounds & (xs[np.clip(idx, 0, xs.size - 1)] == x_array)
        result[on_support] = pmf_vals[idx[on_support]]

        return cast(NumericArray, result[0] if result.shape == (1,) else result)

    pmf_cast = cast(Callable[[NumericArray, KwArg(Any)], NumericArray], _pmf)
    return FittedComputationMethod[NumericArray, NumericArray](
        target=CharacteristicName.PMF,
        sources=[CharacteristicName.CDF],
        func=pmf_cast,
    )


def fit_cdf_to_ppf_1D(
    distribution: Distribution, /, **kwargs: Any,
) -> FittedComputationMethod[NumericArray, NumericArray]:
    """
    Parameters
    ----------
    distribution : Distribution
        Must expose a discrete support on ``.support`` and an array-semantic
        ``cdf`` via the computation strategy.
    **options : Any
        Forwarded to ``cdf`` at fit-time.

    Returns
    -------
    FittedComputationMethod[NumericArray, NumericArray]
        Fitted ``cdf → ppf`` conversion with full array semantics.

    Raises
    ------
    RuntimeError
        If the distribution does not expose a discrete support or if the
        support is empty.
    """
    support = distribution.support
    if support is None or not isinstance(support, DiscreteSupport):
        raise RuntimeError("Discrete support is required for cdf->ppf.")

    cdf_func = _resolve(distribution, CharacteristicName.CDF)

    xs = _collect_support(support)
    if xs.size == 0:
        raise RuntimeError("Discrete support is empty.")

    cdf_vals = np.asarray(cdf_func(xs, **kwargs), dtype=float)
    cdf_vals = np.clip(np.maximum.accumulate(cdf_vals), 0.0, 1.0)

    x_first = float(xs[0])
    x_last = float(xs[-1])

    def _ppf(q: NumericArray, **options: Any) -> NumericArray:
        q_array = np.atleast_1d(np.asarray(q, dtype=float))

        result = np.empty_like(q_array)

        nan_mask = ~np.isfinite(q_array)
        low_mask = (~nan_mask) & (q_array <= 0.0)
        high_mask = (~nan_mask) & (q_array >= 1.0)
        interior = ~nan_mask & ~low_mask & ~high_mask

        result[nan_mask] = np.nan
        result[low_mask] = x_first
        result[high_mask] = x_last

        if np.any(interior):
            q_in = q_array[interior]
            idx = np.searchsorted(cdf_vals, q_in, side="left")
            idx = np.clip(idx, 0, xs.size - 1)
            result[interior] = xs[idx]

        return cast(NumericArray, result[0] if result.shape == (1,) else result)

    ppf_cast = cast(Callable[[NumericArray, KwArg(Any)], NumericArray], _ppf)
    return FittedComputationMethod[NumericArray, NumericArray](
        target=CharacteristicName.PPF,
        sources=[CharacteristicName.CDF],
        func=ppf_cast,
    )


def fit_ppf_to_cdf_1D(
    distribution: Distribution, /, **kwargs: Any,
) -> FittedComputationMethod[NumericArray, NumericArray]:
    """
    Parameters
    ----------
    distribution : Distribution
        Must expose an array-semantic ``ppf`` via the computation strategy.
        No explicit support object is required.
    n_q_grid : int, keyword, default 4096
        Number of q-points used to probe the PPF at fit-time.  Increase if
        the distribution has many closely-spaced support points so that all
        CDF steps are captured.

    Returns
    -------
    FittedComputationMethod[NumericArray, NumericArray]
        Fitted ``ppf → cdf`` conversion with full array semantics.

    """
    n_q_grid = int(kwargs.get("n_q_grid", 4096))

    ppf_func = _resolve(distribution, CharacteristicName.PPF)

    eps = 1.0 / (n_q_grid + 1)
    q_grid = np.linspace(eps, 1.0 - eps, n_q_grid)

    x_grid = np.asarray(ppf_func(q_grid, **kwargs), dtype=float)

    change = np.concatenate(([True], x_grid[1:] != x_grid[:-1]))
    xs_table = x_grid[change]
    cdf_table = q_grid[change]

    next_change_idx = np.where(change)[0]
    right_idx = np.concatenate((next_change_idx[1:] - 1, [n_q_grid - 1]))
    cdf_table = q_grid[right_idx]
    cdf_table = np.clip(np.maximum.accumulate(cdf_table), 0.0, 1.0)

    x_min = float(xs_table[0])
    x_max = float(xs_table[-1])

    def _cdf(x: NumericArray, **options: Any) -> NumericArray:
        x_array = np.atleast_1d(np.asarray(x, dtype=float))

        result = np.empty_like(x_array)

        left_mask = x_array < x_min
        right_mask = x_array >= x_max
        interior = ~left_mask & ~right_mask

        result[left_mask] = 0.0
        result[right_mask] = 1.0

        if np.any(interior):
            xi = x_array[interior]
            idx = np.searchsorted(xs_table, xi, side="right") - 1
            idx = np.clip(idx, 0, cdf_table.size - 1)
            result[interior] = cdf_table[idx]

        return cast(NumericArray, result[0] if result.shape == (1,) else result)

    cdf_cast = cast(Callable[[NumericArray, KwArg(Any)], NumericArray], _cdf)
    return FittedComputationMethod[NumericArray, NumericArray](
        target=CharacteristicName.CDF,
        sources=[CharacteristicName.PPF],
        func=cdf_cast,
    )
