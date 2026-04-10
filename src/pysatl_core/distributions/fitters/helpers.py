from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov, Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pysatl_core.distributions.support import (
    ExplicitTableDiscreteSupport,
    IntegerLatticeDiscreteSupport,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pysatl_core.distributions.distribution import Distribution
    from pysatl_core.distributions.support import DiscreteSupport
    from pysatl_core.types import GenericCharacteristicName, NumericArray


def resolve(
    distribution: Distribution,
    name: GenericCharacteristicName,
) -> Callable[..., NumericArray]:
    """
    Obtain an array-semantic callable for *name* from *distribution*.

    If the underlying method does not natively accept 1-D arrays, a
    ``np.vectorize`` wrapper is applied transparently.  The probe is a
    single-element array ``[0.0]``.

    Parameters
    ----------
    distribution : Distribution
        Source distribution.
    name : GenericCharacteristicName
        Characteristic to resolve (e.g. ``"pdf"``, ``"cdf"``).

    Returns
    -------
    Callable[..., NumericArray]
        Array-semantic callable ``(x, **options) -> NumericArray``.

    Raises
    ------
    RuntimeError
        If the distribution cannot provide the requested characteristic.
    """
    try:
        fn = distribution.query_method(name)
    except AttributeError as exc:
        raise RuntimeError(
            f"Distribution does not expose characteristic '{name}' "
            "via computation_strategy.query_method()."
        ) from exc

    _probe = np.array([0.0])
    try:
        result = fn(_probe)
        if np.ndim(result) == 0:
            raise TypeError
    except (TypeError, ValueError):
        _fn_scalar = fn

        def _vectorised(x: NumericArray, **kwargs: Any) -> NumericArray:
            return cast("NumericArray", np.vectorize(lambda xi: _fn_scalar(float(xi), **kwargs))(x))

        return _vectorised

    return fn


def collect_support(support: DiscreteSupport) -> np.ndarray:
    """
    Materialise a discrete support into a sorted ``float64`` array.

    Parameters
    ----------
    support : DiscreteSupport
        A discrete support object.

    Returns
    -------
    np.ndarray
        Sorted 1-D ``float64`` array of all support points.

    Raises
    ------
    RuntimeError
        If the support cannot be materialised (e.g. unbounded lattice).
    """
    if isinstance(support, ExplicitTableDiscreteSupport):
        return np.asarray(support.points, dtype=float)

    if isinstance(support, IntegerLatticeDiscreteSupport):
        if support.min_k is not None and support.max_k is not None:
            first = support.first()
            if first is None:
                return np.empty(0, dtype=float)
            return np.arange(first, support.max_k + 1, support.modulus, dtype=float)

        if support.min_k is None and support.max_k is not None:
            raise RuntimeError(
                "Left-unbounded IntegerLatticeDiscreteSupport cannot be fully "
                "materialised.  Use build_tail_table for pmf→cdf tail summation."
            )

        raise RuntimeError(
            "Cannot materialise an unbounded IntegerLatticeDiscreteSupport.  "
            "Provide at least one bound."
        )

    xs: list[float] = []

    try:
        xs = [float(v) for v in support]  # type: ignore[attr-defined]
        if xs:
            return np.array(sorted(xs), dtype=float)
    except Exception:
        pass

    for attr in ("values", "to_list"):
        if hasattr(support, attr):
            try:
                xs = [float(v) for v in getattr(support, attr)()]
                if xs:
                    return np.array(sorted(xs), dtype=float)
            except Exception:
                pass

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


def build_tail_table(
    support: IntegerLatticeDiscreteSupport,
    pmf_func: Callable[..., NumericArray],
    **options: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a tail-probability table for a right-bounded, left-unbounded lattice.

    Evaluates ``pmf`` on the finite grid ``[residue, residue + modulus, …, max_k]``
    in a single vectorised call, then computes the reverse cumulative sum.

    Parameters
    ----------
    support : IntegerLatticeDiscreteSupport
        Must satisfy ``support.max_k is not None``.
    pmf_func : Callable[..., NumericArray]
        Array-semantic PMF callable.
    **options : Any
        Forwarded to *pmf_func*.

    Returns
    -------
    xs : np.ndarray
        Lattice points, shape ``(m,)``.
    tail_from : np.ndarray
        Tail probabilities, shape ``(m + 1,)``.
        ``tail_from[i] = P(X >= xs[i])``, normalised so ``tail_from[0] == 1``.
    """
    max_k = support.max_k
    assert max_k is not None, "build_tail_table requires support.max_k to be set"
    residue = support.residue
    modulus = support.modulus

    xs = np.arange(residue, max_k + 1, modulus, dtype=float)
    if xs.size == 0:
        return np.empty(0, dtype=float), np.array([1.0, 0.0])

    pmf_vals = np.clip(np.asarray(pmf_func(xs, **options), dtype=float), 0.0, None)

    tail_cumsum = np.empty(xs.size + 1, dtype=float)
    tail_cumsum[-1] = 0.0
    np.cumsum(pmf_vals[::-1], out=tail_cumsum[:-1])
    tail_cumsum[:-1] = tail_cumsum[:-1][::-1]

    total = float(tail_cumsum[0])
    if total > 0.0:
        tail_cumsum /= total

    return xs, np.clip(tail_cumsum, 0.0, 1.0)


def estimate_support_bounds(
    cdf_func: Callable[..., NumericArray],
    *,
    eps: float = 1e-6,
    x0: float = 0.0,
    max_steps: int = 100,
) -> tuple[float, float]:
    """
    Estimate effective support bounds ``[lo, hi]`` from a CDF.

    Expands exponentially left and right from *x0* until
    ``cdf(lo) <= eps`` and ``cdf(hi) >= 1 - eps``.

    Parameters
    ----------
    cdf_func : Callable[..., NumericArray]
        Array-semantic CDF callable.
    eps : float, default 1e-6
        Tail probability threshold.
    x0 : float, default 0.0
        Starting point.
    max_steps : int, default 100
        Maximum expansion steps per direction.

    Returns
    -------
    lo : float
        Left bound where ``cdf(lo) <= eps``.
    hi : float
        Right bound where ``cdf(hi) >= 1 - eps``.
    """

    def _eval(x: float) -> float:
        return float(np.asarray(cdf_func(np.array([x])), dtype=float).flat[0])

    lo = x0
    step = 1.0
    for _ in range(max_steps):
        if _eval(lo) <= eps:
            break
        lo -= step
        step *= 2.0

    hi = x0
    step = 1.0
    for _ in range(max_steps):
        if _eval(hi) >= 1.0 - eps:
            break
        hi += step
        step *= 2.0

    return lo, hi


def maybe_unwrap_scalar(result: np.ndarray) -> NumericArray:
    """
    If *result* has shape ``(1,)`` return the scalar element, else return as-is.

    This preserves the convention that scalar inputs produce scalar outputs
    while array inputs produce array outputs.

    Parameters
    ----------
    result : np.ndarray
        1-D result array.

    Returns
    -------
    NumericArray
        Scalar or array.
    """
    if result.shape == (1,):
        return cast("NumericArray", result[0])
    return cast("NumericArray", result)
