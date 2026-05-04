"""
Shared helper utilities for fitter and evaluator implementations.

Provides support resolution, tail-table construction, and support-bound
estimation helpers.
"""

from __future__ import annotations

__author__ = "Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, Any

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

    The callable must accept a 1-D NumPy array and return a NumPy array of the
    same length.  If the underlying method is a scalar function (returns a
    0-dimensional result when given a 1-D array, or raises :class:`TypeError`),
    a :class:`TypeError` is raised immediately.

    .. note::
        Scalar functions are **not** supported.  They are significantly slower
        than array-semantic ones because wrapping them with ``numpy.vectorize``
        incurs a per-element Python call overhead.  Implement characteristic
        functions to accept and return NumPy arrays directly.

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
    TypeError
        If the characteristic function is scalar (does not accept or return
        NumPy arrays).
    """
    fn = distribution.query_method(name)

    _probe = np.array([0.5])
    try:
        result = fn(_probe)
        if np.ndim(result) == 0:
            raise TypeError
    except TypeError:
        raise TypeError(
            f"Characteristic '{name}' of distribution "
            f"'{type(distribution).__name__}' is implemented as a scalar "
            "function.  Scalar functions are not supported — implement the "
            "function to accept and return NumPy arrays directly.  "
            "Scalar functions incur a per-element Python call overhead and "
            "are significantly slower for large inputs."
        ) from None

    return fn


def collect_discrete_support(support: DiscreteSupport) -> np.ndarray:
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
        If the support cannot be materialised (e.g. unbounded lattice) or
        is not a recognised concrete type.
    """
    if isinstance(support, ExplicitTableDiscreteSupport):
        return np.asarray(support.points, dtype=float)

    if isinstance(support, IntegerLatticeDiscreteSupport):
        first = support.first()
        last = support.last()
        if first is not None and last is not None:
            return np.arange(first, last + 1, support.modulus, dtype=float)

        if first is None and last is not None:
            raise RuntimeError(
                "Left-unbounded IntegerLatticeDiscreteSupport cannot be fully "
                "materialised.  Use build_tail_table for pmf→cdf tail summation."
            )

        if first is not None and last is None:
            raise RuntimeError(
                "Right-unbounded IntegerLatticeDiscreteSupport cannot be fully "
                "materialised.  Use build_head_table for pmf→cdf prefix summation."
            )

        raise RuntimeError(
            "Cannot materialise an unbounded IntegerLatticeDiscreteSupport.  "
            "Provide at least one bound."
        )

    raise RuntimeError(
        f"Unsupported DiscreteSupport type: {type(support).__name__!r}.  "
        "Only ExplicitTableDiscreteSupport and IntegerLatticeDiscreteSupport are supported."
    )


def build_tail_table(
    support: IntegerLatticeDiscreteSupport,
    pmf_func: Callable[..., NumericArray],
    *,
    eps: float = 1e-12,
    batch_size: int = 256,
    max_batches: int = 100_000,
    **options: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a tail-probability table for a right-bounded, left-unbounded lattice.

    Walks downward from ``max_point`` in steps of ``modulus``, evaluating the PMF
    in batches, and stops as soon as the remaining left-tail probability
    ``1 - cumulative_sum`` drops below *eps*.  This correctly handles
    left-unbounded supports where mass exists arbitrarily far to the left.

    Parameters
    ----------
    support : IntegerLatticeDiscreteSupport
        Must satisfy ``support.max_k is not None``.
    pmf_func : Callable[..., NumericArray]
        Array-semantic PMF callable.
    eps : float, default 1e-12
        Stopping threshold.  The downward walk continues while
        ``1 - cumulative_sum >= eps``; once the remaining left-tail
        probability falls below *eps* it is considered negligible.
    batch_size : int, default 256
        Number of lattice points evaluated per PMF call.
    max_batches : int, default 100_000
        Hard upper limit on the number of batches to prevent infinite loops
        when the PMF sums to zero or is pathologically flat.
    **options : Any
        Forwarded to *pmf_func*.

    Returns
    -------
    xs : np.ndarray
        Lattice points in ascending order, shape ``(m,)``.
    tail_from : np.ndarray
        Tail probabilities, shape ``(m + 1,)``.
        ``tail_from[i] = P(X >= xs[i])``, with ``tail_from[-1] == 0``.
    """
    max_point = support.last()
    assert max_point is not None, "build_tail_table requires support.max_k to be set"
    modulus = support.modulus

    collected_xs: list[np.ndarray] = []
    collected_pmf: list[np.ndarray] = []
    cumsum: float = 0.0
    x_top = float(max_point)

    for _ in range(max_batches):
        batch_xs = x_top - np.arange(batch_size, dtype=float) * modulus
        batch_pmf = np.clip(np.asarray(pmf_func(batch_xs, **options), dtype=float), 0.0, None)

        batch_cumsum = np.cumsum(batch_pmf)
        stop_mask = (1.0 - (cumsum + batch_cumsum)) < eps
        if np.any(stop_mask):
            cut = int(np.argmax(stop_mask)) + 1
            collected_xs.append(batch_xs[:cut])
            collected_pmf.append(batch_pmf[:cut])
            break

        collected_xs.append(batch_xs)
        collected_pmf.append(batch_pmf)
        cumsum += float(batch_cumsum[-1])
        x_top = float(batch_xs[-1]) - modulus

    if not collected_xs:
        return np.empty(0, dtype=float), np.array([1.0, 0.0])

    xs = np.concatenate(collected_xs[::-1])[::-1].copy()
    pmf_vals = np.concatenate(collected_pmf[::-1])[::-1].copy()

    if xs.size == 0:
        return np.empty(0, dtype=float), np.array([1.0, 0.0])

    tail_cumsum = np.empty(xs.size + 1, dtype=float)
    tail_cumsum[-1] = 0.0
    np.cumsum(pmf_vals[::-1], out=tail_cumsum[:-1])
    tail_cumsum[:-1] = tail_cumsum[:-1][::-1]

    total = float(tail_cumsum[0])
    if total > 0.0:
        tail_cumsum /= total

    return xs, np.clip(tail_cumsum, 0.0, 1.0)


def build_head_table(
    support: IntegerLatticeDiscreteSupport,
    pmf_func: Callable[..., NumericArray],
    *,
    eps: float = 1e-12,
    batch_size: int = 256,
    max_batches: int = 100_000,
    **options: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a CDF table for a left-bounded, right-unbounded lattice.

    Mirrors the support and PMF around zero, delegates to
    :func:`build_tail_table` (which walks downward from the mirrored
    ``max_k``), then flips the result back to the original orientation.

    Parameters
    ----------
    support : IntegerLatticeDiscreteSupport
        Must satisfy ``support.min_k is not None``.
    pmf_func : Callable[..., NumericArray]
        Array-semantic PMF callable.
    eps : float, default 1e-12
        Stopping threshold forwarded to :func:`build_tail_table`.
    batch_size : int, default 256
        Number of lattice points evaluated per PMF call.
    max_batches : int, default 100_000
        Hard upper limit on the number of batches.
    **options : Any
        Forwarded to *pmf_func*.

    Returns
    -------
    xs : np.ndarray
        Lattice points in ascending order, shape ``(m,)``.
    cdf_at : np.ndarray
        CDF values, shape ``(m,)``.
        ``cdf_at[i] = P(X <= xs[i])``.
    """
    min_point = support.first()
    assert min_point is not None, "build_head_table requires support.min_k to be set"

    mirrored_support = IntegerLatticeDiscreteSupport(
        residue=(-support.residue) % support.modulus,
        modulus=support.modulus,
        min_k=None,
        max_k=-min_point,
    )

    def mirrored_pmf(x: NumericArray, **kw: Any) -> NumericArray:
        return pmf_func(-x, **kw, **options)

    xs_mirror, tail_from = build_tail_table(
        mirrored_support,
        mirrored_pmf,
        eps=eps,
        batch_size=batch_size,
        max_batches=max_batches,
    )

    if xs_mirror.size == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    xs = -xs_mirror[::-1].copy()
    m = xs_mirror.size
    cdf_at = tail_from[m - 1 :: -1].copy()  # tail_from[m-1], tail_from[m-2], ..., tail_from[0]

    return xs, np.clip(cdf_at, 0.0, 1.0)


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


__all__ = [
    "resolve",
    "collect_discrete_support",
    "build_tail_table",
    "build_head_table",
    "estimate_support_bounds",
]
