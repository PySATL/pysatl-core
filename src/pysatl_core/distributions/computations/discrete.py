"""
Discrete-distribution fitters (1D).

Provides fitter functions and their descriptors for converting between
discrete distribution characteristics (PMF, CDF, PPF).

Option taxonomy used here
-------------------------
``CharacteristicOption``
    * ``_fit_pmf_to_cdf_1D``, ``_fit_ppf_to_cdf_1D``: ``right_closed`` —
      controls the CDF convention:

      * ``True`` (default): right-closed, standard convention
        ``F(x) = P(ξ ≤ x)``.
      * ``False``: right-open convention ``F⁻(x) = P(ξ < x)``.

      The right-open form is useful when computing the CDF of ``-ξ``:
      ``P(-ξ ≤ -x) = P(ξ ≥ x) = 1 - P(ξ < x) = 1 - F⁻(x)``.

      Because this option changes the *meaning* of the result it is a
      ``CharacteristicOption`` and is encoded into the cache key.

``ComputationOption``
    * ``_fit_ppf_to_cdf_1D``: ``n_q_grid`` — grid resolution for probing the
      PPF at fit-time.  Affects only the accuracy of the table construction,
      not the semantic meaning of the CDF.
"""

from __future__ import annotations

__author__ = "Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from pysatl_core.distributions.computations._utils import (
    build_head_table,
    build_tail_table,
    collect_discrete_support,
    resolve,
)
from pysatl_core.distributions.computations.computation import FittedComputationMethod
from pysatl_core.distributions.computations.descriptors import FitterDescriptor
from pysatl_core.distributions.computations.options import CharacteristicOption, ComputationOption
from pysatl_core.distributions.support import (
    DiscreteSupport,
    IntegerLatticeDiscreteSupport,
)
from pysatl_core.types import CharacteristicName, NumericArray

if TYPE_CHECKING:
    from pysatl_core.distributions.distribution import Distribution


def _require_discrete_support(distribution: Distribution, conversion: str) -> DiscreteSupport:
    """Return distribution support or raise a conversion-specific error."""
    support = distribution.support
    if support is None or not isinstance(support, DiscreteSupport):
        raise RuntimeError(f"Discrete support is required for {conversion}.")
    return support


def _fit_pmf_to_cdf_1D(
    distribution: Distribution,
    /,
    eps: float = 1e-12,
    right_closed: bool = True,
) -> FittedComputationMethod[NumericArray, NumericArray]:
    """
    Fit a ``pmf -> cdf`` conversion for discrete distributions.

    Parameters
    ----------
    distribution : Distribution
        Must expose a discrete ``support`` and a ``pmf`` characteristic.
    eps : float, default 1e-12
        *(Computation option)* Stopping threshold for the tail walk when the
        support is one-sided unbounded.

        * **Right-bounded, left-unbounded**: the downward walk continues while
          ``1 - cumulative_sum >= eps``; once the remaining left-tail
          probability falls below *eps* it is considered negligible.
        * **Left-bounded, right-unbounded**: the upward walk (via mirroring)
          continues while ``1 - cumulative_sum >= eps``; once the remaining
          right-tail probability falls below *eps* it is considered negligible.
    right_closed : bool, default True
        *(Characteristic option)* CDF convention:

        * ``True``: right-closed ``F(x) = P(ξ ≤ x)`` (standard).
        * ``False``: right-open ``F⁻(x) = P(ξ < x)``.

        The right-open form satisfies
        ``1 - F⁻(x) = P(ξ ≥ x)``, which is needed when computing the CDF
        of ``-ξ``.

    Returns
    -------
    FittedComputationMethod[NumericArray, NumericArray]

    Raises
    ------
    RuntimeError
        If the support is missing, empty, or a two-sided infinite lattice.
    """
    support = _require_discrete_support(distribution, "pmf->cdf")
    pmf_func = resolve(distribution, CharacteristicName.PMF)
    side: Literal["left", "right"] = "right" if right_closed else "left"

    if (
        isinstance(support, IntegerLatticeDiscreteSupport)
        and not support.is_left_bounded
        and support.is_right_bounded
    ):
        xs, tail_from = build_tail_table(support, pmf_func, eps=eps)
        max_point = float(support.last())  # type: ignore[arg-type]

        def _cdf_tail(x: NumericArray, **options: Any) -> NumericArray:
            x_arr = np.atleast_1d(np.asarray(x, dtype=float))
            idx = np.searchsorted(xs, x_arr, side=side)
            result: NumericArray = np.clip(np.asarray(1.0 - tail_from[idx], dtype=float), 0.0, 1.0)
            if right_closed:
                result[x_arr >= max_point] = 1.0
            else:
                result[x_arr > max_point] = 1.0
            return result

        return FittedComputationMethod[NumericArray, NumericArray](
            target=CharacteristicName.CDF,
            sources=[CharacteristicName.PMF],
            func=_cdf_tail,  # type: ignore[arg-type]
        )

    if (
        isinstance(support, IntegerLatticeDiscreteSupport)
        and support.is_left_bounded
        and not support.is_right_bounded
    ):
        xs, cdf_at = build_head_table(support, pmf_func, eps=eps)
        min_point = float(support.first())  # type: ignore[arg-type]

        def _cdf_head(x: NumericArray, **options: Any) -> NumericArray:
            x_arr = np.atleast_1d(np.asarray(x, dtype=float))
            result = np.empty_like(x_arr)
            below = x_arr < min_point if right_closed else x_arr <= min_point
            result[below] = 0.0
            if xs.size == 0:
                result[~below] = 0.0
                return result
            idx = np.searchsorted(xs, x_arr[~below], side=side) - 1
            idx = np.clip(idx, 0, cdf_at.size - 1)
            result[~below] = cdf_at[idx]
            return result

        return FittedComputationMethod[NumericArray, NumericArray](
            target=CharacteristicName.CDF,
            sources=[CharacteristicName.PMF],
            func=_cdf_head,  # type: ignore[arg-type]
        )

    if (
        isinstance(support, IntegerLatticeDiscreteSupport)
        and not support.is_left_bounded
        and not support.is_right_bounded
    ):
        raise RuntimeError(
            "pmf->cdf for a two-sided infinite integer lattice is not supported "
            "by the generic fitter.  Provide an analytical CDF or a custom fitter."
        )

    xs = collect_discrete_support(support)
    if xs.size == 0:
        raise RuntimeError("Discrete support is empty.")

    pmf_vals = np.clip(np.asarray(pmf_func(xs), dtype=float), 0.0, None)
    cdf_vals = np.clip(np.cumsum(pmf_vals), 0.0, 1.0)
    np.maximum.accumulate(cdf_vals, out=cdf_vals)

    def _cdf(x: NumericArray, **options: Any) -> NumericArray:
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        idx = np.searchsorted(xs, x_arr, side=side) - 1
        result = np.where(idx < 0, 0.0, cdf_vals[np.clip(idx, 0, cdf_vals.size - 1)])
        return result

    return FittedComputationMethod[NumericArray, NumericArray](
        target=CharacteristicName.CDF,
        sources=[CharacteristicName.PMF],
        func=_cdf,  # type: ignore[arg-type]
    )


def _build_pmf_to_cdf_1D() -> FitterDescriptor:
    return FitterDescriptor(
        name="pmf_to_cdf_1D",
        target=CharacteristicName.CDF,
        sources=[CharacteristicName.PMF],
        fitter=_fit_pmf_to_cdf_1D,
        characteristic_options=(
            CharacteristicOption(
                name="right_closed",
                type=bool,
                default=True,
                description=(
                    "CDF convention.  True (default): right-closed F(x) = P(ξ ≤ x).  "
                    "False: right-open F⁻(x) = P(ξ < x).  "
                    "The right-open form satisfies 1 - F⁻(x) = P(ξ ≥ x), "
                    "which is needed when computing the CDF of -ξ."
                ),
            ),
        ),
        computation_options=(
            ComputationOption(
                name="eps",
                type=float,
                default=1e-12,
                description=(
                    "Stopping threshold for the tail walk on one-sided unbounded supports. "
                    "For right-bounded, left-unbounded supports the downward walk continues "
                    "while 1 - cumulative_sum >= eps. "
                    "For left-bounded, right-unbounded supports the upward walk (via mirroring) "
                    "continues while 1 - cumulative_sum >= eps. "
                    "Once the remaining tail probability falls below eps it is "
                    "considered negligible."
                ),
                validate=lambda v: 0.0 < v < 1.0,
            ),
        ),
        constraint_tags=frozenset({"discrete", "univariate"}),
        description=(
            "PMF -> CDF via prefix-sum (finite support) or tail summation "
            "(left-unbounded or right-unbounded).  Supports right-closed and "
            "right-open CDF conventions via the ``right_closed`` characteristic option."
        ),
    )


def _fit_cdf_to_pmf_1D(
    distribution: Distribution,
    /,
) -> FittedComputationMethod[NumericArray, NumericArray]:
    """
    Fit a ``cdf -> pmf`` conversion for discrete distributions.

    Parameters
    ----------
    distribution : Distribution
        Must expose a discrete ``support`` and a ``cdf`` characteristic.

    Returns
    -------
    FittedComputationMethod[NumericArray, NumericArray]

    Raises
    ------
    RuntimeError
        If the support is missing or empty.
    """
    support = _require_discrete_support(distribution, "cdf->pmf")

    cdf_func = resolve(distribution, CharacteristicName.CDF)

    xs = collect_discrete_support(support)
    if xs.size == 0:
        raise RuntimeError("Discrete support is empty.")

    cdf_vals = np.asarray(cdf_func(xs), dtype=float)
    cdf_vals = np.clip(cdf_vals, 0.0, 1.0)
    np.maximum.accumulate(cdf_vals, out=cdf_vals)

    pmf_vals = np.empty_like(cdf_vals)
    pmf_vals[0] = cdf_vals[0]
    pmf_vals[1:] = np.diff(cdf_vals)
    pmf_vals = np.clip(pmf_vals, 0.0, 1.0)

    def _pmf(x: NumericArray, **options: Any) -> NumericArray:
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        result = np.zeros_like(x_arr)

        idx = np.searchsorted(xs, x_arr, side="left")
        in_bounds = (idx >= 0) & (idx < xs.size)
        on_support = in_bounds & (xs[np.clip(idx, 0, xs.size - 1)] == x_arr)
        result[on_support] = pmf_vals[idx[on_support]]

        return result

    return FittedComputationMethod[NumericArray, NumericArray](
        target=CharacteristicName.PMF,
        sources=[CharacteristicName.CDF],
        func=_pmf,  # type: ignore[arg-type]
    )


def _build_cdf_to_pmf_1D() -> FitterDescriptor:
    return FitterDescriptor(
        name="cdf_to_pmf_1D",
        target=CharacteristicName.PMF,
        sources=[CharacteristicName.CDF],
        fitter=_fit_cdf_to_pmf_1D,
        characteristic_options=(),
        computation_options=(),
        constraint_tags=frozenset({"discrete", "univariate"}),
        description="CDF -> PMF via finite differences on the support table.",
    )


def _fit_cdf_to_ppf_1D(
    distribution: Distribution,
    /,
) -> FittedComputationMethod[NumericArray, NumericArray]:
    """
    Fit a ``cdf -> ppf`` conversion for discrete distributions.

    Parameters
    ----------
    distribution : Distribution
        Must expose a discrete ``support`` and a ``cdf`` characteristic.

    Returns
    -------
    FittedComputationMethod[NumericArray, NumericArray]

    Raises
    ------
    RuntimeError
        If the support is missing or empty.
    """
    support = _require_discrete_support(distribution, "cdf->ppf")

    cdf_func = resolve(distribution, CharacteristicName.CDF)

    xs = collect_discrete_support(support)
    if xs.size == 0:
        raise RuntimeError("Discrete support is empty.")

    cdf_vals = np.asarray(cdf_func(xs), dtype=float)
    cdf_vals = np.clip(cdf_vals, 0.0, 1.0)
    np.maximum.accumulate(cdf_vals, out=cdf_vals)

    x_first = float(xs[0])
    x_last = float(xs[-1])

    def _ppf(q: NumericArray, **options: Any) -> NumericArray:
        q_arr = np.atleast_1d(np.asarray(q, dtype=float))
        result = np.empty_like(q_arr)

        nan_mask = ~np.isfinite(q_arr)
        low_mask = (~nan_mask) & (q_arr <= 0.0)
        high_mask = (~nan_mask) & (q_arr >= 1.0)
        interior = ~nan_mask & ~low_mask & ~high_mask

        result[nan_mask] = np.nan
        result[low_mask] = x_first
        result[high_mask] = x_last

        if np.any(interior):
            q_in = q_arr[interior]
            idx = np.searchsorted(cdf_vals, q_in, side="left")
            idx = np.clip(idx, 0, xs.size - 1)
            result[interior] = xs[idx]

        return result

    return FittedComputationMethod[NumericArray, NumericArray](
        target=CharacteristicName.PPF,
        sources=[CharacteristicName.CDF],
        func=_ppf,  # type: ignore[arg-type]
    )


def _build_cdf_to_ppf_1D() -> FitterDescriptor:
    return FitterDescriptor(
        name="cdf_to_ppf_1D",
        target=CharacteristicName.PPF,
        sources=[CharacteristicName.CDF],
        fitter=_fit_cdf_to_ppf_1D,
        characteristic_options=(),
        computation_options=(),
        constraint_tags=frozenset({"discrete", "univariate"}),
        description="CDF -> PPF via searchsorted inversion on the support table.",
    )


def _fit_ppf_to_cdf_1D(
    distribution: Distribution,
    /,
    n_q_grid: int = 4096,
    right_closed: bool = True,
) -> FittedComputationMethod[NumericArray, NumericArray]:
    """
    Fit a ``ppf -> cdf`` conversion for discrete distributions.

    Parameters
    ----------
    distribution : Distribution
        Must expose an array-semantic ``ppf``.
    n_q_grid : int, default 4096
        *(Computation option)* Grid resolution for probing the PPF at
        fit-time.  Increase if the distribution has many closely-spaced
        support points.
    right_closed : bool, default True
        *(Characteristic option)* CDF convention:

        * ``True``: right-closed ``F(x) = P(ξ ≤ x)`` (standard).
        * ``False``: right-open ``F⁻(x) = P(ξ < x)``.

        The right-open form satisfies
        ``1 - F⁻(x) = P(ξ ≥ x)``, which is needed when computing the CDF
        of ``-ξ``.

    Returns
    -------
    FittedComputationMethod[NumericArray, NumericArray]
    """
    ppf_func = resolve(distribution, CharacteristicName.PPF)

    eps = 1.0 / (n_q_grid + 1)
    q_grid = np.linspace(eps, 1.0 - eps, n_q_grid)
    x_grid = np.asarray(ppf_func(q_grid), dtype=float)

    change = np.empty(n_q_grid, dtype=bool)
    change[0] = True
    change[1:] = x_grid[1:] != x_grid[:-1]

    xs_table = x_grid[change]

    change_idx = np.where(change)[0]
    right_idx = np.empty_like(change_idx)
    right_idx[:-1] = change_idx[1:] - 1
    right_idx[-1] = n_q_grid - 1

    left_idx = change_idx.copy()
    cdf_table_closed = np.clip(q_grid[right_idx], 0.0, 1.0)
    cdf_table_open = np.clip(
        np.concatenate([[0.0], q_grid[left_idx[1:] - 1]]),
        0.0,
        1.0,
    )
    np.maximum.accumulate(cdf_table_closed, out=cdf_table_closed)
    np.maximum.accumulate(cdf_table_open, out=cdf_table_open)

    cdf_table = cdf_table_closed if right_closed else cdf_table_open

    x_min = float(xs_table[0])
    x_max = float(xs_table[-1])

    def _cdf(x: NumericArray, **options: Any) -> NumericArray:
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        result = np.empty_like(x_arr)

        if right_closed:
            left_mask = x_arr < x_min
            right_mask = x_arr >= x_max
        else:
            left_mask = x_arr <= x_min
            right_mask = x_arr > x_max

        interior = ~left_mask & ~right_mask

        result[left_mask] = 0.0
        result[right_mask] = 1.0

        if np.any(interior):
            xi = x_arr[interior]
            if right_closed:
                idx = np.searchsorted(xs_table, xi, side="right") - 1
            else:
                idx = np.searchsorted(xs_table, xi, side="left") - 1
                idx = idx + 1
            idx = np.clip(idx, 0, cdf_table.size - 1)
            result[interior] = cdf_table[idx]

        return result

    return FittedComputationMethod[NumericArray, NumericArray](
        target=CharacteristicName.CDF,
        sources=[CharacteristicName.PPF],
        func=_cdf,  # type: ignore[arg-type]
    )


def _build_ppf_to_cdf_1D() -> FitterDescriptor:
    return FitterDescriptor(
        name="ppf_to_cdf_1D",
        target=CharacteristicName.CDF,
        sources=[CharacteristicName.PPF],
        fitter=_fit_ppf_to_cdf_1D,
        characteristic_options=(
            CharacteristicOption(
                name="right_closed",
                type=bool,
                default=True,
                description=(
                    "CDF convention.  True (default): right-closed F(x) = P(ξ ≤ x).  "
                    "False: right-open F⁻(x) = P(ξ < x).  "
                    "The right-open form satisfies 1 - F⁻(x) = P(ξ ≥ x), "
                    "which is needed when computing the CDF of -ξ."
                ),
            ),
        ),
        computation_options=(
            ComputationOption(
                name="n_q_grid",
                type=int,
                default=4096,
                description=(
                    "Number of q-points used to probe the PPF at fit-time.  "
                    "Increase if the distribution has many closely-spaced support points."
                ),
                validate=lambda v: v >= 16,
            ),
        ),
        constraint_tags=frozenset({"discrete", "univariate"}),
        description=(
            "PPF -> CDF via grid probing and step-function table construction.  "
            "Supports right-closed and right-open CDF conventions via the "
            "``right_closed`` characteristic option."
        ),
    )


def _build_discrete_descriptors() -> list[FitterDescriptor]:
    """Build and return all discrete 1D fitter descriptors (lazy factory)."""
    return [
        _build_pmf_to_cdf_1D(),
        _build_cdf_to_pmf_1D(),
        _build_cdf_to_ppf_1D(),
        _build_ppf_to_cdf_1D(),
    ]


__all__: list[str] = []
