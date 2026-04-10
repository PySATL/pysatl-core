from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov, Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, Any

import numpy as np

from pysatl_core.distributions.computation import FittedComputationMethod
from pysatl_core.distributions.fitters.base import FitterDescriptor, FitterOption
from pysatl_core.distributions.fitters.helpers import (
    build_tail_table,
    collect_support,
    maybe_unwrap_scalar,
    resolve,
)
from pysatl_core.distributions.support import (
    DiscreteSupport,
    IntegerLatticeDiscreteSupport,
)
from pysatl_core.types import CharacteristicName, NumericArray

if TYPE_CHECKING:
    from pysatl_core.distributions.distribution import Distribution


def fit_pmf_to_cdf_1D(
    distribution: Distribution,
    /,
    **kwargs: Any,
) -> FittedComputationMethod[NumericArray, NumericArray]:
    """
    Fit a ``pmf -> cdf`` conversion for discrete distributions.

    Parameters
    ----------
    distribution : Distribution
        Must expose a discrete ``support`` and a ``pmf`` characteristic.
    **kwargs : Any
        Forwarded to the PMF at fit-time.

    Returns
    -------
    FittedComputationMethod[NumericArray, NumericArray]

    Raises
    ------
    RuntimeError
        If the support is missing, empty, or a two-sided infinite lattice.
    """
    support = distribution.support
    if support is None or not isinstance(support, DiscreteSupport):
        raise RuntimeError("Discrete support is required for pmf->cdf.")

    pmf_func = resolve(distribution, CharacteristicName.PMF)

    if (
        isinstance(support, IntegerLatticeDiscreteSupport)
        and not support.is_left_bounded
        and support.is_right_bounded
    ):
        xs, tail_from = build_tail_table(support, pmf_func, **kwargs)
        max_k = float(support.max_k)  # type: ignore[arg-type]

        def _cdf_tail(x: NumericArray, **options: Any) -> NumericArray:
            x_arr = np.atleast_1d(np.asarray(x, dtype=float))
            idx = np.searchsorted(xs, x_arr, side="right")
            result = np.clip(1.0 - tail_from[idx], 0.0, 1.0)
            result[x_arr >= max_k] = 1.0
            return maybe_unwrap_scalar(result)

        return FittedComputationMethod[NumericArray, NumericArray](
            target=CharacteristicName.CDF,
            sources=[CharacteristicName.PMF],
            func=_cdf_tail,  # type: ignore[arg-type]
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

    xs = collect_support(support)
    if xs.size == 0:
        raise RuntimeError("Discrete support is empty.")

    pmf_vals = np.clip(np.asarray(pmf_func(xs, **kwargs), dtype=float), 0.0, None)
    cdf_vals = np.clip(np.cumsum(pmf_vals), 0.0, 1.0)
    np.maximum.accumulate(cdf_vals, out=cdf_vals)

    def _cdf(x: NumericArray, **options: Any) -> NumericArray:
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        idx = np.searchsorted(xs, x_arr, side="right") - 1
        result = np.where(idx < 0, 0.0, cdf_vals[np.clip(idx, 0, cdf_vals.size - 1)])
        return maybe_unwrap_scalar(result)

    return FittedComputationMethod[NumericArray, NumericArray](
        target=CharacteristicName.CDF,
        sources=[CharacteristicName.PMF],
        func=_cdf,  # type: ignore[arg-type]
    )


FITTER_PMF_TO_CDF_1D = FitterDescriptor(
    name="pmf_to_cdf_1D",
    target=CharacteristicName.CDF,
    sources=[CharacteristicName.PMF],
    fitter=fit_pmf_to_cdf_1D,
    options=(),
    tags=frozenset({"discrete", "univariate"}),
    priority=0,
    description="PMF -> CDF via prefix-sum (finite support) or tail summation (left-unbounded).",
)


def fit_cdf_to_pmf_1D(
    distribution: Distribution,
    /,
    **kwargs: Any,
) -> FittedComputationMethod[NumericArray, NumericArray]:
    """
    Fit a ``cdf -> pmf`` conversion for discrete distributions.

    Parameters
    ----------
    distribution : Distribution
        Must expose a discrete ``support`` and a ``cdf`` characteristic.
    **kwargs : Any
        Forwarded to the CDF at fit-time.

    Returns
    -------
    FittedComputationMethod[NumericArray, NumericArray]

    Raises
    ------
    RuntimeError
        If the support is missing or empty.
    """
    support = distribution.support
    if support is None or not isinstance(support, DiscreteSupport):
        raise RuntimeError("Discrete support is required for cdf->pmf.")

    cdf_func = resolve(distribution, CharacteristicName.CDF)

    xs = collect_support(support)
    if xs.size == 0:
        raise RuntimeError("Discrete support is empty.")

    cdf_vals = np.asarray(cdf_func(xs, **kwargs), dtype=float)
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

        return maybe_unwrap_scalar(result)

    return FittedComputationMethod[NumericArray, NumericArray](
        target=CharacteristicName.PMF,
        sources=[CharacteristicName.CDF],
        func=_pmf,  # type: ignore[arg-type]
    )


FITTER_CDF_TO_PMF_1D = FitterDescriptor(
    name="cdf_to_pmf_1D",
    target=CharacteristicName.PMF,
    sources=[CharacteristicName.CDF],
    fitter=fit_cdf_to_pmf_1D,
    options=(),
    tags=frozenset({"discrete", "univariate"}),
    priority=0,
    description="CDF -> PMF via finite differences on the support table.",
)


def fit_cdf_to_ppf_1D(
    distribution: Distribution,
    /,
    **kwargs: Any,
) -> FittedComputationMethod[NumericArray, NumericArray]:
    """
    Fit a ``cdf -> ppf`` conversion for discrete distributions.



    Parameters
    ----------
    distribution : Distribution
        Must expose a discrete ``support`` and a ``cdf`` characteristic.
    **kwargs : Any
        Forwarded to the CDF at fit-time.

    Returns
    -------
    FittedComputationMethod[NumericArray, NumericArray]

    Raises
    ------
    RuntimeError
        If the support is missing or empty.
    """
    support = distribution.support
    if support is None or not isinstance(support, DiscreteSupport):
        raise RuntimeError("Discrete support is required for cdf->ppf.")

    cdf_func = resolve(distribution, CharacteristicName.CDF)

    xs = collect_support(support)
    if xs.size == 0:
        raise RuntimeError("Discrete support is empty.")

    cdf_vals = np.asarray(cdf_func(xs, **kwargs), dtype=float)
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

        return maybe_unwrap_scalar(result)

    return FittedComputationMethod[NumericArray, NumericArray](
        target=CharacteristicName.PPF,
        sources=[CharacteristicName.CDF],
        func=_ppf,  # type: ignore[arg-type]
    )


FITTER_CDF_TO_PPF_1D = FitterDescriptor(
    name="cdf_to_ppf_1D",
    target=CharacteristicName.PPF,
    sources=[CharacteristicName.CDF],
    fitter=fit_cdf_to_ppf_1D,
    options=(),
    tags=frozenset({"discrete", "univariate"}),
    priority=0,
    description="CDF -> PPF via searchsorted inversion on the support table.",
)


def fit_ppf_to_cdf_1D(
    distribution: Distribution,
    /,
    **kwargs: Any,
) -> FittedComputationMethod[NumericArray, NumericArray]:
    """
    Fit a ``ppf -> cdf`` conversion for discrete distributions.

    Parameters
    ----------
    distribution : Distribution
        Must expose an array-semantic ``ppf``.
    **kwargs : Any
        ``n_q_grid`` (int, default 4096) — grid resolution.

    Returns
    -------
    FittedComputationMethod[NumericArray, NumericArray]
    """
    opts = FITTER_PPF_TO_CDF_1D.resolve_options(kwargs)
    n_q_grid: int = opts["n_q_grid"]

    ppf_func = resolve(distribution, CharacteristicName.PPF)

    eps = 1.0 / (n_q_grid + 1)
    q_grid = np.linspace(eps, 1.0 - eps, n_q_grid)
    x_grid = np.asarray(ppf_func(q_grid, **kwargs), dtype=float)

    change = np.empty(n_q_grid, dtype=bool)
    change[0] = True
    change[1:] = x_grid[1:] != x_grid[:-1]

    xs_table = x_grid[change]

    change_idx = np.where(change)[0]
    right_idx = np.empty_like(change_idx)
    right_idx[:-1] = change_idx[1:] - 1
    right_idx[-1] = n_q_grid - 1

    cdf_table = q_grid[right_idx]
    cdf_table = np.clip(cdf_table, 0.0, 1.0)
    np.maximum.accumulate(cdf_table, out=cdf_table)

    x_min = float(xs_table[0])
    x_max = float(xs_table[-1])

    def _cdf(x: NumericArray, **options: Any) -> NumericArray:
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        result = np.empty_like(x_arr)

        left_mask = x_arr < x_min
        right_mask = x_arr >= x_max
        interior = ~left_mask & ~right_mask

        result[left_mask] = 0.0
        result[right_mask] = 1.0

        if np.any(interior):
            xi = x_arr[interior]
            idx = np.searchsorted(xs_table, xi, side="right") - 1
            idx = np.clip(idx, 0, cdf_table.size - 1)
            result[interior] = cdf_table[idx]

        return maybe_unwrap_scalar(result)

    return FittedComputationMethod[NumericArray, NumericArray](
        target=CharacteristicName.CDF,
        sources=[CharacteristicName.PPF],
        func=_cdf,  # type: ignore[arg-type]
    )


FITTER_PPF_TO_CDF_1D = FitterDescriptor(
    name="ppf_to_cdf_1D",
    target=CharacteristicName.CDF,
    sources=[CharacteristicName.PPF],
    fitter=fit_ppf_to_cdf_1D,
    options=(
        FitterOption(
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
    tags=frozenset({"discrete", "univariate"}),
    priority=0,
    description="PPF -> CDF via grid probing and step-function table construction.",
)
