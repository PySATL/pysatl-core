"""
Refusing data that cannot carry a fit.

Both checks here run before anything else, so that an unusable sample is
rejected with a message naming the real cause and quoting the numbers, rather
than surfacing later as a puzzling optimizer failure.  Neither knows which
method is about to run: a sample that is two-dimensional, or that lies outside
a support no parameter can move, is unusable for all of them.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING

import numpy as np

from pysatl_core.estimation.errors import FitDataError, InsufficientDataError
from pysatl_core.estimation.parameters.vectors import field_names

if TYPE_CHECKING:
    import numpy.typing as npt
    from numpy.typing import NDArray

    from pysatl_core.families.parametric_family import ParametricFamily
    from pysatl_core.families.parametrizations import Parametrization


def validate_sample(family: ParametricFamily, sample: npt.ArrayLike) -> NDArray[np.float64]:
    """
    Check that a sample can carry a fit, and normalise it.

    Runs before anything else, so that an unusable sample is rejected here,
    with a message naming the real cause and quoting the numbers, rather than
    surfacing later as a puzzling optimizer failure.

    Parameters
    ----------
    family : ParametricFamily
        Family being fitted; its base parametrization determines how many
        observations are the minimum.
    sample : array_like
        Observed values, coerced to a float array.  Declared as ``ArrayLike``
        rather than ``NDArray[np.float64]`` because that is what the body
        accepts: the ``try``/``except`` around ``np.asarray`` is only reachable
        for an argument that is *not* already a float array.

    Returns
    -------
    NDArray[np.float64]
        The sample as a 1-D float array.

    Raises
    ------
    ValueError
        If *sample* cannot be read as a float array, or is not one-dimensional,
        or contains ``NaN`` or infinities.
    InsufficientDataError
        If it holds fewer observations than there are free parameters.

    Notes
    -----
    There is deliberately no constant-sample check, following section 6.1 of
    the specification.  Be aware that its stated rationale does not hold: a
    constant sample drives ``sigma`` to 0 for a normal family and collapses the
    interval for a uniform one, and neither is in fact refused — the closed-form
    """
    try:
        arr = np.asarray(sample, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Sample must be convertible to a float array; got {type(sample).__name__}."
        ) from exc

    if arr.ndim != 1:
        raise ValueError(
            f"Sample must be one-dimensional; got an array with {arr.ndim} dimension(s) "
            f"and shape {arr.shape}. Estimation here is univariate: "
            f"flatten the data or fit one column at a time."
        )

    if not np.all(np.isfinite(arr)):
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        raise ValueError(
            f"Sample must be finite; got {n_nan} NaN and {n_inf} infinite value(s) out of "
            f"{arr.size}. A non-finite observation has no density, so the likelihood is "
            f"undefined: drop or impute those points before fitting."
        )

    free_names = field_names(family.base)
    n_free = len(free_names)
    if arr.size < n_free:
        raise InsufficientDataError(
            f"Family '{family.name}' estimates {n_free} free parameter(s) "
            f"({', '.join(free_names)}), but the sample holds "
            f"{arr.size} observation(s). At least {n_free} are required; with fewer, the "
            f"likelihood has no isolated maximum."
        )

    return arr


def check_fixed_support(
    family: ParametricFamily, sample: NDArray[np.float64], probe: Parametrization
) -> None:
    """
    Reject data lying outside a support that no parameter value can move.

    Raises
    ------
    FitDataError
        If the support does not depend on the parameters and some observation
        falls outside it.
    """
    support = family.support_resolver(probe)
    if support is None:
        return
    inside = np.asarray(support.contains(sample), dtype=bool)
    if bool(inside.all()):
        return
    outside = sample[~inside]
    raise FitDataError(
        f"{outside.size} of {sample.size} observation(s) lie outside the support "
        f"{support} of family '{family.name}', which does not depend on its parameters — "
        f"for example {float(outside[0])!r}. No parameter value can give those points a "
        f"positive density, so the likelihood is zero everywhere and there is nothing to "
        f"maximise. Drop them, or fit a family whose support covers the data."
    )


__all__ = ["check_fixed_support", "validate_sample"]
