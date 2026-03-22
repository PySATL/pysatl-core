"""
Shared utilities for interpolation-based characteristic approximation.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pysatl_core.distributions.computation import AnalyticalComputation
from pysatl_core.types import Kind, NumericArray

if TYPE_CHECKING:
    from pysatl_core.transformations.distribution import DerivedDistribution
    from pysatl_core.types import GenericCharacteristicName


def get_analytical_method(
    distribution: DerivedDistribution,
    characteristic_name: GenericCharacteristicName,
) -> AnalyticalComputation[Any, Any]:
    """
    Get the first analytical method for a characteristic.

    Parameters
    ----------
    distribution : DerivedDistribution
        Distribution containing analytical methods.
    characteristic_name : GenericCharacteristicName
        Characteristic name to fetch.

    Returns
    -------
    AnalyticalComputation[Any, Any]
        First available analytical method for the requested
        characteristic.

    Raises
    ------
    ValueError
        If characteristic is not available analytically.
    """
    methods = distribution.analytical_computations.get(characteristic_name)
    if methods is None:
        raise ValueError(
            "Approximation requires an analytical method for "
            f"characteristic '{characteristic_name}'."
        )

    try:
        return next(iter(methods.values()))
    except StopIteration as exc:
        raise ValueError(
            f"Characteristic '{characteristic_name}' provides no analytical methods."
        ) from exc


def evaluate_on_grid(
    computation: AnalyticalComputation[Any, Any],
    grid: NumericArray,
    *,
    characteristic_name: GenericCharacteristicName,
    **options: Any,
) -> NumericArray:
    """
    Evaluate a method on a 1D grid with strict array semantics.

    Parameters
    ----------
    computation : AnalyticalComputation[Any, Any]
        Source analytical computation.
    grid : NumericArray
        One-dimensional interpolation grid.
    characteristic_name : GenericCharacteristicName
        Name of the approximated characteristic for error messages.
    **options : Any
        Extra options forwarded to the method.

    Returns
    -------
    NumericArray
        Values evaluated on the input grid.
    """
    values = np.asarray(computation(grid, **options), dtype=float)
    if values.shape != grid.shape:
        raise ValueError(
            f"Approximation for characteristic '{characteristic_name}' expects "
            "array semantics with shape-preserving outputs on interpolation grid."
        )
    return cast(NumericArray, values)


def validate_univariate_continuous(distribution: DerivedDistribution) -> None:
    """
    Validate that a distribution is univariate continuous.

    Parameters
    ----------
    distribution : DerivedDistribution
        Distribution to validate.

    Raises
    ------
    TypeError
        If distribution kind or dimension is unsupported.
    """
    distribution_type = distribution.distribution_type
    kind = getattr(distribution_type, "kind", None)
    dimension = getattr(distribution_type, "dimension", None)
    if kind != Kind.CONTINUOUS or dimension != 1:
        raise TypeError(
            "Interpolation approximation currently supports only univariate "
            "continuous distributions."
        )
