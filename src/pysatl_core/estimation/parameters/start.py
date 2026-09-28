"""Where a numerical search begins.

Moment rules are stored in :class:`~pysatl_core.estimation.EstimationFormulaRegistry`.
This module projects their named values onto a family's free parameters and
clips the result to its declared bounds. A rule is only a starting point; a
method-of-moments estimator would be a separate estimation method.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pysatl_core.estimation._attribution import warn_at_caller
from pysatl_core.estimation.errors import EstimationError
from pysatl_core.estimation.formulas import EstimationFormulaRegistry
from pysatl_core.estimation.parameters.bounds import ParameterBox
from pysatl_core.estimation.parameters.vectors import field_names

if TYPE_CHECKING:
    from pysatl_core.families.parametric_family import ParametricFamily
    from pysatl_core.families.parametrizations import Parametrization


def project_onto_base(
    family: ParametricFamily, values: Mapping[str, float]
) -> Parametrization | None:
    """
    Build an instance of the family's base parametrization from named values.

    For a plain family the base parametrization is the full one; for a view it
    holds only the free parameters, and entries naming fixed parameters are
    simply not used.

    Parameters
    ----------
    family : ParametricFamily
        Family whose base parametrization is the target.
    values : Mapping[str, float]
        Candidate values keyed by parameter name.

    Returns
    -------
    Parametrization or None
        The instance, or ``None`` if ``values`` does not cover every field of
        the target class.  That happens for a view fixed in a non-base
        parametrization, where the two sets of names live in different
        coordinate systems.
    """
    fields = list(field_names(family.base))
    if not set(fields).issubset(values):
        return None
    return family.base(**{name: float(values[name]) for name in fields})


def starting_point(
    family: ParametricFamily,
    sample: NDArray[np.float64],
) -> Parametrization:
    """
    Choose the point the optimizer starts from.

    The order is: the registry's moment starting rule if it has one;
    otherwise a vector of ones, projected into the declared parameter bounds.
    Either way the result is clipped into the bounds, so the start is always a
    point the optimizer is allowed to occupy.

    Parameters
    ----------
    family : ParametricFamily
        Family being fitted.
    sample : NDArray[np.float64]
        Validated 1-D sample.

    Returns
    -------
    Parametrization
        An instance of ``family.base``.

    Raises
    ------
    EstimationError
        If the starting values cannot be assembled at all.

    Warns
    -----
    UserWarning
        When the rule returns names that partly match the family's
        free parameters and partly do not — the signature of a misspelled
        name, as opposed to a rule written in another parametrization, which
        shares no names at all and is left alone.
    """
    rule = EstimationFormulaRegistry.moment_start_for(family)
    if rule is not None:
        values = rule(sample)
        projected = project_onto_base(family, values)
        if projected is not None:
            return ParameterBox.of(family).clip(projected)
        names = set(field_names(family.base))
        if names & set(values):
            warn_at_caller(
                f"the method-of-moments rule for family '{family.name}' returned "
                f"{sorted(values)}, which does not cover its free parameters "
                f"{sorted(names)}; starting from a default point instead. Check the "
                f"names the rule returns."
            )

    ones = dict.fromkeys(field_names(family.base), 1.0)
    projected = project_onto_base(family, ones)
    if projected is None:  # pragma: no cover - ``ones`` covers every field by construction
        raise EstimationError(
            f"Cannot build a starting point for family '{family.name}': its base "
            f"parametrization exposes no fields."
        )
    return ParameterBox.of(family).clip(projected)


__all__ = [
    "project_onto_base",
    "starting_point",
]
