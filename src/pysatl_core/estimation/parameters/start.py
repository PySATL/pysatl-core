"""
Where a numerical search begins.

The module was called ``moments`` while the only rules in it were
method-of-moments formulas, and that name is now taken: a *method of moments*
is an estimation method, and one may well appear in
:mod:`pysatl_core.estimation.methods`.  Two different things called "moments"
in one tree is one too many, so this one is named after its job.

Nothing here estimates anything.  A moment formula appears in one role only —
it produces the point an optimizer starts from — and that is a different
question from the one a moment *estimator* answers.  Mixing the two behind one
entry point would make ``fit`` ambiguous.

A good start matters most where the support moves with the parameters.  For a
uniform family the objective is a staircase in the number of unexplained
observations, so a start that does not already cover the sample sits on a flat
plateau of pure penalty and tells the optimizer nothing — which is why that
family's rule pads the observed range outwards.

The rules themselves are not here.  Each one is knowledge about a particular
family — the normal family's start is its own exact estimate, the uniform
family's has to cover the sample — so it is declared beside that family, as
``moment_start``, next to ``mle`` and ``base_score``.  This module only asks
the family for its rule and makes the answer usable: projected onto the free
parameters of a view, and clipped into the declared bounds.

There used to be a registry here instead, a module-level dictionary keyed by
family name with a ``register_moment_start`` function beside it.  It was global
mutable state: a rule outlived the family it described, two families of the
same name collided, and every test that set one had to remember to remove it
again.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pysatl_core.estimation._attribution import warn_at_caller
from pysatl_core.estimation.errors import EstimationError
from pysatl_core.estimation.parameters.bounds import ParameterBox
from pysatl_core.estimation.parameters.vectors import field_names

if TYPE_CHECKING:
    from pysatl_core.families.parametric_family import ParametricFamily
    from pysatl_core.families.parametrizations import Parametrization


type MomentRule = Callable[[NDArray[np.float64]], Mapping[str, float]]


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


def starting_point(family: ParametricFamily, sample: NDArray[np.float64]) -> Parametrization:
    """
    Choose the point the optimizer starts from.

    The order is: the family's own ``moment_start`` rule if it declares one;
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
    rule = family.moment_start
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
    "MomentRule",
    "project_onto_base",
    "starting_point",
]
