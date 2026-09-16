"""
Starting points for the numerical maximum likelihood path.

The method of moments appears here in one role only: it produces the point the
optimizer starts from.  It is *not* offered as an estimation method of its own
(SciPy's ``method="mm"`` has no counterpart in this package), because a moment
estimator answers a different question and mixing the two behind one entry
point would make ``fit`` ambiguous.

A good start matters most where the support moves with the parameters.  For a
uniform family the objective is a staircase in the number of unexplained
observations, so a start that does not already cover the sample sits on a flat
plateau of pure penalty and tells the optimizer nothing.  The uniform rule
below therefore pads the observed range outwards — the same trick SciPy plays
in ``_fit_loc_scale_support``.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import warnings
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pysatl_core.estimation.bounds import clip_to_bounds
from pysatl_core.estimation.errors import MLEError
from pysatl_core.estimation.likelihood import field_names
from pysatl_core.types import FamilyName

if TYPE_CHECKING:
    from pysatl_core.families.parametric_family import ParametricFamily
    from pysatl_core.families.parametrizations import Parametrization


type MomentRule = Callable[[NDArray[np.float64]], Mapping[str, float]]


UNIFORM_START_PADDING: float = 0.05
"""Fraction of the observed range by which the uniform start is widened.

The start has to cover the sample: an interval that excludes even one
observation makes the objective a constant wall of penalty, with no gradient
and no simplex direction leading out of it.
"""


def _normal_start(sample: NDArray[np.float64]) -> dict[str, float]:
    """Moment start for the normal family — which is also its exact estimate."""
    return {"mu": float(sample.mean()), "sigma": float(sample.std())}


def _exponential_start(sample: NDArray[np.float64]) -> dict[str, float]:
    """Moment start for the exponential family: ``lambda = 1 / mean(x)``."""
    mean = float(sample.mean())
    return {"lambda_": 1.0 / mean if mean != 0.0 else 1.0}


def _gamma_start(sample: NDArray[np.float64]) -> dict[str, float]:
    """Moment start for the gamma family: ``k = mean^2 / var``, ``theta = var / mean``."""
    mean = float(sample.mean())
    var = float(sample.var())
    if mean <= 0.0 or var <= 0.0:
        return {"k": 1.0, "theta": 1.0}
    return {"k": mean * mean / var, "theta": var / mean}


def _uniform_start(sample: NDArray[np.float64]) -> dict[str, float]:
    """Moment start for the uniform family: the observed range, padded outwards."""
    low = float(sample.min())
    high = float(sample.max())
    span = high - low
    pad = UNIFORM_START_PADDING * span if span > 0.0 else max(abs(low), 1.0)
    return {"lower_bound": low - pad, "upper_bound": high + pad}


_MOMENT_STARTS: dict[FamilyName | str, MomentRule] = {
    FamilyName.NORMAL: _normal_start,
    FamilyName.EXPONENTIAL: _exponential_start,
    FamilyName.GAMMA: _gamma_start,
    FamilyName.CONTINUOUS_UNIFORM: _uniform_start,
}
"""Method-of-moments starting rules, keyed by family name.

A rule returns values in the family's *base* parametrization.  Entries for
parameters the caller has fixed through ``view`` are dropped by the projection
step in :func:`starting_point`; the fixed values are re-injected by the view
itself.  A family absent from this table starts from a vector of ones, exactly
as SciPy does (``args = (1.0,) * self.numargs``).
"""


def register_moment_start(family_name: FamilyName | str, rule: MomentRule) -> None:
    """
    Register a method-of-moments starting rule for a family.

    This is the extension point for user-defined families: it needs no
    subclassing and no change to the family object itself.

    Parameters
    ----------
    family_name : FamilyName or str
        Name of the family the rule applies to.  ``FamilyName`` is the exact
        type for the built-in families and is written first for that reason;
        a plain ``str`` is admitted because this function exists precisely so
        that a user-defined family, whose name is not in that enumeration, can
        register a rule.
    rule : MomentRule
        Callable mapping a sample to starting values, keyed by the names of
        the family's base parameters.
    """
    _MOMENT_STARTS[family_name] = rule


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
    family: ParametricFamily, sample: NDArray[np.float64], *, stacklevel: int = 2
) -> Parametrization:
    """
    Choose the point the optimizer starts from.

    The order is: a registered method-of-moments rule if the family has one;
    otherwise a vector of ones, projected into the declared parameter bounds.
    Either way the result is clipped into the bounds, so the start is always a
    point the optimizer is allowed to occupy.

    Parameters
    ----------
    family : ParametricFamily
        Family being fitted.
    sample : NDArray[np.float64]
        Validated 1-D sample.
    stacklevel : int, optional
        Frames to skip when attributing the warning below, in the sense of
        :func:`warnings.warn`.  The default points at a direct caller;
        ``fit_family`` passes a larger value so that a misspelled rule is
        reported against the user's ``fit`` call rather than against this
        package.

    Returns
    -------
    Parametrization
        An instance of ``family.base``.

    Raises
    ------
    MLEError
        If the starting values cannot be assembled at all.

    Warns
    -----
    UserWarning
        When a registered rule returns names that partly match the family's
        free parameters and partly do not — the signature of a misspelled
        name, as opposed to a rule written in another parametrization, which
        shares no names at all and is left alone.
    """
    rule = _MOMENT_STARTS.get(family.name)
    if rule is not None:
        values = rule(sample)
        projected = project_onto_base(family, values)
        if projected is not None:
            return clip_to_bounds(family, projected)
        names = set(field_names(family.base))
        if names & set(values):
            warnings.warn(
                f"the method-of-moments rule for family '{family.name}' returned "
                f"{sorted(values)}, which does not cover its free parameters "
                f"{sorted(names)}; starting from a default point instead. Check the "
                f"names the rule returns.",
                UserWarning,
                stacklevel=stacklevel,
            )

    ones = dict.fromkeys(field_names(family.base), 1.0)
    projected = project_onto_base(family, ones)
    if projected is None:  # pragma: no cover - ``ones`` covers every field by construction
        raise MLEError(
            f"Cannot build a starting point for family '{family.name}': its base "
            f"parametrization exposes no fields."
        )
    return clip_to_bounds(family, projected)


__all__ = [
    "UNIFORM_START_PADDING",
    "MomentRule",
    "starting_point",
    "register_moment_start",
    "project_onto_base",
]
