"""
Telling whether a family's support moves with its parameters.

The distinction drives how out-of-support observations are treated, and it is
the one fact about a family that cannot be read off a declaration: the support
is resolved at two points of the parameter space and the results compared.
Comparing them needs a fingerprint that is structural rather than by identity,
which is what the three signature classes are for.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pysatl_core.distributions.support import IntervalSupport, PointSupport
from pysatl_core.estimation.parameters.bounds import ParameterBox
from pysatl_core.estimation.parameters.vectors import from_vector, to_vector

if TYPE_CHECKING:
    from pysatl_core.distributions.support import Support
    from pysatl_core.families.parametric_family import ParametricFamily
    from pysatl_core.families.parametrizations import Parametrization


@dataclass(frozen=True, slots=True)
class IntervalSignature:
    """Fingerprint of a support that is an interval."""

    left: float
    right: float
    left_closed: bool
    right_closed: bool


@dataclass(frozen=True, slots=True)
class PointsSignature:
    """Fingerprint of a support given as an explicit set of points."""

    points: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class OpaqueSignature:
    """
    Fingerprint of a support matching neither shape protocol.

    The last resort, and named as such rather than hidden behind a magic
    string: two supports of an unknown kind are told apart by their type and
    their ``repr``, which compares their *printed form* rather than their
    meaning.  A family whose support lands here should declare
    :class:`~pysatl_core.distributions.support.IntervalSupport` or
    :class:`~pysatl_core.distributions.support.PointSupport` instead.
    """

    type_name: str
    representation: str


type SupportSignature = IntervalSignature | PointsSignature | OpaqueSignature | None
"""Comparable fingerprint of a support.

Every member is a frozen dataclass, so ``==`` is structural and two
fingerprints of the same shape compare field by field.  ``None`` means the
family declares no support at all, which is itself a distinguishable state.
"""


def support_signature(support: Support | None) -> SupportSignature:
    """
    Comparable fingerprint of a support, used to tell two supports apart.

    The fields are read through the shape protocols rather than probed by name:
    every one of them is a declared, typed member of the concrete support
    classes, so a misspelling here is a type error instead of a silent ``None``
    that would send :func:`support_depends_on_params` down the wrong branch —
    and with it the whole error contract of a fit, which either raises
    ``FitDataError`` for out-of-support data or charges a penalty and
    continues.
    """
    if support is None:
        return None
    # New dataclass, not the support itself: IntervalSupport/PointSupport are
    # Protocols, so an arbitrary implementation's own `==` can't be trusted to
    # compare structurally.
    if isinstance(support, IntervalSupport):
        return IntervalSignature(
            left=float(support.left),
            right=float(support.right),
            left_closed=bool(support.left_closed),
            right_closed=bool(support.right_closed),
        )
    if isinstance(support, PointSupport):
        return PointsSignature(
            points=tuple(float(p) for p in np.asarray(support.points).ravel().tolist())
        )
    return OpaqueSignature(type_name=type(support).__name__, representation=repr(support))


def support_depends_on_params(family: ParametricFamily, probe: Parametrization) -> bool:
    """
    Decide whether the family's support moves with its parameters.

    The distinction drives how out-of-support observations are treated.  When
    the support is fixed (normal, gamma, exponential), a point outside it can
    never be explained by any parameter value, so it is a data error and the
    fit stops. When the support moves with the parameters (uniform), the same
    point is only a symptom of the current iterate, so it is charged a penalty
    inside the objective and the search continues.

    This is a heuristic: the support is resolved at two different points of the
    parameter space and the results compared.  It answers correctly for the
    four built-in families.

    Parameters
    ----------
    family : ParametricFamily
        Family being fitted.
    probe : Parametrization
        A plausible point of the parameter space, in the family's base
        parametrization — the first of the two the supports are compared at.
        The sample is not needed here and is no longer asked for: it only ever
        served to produce this point, and computing it inside meant computing
        it three times per fit.  :func:`probe_params` is the usual source.

    Returns
    -------
    bool
        ``True`` if the two supports differ.
    """
    perturbed = _perturb(family, probe)
    first = support_signature(family.support_resolver(probe))
    second = support_signature(family.support_resolver(perturbed))
    return first != second


def _perturb[P: Parametrization](family: ParametricFamily, params: P) -> P:
    """
    Move every parameter to a different value, staying inside the bounds.

    ``2 * v + 1`` is used rather than ``2 * v`` so that a parameter sitting at
    zero also moves, and because it preserves the order of any two values and
    so cannot break a coupled constraint such as ``lower_bound < upper_bound``.
    """
    moved = from_vector(type(params), 2.0 * to_vector(params) + 1.0)
    return ParameterBox.of(family).clip(moved)


__all__ = [
    "IntervalSignature",
    "OpaqueSignature",
    "PointsSignature",
    "SupportSignature",
    "support_depends_on_params",
    "support_signature",
]
