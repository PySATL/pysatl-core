"""
Parameter bounds: from a family's declaration to the box an optimizer accepts.

A family declares ``param_bounds`` as a mapping of open intervals; an optimizer
wants a closed box, one entry per free parameter, in the order the
parametrization declares them.  Translating between the two is the whole of
this module, and it is separate from :mod:`pysatl_core.estimation.mle` for one
concrete reason: :mod:`pysatl_core.estimation.moments` needs
:func:`clip_to_bounds` to keep a starting point admissible, and taking it from
``mle`` — which imports ``moments`` — meant a deferred import inside a function
body to dodge the cycle.  Bounds sit below both, so neither has to.

Bounds and ``@constraint`` predicates stay two separate mechanisms on purpose:
``param_bounds`` cannot validate anything, and a predicate cannot be turned into
a search region.  SciPy reached the same conclusion and added ``_ShapeInfo``
alongside ``_argcheck``.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import warnings
from typing import TYPE_CHECKING

import numpy as np

from pysatl_core.estimation.likelihood import field_names, from_vector, to_vector

if TYPE_CHECKING:
    from pysatl_core.families.parametric_family import ParametricFamily
    from pysatl_core.families.parametrizations import Parametrization


# TODO(mle): ``param_bounds`` cannot say whether a bound is open or closed, and
# this function assumes every one of them is open — it nudges each finite edge
# inwards by one ULP unconditionally.  A family needing ``c >= 0`` rather than
# ``c > 0`` therefore has no way to declare it: the optimizer is never allowed
# to sit on the endpoint.  Closed parameter bounds are ordinary, not exotic —
# SciPy declares them for ``foldnorm`` and ``foldcauchy`` (``c >= 0``), for
# ``erlang`` and ``irwinhall`` (``n >= 1``), and they are the natural shape for
# a mixture weight in [0, 1] or a correlation in [-1, 1].
#
# The workaround today is to declare the bound anyway and accept that an
# estimate sitting exactly on the endpoint comes back as 5e-324 instead of 0.
# Admissibility itself is unaffected: that is decided by the family's
# ``@constraint`` predicates, not by these bounds.  The loss only bites when
# the likelihood maximum lies *on* the boundary.
#
# SciPy solves this with an explicit flag: ``_ShapeInfo`` carries
# ``inclusive=(bool, bool)`` and shifts an endpoint only when it is exclusive.
# Two ways to add the same expressiveness here, both backward compatible — a
# two-element entry keeps meaning "open at both ends":
#
#   1. a third element on the tuple, mirroring SciPy directly:
#          param_bounds={"c": (0, None, (True, False))}
#
#   2. a small declarative object, which reads better at the declaration site
#      and leaves room for further per-parameter metadata (integrality, or the
#      reparametrisation transform of the TODO in ``mle.py``):
#          param_bounds={"c": Bound(low=0, high=None, low_closed=True)}
#
# Option 2 is preferable: a bare ``(0, None, (True, False))`` is hard to read
# and easy to mis-order, and a ``Bound`` dataclass with defaults
# ``low_closed=False, high_closed=False`` reproduces today's behaviour exactly
# while naming what each field means.  The change is local — accept the new
# form in ``ParametricFamily._normalize_param_bounds``, honour the flags here,
# and update ``TestBoundsAgreeWithConstraints``, which currently asserts the
# opposite (that a value *on* the declared edge fails ``validate()``).
def _collect_bounds(family: ParametricFamily) -> tuple[list[tuple[float, float]], bool]:
    """
    Assemble optimizer bounds, reporting whether the family declared any.

    Returns
    -------
    tuple[list[tuple[float, float]], bool]
        Bounds in the order of ``family.base.__dataclass_fields__``, and a flag
        that is ``True`` when at least one of them came from the family rather
        than from the ``(-inf, inf)`` default.
    """
    declared = family.param_bounds
    bounds: list[tuple[float, float]] = []
    any_declared = False

    for name in field_names(family.base):
        entry = declared.get(name)
        if entry is None:
            bounds.append((-np.inf, np.inf))
            continue
        any_declared = True
        raw_low, raw_high = entry
        low = -np.inf if raw_low is None else float(raw_low)
        high = np.inf if raw_high is None else float(raw_high)
        # An entry such as ``("sigma", (0, None))`` states an *open* bound, but
        # an optimizer only understands a closed box. Nudging each finite edge
        # inwards by one ULP is what SciPy does in ``_ShapeInfo``. Formally the
        # gap is 5e-324 and numerically useless on its own — the optimizer can
        # still step into it — which is precisely why the objective returns
        # ``inf`` wherever a constraint fails, so the line search backs off. No
        # separate notion of a "practical" bound is introduced.
        #
        # Every declared bound is treated as open, because the declaration has
        # no way to say otherwise. See the TODO above this function.
        if np.isfinite(low):
            low = float(np.nextafter(low, np.inf))
        if np.isfinite(high):
            high = float(np.nextafter(high, -np.inf))
        bounds.append((low, high))

    return bounds, any_declared


def resolve_bounds(family: ParametricFamily) -> list[tuple[float, float]] | None:
    """
    Build the box of parameter bounds handed to the optimizer.

    The single source is the family's ``param_bounds``, declared beside
    ``base_score`` and ``mle`` in its constructor.  Parameters with no entry get
    ``(-inf, inf)``.  For a view, both the order and the membership follow
    ``family.base.__dataclass_fields__``, that is, the free parameters only.

    Probing a ``@constraint`` predicate numerically is not an alternative: it
    cannot distinguish "no upper bound" from "the bound is at the edge of the
    search region", and for a coupled constraint the answer depends on where
    the other parameter happens to sit.

    Parameters
    ----------
    family : ParametricFamily
        Family being fitted.

    Returns
    -------
    list[tuple[float, float]] or None
        One ``(low, high)`` pair per free parameter, or ``None`` when the
        family declared no bounds at all — the optimisation then runs unbounded.

    Warns
    -----
    UserWarning
        When the family declares no bounds for any free parameter.
    """
    bounds, any_declared = _collect_bounds(family)
    if not any_declared:
        warnings.warn(
            f"Family '{family.name}' declares no 'param_bounds', so the optimizer runs "
            f"without bounds and may probe inadmissible parameters. The objective rejects "
            f"those with 'inf', so the fit is still correct, only slower and less robust. "
            f"Pass 'param_bounds={{...}}' to the family constructor to fix this.",
            UserWarning,
            stacklevel=2,
        )
        return None
    return bounds


def clip_to_bounds[P: Parametrization](family: ParametricFamily, params: P) -> P:
    """
    Move a parametrization inside the declared bounds.

    Parameters
    ----------
    family : ParametricFamily
        Family whose bounds apply.
    params : P
        Candidate parameters, in the family's base parametrization.

    Returns
    -------
    P
        The same values in the same class, each clipped into its bound;
        unchanged if the family declares no bounds.
    """
    bounds, any_declared = _collect_bounds(family)
    if not any_declared:
        return params
    vec = to_vector(params)
    lows = np.array([low for low, _ in bounds], dtype=np.float64)
    highs = np.array([high for _, high in bounds], dtype=np.float64)
    return from_vector(type(params), np.clip(vec, lows, highs))


__all__ = ["resolve_bounds", "clip_to_bounds"]
