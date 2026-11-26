"""
Parameter bounds: from a family's declaration to the box an optimizer accepts.

A family declares ``param_bounds`` as a mapping of open intervals; an optimizer
wants a closed box, one entry per free parameter, in the order the
parametrization declares them.  Translating between the two is the whole of
this module, and it is separate from :mod:`pysatl_core.estimation.methods.mle` for one
concrete reason: :mod:`pysatl_core.estimation.parameters.start` needs the box to keep a
starting point admissible, and taking it from ``mle`` — which imports
``moments`` — meant a deferred import inside a function body to dodge the
cycle.  Bounds sit below both, so neither has to.

The translation is a :class:`ParameterBox`, built once from a family and then
asked for what a caller needs.  It replaces three functions that each took a
``family`` and each re-read the same declaration: on one fit the box used to be
assembled four separate times, once for the optimizer and three more to keep a
candidate point inside it.

Bounds and ``@constraint`` predicates stay two separate mechanisms on purpose:
``param_bounds`` cannot validate anything, and a predicate cannot be turned into
a search region.  SciPy reached the same conclusion and added ``_ShapeInfo``
alongside ``_argcheck``.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pysatl_core.estimation._attribution import warn_at_caller
from pysatl_core.estimation.parameters.vectors import field_names, from_vector, to_vector

if TYPE_CHECKING:
    from pysatl_core.families.parametric_family import ParametricFamily
    from pysatl_core.families.parametrizations import Parametrization


@dataclass(frozen=True, slots=True)
class ParameterBox:
    """
    The region of the parameter space a search is allowed to occupy.

    Built from a family's ``param_bounds`` declaration and then asked for
    whichever form a caller needs: :meth:`as_scipy` for the optimizer,
    :meth:`clip` to pull a candidate point back inside.

    Attributes
    ----------
    family_name : str
        Name of the family the box came from, quoted when warning that it
        declared nothing.
    bounds : tuple[tuple[float, float], ...]
        One ``(low, high)`` pair per free parameter, in the order of
        ``family.base.__dataclass_fields__``.  Parameters with no entry get
        ``(-inf, inf)``, so the tuple always has one entry per parameter even
        when :attr:`declared` is ``False``.
    declared : bool
        Whether at least one bound came from the family rather than from the
        ``(-inf, inf)`` default.  The two states are kept apart because a box
        of nothing but infinities is not the same thing as a family that
        deliberately bounded every parameter to the whole line: only the first
        is worth warning about.
    """

    family_name: str
    bounds: tuple[tuple[float, float], ...]
    declared: bool

    @classmethod
    def of(cls, family: ParametricFamily) -> ParameterBox:
        """
        Read a family's ``param_bounds`` into a box.

        The single source is the family's ``param_bounds``, declared beside
        ``base_score`` and ``mle`` in its constructor.  For a view, both the
        order and the membership follow ``family.base.__dataclass_fields__``,
        that is, the free parameters only.

        Probing a ``@constraint`` predicate numerically is not an alternative:
        it cannot distinguish "no upper bound" from "the bound is at the edge
        of the search region", and for a coupled constraint the answer depends
        on where the other parameter happens to sit.

        Parameters
        ----------
        family : ParametricFamily
            Family being fitted.

        Returns
        -------
        ParameterBox
            The box.  Building it never warns: a caller that merely clips a
            point has no use for the "declares no bounds" warning, which is
            about the *optimizer* running unbounded.  See :meth:`as_scipy`.
        """
        declared_bounds = family.param_bounds
        bounds: list[tuple[float, float]] = []
        any_declared = False

        for name in field_names(family.base):
            entry = declared_bounds.get(name)
            if entry is None:
                bounds.append((-np.inf, np.inf))
                continue
            any_declared = True
            raw_low, raw_high = entry
            low = -np.inf if raw_low is None else float(raw_low)
            high = np.inf if raw_high is None else float(raw_high)
            # An entry such as ``("sigma", (0, None))`` states an *open* bound,
            # but an optimizer only understands a closed box. Nudging each
            # finite edge inwards by one ULP is what SciPy does in
            # ``_ShapeInfo``. Formally the gap is 5e-324 and numerically
            # useless on its own — the optimizer can still step into it — which
            # is precisely why the objective returns ``inf`` wherever a
            # constraint fails, so the line search backs off. No separate
            # notion of a "practical" bound is introduced.
            #
            # Every declared bound is treated as open, because the declaration
            # has no way to say otherwise.
            if np.isfinite(low):
                low = float(np.nextafter(low, np.inf))
            if np.isfinite(high):
                high = float(np.nextafter(high, -np.inf))
            bounds.append((low, high))

        return cls(family_name=family.name, bounds=tuple(bounds), declared=any_declared)

    def as_scipy(self) -> list[tuple[float, float]] | None:
        """
        The box in the form ``scipy.optimize.minimize`` takes, or ``None``.

        Returns
        -------
        list[tuple[float, float]] or None
            One ``(low, high)`` pair per free parameter, or ``None`` when the
            family declared no bounds at all — the optimisation then runs
            unbounded.

        Warns
        -----
        UserWarning
            When the family declares no bounds for any free parameter.  It is
            raised here rather than in :meth:`of` because this is the question
            it answers: the optimizer is about to run without a box.  It is
            charged to the caller's line by
            :func:`~pysatl_core.estimation._attribution.warn_at_caller`.
        """
        if not self.declared:
            warn_at_caller(
                f"Family '{self.family_name}' declares no 'param_bounds', so the optimizer "
                f"runs without bounds and may probe inadmissible parameters. The objective "
                f"rejects those with 'inf', so the fit is still correct, only slower and less "
                f"robust. Pass 'param_bounds={{...}}' to the family constructor to fix this."
            )
            return None
        return list(self.bounds)

    def clip[P: Parametrization](self, params: P) -> P:
        """
        Move a parametrization inside the box.

        Parameters
        ----------
        params : P
            Candidate parameters, in the family's base parametrization.

        Returns
        -------
        P
            The same values in the same class, each clipped into its bound.
            The argument itself is returned, unchanged and un-copied, when the
            family declares no bounds: there is then nothing to clip against,
            and rebuilding the object would only cost.
        """
        if not self.declared:
            return params
        vec = to_vector(params)
        lows = np.array([low for low, _ in self.bounds], dtype=np.float64)
        highs = np.array([high for _, high in self.bounds], dtype=np.float64)
        return from_vector(type(params), np.clip(vec, lows, highs))


__all__ = ["ParameterBox"]
