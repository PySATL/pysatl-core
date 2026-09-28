"""
What an optimizer is, expressed as a type.

The protocol is deliberately ignorant of what is being minimised: it takes a
plain callable of a parameter vector, not a
:class:`~pysatl_core.estimation.methods.mle.likelihood.LogLikelihood`.  That is what
lets a method of moments, a minimum-distance fit or a maximum a posteriori
estimate reuse every solver here without inheriting a line of likelihood code.

The two function aliases live here for the same reason.  They describe what an
optimizer consumes — a function of a flat parameter vector, and its gradient —
and belong to the consumer rather than to whichever method happened to define
one first.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pysatl_core.estimation.optimizers.outcome import OptimizerOutcome


type ObjectiveFunc = Callable[[NDArray[np.float64]], float]
"""A function of a flat parameter vector, returning the value to minimise."""

type GradientFunc = Callable[[NDArray[np.float64]], NDArray[np.float64]]
"""Its gradient, over the same vector."""


@runtime_checkable
class Optimizer(Protocol):
    """
    A search for the minimum of an objective.

    An object rather than a ``(method, options)`` pair passed around, for three
    reasons.  It carries its own configuration, so nothing downstream has to
    thread ``options`` through every call.  It answers ``supports jac?`` and
    ``supports bounds?`` itself, instead of a caller consulting two tables of
    method names.  And because it is a value, a *policy* — try this, and on
    failure that, keeping the better point — is itself an optimizer
    (:class:`WithFallback`), which can be swapped, tested on its own, and
    reused by an estimation method that has nothing to do with likelihood.

    The objective is a plain callable, not a
    :class:`~pysatl_core.estimation.methods.mle.likelihood.LogLikelihood`: an optimizer has
    no business knowing what it is minimising.
    """

    @property
    def name(self) -> str:
        """Readable name, recorded in :attr:`OptimizerOutcome.optimizer`."""
        ...

    def minimize(
        self,
        objective: ObjectiveFunc,
        x0: NDArray[np.float64],
        *,
        gradient: GradientFunc | None,
        bounds: list[tuple[float, float]] | None,
    ) -> OptimizerOutcome:
        """
        Minimise *objective* from *x0*, in declared types.

        Parameters
        ----------
        objective : ObjectiveFunc
            The function to minimise.
        x0 : NDArray[np.float64]
            Starting point, in the order of the family's base parameters.
        gradient : GradientFunc or None
            Analytical gradient when one exists; ``None`` means "difference it
            yourself".  An implementation withholds it from a method that would
            ignore it.
        bounds : list[tuple[float, float]] or None
            Box bounds, or ``None`` to run unbounded.

        Returns
        -------
        OptimizerOutcome
            The estimate and whatever diagnostics were reported.

        Raises
        ------
        EstimationError
            If a solver returns no usable ``x``.
        """
        ...


__all__ = ["GradientFunc", "ObjectiveFunc", "Optimizer"]
