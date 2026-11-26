"""
Trying one optimizer, and another when the first falters.

A policy, expressed as an optimizer.  It used to be twenty-five lines inside
the maximum likelihood fit, where it could not be replaced, tested on its own,
or reused by any other estimation method.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from pysatl_core.estimation.optimizers.outcome import OptimizerOutcome
    from pysatl_core.estimation.optimizers.protocol import (
        GradientFunc,
        ObjectiveFunc,
        Optimizer,
    )


@dataclass(frozen=True, slots=True)
class WithFallback:
    """
    Try one optimizer; if it fails or does not move, try another and keep the
    better point.

    This is a *policy*, and making it an optimizer is the point: it used to be
    twenty-five lines inside the maximum likelihood fit, where it could not be
    replaced, tested on its own, or reused by any other estimation method.

    A quasi-Newton method models the objective through its derivatives and
    stalls on a surface that is not smooth.  Where a parameter moves the
    boundary of the support, the objective is piecewise constant in the number
    of unexplained observations — a staircase, on which a line search finds no
    improvement and returns the starting point untouched.  Hence the retry.

    The retry is derivative-free by design, so the gradient is withheld from
    it whatever it says it supports.

    The second result is a second opinion, not a verdict: adopting it unseen
    returned a worse likelihood on about half of sampled fits, so the two
    points are compared on the objective and the better one wins.  Substituting
    the algorithm silently would be indefensible — two similar fits would
    behave differently with no explanation — so whichever way it goes is
    recorded in :attr:`OptimizerOutcome.notes`.

    Parameters
    ----------
    first : Optimizer
        Tried first.
    then : Optimizer
        Tried when *first* reports failure or takes no step.
    """

    first: Optimizer
    then: Optimizer

    @property
    def name(self) -> str:
        """Both names: which one ran is answered by the outcome, not by this."""
        return f"{self.first.name} or {self.then.name}"

    def minimize(
        self,
        objective: ObjectiveFunc,
        x0: NDArray[np.float64],
        *,
        gradient: GradientFunc | None,
        bounds: list[tuple[float, float]] | None,
    ) -> OptimizerOutcome:
        """Run the first optimizer, and the second only if the first faltered."""
        outcome = self.first.minimize(objective, x0, gradient=gradient, bounds=bounds)
        if outcome.success and outcome.n_iterations != 0:
            return outcome

        verdict = "did not converge" if not outcome.success else "converged without taking a step"
        preamble = (
            f"{outcome.optimizer} {verdict} "
            f"(success={outcome.success}, nit={outcome.n_iterations}: {outcome.message})"
        )
        retry = self.then.minimize(objective, x0, gradient=None, bounds=bounds)
        if objective(retry.x) <= objective(outcome.x):
            return replace(
                retry, notes=(*retry.notes, f"{preamble}; fell back to {retry.optimizer}")
            )
        return replace(
            outcome,
            notes=(
                *outcome.notes,
                (
                    f"{preamble}; {retry.optimizer} was tried and reached a worse point, "
                    f"so the {outcome.optimizer} estimate was kept"
                ),
            ),
        )


__all__ = ["WithFallback"]
