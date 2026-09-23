"""
What a maximum likelihood fit returns.

Everything an estimate of any method carries is inherited from
:class:`~pysatl_core.estimation.result.FitResult`; what is added here is what
only maximising a likelihood produces — the attained log-likelihood, the
information criteria built on it, and the optimizer's diagnostics.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pysatl_core.estimation.result import FitResult
from pysatl_core.families.parametrizations import Parametrization

if TYPE_CHECKING:
    from pysatl_core.estimation.result import FitRoute


@dataclass(frozen=True, slots=True, kw_only=True)
class MLEResult[P: Parametrization](FitResult[P]):
    """
    Outcome of a maximum likelihood fit.

    Everything a result of any method carries is inherited from
    :class:`FitResult`; what is added here is what only maximising a likelihood
    produces.

    Parameters
    ----------
    log_likelihood : float
        Attained log-likelihood ``l(theta_hat)``, computed without the
        out-of-support penalty that the objective function uses internally.
        ``-inf`` if some observation falls outside the support implied by the
        estimate.
    route : FitRoute
        Which branch produced the estimate: the family's closed-form formula,
        or a numerical search.
    optimizer : str or None
        Name of the optimizer actually used; ``None`` for ``closed_form``.
        On a fallback this names the optimizer that produced the final
        estimate, not the one that was tried first.
    n_iterations : int or None, optional
        Iterations reported by the optimizer, when it reports them.
    n_function_evaluations : int or None, optional
        Objective evaluations reported by the optimizer, when it reports them.
    """

    log_likelihood: float
    route: FitRoute
    optimizer: str | None
    n_iterations: int | None = None
    n_function_evaluations: int | None = None

    @property
    def aic(self) -> float:
        """Akaike information criterion, ``2k - 2*l(theta_hat)``."""
        return 2.0 * self.n_params - 2.0 * self.log_likelihood

    @property
    def bic(self) -> float:
        """Bayesian information criterion, ``k*log(n) - 2*l(theta_hat)``."""
        return self.n_params * math.log(self.n_observations) - 2.0 * self.log_likelihood


__all__ = ["MLEResult"]
