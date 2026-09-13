"""
Result object returned by maximum likelihood estimation.

Why an object and not a tuple of parameters: a bare tuple — the shape of the
old ``scipy.stats.rv_continuous.fit`` API — carries neither the attained
log-likelihood, nor a convergence flag, nor a diagnostic message, so a caller
cannot tell a converged fit from a silently failed one.  SciPy itself treated
that as a mistake and moved to an object (``FitResult`` with ``params``,
``success``, ``message`` and ``nllf()``) in the newer ``scipy.stats.fit`` API.
pysatl-core starts from the corrected variant.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pysatl_core.families.distribution import ParametricFamilyDistribution
    from pysatl_core.families.parametrizations import Parametrization


# TODO(mle): inference is not implemented.  Standard errors and confidence
# intervals follow from the observed Fisher information — the negative Hessian
# of the log-likelihood at the estimate.  The building block is already there:
# ``ParametricFamily.score`` gives the per-observation gradient, so the
# observed information can be approximated by the outer-product (BHHH)
# estimator ``sum_i s_i s_i^T`` or by differentiating the score numerically.
# Adding ``standard_errors`` and ``confidence_interval(level)`` here would then
# be a local change, requiring no new family-level API.


@dataclass(frozen=True, slots=True)
class MLEResult:
    """
    Outcome of a maximum likelihood fit.

    Parameters
    ----------
    family_name : str
        Name of the fitted family.
    params : Parametrization
        Estimated parameters, in the parametrization the caller requested
        (the family's base parametrization by default).  For a
        :meth:`~pysatl_core.families.parametric_family.ParametricFamily.view`
        this holds the free parameters only; the fixed ones stay in the view.
    log_likelihood : float
        Attained log-likelihood ``l(theta_hat)``, computed without the
        out-of-support penalty that the objective function uses internally.
        ``-inf`` if some observation falls outside the support implied by the
        estimate.
    n_params : int
        Number of *estimated* (free) parameters.  ``Normal`` has 2,
        ``Normal.view(mu=0)`` has 1.  Kept as a field because a caller cannot
        reconstruct it from ``params`` alone in every case, and because
        :attr:`aic` and :attr:`bic` need it.
    n_observations : int
        Sample size the fit was based on.
    method : {"closed_form", "numeric"}
        Which branch produced the estimate.
    optimizer : str or None
        Name of the optimizer actually used; ``None`` for ``closed_form``.
        On a fallback this names the optimizer that produced the final
        estimate, not the one that was tried first.
    success : bool
        Whether the estimate is trustworthy.  Always ``True`` for a closed-form
        solution.
    message : str
        Human-readable explanation.  Records optimizer fallbacks and the
        caveats attached to a numerical fit of a boundary-supported family.
    n_iterations : int or None, optional
        Iterations reported by the optimizer, when it reports them.
    n_function_evaluations : int or None, optional
        Objective evaluations reported by the optimizer, when it reports them.
    """

    family_name: str
    params: Parametrization
    log_likelihood: float
    n_params: int
    n_observations: int
    method: Literal["closed_form", "numeric"]
    optimizer: str | None
    success: bool
    message: str
    n_iterations: int | None = None
    n_function_evaluations: int | None = None

    @property
    def distribution(self) -> ParametricFamilyDistribution:
        """
        Build the distribution described by the estimate.

        Constructed on access rather than stored, so that a result can be
        inspected, compared or serialised without paying for a distribution
        nobody asked for.

        Returns
        -------
        ParametricFamilyDistribution
            Distribution of the fitted family with the estimated parameters.

        Raises
        ------
        ValueError
            If the estimated parameters violate the family's constraints,
            which can happen for a fit that reports ``success=False``.
        """
        family = type(self.params).__family__
        return family.distribution(**self.params.parameters)

    @property
    def aic(self) -> float:
        """Akaike information criterion, ``2k - 2*l(theta_hat)``."""
        return 2.0 * self.n_params - 2.0 * self.log_likelihood

    @property
    def bic(self) -> float:
        """Bayesian information criterion, ``k*log(n) - 2*l(theta_hat)``."""
        return self.n_params * math.log(self.n_observations) - 2.0 * self.log_likelihood


__all__ = ["MLEResult"]
