"""
Result objects returned by parameter estimation.

Why an object and not a tuple of parameters: a bare tuple — the shape of the
old ``scipy.stats.rv_continuous.fit`` API — carries neither the attained
log-likelihood, nor a convergence flag, nor a diagnostic message, so a caller
cannot tell a converged fit from a silently failed one.  SciPy itself treated
that as a mistake and moved to an object (``FitResult`` with ``params``,
``success``, ``message`` and ``nllf()``) in the newer ``scipy.stats.fit`` API.
pysatl-core starts from the corrected variant.

This module holds only what every estimate has whatever produced it: which
family, which parameters, how many of them were free, how many observations,
whether it worked and what to tell the user.  What only one method can report
lives with that method — :class:`~pysatl_core.estimation.methods.mle.result.MLEResult`
adds the attained log-likelihood and the criteria derived from it.  A method
that measures something else reports it in its own subclass rather than in a
field named for likelihood: a minimum-distance fit has a distance, not a
log-likelihood, and calling the two by one name would make ``aic`` a number
computed from the wrong quantity.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pysatl_core.families.parametrizations import Parametrization

if TYPE_CHECKING:
    from pysatl_core.families.distribution import ParametricFamilyDistribution
    from pysatl_core.types import EstimatorName


type FitRoute = Literal["closed_form", "numeric"]
"""Which branch of an estimator produced an estimate: a formula, or a search.

This is *not* the name of the estimation method — that is
:attr:`FitResult.estimator`.  The two were one field while maximum likelihood
was the only method, and the field was called ``method``; with several methods
the word had to name one of them, so the branch got the narrower name it always
meant and ``method`` went to the method.

Declared once and used by every function that passes the value along, so that a
misspelling is a type error at the call site rather than a string that travels
unchecked into :attr:`MLEResult.route`.
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class FitResult[P: Parametrization]:
    """
    Outcome of a parameter estimation, in the part common to every method.

    The class is generic in the parametrization it carries, so that a caller
    who knows which class a family estimates keeps that knowledge:
    ``FitResult[MeanStd].params.sigma`` is a ``float``, while a misspelled
    ``.sigmaa`` is a type error.  ``fit`` itself can only promise
    ``FitResult[Parametrization]`` — a family is bound to its parametrization
    class at runtime, by a decorator that runs after the family object exists,
    so nothing static knows which class a given family will register.  See the
    note in ``ParametricFamily.fit``.

    The fields are keyword-only.  A result has eight or more of them, several
    adjacent ones are numbers, and a subclass adds more in the middle of the
    order; positional construction would be unreadable at best and silently
    wrong at worst.

    Parameters
    ----------
    family_name : str
        Name of the fitted family.
    params : P
        Estimated parameters, in the parametrization the caller requested
        (the family's base parametrization by default).  For a
        :meth:`~pysatl_core.families.parametric_family.ParametricFamily.view`
        this holds the free parameters only; the fixed ones stay in the view.
    n_params : int
        Number of *estimated* (free) parameters.  ``Normal`` has 2,
        ``Normal.view(mu=0)`` has 1.  Kept as a field because a caller cannot
        reconstruct it from ``params`` alone in every case, and because the
        information criteria need it.
    n_observations : int
        Sample size the fit was based on.  Must be positive: ``bic`` takes its
        logarithm.
    estimator : EstimatorName
        Name of the estimation method that produced this result, as the
        estimator reports it — ``"mle"`` for maximum likelihood.  Recorded so
        that a result read on its own, or one of several collected for a
        comparison, still says how it was obtained.
    success : bool
        Whether the estimate is trustworthy.
    message : str
        Human-readable explanation.  Empty when there was nothing to report.

    Raises
    ------
    ValueError
        If ``n_observations`` or ``n_params`` is negative, or if
        ``n_observations`` is zero.  The type ``int`` admits values this object
        has no meaning for, and :attr:`MLEResult.bic` would otherwise fail deep
        inside ``math.log`` with "expected a positive input", a message that
        names neither the field nor the object.
    """

    family_name: str
    params: P
    n_params: int
    n_observations: int
    estimator: EstimatorName
    success: bool
    message: str

    def __post_init__(self) -> None:
        """Reject counts that no fit can produce."""
        if self.n_observations <= 0:
            raise ValueError(
                f"n_observations must be positive; got {self.n_observations}. A fit is always "
                f"based on at least one observation, and 'bic' takes the logarithm of this "
                f"number."
            )
        if self.n_params < 0:
            raise ValueError(
                f"n_params must not be negative; got {self.n_params}. It counts the free "
                f"parameters the fit estimated."
            )

    @property
    def distribution(self) -> ParametricFamilyDistribution:
        """
        Build the distribution described by the estimate.

        Constructed on access rather than stored, so that a result can be
        inspected, compared or serialised without paying for a distribution
        nobody asked for.

        The family is taken from the parametrization class rather than from
        :attr:`family_name`: the name is a label, while ``__family__`` is the
        object the parameters were actually produced by, so the two cannot
        drift apart here.

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
        return family.distribution_from(self.params)


__all__ = ["FitResult", "FitRoute"]
