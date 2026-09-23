"""
Maximum likelihood estimation for parametric families.

This module owns the *policy* of a likelihood fit: the order of the steps, the
choice between a closed-form solution and a numerical search, the fallback
between optimizers, and the packing of the outcome.  The numerical core it
calls lives in :mod:`pysatl_core.estimation.methods.mle.likelihood` and knows nothing of
any of this; the steps it shares with every other estimation method live in
:mod:`pysatl_core.estimation.problem.problem` and know nothing of likelihood.

The policy is an object — :class:`MLE`, an
:class:`~pysatl_core.estimation.estimator.Estimator` — rather than a function
with keyword arguments.  Its configuration (which optimizer, which options) is
stated once, in typed fields, and is then applied to as many samples or
families as the caller likes; and a second estimation method is a second class,
not another branch here.  There is no function form beside it: a second
spelling of the same fit would be a second place to configure one, and the
whole point of the object is that there is exactly one.

Estimation is an operation on a *family*, never on a distribution: a
distribution already has its parameters pinned, so there is nothing in it left
to estimate.  Maximising the likelihood over all densities has no solution at
all — the supremum is unbounded, since ever narrower spikes placed at the
observations drive it up without limit — so the family is what makes the
problem well posed, and the caller always supplies it.

Three neighbours carry what policy merely *uses*, so that this module is about
the decisions and not about their machinery:
:mod:`pysatl_core.estimation.methods.mle.likelihood` builds the objective and its gradient,
:mod:`pysatl_core.estimation.parameters.bounds` turns a family's declaration into the box
an optimizer accepts, and :mod:`pysatl_core.estimation.optimizers` is the one
door to ``scipy.optimize`` — it is what keeps ``OptimizeResult``, whose every
attribute types as ``Any``, from travelling any further than the call that
produced it.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pysatl_core.estimation._attribution import warn_at_caller
from pysatl_core.estimation.errors import EstimationError
from pysatl_core.estimation.methods.mle.likelihood import LogLikelihood
from pysatl_core.estimation.methods.mle.result import MLEResult
from pysatl_core.estimation.optimizers import (
    DEFAULT_OPTIMIZER,
    FALLBACK_OPTIMIZER,
    MinimizeMethod,
    MinimizeOptions,
    MinimizeSolver,
    Optimizer,
    ScipyMethod,
    WithFallback,
    optimizer_for,
)
from pysatl_core.estimation.parameters.start import project_onto_base
from pysatl_core.estimation.parameters.vectors import field_names, from_vector, to_vector
from pysatl_core.estimation.problem.problem import FitProblem
from pysatl_core.types import FamilyName

if TYPE_CHECKING:
    import numpy.typing as npt

    from pysatl_core.estimation.estimator import Estimator
    from pysatl_core.estimation.result import FitRoute
    from pysatl_core.families.parametric_family import ParametricFamily
    from pysatl_core.families.parametrizations import Parametrization
    from pysatl_core.types import EstimatorName, ParametrizationName


MLE_NAME: EstimatorName = "mle"
"""Name maximum likelihood reports in :attr:`~.result.FitResult.estimator`."""


@dataclass(frozen=True, slots=True)
class MLE:
    """
    Maximum likelihood estimation.

    Maximum likelihood picks the parameters under which the observed sample is
    most probable.  In practice the logarithm is maximised — the maximum is the
    same, since the logarithm is monotone, but a product of a thousand
    densities would underflow to zero — so the criterion is

    ``l(theta) = sum_i log f(x_i; theta) -> max``,

    and the objective handed to the optimizer is ``-l(theta)``.

    Parameters
    ----------
    optimizer : MinimizeMethod or MinimizeSolver or Optimizer or None, optional
        What to search with, in any of three spellings: a
        ``scipy.optimize.minimize`` method name (``"Nelder-Mead"``,
        ``"Powell"``, ...), a solver callable with the ``minimize`` signature,
        or a ready-made
        :class:`~pysatl_core.estimation.optimizers.Optimizer` — which may
        itself be a policy such as
        :class:`~pysatl_core.estimation.optimizers.WithFallback`.

        This field holds what the caller *wrote*; :attr:`search` is what it
        *means*, resolved into an optimizer object.  Read that one to find out
        what will actually run, including when nothing was named here.

        Naming anything forces the numerical path even for a family that has a
        closed-form solution, and a ``UserWarning`` says so.  Unlike SciPy —
        where an overridden ``fit`` silently discards this argument — the
        request is honoured.
    options : MinimizeOptions, optional
        Extra keyword arguments forwarded to ``scipy.optimize.minimize``, for
        example ``{"tol": 1e-12}`` or ``{"options": {"maxiter": 500}}``.  The
        accepted keys are listed in
        :class:`~pysatl_core.estimation.optimizers.MinimizeOptions`, so a
        misspelling is a type error here rather than a value that travels
        unchecked into SciPy.

        They configure the optimizer this class builds from *optimizer*, so
        they are meaningless beside a ready-made one, which carries its own —
        and passing both is a ``ValueError`` rather than a silently dropped
        tolerance.

    Raises
    ------
    ValueError
        If *options* is given alongside a ready-made ``Optimizer``.

    Notes
    -----
    **Which route is taken.** If the family declared a closed-form solution
    through the ``mle`` constructor argument and no ``optimizer`` was
    configured, that formula is used and no optimizer runs at all.  Otherwise
    the search starts from a method-of-moments point and runs L-BFGS-B, using
    the analytical gradient from
    :meth:`~pysatl_core.families.parametric_family.ParametricFamily.score` when
    the family provides one.  A family without ``score`` is still fitted, with
    ``minimize`` differencing the objective numerically — correct, but several
    times more objective evaluations.  If L-BFGS-B reports failure or takes no
    step, the fit is retried with Nelder-Mead and the fallback is recorded in
    ``MLEResult.message``.

    **Which coordinates are used.** The search always runs in the family's base
    parametrization.  Maximum likelihood is invariant to reparametrisation in
    theory, but numerically the variants are different problems with different
    conditioning, and the estimate should not depend on the coordinates it was
    asked for.

    **Hashing.** ``options`` is a mapping, so a configured ``MLE`` compares by
    value but cannot be hashed.  Estimators go in lists, not in sets.

    Examples
    --------
    >>> MLE().estimate(Normal, sample).route            # doctest: +SKIP
    'closed_form'
    >>> MLE(optimizer="Powell").estimate(Normal, sample).route   # doctest: +SKIP
    'numeric'
    """

    optimizer: MinimizeMethod | MinimizeSolver | Optimizer | None = None
    options: MinimizeOptions = field(default_factory=MinimizeOptions)

    def __post_init__(self) -> None:
        """Refuse options that would be silently ignored.

        A ready-made :class:`~pysatl_core.estimation.optimizers.Optimizer`
        carries its own options, so ``options`` passed alongside one would go
        nowhere — and a caller who wrote ``tol=`` has every right to assume a
        tolerance was applied.
        """
        if isinstance(self.optimizer, Optimizer) and self.options:
            raise ValueError(
                f"MLE got both a ready-made optimizer ({self.optimizer.name!r}) and "
                f"'options={dict(self.options)}'. An optimizer carries its own options, so "
                f"these would be ignored: put them on the optimizer itself."
            )

    @property
    def name(self) -> EstimatorName:
        """The name this method records in its results: ``"mle"``."""
        return MLE_NAME

    @property
    def search(self) -> Optimizer:
        """
        The optimizer this configuration means, as an object.

        With none named, that is the package's policy rather than a single
        algorithm: L-BFGS-B, falling back to Nelder-Mead when it reports
        failure or takes no step.  Naming one replaces the whole policy,
        fallback included — the request is honoured rather than wrapped in
        second-guessing.
        """
        if self.optimizer is None:
            return WithFallback(
                first=ScipyMethod(DEFAULT_OPTIMIZER, self.options),
                then=ScipyMethod(FALLBACK_OPTIMIZER, self.options, use_gradient=False),
            )
        return optimizer_for(self.optimizer, self.options)

    def estimate(
        self,
        family: ParametricFamily,
        sample: npt.ArrayLike,
        *,
        parametrization: ParametrizationName | None = None,
    ) -> MLEResult[Parametrization]:
        """
        Estimate the parameters of *family* from *sample* by maximum likelihood.

        The steps are: validate the sample; reject data that no parameter value
        could explain; take the closed-form solution if the family provides one
        and this estimator names no optimizer; otherwise search numerically
        from a method-of-moments start; finally recompute the clean
        log-likelihood and pack the outcome.

        Parameters
        ----------
        family : ParametricFamily
            Family to fit.  A view produced by
            :meth:`~pysatl_core.families.parametric_family.ParametricFamily.view`
            works unchanged: it *is* a family, whose base parametrization holds
            only the free parameters.
        sample : array_like
            Observed values; 1-D and finite.
        parametrization : ParametrizationName or None, optional
            Parametrization the estimate should be reported in.  Only the
            family's base parametrization is currently supported.

        Returns
        -------
        MLEResult[Parametrization]
            The estimate together with its log-likelihood, convergence flag and
            diagnostics.  The parameter is the base ``Parametrization`` because
            a family is bound to its parametrization class at runtime; a caller
            that knows the class can narrow the result itself.

        Raises
        ------
        ValueError
            If the sample is not 1-D or not finite.
        InsufficientDataError
            If there are fewer observations than free parameters.
        FitDataError
            If observations fall outside a support that does not depend on the
            parameters.  A closed-form rule may raise it for a second reason —
            data that contradict the parameters fixed in a view — but only on
            its own branch: the numerical path reports the same situation as a
            returned result with ``success=False`` and
            ``log_likelihood == -inf`` rather than as an exception.  Do not rely
            on the exception to detect it; test ``success`` instead, which is
            correct on both paths.
        EstimationError
            If the family declares no ``lpdf``, or no usable starting point
            exists.
        NotImplementedError
            If a non-base *parametrization* is requested.
        """
        problem = FitProblem.prepare(family, sample)
        problem.reject_data_outside_a_fixed_support()

        objective = LogLikelihood.of(family, problem.sample)
        closed_form = family.closed_form_for(self.name)
        fixed = problem.fixed

        if closed_form is not None and fixed.in_base_parametrization and self.optimizer is None:
            params = closed_form(problem.sample, fixed.values)
            if params is not None:
                projected = project_onto_base(family, params.parameters)
                if projected is not None:
                    return self._build_result(
                        problem,
                        objective,
                        projected,
                        parametrization,
                        route="closed_form",
                        optimizer=None,
                        success=True,
                        message="closed-form maximum likelihood solution",
                    )

        formula_applies = closed_form is not None and fixed.in_base_parametrization
        notes: list[str] = []
        if formula_applies and self.optimizer is not None:
            note = (
                f"family '{family.name}' has a closed-form MLE, but 'optimizer="
                f"{self.search.name}' was passed, so the numerical path is used"
            )
            notes.append(note)
            warn_at_caller(note)
            if family.name == FamilyName.CONTINUOUS_UNIFORM:
                caveat = (
                    "the numerical path is unreliable for a uniform family: the likelihood "
                    "maximum sits on the boundary of the admissible region (at min(x) and "
                    "max(x)) and the objective surface is discontinuous there, so a gradient "
                    "method cannot reach it; prefer the closed-form solution by omitting "
                    "'optimizer'"
                )
                notes.append(caveat)
                warn_at_caller(caveat)

        return self._fit_numerically(problem, objective, parametrization, notes=notes)

    def _fit_numerically(
        self,
        problem: FitProblem,
        objective: LogLikelihood,
        parametrization: ParametrizationName | None,
        *,
        notes: list[str],
    ) -> MLEResult[Parametrization]:
        """
        Hand the objective to the optimizer and pack what comes back.

        Which optimizer, and whether a failure is retried with another, is
        :attr:`search`; this method only refuses to start from a point where
        the objective is not finite, and records whatever the optimizer asked
        to have recorded.
        """
        family = problem.family
        fun = objective
        jac = objective.gradient_or_none
        x0 = to_vector(problem.probe)
        bounds = problem.box.as_scipy()

        start_value = fun(x0)
        if not np.isfinite(start_value):
            raise EstimationError(
                f"The objective is not finite at the starting point for family "
                f"'{family.name}' (parameters "
                f"{from_vector(family.base, x0).parameters}, objective {start_value}). "
                f"The optimizer has no direction to follow from there. Declare a "
                f"method-of-moments rule for this family, as the 'moment_start' argument of "
                f"'ParametricFamily', so that the search starts where the data have a "
                f"positive density."
            )

        outcome = self.search.minimize(fun, x0, gradient=jac, bounds=bounds)

        params = from_vector(family.base, outcome.x)
        notes.extend(outcome.notes)
        if outcome.message:
            notes.append(outcome.message)
        if outcome.success is None:
            notes.append("the optimizer reported no convergence flag, so success is not claimed")

        return self._build_result(
            problem,
            objective,
            params,
            parametrization,
            route="numeric",
            optimizer=outcome.optimizer,
            success=bool(outcome.success),
            message="; ".join(notes),
            n_iterations=outcome.n_iterations,
            n_function_evaluations=outcome.n_function_evaluations,
        )

    def _build_result(
        self,
        problem: FitProblem,
        objective: LogLikelihood,
        estimated_params: Parametrization,
        parametrization: ParametrizationName | None,
        *,
        route: FitRoute,
        optimizer: str | None,
        success: bool,
        message: str,
        n_iterations: int | None = None,
        n_function_evaluations: int | None = None,
    ) -> MLEResult[Parametrization]:
        """Recompute the clean log-likelihood and pack everything into a result.

        The log-likelihood comes from the same object the search minimised, so
        the family's log-density provider is fetched once per fit rather than
        once per branch.
        """
        value = objective.at(estimated_params)
        if success and value == -np.inf:
            success = False
            unusable = (
                "the data have zero likelihood under this estimate - an observation lies "
                "outside the support it implies, or has zero density there - so the estimate "
                "is not usable"
            )
            message = f"{message}; {unusable}" if message else unusable
        return MLEResult(
            family_name=problem.family.name,
            params=problem.report_in(estimated_params, parametrization),
            log_likelihood=value,
            n_params=len(field_names(problem.family.base)),
            n_observations=int(problem.sample.size),
            estimator=self.name,
            route=route,
            optimizer=optimizer,
            success=success,
            message=message,
            n_iterations=n_iterations,
            n_function_evaluations=n_function_evaluations,
        )


if TYPE_CHECKING:
    # States in one line what the class promises: a checker rejects this
    # assignment the moment 'MLE' drifts away from the protocol.
    _: Estimator[MLEResult[Parametrization]] = MLE()


__all__ = ["MLE", "MLE_NAME"]
