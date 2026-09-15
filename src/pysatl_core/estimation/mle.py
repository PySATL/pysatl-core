"""
Maximum likelihood estimation for parametric families.

This module owns the *policy* of a fit: the order of the steps, the choice
between a closed-form solution and a numerical search, the fallback between
optimizers, and the packing of the outcome.  The numerical core it calls lives
in :mod:`pysatl_core.estimation.likelihood` and knows nothing of any of this.

Estimation is an operation on a *family*, never on a distribution: a
distribution already has its parameters pinned, so there is nothing in it left
to estimate.  Maximising the likelihood over all densities has no solution at
all — the supremum is unbounded, since ever narrower spikes placed at the
observations drive it up without limit — so the family is what makes the
problem well posed, and the caller always supplies it.

Three neighbours carry what policy merely *uses*, so that this module is about
the decisions and not about their machinery:
:mod:`pysatl_core.estimation.likelihood` builds the objective and its gradient,
:mod:`pysatl_core.estimation.bounds` turns a family's declaration into the box
an optimizer accepts, and :mod:`pysatl_core.estimation.optimizers` is the one
door to ``scipy.optimize`` — it is what keeps ``OptimizeResult``, whose every
attribute types as ``Any``, from travelling any further than the call that
produced it.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import warnings
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Unpack

import numpy as np

from pysatl_core.distributions.support import IntervalSupport, PointSupport
from pysatl_core.estimation.bounds import clip_to_bounds, resolve_bounds
from pysatl_core.estimation.errors import FitDataError, InsufficientDataError, MLEError
from pysatl_core.estimation.likelihood import (
    field_names,
    from_vector,
    log_likelihood,
    make_objective,
    to_vector,
)
from pysatl_core.estimation.moments import project_onto_base, starting_point
from pysatl_core.estimation.optimizers import (
    DEFAULT_OPTIMIZER,
    FALLBACK_OPTIMIZER,
    MinimizeMethod,
    MinimizeOptions,
    MinimizeSolver,
    optimizer_name,
    run_optimizer,
)
from pysatl_core.estimation.result import MLEResult
from pysatl_core.types import FamilyName

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt
    from numpy.typing import NDArray

    from pysatl_core.distributions.support import Support
    from pysatl_core.estimation.result import FitMethod
    from pysatl_core.families.parametric_family import ParametricFamily
    from pysatl_core.families.parametrizations import Parametrization
    from pysatl_core.types import ParametrizationName


def validate_sample(family: ParametricFamily, sample: npt.ArrayLike) -> NDArray[np.float64]:
    """
    Check that a sample can carry a maximum likelihood fit, and normalise it.

    Runs before anything else, so that an unusable sample is rejected here,
    with a message naming the real cause and quoting the numbers, rather than
    surfacing later as a puzzling optimizer failure.

    Parameters
    ----------
    family : ParametricFamily
        Family being fitted; its base parametrization determines how many
        observations are the minimum.
    sample : array_like
        Observed values, coerced to a float array.  Declared as ``ArrayLike``
        rather than ``NDArray[np.float64]`` because that is what the body
        accepts: the ``try``/``except`` around ``np.asarray`` is only reachable
        for an argument that is *not* already a float array.

    Returns
    -------
    NDArray[np.float64]
        The sample as a 1-D float array.

    Raises
    ------
    ValueError
        If *sample* cannot be read as a float array, or is not one-dimensional,
        or contains ``NaN`` or infinities.
    InsufficientDataError
        If it holds fewer observations than there are free parameters.

    Notes
    -----
    There is deliberately no constant-sample check, following section 6.1 of
    the specification.  Be aware that its stated rationale does not hold: a
    constant sample drives ``sigma`` to 0 for a normal family and collapses the
    interval for a uniform one, and neither is in fact refused — the closed-form
    """
    try:
        arr = np.asarray(sample, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Sample must be convertible to a float array; got {type(sample).__name__}."
        ) from exc

    if arr.ndim != 1:
        raise ValueError(
            f"Sample must be one-dimensional; got an array with {arr.ndim} dimension(s) "
            f"and shape {arr.shape}. Maximum likelihood estimation here is univariate: "
            f"flatten the data or fit one column at a time."
        )

    if not np.all(np.isfinite(arr)):
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        raise ValueError(
            f"Sample must be finite; got {n_nan} NaN and {n_inf} infinite value(s) out of "
            f"{arr.size}. A non-finite observation has no density, so the likelihood is "
            f"undefined: drop or impute those points before fitting."
        )

    free_names = field_names(family.base)
    n_free = len(free_names)
    if arr.size < n_free:
        raise InsufficientDataError(
            f"Family '{family.name}' estimates {n_free} free parameter(s) "
            f"({', '.join(free_names)}), but the sample holds "
            f"{arr.size} observation(s). At least {n_free} are required; with fewer, the "
            f"likelihood has no isolated maximum."
        )

    return arr


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


def _support_signature(support: Support | None) -> SupportSignature:
    """
    Comparable fingerprint of a support, used to tell two supports apart.

    The fields are read through the shape protocols rather than probed by name:
    every one of them is a declared, typed member of the concrete support
    classes, so a misspelling here is a type error instead of a silent ``None``
    that would send :func:`support_depends_on_params` down the wrong branch —
    and with it the whole error contract of :func:`fit_family`, which either
    raises ``FitDataError`` for out-of-support data or charges a penalty and
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
        it three times per fit.  :func:`~pysatl_core.estimation.moments.
        starting_point` is the usual source.

    Returns
    -------
    bool
        ``True`` if the two supports differ.
    """
    perturbed = _perturb(family, probe)
    first = _support_signature(family.support_resolver(probe))
    second = _support_signature(family.support_resolver(perturbed))
    return first != second


def _probe_params(family: ParametricFamily, sample: NDArray[np.float64]) -> Parametrization:
    """
    A plausible point of the parameter space, obtained without raising.

    The caught exceptions are the ones a *moment rule* can legitimately produce
    on awkward data: a value it refuses (``ValueError``), arithmetic that does
    not work out (``ArithmeticError`` and its three subclasses), or a start
    that cannot be assembled at all (``MLEError``).  Anything else — a
    ``TypeError``, a ``KeyError`` — is a bug in the rule rather than a verdict
    about the data, and is left to surface.

    Either way the substitution is announced: a rule that fails every time is a
    defect worth seeing, and swallowing it in silence is what would really blur
    the line between "the rule declined" and "the rule is broken".
    """
    try:
        return starting_point(family, sample, stacklevel=5)
    except (MLEError, ValueError, ArithmeticError) as exc:
        ones = project_onto_base(family, dict.fromkeys(field_names(family.base), 1.0))
        if ones is None:  # pragma: no cover - ones covers every field by construction
            raise
        warnings.warn(
            f"the method-of-moments starting rule for family '{family.name}' failed "
            f"({type(exc).__name__}: {exc}); starting from a default probe instead. The fit "
            f"continues, but it starts further from the answer than it needs to.",
            UserWarning,
            stacklevel=4,
        )

        return clip_to_bounds(family, ones)


def _perturb[P: Parametrization](family: ParametricFamily, params: P) -> P:
    """
    Move every parameter to a different value, staying inside the bounds.

    ``2 * v + 1`` is used rather than ``2 * v`` so that a parameter sitting at
    zero also moves, and because it preserves the order of any two values and
    so cannot break a coupled constraint such as ``lower_bound < upper_bound``.
    """
    moved = from_vector(type(params), 2.0 * to_vector(params) + 1.0)
    return clip_to_bounds(family, moved)


def _check_fixed_support(
    family: ParametricFamily, sample: NDArray[np.float64], probe: Parametrization
) -> None:
    """
    Reject data lying outside a support that no parameter value can move.

    Raises
    ------
    FitDataError
        If the support does not depend on the parameters and some observation
        falls outside it.
    """
    support = family.support_resolver(probe)
    if support is None:
        return
    inside = np.asarray(support.contains(sample), dtype=bool)
    if bool(inside.all()):
        return
    outside = sample[~inside]
    raise FitDataError(
        f"{outside.size} of {sample.size} observation(s) lie outside the support "
        f"{support} of family '{family.name}', which does not depend on its parameters — "
        f"for example {float(outside[0])!r}. No parameter value can give those points a "
        f"positive density, so the likelihood is zero everywhere and there is nothing to "
        f"maximise. Drop them, or fit a family whose support covers the data."
    )


@dataclass(frozen=True, slots=True)
class FixedParameters:
    """
    Parameters pinned through ``view``, and the coordinates they were pinned in.

    A named pair rather than a bare tuple: at the call site ``fixed.values`` and
    ``fixed.in_base_parametrization`` say what they are, where the second
    element of a ``tuple[Mapping[str, float], bool]`` said only ``True``.
    """

    values: Mapping[str, float]
    """The fixed values, keyed by parameter name."""

    in_base_parametrization: bool
    """Whether they were fixed in the parent family's *base* parametrization.

    A closed-form rule is written against the base parametrization, so it
    cannot be handed values expressed in any other.
    """


def _fixed_parameters(family: ParametricFamily) -> FixedParameters:
    """Report which parameters a view has pinned, and in which coordinates."""
    from pysatl_core.families.parametric_family import PartialParametricFamily

    if not isinstance(family, PartialParametricFamily):
        return FixedParameters(values=MappingProxyType({}), in_base_parametrization=True)
    # A view's own ``base_parametrization_name`` is the parametrization the
    # parameters were fixed in: ``PartialParametricFamily`` registers exactly
    # that one and nothing else.
    return FixedParameters(
        values=family.fixed_parameters,
        in_base_parametrization=(
            family.base_parametrization_name == family.parent_family.base_parametrization_name
        ),
    )


def _convert_parametrization(
    family: ParametricFamily,
    params: Parametrization,
    parametrization: ParametrizationName | None,
) -> Parametrization:
    """
    Express the estimate in the parametrization the caller asked for.

    Raises
    ------
    NotImplementedError
        For any parametrization other than the base one.
    """
    if parametrization is None or parametrization == family.base_parametrization_name:
        return params
    raise NotImplementedError(
        f"Cannot return the estimate in parametrization '{parametrization}': converting "
        f"from the base parametrization '{family.base_parametrization_name}' requires an "
        f"inverse transform, and the 'Parametrization' API offers only "
        f"'transform_to_base_parametrization'. Fit in the base parametrization and convert "
        f"the values by hand, or add the inverse transform to the family's "
        f"parametrization class."
    )


def _build_result(
    family: ParametricFamily,
    params: Parametrization,
    sample: NDArray[np.float64],
    parametrization: ParametrizationName | None,
    *,
    method: FitMethod,
    optimizer: str | None,
    success: bool,
    message: str,
    n_iterations: int | None = None,
    n_function_evaluations: int | None = None,
) -> MLEResult[Parametrization]:
    """Recompute the clean log-likelihood and pack everything into a result."""
    value = log_likelihood(family, params, sample)
    if success and value == -np.inf:
        success = False
        unusable = (
            "the data have zero likelihood under this estimate - an observation lies "
            "outside the support it implies, or has zero density there - so the estimate "
            "is not usable"
        )
        message = f"{message}; {unusable}" if message else unusable
    return MLEResult(
        family_name=family.name,
        params=_convert_parametrization(family, params, parametrization),
        log_likelihood=value,
        n_params=len(field_names(family.base)),
        n_observations=int(sample.size),
        method=method,
        optimizer=optimizer,
        success=success,
        message=message,
        n_iterations=n_iterations,
        n_function_evaluations=n_function_evaluations,
    )


def fit_family(
    family: ParametricFamily,
    sample: npt.ArrayLike,
    *,
    parametrization: ParametrizationName | None = None,
    optimizer: MinimizeMethod | MinimizeSolver | None = None,
    **options: Unpack[MinimizeOptions],
) -> MLEResult[Parametrization]:
    """
    Estimate the parameters of *family* from *sample* by maximum likelihood.

    The steps are: validate the sample; reject data that no parameter value
    could explain; take the closed-form solution if the family provides one and
    the caller did not ask for an optimizer; otherwise search numerically from
    a method-of-moments start; finally recompute the clean log-likelihood and
    pack the outcome.

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
        Parametrization the estimate should be reported in.  Only the family's
        base parametrization is currently supported.
    optimizer : MinimizeMethod or MinimizeSolver or None, optional
        ``scipy.optimize.minimize`` method name, or a solver callable with the
        ``minimize`` signature.  Passing it forces the numerical path even when
        a closed-form solution exists.
    **options
        Extra keyword arguments forwarded to ``scipy.optimize.minimize``, for
        example ``tol=1e-12`` or ``options={"maxiter": 500}``.  The accepted
        keys are listed in :class:`MinimizeOptions`.

    Returns
    -------
    MLEResult[Parametrization]
        The estimate together with its log-likelihood, convergence flag and
        diagnostics.  The parameter is the base ``Parametrization`` because a
        family is bound to its parametrization class at runtime; a caller that
        knows the class can narrow the result itself.

    Raises
    ------
    ValueError
        If the sample is not 1-D or not finite.
    InsufficientDataError
        If there are fewer observations than free parameters.
    FitDataError
        If observations fall outside a support that does not depend on the
        parameters.  A closed-form rule may raise it for a second reason — data
        that contradict the parameters fixed in a view — but only on its own
        branch: the numerical path reports the same situation as a returned
        result with ``success=False`` and ``log_likelihood == -inf`` rather
        than as an exception.  Do not rely on the exception to detect it; test
        ``success`` instead, which is correct on both paths.
    MLEError
        If the family declares no ``lpdf``, or no usable starting point exists.
    NotImplementedError
        If a non-base *parametrization* is requested.
    """
    data = validate_sample(family, sample)

    probe = _probe_params(family, data)
    if not support_depends_on_params(family, probe):
        _check_fixed_support(family, data, probe)

    closed_form = family.mle
    fixed = _fixed_parameters(family)

    if closed_form is not None and fixed.in_base_parametrization and optimizer is None:
        params = closed_form(data, fixed.values)
        if params is not None:
            projected = project_onto_base(family, params.parameters)
            if projected is not None:
                return _build_result(
                    family,
                    projected,
                    data,
                    parametrization,
                    method="closed_form",
                    optimizer=None,
                    success=True,
                    message="closed-form maximum likelihood solution",
                )

    formula_applies = closed_form is not None and fixed.in_base_parametrization
    notes: list[str] = []
    if formula_applies and optimizer is not None:
        note = (
            f"family '{family.name}' has a closed-form MLE, but 'optimizer="
            f"{optimizer_name(optimizer)}' was passed, so the numerical path is used"
        )
        notes.append(note)
        warnings.warn(note, UserWarning, stacklevel=3)
        if family.name == FamilyName.CONTINUOUS_UNIFORM:
            caveat = (
                "the numerical path is unreliable for a uniform family: the likelihood "
                "maximum sits on the boundary of the admissible region (at min(x) and "
                "max(x)) and the objective surface is discontinuous there, so a gradient "
                "method cannot reach it; prefer the closed-form solution by omitting "
                "'optimizer'"
            )
            notes.append(caveat)
            warnings.warn(caveat, UserWarning, stacklevel=3)

    return _fit_numerically(
        family, data, parametrization, optimizer, start=probe, notes=notes, **options
    )


def _fit_numerically(
    family: ParametricFamily,
    sample: NDArray[np.float64],
    parametrization: ParametrizationName | None,
    optimizer: MinimizeMethod | MinimizeSolver | None,
    *,
    start: Parametrization,
    notes: list[str],
    **options: Unpack[MinimizeOptions],
) -> MLEResult[Parametrization]:
    """
    Run the numerical search, applying the optimizer fallback policy.

    With no optimizer named, L-BFGS-B runs first, using the analytical gradient
    from ``score`` when the family provides one and letting ``minimize``
    difference the objective when it does not.  If L-BFGS-B reports failure or
    does not move at all, the fit is retried with Nelder-Mead from the same
    start.  Substituting the algorithm silently would be indefensible — two
    similar fits would behave differently with no explanation — so the fallback
    is always recorded in ``MLEResult.message``.
    """
    fun, jac = make_objective(family, sample)
    x0 = to_vector(start)
    bounds = resolve_bounds(family, stacklevel=5)

    start_value = fun(x0)
    if not np.isfinite(start_value):
        raise MLEError(
            f"The objective is not finite at the starting point for family "
            f"'{family.name}' (parameters "
            f"{from_vector(family.base, x0).parameters}, objective {start_value}). "
            f"The optimizer has no direction to follow from there. Register a "
            f"method-of-moments rule for this family with "
            f"'pysatl_core.estimation.moments.register_moment_start(name, rule)', so that "
            f"the search starts where the data have a positive density."
        )

    method: MinimizeMethod | MinimizeSolver = (
        optimizer if optimizer is not None else DEFAULT_OPTIMIZER
    )
    outcome = run_optimizer(fun, x0, jac=jac, bounds=bounds, method=method, options=options)

    if optimizer is None and (not outcome.success or outcome.n_iterations == 0):
        # A retry is a second opinion, not a verdict — adopting it unseen
        # returned a worse likelihood on ~half of sampled fits.
        first = outcome
        first_name = optimizer_name(method)
        verdict = "did not converge" if not first.success else "converged without taking a step"
        # No gradient on the retry: the fallback is derivative-free by design.
        retry = run_optimizer(
            fun, x0, jac=None, bounds=bounds, method=FALLBACK_OPTIMIZER, options=options
        )
        preamble = (
            f"{first_name} {verdict} "
            f"(success={first.success}, nit={first.n_iterations}: {first.message})"
        )
        if fun(retry.x) <= fun(first.x):
            outcome = retry
            method = FALLBACK_OPTIMIZER
            notes.append(f"{preamble}; fell back to {FALLBACK_OPTIMIZER}")
        else:
            notes.append(
                f"{preamble}; {FALLBACK_OPTIMIZER} was tried and reached a worse point, "
                f"so the {first_name} estimate was kept"
            )

    params = from_vector(family.base, outcome.x)
    if outcome.message:
        notes.append(outcome.message)
    if outcome.success is None:
        notes.append("the optimizer reported no convergence flag, so success is not claimed")

    return _build_result(
        family,
        params,
        sample,
        parametrization,
        method="numeric",
        optimizer=optimizer_name(method),
        success=bool(outcome.success),
        message="; ".join(notes),
        n_iterations=outcome.n_iterations,
        n_function_evaluations=outcome.n_function_evaluations,
    )


__all__ = [
    "FixedParameters",
    "IntervalSignature",
    "OpaqueSignature",
    "PointsSignature",
    "SupportSignature",
    "fit_family",
    "support_depends_on_params",
    "validate_sample",
]
