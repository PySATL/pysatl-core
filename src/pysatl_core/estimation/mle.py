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
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import warnings
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from pysatl_core.estimation.errors import FitDataError, InsufficientDataError, MLEError
from pysatl_core.estimation.likelihood import (
    field_names,
    from_vector,
    log_likelihood,
    make_objective,
    to_vector,
)
from pysatl_core.estimation.moments import project_onto_base, starting_point
from pysatl_core.estimation.result import MLEResult
from pysatl_core.types import FamilyName

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from numpy.typing import NDArray

    from pysatl_core.distributions.support import Support
    from pysatl_core.families.parametric_family import ParametricFamily
    from pysatl_core.families.parametrizations import Parametrization
    from pysatl_core.types import ParametrizationName


DEFAULT_OPTIMIZER: str = "L-BFGS-B"
"""Optimizer used when the caller names none and no closed form applies.

With the analytical gradient from ``ParametricFamily.score`` it is markedly
more economical than a simplex method on smooth problems — on a normal sample
of 1000 points it reaches the same estimate in 14 objective evaluations against
147 for Nelder-Mead.
"""

FALLBACK_OPTIMIZER: str = "Nelder-Mead"
"""Derivative-free optimizer retried when the default reports failure.

A quasi-Newton method models the objective through its derivatives and stalls
on a surface that is not smooth.  Where a parameter moves the boundary of the
support, the objective is piecewise constant in the number of unexplained
observations — a staircase, on which a line search finds no improvement and
returns the starting point untouched.
"""

_GRADIENT_METHODS: frozenset[str] = frozenset(
    {
        "cg",
        "bfgs",
        "newton-cg",
        "l-bfgs-b",
        "tnc",
        "slsqp",
        "dogleg",
        "trust-ncg",
        "trust-krylov",
        "trust-exact",
        "trust-constr",
    }
)
"""``scipy.optimize.minimize`` methods that make use of ``jac``."""

_BOUNDED_METHODS: frozenset[str] = frozenset(
    {
        "nelder-mead",
        "l-bfgs-b",
        "tnc",
        "slsqp",
        "powell",
        "trust-constr",
        "cobyla",
        "cobyqa",
    }
)
"""``scipy.optimize.minimize`` methods that accept ``bounds``."""


# TODO(mle): selecting a family from a list of candidates by AIC/BIC is not
# implemented.  It is a loop over ``fit_family`` that keeps the candidate with
# the smallest criterion; ``MLEResult`` already carries ``n_params``,
# ``n_observations`` and ``log_likelihood``, so no new plumbing is needed.  The
# open question is what to do with candidates whose fit reports
# ``success=False``, which is why it is not decided here.

# TODO(mle): parameters with a one-sided bound are optimised in their natural
# coordinates.  Re-parametrising them (``sigma -> log sigma``, ``k -> log k``)
# would remove the bound entirely and improve conditioning near zero.  It needs
# a per-parameter transform declared next to ``param_bounds`` plus the matching
# chain rule applied to the gradient before it reaches the optimizer.

# TODO(mle): whether the support depends on the parameters is decided by the
# heuristic in ``support_depends_on_params`` — it compares the support at two
# points of the parameter space.  It answers correctly for all four built-in
# families, but a family whose support happens to coincide at those two points
# would be misclassified.  The proper fix is a declarative flag on
# ``ParametricFamily`` (say ``support_depends_on_parameters: bool``), set by the
# family author, with this heuristic kept only as the default.

# TODO(mle): a result cannot be converted from the base parametrization into an
# arbitrary one.  ``Parametrization`` offers only
# ``transform_to_base_parametrization``; there is no inverse, and it cannot be
# synthesised from the forward map.  Adding
# ``transform_from_base_parametrization`` to the parametrizations of the
# built-in families would make ``fit(..., parametrization=...)`` work for every
# declared parametrization.

# TODO(mle): discrete families are not supported.  None are registered, and
# ``CharacteristicName`` has ``PMF`` but no ``LPMF``, so there is no log-mass
# characteristic for the objective to sum.  Nothing here assumes continuity
# beyond that missing characteristic: adding ``LPMF`` and selecting it by
# distribution type in ``lpdf_provider`` would be the whole change.


def validate_sample(family: ParametricFamily, sample: NDArray[np.float64]) -> NDArray[np.float64]:
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
        Observed values, coerced to a float array.

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
    branch never calls ``validate()``, so such a fit is reported as successful.
    See the TODO above ``_build_result``.  Observation weights and censored data
    are not supported.
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
#      reparametrisation transform of the TODO above):
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

    Bounds and constraints are two separate mechanisms on purpose:
    ``param_bounds`` cannot validate anything, and a ``@constraint`` predicate
    cannot be turned into a box.  Probing a predicate numerically does not work
    either — it cannot distinguish "no upper bound" from "the bound is at the
    edge of the search region", and for a coupled constraint the answer depends
    on where the other parameter happens to sit.  SciPy reached the same
    conclusion and added ``_ShapeInfo`` alongside ``_argcheck``.

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


def clip_to_bounds(family: ParametricFamily, params: Parametrization) -> Parametrization:
    """
    Move a parametrization inside the declared bounds.

    Parameters
    ----------
    family : ParametricFamily
        Family whose bounds apply.
    params : Parametrization
        Candidate parameters, in the family's base parametrization.

    Returns
    -------
    Parametrization
        The same values, each clipped into its bound; unchanged if the family
        declares no bounds.
    """
    bounds, any_declared = _collect_bounds(family)
    if not any_declared:
        return params
    vec = to_vector(params)
    lows = np.array([low for low, _ in bounds], dtype=np.float64)
    highs = np.array([high for _, high in bounds], dtype=np.float64)
    return from_vector(type(params), np.clip(vec, lows, highs))


def _support_signature(support: Support | None) -> object:
    """Comparable fingerprint of a support, used to tell two supports apart."""
    if support is None:
        return None
    left = getattr(support, "left", None)
    right = getattr(support, "right", None)
    if left is not None or right is not None:
        return (
            "interval",
            float(left) if left is not None else None,
            float(right) if right is not None else None,
            bool(getattr(support, "left_closed", True)),
            bool(getattr(support, "right_closed", True)),
        )
    points = getattr(support, "points", None)
    if points is not None:
        return ("points", tuple(np.asarray(points).ravel().tolist()))
    return ("repr", type(support).__name__, repr(support))


def support_depends_on_params(family: ParametricFamily, sample: NDArray[np.float64]) -> bool:
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
    sample : NDArray[np.float64]
        Validated sample, used only to pick a plausible probe point.

    Returns
    -------
    bool
        ``True`` if the two supports differ.
    """
    probe = _probe_params(family, sample)
    perturbed = _perturb(family, probe)
    first = _support_signature(family.support_resolver(probe))
    second = _support_signature(family.support_resolver(perturbed))
    return first != second


def _probe_params(family: ParametricFamily, sample: NDArray[np.float64]) -> Parametrization:
    """A plausible point of the parameter space, obtained without raising."""
    try:
        return starting_point(family, sample)
    except (MLEError, ValueError, ArithmeticError):
        ones = project_onto_base(family, dict.fromkeys(field_names(family.base), 1.0))
        if ones is None:  # pragma: no cover - ones covers every field by construction
            raise
        return ones


def _perturb(family: ParametricFamily, params: Parametrization) -> Parametrization:
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


def _fixed_parameters(family: ParametricFamily) -> tuple[Mapping[str, Any], bool]:
    """
    Report which parameters a view has pinned, and in which coordinates.

    Returns
    -------
    tuple[Mapping[str, Any], bool]
        The fixed values, and whether they were fixed in the parent family's
        *base* parametrization.  A closed-form rule is written against the base
        parametrization, so it cannot be handed values expressed in any other.
    """
    from pysatl_core.families.parametric_family import PartialParametricFamily

    if not isinstance(family, PartialParametricFamily):
        return {}, True
    # A view's own ``base_parametrization_name`` is the parametrization the
    # parameters were fixed in: ``PartialParametricFamily`` registers exactly
    # that one and nothing else.
    fixed_in_base = (
        family.base_parametrization_name == family.parent_family.base_parametrization_name
    )
    return dict(family.fixed_parameters), fixed_in_base


def _optimizer_name(optimizer: str | Callable[..., Any]) -> str:
    """Readable name for a method string or a custom solver callable."""
    if isinstance(optimizer, str):
        return optimizer
    return getattr(optimizer, "__name__", None) or repr(optimizer)


def _success_verdict(result: OptimizeResult) -> bool | None:
    """
    The solver's convergence verdict, or ``None`` when it stated none.

    Three states, not two: converged, did not converge, and said nothing.
    ``scipy.optimize.minimize`` always fills ``success``, but a solver handed in
    through ``optimizer=`` is only obliged to return an ``OptimizeResult`` — the
    remaining fields are conventions its built-in methods follow.  The caller
    needs the third state to word ``MLEResult.message`` honestly, which is why
    this returns ``bool | None`` rather than collapsing to a bool here.

    A missing verdict is treated as failure downstream: claiming a convergence
    nobody reported would be the one failure mode this package exists to avoid.
    """
    stated = getattr(result, "success", None)
    return None if stated is None else bool(stated)


def _reported_message(result: OptimizeResult) -> str:
    """The solver's diagnostic text, or an empty string when it stated none."""
    message = getattr(result, "message", None)
    return "" if message is None else str(message)


def _supports_jac(optimizer: str | Callable[..., Any]) -> bool:
    """Whether ``minimize`` would actually use a gradient for this method."""
    if not isinstance(optimizer, str):
        return True
    return optimizer.lower() in _GRADIENT_METHODS


def _supports_bounds(optimizer: str | Callable[..., Any]) -> bool:
    """Whether ``minimize`` accepts ``bounds`` for this method."""
    if not isinstance(optimizer, str):
        return True
    return optimizer.lower() in _BOUNDED_METHODS


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


# TODO(mle): a fit is never checked against the family's own ``@constraint``
# predicates, so an estimate that violates them is reported with
# ``success=True``.  A constant sample is the reachable case: the closed form
# returns ``sigma = 0`` for ``Normal`` and ``lower_bound == upper_bound`` for
# ``ContinuousUniform``, both of which ``validate()`` rejects, yet ``fit``
# succeeds and only ``log_likelihood == -inf`` hints at the problem — the
# failure surfaces later, and elsewhere, as a ``ValueError`` from
# ``MLEResult.distribution``.
#
# Note this contradicts the reasoning in section 6.1 of the specification,
# which omitted a constant-sample check on the grounds that ``sigma > 0`` and
# ``lower_bound < upper_bound`` "already reject" such an estimate.  They do
# not: no code path on the closed-form branch calls ``validate()``.
#
# The fix belongs in ``_build_result``: call ``params.validate()`` and turn the
# resulting ``ValueError`` into a ``FitDataError`` naming the constraint and
# the sample property that caused it.  Left undone here because it changes the
# error contract of ``fit`` (a case that currently returns would start
# raising), which is the specification author's call to make.
def _build_result(
    family: ParametricFamily,
    params: Parametrization,
    sample: NDArray[np.float64],
    parametrization: ParametrizationName | None,
    *,
    method: str,
    optimizer: str | None,
    success: bool,
    message: str,
    n_iterations: int | None = None,
    n_function_evaluations: int | None = None,
) -> MLEResult:
    """Recompute the clean log-likelihood and pack everything into a result."""
    value = log_likelihood(family, params, sample)
    return MLEResult(
        family_name=family.name,
        params=_convert_parametrization(family, params, parametrization),
        log_likelihood=value,
        n_params=len(field_names(family.base)),
        n_observations=int(sample.size),
        method=cast("Any", method),
        optimizer=optimizer,
        success=success,
        message=message,
        n_iterations=n_iterations,
        n_function_evaluations=n_function_evaluations,
    )


def fit_family(
    family: ParametricFamily,
    sample: NDArray[np.float64],
    *,
    parametrization: ParametrizationName | None = None,
    optimizer: str | Callable[..., Any] | None = None,
    **options: Any,
) -> MLEResult:
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
    sample : NDArray[np.float64]
        Observed values; 1-D and finite.
    parametrization : ParametrizationName or None, optional
        Parametrization the estimate should be reported in.  Only the family's
        base parametrization is currently supported.
    optimizer : str or Callable or None, optional
        ``scipy.optimize.minimize`` method name, or a solver callable with the
        ``minimize`` signature.  Passing it forces the numerical path even when
        a closed-form solution exists.
    **options
        Extra keyword arguments forwarded to ``scipy.optimize.minimize``, for
        example ``tol=1e-12`` or ``options={"maxiter": 500}``.

    Returns
    -------
    MLEResult
        The estimate together with its log-likelihood, convergence flag and
        diagnostics.

    Raises
    ------
    ValueError
        If the sample is not 1-D or not finite.
    InsufficientDataError
        If there are fewer observations than free parameters.
    FitDataError
        If observations fall outside a support that does not depend on the
        parameters, or contradict the parameters fixed in a view.
    MLEError
        If the family declares no ``lpdf``, or no usable starting point exists.
    NotImplementedError
        If a non-base *parametrization* is requested.
    """
    sample = validate_sample(family, sample)

    probe = _probe_params(family, sample)
    if not support_depends_on_params(family, sample):
        _check_fixed_support(family, sample, probe)

    closed_form = family.mle
    fixed, fixed_in_base = _fixed_parameters(family)
    formula_applies = closed_form is not None and fixed_in_base

    if formula_applies and optimizer is None:
        params = closed_form(sample, fixed)  # type: ignore[misc]
        if params is not None:
            projected = project_onto_base(family, params.parameters)
            if projected is not None:
                return _build_result(
                    family,
                    projected,
                    sample,
                    parametrization,
                    method="closed_form",
                    optimizer=None,
                    success=True,
                    message="closed-form maximum likelihood solution",
                )

    notes: list[str] = []
    if formula_applies and optimizer is not None:
        note = (
            f"family '{family.name}' has a closed-form MLE, but 'optimizer="
            f"{_optimizer_name(optimizer)}' was passed, so the numerical path is used"
        )
        notes.append(note)
        warnings.warn(note, UserWarning, stacklevel=3)
        if family.name == FamilyName.CONTINUOUS_UNIFORM:
            # Unlike SciPy, which drops an explicitly passed ``optimizer`` on
            # the floor in every overridden ``fit`` (``_remove_optimizer_
            # parameters``), the request is honoured here — but not silently,
            # because for this family the numerical path is genuinely unsound.
            caveat = (
                "the numerical path is unreliable for a uniform family: the likelihood "
                "maximum sits on the boundary of the admissible region (at min(x) and "
                "max(x)) and the objective surface is discontinuous there, so a gradient "
                "method cannot reach it; prefer the closed-form solution by omitting "
                "'optimizer'"
            )
            notes.append(caveat)
            warnings.warn(caveat, UserWarning, stacklevel=3)

    return _fit_numerically(family, sample, parametrization, optimizer, notes=notes, **options)


def _fit_numerically(
    family: ParametricFamily,
    sample: NDArray[np.float64],
    parametrization: ParametrizationName | None,
    optimizer: str | Callable[..., Any] | None,
    *,
    notes: list[str],
    **options: Any,
) -> MLEResult:
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
    x0 = to_vector(starting_point(family, sample))
    bounds = resolve_bounds(family)

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

    method: str | Callable[..., Any] = optimizer if optimizer is not None else DEFAULT_OPTIMIZER
    result = _minimize(fun, x0, jac, bounds, method, options)

    n_iterations = getattr(result, "nit", None)
    verdict = _success_verdict(result)
    if optimizer is None and (not verdict or n_iterations == 0):
        notes.append(
            f"{_optimizer_name(method)} did not converge "
            f"(success={verdict}, nit={n_iterations}: {_reported_message(result)}); "
            f"fell back to {FALLBACK_OPTIMIZER}"
        )
        method = FALLBACK_OPTIMIZER
        result = _minimize(fun, x0, None, bounds, method, options)
        n_iterations = getattr(result, "nit", None)
        verdict = _success_verdict(result)

    if getattr(result, "x", None) is None:
        raise MLEError(
            f"Optimizer '{_optimizer_name(method)}' returned no 'x' field, so there is no "
            f"estimate to report. A solver passed as 'optimizer=' must return an "
            f"'OptimizeResult' carrying at least 'x'."
        )
    params = from_vector(family.base, np.asarray(result.x, dtype=np.float64))
    message = _reported_message(result)
    if message:
        notes.append(message)
    if verdict is None:
        notes.append("the optimizer reported no convergence flag, so success is not claimed")

    return _build_result(
        family,
        params,
        sample,
        parametrization,
        method="numeric",
        optimizer=_optimizer_name(method),
        success=bool(verdict),
        message="; ".join(notes),
        n_iterations=None if n_iterations is None else int(n_iterations),
        n_function_evaluations=(
            None if getattr(result, "nfev", None) is None else int(result.nfev)
        ),
    )


def _minimize(
    fun: Callable[[NDArray[np.float64]], float],
    x0: NDArray[np.float64],
    jac: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None,
    bounds: list[tuple[float, float]] | None,
    method: str | Callable[..., Any],
    options: Mapping[str, Any],
) -> OptimizeResult:
    """
    Call ``scipy.optimize.minimize``, passing only what the method can use.

    ``jac`` and ``bounds`` are withheld from methods that ignore them, so that
    naming, say, ``optimizer="Powell"`` does not fill the caller's output with
    SciPy warnings about arguments the method never asked for.
    """
    kwargs: dict[str, Any] = dict(options)
    if jac is not None and _supports_jac(method):
        kwargs["jac"] = jac
    if bounds is not None and _supports_bounds(method):
        kwargs["bounds"] = bounds
    # The SciPy stubs restrict ``method`` to a Literal of the built-in names,
    # but ``minimize`` also accepts a custom solver callable — which this
    # package documents and supports — so the call is made through an
    # untyped view of the same function.
    solver = cast("Callable[..., OptimizeResult]", minimize)
    return solver(fun, x0, method=method, **kwargs)


__all__ = [
    "DEFAULT_OPTIMIZER",
    "FALLBACK_OPTIMIZER",
    "fit_family",
    "resolve_bounds",
    "clip_to_bounds",
    "support_depends_on_params",
    "validate_sample",
]
