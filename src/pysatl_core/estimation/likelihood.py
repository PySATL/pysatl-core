"""
Objective function, gradient and log-likelihood for maximum likelihood fits.

This module is the numerical core of :mod:`pysatl_core.estimation`.  It knows
nothing about optimizers, about parameter fixing, or about who called it: it
turns a family plus a sample into a pair of plain functions of a flat parameter
vector.  Keeping it free of those concerns is what makes it reusable later for
information criteria, Fisher information and goodness-of-fit statistics.

Two facts about the existing family API shape everything here.

1.  ``lpdf`` degrades pointwise — it returns ``-inf`` outside the support —
    while ``ParametricFamily.score`` raises ``ValueError`` if *any* evaluation
    point lies outside the support.  A gradient computed over the raw sample
    would therefore blow up on a single stray point where the objective merely
    grows.  Both are consequently evaluated over the *same* subset of the
    sample.
2.  Going through ``Distribution`` costs about fifteen times more per call than
    calling the family's raw analytical provider, and ``family.distribution()``
    validates parameters by raising ``ValueError``.  Inside an optimisation
    loop neither is acceptable, so the raw provider is used directly and
    constraint violations are reported as ``inf``, which is a value the line
    search can act on.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, cast

import numpy as np

from pysatl_core.estimation.errors import MLEError
from pysatl_core.types import DEFAULT_ANALYTICAL_COMPUTATION_LABEL, CharacteristicName

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from pysatl_core.families.parametric_family import ParametricFamily
    from pysatl_core.families.parametrizations import Parametrization

    type ObjectiveFunc = Callable[[NDArray[np.float64]], float]
    type GradientFunc = Callable[[NDArray[np.float64]], NDArray[np.float64]]


OUT_OF_SUPPORT_PENALTY: float = float(np.log(np.finfo(np.float64).max) * 100)
"""Penalty charged per observation that the current parameters cannot explain (~70978.27).

The rule is taken from SciPy (``_distn_infrastructure._nlff_and_penalty``):
points outside the support neither break the computation nor turn the objective
into ``inf``; they are charged a finite amount proportional to how many of them
there are.

The motive is directional information.  ``inf`` is a plateau — every point on it
looks equally bad, so a line search has nothing to descend along.  A finite
per-violation charge instead encodes "reduce the number of violations", which is
a direction.  The constant is large enough that no admissible configuration can
ever compete with one that explains more of the data.
"""


# TODO(mle): coupled constraints (``lower_bound < upper_bound`` in the uniform
# family) are checked by predicate here but never handed to the optimizer: a box
# of per-parameter bounds cannot express a relation between two parameters.  To
# support them, translate the family's coupled ``@constraint`` predicates into
# ``scipy.optimize.minimize(constraints=...)`` entries and switch the numerical
# path to SLSQP or trust-constr when any are present.

# TODO(mle): a family that does not declare ``lpdf`` is unsupported.  The
# characteristic graph has no ``pdf -> lpdf`` edge (see the comment on
# ``CharacteristicName.LPDF`` in ``types.py``), so nothing can derive the
# log-density for such a family.  Adding that edge to
# ``distributions/registry/`` would let this module fall back to the graph.  A
# local ``log(pdf)`` fallback is deliberately *not* used: it silently loses
# precision in the tails, exactly where the log-density matters.


def field_names(params_or_class: Parametrization | type[Parametrization]) -> tuple[str, ...]:
    """
    List a parametrization's fields, in declaration order.

    ``Parametrization`` is an ABC that the ``@parametrization`` decorator turns
    into a dataclass, and views get a class synthesised at runtime, so
    ``__dataclass_fields__`` exists on every concrete parametrization but is
    not part of the declared base class. Reading it through ``getattr`` keeps
    that fact in one place instead of scattering it across the package.

    Parameters
    ----------
    params_or_class : Parametrization or type[Parametrization]
        A parametrization instance or class.

    Returns
    -------
    tuple[str, ...]
        Field names in declaration order. For a view this is the free
        parameters only.
    """
    return tuple(getattr(params_or_class, "__dataclass_fields__", {}))


def to_vector(params: Parametrization) -> NDArray[np.float64]:
    """
    Flatten a parametrization into the vector the optimizer works on.

    Parameters
    ----------
    params : Parametrization
        Parameters to flatten.

    Returns
    -------
    NDArray[np.float64]
        Values ordered by ``params.__dataclass_fields__``, that is, by
        declaration order of the parametrization's fields.
    """
    return np.array(
        [float(getattr(params, name)) for name in field_names(params)],
        dtype=np.float64,
    )


def from_vector(param_cls: type[Parametrization], vec: NDArray[np.float64]) -> Parametrization:
    """
    Rebuild a parametrization from a flat vector.

    Parameters
    ----------
    param_cls : type[Parametrization]
        Parametrization class to instantiate.  For a view this is the
        lightweight class holding only the free parameters.
    vec : NDArray[np.float64]
        Values in the order of ``param_cls.__dataclass_fields__``.

    Returns
    -------
    Parametrization
        Instance carrying those values.  The instance is *not* validated:
        rejecting inadmissible parameters is the objective function's job, and
        it does so with a value rather than an exception.
    """
    names = field_names(param_cls)
    return param_cls(**{name: float(value) for name, value in zip(names, vec, strict=True)})


def satisfies_constraints(params: Parametrization) -> bool:
    """
    Check a parametrization against its own ``@constraint`` predicates.

    The predicates are evaluated directly rather than through
    ``Parametrization.validate`` (which raises) or ``family.distribution``
    (which raises *and* costs about fifteen times more): inside an optimisation
    loop the answer has to be a value.

    Parameters
    ----------
    params : Parametrization
        Candidate parameters.

    Returns
    -------
    bool
        ``True`` if every constraint holds.  A predicate that raises — which
        happens when the optimizer probes values a family's constraint was
        never written to handle, such as ``nan`` — counts as not holding.
    """
    for constraint in params.constraints:
        try:
            if not constraint.check(params):
                return False
        except (ValueError, ArithmeticError, TypeError):
            return False
    return True


def lpdf_provider(
    family: ParametricFamily,
) -> Callable[[Parametrization, NDArray[np.float64]], NDArray[np.float64]]:
    """
    Fetch the family's raw analytical log-density provider.

    The provider is taken straight out of ``family.distr_characteristics``
    rather than through a ``Distribution`` instance.  That is roughly fifteen
    times cheaper per call and, more importantly, never raises for inadmissible
    parameters — both properties matter once the function is called thousands
    of times inside an optimisation loop.

    Parameters
    ----------
    family : ParametricFamily
        Family to fit.  For a view, its base parametrization is the free
        parameter class and the provider already re-injects the fixed values.

    Returns
    -------
    Callable[[Parametrization, NDArray[np.float64]], NDArray[np.float64]]
        Callable mapping ``(params, x)`` to log-density values.

    Raises
    ------
    MLEError
        If the family does not declare ``lpdf``.
    """
    by_parametrization = family.distr_characteristics.get(CharacteristicName.LPDF)
    base_name = family.base_parametrization_name
    labeled = None if by_parametrization is None else by_parametrization.get(base_name)
    if not labeled:
        raise MLEError(
            f"Family '{family.name}' does not declare the 'lpdf' characteristic for its base "
            f"parametrization '{base_name}'. Maximum likelihood estimation needs the logarithm "
            f"of the density: add 'CharacteristicName.LPDF' to the family's "
            f"'distr_characteristics'. It cannot be derived here — the characteristic graph has "
            f"no 'pdf -> lpdf' edge — and 'log(pdf)' is not used as a substitute because it "
            f"loses precision in the tails."
        )
    provider = labeled.get(DEFAULT_ANALYTICAL_COMPUTATION_LABEL) or next(iter(labeled.values()))
    return cast("Callable[[Parametrization, NDArray[np.float64]], NDArray[np.float64]]", provider)


def _inside_support(
    family: ParametricFamily, params: Parametrization, sample: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """Boolean mask of sample points lying in the support implied by ``params``."""
    support = family.support_resolver(params)
    if support is None:
        return np.ones(sample.shape, dtype=bool)
    return np.asarray(support.contains(sample), dtype=bool)


def _usable_mask(
    family: ParametricFamily,
    params: Parametrization,
    sample: NDArray[np.float64],
    provider: Callable[[Parametrization, NDArray[np.float64]], NDArray[np.float64]],
) -> tuple[NDArray[np.bool_], NDArray[np.float64], int]:
    """
    Split the sample into the part the current parameters explain and the rest.

    Returns
    -------
    tuple[NDArray[np.bool_], NDArray[np.float64], int]
        The mask of usable points, their log-density values, and the number of
        points that are not usable (outside the support, or carrying a
        non-finite log-density).
    """
    inside = _inside_support(family, params, sample)
    terms = np.full(sample.shape, -np.inf, dtype=np.float64)
    if inside.any():
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            terms[inside] = np.asarray(provider(params, sample[inside]), dtype=np.float64)
    usable = inside & np.isfinite(terms)
    return usable, terms, int(sample.size - usable.sum())


def make_objective(
    family: ParametricFamily, sample: NDArray[np.float64]
) -> tuple[ObjectiveFunc, GradientFunc | None]:
    """
    Build the negative log-likelihood and its gradient as functions of a vector.

    Both returned callables take a flat parameter vector ordered like
    ``family.base.__dataclass_fields__`` — the only place in this package where
    a flat vector exists at all.

    The objective is

    ``-sum_{i usable} log f(x_i; theta) + n_unusable * OUT_OF_SUPPORT_PENALTY``

    where a point is *usable* when it lies inside the support implied by
    ``theta`` and its log-density is finite.  Parameters violating the family's
    constraints yield ``inf``, which is what keeps the optimizer away from the
    one-ULP-wide gap that an open bound such as ``sigma > 0`` leaves once it is
    handed to the optimizer as a closed box.

    The gradient is evaluated over exactly the same usable subset.  This is not
    cosmetic: ``lpdf`` degrades pointwise to ``-inf`` while ``score`` raises on
    the whole array if one point is outside the support, so a differently
    masked gradient would hand the optimizer a value/derivative pair that do
    not describe the same function.

    Parameters
    ----------
    family : ParametricFamily
        Family being fitted.
    sample : NDArray[np.float64]
        Validated 1-D sample.

    Returns
    -------
    tuple[ObjectiveFunc, GradientFunc or None]
        The objective and its analytical gradient.  The gradient is ``None``
        when the family provides no ``score``; the caller is then expected to
        let the optimizer difference the objective numerically.

    Raises
    ------
    MLEError
        If the family does not declare ``lpdf``.
    """
    provider = lpdf_provider(family)
    param_cls = family.base
    n_free = len(field_names(param_cls))
    has_score = family.base_score is not None

    def fun(vec: NDArray[np.float64]) -> float:
        params = from_vector(param_cls, np.asarray(vec, dtype=np.float64))
        if not satisfies_constraints(params):
            return float(np.inf)
        usable, terms, n_bad = _usable_mask(family, params, sample, provider)
        total = -float(terms[usable].sum()) + n_bad * OUT_OF_SUPPORT_PENALTY
        return total if np.isfinite(total) else float(np.inf)

    if not has_score:
        return fun, None

    def jac(vec: NDArray[np.float64]) -> NDArray[np.float64]:
        params = from_vector(param_cls, np.asarray(vec, dtype=np.float64))
        zeros = np.zeros(n_free, dtype=np.float64)
        if not satisfies_constraints(params):
            # ``fun`` already rejects this point with ``inf``; the gradient is
            # never used to move away from it, so its value only has to be finite.
            return zeros
        usable, _terms, _n_bad = _usable_mask(family, params, sample, provider)
        if not usable.any():
            return zeros
        try:
            scores = np.asarray(family.score(params, sample[usable]), dtype=np.float64)
        except ValueError:
            # A family may declare its score undefined on a boundary point that
            # still carries a finite log-density.  Reporting a zero gradient
            # there is honest enough for the caller: a gradient method reads it
            # as a stationary point, stops, and the fallback policy in
            # ``mle.py`` retries with a derivative-free method.
            return zeros
        grad = -scores.sum(axis=0)
        return np.where(np.isfinite(grad), grad, 0.0).astype(np.float64)

    return fun, jac


def log_likelihood(
    family: ParametricFamily, params: Parametrization, sample: NDArray[np.float64]
) -> float:
    """
    Compute the plain log-likelihood, with no penalty term.

    This is the quantity reported in :attr:`MLEResult.log_likelihood` and the
    one information criteria are built on, so it must not be contaminated by
    the optimisation penalty.

    Parameters
    ----------
    family : ParametricFamily
        Family the parameters belong to.
    params : Parametrization
        Parameters, in the family's base parametrization.
    sample : NDArray[np.float64]
        Observed values.

    Returns
    -------
    float
        ``sum_i log f(x_i; theta)``, or ``-inf`` if any observation lies
        outside the support implied by ``params`` or has a non-finite
        log-density: such a sample genuinely has zero likelihood under these
        parameters.

    Raises
    ------
    MLEError
        If the family does not declare ``lpdf``.
    """
    provider = lpdf_provider(family)
    usable, terms, n_bad = _usable_mask(family, params, sample, provider)
    if n_bad:
        return float(-np.inf)
    return float(terms[usable].sum())


__all__ = [
    "OUT_OF_SUPPORT_PENALTY",
    "field_names",
    "make_objective",
    "log_likelihood",
    "lpdf_provider",
    "satisfies_constraints",
    "to_vector",
    "from_vector",
]
