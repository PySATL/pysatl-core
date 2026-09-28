"""
The log-likelihood of a sample under a family, as one object.

This module is the numerical core of maximum likelihood estimation.  It knows
nothing about optimizers, about parameter fixing, or about who called it: it
turns a family plus a sample into an object that is callable as the objective,
answers for its own gradient, and reports the plain ``l(theta)`` a result
carries.

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

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

from pysatl_core.estimation.errors import EstimationError
from pysatl_core.estimation.parameters.vectors import (
    field_names,
    from_vector,
    satisfies_constraints,
)
from pysatl_core.families.parametrizations import Parametrization
from pysatl_core.types import DEFAULT_ANALYTICAL_COMPUTATION_LABEL, CharacteristicName

if TYPE_CHECKING:
    from pysatl_core.estimation.optimizers.protocol import GradientFunc
    from pysatl_core.families.parametric_family import (
        ParametricFamily,
        ParametricFamilyCharacteristic,
    )

type LpdfProvider = Callable[[Parametrization, NDArray[np.float64]], NDArray[np.float64]]


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


def _accepts_params_and_points(provider: ParametricFamilyCharacteristic[object, object]) -> bool:
    """
    Whether a characteristic provider has the ``(params, x)`` calling shape.

    ``distr_characteristics`` admits three shapes — ``f()``, ``f(params)`` and
    ``f(params, x)`` — and the declared type is their union, so nothing in it
    says which one a given ``lpdf`` entry is.  ``ParametricFamily.
    _bind_parametrization`` answers the same question the same way, by reading
    the signature; asking it here is what turns "the wrong shape" into a
    message at fit time instead of a bare ``TypeError`` thrown thousands of
    iterations deep inside the optimizer.
    """
    try:
        parameters = list(inspect.signature(provider).parameters.values())
    except (TypeError, ValueError):  # pragma: no cover - builtins without a signature
        # A provider whose signature cannot be read is given the benefit of the
        # doubt: refusing it would reject a legitimate C-implemented callable.
        return True
    positional = [
        p
        for p in parameters
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    takes_var_positional = any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in parameters)
    return len(positional) >= 2 or (len(positional) >= 1 and takes_var_positional)


def lpdf_provider(family: ParametricFamily) -> LpdfProvider:
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
    LpdfProvider
        Callable mapping ``(params, x)`` to log-density values.

    Raises
    ------
    EstimationError
        If the family does not declare ``lpdf``, or declares it with a calling
        shape other than ``(params, x)``.
    """
    by_parametrization = family.distr_characteristics.get(CharacteristicName.LPDF)
    base_name = family.base_parametrization_name
    labeled = None if by_parametrization is None else by_parametrization.get(base_name)
    if not labeled:
        raise EstimationError(
            f"Family '{family.name}' does not declare the 'lpdf' characteristic for its base "
            f"parametrization '{base_name}'. Maximum likelihood estimation needs the logarithm "
            f"of the density: add 'CharacteristicName.LPDF' to the family's "
            f"'distr_characteristics'. It cannot be derived here — the characteristic graph has "
            f"no 'pdf -> lpdf' edge — and 'log(pdf)' is not used as a substitute because it "
            f"loses precision in the tails."
        )
    # A label chosen here has to be deterministic: two fits of the same family
    # must sum the same log-density.  ``dict`` preserves insertion order, so
    # falling back to the first declared provider is reproducible, and the
    # default label is preferred whenever the family declares one.
    provider = labeled.get(DEFAULT_ANALYTICAL_COMPUTATION_LABEL) or next(iter(labeled.values()))
    if not _accepts_params_and_points(provider):
        raise EstimationError(
            f"Family '{family.name}' declares 'lpdf' for parametrization '{base_name}' as a "
            f"provider that takes no evaluation points "
            f"({inspect.signature(provider)}). Maximum likelihood estimation needs the "
            f"log-density *at the observations*, so the provider must have the "
            f"'(parameters, x)' shape that 'pdf', 'cdf' and 'ppf' use."
        )
    # Guarded by the arity check above: the declared union admits three calling
    # shapes and only this one survives it.
    return cast("LpdfProvider", provider)


@dataclass(frozen=True, slots=True, eq=False)
class Evaluation:
    """
    The split of a sample at one point of the parameter space.

    A named triple rather than a bare tuple: ``ev.n_unusable`` says what it is,
    where ``split[2]`` said only ``3``.  It is what the objective and the
    gradient both need, and the reason they can share one computation.


    Not comparable — ``eq=False`` — because a field holding a NumPy array has no
    sensible ``==``: the generated one compares field tuples, and a tuple
    comparison of two distinct-but-equal arrays yields an array, whose truth
    value then raises.  Identity comparison is both meaningful and what every
    use here wants.
    """

    usable: NDArray[np.bool_]
    """Mask of points the parameters explain: inside the support, finite density."""

    terms: NDArray[np.float64]
    """Log-density at every point; ``-inf`` outside the support."""

    n_unusable: int
    """How many points are not usable — what the penalty is charged for."""


@dataclass(slots=True)
class _Memo:
    """The last evaluation, kept so that value and gradient do not repeat it.

    Mutable, and the only mutable thing in this module.  It is held by a frozen
    :class:`LogLikelihood` as a private field: the object's *meaning* does not
    change, only what it remembers having computed.
    """

    key: bytes | None = None
    value: Evaluation | None = None


@dataclass(frozen=True, slots=True, eq=False)
class LogLikelihood:
    """
    The log-likelihood of one sample under one family, as an object.

    It is three things at once, and they belong together because they share
    both state and work:

    - the objective an optimizer minimises, ``-l(theta)`` with the
      out-of-support penalty — the object is callable, so it *is* an
      :data:`ObjectiveFunc`;
    - its gradient, over exactly the same subset of the sample;
    - the clean ``l(theta)`` reported in a result, through :meth:`at`.

    Why an object and not the three functions it replaces.  ``make_objective``
    used to return two closures over five shared variables, and
    ``log_likelihood`` was a third function that fetched the family's provider
    all over again.  The shared state was real but hidden, so nothing could be
    reused: on a gamma fit of 1000 points the value and the gradient between
    them walked the sample **14 times for 7 optimizer points**, because the
    gradient rebuilt the very mask the value had just computed at the same
    vector.  Here that mask is computed once per point and remembered (see
    :class:`_Memo`), and the provider is fetched once per object.

    Not comparable — ``eq=False`` — because ``sample`` is a NumPy array and the
    generated ``__eq__`` would compare field tuples: for two distinct-but-equal
    arrays that yields an array, whose truth value raises.  Identity is the
    only comparison this object has a use for.

    Attributes
    ----------
    family : ParametricFamily
        Family being fitted.
    sample : NDArray[np.float64]
        Validated 1-D sample.
    provider : LpdfProvider
        The family's raw log-density, fetched once.
    param_cls : type[Parametrization]
        The class a parameter vector is rebuilt into — the family's base, that
        is, its *free* parameters for a view.
    n_free : int
        Length of the parameter vector.
    has_gradient : bool
        Whether the family provides ``score``.  When it does not, the caller is
        expected to let the optimizer difference the objective numerically, and
        :meth:`gradient` refuses rather than returning a fabricated zero.
    """

    family: ParametricFamily
    sample: NDArray[np.float64]
    provider: LpdfProvider
    param_cls: type[Parametrization]
    n_free: int
    has_gradient: bool
    _memo: _Memo = field(default_factory=_Memo, repr=False)

    @classmethod
    def of(cls, family: ParametricFamily, sample: NDArray[np.float64]) -> LogLikelihood:
        """
        Build the log-likelihood of *sample* under *family*.

        Everything derived from the family is resolved here, once: the
        log-density provider, the parameter class, how many free parameters
        there are, and whether a gradient is available.

        Raises
        ------
        EstimationError
            If the family does not declare ``lpdf``.
        """
        return cls(
            family=family,
            sample=sample,
            provider=lpdf_provider(family),
            param_cls=family.base,
            n_free=len(field_names(family.base)),
            has_gradient=family.base_score is not None,
        )

    def __call__(self, vec: NDArray[np.float64]) -> float:
        """
        The objective: ``-l(theta)`` plus the out-of-support penalty.

        The value is

        ``-sum_{i usable} log f(x_i; theta) + n_unusable * OUT_OF_SUPPORT_PENALTY``

        where a point is *usable* when it lies inside the support implied by
        ``theta`` and its log-density is finite.  Parameters violating the
        family's constraints yield ``inf``, which is what keeps the optimizer
        away from the one-ULP-wide gap that an open bound such as ``sigma > 0``
        leaves once it is handed to the optimizer as a closed box.

        Parameters
        ----------
        vec : NDArray[np.float64]
            A flat parameter vector, ordered like
            ``param_cls.__dataclass_fields__`` — the only place in this package
            where a flat vector exists at all.
        """
        params = from_vector(self.param_cls, np.asarray(vec, dtype=np.float64))
        if not satisfies_constraints(params):
            return float(np.inf)
        ev = self._evaluate(params, vec)
        total = -float(ev.terms[ev.usable].sum()) + ev.n_unusable * OUT_OF_SUPPORT_PENALTY
        return total if np.isfinite(total) else float(np.inf)

    def gradient(self, vec: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Gradient of the objective, over exactly the same usable subset.

        Sharing the subset is not cosmetic: ``lpdf`` degrades pointwise to
        ``-inf`` while ``ParametricFamily.score`` raises on the whole array if
        one point is outside the support, so a differently masked gradient
        would hand the optimizer a value and a derivative that do not describe
        the same function.

        Raises
        ------
        EstimationError
            If the family provides no ``score``.  Returning zeros instead would
            tell the optimizer it had reached a stationary point — a fabricated
            answer, and a far worse failure than a refusal.  Check
            :attr:`has_gradient` first.
        """
        if not self.has_gradient:
            raise EstimationError(
                f"Family '{self.family.name}' provides no 'score', so this log-likelihood has "
                f"no analytical gradient. Test 'has_gradient' before asking for one, and let "
                f"the optimizer difference the objective numerically when there is none."
            )
        params = from_vector(self.param_cls, np.asarray(vec, dtype=np.float64))
        zeros = np.zeros(self.n_free, dtype=np.float64)
        if not satisfies_constraints(params):
            return zeros
        ev = self._evaluate(params, vec)
        if not ev.usable.any():
            return zeros
        try:
            scores = np.asarray(self.family.score(params, self.sample[ev.usable]), dtype=np.float64)
        except ValueError:
            return zeros
        grad = -scores.sum(axis=0)
        return np.where(np.isfinite(grad), grad, 0.0).astype(np.float64)

    @property
    def gradient_or_none(self) -> GradientFunc | None:
        """
        :meth:`gradient` when the family has one, ``None`` when it does not.

        What an optimizer wants: ``None`` is how "difference it yourself" is
        spelled to ``scipy.optimize.minimize``.
        """
        return self.gradient if self.has_gradient else None

    def at(self, params: Parametrization) -> float:
        """
        The plain log-likelihood ``l(theta)``, with no penalty term.

        This is the quantity reported in
        :attr:`~pysatl_core.estimation.result.MLEResult.log_likelihood` and the
        one the information criteria are built on, so it must not be
        contaminated by the optimisation penalty.

        Parameters
        ----------
        params : Parametrization
            Parameters, in the family's base parametrization.

        Returns
        -------
        float
            ``sum_i log f(x_i; theta)``, or ``-inf`` if any observation lies
            outside the support implied by *params* or has a non-finite
            log-density: such a sample genuinely has zero likelihood under
            these parameters.
        """
        ev = self._split(params)
        if ev.n_unusable:
            return float(-np.inf)
        return float(ev.terms[ev.usable].sum())

    def _evaluate(self, params: Parametrization, vec: NDArray[np.float64]) -> Evaluation:
        """
        :meth:`_split`, but computed at most once per parameter vector.

        ``scipy.optimize`` asks for the value and the gradient at the same
        point, one after the other; the bytes of the vector are an exact key,
        so the second question is answered from the first answer.
        """
        key = np.asarray(vec, dtype=np.float64).tobytes()
        memo = self._memo
        if memo.key == key and memo.value is not None:
            return memo.value
        value = self._split(params)
        memo.key = key
        memo.value = value
        return value

    def _split(self, params: Parametrization) -> Evaluation:
        """Split the sample into the part these parameters explain and the rest."""
        inside = self._inside_support(params)
        terms = np.full(self.sample.shape, -np.inf, dtype=np.float64)
        if inside.any():
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                terms[inside] = np.asarray(
                    self.provider(params, self.sample[inside]), dtype=np.float64
                )
        usable = inside & np.isfinite(terms)
        return Evaluation(
            usable=usable, terms=terms, n_unusable=int(self.sample.size - usable.sum())
        )

    def _inside_support(self, params: Parametrization) -> NDArray[np.bool_]:
        """Mask of sample points lying in the support implied by *params*."""
        support = self.family.support_resolver(params)
        if support is None:
            return np.ones(self.sample.shape, dtype=bool)
        return np.asarray(support.contains(self.sample), dtype=bool)


__all__ = [
    "OUT_OF_SUPPORT_PENALTY",
    "Evaluation",
    "LogLikelihood",
    "LpdfProvider",
    "lpdf_provider",
]
