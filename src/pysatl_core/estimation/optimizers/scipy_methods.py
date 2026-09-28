"""
The single point of contact with ``scipy.optimize``.

What the caller may name as an optimizer, and what may be forwarded to
``minimize``, are spelled out as :data:`MinimizeMethod`,
:class:`MinimizeSolver` and :class:`MinimizeOptions` rather than left as
``str | Callable[..., Any]`` and ``**options: Any``.

Keeping all of it in one module is what lets a policy such as
:class:`~pysatl_core.estimation.optimizers.fallback.WithFallback`, and the
estimation methods above it, be about decisions instead of about SciPy's
calling conventions.  It is also the part most likely to move when SciPy or its
stubs change, and it moves alone.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypedDict, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize

from pysatl_core.estimation.optimizers.outcome import OptimizerOutcome
from pysatl_core.estimation.optimizers.protocol import Optimizer

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pysatl_core.estimation.optimizers.protocol import GradientFunc, ObjectiveFunc


type MinimizeMethod = Literal[
    "Nelder-Mead",
    "Powell",
    "CG",
    "BFGS",
    "Newton-CG",
    "L-BFGS-B",
    "TNC",
    "COBYLA",
    "COBYQA",
    "SLSQP",
    "trust-constr",
    "dogleg",
    "trust-ncg",
    "trust-exact",
    "trust-krylov",
]
"""Method names ``scipy.optimize.minimize`` understands.

Spelled as SciPy spells them.  SciPy itself compares method names
case-insensitively and so does this module, but the alias fixes one spelling so
that a typo is caught: ``optimizer="L-BFSG-B"`` is a type error here, where a
plain ``str`` would have carried it all the way into SciPy.
"""


@runtime_checkable
class MinimizeSolver(Protocol):
    """
    A caller-supplied solver with the ``scipy.optimize.minimize`` calling shape.

    SciPy invokes a custom ``method`` as ``method(fun, x0, args=..., **options)``
    and expects an ``OptimizeResult`` back.  Spelling that out is what replaces
    ``Callable[..., Any]``, which checked neither the arguments nor the result.

    The first two parameters are matched by name, as SciPy's own documentation
    spells them: a solver declaring ``**kwargs`` alongside positional-only
    parameters could otherwise be handed ``fun=`` twice, and a checker is right
    to refuse it.  Any further parameters — ``args``, the method's own options —
    are free.

    The only field this package requires of the returned object is ``x``; the
    rest are read when present.  See :class:`OptimizerOutcome`.
    """

    def __call__(
        self,
        fun: ObjectiveFunc,
        x0: NDArray[np.float64],
        **kwargs: object,
    ) -> OptimizeResult[object]: ...


type MinimizeCallback = (
    Callable[[NDArray[np.float64]], None] | Callable[[OptimizeResult[object]], None]
)
"""The two callback shapes ``scipy.optimize.minimize`` documents.

Older methods call back with the current iterate, newer ones with an
intermediate ``OptimizeResult``; ``Callable[..., object]`` would have covered
both by checking neither.
"""


@runtime_checkable
class _NamedCallable(Protocol):
    """A callable that carries a ``__name__``, as plain functions and classes do."""

    __name__: str


class MinimizeOptions(TypedDict, total=False):
    """
    Keyword arguments :class:`~pysatl_core.estimation.methods.mle.MLE` forwards to SciPy.

    Declaring the shape is what makes ``tolerance=1e-9`` — a misspelling of
    ``tol`` — an error at the call site.  Passed as ``**options: Any`` it used
    to travel silently into SciPy, which either ignored it or complained far
    from where it was written.

    ``jac`` and ``bounds`` are deliberately absent: :func:`run_optimizer`
    decides both, from the family's ``score`` and ``param_bounds`` and from what
    the chosen method can actually use, and a value passed here would be
    overwritten.

    ``args``, ``hess`` and ``hessp`` are absent for the same reason.  The
    objective built by :func:`~pysatl_core.estimation.methods.mle.likelihood.make_objective`
    is a function of the parameter vector alone, so an ``args`` tuple would be
    forwarded to SciPy and reach ``fun(x, *args)`` as a ``TypeError`` raised
    from inside the objective — the opposite of what declaring this shape is
    for.  ``hess`` and ``hessp`` are ignored by both the default method and the
    fallback, and naming them only fills the caller's output with SciPy's
    "does not use Hessian information" warnings, the very noise
    :func:`run_optimizer` withholds ``jac`` and ``bounds`` to avoid.

    ``constraints`` stays: it neither fails nor warns, and it is what the
    coupled-constraint work in ``docs/estimation_todos.md`` #2 will need.
    """

    tol: float
    options: Mapping[str, object]
    callback: MinimizeCallback
    constraints: object


class _MinimizeFunc(Protocol):
    """``scipy.optimize.minimize``, with ``method`` widened to admit a solver.

    The SciPy stubs type ``method`` as a ``Literal`` of the built-in names, but
    ``minimize`` also accepts a callable — which this package documents and
    supports — so it is reached through this protocol instead.  Everything
    except ``method`` keeps the stubs' meaning.
    """

    def __call__(
        self,
        fun: ObjectiveFunc,
        x0: NDArray[np.float64],
        /,
        *,
        method: MinimizeMethod | MinimizeSolver,
        **kwargs: object,
    ) -> OptimizeResult[object]: ...


_minimize_impl: Final[_MinimizeFunc] = cast("_MinimizeFunc", minimize)


DEFAULT_OPTIMIZER: Final[Literal["L-BFGS-B"]] = "L-BFGS-B"
"""Optimizer used when the caller names none and no closed form applies.

With the analytical gradient from ``ParametricFamily.score`` it is markedly
more economical than a simplex method on smooth problems — on a normal sample
of 1000 points it reaches the same estimate in 14 objective evaluations against
147 for Nelder-Mead.
"""

FALLBACK_OPTIMIZER: Final[Literal["Nelder-Mead"]] = "Nelder-Mead"
"""Derivative-free optimizer retried when the default reports failure.

A quasi-Newton method models the objective through its derivatives and stalls
on a surface that is not smooth.  Where a parameter moves the boundary of the
support, the objective is piecewise constant in the number of unexplained
observations — a staircase, on which a line search finds no improvement and
returns the starting point untouched.
"""

_GRADIENT_METHODS: Final[frozenset[str]] = frozenset(
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

_BOUNDED_METHODS: Final[frozenset[str]] = frozenset(
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


def optimizer_name(optimizer: MinimizeMethod | MinimizeSolver) -> str:
    """
    Readable name for a method string or a custom solver callable.

    ``__name__`` is read through a protocol rather than probed with ``getattr``:
    a plain function, a method and a class all declare it, while a
    ``functools.partial`` or a callable instance does not, and the two cases are
    told apart by the check instead of by a default value.
    """
    if isinstance(optimizer, str):
        return optimizer
    if isinstance(optimizer, _NamedCallable):
        return optimizer.__name__
    return repr(optimizer)


@dataclass(frozen=True, slots=True)
class ScipyMethod:
    """
    One of ``scipy.optimize.minimize``'s built-in methods.

    ``jac`` and ``bounds`` are withheld from a method that ignores them, so
    that naming, say, ``"Powell"`` does not fill the caller's output with SciPy
    warnings about arguments the method never asked for.  Which method uses
    what is the class's own business — it is why the two tables live beside it
    rather than in the code that calls it.

    Parameters
    ----------
    method : MinimizeMethod
        The method name, spelled as SciPy spells it.
    options : MinimizeOptions, optional
        Extra keyword arguments forwarded to ``minimize``.
    use_gradient : bool, optional
        Whether to pass a gradient at all when the method could use one.
        ``False`` is how a derivative-free retry is expressed: see
        :class:`WithFallback`.
    """

    method: MinimizeMethod
    options: MinimizeOptions = field(default_factory=MinimizeOptions)
    use_gradient: bool = True

    @property
    def name(self) -> str:
        """The method name."""
        return self.method

    def minimize(
        self,
        objective: ObjectiveFunc,
        x0: NDArray[np.float64],
        *,
        gradient: GradientFunc | None,
        bounds: list[tuple[float, float]] | None,
    ) -> OptimizerOutcome:
        """Run ``scipy.optimize.minimize`` and normalise what it returns."""
        kwargs: dict[str, object] = dict(self.options)
        if gradient is not None and self.use_gradient and self._uses_gradient:
            kwargs["jac"] = gradient
        if bounds is not None and self._accepts_bounds:
            kwargs["bounds"] = bounds
        result = _minimize_impl(objective, x0, method=self.method, **kwargs)
        return OptimizerOutcome.from_scipy(result, self.name)

    @property
    def _uses_gradient(self) -> bool:
        """Whether ``minimize`` would actually make use of ``jac`` here."""
        return self.method.lower() in _GRADIENT_METHODS

    @property
    def _accepts_bounds(self) -> bool:
        """Whether ``minimize`` accepts ``bounds`` for this method."""
        return self.method.lower() in _BOUNDED_METHODS


@dataclass(frozen=True, slots=True)
class CustomSolver:
    """
    A solver the caller supplied, with the ``minimize`` calling shape.

    It is reached through ``minimize(method=...)``, exactly as SciPy documents,
    and is handed both the gradient and the bounds when they exist: a caller
    who wrote the solver knows what it does with them, and this class has no
    table to consult.

    Parameters
    ----------
    solver : MinimizeSolver
        The callable.
    options : MinimizeOptions, optional
        Extra keyword arguments forwarded to ``minimize``.
    """

    solver: MinimizeSolver
    options: MinimizeOptions = field(default_factory=MinimizeOptions)

    @property
    def name(self) -> str:
        """The callable's ``__name__``, or its ``repr`` when it has none."""
        return optimizer_name(self.solver)

    def minimize(
        self,
        objective: ObjectiveFunc,
        x0: NDArray[np.float64],
        *,
        gradient: GradientFunc | None,
        bounds: list[tuple[float, float]] | None,
    ) -> OptimizerOutcome:
        """Hand the problem to the caller's solver through ``minimize``."""
        kwargs: dict[str, object] = dict(self.options)
        if gradient is not None:
            kwargs["jac"] = gradient
        if bounds is not None:
            kwargs["bounds"] = bounds
        result = _minimize_impl(objective, x0, method=self.solver, **kwargs)
        return OptimizerOutcome.from_scipy(result, self.name)


def optimizer_for(
    spec: MinimizeMethod | MinimizeSolver | Optimizer,
    options: MinimizeOptions,
) -> Optimizer:
    """
    Turn whatever a caller named into an :class:`Optimizer`.

    The three spellings are a method name, a solver callable, and a ready-made
    optimizer.  The order of the checks matters: every ``Optimizer`` here is
    also an object, and :class:`MinimizeSolver` matches any callable at all, so
    the narrowest test comes first.

    Parameters
    ----------
    spec : MinimizeMethod or MinimizeSolver or Optimizer
        What the caller named.
    options : MinimizeOptions
        Options to attach, ignored when *spec* is already an optimizer — it
        carries its own.

    Returns
    -------
    Optimizer
        *spec* itself when it is already one, otherwise a wrapper around it.
    """
    if isinstance(spec, str):
        return ScipyMethod(spec, options)
    if isinstance(spec, Optimizer):
        return spec
    return CustomSolver(spec, options)


__all__ = [
    "DEFAULT_OPTIMIZER",
    "FALLBACK_OPTIMIZER",
    "CustomSolver",
    "MinimizeCallback",
    "MinimizeMethod",
    "MinimizeOptions",
    "MinimizeSolver",
    "ScipyMethod",
    "optimizer_for",
    "optimizer_name",
]
