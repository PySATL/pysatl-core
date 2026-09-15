"""
The one door through which this package reaches ``scipy.optimize``.

``scipy.optimize.OptimizeResult`` is a bag of attributes — its stubs declare
``__getattr__(str) -> Any``, so letting such an object travel through the
estimator would make every value derived from it unverifiable.
:class:`OptimizerOutcome` normalises it once, here, and everything downstream
works with declared types.

The same applies in the other direction: what the caller may name as an
optimizer, and what may be forwarded to ``minimize``, are spelled out as
:data:`MinimizeMethod`, :class:`MinimizeSolver` and :class:`MinimizeOptions`
rather than left as ``str | Callable[..., Any]`` and ``**options: Any``.

Keeping all of it in its own module is what lets
:mod:`pysatl_core.estimation.mle` be about the *policy* of a fit — the order of
the steps, closed form against numerical search, the fallback rule — instead of
also being about SciPy's calling conventions.  It is also the part most likely
to move when SciPy or its stubs change, and it moves alone.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    Protocol,
    TypedDict,
    cast,
    runtime_checkable,
)

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from pysatl_core.estimation.errors import MLEError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from numpy.typing import NDArray

    from pysatl_core.estimation.likelihood import GradientFunc, ObjectiveFunc


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
    Keyword arguments :func:`~pysatl_core.estimation.mle.fit_family` forwards to SciPy.

    Declaring the shape is what makes ``tolerance=1e-9`` — a misspelling of
    ``tol`` — an error at the call site.  Passed as ``**options: Any`` it used
    to travel silently into SciPy, which either ignored it or complained far
    from where it was written.

    ``jac`` and ``bounds`` are deliberately absent: :func:`run_optimizer`
    decides both, from the family's ``score`` and ``param_bounds`` and from what
    the chosen method can actually use, and a value passed here would be
    overwritten.

    ``args``, ``hess`` and ``hessp`` are absent for the same reason.  The
    objective built by :func:`~pysatl_core.estimation.likelihood.make_objective`
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


def _supports_jac(optimizer: MinimizeMethod | MinimizeSolver) -> bool:
    """Whether ``minimize`` would actually use a gradient for this method."""
    if not isinstance(optimizer, str):
        return True
    return optimizer.lower() in _GRADIENT_METHODS


def _supports_bounds(optimizer: MinimizeMethod | MinimizeSolver) -> bool:
    """Whether ``minimize`` accepts ``bounds`` for this method."""
    if not isinstance(optimizer, str):
        return True
    return optimizer.lower() in _BOUNDED_METHODS


@dataclass(frozen=True, slots=True)
class OptimizerOutcome:
    """
    A solver's result, normalised into declared types.

    The single point of contact with ``scipy.optimize.OptimizeResult``, which
    answers any attribute access with ``Any``.  Reading the fields here, once,
    means a misspelling is caught — and it means the package states plainly what
    it needs from a solver and what it merely reads when offered.

    Attributes
    ----------
    x : NDArray[np.float64]
        The estimate.  The only field a solver must provide.
    success : bool or None
        Three states, not two: converged, did not converge, and said nothing.
        ``scipy.optimize.minimize`` always fills ``success``, but a solver
        handed in through ``optimizer=`` is only obliged to return an
        ``OptimizeResult`` — the remaining fields are conventions its built-in
        methods follow.  The caller needs the third state to word
        ``MLEResult.message`` honestly, so it is not collapsed into a ``bool``
        here.  Downstream a missing verdict counts as failure: claiming a
        convergence nobody reported would be the one failure mode this package
        exists to avoid.
    message : str
        The solver's diagnostic text, or empty when it stated none.
    n_iterations, n_function_evaluations : int or None
        Counters, when the solver reports them.
    """

    x: NDArray[np.float64]
    success: bool | None
    message: str
    n_iterations: int | None
    n_function_evaluations: int | None

    @classmethod
    def from_scipy(cls, result: OptimizeResult[object], name: str) -> OptimizerOutcome:
        """
        Read an ``OptimizeResult``, narrowing every field to a declared type.

        The fields are fetched through the ``Mapping`` interface rather than as
        attributes, because ``result.get(...)`` yields ``object`` where
        ``result.nit`` yields ``Any``: ``object`` forces the narrowing below to
        be written down, and refuses anything that does not fit.

        Parameters
        ----------
        result : OptimizeResult
            Whatever the solver returned.
        name : str
            Name of the optimizer, quoted in the error message when the result
            is unusable.

        Raises
        ------
        MLEError
            If the result carries no ``x``, or an ``x`` that is not an array of
            numbers — there is then no estimate to report.
        """
        raw_x: object = result.get("x")
        if raw_x is None:
            raise MLEError(
                f"Optimizer '{name}' returned no 'x' field, so there is no "
                f"estimate to report. A solver passed as 'optimizer=' must return an "
                f"'OptimizeResult' carrying at least 'x'."
            )
        if not isinstance(raw_x, (np.ndarray, list, tuple)):
            raise MLEError(
                f"Optimizer '{name}' returned an 'x' of type "
                f"{type(raw_x).__name__}, which is not an array of parameter values. A "
                f"solver passed as 'optimizer=' must return an 'OptimizeResult' whose 'x' "
                f"is the estimate, in the order of the family's base parameters."
            )
        raw_success: object = result.get("success")
        raw_message: object = result.get("message")
        raw_iterations: object = result.get("nit")
        raw_evaluations: object = result.get("nfev")
        return cls(
            x=np.asarray(raw_x, dtype=np.float64),
            success=None if raw_success is None else bool(raw_success),
            message="" if raw_message is None else str(raw_message),
            n_iterations=_as_count(raw_iterations),
            n_function_evaluations=_as_count(raw_evaluations),
        )


def _as_count(value: object) -> int | None:
    """A solver's counter as an ``int``, or ``None`` when it reported none.

    A value of an unexpected type is read as "not reported" rather than
    coerced: a counter this package cannot interpret is exactly as informative
    as a missing one, and guessing would put a fabricated number in the result.
    """
    if isinstance(value, (int, np.integer)):
        return int(value)
    return None


def run_optimizer(
    fun: ObjectiveFunc,
    x0: NDArray[np.float64],
    *,
    jac: GradientFunc | None,
    bounds: list[tuple[float, float]] | None,
    method: MinimizeMethod | MinimizeSolver,
    options: MinimizeOptions,
) -> OptimizerOutcome:
    """
    Minimise *fun* from *x0* and return the result in declared types.

    The call and the normalisation are one step on purpose: an
    ``OptimizeResult`` that escaped this function would carry ``Any`` into the
    caller, which is the whole thing this module exists to prevent.

    ``jac`` and ``bounds`` are withheld from methods that ignore them, so that
    naming, say, ``optimizer="Powell"`` does not fill the caller's output with
    SciPy warnings about arguments the method never asked for.

    Parameters
    ----------
    fun : ObjectiveFunc
        Objective to minimise.
    x0 : NDArray[np.float64]
        Starting point, in the order of the family's base parameters.
    jac : GradientFunc or None
        Analytical gradient, when the family provides one.
    bounds : list[tuple[float, float]] or None
        Box bounds, or ``None`` to run unbounded.
    method : MinimizeMethod or MinimizeSolver
        Method name, or a caller-supplied solver.
    options : MinimizeOptions
        Extra keyword arguments forwarded to ``scipy.optimize.minimize``.

    Returns
    -------
    OptimizerOutcome
        The estimate and whatever diagnostics the solver reported.

    Raises
    ------
    MLEError
        If the solver returns no usable ``x``.
    """
    kwargs: dict[str, object] = dict(options)
    if jac is not None and _supports_jac(method):
        kwargs["jac"] = jac
    if bounds is not None and _supports_bounds(method):
        kwargs["bounds"] = bounds
    result = _minimize_impl(fun, x0, method=method, **kwargs)
    return OptimizerOutcome.from_scipy(result, optimizer_name(method))


__all__ = [
    "DEFAULT_OPTIMIZER",
    "FALLBACK_OPTIMIZER",
    "MinimizeCallback",
    "MinimizeMethod",
    "MinimizeOptions",
    "MinimizeSolver",
    "OptimizerOutcome",
    "optimizer_name",
    "run_optimizer",
]
