"""
A solver's result, normalised into declared types.

``scipy.optimize.OptimizeResult`` is a bag of attributes — its stubs declare
``__getattr__(str) -> Any``, so letting such an object travel through an
estimator would make every value derived from it unverifiable.
:class:`OptimizerOutcome` normalises it once, here, and everything downstream
works with declared types.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult

from pysatl_core.estimation.errors import EstimationError

if TYPE_CHECKING:
    pass


@dataclass(frozen=True, slots=True, eq=False)
class OptimizerOutcome:
    """
    A solver's result, normalised into declared types.

    Not comparable — ``eq=False`` — because ``x`` is a NumPy array: the
    generated ``__eq__`` compares field tuples, and a tuple comparison of two
    distinct-but-equal arrays yields an array whose truth value raises.

    The single point of contact with ``scipy.optimize.OptimizeResult``, which
    answers any attribute access with ``Any``.  Reading the fields here, once,
    means a misspelling is caught — and it means the package states plainly what
    it needs from a solver and what it merely reads when offered.

    Attributes
    ----------
    x : NDArray[np.float64]
        The estimate.  The only field a solver must provide.
    optimizer : str
        Name of the optimizer that produced *this* estimate.  It is carried
        here rather than tracked by the caller because a composite optimizer
        such as :class:`WithFallback` decides which of several actually ran,
        and only it knows the answer.
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
    notes : tuple[str, ...]
        What the optimizer wants recorded in the fit's message beyond its own
        ``message`` — an optimizer that was substituted for another explains
        the substitution here.  Empty for a plain run.
    """

    x: NDArray[np.float64]
    optimizer: str
    success: bool | None
    message: str
    n_iterations: int | None
    n_function_evaluations: int | None
    notes: tuple[str, ...] = ()

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
        EstimationError
            If the result carries no ``x``, or an ``x`` that is not an array of
            numbers — there is then no estimate to report.
        """
        raw_x: object = result.get("x")
        if raw_x is None:
            raise EstimationError(
                f"Optimizer '{name}' returned no 'x' field, so there is no "
                f"estimate to report. A solver passed as 'optimizer=' must return an "
                f"'OptimizeResult' carrying at least 'x'."
            )
        if not isinstance(raw_x, (np.ndarray, list, tuple)):
            raise EstimationError(
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
            optimizer=name,
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


__all__ = ["OptimizerOutcome"]
