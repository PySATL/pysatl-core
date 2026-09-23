"""
Exception taxonomy for parameter estimation.

The split follows one rule: an exception marks a situation an estimator cannot
recover from — data that no parameter value can explain, a sample that is not
usable at all, a family that lacks what the method needs.  A merely
unsuccessful optimisation is *not* such a situation: it is reported as a
:class:`~pysatl_core.estimation.result.FitResult` with ``success=False`` and a
filled ``message``.

This is a deliberate departure from ``scipy.stats``, where the two are mixed:
``rv_continuous.fit`` returns a bare tuple that carries no convergence flag at
all, yet raises ``FitError`` when the optimiser wanders outside the admissible
parameter region.  A caller there cannot tell "did not converge" from "wrong
data" without parsing an exception message.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


class EstimationError(Exception):
    """Base exception for parameter estimation.

    It was called ``MLEError`` while maximum likelihood was the only method,
    and the name was already wrong by then: it is raised where no start can be
    assembled and where a solver returns no estimate, neither of which is about
    likelihood.  None of the situations it marks is specific to one criterion,
    so none of them earns a per-method class.
    """


class FitDataError(EstimationError, ValueError):
    """
    Data are incompatible with the family or with the fixed parameters.

    Raised, for example, when a sample holds a negative value while a gamma
    family is being fitted (its support ``[0, inf)`` does not depend on the
    parameters, so no parameter value can accommodate that point), or when
    ``min(x) < lower_bound`` for a uniform family whose ``lower_bound`` was
    fixed through :meth:`~pysatl_core.families.parametric_family.ParametricFamily.view`.
    """


class InsufficientDataError(EstimationError, ValueError):
    """Fewer observations than the number of parameters being estimated."""


__all__ = [
    "EstimationError",
    "FitDataError",
    "InsufficientDataError",
]
