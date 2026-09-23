"""
Searching for the minimum of an objective, in declared types.

An optimizer is an object: it carries its own configuration, answers for what
it can use, and — because a *policy* is itself an optimizer — composes.  None
of them knows what is being minimised, so any estimation method can reuse any
of them.

``protocol`` declares what an optimizer is and what it consumes, ``outcome``
normalises what a solver returns, ``scipy_methods`` is the one door to
``scipy.optimize``, and ``fallback`` is a policy built from two others.
"""

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_core.estimation.optimizers.fallback import WithFallback
from pysatl_core.estimation.optimizers.outcome import OptimizerOutcome
from pysatl_core.estimation.optimizers.protocol import GradientFunc, ObjectiveFunc, Optimizer
from pysatl_core.estimation.optimizers.scipy_methods import (
    DEFAULT_OPTIMIZER,
    FALLBACK_OPTIMIZER,
    CustomSolver,
    MinimizeCallback,
    MinimizeMethod,
    MinimizeOptions,
    MinimizeSolver,
    ScipyMethod,
    optimizer_for,
    optimizer_name,
)

__all__ = [
    "DEFAULT_OPTIMIZER",
    "FALLBACK_OPTIMIZER",
    "CustomSolver",
    "GradientFunc",
    "MinimizeCallback",
    "MinimizeMethod",
    "MinimizeOptions",
    "MinimizeSolver",
    "ObjectiveFunc",
    "Optimizer",
    "OptimizerOutcome",
    "ScipyMethod",
    "WithFallback",
    "optimizer_for",
    "optimizer_name",
]
