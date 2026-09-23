"""
Parameter estimation for parametric families.

An estimation method is an object — an
:class:`~pysatl_core.estimation.estimator.Estimator` — carrying its own
configuration in its own typed fields.  Maximum likelihood,
:class:`~pysatl_core.estimation.methods.mle.MLE`, is the one implemented today
and the default of
:meth:`pysatl_core.families.parametric_family.ParametricFamily.fit`.

The package is laid out along one split: what every method is built from, and
the methods themselves.

At the top level sit the things nothing can be written without — what a method
*is* (``estimator``), what it returns (``result``), what it raises (``errors``)
and how a warning finds the caller's line (``_attribution``) — and three
sub-packages, none of which knows what is being estimated:

``problem``
    One family, one sample, and everything derived from the pair: validation,
    the support's behaviour, what a view pinned, and the
    :class:`~pysatl_core.estimation.problem.FitProblem` they hang off.
``parameters``
    The parameter space — flat vectors, declared bounds, where a search starts.
``optimizers``
    The search itself, including the fallback policy, with no notion of a
    criterion.

``methods`` holds the methods, one directory (or module) each.  The boundary is
checkable rather than promised: nothing outside ``methods`` imports from it,
and no method imports from a sibling.

What this facade exports
------------------------
The shared layer in full, and from each method only the two names a caller
needs: the estimator class and its result.  A method's internals — the
criterion it minimises, its own helpers — stay behind its own package, so that
this list does not grow by a dozen names with every method added::

    from pysatl_core.estimation import MLE, MLEResult            # the method
    from pysatl_core.estimation.methods.mle import LogLikelihood  # its insides
"""

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_core.estimation.errors import (
    EstimationError,
    FitDataError,
    InsufficientDataError,
)
from pysatl_core.estimation.estimator import AnyFitResult, Estimator
from pysatl_core.estimation.methods import MLE, MLEResult
from pysatl_core.estimation.optimizers import (
    DEFAULT_OPTIMIZER,
    FALLBACK_OPTIMIZER,
    CustomSolver,
    GradientFunc,
    MinimizeCallback,
    MinimizeMethod,
    MinimizeOptions,
    MinimizeSolver,
    ObjectiveFunc,
    Optimizer,
    OptimizerOutcome,
    ScipyMethod,
    WithFallback,
    optimizer_for,
    optimizer_name,
)
from pysatl_core.estimation.parameters import (
    MomentRule,
    ParameterBox,
    field_names,
    from_vector,
    starting_point,
    to_vector,
)
from pysatl_core.estimation.problem import (
    FitProblem,
    FixedParameters,
    IntervalSignature,
    OpaqueSignature,
    PointsSignature,
    SupportSignature,
    check_fixed_support,
    convert_parametrization,
    fixed_parameters,
    probe_params,
    support_depends_on_params,
    support_signature,
    validate_sample,
)
from pysatl_core.estimation.result import FitResult, FitRoute

__all__ = [
    # What a method is
    "Estimator",
    "AnyFitResult",
    "FitResult",
    "FitRoute",
    # The methods: one class and one result each
    "MLE",
    "MLEResult",
    # Errors
    "EstimationError",
    "FitDataError",
    "InsufficientDataError",
    # The problem: one family, one sample
    "FitProblem",
    "validate_sample",
    "probe_params",
    "support_depends_on_params",
    "support_signature",
    "check_fixed_support",
    "fixed_parameters",
    "convert_parametrization",
    # The parameter space
    "ParameterBox",
    "starting_point",
    "to_vector",
    "from_vector",
    "field_names",
    # Optimizers
    "Optimizer",
    "ScipyMethod",
    "CustomSolver",
    "WithFallback",
    "optimizer_for",
    "optimizer_name",
    "DEFAULT_OPTIMIZER",
    "FALLBACK_OPTIMIZER",
    # Declared shapes: what used to be probed, cast or left as ``Any``
    "FixedParameters",
    "OptimizerOutcome",
    "MinimizeMethod",
    "MinimizeOptions",
    "MinimizeSolver",
    "MinimizeCallback",
    "MomentRule",
    "ObjectiveFunc",
    "GradientFunc",
    "SupportSignature",
    "IntervalSignature",
    "PointsSignature",
    "OpaqueSignature",
]
