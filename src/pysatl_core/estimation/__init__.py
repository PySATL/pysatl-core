"""
Parameter estimation for parametric families.

Currently this package implements maximum likelihood estimation.  The entry
point users normally reach for is
:meth:`pysatl_core.families.parametric_family.ParametricFamily.fit`, which
delegates to :func:`fit_family` here.

The modules form a stack, each layer knowing nothing of the ones above it:

``errors`` → ``likelihood`` (objective and gradient) → ``bounds`` (the box the
optimizer is given) → ``moments`` (where the search starts) → ``mle`` (the
policy of a fit) → ``result`` (what comes back), with ``optimizers`` alongside
as the single point of contact with ``scipy.optimize``.
"""

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_core.estimation.bounds import clip_to_bounds, resolve_bounds
from pysatl_core.estimation.errors import (
    FitDataError,
    InsufficientDataError,
    MLEError,
)
from pysatl_core.estimation.likelihood import (
    OUT_OF_SUPPORT_PENALTY,
    GradientFunc,
    LpdfProvider,
    ObjectiveFunc,
    from_vector,
    log_likelihood,
    make_objective,
    to_vector,
)
from pysatl_core.estimation.mle import (
    FixedParameters,
    IntervalSignature,
    OpaqueSignature,
    PointsSignature,
    SupportSignature,
    fit_family,
    support_depends_on_params,
    validate_sample,
)
from pysatl_core.estimation.moments import MomentRule, register_moment_start, starting_point
from pysatl_core.estimation.optimizers import (
    DEFAULT_OPTIMIZER,
    FALLBACK_OPTIMIZER,
    MinimizeCallback,
    MinimizeMethod,
    MinimizeOptions,
    MinimizeSolver,
    OptimizerOutcome,
    optimizer_name,
    run_optimizer,
)
from pysatl_core.estimation.result import FitMethod, MLEResult

__all__ = [
    "MLEResult",
    "FitMethod",
    "MLEError",
    "FitDataError",
    "InsufficientDataError",
    "fit_family",
    "make_objective",
    "log_likelihood",
    "starting_point",
    "register_moment_start",
    "resolve_bounds",
    "clip_to_bounds",
    "support_depends_on_params",
    "validate_sample",
    "to_vector",
    "from_vector",
    "OUT_OF_SUPPORT_PENALTY",
    "DEFAULT_OPTIMIZER",
    "FALLBACK_OPTIMIZER",
    "optimizer_name",
    "run_optimizer",
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
    "LpdfProvider",
    "SupportSignature",
    "IntervalSignature",
    "PointsSignature",
    "OpaqueSignature",
]
