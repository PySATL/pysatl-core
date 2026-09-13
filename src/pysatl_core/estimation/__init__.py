"""
Parameter estimation for parametric families.

Currently this package implements maximum likelihood estimation.  The entry
point users normally reach for is
:meth:`pysatl_core.families.parametric_family.ParametricFamily.fit`, which
delegates to :func:`fit_family` here.
"""

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_core.estimation.errors import (
    FitDataError,
    InsufficientDataError,
    MLEError,
)
from pysatl_core.estimation.likelihood import (
    OUT_OF_SUPPORT_PENALTY,
    from_vector,
    log_likelihood,
    make_objective,
    to_vector,
)
from pysatl_core.estimation.mle import (
    DEFAULT_OPTIMIZER,
    FALLBACK_OPTIMIZER,
    clip_to_bounds,
    fit_family,
    resolve_bounds,
    support_depends_on_params,
    validate_sample,
)
from pysatl_core.estimation.moments import register_moment_start, starting_point
from pysatl_core.estimation.result import MLEResult

__all__ = [
    "MLEResult",
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
]
