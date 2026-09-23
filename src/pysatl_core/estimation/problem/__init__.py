"""
What every estimation method does before it does anything of its own.

``validation`` refuses data that cannot carry a fit, ``support`` decides
whether the family's support moves with its parameters, ``views`` reports what
a view pinned and in which coordinates an estimate is expressed, and
``problem`` holds the object the other three hang off.
"""

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_core.estimation.problem.problem import FitProblem, probe_params
from pysatl_core.estimation.problem.support import (
    IntervalSignature,
    OpaqueSignature,
    PointsSignature,
    SupportSignature,
    support_depends_on_params,
    support_signature,
)
from pysatl_core.estimation.problem.validation import check_fixed_support, validate_sample
from pysatl_core.estimation.problem.views import (
    FixedParameters,
    convert_parametrization,
    fixed_parameters,
)

__all__ = [
    "FitProblem",
    "FixedParameters",
    "IntervalSignature",
    "OpaqueSignature",
    "PointsSignature",
    "SupportSignature",
    "check_fixed_support",
    "convert_parametrization",
    "fixed_parameters",
    "probe_params",
    "support_depends_on_params",
    "support_signature",
    "validate_sample",
]
