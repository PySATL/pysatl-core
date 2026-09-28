"""
Maximum likelihood estimation.

One estimation method, kept in one directory: the policy of a fit
(``estimator``), the objective it minimises (``likelihood``), and what it
returns (``result``).  A second method is a sibling directory, not an edit
here — nothing outside this package imports from it.
"""

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_core.estimation.methods.mle.estimator import MLE, MLE_NAME
from pysatl_core.estimation.methods.mle.likelihood import (
    OUT_OF_SUPPORT_PENALTY,
    Evaluation,
    LogLikelihood,
    LpdfProvider,
    lpdf_provider,
)
from pysatl_core.estimation.methods.mle.result import MLEResult

__all__ = [
    "MLE",
    "MLE_NAME",
    "OUT_OF_SUPPORT_PENALTY",
    "Evaluation",
    "LogLikelihood",
    "LpdfProvider",
    "MLEResult",
    "lpdf_provider",
]
