from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov, Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_core.distributions.fitters.base import (
    FitterDescriptor,
    FitterOption,
)
from pysatl_core.distributions.fitters.continuous import (
    FITTER_CDF_TO_PDF_1C,
    FITTER_CDF_TO_PPF_1C,
    FITTER_PDF_TO_CDF_1C,
    FITTER_PPF_TO_CDF_1C,
    fit_cdf_to_pdf_1C,
    fit_cdf_to_ppf_1C,
    fit_pdf_to_cdf_1C,
    fit_ppf_to_cdf_1C,
)
from pysatl_core.distributions.fitters.discrete import (
    FITTER_CDF_TO_PMF_1D,
    FITTER_CDF_TO_PPF_1D,
    FITTER_PMF_TO_CDF_1D,
    FITTER_PPF_TO_CDF_1D,
    fit_cdf_to_pmf_1D,
    fit_cdf_to_ppf_1D,
    fit_pmf_to_cdf_1D,
    fit_ppf_to_cdf_1D,
)
from pysatl_core.distributions.fitters.registry import FitterRegistry

ALL_FITTER_DESCRIPTORS: list[FitterDescriptor] = [
    FITTER_PDF_TO_CDF_1C,
    FITTER_CDF_TO_PDF_1C,
    FITTER_CDF_TO_PPF_1C,
    FITTER_PPF_TO_CDF_1C,
    FITTER_PMF_TO_CDF_1D,
    FITTER_CDF_TO_PMF_1D,
    FITTER_CDF_TO_PPF_1D,
    FITTER_PPF_TO_CDF_1D,
]

__all__ = [
    # Abstractions
    "FitterOption",
    "FitterDescriptor",
    "FitterRegistry",
    # Continuous fitters
    "fit_pdf_to_cdf_1C",
    "fit_cdf_to_pdf_1C",
    "fit_cdf_to_ppf_1C",
    "fit_ppf_to_cdf_1C",
    # Discrete fitters
    "fit_pmf_to_cdf_1D",
    "fit_cdf_to_pmf_1D",
    "fit_cdf_to_ppf_1D",
    "fit_ppf_to_cdf_1D",
    # Descriptors
    "FITTER_PDF_TO_CDF_1C",
    "FITTER_CDF_TO_PDF_1C",
    "FITTER_CDF_TO_PPF_1C",
    "FITTER_PPF_TO_CDF_1C",
    "FITTER_PMF_TO_CDF_1D",
    "FITTER_CDF_TO_PMF_1D",
    "FITTER_CDF_TO_PPF_1D",
    "FITTER_PPF_TO_CDF_1D",
    # Aggregate
    "ALL_FITTER_DESCRIPTORS",
]
