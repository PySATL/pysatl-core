"""
Distributions Subpackage
========================

Interfaces and default implementations for probability distributions used by
PySATL Core. This subpackage provides:

- characteristic wrappers (:mod:`.characteristics`);
- analytical and fitted computation primitives and canonical 1C conversions
  (:mod:`.computation`);
- a minimal standalone univariate distribution and the public distribution
  protocol (:mod:`.distribution`);
- numerical fitters between PDF/CDF/PPF (:mod:`.fitters`);
- a directed graph registry for characteristics, with invariants
  (:mod:`.registry`);
- sampling protocol and an array-backed sample type (:mod:`.sampling`);
- pluggable computation/sampling strategies with defaults (:mod:`.strategies`).

The public symbols are re-exported here for convenience, so typical users can
import from :mod:`pysatl_core.distributions` directly.

Notes
-----
- The univariate API is intentionally **scalar** for characteristics
  (``float -> float``). Sampling returns 2D arrays of shape ``(n, d)`` with
  ``d == 1`` for the univariate case.
- Conversions between characteristics are discovered over a directed graph
  and fitted on demand via strategy logic.

Examples
--------
Create a trivial continuous univariate distribution with analytical ``ppf(q)=q``,
draw a sample, and compute its log-likelihood under a uniform PDF:

>>> from pysatl_core.types import Kind
>>> from pysatl_core.distributions import (
...     AnalyticalComputation, StandaloneEuclideanUnivariateDistribution
... )
>>> PPF = "ppf"; PDF = "pdf"
>>> dist = StandaloneEuclideanUnivariateDistribution(
...     kind=Kind.CONTINUOUS,
...     analytical_computations=[AnalyticalComputation[float, float](PPF, lambda q: q)],
... )
>>> sample = dist.sample(5)           # shape (5, 1)
>>> sample.shape
(5, 1)
>>> # Provide an analytical uniform pdf to compute log-likelihood:
>>> dist_uniform_pdf = StandaloneEuclideanUnivariateDistribution(
...     kind=Kind.CONTINUOUS,
...     analytical_computations= \
...         [AnalyticalComputation[float, float](PDF, lambda x: 1.0 if 0.0 <= x <= 1.0 else 0.0)],
... )
>>> dist_uniform_pdf.log_likelihood(sample)  # log(1)*5 == 0.0
0.0
"""

__author__ = "Leonid Elkin, Mikhail, Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


# Re-exports: computation primitives and canonical conversions
from .computation import (
    AnalyticalComputation,
    ComputationMethod,
    FittedComputationMethod,
    cdf_to_pdf_1C,
    cdf_to_ppf_1C,
    pdf_to_cdf_1C,
    ppf_to_cdf_1C,
)

# Re-exports: distribution protocol and a minimal implementation
from .distribution import Distribution

# Re-exports: characteristic graph registry and helpers
from .registry import (
    DEFAULT_COMPUTATION_KEY,
    DistributionTypeRegister,
    GenericCharacteristicRegister,
    GraphInvariantError,
    distribution_type_register,
)

# Re-exports: sampling protocol and array-backed sample
from .sampling import ArraySample, Sample

# Re-exports: strategies (computation & sampling)
from .strategies import (
    ComputationStrategy,
    DefaultComputationStrategy,
    DefaultSamplingUnivariateStrategy,
    SamplingStrategy,
)

__all__ = [
    # characteristics
    "GenericCharacteristic",
    # computation
    "AnalyticalComputation",
    "ComputationMethod",
    "FittedComputationMethod",
    "pdf_to_cdf_1C",
    "cdf_to_pdf_1C",
    "cdf_to_ppf_1C",
    "ppf_to_cdf_1C",
    # distribution
    "Distribution",
    "StandaloneEuclideanUnivariateDistribution",
    # sampling
    "Sample",
    "ArraySample",
    # strategies
    "ComputationStrategy",
    "DefaultComputationStrategy",
    "SamplingStrategy",
    "DefaultSamplingUnivariateStrategy",
    # registry
    "DEFAULT_COMPUTATION_KEY",
    "GraphInvariantError",
    "GenericCharacteristicRegister",
    "DistributionTypeRegister",
    "distribution_type_register",
]
