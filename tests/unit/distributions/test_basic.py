from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import math

from pysatl_core.distributions.computation import AnalyticalComputation
from pysatl_core.types import Kind
from tests.utils.mocks import (
    DiscreteSupport,
    StandaloneEuclideanUnivariateDistribution,
)


class DistributionTestBase:
    """Common helpers and constants for distribution tests."""

    PDF = "pdf"
    CDF = "cdf"
    PPF = "ppf"
    PMF = "pmf"

    # ---- factories: continuous -------------------------------------------------

    def make_uniform_ppf_distribution(
        self,
    ) -> StandaloneEuclideanUnivariateDistribution:
        return StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations=[
                AnalyticalComputation[float, float](target=self.PPF, func=lambda q: q),
            ],
        )

    def make_logistic_cdf_distribution(
        self,
    ) -> StandaloneEuclideanUnivariateDistribution:
        def logistic_cdf(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        return StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations=[
                AnalyticalComputation[float, float](target=self.CDF, func=logistic_cdf),
            ],
        )

    def make_uniform_pdf_distribution(
        self,
    ) -> StandaloneEuclideanUnivariateDistribution:
        def uniform_pdf(x: float) -> float:
            return 1.0 if 0.0 <= x <= 1.0 else 0.0

        return StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations=[
                AnalyticalComputation[float, float](target=self.PDF, func=uniform_pdf),
            ],
        )

    def make_plateau_cdf_distribution(
        self,
    ) -> StandaloneEuclideanUnivariateDistribution:
        def plateau_cdf(x: float) -> float:
            if x < 0.0:
                return 0.0
            if x < 1.0:
                return 0.5
            return 1.0

        return StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations=[
                AnalyticalComputation[float, float](target=self.CDF, func=plateau_cdf),
            ],
        )

    # ---- factories: discrete ---------------------------------------------------

    def make_discrete_point_pmf_distribution(
        self,
    ) -> StandaloneEuclideanUnivariateDistribution:
        masses = {0.0: 0.2, 1.0: 0.5, 2.0: 0.3}

        def pmf(x: float) -> float:
            return masses.get(float(x), 0.0)

        return StandaloneEuclideanUnivariateDistribution(
            kind=Kind.DISCRETE,
            analytical_computations=[
                AnalyticalComputation[float, float](target=self.PMF, func=pmf),
            ],
        )

    def make_discrete_support(self) -> DiscreteSupport:
        return DiscreteSupport([0.0, 1.0, 2.0])
