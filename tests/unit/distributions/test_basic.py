from __future__ import annotations

import math
from collections.abc import Callable

from pysatl_core.distributions.computation import AnalyticalComputation
from pysatl_core.distributions.distribution import StandaloneEuclideanUnivariateDistribution
from pysatl_core.types import Kind


class DistributionTestBase:
    """Common helpers and constants for distribution tests."""

    PDF = "pdf"
    CDF = "cdf"
    PPF = "ppf"

    # ---- factories ---------------------------------------------------------

    def make_uniform_ppf_distribution(self) -> StandaloneEuclideanUnivariateDistribution:
        """Return a distribution with only PPF = identity on [0,1]."""
        return StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations=[
                AnalyticalComputation[float, float](target=self.PPF, func=lambda q: q),
            ],
        )

    def make_uniform_pdf_distribution(self) -> StandaloneEuclideanUnivariateDistribution:
        """Return a distribution with only PDF for U(0,1)."""

        def pdf_func(x: float) -> float:
            return 1.0 if (0.0 <= x <= 1.0) else 0.0

        return StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations=[
                AnalyticalComputation[float, float](target=self.PDF, func=pdf_func),
            ],
        )

    def make_normal_pdf_function(self, mu: float, sigma: float) -> Callable[[float], float]:
        """Return a Gaussian PDF function with mean mu and std sigma."""
        inv = 1.0 / (sigma * math.sqrt(2.0 * math.pi))

        def pdf(x: float) -> float:
            z = (x - mu) / sigma
            return inv * math.exp(-0.5 * z * z)

        return pdf

    def make_plateau_cdf_distribution(self) -> StandaloneEuclideanUnivariateDistribution:
        """Return a distribution with a plateau CDF: 0 below 0, 0.5 on [0,1), 1 above 1."""

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
