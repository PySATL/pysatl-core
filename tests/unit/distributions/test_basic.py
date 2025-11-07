from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

from mypy_extensions import KwArg

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

    def make_uniform_ppf_distribution(self) -> StandaloneEuclideanUnivariateDistribution:
        """Return a distribution with only PPF = identity on [0,1]."""
        ppf_func = cast(Callable[[float, KwArg(Any)], float], lambda q, **kwargs: q)

        return StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations=[
                AnalyticalComputation[float, float](target=self.PPF, func=ppf_func),
            ],
        )

    def make_logistic_cdf_distribution(
        self,
    ) -> StandaloneEuclideanUnivariateDistribution:
        def logistic_cdf(x: float, **kwargs: Any) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        cdf_func = cast(Callable[[float, KwArg(Any)], float], logistic_cdf)

        return StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations=[
                AnalyticalComputation[float, float](target=self.CDF, func=cdf_func),
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

    def make_plateau_cdf_distribution(self) -> StandaloneEuclideanUnivariateDistribution:
        """Return a distribution with a plateau CDF: 0 below 0, 0.5 on [0,1), 1 above 1."""

        def _plateau_cdf(x: float, **kwargs: Any) -> float:
            if x < 0.0:
                return 0.0
            if x < 1.0:
                return 0.5
            return 1.0

        plateau_cdf = cast(Callable[[float, KwArg(Any)], float], _plateau_cdf)

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
