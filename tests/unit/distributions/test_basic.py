from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from mypy_extensions import KwArg

from pysatl_core.distributions.computation import (
    AnalyticalComputation,
    ComputationMethod,
    FittedComputationMethod,
)
from pysatl_core.distributions.support import (
    ContinuousSupport,
    ExplicitTableDiscreteSupport,
)
from pysatl_core.types import Kind
from tests.utils.mocks import (
    StandaloneEuclideanUnivariateDistribution,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class DistributionTestBase:
    PDF = "pdf"
    CDF = "cdf"
    PPF = "ppf"
    PMF = "pmf"

    def make_uniform_ppf_distribution(
        self,
    ) -> StandaloneEuclideanUnivariateDistribution:
        ppf_func = cast(Callable[[float, KwArg(Any)], float], lambda q, **kwargs: q)
        return StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations=[
                AnalyticalComputation[float, float](target=self.PPF, func=ppf_func),
            ],
            support=ContinuousSupport(0, 1),
        )

    def make_logistic_cdf_distribution(
        self,
    ) -> StandaloneEuclideanUnivariateDistribution:
        def logistic_cdf(x: float, **_: Any) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        logistic_cdf_func = cast(Callable[[float, KwArg(Any)], float], logistic_cdf)
        return StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations=[
                AnalyticalComputation[float, float](target=self.CDF, func=logistic_cdf_func),
            ],
            support=ContinuousSupport(),
        )

    def make_uniform_pdf_distribution(
        self,
    ) -> StandaloneEuclideanUnivariateDistribution:
        def uniform_pdf(x: float, **_: Any) -> float:
            return 1.0 if 0.0 <= x <= 1.0 else 0.0

        uniform_pdf_func = cast(Callable[[float, KwArg(Any)], float], uniform_pdf)

        return StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations=[
                AnalyticalComputation[float, float](target=self.PDF, func=uniform_pdf_func),
            ],
            support=ContinuousSupport(0, 1),
        )

    def make_discrete_point_pmf_distribution(
        self, is_with_support: bool = True
    ) -> StandaloneEuclideanUnivariateDistribution:
        masses = {0.0: 0.2, 1.0: 0.5, 2.0: 0.3}

        def pmf(x: float) -> float:
            return masses.get(float(x), 0.0)

        pmf_func = cast(Callable[[float, KwArg(Any)], float], pmf)

        support = ExplicitTableDiscreteSupport([0, 1, 2]) if is_with_support else None

        return StandaloneEuclideanUnivariateDistribution(
            kind=Kind.DISCRETE,
            analytical_computations=[
                AnalyticalComputation[float, float](target=self.PMF, func=pmf_func),
            ],
            support=support,
        )

    @staticmethod
    def make_fictitious_computation_method(
        target: str, sources: Sequence[str]
    ) -> ComputationMethod[Any, Any]:
        def _fitted_const(val: Any) -> FittedComputationMethod[Any, Any]:
            def _impl(*_args: Any, **_kwargs: Any) -> Any:
                return val

            return cast(FittedComputationMethod[Any, Any], _impl)

        return ComputationMethod(
            target=target, sources=sources, fitter=lambda *_a, **_k: _fitted_const(None)
        )
