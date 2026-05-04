from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from mypy_extensions import KwArg

from pysatl_core.distributions.computations.computation import (
    AnalyticalComputation,
    FittedComputationMethod,
    FitterMethod,
)
from pysatl_core.distributions.support import (
    ContinuousSupport,
    ExplicitTableDiscreteSupport,
)
from pysatl_core.types import (
    DEFAULT_ANALYTICAL_COMPUTATION_LABEL,
    CharacteristicName,
    Kind,
    NumericArray,
)
from tests.utils.mocks import (
    StandaloneEuclideanUnivariateDistribution,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_ANALYTICAL_LABEL = "default"


class DistributionTestBase:
    def make_uniform_ppf_distribution(
        self,
    ) -> StandaloneEuclideanUnivariateDistribution:
        def ppf_func(q: NumericArray, **kwargs: Any) -> NumericArray:
            return np.atleast_1d(np.asarray(q, dtype=float))

        return StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations={
                CharacteristicName.PPF: {
                    DEFAULT_ANALYTICAL_LABEL: AnalyticalComputation[NumericArray, NumericArray](
                        target=CharacteristicName.PPF,
                        func=ppf_func,  # type: ignore[arg-type]
                    )
                }
            },
            support=ContinuousSupport(0, 1),
        )

    def make_logistic_cdf_distribution(
        self,
    ) -> StandaloneEuclideanUnivariateDistribution:
        def logistic_cdf(x: NumericArray, **_: Any) -> NumericArray:
            x_arr = np.atleast_1d(np.asarray(x, dtype=float))
            return cast(NumericArray, 1.0 / (1.0 + np.exp(-x_arr)))

        return StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations={
                CharacteristicName.CDF: {
                    DEFAULT_ANALYTICAL_LABEL: AnalyticalComputation[NumericArray, NumericArray](
                        target=CharacteristicName.CDF,
                        func=logistic_cdf,  # type: ignore[arg-type]
                    )
                }
            },
            support=ContinuousSupport(),
        )

    def make_uniform_pdf_distribution(
        self,
    ) -> StandaloneEuclideanUnivariateDistribution:
        def uniform_pdf(x: NumericArray, **_: Any) -> NumericArray:
            x_arr = np.atleast_1d(np.asarray(x, dtype=float))
            return cast(NumericArray, np.where((x_arr >= 0.0) & (x_arr <= 1.0), 1.0, 0.0))

        return StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations={
                CharacteristicName.PDF: {
                    DEFAULT_ANALYTICAL_LABEL: AnalyticalComputation[NumericArray, NumericArray](
                        target=CharacteristicName.PDF,
                        func=uniform_pdf,  # type: ignore[arg-type]
                    )
                }
            },
            support=ContinuousSupport(0, 1),
        )

    def make_discrete_point_pmf_distribution(
        self, is_with_support: bool = True
    ) -> StandaloneEuclideanUnivariateDistribution:
        masses = {0.0: 0.2, 1.0: 0.5, 2.0: 0.3}

        def pmf(x: NumericArray, **_: Any) -> NumericArray:
            x_arr = np.atleast_1d(np.asarray(x, dtype=float))
            return cast(
                NumericArray,
                np.array([masses.get(float(xi), 0.0) for xi in x_arr]),
            )

        support = ExplicitTableDiscreteSupport([0, 1, 2]) if is_with_support else None

        return StandaloneEuclideanUnivariateDistribution(
            kind=Kind.DISCRETE,
            analytical_computations={
                CharacteristicName.PMF: {
                    DEFAULT_ANALYTICAL_LABEL: AnalyticalComputation[NumericArray, NumericArray](
                        target=CharacteristicName.PMF,
                        func=pmf,  # type: ignore[arg-type]
                    )
                }
            },
            support=support,
        )

    @staticmethod
    def make_fictitious_computation_method(target: str, sources: Sequence[str]) -> FitterMethod:
        def _fitted_const(val: Any) -> FittedComputationMethod[Any, Any]:
            def _impl(*_args: Any, **_kwargs: Any) -> Any:
                return val

            return cast(FittedComputationMethod[Any, Any], _impl)

        return FitterMethod(
            target=target, sources=sources, fitter=lambda *_a, **_k: _fitted_const(None)
        )


class TestDistributionInitialization:
    def test_distribution_accepts_unlabeled_analytical_mapping(self) -> None:
        ppf_func = cast(Callable[[float, KwArg(Any)], float], lambda q, **_kwargs: q)
        ppf_method = AnalyticalComputation[float, float](
            target=CharacteristicName.PPF, func=ppf_func
        )

        distr = StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations={CharacteristicName.PPF: ppf_method},
            support=ContinuousSupport(0, 1),
        )

        methods = distr.analytical_computations[CharacteristicName.PPF]
        assert set(methods.keys()) == {DEFAULT_ANALYTICAL_COMPUTATION_LABEL}
        assert methods[DEFAULT_ANALYTICAL_COMPUTATION_LABEL](0.42) == pytest.approx(0.42)

    def test_distribution_rejects_empty_labeled_analytical_computations(self) -> None:
        with pytest.raises(
            ValueError,
            match="Characteristic 'cdf' must provide at least one analytical computation.",
        ):
            StandaloneEuclideanUnivariateDistribution(
                kind=Kind.CONTINUOUS,
                analytical_computations={CharacteristicName.CDF: {}},
            )
