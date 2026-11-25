from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from mypy_extensions import KwArg

from pysatl_core.arrays import ForeignArrayError
from pysatl_core.distributions.computations.computation import (
    AnalyticalComputation,
    EvaluatorMethod,
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
    ComputationFunc,
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


class TestPublicArrayInput(DistributionTestBase):
    def test_query_method_normalizes_bound_evaluator_keyword_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        distribution = self.make_logistic_cdf_distribution()

        def evaluate(_distribution: object, data: Any, **options: Any) -> Any:
            return data

        fitted = EvaluatorMethod(
            target=CharacteristicName.CDF,
            sources=[CharacteristicName.PDF],
            evaluator=evaluate,
        ).prepare(distribution)
        monkeypatch.setattr(
            distribution.computation_strategy, "query_method", lambda *_args, **_kwargs: fitted
        )

        method = distribution.query_method(CharacteristicName.CDF)
        value = [0.0, 1.0]
        result = method(data=value)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, value)
        assert isinstance(method(value), np.ndarray)

    def test_query_method_preserves_fitted_method_sources(self) -> None:
        distribution = self.make_logistic_cdf_distribution()
        raw = distribution.computation_strategy.query_method(CharacteristicName.PDF, distribution)
        public = distribution.query_method(CharacteristicName.PDF)

        assert isinstance(raw, FittedComputationMethod)
        assert isinstance(public, FittedComputationMethod)
        assert public.target == raw.target
        assert public.sources == raw.sources

    def test_query_method_preserves_native_input_identity(self) -> None:
        def identity(data: Any, **options: Any) -> Any:
            return data

        distribution = StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations={
                "identity": AnalyticalComputation[Any, Any](
                    target="identity", func=cast(ComputationFunc[Any, Any], identity)
                )
            },
        )
        value = np.array([0.0, 1.0])

        method = distribution.query_method("identity")
        assert method(value) is value
        assert method(data=value) is value

    @pytest.mark.parametrize(
        ("value", "expected_type"), [([0.0, 1.0], np.ndarray), (2.0, np.float64)]
    )
    def test_query_method_normalizes_keyword_input(self, value: Any, expected_type: type) -> None:
        def identity(x: Any, **options: Any) -> Any:
            return x

        distribution = StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations={
                "identity": AnalyticalComputation[Any, Any](
                    target="identity", func=cast(ComputationFunc[Any, Any], identity)
                )
            },
        )
        method = distribution.query_method("identity")
        result = method(x=value)

        assert isinstance(result, expected_type)
        np.testing.assert_array_equal(result, value)

    @pytest.mark.parametrize("value", [(0.0, 1.0), object()])
    def test_query_method_rejects_foreign_keyword_input(self, value: object) -> None:
        method = self.make_uniform_ppf_distribution().query_method(CharacteristicName.PPF)

        with pytest.raises(ForeignArrayError):
            method(q=value)
        with pytest.raises(ForeignArrayError, match="Tuple inputs"):
            method((0.0, 1.0))

    def test_query_method_normalizes_callable_keyword_input(self) -> None:
        class Identity:
            def __call__(self, data: Any, **options: Any) -> Any:
                return data

        distribution = StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations={
                "identity": AnalyticalComputation[Any, Any](
                    target="identity", func=cast(ComputationFunc[Any, Any], Identity())
                )
            },
        )
        method = distribution.query_method("identity")
        value = np.array([0.0, 1.0])

        assert method(value) is value
        assert method(data=value) is value
        np.testing.assert_array_equal(method(data=[0.0, 1.0]), value)

    def test_query_method_preserves_options_named_like_positional_only_input(self) -> None:
        def identity(x: Any, /, **options: Any) -> Any:
            assert options["x"] == (2.0, 3.0)
            return x

        distribution = StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations={
                "identity": AnalyticalComputation[Any, Any](target="identity", func=identity)
            },
        )
        value = np.array([0.0, 1.0])

        assert distribution.query_method("identity")(value, x=(2.0, 3.0)) is value

    def test_query_method_preserves_nullary_keyword_options(self) -> None:
        def kurt(*, excess: bool = False) -> float:
            return 0.0 if excess else 3.0

        distribution = StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations={
                CharacteristicName.KURT: AnalyticalComputation[Any, Any](
                    target=CharacteristicName.KURT, func=cast(ComputationFunc[Any, Any], kurt)
                )
            },
        )

        method = distribution.query_method(CharacteristicName.KURT)
        assert method(excess=True) == 0.0
        assert method() == 3.0

    def test_query_method_normalizes_list_input(self) -> None:
        method = self.make_logistic_cdf_distribution().query_method(CharacteristicName.CDF)

        result = method([0.0, 1.0])

        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_query_method_rejects_tuple_input(self) -> None:
        method = self.make_logistic_cdf_distribution().query_method(CharacteristicName.CDF)

        with pytest.raises(ForeignArrayError, match="Tuple inputs"):
            method((0.0, 1.0))
