from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any, cast

import numpy as np
import pytest

from pysatl_core.distributions.computation import AnalyticalComputation
from pysatl_core.distributions.support import ContinuousSupport, ExplicitTableDiscreteSupport
from pysatl_core.transformations.distribution import ApproximatedDistribution
from pysatl_core.transformations.operations import binary
from pysatl_core.transformations.operations.binary import (
    DivisionBinaryDistribution,
    LinearBinaryDistribution,
    MultiplicationBinaryDistribution,
)
from pysatl_core.types import (
    DEFAULT_ANALYTICAL_COMPUTATION_LABEL,
    BinaryOperationName,
    CharacteristicName,
    ComputationFunc,
    Kind,
)
from tests.unit.distributions.test_basic import DistributionTestBase
from tests.utils.mocks import StandaloneEuclideanUnivariateDistribution


class TestBinaryDistribution(DistributionTestBase):
    @staticmethod
    def _make_uniform_full_distribution(
        left: float,
        right: float,
    ) -> StandaloneEuclideanUnivariateDistribution:
        width = right - left

        def _pdf(data: float, **_options: Any) -> float:
            return 1.0 / width if left <= data <= right else 0.0

        def _cdf(data: float, **_options: Any) -> float:
            if data <= left:
                return 0.0
            if data >= right:
                return 1.0
            return (data - left) / width

        def _ppf(data: float, **_options: Any) -> float:
            if not 0.0 <= data <= 1.0:
                raise ValueError("PPF input must be in [0, 1].")
            return left + width * data

        def _cf(data: float, **_options: Any) -> complex:
            if data == 0.0:
                return 1.0 + 0.0j
            numerator = np.exp(1j * data * right) - np.exp(1j * data * left)
            return cast(complex, numerator / (1j * data * width))

        def _mean(**_options: Any) -> float:
            return 0.5 * (left + right)

        def _var(**_options: Any) -> float:
            return width**2 / 12.0

        def _skew(**_options: Any) -> float:
            return 0.0

        def _kurt(*, excess: bool = False, **_options: Any) -> float:
            return -1.2 if excess else 1.8

        return StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations={
                CharacteristicName.PDF: AnalyticalComputation[float, float](
                    target=CharacteristicName.PDF,
                    func=cast(ComputationFunc[float, float], _pdf),
                ),
                CharacteristicName.CDF: AnalyticalComputation[float, float](
                    target=CharacteristicName.CDF,
                    func=cast(ComputationFunc[float, float], _cdf),
                ),
                CharacteristicName.PPF: AnalyticalComputation[float, float](
                    target=CharacteristicName.PPF,
                    func=cast(ComputationFunc[float, float], _ppf),
                ),
                CharacteristicName.CF: AnalyticalComputation[float, complex](
                    target=CharacteristicName.CF,
                    func=cast(ComputationFunc[float, complex], _cf),
                ),
                CharacteristicName.MEAN: AnalyticalComputation[Any, float](
                    target=CharacteristicName.MEAN,
                    func=cast(ComputationFunc[Any, float], _mean),
                ),
                CharacteristicName.VAR: AnalyticalComputation[Any, float](
                    target=CharacteristicName.VAR,
                    func=cast(ComputationFunc[Any, float], _var),
                ),
                CharacteristicName.SKEW: AnalyticalComputation[Any, float](
                    target=CharacteristicName.SKEW,
                    func=cast(ComputationFunc[Any, float], _skew),
                ),
                CharacteristicName.KURT: AnalyticalComputation[Any, float](
                    target=CharacteristicName.KURT,
                    func=cast(ComputationFunc[Any, float], _kurt),
                ),
            },
            support=ContinuousSupport(left, right),
        )

    @staticmethod
    def _make_discrete_pmf_distribution(
        points: list[float],
        masses: list[float],
    ) -> StandaloneEuclideanUnivariateDistribution:
        table = {float(point): float(mass) for point, mass in zip(points, masses, strict=True)}

        def _pmf(data: float, **_options: Any) -> float:
            return table.get(float(data), 0.0)

        return StandaloneEuclideanUnivariateDistribution(
            kind=Kind.DISCRETE,
            analytical_computations={
                CharacteristicName.PMF: AnalyticalComputation[float, float](
                    target=CharacteristicName.PMF,
                    func=cast(ComputationFunc[float, float], _pmf),
                )
            },
            support=ExplicitTableDiscreteSupport(points=points, assume_sorted=False),
        )

    @staticmethod
    def _make_statistics_distribution(
        *,
        mean: float | None = None,
        variance: float | None = None,
        skewness: float | None = None,
        kurtosis: float | None = None,
    ) -> StandaloneEuclideanUnivariateDistribution:
        analytical_computations: dict[str, AnalyticalComputation[Any, Any]] = {}

        if mean is not None:
            analytical_computations[CharacteristicName.MEAN] = AnalyticalComputation[Any, float](
                target=CharacteristicName.MEAN,
                func=cast(ComputationFunc[Any, float], lambda **_options: mean),
            )
        if variance is not None:
            analytical_computations[CharacteristicName.VAR] = AnalyticalComputation[Any, float](
                target=CharacteristicName.VAR,
                func=cast(ComputationFunc[Any, float], lambda **_options: variance),
            )
        if skewness is not None:
            analytical_computations[CharacteristicName.SKEW] = AnalyticalComputation[Any, float](
                target=CharacteristicName.SKEW,
                func=cast(ComputationFunc[Any, float], lambda **_options: skewness),
            )
        if kurtosis is not None:
            analytical_computations[CharacteristicName.KURT] = AnalyticalComputation[Any, float](
                target=CharacteristicName.KURT,
                func=cast(
                    ComputationFunc[Any, float],
                    lambda *, excess=False, **_options: kurtosis - 3.0 if excess else kurtosis,
                ),
            )

        return StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations=analytical_computations,
        )

    def test_add_continuous_exposes_full_characteristic_set(self) -> None:
        left = self._make_uniform_full_distribution(0.0, 1.0)
        right = self._make_uniform_full_distribution(1.0, 2.0)

        transformed = binary(left, right, operation=BinaryOperationName.ADD)
        assert isinstance(transformed, LinearBinaryDistribution)
        assert set(transformed.analytical_computations) == {
            CharacteristicName.CF,
            CharacteristicName.MEAN,
            CharacteristicName.VAR,
            CharacteristicName.SKEW,
            CharacteristicName.KURT,
            CharacteristicName.CDF,
            CharacteristicName.PDF,
            CharacteristicName.PPF,
        }

        mean = transformed.query_method(CharacteristicName.MEAN)
        var = transformed.query_method(CharacteristicName.VAR)
        pdf = transformed.query_method(CharacteristicName.PDF)
        cdf = transformed.query_method(CharacteristicName.CDF)
        ppf = transformed.query_method(CharacteristicName.PPF)
        cf = transformed.query_method(CharacteristicName.CF)

        assert mean() == pytest.approx(2.0)
        assert var() == pytest.approx(1.0 / 6.0, rel=1e-2)
        assert pdf(2.0) == pytest.approx(1.0, rel=2e-2)
        assert cdf(2.0) == pytest.approx(0.5, rel=2e-2)
        assert ppf(0.5) == pytest.approx(2.0, rel=2e-2)
        assert cf(0.0) == pytest.approx(1.0 + 0.0j)

    def test_sub_mul_div_continuous_are_queryable(self) -> None:
        sub_left = self._make_uniform_full_distribution(0.0, 1.0)
        sub_right = self._make_uniform_full_distribution(1.0, 2.0)
        sub_transformed = binary(sub_left, sub_right, operation=BinaryOperationName.SUB)
        assert isinstance(sub_transformed, LinearBinaryDistribution)

        sub_mean = sub_transformed.query_method(CharacteristicName.MEAN)
        sub_var = sub_transformed.query_method(CharacteristicName.VAR)
        assert sub_mean() == pytest.approx(-1.0)
        assert sub_var() == pytest.approx(1.0 / 6.0, rel=1e-2)

        mul_left = self._make_uniform_full_distribution(1.0, 2.0)
        mul_right = self._make_uniform_full_distribution(2.0, 3.0)
        mul_transformed = binary(mul_left, mul_right, operation=BinaryOperationName.MUL)
        assert isinstance(mul_transformed, MultiplicationBinaryDistribution)

        mul_mean = mul_transformed.query_method(CharacteristicName.MEAN)
        mul_var = mul_transformed.query_method(CharacteristicName.VAR)
        mul_pdf = mul_transformed.query_method(CharacteristicName.PDF)
        mul_cdf = mul_transformed.query_method(CharacteristicName.CDF)
        mul_ppf = mul_transformed.query_method(CharacteristicName.PPF)
        assert mul_mean() == pytest.approx(3.75, rel=2e-3)
        assert mul_var() == pytest.approx(0.7152777778, rel=1e-2)
        assert float(mul_pdf(3.0)) >= 0.0
        assert 0.0 <= float(mul_cdf(3.0)) <= 1.0
        assert np.isfinite(float(mul_ppf(0.5)))

        div_left = self._make_uniform_full_distribution(2.0, 3.0)
        div_right = self._make_uniform_full_distribution(1.0, 2.0)
        div_transformed = binary(div_left, div_right, operation=BinaryOperationName.DIV)
        assert isinstance(div_transformed, DivisionBinaryDistribution)

        div_mean = div_transformed.query_method(CharacteristicName.MEAN)
        div_var = div_transformed.query_method(CharacteristicName.VAR)
        div_pdf = div_transformed.query_method(CharacteristicName.PDF)
        div_cdf = div_transformed.query_method(CharacteristicName.CDF)
        div_ppf = div_transformed.query_method(CharacteristicName.PPF)
        assert div_mean() == pytest.approx(2.5 * np.log(2.0), rel=1e-2)
        assert div_var() == pytest.approx(0.163824299, rel=2e-2)
        assert float(div_pdf(1.5)) >= 0.0
        assert 0.0 <= float(div_cdf(1.5)) <= 1.0
        assert np.isfinite(float(div_ppf(0.5)))

    def test_linear_mean_requires_only_parent_means(self) -> None:
        left = self._make_statistics_distribution(mean=1.5)
        right = self._make_statistics_distribution(mean=-0.25)

        transformed = binary(left, right, operation=BinaryOperationName.ADD)
        assert isinstance(transformed, LinearBinaryDistribution)
        assert set(transformed.analytical_computations) == {CharacteristicName.MEAN}
        assert transformed.query_method(CharacteristicName.MEAN)() == pytest.approx(1.25)

    def test_linear_var_requires_only_mean_and_variance(self) -> None:
        left = self._make_statistics_distribution(mean=2.0, variance=1.0)
        right = self._make_statistics_distribution(mean=-1.0, variance=0.25)

        transformed = binary(left, right, operation=BinaryOperationName.ADD)
        assert isinstance(transformed, LinearBinaryDistribution)
        assert set(transformed.analytical_computations) == {
            CharacteristicName.MEAN,
            CharacteristicName.VAR,
        }
        assert transformed.query_method(CharacteristicName.MEAN)() == pytest.approx(1.0)
        assert transformed.query_method(CharacteristicName.VAR)() == pytest.approx(1.25)

    def test_discrete_operations_expose_pmf_cdf_ppf(self) -> None:
        left = self._make_discrete_pmf_distribution([0.0, 1.0, 2.0], [0.2, 0.5, 0.3])
        right = self._make_discrete_pmf_distribution([1.0, 2.0], [0.6, 0.4])

        for operation in BinaryOperationName:
            transformed = binary(left, right, operation=operation)
            assert {
                CharacteristicName.PMF,
                CharacteristicName.CDF,
                CharacteristicName.PPF,
            }.issubset(set(transformed.analytical_computations))

            pmf = transformed.query_method(CharacteristicName.PMF)
            cdf = transformed.query_method(CharacteristicName.CDF)
            ppf = transformed.query_method(CharacteristicName.PPF)

            support = transformed.support
            assert isinstance(support, ExplicitTableDiscreteSupport)
            points = np.asarray(support.points, dtype=float)
            masses = np.asarray([pmf(float(x)) for x in points], dtype=float)
            assert float(np.sum(masses)) == pytest.approx(1.0, rel=1e-9)
            assert float(cdf(float(points[0] - 1.0))) == pytest.approx(0.0)
            assert float(cdf(float(points[-1] + 1.0))) == pytest.approx(1.0)
            assert float(ppf(0.0)) == pytest.approx(float(points[0]))
            assert float(ppf(1.0)) == pytest.approx(float(points[-1]))

    def test_non_analytical_parent_marks_loop_as_non_analytical(self) -> None:
        left_base = self._make_uniform_full_distribution(0.0, 1.0)
        right = self._make_uniform_full_distribution(1.0, 2.0)

        left = ApproximatedDistribution(
            distribution_type=left_base.distribution_type,
            analytical_computations=left_base.analytical_computations,
            support=left_base.support,
        )
        transformed = binary(left, right, operation=BinaryOperationName.ADD)

        assert not transformed.loop_is_analytical(
            CharacteristicName.MEAN,
            DEFAULT_ANALYTICAL_COMPUTATION_LABEL,
        )

    @pytest.mark.parametrize(
        ("operation", "left_samples", "right_samples", "expected"),
        [
            (
                BinaryOperationName.ADD,
                [1.0, 2.0, 3.0],
                [10.0, 20.0, 30.0],
                [11.0, 22.0, 33.0],
            ),
            (
                BinaryOperationName.SUB,
                [5.0, 4.0, 3.0],
                [1.0, 2.0, 3.0],
                [4.0, 2.0, 0.0],
            ),
            (
                BinaryOperationName.MUL,
                [3.0, 4.0, 5.0],
                [2.0, 3.0, 4.0],
                [6.0, 12.0, 20.0],
            ),
            (
                BinaryOperationName.DIV,
                [2.0, 4.0, 6.0],
                [1.0, 2.0, 3.0],
                [2.0, 2.0, 2.0],
            ),
        ],
    )
    def test_sample_uses_both_parents_and_applies_operation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        operation: BinaryOperationName,
        left_samples: list[float],
        right_samples: list[float],
        expected: list[float],
    ) -> None:
        left = self._make_uniform_full_distribution(1.0, 2.0)
        right = self._make_uniform_full_distribution(1.0, 2.0)
        transformed = binary(left, right, operation=operation)
        captured: list[tuple[int, object, dict[str, object]]] = []

        def _fake_sample(n: int, distr: object, **options: object) -> object:
            captured.append((n, distr, dict(options)))
            if distr is transformed.left_distribution:
                return left_samples
            if distr is transformed.right_distribution:
                return right_samples
            raise AssertionError("Unexpected parent distribution passed to binary sampler.")

        monkeypatch.setattr(transformed.sampling_strategy, "sample", _fake_sample)
        samples = transformed.sample(3, token="binary")

        assert captured == [
            (3, transformed.left_distribution, {"token": "binary"}),
            (3, transformed.right_distribution, {"token": "binary"}),
        ]
        assert samples == pytest.approx(expected)
