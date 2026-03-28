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
from pysatl_core.transformations.operations import discrete_mixture, finite_mixture
from pysatl_core.transformations.operations.mixture import FiniteMixtureDistribution
from pysatl_core.types import (
    DEFAULT_ANALYTICAL_COMPUTATION_LABEL,
    CharacteristicName,
    ComplexArray,
    ComputationFunc,
    Kind,
    NumericArray,
)
from tests.unit.distributions.test_basic import DistributionTestBase
from tests.utils.mocks import StandaloneEuclideanUnivariateDistribution


class TestFiniteMixtureDistribution(DistributionTestBase):
    @staticmethod
    def _make_uniform_full_distribution(
        left: float,
        right: float,
    ) -> StandaloneEuclideanUnivariateDistribution:
        width = right - left

        def _pdf(data: NumericArray, **_options: Any) -> NumericArray:
            array = np.asarray(data, dtype=float)
            values = np.where((left <= array) & (array <= right), 1.0 / width, 0.0)
            return cast(NumericArray, values)

        def _cdf(data: NumericArray, **_options: Any) -> NumericArray:
            array = np.asarray(data, dtype=float)
            values = np.clip((array - left) / width, 0.0, 1.0)
            return cast(NumericArray, values)

        def _ppf(data: NumericArray, **_options: Any) -> NumericArray:
            array = np.asarray(data, dtype=float)
            if np.any((array < 0.0) | (array > 1.0)):
                raise ValueError("PPF input must be in [0, 1].")
            values = left + width * array
            return cast(NumericArray, values)

        def _cf(data: NumericArray, **_options: Any) -> ComplexArray:
            array = np.asarray(data, dtype=float)
            numerator = np.exp(1j * array * right) - np.exp(1j * array * left)
            denominator = 1j * array * width
            values = np.where(np.isclose(array, 0.0), 1.0 + 0.0j, numerator / denominator)
            return cast(ComplexArray, np.asarray(values, dtype=complex))

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
                CharacteristicName.PDF: AnalyticalComputation[NumericArray, NumericArray](
                    target=CharacteristicName.PDF,
                    func=cast(ComputationFunc[NumericArray, NumericArray], _pdf),
                ),
                CharacteristicName.CDF: AnalyticalComputation[NumericArray, NumericArray](
                    target=CharacteristicName.CDF,
                    func=cast(ComputationFunc[NumericArray, NumericArray], _cdf),
                ),
                CharacteristicName.PPF: AnalyticalComputation[NumericArray, NumericArray](
                    target=CharacteristicName.PPF,
                    func=cast(ComputationFunc[NumericArray, NumericArray], _ppf),
                ),
                CharacteristicName.CF: AnalyticalComputation[NumericArray, ComplexArray](
                    target=CharacteristicName.CF,
                    func=cast(ComputationFunc[NumericArray, ComplexArray], _cf),
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

        def _pmf(data: NumericArray, **_options: Any) -> NumericArray:
            array = np.asarray(data, dtype=float)
            values = np.zeros(array.shape, dtype=float)
            for point, mass in table.items():
                values[np.isclose(array, point, atol=1e-12, rtol=0.0)] = mass
            return cast(NumericArray, values)

        return StandaloneEuclideanUnivariateDistribution(
            kind=Kind.DISCRETE,
            analytical_computations={
                CharacteristicName.PMF: AnalyticalComputation[NumericArray, NumericArray](
                    target=CharacteristicName.PMF,
                    func=cast(ComputationFunc[NumericArray, NumericArray], _pmf),
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

    def test_discrete_mixture_with_arbitrary_number_of_components(self) -> None:
        first = self._make_discrete_pmf_distribution([0.0, 1.0], [0.7, 0.3])
        second = self._make_discrete_pmf_distribution([1.0, 2.0], [0.2, 0.8])
        third = self._make_discrete_pmf_distribution([2.0], [1.0])

        mixture = discrete_mixture(
            [(0.2, first), (0.5, second), (0.3, third)],
        )
        assert isinstance(mixture, FiniteMixtureDistribution)
        assert set(mixture.analytical_computations) == {
            CharacteristicName.PMF,
            CharacteristicName.CDF,
            CharacteristicName.PPF,
        }

        pmf = mixture.query_method(CharacteristicName.PMF)
        cdf = mixture.query_method(CharacteristicName.CDF)
        ppf = mixture.query_method(CharacteristicName.PPF)

        assert float(pmf(0.0)) == pytest.approx(0.14)
        assert float(pmf(1.0)) == pytest.approx(0.16)
        assert float(pmf(2.0)) == pytest.approx(0.70)
        assert float(cdf(1.0)) == pytest.approx(0.30)
        assert float(cdf(2.0)) == pytest.approx(1.0)
        assert float(ppf(0.0)) == pytest.approx(0.0)
        assert float(ppf(0.30)) == pytest.approx(1.0)
        assert float(ppf(0.31)) == pytest.approx(2.0)

        support = mixture.support
        assert isinstance(support, ExplicitTableDiscreteSupport)
        assert support.points == pytest.approx([0.0, 1.0, 2.0])

    def test_continuous_mixture_weighted_characteristics(self) -> None:
        left = self._make_uniform_full_distribution(0.0, 1.0)
        right = self._make_uniform_full_distribution(1.0, 2.0)

        mixture = finite_mixture([(0.25, left), (0.75, right)])
        assert isinstance(mixture, FiniteMixtureDistribution)

        mean = mixture.query_method(CharacteristicName.MEAN)
        pdf = mixture.query_method(CharacteristicName.PDF)
        cdf = mixture.query_method(CharacteristicName.CDF)

        assert mean() == pytest.approx(1.25)
        assert float(pdf(0.5)) == pytest.approx(0.25)
        assert float(pdf(1.5)) == pytest.approx(0.75)
        assert float(cdf(0.5)) == pytest.approx(0.125)
        assert float(cdf(1.5)) == pytest.approx(0.625)
        assert CharacteristicName.PPF not in mixture.analytical_computations

    def test_mixture_mean_requires_only_component_means(self) -> None:
        first = self._make_statistics_distribution(mean=-1.0)
        second = self._make_statistics_distribution(mean=3.0)
        mixture = finite_mixture([(0.25, first), (0.75, second)])

        assert set(mixture.analytical_computations) == {CharacteristicName.MEAN}
        assert mixture.query_method(CharacteristicName.MEAN)() == pytest.approx(2.0)

    def test_mixture_var_requires_only_mean_and_variance(self) -> None:
        first = self._make_statistics_distribution(mean=0.0, variance=1.0)
        second = self._make_statistics_distribution(mean=2.0, variance=2.0)
        mixture = finite_mixture([(0.4, first), (0.6, second)])

        expected_mean = 0.4 * 0.0 + 0.6 * 2.0
        expected_second_raw = 0.4 * (1.0 + 0.0**2) + 0.6 * (2.0 + 2.0**2)
        expected_var = expected_second_raw - expected_mean**2

        assert set(mixture.analytical_computations) == {
            CharacteristicName.MEAN,
            CharacteristicName.VAR,
        }
        assert mixture.query_method(CharacteristicName.MEAN)() == pytest.approx(expected_mean)
        assert mixture.query_method(CharacteristicName.VAR)() == pytest.approx(expected_var)

    def test_non_analytical_component_marks_loop_as_non_analytical(self) -> None:
        left_base = self._make_uniform_full_distribution(0.0, 1.0)
        right = self._make_uniform_full_distribution(1.0, 2.0)
        left = ApproximatedDistribution(
            distribution_type=left_base.distribution_type,
            analytical_computations=left_base.analytical_computations,
            support=left_base.support,
        )

        mixture = finite_mixture([(0.5, left), (0.5, right)])
        assert not mixture.loop_is_analytical(
            CharacteristicName.MEAN,
            DEFAULT_ANALYTICAL_COMPUTATION_LABEL,
        )

    def test_component_and_weight_accessors_are_safe(self) -> None:
        first = self._make_discrete_pmf_distribution([0.0], [1.0])
        second = self._make_discrete_pmf_distribution([1.0], [1.0])
        mixture = finite_mixture([(0.3, first), (0.7, second)])

        weights = mixture.weights
        assert weights.tolist() == pytest.approx([0.3, 0.7])
        weights[1] = 0.0
        assert mixture.weights.tolist() == pytest.approx([0.3, 0.7])

        components = mixture.components
        assert len(components) == 2
        assert components[0] is not components[1]
        assert components[0].distribution_type == first.distribution_type
        assert components[1].distribution_type == second.distribution_type
        assert components[1].analytical_computations == second.analytical_computations

    def test_validation_rejects_invalid_inputs(self) -> None:
        component = self._make_discrete_pmf_distribution([0.0], [1.0])
        continuous = self._make_uniform_full_distribution(0.0, 1.0)

        with pytest.raises(ValueError, match="at least one component"):
            finite_mixture([])
        with pytest.raises(ValueError, match="equal to 1.0"):
            finite_mixture([(0.6, component)])
        with pytest.raises(ValueError, match="non-negative"):
            finite_mixture([(-1.0, component)])
        with pytest.raises(TypeError, match="same distribution kind"):
            finite_mixture([(0.5, component), (0.5, continuous)])

    def test_sample_selects_components_by_weights(self, monkeypatch: pytest.MonkeyPatch) -> None:
        first = self._make_discrete_pmf_distribution([0.0], [1.0])
        second = self._make_discrete_pmf_distribution([1.0], [1.0])
        mixture = finite_mixture([(0.3, first), (0.7, second)])
        selected_indices = np.asarray([1, 0, 1, 1, 0], dtype=int)
        captured: list[tuple[int, object, dict[str, object]]] = []

        class _FakeRng:
            def choice(self, a: int, *, size: int, p: NumericArray) -> NumericArray:
                assert a == 2
                assert size == selected_indices.size
                np.testing.assert_allclose(
                    np.asarray(p, dtype=float),
                    np.asarray([0.3, 0.7], dtype=float),
                )
                return selected_indices

        def _fake_sample(n: int, distr: object, **options: object) -> NumericArray:
            sample_size = int(n)
            captured.append((sample_size, distr, dict(options)))
            if distr is mixture.components[0]:
                return cast(NumericArray, np.asarray([100.0, 101.0], dtype=float))
            if distr is mixture.components[1]:
                return cast(NumericArray, np.asarray([200.0, 201.0, 202.0], dtype=float))
            raise AssertionError("Unexpected component passed to mixture sampler.")

        monkeypatch.setattr(
            "pysatl_core.transformations.operations.mixture.np.random.default_rng",
            lambda: _FakeRng(),
        )
        monkeypatch.setattr(mixture.sampling_strategy, "sample", _fake_sample)
        samples = mixture.sample(5, token="mixture")

        assert captured == [
            (2, mixture.components[0], {"token": "mixture"}),
            (3, mixture.components[1], {"token": "mixture"}),
        ]
        assert samples == pytest.approx([200.0, 100.0, 201.0, 202.0, 101.0])
