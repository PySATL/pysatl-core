"""
Unit tests for EmpiricalDistribution: validation, estimators, and estimator changes.

Tests here assert observable behaviour through the public API.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from pysatl_core.distributions.computations.computation import AnalyticalComputation
from pysatl_core.distributions.empirical import (
    EmpiricalCdf,
    EmpiricalDistribution,
    ScipyGaussianKde,
)
from pysatl_core.distributions.strategies import DefaultComputationStrategy
from pysatl_core.distributions.support import ExplicitTableDiscreteSupport, Support
from pysatl_core.types import (
    CharacteristicName,
    ComputationFunc,
    DistributionType,
    UnivariateContinuous,
    UnivariateDiscrete,
)


class _ConstantEstimate:
    def __init__(self, pdf_value: float, cdf_value: float) -> None:
        self._pdf_value = pdf_value
        self._cdf_value = cdf_value
        self.fit_calls = 0

    def pdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return np.full_like(x, self._pdf_value)

    def cdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return np.full_like(x, self._cdf_value)


class _ConstantEstimator:
    def __init__(self, pdf_value: float, cdf_value: float) -> None:
        self._pdf_value = pdf_value
        self._cdf_value = cdf_value
        self.fit_calls = 0

    @property
    def distribution_type(self) -> DistributionType:
        return UnivariateContinuous

    def resolve_support(
        self, sample: NDArray[np.float64], support: Support | None
    ) -> Support | None:
        return ScipyGaussianKde().resolve_support(sample, support)

    def fit(self, sample: NDArray[np.float64]) -> dict[str, AnalyticalComputation[Any, Any]]:
        self.fit_calls += 1
        fitted = _ConstantEstimate(self._pdf_value, self._cdf_value)
        return {
            CharacteristicName.CDF: AnalyticalComputation(
                CharacteristicName.CDF, cast(ComputationFunc[Any, Any], fitted.cdf)
            ),
            CharacteristicName.PDF: AnalyticalComputation(
                CharacteristicName.PDF, cast(ComputationFunc[Any, Any], fitted.pdf)
            ),
        }


@pytest.fixture
def sample() -> NDArray[np.float64]:
    rng = np.random.default_rng(42)
    return rng.normal(0.0, 1.0, 300)


@pytest.fixture
def distr(sample: NDArray[np.float64]) -> EmpiricalDistribution:
    return EmpiricalDistribution(sample)


class TestValidateSample:
    """Common checks precede fitting; KDE adds density-specific checks."""

    def test_accepts_a_normal_sample(self, sample: NDArray[np.float64]) -> None:
        validated = EmpiricalDistribution.validate_sample(sample)
        assert validated.ndim == 1
        assert validated.dtype == np.float64
        assert np.array_equal(validated, sample)

    def test_coerces_a_python_list(self) -> None:
        validated = EmpiricalDistribution.validate_sample(cast(Any, [1.0, 2.0, 3.5]))
        assert isinstance(validated, np.ndarray)
        assert validated.dtype == np.float64

    def test_one_observation_is_valid_for_an_empirical_cdf(self) -> None:
        assert EmpiricalDistribution.validate_sample(np.array([1.0])).size == 1

    # --- rejections -------------------------------------------------------

    def test_rejects_two_dimensional_sample(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match=r"univariate.*1-D.*shape \(2, 300\)"):
            EmpiricalDistribution.validate_sample(rng.normal(0.0, 1.0, (2, 300)))

    def test_rejects_empty_sample(self) -> None:
        with pytest.raises(ValueError, match="at least 1 observation, got 0"):
            EmpiricalDistribution.validate_sample(np.array([]))

    def test_kde_rejects_single_observation(self) -> None:
        with pytest.raises(ValueError, match="at least 2 observations, got 1"):
            EmpiricalDistribution(np.array([1.0]))

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match=r"1 of 3 values are NaN or infinite"):
            EmpiricalDistribution.validate_sample(np.array([1.0, 2.0, np.nan]))

    def test_rejects_infinity(self) -> None:
        with pytest.raises(ValueError, match=r"NaN or infinite"):
            EmpiricalDistribution.validate_sample(np.array([1.0, 2.0, np.inf]))

    def test_non_finite_message_reports_positions(self) -> None:
        bad = np.array([np.nan, 1.0, 2.0, np.inf, 3.0])
        with pytest.raises(ValueError, match=r"2 of 5 values .*at index 0, 3"):
            EmpiricalDistribution.validate_sample(bad)

    def test_non_finite_message_truncates_long_position_lists(self) -> None:
        bad = np.full(20, np.nan)
        with pytest.raises(ValueError, match=r"at index 0, 1, 2, 3, 4, \.\.\."):
            EmpiricalDistribution.validate_sample(bad)

    def test_kde_rejects_constant_sample(self) -> None:
        """The silent-wrong-answer case: KDE collapses to an unusable spike."""
        with pytest.raises(ValueError, match=r"constant \(all 100 values equal 3\.0\)"):
            EmpiricalDistribution(np.full(100, 3.0))

    # --- the constructor must apply all of the above ----------------------

    @pytest.mark.parametrize(
        ("label", "bad_sample"),
        [
            ("two_dimensional", np.zeros((2, 5))),
            ("empty", np.array([])),
            ("nan", np.array([1.0, 2.0, np.nan])),
        ],
    )
    def test_constructor_rejects_before_fitting(
        self, label: str, bad_sample: NDArray[np.float64]
    ) -> None:
        estimator = _ConstantEstimator(pdf_value=1.0, cdf_value=0.5)
        with pytest.raises(ValueError):
            EmpiricalDistribution(bad_sample, estimator=estimator)
        assert estimator.fit_calls == 0, f"{label}: estimator must not be fitted on a bad sample"


class TestSupportIsRejectedWhenBounded:
    """
    A bounded support is refused rather than silently violated.

    A Gaussian kernel spreads mass across the whole line, so a finite bound
    cannot be honoured without boundary correction: on positive data with
    ``support=[0, inf)`` the fit used to put ~2% of its mass below zero, with
    ``pdf(-0.5) > 0`` and ``cdf(0) > 0``.
    """

    def test_left_bounded_support_is_refused(self, sample: NDArray[np.float64]) -> None:
        from pysatl_core.distributions.support import ContinuousSupport

        with pytest.raises(NotImplementedError, match=r"bounded support \[0\.0, inf\]"):
            EmpiricalDistribution(sample, support=ContinuousSupport(left=0.0, right=np.inf))

    def test_right_bounded_support_is_refused(self, sample: NDArray[np.float64]) -> None:
        from pysatl_core.distributions.support import ContinuousSupport

        with pytest.raises(NotImplementedError, match="boundary correction"):
            EmpiricalDistribution(sample, support=ContinuousSupport(left=-np.inf, right=1.0))

    def test_fully_bounded_support_is_refused(self, sample: NDArray[np.float64]) -> None:
        from pysatl_core.distributions.support import ContinuousSupport

        with pytest.raises(NotImplementedError):
            EmpiricalDistribution(sample, support=ContinuousSupport(left=0.0, right=1.0))

    def test_unbounded_support_is_accepted(self, sample: NDArray[np.float64]) -> None:
        from pysatl_core.distributions.support import ContinuousSupport

        distr = EmpiricalDistribution(sample, support=ContinuousSupport(left=-np.inf, right=np.inf))
        assert distr.support is not None

    def test_no_support_is_the_default(self, distr: EmpiricalDistribution) -> None:
        assert distr.support is None

    def test_refusal_happens_before_fitting(self, sample: NDArray[np.float64]) -> None:
        from pysatl_core.distributions.support import ContinuousSupport

        estimator = _ConstantEstimator(pdf_value=1.0, cdf_value=0.5)
        with pytest.raises(NotImplementedError):
            EmpiricalDistribution(
                sample, estimator=estimator, support=ContinuousSupport(left=0.0, right=np.inf)
            )
        assert estimator.fit_calls == 0


class TestScipyGaussianKdePdf:
    def test_mixed_finite_infinite_and_nan_input(self, sample: NDArray[np.float64]) -> None:
        from scipy.stats import gaussian_kde

        fitted = ScipyGaussianKde().fit(sample)
        result = fitted[CharacteristicName.PDF](np.array([-1.0, -np.inf, np.nan, 1.0, np.inf]))

        np.testing.assert_allclose(result[[0, 3]], gaussian_kde(sample)([-1.0, 1.0]))
        assert result[1] == 0.0
        assert np.isnan(result[2])
        assert result[4] == 0.0

    def test_scalar_nan_propagates(self, sample: NDArray[np.float64]) -> None:
        fitted = ScipyGaussianKde().fit(sample)
        result = fitted[CharacteristicName.PDF](cast(NDArray[np.float64], np.float64(np.nan)))

        assert np.ndim(result) == 0
        assert np.isnan(result)


class TestScipyGaussianKdeCdf:
    """
    The KDE CDF is evaluated in closed form rather than by integration.

    ``scipy.stats.gaussian_kde.integrate_box_1d`` is the reference these tests
    check against: it is what the wrapper used to call point by point, so
    agreeing with it to round-off is exactly the contract that must not drift
    when the closed form is touched.
    """

    @pytest.mark.parametrize("bandwidth", ["scott", "silverman", 0.05])
    def test_matches_scipy_integration(self, sample: NDArray[np.float64], bandwidth: Any) -> None:
        from scipy.stats import gaussian_kde

        fitted = ScipyGaussianKde(bandwidth=bandwidth).fit(sample)
        kde = gaussian_kde(sample, bw_method=bandwidth)
        x = np.linspace(sample.min() - 3.0, sample.max() + 3.0, 257)
        reference = np.array([kde.integrate_box_1d(-np.inf, xi) for xi in x], dtype=float)
        assert np.allclose(fitted[CharacteristicName.CDF](x), reference, rtol=0.0, atol=1e-12)

    def test_infinities_saturate(self, sample: NDArray[np.float64]) -> None:
        fitted = ScipyGaussianKde().fit(sample)
        result = fitted[CharacteristicName.CDF](np.array([-np.inf, np.inf]))
        assert result[0] == 0.0
        assert result[1] == pytest.approx(1.0)

    def test_nan_propagates(self, sample: NDArray[np.float64]) -> None:
        fitted = ScipyGaussianKde().fit(sample)
        assert np.isnan(fitted[CharacteristicName.CDF](np.array([np.nan]))).all()

    def test_scalar_input_returns_scalar(self, sample: NDArray[np.float64]) -> None:
        fitted = ScipyGaussianKde().fit(sample)
        # The protocol is typed for arrays; passing a scalar is the point of
        # this test, so the cast states the intent rather than hiding a slip.
        result = fitted[CharacteristicName.CDF](cast(NDArray[np.float64], np.float64(0.0)))
        assert np.ndim(result) == 0
        assert 0.0 < float(result) < 1.0

    def test_result_is_monotonic_and_bounded(self, sample: NDArray[np.float64]) -> None:
        fitted = ScipyGaussianKde().fit(sample)
        values = fitted[CharacteristicName.CDF](np.linspace(-10.0, 10.0, 1001))
        assert np.all(np.diff(values) >= 0.0)
        # gaussian_kde's own weights sum to 1 only up to rounding, so the last
        # bit can sit just above 1.0 -- integrate_box_1d overshoots identically.
        assert values.min() >= 0.0
        assert values.max() <= 1.0 + 1e-12

    def test_chunking_does_not_change_the_result(
        self, sample: NDArray[np.float64], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Reaches into the chunk-size constant on purpose: the whole point is that
        splitting the work is an internal memory concern with no observable
        effect, and only forcing many chunks can demonstrate that.
        """
        from pysatl_core.distributions.empirical.estimators import scipy_gaussian_kde as module

        fitted = ScipyGaussianKde().fit(sample)
        x = np.linspace(-5.0, 5.0, 401)
        single_chunk = fitted[CharacteristicName.CDF](x)

        monkeypatch.setattr(module, "_CDF_CHUNK_ELEMENTS", 100)
        many_chunks = fitted[CharacteristicName.CDF](x)

        # Not bit-identical: a matvec over fewer rows can pick a different BLAS
        # kernel, and with it a different summation order.
        assert np.allclose(many_chunks, single_chunk, rtol=0.0, atol=1e-12)

    def test_shape_is_preserved(self, sample: NDArray[np.float64]) -> None:
        fitted = ScipyGaussianKde().fit(sample)
        assert fitted[CharacteristicName.CDF](np.zeros((3, 4))).shape == (3, 4)


class TestEmpiricalCdf:
    def test_repeated_values_define_cdf_pmf_and_discrete_support(self) -> None:
        distr = EmpiricalDistribution(np.array([1.0, 1.0, 3.0, 5.0, 5.0]), EmpiricalCdf())
        x = np.array([-np.inf, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, np.inf])

        assert distr.distribution_type == UnivariateDiscrete
        assert set(distr.analytical_computations) == {
            CharacteristicName.CDF,
            CharacteristicName.PMF,
        }
        assert isinstance(distr.support, ExplicitTableDiscreteSupport)
        np.testing.assert_array_equal(distr.support.points, [1.0, 3.0, 5.0])
        np.testing.assert_allclose(
            distr.calculate_characteristic(CharacteristicName.CDF, x),
            [0.0, 0.0, 0.4, 0.4, 0.6, 0.6, 1.0, 1.0],
        )
        np.testing.assert_allclose(
            distr.calculate_characteristic(CharacteristicName.PMF, x),
            [0.0, 0.0, 0.4, 0.0, 0.2, 0.0, 0.4, 0.0],
        )
        with pytest.raises(RuntimeError):
            distr.query_method(CharacteristicName.PDF)

    def test_single_constant_observation_is_valid(self) -> None:
        distr = EmpiricalDistribution(np.array([2.0]), EmpiricalCdf())
        np.testing.assert_array_equal(
            distr.calculate_characteristic(CharacteristicName.CDF, np.array([1.0, 2.0, 3.0])),
            [0.0, 1.0, 1.0],
        )
        np.testing.assert_array_equal(
            distr.calculate_characteristic(CharacteristicName.PMF, np.array([1.0, 2.0, 3.0])),
            [0.0, 1.0, 0.0],
        )

    def test_nan_propagates_and_shape_is_preserved(self) -> None:
        distr = EmpiricalDistribution(np.array([1.0, 2.0]), EmpiricalCdf())
        x = np.array([[np.nan, 1.0], [2.0, np.inf]])
        for name in (CharacteristicName.CDF, CharacteristicName.PMF):
            result = distr.calculate_characteristic(name, x)
            assert result.shape == x.shape
            assert np.isnan(result[0, 0])

    def test_explicit_support_must_match_observations(self) -> None:
        sample = np.array([1.0, 2.0])
        support = ExplicitTableDiscreteSupport([1.0, 3.0])
        with pytest.raises(ValueError, match="exactly the observed values"):
            EmpiricalDistribution(sample, EmpiricalCdf(), support=support)

    def test_ppf_uses_discrete_graph(self) -> None:
        distr = EmpiricalDistribution(np.array([1.0, 1.0, 3.0, 5.0]), EmpiricalCdf())
        result = distr.calculate_characteristic(
            CharacteristicName.PPF, np.array([0.1, 0.5, 0.75, 0.9])
        )
        np.testing.assert_array_equal(result, [1.0, 1.0, 3.0, 5.0])

    def test_sampling_uses_observed_values(self) -> None:
        distr = EmpiricalDistribution(np.array([1.0, 1.0, 3.0, 5.0]), EmpiricalCdf())
        samples = distr.sample(20)
        assert samples.shape[0] == 20
        assert np.isin(samples, [1.0, 3.0, 5.0]).all()

    def test_estimator_swap_changes_kind_support_and_available_functions(self) -> None:
        sample = np.array([1.0, 1.0, 3.0, 5.0])
        distr = EmpiricalDistribution(sample)
        distr.calculate_characteristic(CharacteristicName.PPF, np.array([0.5]))

        distr.set_estimator(EmpiricalCdf())
        assert distr.distribution_type == UnivariateDiscrete
        assert isinstance(distr.support, ExplicitTableDiscreteSupport)
        assert CharacteristicName.PDF not in distr.analytical_computations
        assert distr.calculate_characteristic(CharacteristicName.PMF, np.array([1.0]))[0] == 0.5

        distr.set_estimator(ScipyGaussianKde())
        assert distr.distribution_type == UnivariateContinuous
        assert distr.support is None
        assert CharacteristicName.PMF not in distr.analytical_computations
        assert np.isfinite(
            distr.calculate_characteristic(CharacteristicName.PDF, np.array([1.0]))[0]
        )

    def test_with_estimator_preserves_original_kind_and_support(self) -> None:
        original = EmpiricalDistribution(np.array([1.0, 2.0, 3.0]))
        discrete = original.with_estimator(EmpiricalCdf())
        assert original.distribution_type == UnivariateContinuous
        assert original.support is None
        assert discrete.distribution_type == UnivariateDiscrete
        assert isinstance(discrete.support, ExplicitTableDiscreteSupport)

    def test_cdf_only_estimator_uses_distribution_mapping(self) -> None:
        class _CdfOnly(_ConstantEstimator):
            def fit(
                self, sample: NDArray[np.float64]
            ) -> dict[str, AnalyticalComputation[Any, Any]]:
                return {CharacteristicName.CDF: super().fit(sample)[CharacteristicName.CDF]}

        distr = EmpiricalDistribution(np.array([1.0, 2.0]), _CdfOnly(0.0, 0.5))
        assert list(distr.analytical_computations) == [CharacteristicName.CDF]
        assert distr.calculate_characteristic(CharacteristicName.CDF, np.array([0.0]))[0] == 0.5

    def test_empty_fit_is_rejected_before_estimator_swap(self) -> None:
        class _Empty(_ConstantEstimator):
            def fit(
                self, sample: NDArray[np.float64]
            ) -> dict[str, AnalyticalComputation[Any, Any]]:
                return {}

        distr = EmpiricalDistribution(np.array([1.0, 2.0]))
        before = distr.analytical_computations
        with pytest.raises(ValueError, match="at least one analytical computation"):
            distr.set_estimator(_Empty(0.0, 0.0))
        assert distr.analytical_computations is before
        assert distr.distribution_type == UnivariateContinuous


class TestEstimatorSwapIsVisibleThroughCharacteristics:
    """
    Swapping the estimator must take effect on every characteristic at once.

    Internally this works because the analytical entries call back into the
    distribution rather than closing over a fixed estimate, but the contract
    worth pinning is the observable one: after a swap, PDF and CDF both come
    from the new estimator without the caller rebuilding anything.
    """

    def test_pdf_follows_the_current_estimator(self, distr: EmpiricalDistribution) -> None:
        distr.set_estimator(_ConstantEstimator(pdf_value=42.0, cdf_value=0.5))
        result = distr.calculate_characteristic(CharacteristicName.PDF, np.array([0.0, 1.0]))
        assert np.allclose(result, 42.0)

    def test_cdf_follows_the_current_estimator(self, distr: EmpiricalDistribution) -> None:
        distr.set_estimator(_ConstantEstimator(pdf_value=1.0, cdf_value=0.7))
        result = distr.calculate_characteristic(CharacteristicName.CDF, np.array([0.0, 2.0]))
        assert np.allclose(result, 0.7)

    def test_both_characteristics_follow_a_second_swap(self, distr: EmpiricalDistribution) -> None:
        distr.set_estimator(_ConstantEstimator(pdf_value=42.0, cdf_value=0.5))
        distr.set_estimator(_ConstantEstimator(pdf_value=7.0, cdf_value=0.9))
        x = np.array([0.0, 1.0])
        assert np.allclose(distr.calculate_characteristic(CharacteristicName.PDF, x), 7.0)
        assert np.allclose(distr.calculate_characteristic(CharacteristicName.CDF, x), 0.9)


class TestAnalyticalComputationMapping:
    """The fitted estimator contributes directly to Distribution's normal mapping."""

    def test_exposes_the_characteristics_the_fit_produced(
        self, sample: NDArray[np.float64]
    ) -> None:
        estimator = _ConstantEstimator(pdf_value=1.0, cdf_value=0.5)
        d = EmpiricalDistribution(sample, estimator=estimator)
        assert set(d.analytical_computations) == {CharacteristicName.CDF, CharacteristicName.PDF}

    def test_is_stable_across_reads(self, distr: EmpiricalDistribution) -> None:
        assert distr.analytical_computations is distr.analytical_computations

    def test_set_estimator_replaces_the_mapping(self, distr: EmpiricalDistribution) -> None:
        before = distr.analytical_computations
        distr.set_estimator(ScipyGaussianKde(bandwidth="silverman"))
        assert distr.analytical_computations is not before

    def test_with_estimator_leaves_the_original_mapping_alone(
        self, distr: EmpiricalDistribution
    ) -> None:
        before = distr.analytical_computations
        clone = distr.with_estimator(ScipyGaussianKde(bandwidth="silverman"))
        assert distr.analytical_computations is before
        assert clone.analytical_computations is not before

    def test_is_read_only(self, distr: EmpiricalDistribution) -> None:
        with pytest.raises(AttributeError):
            distr.analytical_computations = distr.analytical_computations  # type: ignore[misc]


class TestDefaultStrategy:
    def test_default_computation_strategy_caches(self, distr: EmpiricalDistribution) -> None:
        strategy = distr.computation_strategy
        assert type(strategy) is DefaultComputationStrategy
        assert strategy.is_caching_enabled


class TestSampleSnapshot:
    def test_mutations_after_ppf_preparation_leave_original_and_clone_unchanged(self) -> None:
        sample = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        distr = EmpiricalDistribution(sample)
        clone = distr.with_estimator(ScipyGaussianKde(bandwidth="silverman"))
        x = np.array([-0.5, 0.5])
        q = np.array([0.25, 0.5, 0.75])

        before = [
            (
                current.calculate_characteristic(CharacteristicName.PDF, x),
                current.calculate_characteristic(CharacteristicName.CDF, x),
                current.calculate_characteristic(CharacteristicName.PPF, q),
            )
            for current in (distr, clone)
        ]

        sample[:] += 100.0
        public_data = distr.data
        with pytest.raises(ValueError):
            public_data[:] += 100.0
        with pytest.raises(ValueError):
            public_data.setflags(write=True)

        assert clone.data is public_data
        np.testing.assert_array_equal(public_data, [-2.0, -1.0, 0.0, 1.0, 2.0])
        for current, (pdf, cdf, ppf) in zip((distr, clone), before, strict=True):
            np.testing.assert_array_equal(
                current.calculate_characteristic(CharacteristicName.PDF, x), pdf
            )
            np.testing.assert_array_equal(
                current.calculate_characteristic(CharacteristicName.CDF, x), cdf
            )
            np.testing.assert_array_equal(
                current.calculate_characteristic(CharacteristicName.PPF, q), ppf
            )


class TestWithEstimator:
    def test_returns_new_instance(self, distr: EmpiricalDistribution) -> None:
        clone = distr.with_estimator(ScipyGaussianKde(bandwidth="silverman"))
        assert clone is not distr
        assert isinstance(clone, EmpiricalDistribution)

    def test_shares_sample(self, distr: EmpiricalDistribution) -> None:
        clone = distr.with_estimator(ScipyGaussianKde(bandwidth="silverman"))
        assert clone.data is distr.data

    def test_changes_pdf(self, sample: NDArray[np.float64]) -> None:
        scott = EmpiricalDistribution(sample, estimator=ScipyGaussianKde(bandwidth="scott"))
        silverman = scott.with_estimator(ScipyGaussianKde(bandwidth="silverman"))

        x = np.array([0.0, 0.5, 1.0])
        pdf_scott = scott.calculate_characteristic(CharacteristicName.PDF, x)
        pdf_silverman = silverman.calculate_characteristic(CharacteristicName.PDF, x)

        assert not np.allclose(pdf_scott, pdf_silverman)

    def test_preserves_original(self, distr: EmpiricalDistribution) -> None:
        x = np.array([0.0, 1.0])
        pdf_before = distr.calculate_characteristic(CharacteristicName.PDF, x)

        distr.with_estimator(ScipyGaussianKde(bandwidth="silverman"))

        pdf_after = distr.calculate_characteristic(CharacteristicName.PDF, x)
        assert np.allclose(pdf_before, pdf_after)

    def test_independent_strategies(self, distr: EmpiricalDistribution) -> None:
        clone = distr.with_estimator(ScipyGaussianKde(bandwidth="silverman"))
        assert clone.sampling_strategy is not distr.sampling_strategy
        assert clone.computation_strategy is not distr.computation_strategy

    def test_sampling_config_is_independent_of_clone(self, sample: NDArray[np.float64]) -> None:
        from pysatl_core.sampling.unuran.core.unuran_sampling_strategy import (
            DefaultUnuranSamplingStrategy,
        )
        from pysatl_core.sampling.unuran.method_config import UnuranMethodConfig

        strategy = DefaultUnuranSamplingStrategy(
            UnuranMethodConfig(method_params={"nested": {"values": [1]}})
        )
        distr = EmpiricalDistribution(sample, sampling_strategy=strategy)
        clone = distr.with_estimator(ScipyGaussianKde(bandwidth="silverman"))

        assert isinstance(clone.sampling_strategy, DefaultUnuranSamplingStrategy)
        clone_params = clone.sampling_strategy.config.method_params
        original_params = strategy.config.method_params
        assert clone_params is not None
        assert original_params is not None
        clone_params["nested"]["values"].append(2)
        assert original_params["nested"]["values"] == [1]

    def test_estimator_property_reflects_new_estimator(self, distr: EmpiricalDistribution) -> None:
        new_estimator = ScipyGaussianKde(bandwidth="silverman")
        clone = distr.with_estimator(new_estimator)
        assert clone.estimator is new_estimator
        assert distr.estimator is not new_estimator

    def test_calls_fit_exactly_once_on_new_estimator(self, sample: NDArray[np.float64]) -> None:
        estimator = _ConstantEstimator(pdf_value=1.0, cdf_value=0.5)
        distr = EmpiricalDistribution(sample)

        clone = distr.with_estimator(estimator)

        assert estimator.fit_calls == 1
        assert np.allclose(
            clone.calculate_characteristic(CharacteristicName.PDF, np.array([0.0])),
            1.0,
        )

    def test_composes_with_sampling_strategy_override(self, distr: EmpiricalDistribution) -> None:
        from pysatl_core.sampling.unuran.core.unuran_sampling_strategy import (
            DefaultUnuranSamplingStrategy,
        )

        new_sampling = DefaultUnuranSamplingStrategy()
        chained = distr.with_estimator(
            ScipyGaussianKde(bandwidth="silverman")
        ).with_sampling_strategy(new_sampling)
        assert chained.sampling_strategy is new_sampling
        assert chained.estimator is not distr.estimator


class TestSetEstimator:
    def test_default_caching_strategy_rebuilds_ppf_after_estimator_swap(
        self, sample: NDArray[np.float64]
    ) -> None:
        strategy = DefaultComputationStrategy(enable_caching=True)
        distr = EmpiricalDistribution(
            sample,
            estimator=ScipyGaussianKde(bandwidth=2.0),
            computation_strategy=strategy,
        )
        q = np.array([0.1, 0.9])
        before = distr.calculate_characteristic(CharacteristicName.PPF, q)

        new_estimator = ScipyGaussianKde(bandwidth=0.05)
        distr.set_estimator(new_estimator)
        after = distr.calculate_characteristic(CharacteristicName.PPF, q)
        fresh = EmpiricalDistribution(sample, estimator=new_estimator)
        expected = fresh.calculate_characteristic(CharacteristicName.PPF, q)

        assert not np.allclose(after, before)
        np.testing.assert_allclose(after, expected)

    def test_failed_fit_preserves_current_distribution(self, distr: EmpiricalDistribution) -> None:
        class _FailingEstimator(_ConstantEstimator):
            def fit(
                self, sample: NDArray[np.float64]
            ) -> dict[str, AnalyticalComputation[Any, Any]]:
                raise ValueError("cannot fit")

        original_estimator = distr.estimator
        original_computations = distr.analytical_computations
        x = np.array([-0.5, 0.5])
        pdf_before = distr.calculate_characteristic(CharacteristicName.PDF, x)
        ppf_before = distr.calculate_characteristic(CharacteristicName.PPF, np.array([0.25, 0.75]))

        with pytest.raises(ValueError, match="cannot fit"):
            distr.set_estimator(_FailingEstimator(1.0, 0.5))

        assert distr.estimator is original_estimator
        assert distr.analytical_computations is original_computations
        np.testing.assert_array_equal(
            distr.calculate_characteristic(CharacteristicName.PDF, x), pdf_before
        )
        np.testing.assert_array_equal(
            distr.calculate_characteristic(CharacteristicName.PPF, np.array([0.25, 0.75])),
            ppf_before,
        )

    def test_mutates_in_place_returns_none(self, distr: EmpiricalDistribution) -> None:
        original_id = id(distr)
        distr.set_estimator(ScipyGaussianKde(bandwidth="silverman"))
        assert id(distr) == original_id

    def test_estimator_property_reflects_new_estimator(self, distr: EmpiricalDistribution) -> None:
        new_estimator = ScipyGaussianKde(bandwidth="silverman")
        distr.set_estimator(new_estimator)
        assert distr.estimator is new_estimator

    def test_pdf_changes_after_swap(self, distr: EmpiricalDistribution) -> None:
        x = np.array([0.0, 0.5, 1.0])
        pdf_before = distr.calculate_characteristic(CharacteristicName.PDF, x)
        distr.set_estimator(ScipyGaussianKde(bandwidth="silverman"))
        pdf_after = distr.calculate_characteristic(CharacteristicName.PDF, x)
        assert not np.allclose(pdf_before, pdf_after)

    def test_cdf_changes_after_swap(self, distr: EmpiricalDistribution) -> None:
        x = np.array([-1.0, 0.0, 1.0])
        cdf_before = distr.calculate_characteristic(CharacteristicName.CDF, x)
        distr.set_estimator(ScipyGaussianKde(bandwidth=0.05))
        cdf_after = distr.calculate_characteristic(CharacteristicName.CDF, x)
        assert not np.allclose(cdf_before, cdf_after)

    def test_calls_fit_exactly_once(self, distr: EmpiricalDistribution) -> None:
        estimator = _ConstantEstimator(pdf_value=7.0, cdf_value=0.3)
        distr.set_estimator(estimator)
        assert estimator.fit_calls == 1
        assert np.allclose(
            distr.calculate_characteristic(CharacteristicName.PDF, np.array([0.0])),
            7.0,
        )

    def test_sample_after_swap_rebuilds_sampler(
        self, distr: EmpiricalDistribution, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        built: list[Any] = []

        class _StubSampler:
            def __init__(self, distr: Any, config: Any) -> None:
                built.append(self)

            def sample(self, n: int) -> NDArray[np.float64]:
                return np.zeros(n)

        monkeypatch.setattr(
            "pysatl_core.sampling.unuran.core.unuran_sampling_strategy.DefaultUnuranSampler",
            _StubSampler,
        )
        _ = distr.sample(5)
        distr.set_estimator(ScipyGaussianKde(bandwidth="silverman"))
        out = distr.sample(50)

        assert out.shape[0] == 50
        assert len(built) == 2
        assert built[0] is not built[1]

    def test_works_when_strategies_lack_invalidate(self, sample: NDArray[np.float64]) -> None:
        class _BareSampling:
            def sample(self, n: int, distr: Any, **options: Any) -> NDArray[np.float64]:
                return np.zeros(n)

        class _BareComputation:
            def query_method(self, state: Any, distr: Any, **options: Any) -> Any:
                return distr.analytical_computations[CharacteristicName.PDF][
                    next(iter(distr.analytical_computations[CharacteristicName.PDF]))
                ]

            def explain_computation_path(self, state: Any, distr: Any) -> Any:
                raise NotImplementedError

        d = EmpiricalDistribution(
            sample,
            sampling_strategy=_BareSampling(),
            computation_strategy=_BareComputation(),  # type: ignore[arg-type]
        )

        d.set_estimator(ScipyGaussianKde(bandwidth="silverman"))
