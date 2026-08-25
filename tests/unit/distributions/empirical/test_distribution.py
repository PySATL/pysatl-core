"""
Unit tests for EmpiricalDistribution: sample validation, with_method, set_method.

Tests here assert observable behaviour through the public API.  The one
deliberate exception -- ``test_computation_cache_cleared_via_estimator_tracking``
-- reaches into the strategy cache and says so in its own docstring.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Hashable
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from pysatl_core.distributions.empirical import (
    EmpiricalComputationStrategy,
    EmpiricalDistribution,
    FittedEmpirical,
    ScipyGaussianKde,
)
from pysatl_core.types import CharacteristicName


class _ConstantFittedEmpirical:
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


class _ConstantMethod:
    def __init__(self, pdf_value: float, cdf_value: float) -> None:
        self._pdf_value = pdf_value
        self._cdf_value = cdf_value
        self.fit_calls = 0

    def fit(self, sample: NDArray[np.float64]) -> FittedEmpirical:
        self.fit_calls += 1
        return _ConstantFittedEmpirical(self._pdf_value, self._cdf_value)


@pytest.fixture
def sample() -> NDArray[np.float64]:
    rng = np.random.default_rng(42)
    return rng.normal(0.0, 1.0, 300)


@pytest.fixture
def distr(sample: NDArray[np.float64]) -> EmpiricalDistribution:
    return EmpiricalDistribution(sample)


class TestValidateSample:
    """
    Input validation happens in the constructor, before the estimator is fitted.

    Each case below used to surface either as an error from scipy's internals
    at an unpredictable moment, or -- for a constant sample -- not at all.
    """

    def test_accepts_a_normal_sample(self, sample: NDArray[np.float64]) -> None:
        validated = EmpiricalDistribution.validate_sample(sample)
        assert validated.ndim == 1
        assert validated.dtype == np.float64
        assert np.array_equal(validated, sample)

    def test_coerces_a_python_list(self) -> None:
        validated = EmpiricalDistribution.validate_sample(cast(Any, [1.0, 2.0, 3.5]))
        assert isinstance(validated, np.ndarray)
        assert validated.dtype == np.float64

    def test_minimum_two_observations_is_enough(self) -> None:
        assert EmpiricalDistribution.validate_sample(np.array([1.0, 2.0])).size == 2

    # --- rejections -------------------------------------------------------

    def test_rejects_two_dimensional_sample(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match=r"univariate.*1-D.*shape \(2, 300\)"):
            EmpiricalDistribution.validate_sample(rng.normal(0.0, 1.0, (2, 300)))

    def test_rejects_empty_sample(self) -> None:
        with pytest.raises(ValueError, match="at least 2 observations, got 0"):
            EmpiricalDistribution.validate_sample(np.array([]))

    def test_rejects_single_observation(self) -> None:
        with pytest.raises(ValueError, match="at least 2 observations, got 1"):
            EmpiricalDistribution.validate_sample(np.array([1.0]))

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

    def test_rejects_constant_sample(self) -> None:
        """The silent-wrong-answer case: KDE collapses to an unusable spike."""
        with pytest.raises(ValueError, match=r"constant \(all 100 values equal 3\.0\)"):
            EmpiricalDistribution.validate_sample(np.full(100, 3.0))

    # --- the constructor must apply all of the above ----------------------

    @pytest.mark.parametrize(
        ("label", "bad_sample"),
        [
            ("two_dimensional", np.zeros((2, 5))),
            ("empty", np.array([])),
            ("single", np.array([1.0])),
            ("nan", np.array([1.0, 2.0, np.nan])),
            ("constant", np.full(10, 3.0)),
        ],
    )
    def test_constructor_rejects_before_fitting(
        self, label: str, bad_sample: NDArray[np.float64]
    ) -> None:
        method = _ConstantMethod(pdf_value=1.0, cdf_value=0.5)
        with pytest.raises(ValueError):
            EmpiricalDistribution(bad_sample, method=method)
        assert method.fit_calls == 0, f"{label}: estimator must not be fitted on a bad sample"


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

        with pytest.raises(NotImplementedError, match="boundary-corrected"):
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

        method = _ConstantMethod(pdf_value=1.0, cdf_value=0.5)
        with pytest.raises(NotImplementedError):
            EmpiricalDistribution(
                sample, method=method, support=ContinuousSupport(left=0.0, right=np.inf)
            )
        assert method.fit_calls == 0


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
        assert np.allclose(fitted.cdf(x), reference, rtol=0.0, atol=1e-12)

    def test_infinities_saturate(self, sample: NDArray[np.float64]) -> None:
        fitted = ScipyGaussianKde().fit(sample)
        result = fitted.cdf(np.array([-np.inf, np.inf]))
        assert result[0] == 0.0
        assert result[1] == pytest.approx(1.0)

    def test_nan_propagates(self, sample: NDArray[np.float64]) -> None:
        fitted = ScipyGaussianKde().fit(sample)
        assert np.isnan(fitted.cdf(np.array([np.nan]))).all()

    def test_scalar_input_returns_scalar(self, sample: NDArray[np.float64]) -> None:
        fitted = ScipyGaussianKde().fit(sample)
        # The protocol is typed for arrays; passing a scalar is the point of
        # this test, so the cast states the intent rather than hiding a slip.
        result = fitted.cdf(cast(NDArray[np.float64], np.float64(0.0)))
        assert np.ndim(result) == 0
        assert 0.0 < float(result) < 1.0

    def test_result_is_monotonic_and_bounded(self, sample: NDArray[np.float64]) -> None:
        fitted = ScipyGaussianKde().fit(sample)
        values = fitted.cdf(np.linspace(-10.0, 10.0, 1001))
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
        from pysatl_core.distributions.empirical import distribution as module

        fitted = ScipyGaussianKde().fit(sample)
        x = np.linspace(-5.0, 5.0, 401)
        single_chunk = fitted.cdf(x)

        monkeypatch.setattr(module, "_CDF_CHUNK_ELEMENTS", 100)
        many_chunks = fitted.cdf(x)

        # Not bit-identical: a matvec over fewer rows can pick a different BLAS
        # kernel, and with it a different summation order.
        assert np.allclose(many_chunks, single_chunk, rtol=0.0, atol=1e-12)

    def test_shape_is_preserved(self, sample: NDArray[np.float64]) -> None:
        fitted = ScipyGaussianKde().fit(sample)
        assert fitted.cdf(np.zeros((3, 4))).shape == (3, 4)


class TestMethodSwapIsVisibleThroughCharacteristics:
    """
    Swapping the method must take effect on every characteristic at once.

    Internally this works because the analytical entries call back into the
    distribution rather than closing over a fixed estimator, but the contract
    worth pinning is the observable one: after a swap, PDF and CDF both come
    from the new method without the caller rebuilding anything.
    """

    def test_pdf_follows_the_current_method(self, distr: EmpiricalDistribution) -> None:
        distr.set_method(_ConstantMethod(pdf_value=42.0, cdf_value=0.5))
        result = distr.calculate_characteristic(CharacteristicName.PDF, np.array([0.0, 1.0]))
        assert np.allclose(result, 42.0)

    def test_cdf_follows_the_current_method(self, distr: EmpiricalDistribution) -> None:
        distr.set_method(_ConstantMethod(pdf_value=1.0, cdf_value=0.7))
        result = distr.calculate_characteristic(CharacteristicName.CDF, np.array([0.0, 2.0]))
        assert np.allclose(result, 0.7)

    def test_both_characteristics_follow_a_second_swap(self, distr: EmpiricalDistribution) -> None:
        distr.set_method(_ConstantMethod(pdf_value=42.0, cdf_value=0.5))
        distr.set_method(_ConstantMethod(pdf_value=7.0, cdf_value=0.9))
        x = np.array([0.0, 1.0])
        assert np.allclose(distr.calculate_characteristic(CharacteristicName.PDF, x), 7.0)
        assert np.allclose(distr.calculate_characteristic(CharacteristicName.CDF, x), 0.9)


class TestEstimatorProperty:
    """
    ``estimator`` is public API, not an accessor added for convenience.

    EmpiricalComputationStrategy invalidates its cache by comparing this value
    by identity, so the contract pinned here -- a stable object that is
    *replaced* on every refit -- is what that invalidation rests on.
    """

    def test_exposes_the_object_the_method_produced(self, sample: NDArray[np.float64]) -> None:
        method = _ConstantMethod(pdf_value=1.0, cdf_value=0.5)
        d = EmpiricalDistribution(sample, method=method)
        assert isinstance(d.estimator, _ConstantFittedEmpirical)

    def test_is_stable_across_reads(self, distr: EmpiricalDistribution) -> None:
        assert distr.estimator is distr.estimator

    def test_set_method_replaces_the_object(self, distr: EmpiricalDistribution) -> None:
        before = distr.estimator
        distr.set_method(ScipyGaussianKde(bandwidth="silverman"))
        assert distr.estimator is not before

    def test_with_method_leaves_the_original_estimator_alone(
        self, distr: EmpiricalDistribution
    ) -> None:
        before = distr.estimator
        clone = distr.with_method(ScipyGaussianKde(bandwidth="silverman"))
        assert distr.estimator is before
        assert clone.estimator is not before

    def test_is_read_only(self, distr: EmpiricalDistribution) -> None:
        """
        No setter: rebinding the estimator alone would leave the sampler holding
        state derived from the previous fit.  set_method() is the supported way.
        """
        with pytest.raises(AttributeError):
            distr.estimator = distr.estimator  # type: ignore[misc]


class TestDefaultStrategy:
    def test_default_computation_strategy_is_empirical(self, distr: EmpiricalDistribution) -> None:
        assert isinstance(distr.computation_strategy, EmpiricalComputationStrategy)


class TestWithMethod:
    def test_returns_new_instance(self, distr: EmpiricalDistribution) -> None:
        clone = distr.with_method(ScipyGaussianKde(bandwidth="silverman"))
        assert clone is not distr
        assert isinstance(clone, EmpiricalDistribution)

    def test_shares_sample(self, distr: EmpiricalDistribution) -> None:
        clone = distr.with_method(ScipyGaussianKde(bandwidth="silverman"))
        assert clone.data is distr.data

    def test_changes_pdf(self, sample: NDArray[np.float64]) -> None:
        scott = EmpiricalDistribution(sample, method=ScipyGaussianKde(bandwidth="scott"))
        silverman = scott.with_method(ScipyGaussianKde(bandwidth="silverman"))

        x = np.array([0.0, 0.5, 1.0])
        pdf_scott = scott.calculate_characteristic(CharacteristicName.PDF, x)
        pdf_silverman = silverman.calculate_characteristic(CharacteristicName.PDF, x)

        assert not np.allclose(pdf_scott, pdf_silverman)

    def test_preserves_original(self, distr: EmpiricalDistribution) -> None:
        x = np.array([0.0, 1.0])
        pdf_before = distr.calculate_characteristic(CharacteristicName.PDF, x)

        distr.with_method(ScipyGaussianKde(bandwidth="silverman"))

        pdf_after = distr.calculate_characteristic(CharacteristicName.PDF, x)
        assert np.allclose(pdf_before, pdf_after)

    def test_independent_strategies(self, distr: EmpiricalDistribution) -> None:
        clone = distr.with_method(ScipyGaussianKde(bandwidth="silverman"))
        assert clone.sampling_strategy is not distr.sampling_strategy
        assert clone.computation_strategy is not distr.computation_strategy

    def test_method_property_reflects_new_method(self, distr: EmpiricalDistribution) -> None:
        new_method = ScipyGaussianKde(bandwidth="silverman")
        clone = distr.with_method(new_method)
        assert clone.method is new_method
        assert distr.method is not new_method

    def test_calls_fit_exactly_once_on_new_method(self, sample: NDArray[np.float64]) -> None:
        method = _ConstantMethod(pdf_value=1.0, cdf_value=0.5)
        distr = EmpiricalDistribution(sample)

        clone = distr.with_method(method)

        assert method.fit_calls == 1
        assert np.allclose(
            clone.calculate_characteristic(CharacteristicName.PDF, np.array([0.0])),
            1.0,
        )

    def test_composes_with_sampling_strategy_override(self, distr: EmpiricalDistribution) -> None:
        from pysatl_core.sampling.unuran.core.unuran_sampling_strategy import (
            DefaultUnuranSamplingStrategy,
        )

        new_sampling = DefaultUnuranSamplingStrategy()
        chained = distr.with_method(ScipyGaussianKde(bandwidth="silverman")).with_sampling_strategy(
            new_sampling
        )
        assert chained.sampling_strategy is new_sampling
        assert chained.method is not distr.method


class TestSetMethod:
    def test_mutates_in_place_returns_none(self, distr: EmpiricalDistribution) -> None:
        original_id = id(distr)
        distr.set_method(ScipyGaussianKde(bandwidth="silverman"))
        assert id(distr) == original_id

    def test_method_property_reflects_new_method(self, distr: EmpiricalDistribution) -> None:
        new_method = ScipyGaussianKde(bandwidth="silverman")
        distr.set_method(new_method)
        assert distr.method is new_method

    def test_pdf_changes_after_swap(self, distr: EmpiricalDistribution) -> None:
        x = np.array([0.0, 0.5, 1.0])
        pdf_before = distr.calculate_characteristic(CharacteristicName.PDF, x)
        distr.set_method(ScipyGaussianKde(bandwidth="silverman"))
        pdf_after = distr.calculate_characteristic(CharacteristicName.PDF, x)
        assert not np.allclose(pdf_before, pdf_after)

    def test_cdf_changes_after_swap(self, distr: EmpiricalDistribution) -> None:
        x = np.array([-1.0, 0.0, 1.0])
        cdf_before = distr.calculate_characteristic(CharacteristicName.CDF, x)
        distr.set_method(ScipyGaussianKde(bandwidth=0.05))
        cdf_after = distr.calculate_characteristic(CharacteristicName.CDF, x)
        assert not np.allclose(cdf_before, cdf_after)

    def test_calls_fit_exactly_once(self, distr: EmpiricalDistribution) -> None:
        method = _ConstantMethod(pdf_value=7.0, cdf_value=0.3)
        distr.set_method(method)
        assert method.fit_calls == 1
        assert np.allclose(
            distr.calculate_characteristic(CharacteristicName.PDF, np.array([0.0])),
            7.0,
        )

    def test_computation_cache_cleared_via_estimator_tracking(
        self, sample: NDArray[np.float64]
    ) -> None:
        """
        White-box: pins the invalidation *mechanism*, not a public contract.

        Cache clearing has no cheap observable of its own -- the behavioural
        guarantee it backs ("characteristics follow the current method") is
        covered by TestMethodSwapIsVisibleThroughCharacteristics.  Expect this
        test to be rewritten alongside any change to how fits are cached; a
        failure here does not by itself mean behaviour regressed.
        """
        strategy = EmpiricalComputationStrategy(enable_caching=True)
        d = EmpiricalDistribution(sample, computation_strategy=strategy)

        from pysatl_core.distributions.computations.computation import FittedComputationMethod

        strategy.query_method(CharacteristicName.PDF, d)
        # Key shape mirrors DefaultComputationStrategy._cache:
        # (distr_id, edge_id, target, frozen_options).
        sentinel_key: tuple[int, int, str, frozenset[tuple[str, Hashable]]] = (
            0,
            0,
            CharacteristicName.PPF,
            frozenset(),
        )
        strategy._cache[sentinel_key] = FittedComputationMethod[Any, Any](
            target=CharacteristicName.PPF,
            sources=(CharacteristicName.CDF,),
            func=lambda *a, **kw: 0.0,
        )
        assert sentinel_key in strategy._cache

        d.set_method(ScipyGaussianKde(bandwidth="silverman"))

        strategy.query_method(CharacteristicName.PDF, d)
        assert sentinel_key not in strategy._cache

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
        distr.set_method(ScipyGaussianKde(bandwidth="silverman"))
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

        d.set_method(ScipyGaussianKde(bandwidth="silverman"))
