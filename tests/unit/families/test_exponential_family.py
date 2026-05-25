from collections.abc import Callable

__author__ = "Vinogradov Ilya"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import itertools
from typing import cast

import numpy as np
import pytest
import scipy
from numpy.testing import assert_allclose

from pysatl_core.distributions.support import ContinuousNDSupport, PredicateSupport
from pysatl_core.families import (
    ContinuousExponentialClassFamily,
    ExponentialConjugateHyperparameters,
    ExponentialFamilyParametrization,
)
from pysatl_core.families.registry import ParametricFamilyRegister
from pysatl_core.types import (
    CharacteristicName,
    Interval1D,
    Number,
    NumericArray,
    UnivariateContinuous,
)


def gamma_pdf(alpha: float, beta: float, x: float) -> float:
    return scipy.stats.gamma(a=alpha, scale=1 / beta).pdf(x).item()  # type: ignore[attr-defined]


def lomax_pdf(shape: float, scale: float, x: float) -> float:
    return scipy.stats.lomax(c=shape, scale=scale).pdf(x).item()  # type: ignore[attr-defined]


def exponential_log_partition(parametrization):
    return np.log(-parametrization)


def _make_exponential_family() -> ContinuousExponentialClassFamily:
    support_neg = PredicateSupport(
        predicate=lambda x: bool(
            ContinuousNDSupport(
                intervals=[Interval1D(-np.inf, 0, left_closed=False, right_closed=False)]
            ).contains(np.array([x]))
        )
    )
    support_pos = PredicateSupport(
        predicate=lambda x: bool(
            ContinuousNDSupport(
                intervals=[Interval1D(0, np.inf, left_closed=False, right_closed=False)]
            ).contains(np.array([x]))
        )
    )
    return ContinuousExponentialClassFamily(
        name="ExponentialFamily",
        log_partition=exponential_log_partition,
        sufficient_statistics=lambda x: x,
        normalization_constant=lambda _: 1,
        parameter_space=support_neg,
        sufficient_statistics_values=support_pos,
        support=support_pos,
        distr_type=UnivariateContinuous,
        distr_parametrizations=["theta"],
    )


@pytest.fixture(scope="function")
def exponential_family() -> ContinuousExponentialClassFamily:
    return _make_exponential_family()


@pytest.fixture(scope="function")
def conjugate_for_exponential() -> ContinuousExponentialClassFamily:
    def transform_function(x: NumericArray) -> NumericArray:
        return -x

    fam = _make_exponential_family()
    conjugate_fam = fam.conjugate_prior_family.transform(transform_function)
    ParametricFamilyRegister().register(conjugate_fam)
    return cast(
        ContinuousExponentialClassFamily,
        ParametricFamilyRegister().get("TransformedExponentialFamily"),
    )


def test_log_density_matches_exponential_form(
    exponential_family: ContinuousExponentialClassFamily,
) -> None:
    params = ExponentialFamilyParametrization(theta=np.array([-2.0]))
    log_density_func = cast(
        Callable[[ExponentialFamilyParametrization, NumericArray], Number],
        exponential_family.log_density,
    )
    density_func = cast(
        Callable[[ExponentialFamilyParametrization, NumericArray], Number],
        exponential_family.density,
    )

    log_density = log_density_func(params, np.asarray(0.5))
    density = density_func(params, np.asarray(0.5))

    assert log_density == pytest.approx(np.log(2.0) - 1.0)
    assert density == pytest.approx(2.0 * np.exp(-1.0))


def test_constructor_merges_custom_characteristics() -> None:
    support_neg = PredicateSupport(
        predicate=lambda x: bool(
            ContinuousNDSupport(intervals=[Interval1D(-np.inf, 0)]).contains(np.array([x]))
        )
    )
    support_pos = PredicateSupport(
        predicate=lambda x: bool(
            ContinuousNDSupport(intervals=[Interval1D(0, np.inf)]).contains(np.array([x]))
        )
    )
    family = ContinuousExponentialClassFamily(
        name="ExponentialFamily",
        log_partition=exponential_log_partition,
        sufficient_statistics=lambda x: x,
        normalization_constant=lambda _: 1,
        parameter_space=support_neg,
        sufficient_statistics_values=support_pos,
        support=support_pos,
        distr_type=UnivariateContinuous,
        distr_parametrizations=["theta"],
        distr_characteristics={CharacteristicName.CDF: lambda _params, x: x / (1 + x)},
    )

    assert CharacteristicName.CDF in family.distr_characteristics
    assert CharacteristicName.PDF in family.distr_characteristics
    assert CharacteristicName.MEAN in family.distr_characteristics
    assert CharacteristicName.VAR in family.distr_characteristics


def test_log_density_is_minus_infinity_outside_support(
    exponential_family: ContinuousExponentialClassFamily,
) -> None:
    params = ExponentialFamilyParametrization(theta=np.array([-2.0]))
    log_density_func = cast(
        Callable[[ExponentialFamilyParametrization, NumericArray], Number],
        exponential_family.log_density,
    )
    density_func = cast(
        Callable[[ExponentialFamilyParametrization, NumericArray], Number],
        exponential_family.density,
    )

    assert log_density_func(params, np.asarray(-0.1)) == -np.inf
    assert density_func(params, np.asarray(-0.1)) == 0.0


def test_distribution_rejects_theta_outside_parameter_space(
    exponential_family: ContinuousExponentialClassFamily,
) -> None:
    with pytest.raises(ValueError, match="theta belongs to parameter_space"):
        exponential_family(theta=np.array([0.0]), parametrization_name="theta")


def test_transform_with_negation_moves_support_and_preserves_density(
    exponential_family: ContinuousExponentialClassFamily,
) -> None:
    transformed = exponential_family.transform(lambda x: -x)
    params = ExponentialFamilyParametrization(theta=np.array([-1.5]))
    transformed_log_density = cast(
        Callable[[ExponentialFamilyParametrization, NumericArray], Number],
        transformed.log_density,
    )
    log_density = cast(
        Callable[[ExponentialFamilyParametrization, NumericArray], Number],
        exponential_family.log_density,
    )

    assert transformed.name == "TransformedExponentialFamily"
    assert transformed_log_density(params, np.asarray(-2.0)) == pytest.approx(
        log_density(params, np.asarray(2.0))
    )
    assert transformed_log_density(params, np.asarray(2.0)) == -np.inf


def test_posterior_hyperparameters_updates_sample_without_mutating_input(
    exponential_family: ContinuousExponentialClassFamily,
) -> None:
    prior = ExponentialConjugateHyperparameters(
        effective_suff_stat_value=np.array([3.0]),
        effective_sample_size=2.0,
    )

    posterior = exponential_family.posterior_hyperparameters(prior, sample=[0.5, 1.5])

    assert_allclose(posterior.effective_suff_stat_value, np.array([5.0]))
    assert posterior.effective_sample_size == 4.0
    assert_allclose(prior.effective_suff_stat_value, np.array([3.0]))
    assert prior.effective_sample_size == 2.0


def test_posterior_hyperparameters_accepts_single_observation(
    exponential_family: ContinuousExponentialClassFamily,
) -> None:
    prior = ExponentialConjugateHyperparameters(
        effective_suff_stat_value=np.array([3.0]),
        effective_sample_size=2.0,
    )

    posterior = exponential_family.posterior_hyperparameters(prior, sample=0.5)  # type: ignore[arg-type]

    assert_allclose(posterior.effective_suff_stat_value, np.array([3.5]))
    assert posterior.effective_sample_size == 3.0


def test_posterior_predictive_builds_family_with_posterior_parametrization(
    exponential_family: ContinuousExponentialClassFamily,
) -> None:
    predictive_family = exponential_family.posterior_predictive

    assert predictive_family.name == "PosteriorPredictiveExponentialFamily"
    assert predictive_family.parametrization_names == ["posterior"]
    assert predictive_family.get_parametrization("posterior") is ExponentialConjugateHyperparameters
    assert CharacteristicName.PDF in predictive_family.distr_characteristics


@pytest.mark.parametrize(
    ("xi", "nu"),
    itertools.product((2.0, 3.0, 4.0), (2.0, 3.0, 4.0)),
)
def test_posterior_predictive_matches_lomax_density(
    exponential_family: ContinuousExponentialClassFamily,
    xi: float,
    nu: float,
) -> None:
    predictive = exponential_family.posterior_predictive.distribution(
        parametrization_name="posterior",
        effective_suff_stat_value=np.array([xi]),
        effective_sample_size=nu,
    )
    pdf = predictive.computation_strategy.query_method("pdf", distr=predictive)
    x_values = np.array([0.0, 0.5, 1.5, 3.0, 6.0])

    actual = np.asarray([pdf(x) for x in x_values], dtype=float).reshape(-1)
    expected = np.asarray([lomax_pdf(shape=nu + 1, scale=xi, x=x) for x in x_values])

    assert_allclose(actual, expected, rtol=1e-6)


@pytest.mark.parametrize(
    ("theta1", "theta2"),
    itertools.product(range(2, 5), range(2, 5)),
)
def test_exponential_pdf(theta1, theta2, conjugate_for_exponential):
    gamma_family: ContinuousExponentialClassFamily = conjugate_for_exponential

    alpha = theta2 + 1
    beta = theta1

    exponential = gamma_family(theta=np.array([theta1, theta2]), parametrization_name="theta")
    pdf = exponential.computation_strategy.query_method("pdf", distr=exponential)

    x = [i / 10 for i in range(100)]

    assert_allclose([pdf(xx) for xx in x], [gamma_pdf(alpha, beta, xx) for xx in x], rtol=1e-6)


@pytest.mark.parametrize(
    ("theta1", "theta2"),
    itertools.product(range(2, 5), range(2, 5)),
)
def test_exponential_mean(theta1, theta2, conjugate_for_exponential):
    gamma_family: ContinuousExponentialClassFamily = conjugate_for_exponential

    alpha = theta2 + 1
    beta = theta1

    exponential = gamma_family(theta=np.array([theta1, theta2]), parametrization_name="theta")
    mean = exponential.computation_strategy.query_method(CharacteristicName.MEAN, distr=exponential)
    assert np.isclose(mean(), alpha / beta, rtol=1e-6)


@pytest.mark.parametrize(
    ("theta1", "theta2"),
    itertools.product(range(2, 5), range(2, 5)),
)
def test_exponential_var(theta1, theta2, conjugate_for_exponential):
    gamma_family: ContinuousExponentialClassFamily = conjugate_for_exponential

    alpha = theta2 + 1
    beta = theta1

    exponential = gamma_family(theta=np.array([theta1, theta2]), parametrization_name="theta")
    var = exponential.computation_strategy.query_method(CharacteristicName.VAR, distr=exponential)
    assert np.isclose(var(), alpha / beta**2, rtol=1e-6)
