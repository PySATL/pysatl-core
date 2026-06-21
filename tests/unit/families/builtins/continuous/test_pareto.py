"""
Tests for Pareto distribution family.
"""

__author__ = "Vinogradov Ilya"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import pareto

from pysatl_core.distributions.support import ContinuousSupport
from pysatl_core.families.configuration import configure_families_register
from pysatl_core.families.exponential_family import (
    ContinuousExponentialClassFamily,
    ExponentialConjugateHyperparameters,
)
from pysatl_core.types import (
    CharacteristicName,
    ContinuousSupportShape1D,
    FamilyName,
    UnivariateContinuous,
)

from .base import BaseDistributionTest


class TestParetoFamily(BaseDistributionTest):
    """Test suite for Pareto distribution family."""

    def setup_method(self) -> None:
        registry = configure_families_register()
        self.pareto_family = cast(ContinuousExponentialClassFamily, registry.get(FamilyName.PARETO))
        self.pareto_dist_example = self.pareto_family(parametrization_name="shape", alpha=3.0)

    def test_family_properties(self) -> None:
        assert self.pareto_family.name == FamilyName.PARETO
        assert isinstance(self.pareto_family, ContinuousExponentialClassFamily)
        assert self.pareto_family.distribution(
            parametrization_name="shape", alpha=3.0
        ).family_name == (FamilyName.PARETO)
        assert set(self.pareto_family.parametrization_names) == {"theta", "shape"}
        assert self.pareto_family.base_parametrization_name == "theta"

    def test_shape_parametrization_creation(self) -> None:
        dist = self.pareto_family(parametrization_name="shape", alpha=3.0)

        assert dist.distribution_type == UnivariateContinuous
        assert dist.parameters == {"alpha": 3.0}
        assert dist.parametrization_name == "shape"

    def test_parametrization_constraints(self) -> None:
        with pytest.raises(ValueError, match="alpha > 0"):
            self.pareto_family(parametrization_name="shape", alpha=0.0)

    def test_parametrization_conversion_to_theta(self) -> None:
        shape_params = cast(Any, self.pareto_family.get_parametrization("shape"))
        base_params = cast(
            Any,
            self.pareto_family.to_base(shape_params(alpha=3.0)),
        )

        assert_allclose(base_params.theta, np.array([-4.0]))

    def test_analytical_computations_availability(self) -> None:
        comp = self.pareto_dist_example.analytical_computations

        expected_chars = {
            CharacteristicName.PDF,
            CharacteristicName.MEAN,
            CharacteristicName.VAR,
        }
        assert set(comp.keys()) == expected_chars

    @pytest.mark.parametrize(
        "char_name, test_data, scipy_func",
        [
            (CharacteristicName.PDF, [0.5, 1.0, 1.5, 2.0, 4.0], pareto.pdf),
        ],
    )
    def test_main_characteristics_against_scipy(
        self,
        char_name: CharacteristicName,
        test_data: list[float],
        scipy_func: Any,
    ) -> None:
        char_func = self.pareto_dist_example.query_method(char_name)
        input_array = np.array(test_data)
        if char_name == CharacteristicName.PDF:
            actual = np.asarray([char_func(x) for x in input_array], dtype=float)
        else:
            actual = char_func(input_array)
        expected = scipy_func(input_array, 3.0, scale=1.0)

        assert actual.shape == input_array.shape
        self.assert_arrays_almost_equal(actual, expected, precision=1e-8)

    def test_log_pdf_matches_log_of_pdf(self) -> None:
        pdf = self.pareto_dist_example.query_method(CharacteristicName.PDF)
        x_values = np.array([1.0, 1.5, 2.0, 4.0])

        actual = np.log(np.asarray([pdf(x) for x in x_values], dtype=float))
        expected = pareto.logpdf(x_values, 3.0, scale=1.0)

        self.assert_arrays_almost_equal(actual, expected, precision=1e-8)

    def test_moments(self) -> None:
        mean_func = self.pareto_dist_example.query_method(CharacteristicName.MEAN)
        var_func = self.pareto_dist_example.query_method(CharacteristicName.VAR)

        scipy_pareto = pareto(b=3.0, scale=1.0)

        assert mean_func() == pytest.approx(float(scipy_pareto.mean()), rel=1e-6)
        assert var_func() == pytest.approx(float(scipy_pareto.var()), rel=1e-6)

    def test_support(self) -> None:
        dist = self.pareto_dist_example

        assert dist.support is not None
        assert isinstance(dist.support, ContinuousSupport)
        assert dist.support.left == 1.0
        assert dist.support.right == float("inf")
        assert dist.support.left_closed
        assert not dist.support.right_closed
        assert dist.support.shape == ContinuousSupportShape1D.RAY_RIGHT

        assert dist.support.contains(np.asarray(1.0)) is True
        assert dist.support.contains(np.asarray(2.0)) is True
        assert dist.support.contains(np.asarray(0.99)) is False

    def test_posterior_hyperparameters(self) -> None:
        predictive_family = self.pareto_family.posterior_predictive
        posterior_params_cls = cast(Any, predictive_family.get_parametrization("posterior"))
        prior = cast(
            ExponentialConjugateHyperparameters,
            posterior_params_cls(
                effective_suff_stat_value=np.array([2.0]),
                effective_sample_size=3.0,
            ),
        )
        posterior = self.pareto_family.posterior_hyperparameters(prior, sample=[2.0, 4.0])

        assert_allclose(
            posterior.effective_suff_stat_value,
            np.array([2.0 + np.log(2.0) + np.log(4.0)]),
        )
        assert posterior.effective_sample_size == 5.0

    def test_posterior_predictive_pdf(self) -> None:
        xi = 2.0
        nu = 3.0
        predictive = self.pareto_family.posterior_predictive(
            parametrization_name="posterior",
            effective_suff_stat_value=np.array([xi]),
            effective_sample_size=nu,
        )
        pdf = predictive.query_method(CharacteristicName.PDF)
        x_values = np.array([1.0, 1.5, 2.0, 4.0])

        actual = np.asarray([pdf(x) for x in x_values], dtype=float)
        expected = (
            (nu + 1.0) * xi ** (nu + 1.0) / (x_values * (xi + np.log(x_values)) ** (nu + 2.0))
        )
        self.assert_arrays_almost_equal(actual, expected, precision=1e-6)
