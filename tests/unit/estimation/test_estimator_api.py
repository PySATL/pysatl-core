"""
Estimation methods as objects.

The claim this file exists to check is the extensibility one: a method defined
*outside* this package, implementing nothing but the protocol and reusing the
shared steps, is accepted by ``ParametricFamily.fit`` and its own result type
reaches the caller unchanged — with no edit to the family, to ``fit``, or to
the estimation package.  ``MomentMatching`` below is that outside method,
deliberately written here rather than shipped in ``src``.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pysatl_core.estimation import (
    MLE,
    Estimator,
    FitResult,
    MLEResult,
    check_fixed_support,
    convert_parametrization,
    probe_params,
    support_depends_on_params,
    validate_sample,
)
from pysatl_core.estimation.parameters.vectors import field_names
from pysatl_core.families.parametrizations import Parametrization

if TYPE_CHECKING:
    import numpy.typing as npt

    from pysatl_core.families.parametric_family import ParametricFamily
    from pysatl_core.types import EstimatorName, ParametrizationName


@dataclass(frozen=True, slots=True, kw_only=True)
class MomentResult[P: Parametrization](FitResult[P]):
    """What a moment estimator reports beyond the common part."""

    matched_moments: int


@dataclass(frozen=True, slots=True)
class MomentMatching:
    """A method of moments, defined entirely outside ``pysatl_core.estimation``.

    It reuses the shared steps and reports a result of its own shape, with no
    ``log_likelihood`` — which is the point: a field named for a likelihood has
    no meaning here.
    """

    matched_moments: int = 2

    @property
    def name(self) -> EstimatorName:
        return "mm"

    def estimate(
        self,
        family: ParametricFamily,
        sample: npt.ArrayLike,
        *,
        parametrization: ParametrizationName | None = None,
    ) -> MomentResult[Parametrization]:
        data = validate_sample(family, sample)
        params = probe_params(family, data)
        if not support_depends_on_params(family, params):
            check_fixed_support(family, data, params)
        return MomentResult(
            family_name=family.name,
            params=convert_parametrization(family, params, parametrization),
            n_params=len(field_names(family.base)),
            n_observations=int(data.size),
            estimator=self.name,
            success=True,
            message="",
            matched_moments=self.matched_moments,
        )


class TestTheProtocol:
    def test_mle_is_an_estimator(self):
        assert isinstance(MLE(), Estimator)

    def test_a_method_defined_elsewhere_is_one_too(self):
        assert isinstance(MomentMatching(), Estimator)

    def test_an_object_without_estimate_is_not(self):
        assert not isinstance("mle", Estimator)

    def test_an_estimator_is_a_value(self):
        # Configuration is separate from data: built once, compared by value,
        # applied to as many samples as the caller likes.
        assert MLE(optimizer="Powell") == MLE(optimizer="Powell")
        assert MLE(optimizer="Powell") != MLE()
        with pytest.raises(dataclasses.FrozenInstanceError):
            # mypy rejects this line too, which is half the guarantee; the
            # other half is that it also fails at runtime, for a caller who
            # shares one estimator across several fits.
            MLE().optimizer = "Powell"  # type: ignore[misc]


class TestTheDefaultEstimator:
    def test_fit_without_an_estimator_is_maximum_likelihood(self, normal_family, rng):
        result = normal_family.fit(rng.normal(0.0, 1.0, 100))
        assert isinstance(result, MLEResult)
        assert result.estimator == "mle"

    def test_the_configuration_reaches_scipy(self, gamma_family, fixed_rng):
        # The budget is asserted on, not just the estimate: a run that ignored
        # 'maxiter' would land on the same parameters anyway, so only the
        # iteration count shows whether the option reached SciPy at all.
        sample = fixed_rng.gamma(2.0, 1.0, 200)
        budgeted = gamma_family.fit(
            sample, estimator=MLE(optimizer="Nelder-Mead", options={"options": {"maxiter": 1}})
        )
        unbudgeted = gamma_family.fit(sample, estimator=MLE(optimizer="Nelder-Mead"))
        assert budgeted.n_iterations == 1
        assert unbudgeted.n_iterations is not None
        assert unbudgeted.n_iterations > 1

    def test_estimate_is_a_complete_entry_point(self, normal_family):
        # A raw list, not a validated array: an estimator validates its own
        # input, so 'estimate' is usable without going through 'fit'.
        result = MLE().estimate(normal_family, [1.0, 2.0, 3.0, 4.0])
        assert result.params.parameters["mu"] == pytest.approx(2.5)

    def test_one_estimator_serves_many_samples(self, normal_family, fixed_rng):
        estimator = MLE()
        means = [
            estimator.estimate(normal_family, fixed_rng.normal(m, 1.0, 400)).params.parameters["mu"]
            for m in (0.0, 5.0)
        ]
        assert means[0] == pytest.approx(0.0, abs=0.2)
        assert means[1] == pytest.approx(5.0, abs=0.2)


class TestAMethodDefinedOutside:
    def test_it_reaches_fit_unchanged(self, normal_family, rng):
        sample = rng.normal(4.0, 2.0, 500)
        result = normal_family.fit(sample, estimator=MomentMatching())
        assert result.estimator == "mm"
        assert result.params.parameters["mu"] == pytest.approx(float(sample.mean()))

    def test_its_own_fields_survive_the_round_trip(self, normal_family, rng):
        result = normal_family.fit(rng.normal(0.0, 1.0, 50), estimator=MomentMatching(3))
        assert result.matched_moments == 3
        assert not hasattr(result, "log_likelihood")

    def test_it_inherits_the_common_part(self, normal_family, rng):
        result = normal_family.fit(rng.normal(0.0, 1.0, 50), estimator=MomentMatching())
        assert isinstance(result, FitResult)
        assert result.n_params == 2
        assert result.n_observations == 50
        assert result.distribution is not None

    def test_it_gets_the_shared_validation(self, normal_family):
        with pytest.raises(ValueError, match="one-dimensional"):
            normal_family.fit(np.zeros((4, 4)), estimator=MomentMatching())

    def test_a_view_works_through_it_too(self, normal_family, rng):
        view = normal_family.view(mu=0.0)
        result = view.fit(rng.normal(0.0, 2.0, 200), estimator=MomentMatching())
        assert result.n_params == 1
        assert "mu" not in result.params.parameters


class TestWhatFitDoesNotAccept:
    def test_no_method_specific_keyword_is_taken(self, normal_family, rng):
        # 'optimizer' belongs to one method and means nothing to the others,
        # so it lives on the estimator rather than in this signature. Leaving
        # it here would have forced every future method either to ignore it or
        # to reject it by hand.
        with pytest.raises(TypeError, match="optimizer"):
            normal_family.fit(rng.normal(0.0, 1.0, 50), optimizer="Powell")

    def test_something_that_is_not_an_estimator_is_refused(self, normal_family, rng):
        with pytest.raises(TypeError, match="expected an estimator"):
            normal_family.fit(rng.normal(0.0, 1.0, 50), estimator="mle")


class TestTheRouteField:
    def test_route_names_the_branch(self, normal_family, uniform_family, rng):
        assert normal_family.fit(rng.normal(0.0, 1.0, 50)).route == "closed_form"
        with pytest.warns(UserWarning, match="closed-form MLE"):
            numeric = uniform_family.fit(
                rng.uniform(0.0, 1.0, 50), estimator=MLE(optimizer="Powell")
            )
        assert numeric.route == "numeric"

    def test_the_estimator_name_is_not_the_route(self, normal_family, rng):
        # The two were one field, called 'method', while there was only one
        # method to name.
        result = normal_family.fit(rng.normal(0.0, 1.0, 50))
        assert (result.estimator, result.route) == ("mle", "closed_form")
