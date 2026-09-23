"""
The declared shapes that replaced attribute probing, casts and ``Any``.

Each class here pins behaviour that used to rest on a string literal or on a
type the checker could not see through: the support fingerprint, the normalised
optimizer result, the fixed-parameter pair, and the guards that turn a
malformed family or a malformed solver result into a message instead of a
``TypeError`` thrown from the middle of an optimisation loop.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult

from pysatl_core.distributions.support import (
    ContinuousSupport,
    ExplicitTableDiscreteSupport,
    IntegerLatticeDiscreteSupport,
    IntervalSupport,
    PointSupport,
)
from pysatl_core.estimation import (
    MLE,
    EstimationError,
    FitProblem,
    MLEResult,
)
from pysatl_core.estimation.methods.mle import LogLikelihood
from pysatl_core.estimation.optimizers import ObjectiveFunc, OptimizerOutcome, optimizer_name
from pysatl_core.estimation.parameters.vectors import field_names
from pysatl_core.estimation.problem import (
    FixedParameters,
    IntervalSignature,
    OpaqueSignature,
    PointsSignature,
    fixed_parameters,
    support_signature,
)
from pysatl_core.families.parametric_family import ParametricFamily
from pysatl_core.families.parametrizations import Parametrization, parametrization
from pysatl_core.types import CharacteristicName, UnivariateContinuous


class TestSupportShapeProtocols:
    """``Support`` declares only ``contains``; the shape protocols declare the rest."""

    def test_an_interval_support_is_recognised_by_its_endpoints(self):
        support = ContinuousSupport(left=0.0, right=1.0)
        assert isinstance(support, IntervalSupport)
        assert not isinstance(support, PointSupport)

    def test_a_point_support_is_recognised_by_its_points(self):
        support = ExplicitTableDiscreteSupport([1.0, 2.0, 3.0])
        assert isinstance(support, PointSupport)
        assert not isinstance(support, IntervalSupport)

    def test_a_lattice_support_matches_neither(self):
        support = IntegerLatticeDiscreteSupport(residue=0, modulus=1)
        assert not isinstance(support, IntervalSupport)
        assert not isinstance(support, PointSupport)


class TestSupportSignature:
    def test_no_support_is_its_own_state(self):
        assert support_signature(None) is None

    def test_an_interval_becomes_an_interval_signature(self):
        signature = support_signature(ContinuousSupport(left=-1.0, right=2.0, right_closed=False))
        assert signature == IntervalSignature(
            left=-1.0, right=2.0, left_closed=True, right_closed=False
        )

    def test_equality_is_structural_not_textual(self):
        # Two separately built supports with the same endpoints must compare
        # equal without either object's __repr__ taking part.
        first = support_signature(ContinuousSupport(left=0.0, right=1.0))
        second = support_signature(ContinuousSupport(left=0.0, right=1.0))
        assert first == second
        assert first != support_signature(ContinuousSupport(left=0.0, right=1.5))

    def test_a_closure_flag_alone_distinguishes_two_supports(self):
        # The branch that used to read 'left_closed' through getattr with a
        # default of True: a misspelling made both supports look closed.
        closed = support_signature(ContinuousSupport(left=0.0, right=1.0, left_closed=True))
        half_open = support_signature(ContinuousSupport(left=0.0, right=1.0, left_closed=False))
        assert closed != half_open

    def test_a_discrete_support_becomes_a_points_signature(self):
        signature = support_signature(ExplicitTableDiscreteSupport([3.0, 1.0, 2.0]))
        assert signature == PointsSignature(points=(1.0, 2.0, 3.0))

    def test_an_unrecognised_support_falls_back_to_an_opaque_signature(self):
        support = IntegerLatticeDiscreteSupport(residue=0, modulus=2, min_k=0, max_k=10)
        signature = support_signature(support)
        assert isinstance(signature, OpaqueSignature)
        assert signature.type_name == "IntegerLatticeDiscreteSupport"
        assert signature.representation == repr(support)

    def test_two_unrecognised_supports_are_still_told_apart(self):
        first = support_signature(IntegerLatticeDiscreteSupport(residue=0, modulus=2))
        second = support_signature(IntegerLatticeDiscreteSupport(residue=1, modulus=2))
        assert first != second


class TestNoArrayEquality:
    """A frozen dataclass holding an array compares field tuples, and a tuple
    comparison of two distinct-but-equal arrays yields an array whose truth
    value raises. Every such class here declares ``eq=False`` instead.
    """

    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(
                lambda fam, x: LogLikelihood.of(fam, x),
                id="LogLikelihood",
            ),
            pytest.param(
                lambda fam, x: OptimizerOutcome(
                    x=x,
                    optimizer="stub",
                    success=True,
                    message="",
                    n_iterations=1,
                    n_function_evaluations=1,
                ),
                id="OptimizerOutcome",
            ),
            pytest.param(lambda fam, x: FitProblem.prepare(fam, x), id="FitProblem"),
        ],
    )
    def test_two_objects_over_equal_arrays_compare_without_raising(self, normal_family, build):
        sample = np.array([1.0, 2.0, 3.0, 4.0])
        first = build(normal_family, sample)
        second = build(normal_family, sample.copy())
        assert first != second
        assert first == first


class TestOptimizerOutcome:
    """The single point of contact with ``scipy.optimize.OptimizeResult``."""

    def test_reads_every_field_a_built_in_method_reports(self):
        result = OptimizeResult()
        result["x"] = np.array([1.0, 2.0])
        result["success"] = True
        result["message"] = "converged"
        result["nit"] = 7
        result["nfev"] = 19
        outcome = OptimizerOutcome.from_scipy(result, "L-BFGS-B")
        assert outcome.success is True
        assert outcome.message == "converged"
        assert outcome.n_iterations == 7
        assert outcome.n_function_evaluations == 19
        np.testing.assert_array_equal(outcome.x, np.array([1.0, 2.0]))

    def test_an_unreported_verdict_stays_a_third_state(self):
        result = OptimizeResult()
        result["x"] = np.array([1.0])
        outcome = OptimizerOutcome.from_scipy(result, "custom")
        assert outcome.success is None
        assert outcome.message == ""
        assert outcome.n_iterations is None
        assert outcome.n_function_evaluations is None

    def test_a_missing_estimate_is_refused_by_name(self):
        with pytest.raises(EstimationError, match="returned no 'x' field"):
            OptimizerOutcome.from_scipy(OptimizeResult(), "custom")

    def test_an_estimate_that_is_not_an_array_is_refused(self):
        result = OptimizeResult()
        result["x"] = "not an estimate"
        with pytest.raises(EstimationError, match="which is not an array of parameter values"):
            OptimizerOutcome.from_scipy(result, "custom")

    def test_an_uninterpretable_counter_reads_as_not_reported(self):
        result = OptimizeResult()
        result["x"] = np.array([1.0])
        result["nit"] = "several"
        assert OptimizerOutcome.from_scipy(result, "custom").n_iterations is None

    def test_a_numpy_integer_counter_is_accepted(self):
        result = OptimizeResult()
        result["x"] = np.array([1.0])
        result["nfev"] = np.int64(12)
        assert OptimizerOutcome.from_scipy(result, "custom").n_function_evaluations == 12


class TestOptimizerName:
    def test_a_method_name_is_itself(self):
        assert optimizer_name("Nelder-Mead") == "Nelder-Mead"

    def test_a_function_is_named_by_its_own_name(self):
        def my_solver(
            fun: ObjectiveFunc, x0: NDArray[np.float64], **kwargs: object
        ) -> OptimizeResult:  # pragma: no cover - never called
            raise AssertionError

        assert optimizer_name(my_solver) == "my_solver"

    def test_a_callable_without_a_name_falls_back_to_its_repr(self):
        class Solver:
            def __call__(
                self, fun: ObjectiveFunc, x0: NDArray[np.float64], **kwargs: object
            ) -> OptimizeResult:  # pragma: no cover - never called
                raise AssertionError

        solver = Solver()
        assert optimizer_name(solver) == repr(solver)


class TestFixedParameters:
    def test_a_plain_family_pins_nothing_in_its_own_coordinates(self, normal_family):
        fixed = fixed_parameters(normal_family)
        assert fixed == FixedParameters(values={}, in_base_parametrization=True)

    def test_a_view_reports_what_it_pinned(self, normal_family):
        fixed = fixed_parameters(normal_family.view(mu=0.0))
        assert dict(fixed.values) == {"mu": 0.0}
        assert fixed.in_base_parametrization is True

    def test_a_view_in_foreign_coordinates_says_so(self, normal_family):
        # The closed-form rule is written against the base parametrization, so
        # values fixed in another one cannot be handed to it. This flag is what
        # sends such a fit down the numerical path.
        fixed = fixed_parameters(normal_family.view(parametrization_name="meanVar", mu=0.0))
        assert fixed.in_base_parametrization is False

    def test_such_a_view_is_fitted_numerically(self, normal_family, rng):
        view = normal_family.view(parametrization_name="meanVar", mu=0.0)
        result = view.fit(rng.normal(0.0, 2.0, 200))
        assert result.route == "numeric"


class TestResultValidation:
    def test_a_sample_size_of_zero_is_refused_where_it_is_written(self, normal_family):
        with pytest.raises(ValueError, match="n_observations must be positive"):
            MLEResult(
                family_name="Normal",
                params=normal_family.base(mu=0.0, sigma=1.0),
                log_likelihood=-1.0,
                n_params=2,
                n_observations=0,
                estimator="mle",
                route="numeric",
                optimizer=None,
                success=True,
                message="",
            )

    def test_a_negative_parameter_count_is_refused(self, normal_family):
        with pytest.raises(ValueError, match="n_params must not be negative"):
            MLEResult(
                family_name="Normal",
                params=normal_family.base(mu=0.0, sigma=1.0),
                log_likelihood=-1.0,
                n_params=-1,
                n_observations=10,
                estimator="mle",
                route="numeric",
                optimizer=None,
                success=True,
                message="",
            )

    def test_the_distribution_comes_from_the_parametrization_not_the_label(
        self, normal_family, rng
    ):
        result = normal_family.fit(rng.normal(5.0, 1.0, 200))
        assert result.distribution.parametrization == result.params


class TestFieldNames:
    def test_reads_the_declaration_order(self, normal_family):
        assert field_names(normal_family.base) == ("mu", "sigma")

    def test_a_view_exposes_only_its_free_parameters(self, normal_family):
        assert field_names(normal_family.view(mu=0.0).base) == ("sigma",)

    def test_a_non_parametrization_is_refused_rather_than_read_as_empty(self):
        with pytest.raises(TypeError, match="dataclass"):
            field_names(object())  # type: ignore[arg-type]


class TestLpdfProviderShape:
    def test_a_nullary_lpdf_provider_is_refused_with_an_explanation(self):
        """``distr_characteristics`` admits three calling shapes; ``lpdf`` needs one.

        Assuming the right one used to be a bare ``cast``: a family declaring
        ``lpdf`` without evaluation points reached a ``TypeError`` thousands of
        iterations into the search, far from the declaration that caused it.
        """
        family = ParametricFamily(
            name="NullaryLpdf",
            distr_type=UnivariateContinuous,
            distr_parametrizations=["p"],
            distr_characteristics={CharacteristicName.LPDF: lambda: 0.0},
        )

        @parametrization(family=family, name="p")
        class _P(Parametrization):
            a: float

        with pytest.raises(EstimationError, match="takes no evaluation points"):
            LogLikelihood.of(family, np.array([1.0, 2.0]))

    def test_a_two_argument_provider_is_accepted(self):
        family = ParametricFamily(
            name="PointwiseLpdf",
            distr_type=UnivariateContinuous,
            distr_parametrizations=["p"],
            distr_characteristics={
                CharacteristicName.LPDF: lambda params, x: np.zeros_like(x),
            },
        )

        @parametrization(family=family, name="p")
        class _P(Parametrization):
            a: float

        objective = LogLikelihood.of(family, np.array([1.0, 2.0]))
        assert objective(np.array([1.0])) == 0.0

    def test_a_view_wrapped_provider_is_still_accepted(self, normal_family, rng):
        """A view wraps its providers as ``(params, *args, **kwargs)``.

        The arity check has to count that as the pointwise shape, or every
        fit of a view would be refused.
        """
        result = normal_family.view(mu=0.0).fit(rng.normal(0.0, 2.0, 100))
        assert np.isfinite(result.log_likelihood)


def _identity_solver(
    fun: ObjectiveFunc, x0: NDArray[np.float64], **kwargs: object
) -> OptimizeResult:
    """A solver matching :class:`MinimizeSolver`: it returns where it started."""
    outcome = OptimizeResult()
    outcome["x"] = np.asarray(x0)
    outcome["success"] = True
    return outcome


class TestMinimizeSolverProtocol:
    def test_a_conforming_solver_is_driven_end_to_end(self, gamma_family, rng):
        sample = rng.gamma(3.0, 2.0, 200)
        result = gamma_family.fit(sample, estimator=MLE(optimizer=_identity_solver))
        assert result.optimizer == "_identity_solver"
        assert result.success is True
        assert result.route == "numeric"

    def test_options_still_reach_the_solver(self, gamma_family, rng):
        seen: list[object] = []

        def recording_solver(
            fun: ObjectiveFunc, x0: NDArray[np.float64], **kwargs: object
        ) -> OptimizeResult:
            seen.append(kwargs.get("tol"))
            return _identity_solver(fun, x0, **kwargs)

        gamma_family.fit(
            rng.gamma(3.0, 2.0, 100),
            estimator=MLE(optimizer=recording_solver, options={"tol": 1e-9}),
        )
        assert seen == [1e-9]
