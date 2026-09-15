"""
Parameter bounds: assembly, ordering, and agreement with ``@constraint``.

The last group is the important one.  ``param_bounds`` and ``@constraint``
record the same condition twice — deliberately, since neither mechanism can do
the other's job — and duplicated knowledge drifts apart unless something holds
it together.  These tests are that something.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from pysatl_core.estimation import clip_to_bounds, resolve_bounds
from pysatl_core.families.parametric_family import ParametricFamily
from pysatl_core.families.parametrizations import (
    Parametrization,
    constraint,
    parametrization,
)
from pysatl_core.types import UnivariateContinuous
from tests.unit.estimation.conftest import declared_bounds


class TestResolveBounds:
    def test_follows_the_field_order_of_the_base_parametrization(self, normal_family):
        assert list(normal_family.base.__dataclass_fields__) == ["mu", "sigma"]
        bounds = declared_bounds(normal_family)
        assert len(bounds) == 2
        assert bounds[0] == (-np.inf, np.inf), "mu has no declared bound"
        assert bounds[1][1] == np.inf

    def test_undeclared_parameters_are_unbounded(self, normal_family):
        assert declared_bounds(normal_family)[0] == (-np.inf, np.inf)

    def test_an_open_lower_bound_is_nudged_off_the_edge(self, normal_family):
        low, _high = declared_bounds(normal_family)[1]
        assert low > 0.0, "sigma > 0 is open; the optimizer must not be allowed to sit at 0"
        assert low == np.nextafter(0.0, np.inf)

    def test_every_gamma_parameter_is_bounded_below(self, gamma_family):
        bounds = declared_bounds(gamma_family)
        assert [b[0] > 0.0 for b in bounds] == [True, True]
        assert [b[1] for b in bounds] == [np.inf, np.inf]

    def test_a_view_keeps_only_free_parameters_in_declaration_order(self, gamma_family):
        view = gamma_family.view(k=2.0)
        assert view.free_parameter_names == ("theta",)
        bounds = declared_bounds(view)
        assert len(bounds) == 1
        assert bounds[0][0] > 0.0

    def test_a_normal_view_drops_the_bound_of_the_fixed_parameter(self, normal_family):
        assert len(declared_bounds(normal_family.view(mu=0.0))) == 1

    def test_a_family_without_bounds_warns_and_runs_unbounded(self, uniform_family):
        with pytest.warns(UserWarning, match="declares no 'param_bounds'"):
            assert resolve_bounds(uniform_family) is None

    def test_the_uniform_family_declares_none_on_purpose(self, uniform_family):
        # Its only restriction couples two parameters, and a box cannot express
        # a relation between them.
        assert uniform_family.param_bounds == {}


class TestClipToBounds:
    def test_moves_a_point_into_the_box(self, normal_family):
        clipped = clip_to_bounds(normal_family, normal_family.base(mu=1.0, sigma=-5.0))
        assert clipped.parameters["mu"] == 1.0
        assert clipped.parameters["sigma"] > 0.0

    def test_leaves_an_unbounded_family_alone(self, uniform_family):
        params = uniform_family.base(lower_bound=-10.0, upper_bound=10.0)
        assert clip_to_bounds(uniform_family, params) is params


class TestDeclarationValidation:
    def test_an_unknown_parameter_name_is_rejected(self):
        family = ParametricFamily(
            name="UnknownBoundName",
            distr_type=UnivariateContinuous,
            distr_parametrizations=["p"],
            distr_characteristics={},
            param_bounds={"sigmaa": (0, None)},
        )
        with pytest.raises(ValueError, match="unknown parameter"):

            @parametrization(family=family, name="p")
            class _P(Parametrization):
                sigma: float

    @pytest.mark.parametrize(
        "bad, message",
        [
            ({"a": (1.0, 0.0)}, "is empty"),
            ({"a": (0.0,)}, "must be a"),
            ({"a": "positive"}, "must be a"),
            ({"a": ("x", None)}, "non-numeric"),
        ],
    )
    def test_a_malformed_entry_is_rejected(self, bad, message):
        with pytest.raises(ValueError, match=message):
            ParametricFamily(
                name="MalformedBounds",
                distr_type=UnivariateContinuous,
                distr_parametrizations=["p"],
                distr_characteristics={},
                param_bounds=bad,
            )


class TestBoundsAgreeWithConstraints:
    """A value just inside a declared bound must validate; just outside must not.

    Read this class as describing *today's* behaviour, not a requirement.  It
    asserts that a value sitting exactly *on* a declared edge fails
    ``validate()``, which is true only because ``param_bounds`` has no way to
    say that a bound is closed — see ``docs/estimation_todos.md`` #4.
    Closed bounds are ordinary (SciPy declares ``c >= 0`` for ``foldnorm``,
    ``n >= 1`` for ``erlang``), so when the declaration gains that
    expressiveness this class has to change with it: a family declaring a
    closed edge must then accept the endpoint, and these assertions have to
    apply only to the families that declare an open one.
    """

    @pytest.mark.parametrize(
        "family_fixture, valid_params",
        [
            ("normal_family", {"mu": 0.0, "sigma": 1.0}),
            ("gamma_family", {"k": 2.0, "theta": 3.0}),
            ("exponential_family", {"lambda_": 1.5}),
        ],
    )
    def test_each_declared_edge_is_the_edge_the_constraints_enforce(
        self, request, family_fixture, valid_params
    ):
        family = request.getfixturevalue(family_fixture)
        assert family.param_bounds, "this family is expected to declare bounds"

        for name, (low, high) in family.param_bounds.items():
            if low is not None:
                inside = dict(valid_params, **{name: np.nextafter(low, np.inf)})
                outside = dict(valid_params, **{name: low})
                family.base(**inside).validate()
                with pytest.raises(ValueError):
                    family.base(**outside).validate()
            if high is not None:
                inside = dict(valid_params, **{name: np.nextafter(high, -np.inf)})
                outside = dict(valid_params, **{name: high})
                family.base(**inside).validate()
                with pytest.raises(ValueError):
                    family.base(**outside).validate()

    def test_the_resolved_box_never_contains_an_inadmissible_edge(self, normal_family):
        bounds = declared_bounds(normal_family)
        edges = normal_family.base(
            mu=bounds[0][0] if np.isfinite(bounds[0][0]) else 0.0, sigma=bounds[1][0]
        )
        edges.validate()


class TestPropagationToViews:
    def test_a_view_inherits_the_closed_form_and_the_bounds(self, normal_family):
        view = normal_family.view(mu=0.0)
        assert view.mle is normal_family.mle
        assert view.param_bounds == {"sigma": (0.0, None)}

    def test_a_view_fixed_in_another_parametrization_drops_the_bounds(self, normal_family):
        # The declared names live in the base parametrization's coordinates and
        # say nothing about 'var'.
        view = normal_family.view(parametrization_name="meanVar", mu=0.0)
        assert view.param_bounds == {}
        with pytest.warns(UserWarning, match="declares no 'param_bounds'"):
            assert resolve_bounds(view) is None


class TestDeclarationAccessors:
    """``mle`` and ``base_score`` expose what the family declared.

    Both answer a question the rest of the public surface cannot: ``score``
    evaluates the gradient but raises when there is none, so a caller has no
    way to ask whether one exists without provoking the error.
    """

    def test_a_family_with_a_closed_form_exposes_it(self, normal_family):
        assert normal_family.mle is not None
        assert callable(normal_family.mle)

    def test_a_family_without_a_closed_form_reports_none(self, gamma_family):
        assert gamma_family.mle is None

    def test_base_score_is_exposed_when_declared(self, normal_family):
        assert normal_family.base_score is not None

    def test_base_score_is_none_when_absent(self, scoreless_family):
        assert scoreless_family.base_score is None
        with pytest.raises(ValueError, match="does not provide score"):
            scoreless_family.score(scoreless_family.base(mu=0.0, sigma=1.0), np.array([0.0]))

    def test_a_view_inherits_both(self, normal_family):
        view = normal_family.view(mu=0.0)
        assert view.mle is normal_family.mle
        assert view.base_score is normal_family.base_score


class TestConstraintFamilyBounds:
    def test_a_bound_declared_next_to_the_family_reaches_the_optimizer(self):
        family = ParametricFamily(
            name="BoundedProbe",
            distr_type=UnivariateContinuous,
            distr_parametrizations=["p"],
            distr_characteristics={},
            param_bounds={"a": (-2.0, 5.0)},
        )

        @parametrization(family=family, name="p")
        class _P(Parametrization):
            a: float
            b: float

            @constraint(description="-2 < a < 5")
            def check_a(self) -> bool:
                return -2.0 < self.a < 5.0

        low, high = declared_bounds(family)[0]
        assert -2.0 < low < high < 5.0, (
            "the declared edges must be nudged strictly inside the admissible "
            "interval, since '-2 < a < 5' is open at both ends"
        )
        assert declared_bounds(family)[1] == (-np.inf, np.inf), (
            "'b' has no entry in param_bounds, so it must come back unbounded"
        )
