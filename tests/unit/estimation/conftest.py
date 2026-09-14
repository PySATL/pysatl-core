"""
Shared fixtures for the estimation tests.

Besides the four built-in families, this module builds a few deliberately
incomplete families.  They exercise code paths that no built-in family can
reach: a family without a closed-form solution but with a support that moves
with its parameters (the only way to observe the optimizer fallback, since
every built-in family in that situation has a formula), a family without
``score``, and a family without ``lpdf``.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any, cast

import numpy as np
import pytest

from pysatl_core.families.configuration import configure_families_register
from pysatl_core.families.parametric_family import ParametricFamily
from pysatl_core.families.parametrizations import (
    Parametrization,
    constraint,
    parametrization,
)
from pysatl_core.types import (
    CharacteristicName,
    FamilyName,
    NumericArray,
    UnivariateContinuous,
)

SEED = 20250901

SEEDS = (20250901, 7, 42, 777, 2024, 31337)
"""Seeds every sample-driven test is repeated on.

A single fixed seed lets a tolerance quietly settle around one realisation:
three of the assertions in this suite were tuned to ``20250901`` and failed on
other draws.  Running each test on several samples makes that visible the
moment it happens rather than months later, and costs a few seconds.
"""


def declared_bounds(family: ParametricFamily) -> list[tuple[float, float]]:
    """``resolve_bounds`` for a family that is expected to declare some.

    Keeps the ``None`` branch — which means "this family declares nothing" —
    out of tests that are about the values themselves.
    """
    from pysatl_core.estimation import resolve_bounds

    bounds = resolve_bounds(family)
    assert bounds is not None, f"family '{family.name}' was expected to declare bounds"
    return bounds


@pytest.fixture(params=SEEDS, ids=lambda seed: f"seed{seed}")
def rng(request: pytest.FixtureRequest) -> np.random.Generator:
    """A fresh generator per seed; every test using it runs once per seed."""
    return np.random.default_rng(request.param)


@pytest.fixture
def fixed_rng() -> np.random.Generator:
    """The single-seed generator, for tests whose subject is not the sample.

    Used where repeating the test across seeds would only repeat the same
    arithmetic — a warning being raised, a field being copied — and the sample
    is merely something to hand the fitter.
    """
    return np.random.default_rng(SEED)


@pytest.fixture
def normal_family() -> ParametricFamily:
    return configure_families_register().get(FamilyName.NORMAL)


@pytest.fixture
def uniform_family() -> ParametricFamily:
    return configure_families_register().get(FamilyName.CONTINUOUS_UNIFORM)


@pytest.fixture
def exponential_family() -> ParametricFamily:
    return configure_families_register().get(FamilyName.EXPONENTIAL)


@pytest.fixture
def gamma_family() -> ParametricFamily:
    return configure_families_register().get(FamilyName.GAMMA)


def _make_boxed_family(name: str) -> ParametricFamily:
    """A uniform-like family with no closed-form MLE and no declared bounds.

    Its support moves with the parameters, so the objective is a staircase in
    the number of unexplained observations — exactly the surface on which a
    quasi-Newton method stalls.  Every built-in family of that shape declares a
    closed form, which is why this one has to be built by hand.
    """
    from pysatl_core.distributions.support import ContinuousSupport

    def lpdf(parameters: Parametrization, x: NumericArray) -> NumericArray:
        params = cast(Any, parameters)
        a, b = params.low, params.high
        return np.where((x >= a) & (x <= b), -np.log(b - a), -np.inf)

    def base_score(parameters: Parametrization, x: NumericArray) -> NumericArray:
        params = cast(Any, parameters)
        a, b = params.low, params.high
        if np.any((x < a) | (x > b)):
            raise ValueError("score is undefined outside the support")
        width = b - a
        return np.stack([np.full_like(x, 1.0 / width), np.full_like(x, -1.0 / width)], axis=-1)

    def support(parameters: Parametrization) -> ContinuousSupport:
        params = cast(Any, parameters.transform_to_base_parametrization())
        return ContinuousSupport(left=params.low, right=params.high)

    family = ParametricFamily(
        name=name,
        distr_type=UnivariateContinuous,
        distr_parametrizations=["box"],
        distr_characteristics={CharacteristicName.LPDF: lpdf},
        support_by_parametrization=support,
        base_score=base_score,
    )

    @parametrization(family=family, name="box")
    class _Box(Parametrization):
        low: float
        high: float

        @constraint(description="low < high")
        def check_order(self) -> bool:
            return self.low < self.high

    return family


@pytest.fixture
def make_boxed_family():
    """Factory for fresh copies of the formula-less moving-support family."""
    return _make_boxed_family


@pytest.fixture
def boxed_family():
    """A formula-less moving-support family with a plain method-of-moments start.

    The rule is the textbook moment estimate ``mean +- sqrt(3) * std``, which
    for a uniform sample lands just *inside* the observed range and therefore
    leaves a few observations unexplained.  That is the interesting case: the
    objective at the start is a staircase in the number of violations, which is
    the surface a quasi-Newton method cannot descend.  Padding the start
    outwards instead — what the built-in uniform family does — would let
    L-BFGS-B move and never trigger the fallback.
    """
    from pysatl_core.estimation.moments import _MOMENT_STARTS, register_moment_start

    family = _make_boxed_family("BoxedNoFormula")

    def rule(sample: NumericArray) -> dict[str, float]:
        center = float(np.mean(sample))
        half_width = float(np.sqrt(3.0) * np.std(sample))
        return {"low": center - half_width, "high": center + half_width}

    register_moment_start(family.name, rule)
    try:
        yield family
    finally:
        _MOMENT_STARTS.pop(family.name, None)


@pytest.fixture
def scoreless_family() -> ParametricFamily:
    """A normal family stripped of its ``score``, forcing a numerical gradient."""

    def lpdf(parameters: Parametrization, x: NumericArray) -> NumericArray:
        params = cast(Any, parameters)
        return cast(
            NumericArray,
            -0.5 * np.log(2 * np.pi)
            - np.log(params.sigma)
            - ((x - params.mu) ** 2) / (2 * params.sigma**2),
        )

    family = ParametricFamily(
        name="ScorelessNormal",
        distr_type=UnivariateContinuous,
        distr_parametrizations=["ms"],
        distr_characteristics={CharacteristicName.LPDF: lpdf},
        param_bounds={"sigma": (0, None)},
    )

    @parametrization(family=family, name="ms")
    class _MS(Parametrization):
        mu: float
        sigma: float

        @constraint(description="sigma > 0")
        def check_sigma(self) -> bool:
            return self.sigma > 0

    return family


@pytest.fixture
def lpdfless_family() -> ParametricFamily:
    """A family declaring only ``pdf``, which maximum likelihood cannot use."""

    def pdf(parameters: Parametrization, x: NumericArray) -> NumericArray:
        params = cast(Any, parameters)
        return cast(NumericArray, params.rate * np.exp(-params.rate * x))

    family = ParametricFamily(
        name="NoLogDensity",
        distr_type=UnivariateContinuous,
        distr_parametrizations=["r"],
        distr_characteristics={CharacteristicName.PDF: pdf},
        param_bounds={"rate": (0, None)},
    )

    @parametrization(family=family, name="r")
    class _R(Parametrization):
        rate: float

        @constraint(description="rate > 0")
        def check_rate(self) -> bool:
            return self.rate > 0

    return family
