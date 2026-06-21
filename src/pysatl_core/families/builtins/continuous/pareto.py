"""
Pareto distribution family implementation.

This module provides a Pareto Type I distribution with fixed known minimum
``x_m = 1`` as a continuous exponential family.
"""

from __future__ import annotations

__author__ = "Vinogradov Ilya"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np

from pysatl_core.distributions.support import ContinuousSupport
from pysatl_core.families.exponential_family import (
    ContinuousExponentialClassFamily,
    ExponentialFamilyParametrization,
)
from pysatl_core.families.parametrizations import Parametrization, constraint, parametrization
from pysatl_core.families.registry import ParametricFamilyRegister
from pysatl_core.types import FamilyName, NumericArray, UnivariateContinuous

_MINIMUM = 1.0


def configure_pareto_family() -> None:
    """
    Configure and register the Pareto distribution family with fixed minimum 1.
    """

    if ParametricFamilyRegister.contains(FamilyName.PARETO):
        return

    PARETO_DOC = """
    Pareto Type I distribution with known minimum x_m = 1.

    The distribution is parameterized by the shape parameter ``alpha > 0``.

    Probability density function:
        f(x) = alpha / x^(alpha + 1),  x >= 1

    In natural form this is written as
        f(x | theta) = exp(theta log(x) + B(theta)),  theta < -1
    where ``theta = -(alpha + 1)``.

    With the more common convention
        f(x | theta) = exp(theta log(x) - A(theta)),
    we have
        A(theta) = -log(alpha * x_m^alpha)
    and therefore
        B(theta) = -A(theta) = log(alpha * x_m^alpha).
    """

    def _theta_to_alpha(theta: NumericArray) -> float:
        theta_arr = np.atleast_1d(np.asarray(theta, dtype=float))
        return float(-theta_arr[0] - 1.0)

    def log_partition(theta: NumericArray) -> NumericArray:
        alpha = _theta_to_alpha(theta)
        return np.array([np.log(alpha) + alpha * np.log(_MINIMUM)])

    def sufficient_statistics(x: NumericArray) -> NumericArray:
        x_arr = np.asarray(x, dtype=float)
        return np.array([np.log(x_arr).item()])

    def normalization_constant(_: NumericArray) -> float:
        return 1.0

    def _support(_: Parametrization) -> ContinuousSupport:
        return ContinuousSupport(left=_MINIMUM)

    pareto_family = ContinuousExponentialClassFamily(
        name=FamilyName.PARETO,
        log_partition=log_partition,
        sufficient_statistics=sufficient_statistics,
        normalization_constant=normalization_constant,
        support=ContinuousSupport(left=_MINIMUM),
        parameter_space=ContinuousSupport(right=-1.0, right_closed=False),
        sufficient_statistics_values=ContinuousSupport(left=np.log(_MINIMUM)),
        distr_type=UnivariateContinuous,
        distr_parametrizations=["theta", "shape"],
        support_by_parametrization=_support,
    )
    pareto_family.__doc__ = PARETO_DOC

    @parametrization(family=pareto_family, name="shape")
    class _Shape(Parametrization):
        """
        Shape parametrization of Pareto distribution.

        Parameters
        ----------
        alpha : float
            Shape parameter of the distribution.
        """

        alpha: float

        @constraint(description="alpha > 0")
        def check_alpha_positive(self) -> bool:
            return self.alpha > 0

        def transform_to_base_parametrization(self) -> ExponentialFamilyParametrization:
            return ExponentialFamilyParametrization(theta=np.array([-(self.alpha + 1.0)]))

    ParametricFamilyRegister.register(pareto_family)
