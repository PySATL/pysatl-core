"""
Log-Normal distribution family implementation.

Contains the LogNormal family with multiple parameterizations.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025"
__license__ = "SPDX-License-Identifier: MIT"

from typing import cast

import numpy as np
from scipy.special import erf, erfinv

from pysatl_core.distributions.support import ContinuousSupport
from pysatl_core.families.parametric_family import ParametricFamily
from pysatl_core.families.parametrizations import (
    Parametrization,
    constraint,
    parametrization,
)
from pysatl_core.families.registry import ParametricFamilyRegister
from pysatl_core.types import (
    CharacteristicName,
    FamilyName,
    NumericArray,
    UnivariateContinuous,
)


def configure_lognormal_family() -> None:
    """Configure and register the LogNormal distribution family."""

    if ParametricFamilyRegister.contains(FamilyName.LOGNORMAL):
        return

    LOGNORMAL_DOC = """
    Log-Normal distribution.

    If a random variable Y is normally distributed with mean μ and standard deviation σ,
    then X = exp(Y) follows a log‑normal distribution. Its probability density function is:

        f(x) = 1/(x σ √(2π)) * exp(-(ln x - μ)²/(2σ²)),   for x > 0.

    The distribution is often used to model quantities that cannot be negative,
    such as incomes, stock prices, or lifetimes.
    """

    def pdf(parameters: Parametrization, x: NumericArray) -> NumericArray:
        """Probability density function of the log‑normal distribution."""
        params = cast(_MeanStd, parameters)
        mu = params.mu
        sigma = params.sigma

        if x <= 0:
            raise ValueError("X must be in [0, +inf)")

        exponent = np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma**2))
        coefficient = 1 / (x * sigma * np.sqrt(np.pi * 2))

        return cast(NumericArray, coefficient * exponent)

    def cdf(parameters: Parametrization, x: NumericArray) -> NumericArray:
        parameters = cast(_MeanStd, parameters)
        if x <= 0:
            raise ValueError("X must be in [0, +inf)")

        z = (np.log(x) - parameters.mu) / (parameters.sigma * np.sqrt(2))
        return cast(NumericArray, 0.5 * (1 + erf(z)))

    def ppf(parameters: Parametrization, p: NumericArray) -> NumericArray:
        if np.any((p < 0) | (p > 1)):
            raise ValueError("Probability must be in [0, 1]")

        parameters = cast(_MeanStd, parameters)
        result = np.exp(parameters.mu + np.sqrt(2 * parameters.sigma**2) * erfinv(2 * p - 1))
        return cast(NumericArray, result)

    def lpdf(parameters: Parametrization, x: NumericArray) -> NumericArray:
        return np.log(pdf(parameters, x))

    def mean_func(parameters: Parametrization) -> float:
        """Mean of normal distribution."""
        parameters = cast(_MeanStd, parameters)
        return cast(float, np.exp(parameters.mu + parameters.sigma**2 / 2))

    def var_func(parameters: Parametrization) -> float:
        """Variance of normal distribution."""
        parameters = cast(_MeanStd, parameters)
        return cast(
            float,
            (np.exp(parameters.sigma**2) - 1) * np.exp(2 * parameters.mu + parameters.sigma**2),
        )

    def skew_func(parameters: Parametrization) -> float:
        """Skewness of normal distribution (always 0)."""
        parameters = cast(_MeanStd, parameters)
        return cast(
            float, (np.exp(parameters.sigma**2) + 2) * np.sqrt(np.exp(parameters.sigma**2) - 1)
        )

    def kurt_func(parameters: Parametrization, excess: bool = False) -> float:
        parameters = cast(_MeanStd, parameters)
        if not excess:
            return 0.0
        else:
            return cast(
                float,
                np.exp(4 * parameters.sigma**2)
                + 2 * np.exp(3 * parameters.sigma**2)
                + 3 * np.exp(2 * parameters.sigma**2)
                - 6,
            )

    def _support(_: Parametrization) -> ContinuousSupport:
        """Support of the log‑normal distribution (0, ∞)."""
        return ContinuousSupport(left=0.0, left_closed=False, right=np.inf, right_closed=False)

    LogNormal = ParametricFamily(
        name=FamilyName.LOGNORMAL,
        distr_type=UnivariateContinuous,
        distr_parametrizations=["meanStd"],
        distr_characteristics={
            CharacteristicName.PDF: pdf,
            CharacteristicName.CDF: cdf,
            CharacteristicName.PPF: ppf,
            CharacteristicName.LPDF: lpdf,
            CharacteristicName.MEAN: mean_func,
            CharacteristicName.VAR: var_func,
            CharacteristicName.SKEW: skew_func,
            CharacteristicName.KURT: kurt_func,
        },
        support_by_parametrization=_support,
    )
    LogNormal.__doc__ = LOGNORMAL_DOC

    @parametrization(family=LogNormal, name="meanStd")  # family will be set after Normal is created
    class _MeanStd(Parametrization):
        mu: float
        sigma: float

        @constraint(description="sigma > 0")
        def check_sigma_positive(self) -> bool:
            return self.sigma > 0

    ParametricFamilyRegister.register(LogNormal)
