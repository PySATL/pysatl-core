"""
Normal distribution family implementation.

Contains the Normal family with multiple parameterizations.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import math
from typing import TYPE_CHECKING, cast

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
    ComplexArray,
    FamilyName,
    NumericArray,
    UnivariateContinuous,
)

if TYPE_CHECKING:
    from typing import Any


def configure_normal_family() -> None:
    """
    Configure and register the Normal distribution family.
    """

    if ParametricFamilyRegister.contains(FamilyName.NORMAL):
        return

    NORMAL_DOC = """
    Normal (Gaussian) distribution.

    The normal distribution is a continuous probability distribution characterized
    by its bell-shaped curve. It is symmetric about its mean and is defined by
    two parameters: mean (μ) and standard deviation (σ).

    Probability density function:
        f(x) = 1/(σ√(2π)) * exp(-(x-μ)²/(2σ²))

    The normal distribution is widely used in statistics, natural sciences,
    and social sciences as a simple model for complex random phenomena.
    """

    def pdf(parameters: Parametrization, x: NumericArray) -> NumericArray:
        """
        Probability density function for normal distribution.

        Parameters
        ----------
        parameters : Parametrization ()
        Distribution parameters object with fields:
            - mu: float (mean)
            - sigma: float (standard deviation)
        x : NumericArray
            Points at which to evaluate the probability density function

        Returns
        -------
        NumericArray
            Probability density values at points x
        """
        parameters = cast(_MeanStd, parameters)

        sigma = parameters.sigma
        mu = parameters.mu

        coefficient = 1.0 / (sigma * np.sqrt(2 * np.pi))
        exponent = -((x - mu) ** 2) / (2 * sigma**2)

        return cast(NumericArray, coefficient * np.exp(exponent))

    def cdf(parameters: Parametrization, x: NumericArray) -> NumericArray:
        """
        Cumulative distribution function for normal distribution.

        Parameters
        ----------
        parameters : Parametrization
            Distribution parameters object with fields:
            - mu: float (mean)
            - sigma: float (standard deviation)
        x : NumericArray
            Points at which to evaluate the cumulative distribution function

        Returns
        -------
        NumericArray
            Probabilities P(X ≤ x) for each point x
        """
        parameters = cast(_MeanStd, parameters)

        z = (x - parameters.mu) / (parameters.sigma * np.sqrt(2))
        return cast(NumericArray, 0.5 * (1 + erf(z)))

    def ppf(parameters: Parametrization, p: NumericArray) -> NumericArray:
        """
        Percent point function (inverse CDF) for normal distribution.

        Parameters
        ----------
        parameters : Parametrization
            Distribution parameters object with fields:
            - mu: float (mean)
            - sigma: float (standard deviation)
        p : NumericArray
            Probability from [0, 1]

        Returns
        -------
        NumericArray
            Quantiles corresponding to probabilities p
            If p[i] is 0 or 1, then the result[i] is -inf and inf correspondingly

        Raises
        ------
        ValueError
            If probability is outside [0, 1]
        """
        if np.any((p < 0) | (p > 1)):
            raise ValueError("Probability must be in [0, 1]")

        parameters = cast(_MeanStd, parameters)

        result = cast(
            NumericArray,
            parameters.mu + parameters.sigma * np.sqrt(2) * erfinv(2 * p - 1),
        )
        return result

    def char_func(parameters: Parametrization, t: NumericArray) -> ComplexArray:
        """
        Characteristic function of normal distribution.

        Parameters
        ----------
        parameters : Parametrization
            Distribution parameters object with fields:
            - mu: float (mean)
            - sigma: float (standard deviation)
        x : NumericArray
            Points at which to evaluate the characteristic function

        Returns
        -------
        ComplexArray
            Characteristic function values at points x
        """
        parameters = cast(_MeanStd, parameters)

        sigma = parameters.sigma
        mu = parameters.mu
        return cast(ComplexArray, np.exp(1j * mu * t - 0.5 * (sigma**2) * (t**2)))

    def mean_func(parameters: Parametrization, _: Any) -> float:
        """Mean of normal distribution."""
        parameters = cast(_MeanStd, parameters)
        return parameters.mu

    def var_func(parameters: Parametrization, _: Any) -> float:
        """Variance of normal distribution."""
        parameters = cast(_MeanStd, parameters)
        return parameters.sigma**2

    def skew_func(_1: Parametrization, _2: Any) -> int:
        """Skewness of normal distribution (always 0)."""
        return 0

    def kurt_func(_1: Parametrization, _2: Any, excess: bool = False) -> int:
        """Raw or excess kurtosis of normal distribution.

        Parameters
        ----------
        _1 : Parametrization
            Needed by architecture parameter
        excess : bool
            A value defines if there will be raw or excess kurtosis
            default is False

        Returns
        -------
        int
            Kurtosis value
        """
        if not excess:
            return 3
        else:
            return 0

    def _support(_: Parametrization) -> ContinuousSupport:
        """Support of normal distribution"""
        return ContinuousSupport()

    Normal = ParametricFamily(
        name=FamilyName.NORMAL,
        distr_type=UnivariateContinuous,
        distr_parametrizations=["meanStd", "meanPrec", "exponential"],
        distr_characteristics={
            CharacteristicName.PDF: pdf,
            CharacteristicName.CDF: cdf,
            CharacteristicName.PPF: ppf,
            CharacteristicName.CF: char_func,
            CharacteristicName.MEAN: mean_func,
            CharacteristicName.VAR: var_func,
            CharacteristicName.SKEW: skew_func,
            CharacteristicName.KURT: kurt_func,
        },
        support_by_parametrization=_support,
    )
    Normal.__doc__ = NORMAL_DOC

    @parametrization(family=Normal, name="meanStd")
    class _MeanStd(Parametrization):
        """
        Standard parametrization of normal distribution.

        Parameters
        ----------
        mu : float
            Mean of the distribution
        sigma : float
            Standard deviation of the distribution
        """

        mu: float
        sigma: float

        @constraint(description="sigma > 0")
        def check_sigma_positive(self) -> bool:
            """Check that standard deviation is positive."""
            return self.sigma > 0

    @parametrization(family=Normal, name="meanPrec")
    class _MeanPrec(Parametrization):
        """
        Mean-precision parametrization of normal distribution.

        Parameters
        ----------
        mu : float
            Mean of the distribution
        tau : float
            Precision parameter (inverse variance)
        """

        mu: float
        tau: float

        @constraint(description="tau > 0")
        def check_tau_positive(self) -> bool:
            """Check that precision parameter is positive."""
            return self.tau > 0

        def transform_to_base_parametrization(self) -> Parametrization:
            """
            Transform to Standard parametrization.

            Returns
            -------
            Parametrization
                Standard parametrization instance
            """
            sigma = math.sqrt(1 / self.tau)
            return _MeanStd(mu=self.mu, sigma=sigma)

    @parametrization(family=Normal, name="exponential")
    class _Exp(Parametrization):
        """
        Exponential family parametrization of normal distribution.
            Uses the form: y = exp(a*x² + b*x + c)

        Parameters
        ----------
        a : float
            Quadratic term coefficient in exponential form
        b : float
            Linear term coefficient in exponential form
        """

        a: float
        b: float

        @property
        def c(self) -> float:
            """
            Calculate the normalization constant c.

            Returns
            -------
            float
                Normalization constant
            """
            return (self.b**2) / (4 * self.a) - (1 / 2) * math.log(math.pi / (-self.a))

        @constraint(description="a < 0")
        def check_a_negative(self) -> bool:
            """Check that quadratic term coefficient is negative."""
            return self.a < 0

        def transform_to_base_parametrization(self) -> Parametrization:
            """
            Transform to Standard parametrization.
            Returns
            -------
            Parametrization
                Standard parametrization instance
            """
            mu = -self.b / (2 * self.a)
            sigma = math.sqrt(-1 / (2 * self.a))
            return _MeanStd(mu=mu, sigma=sigma)

    ParametricFamilyRegister.register(Normal)
