"""
Distribution Families Configuration
====================================

This module defines and configures parametric distribution families for the PySATL library:

- :class:`Normal Family` — Gaussian distribution with multiple parameterizations.

Notes
-----
- All families are registered in the global ParametricFamilyRegister.
- Each family supports multiple parameterizations with automatic conversions.
- Analytical implementations are provided where available, with fallbacks to numerical methods.
- Families are designed to be extensible with additional characteristics and parameterizations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from scipy.special import erf, erfinv

from pysatl_core.distributions import DefaultSamplingUnivariateStrategy
from pysatl_core.families.parametric_family import ParametricFamily
from pysatl_core.families.parametrizations import (
    Parametrization,
    constraint,
    parametrization,
)
from pysatl_core.families.registry import ParametricFamilyRegister
from pysatl_core.types import UnivariateContinuous

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


def configure_family_register() -> None:
    """
    Configure and register all distribution families in the global registry.

    This function initializes all parametric families with their respective
    parameterizations, characteristics, and sampling strategies. It should be
    called during application startup to make distributions available.
    """
    _configure_normal_family()


PDF = "pdf"
CDF = "cdf"
PPF = "ppf"
CF = "char_func"
MEAN = "mean"
VAR = "var"
SKEW = "skewness"
RAWKURT = "raw_kurtosis"
EXKURT = "excess_kurtosis"


@dataclass
class MeanVarParametrization(Parametrization):
    """
    Mean-variance parametrization of normal distribution.

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


@dataclass
class MeanPrecParametrization(Parametrization):
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
        Transform to mean-variance parametrization.

        Returns
        -------
        Parametrization
            Mean-variance parametrization instance
        """
        sigma = math.sqrt(1 / self.tau)
        return MeanVarParametrization(mu=self.mu, sigma=sigma)


@dataclass
class ExpParametrization(Parametrization):
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
    def calculate_c(self) -> float:
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
        Transform to mean-variance parametrization.
        Returns
        -------
        Parametrization
            Mean-variance parametrization instance
        """
        mu = -self.b / (2 * self.a)
        sigma = math.sqrt(-1 / (2 * self.a))
        return MeanVarParametrization(mu=mu, sigma=sigma)


def _configure_normal_family() -> None:
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

    def normal_pdf(
        parameters: Parametrization, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Probability density function for normal distribution.

        Parameters
        ----------
        parameters : Parametrization ()
        Distribution parameters object with fields:
            - mu: float (mean)
            - sigma: float (standard deviation)
        x : npt.NDArray[np.float64]
            Points at which to evaluate the probability density function

        Returns
        -------
        npt.NDArray[np.float64]
            Probability density values at points x
        """
        parameters = cast(MeanVarParametrization, parameters)

        sigma = parameters.sigma
        mu = parameters.mu

        coefficient = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
        exponent = -((x - mu) ** 2) / (2 * sigma**2)

        return cast(npt.NDArray[np.float64], coefficient * np.exp(exponent))

    def normal_cdf(
        parameters: Parametrization, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Cumulative distribution function for normal distribution.

        Parameters
        ----------
        parameters : Parametrization
            Distribution parameters object with fields:
            - mu: float (mean)
            - sigma: float (standard deviation)
        x : npt.NDArray[np.float64]
            Points at which to evaluate the cumulative distribution function

        Returns
        -------
        npt.NDArray[np.float64]
            Probabilities P(X ≤ x) for each point x
        """
        parameters = cast(MeanVarParametrization, parameters)

        z = (x - parameters.mu) / (parameters.sigma * np.sqrt(2))
        return cast(npt.NDArray[np.float64], 0.5 * (1 + erf(z)))

    def normal_ppf(
        parameters: Parametrization, p: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Percent point function (inverse CDF) for normal distribution.

        Parameters
        ----------
        parameters : Parametrization
            Distribution parameters object with fields:
            - mu: float (mean)
            - sigma: float (standard deviation)
        p : npt.NDArray[np.float64]
            Probability from [0, 1]

        Returns
        -------
        npt.NDArray[np.float64]
            Quantiles corresponding to probabilities p

        Raises
        ------
        ValueError
            If probability is outside [0, 1]
        """
        if np.any((p < 0) | (p > 1)):
            raise ValueError("Probability must be in [0, 1]")

        parameters = cast(MeanVarParametrization, parameters)

        return cast(
            npt.NDArray[np.float64],
            parameters.mu + parameters.sigma * np.sqrt(2) * erfinv(2 * p - 1),
        )

    def normal_char_func(
        parameters: Parametrization, x: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.complex128]:
        """
        Characteristic function of normal distribution.

        Parameters
        ----------
        parameters : Parametrization
            Distribution parameters object with fields:
            - mu: float (mean)
            - sigma: float (standard deviation)
        x : npt.NDArray[np.complex128]
            Points at which to evaluate the characteristic function

        Returns
        -------
        npt.NDArray[np.complex128]
            Characteristic function values at points x
        """
        parameters = cast(MeanVarParametrization, parameters)

        sigma = parameters.sigma
        mu = parameters.mu
        return cast(npt.NDArray[np.complex128], np.exp(1j * mu * x - 0.5 * (sigma * x) ** 2))

    def mean_func(parameters: Parametrization, __: Any = None) -> float:
        """Mean of normal distribution."""
        parameters = cast(MeanVarParametrization, parameters)
        return parameters.mu

    def var_func(parameters: Parametrization, __: Any = None) -> float:
        """Variance of normal distribution."""
        parameters = cast(MeanVarParametrization, parameters)
        return parameters.sigma**2

    def skew_func(_: Parametrization, __: Any = None) -> int:
        """Skewness of normal distribution (always 0)."""
        return 0

    def raw_kurt_func(_: Parametrization, __: Any = None) -> int:
        """Raw kurtosis of normal distribution (always 3)."""
        return 3

    def ex_kurt_func(_: Parametrization, __: Any) -> int:
        """Excess kurtosis of normal distribution (always 0)."""
        return 0

    Normal = ParametricFamily(
        name="Normal Family",
        distr_type=UnivariateContinuous,
        distr_parametrizations=["meanVar", "meanPrec", "exponential"],
        distr_characteristics={
            PDF: normal_pdf,
            CDF: normal_cdf,
            PPF: normal_ppf,
            CF: normal_char_func,
            MEAN: mean_func,
            VAR: var_func,
            SKEW: skew_func,
            RAWKURT: raw_kurt_func,
            EXKURT: ex_kurt_func,
        },
        sampling_strategy=DefaultSamplingUnivariateStrategy(),
    )
    Normal.__doc__ = NORMAL_DOC

    parametrization(family=Normal, name="meanVar")(MeanVarParametrization)
    parametrization(family=Normal, name="meanPrec")(MeanPrecParametrization)
    parametrization(family=Normal, name="exponential")(ExpParametrization)

    ParametricFamilyRegister.register(Normal)
