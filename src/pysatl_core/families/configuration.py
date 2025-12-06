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

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
from scipy.special import erf, erfinv

from pysatl_core.distributions.strategies import DefaultSamplingUnivariateStrategy
from pysatl_core.distributions.support import ContinuousSupport
from pysatl_core.families.parametric_family import ParametricFamily
from pysatl_core.families.parametrizations import (
    Parametrization,
    constraint,
    parametrization,
)
from pysatl_core.families.registry import ParametricFamilyRegister
from pysatl_core.types import UnivariateContinuous

if TYPE_CHECKING:
    from typing import Any


PDF = "pdf"
CDF = "cdf"
PPF = "ppf"
CF = "char_func"
MEAN = "mean"
VAR = "var"
SKEW = "skewness"
KURT = "kurtosis"


@lru_cache(maxsize=1)
def configure_families_register() -> ParametricFamilyRegister:
    """
    Configure and register all distribution families in the global registry.

    This function initializes all parametric families with their respective
    parameterizations, characteristics, and sampling strategies. It should be
    called during application startup to make distributions available.

    Returns
    -------
    ParametricFamilyRegister
        The global registry of parametric families.
    """
    _configure_normal_family()
    return ParametricFamilyRegister()


@dataclass
class NormalMeanStdParametrization(Parametrization):
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


@dataclass
class NormalMeanPrecParametrization(Parametrization):
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
        return NormalMeanStdParametrization(mu=self.mu, sigma=sigma)


@dataclass
class NormalExpParametrization(Parametrization):
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
        return NormalMeanStdParametrization(mu=mu, sigma=sigma)


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
        parameters = cast(NormalMeanStdParametrization, parameters)

        sigma = parameters.sigma
        mu = parameters.mu

        coefficient = 1.0 / (sigma * np.sqrt(2 * np.pi))
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
        parameters = cast(NormalMeanStdParametrization, parameters)

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
            If p[i] is 0 or 1, then the result[i] is -inf and inf correspondingly

        Raises
        ------
        ValueError
            If probability is outside [0, 1]
        """
        if np.any((p < 0) | (p > 1)):
            raise ValueError("Probability must be in [0, 1]")

        parameters = cast(NormalMeanStdParametrization, parameters)

        result = cast(
            npt.NDArray[np.float64],
            parameters.mu + parameters.sigma * np.sqrt(2) * erfinv(2 * p - 1),
        )
        return result

    def normal_char_func(
        parameters: Parametrization, t: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.complex128]:
        """
        Characteristic function of normal distribution.

        Parameters
        ----------
        parameters : Parametrization
            Distribution parameters object with fields:
            - mu: float (mean)
            - sigma: float (standard deviation)
        x : npt.NDArray[np.float64]
            Points at which to evaluate the characteristic function

        Returns
        -------
        npt.NDArray[np.complex128]
            Characteristic function values at points x
        """
        parameters = cast(NormalMeanStdParametrization, parameters)

        sigma = parameters.sigma
        mu = parameters.mu
        return cast(npt.NDArray[np.complex128], np.exp(1j * mu * t - 0.5 * (sigma**2) * (t**2)))

    def mean_func(parameters: Parametrization, _: Any) -> float:
        """Mean of normal distribution."""
        parameters = cast(NormalMeanStdParametrization, parameters)
        return parameters.mu

    def var_func(parameters: Parametrization, _: Any) -> float:
        """Variance of normal distribution."""
        parameters = cast(NormalMeanStdParametrization, parameters)
        return parameters.sigma**2

    def skew_func(_1: Parametrization, _2: Any) -> int:
        """Skewness of normal distribution (always 0)."""
        return 0

    def kurt_func(_1: Parametrization, _2: Any, excess: bool = False) -> int:
        """Raw or excess kurtosis of normal distribution (always 3).

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

    def _normal_support(_: Parametrization) -> ContinuousSupport:
        return ContinuousSupport()

    Normal = ParametricFamily(
        name="Normal Family",
        distr_type=UnivariateContinuous,
        distr_parametrizations=["meanStd", "meanPrec", "exponential"],
        distr_characteristics={
            PDF: normal_pdf,
            CDF: normal_cdf,
            PPF: normal_ppf,
            CF: normal_char_func,
            MEAN: mean_func,
            VAR: var_func,
            SKEW: skew_func,
            KURT: kurt_func,
        },
        sampling_strategy=DefaultSamplingUnivariateStrategy(),
        support_by_parametrization=_normal_support,
    )
    Normal.__doc__ = NORMAL_DOC

    parametrization(family=Normal, name="meanStd")(NormalMeanStdParametrization)
    parametrization(family=Normal, name="meanPrec")(NormalMeanPrecParametrization)
    parametrization(family=Normal, name="exponential")(NormalExpParametrization)

    ParametricFamilyRegister.register(Normal)


def reset_families_register() -> None:
    configure_families_register.cache_clear()
    ParametricFamilyRegister._reset()
