"""
Distribution Families Configuration
====================================

This module defines and configures parametric distribution families for the PySATL library:

- :class:`Normal Family` — Gaussian distribution with multiple parameterizations.
- :class:`Uniform Family` — Gaussian distribution with multiple parameterizations.

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
from pysatl_core.types import CharacteristicName, UnivariateContinuous

if TYPE_CHECKING:
    from typing import Any


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
    _configure_uniform_family()
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

    def _normal_support(_: Parametrization) -> ContinuousSupport:
        """Support of normal distribution"""
        return ContinuousSupport()

    Normal = ParametricFamily(
        name="Normal",
        distr_type=UnivariateContinuous,
        distr_parametrizations=["meanStd", "meanPrec", "exponential"],
        distr_characteristics={
            CharacteristicName.PDF: normal_pdf,
            CharacteristicName.CDF: normal_cdf,
            CharacteristicName.PPF: normal_ppf,
            CharacteristicName.CF: normal_char_func,
            CharacteristicName.MEAN: mean_func,
            CharacteristicName.VAR: var_func,
            CharacteristicName.SKEW: skew_func,
            CharacteristicName.KURT: kurt_func,
        },
        sampling_strategy=DefaultSamplingUnivariateStrategy(),
        support_by_parametrization=_normal_support,
    )
    Normal.__doc__ = NORMAL_DOC

    parametrization(family=Normal, name="meanStd")(NormalMeanStdParametrization)
    parametrization(family=Normal, name="meanPrec")(NormalMeanPrecParametrization)
    parametrization(family=Normal, name="exponential")(NormalExpParametrization)

    ParametricFamilyRegister.register(Normal)


@dataclass
class UniformStandardParametrization(Parametrization):
    """
    Standard parametrization of uniform distribution.

    Parameters
    ----------
    lower_bound : float
        Lower bound of the distribution
    upper_bound : float
        Upper bound of the distribution
    """

    lower_bound: float
    upper_bound: float

    @constraint(description="lower_bound < upper_bound")
    def check_lower_less_than_upper(self) -> bool:
        """Check that lower bound is less than upper bound."""
        return self.lower_bound < self.upper_bound


@dataclass
class UniformMeanWidthParametrization(Parametrization):
    """
    Mean-width parametrization of uniform distribution.

    Parameters
    ----------
    mean : float
        Mean (center) of the distribution
    width : float
        Width of the distribution (upper_bound - lower_bound)
    """

    mean: float
    width: float

    @constraint(description="width > 0")
    def check_width_positive(self) -> bool:
        """Check that width is positive."""
        return self.width > 0

    def transform_to_base_parametrization(self) -> Parametrization:
        """
        Transform to Standard parametrization.

        Returns
        -------
        Parametrization
            Standard parametrization instance
        """
        half_width = self.width / 2
        return UniformStandardParametrization(
            lower_bound=self.mean - half_width, upper_bound=self.mean + half_width
        )


@dataclass
class UniformMinRangeParametrization(Parametrization):
    """
    Minimum-range parametrization of uniform distribution.

    Parameters
    ----------
    minimum : float
        Minimum value (lower bound)
    range_val : float
        Range of the distribution (upper_bound - lower_bound)
    """

    minimum: float
    range_val: float

    @constraint(description="range_val > 0")
    def check_range_positive(self) -> bool:
        """Check that range is positive."""
        return self.range_val > 0

    def transform_to_base_parametrization(self) -> Parametrization:
        """
        Transform to Standard parametrization.

        Returns
        -------
        Parametrization
            Standard parametrization instance
        """
        return UniformStandardParametrization(
            lower_bound=self.minimum, upper_bound=self.minimum + self.range_val
        )


def _configure_uniform_family() -> None:
    UNIFORM_DOC = """
    Uniform (continuous) distribution.

    The uniform distribution is a continuous probability distribution where
    all intervals of the same length are equally probable. It is defined by
    two parameters: lower bound and upper bound.

    Probability density function:
        f(x) = 1/(upper_bound - lower_bound) for x in [lower_bound, upper_bound], 0 otherwise

    The uniform distribution is often used when there is no prior knowledge
    about the possible values of a variable, representing maximum uncertainty.
    """

    def uniform_pdf(
        parameters: Parametrization, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Probability density function for uniform distribution.
            - For x < lower_bound: returns 0
            - For x > upper_bound: returns 0
            - Otherwise: returns (1 / (upper_bound - lower_bound))

        Parameters
        ----------
        parameters : Parametrization
            Distribution parameters object with fields:
            - lower_bound: float (lower bound)
            - upper_bound: float (upper bound)
        x : npt.NDArray[np.float64]
            Points at which to evaluate the probability density function

        Returns
        -------
        npt.NDArray[np.float64]
            Probability density values at points x
        """
        parameters = cast(UniformStandardParametrization, parameters)

        lower_bound = parameters.lower_bound
        upper_bound = parameters.upper_bound

        return np.where(
            (x >= lower_bound) & (x <= upper_bound), 1.0 / (upper_bound - lower_bound), 0.0
        )

    def uniform_cdf(
        parameters: Parametrization, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Cumulative distribution function for uniform distribution.
        Uses np.clip for vectorized computation:
            - For x < lower_bound: returns 0
            - For x > upper_bound: returns 1

        Parameters
        ----------
        parameters : Parametrization
            Distribution parameters object with fields:
            - lower_bound: float (lower bound)
            - upper_bound: float (upper bound)
        x : npt.NDArray[np.float64]
            Points at which to evaluate the cumulative distribution function

        Returns
        -------
        npt.NDArray[np.float64]
            Probabilities P(X ≤ x) for each point x
        """
        parameters = cast(UniformStandardParametrization, parameters)

        lower_bound = parameters.lower_bound
        upper_bound = parameters.upper_bound

        return np.clip((x - lower_bound) / (upper_bound - lower_bound), 0.0, 1.0)

    def uniform_ppf(
        parameters: Parametrization, p: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Percent point function (inverse CDF) for uniform distribution.

        For uniform distribution on [lower_bound, upper_bound]:
        - For p = 0: returns lower_bound
        - For p = 1: returns upper_bound
        - For p in (0, 1): returns lower_bound + p × (upper_bound - lower_bound)

        Parameters
        ----------
        parameters : Parametrization
            Distribution parameters object with fields:
            - lower_bound: float (lower bound)
            - upper_bound: float (upper bound)
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

        parameters = cast(UniformStandardParametrization, parameters)
        lower_bound = parameters.lower_bound
        upper_bound = parameters.upper_bound

        return cast(npt.NDArray[np.float64], lower_bound + p * (upper_bound - lower_bound))

    def uniform_char_func(
        parameters: Parametrization, t: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.complex128]:
        """
        Characteristic function of uniform distribution.

        Characteristic function formula for uniform distribution on [lower_bound, upper bound]:
            φ(t) = sinc((upper bound - lower_bound) * t / 2) *
            * exp(i * (lower_bound + upper bound) * t / 2)
        where sinc(x) = sin(πx)/(πx) as defined by numpy.

        Parameters
        ----------
        parameters : Parametrization
            Distribution parameters object with fields:
            - lower_bound: float (lower bound)
            - upper_bound: float (upper bound)
        t : npt.NDArray[np.float64]
            Points at which to evaluate the characteristic function

        Returns
        -------
        npt.NDArray[np.complex128]
            Characteristic function values at points t
        """
        parameters = cast(UniformStandardParametrization, parameters)

        lower_bound = parameters.lower_bound
        upper_bound = parameters.upper_bound

        width = upper_bound - lower_bound
        center = (lower_bound + upper_bound) / 2

        t_arr = np.asarray(t, dtype=np.float64)

        x = width * t_arr / (2 * np.pi)
        sinc_val = np.sinc(x)

        return cast(npt.NDArray[np.complex128], sinc_val * np.exp(1j * center * t_arr))

    def mean_func(parameters: Parametrization, _: Any) -> float:
        """Mean of uniform distribution."""
        parameters = cast(UniformStandardParametrization, parameters)
        return (parameters.lower_bound + parameters.upper_bound) / 2

    def var_func(parameters: Parametrization, _: Any) -> float:
        """Variance of uniform distribution."""
        parameters = cast(UniformStandardParametrization, parameters)
        width = parameters.upper_bound - parameters.lower_bound
        return width**2 / 12

    def skew_func(_1: Parametrization, _2: Any) -> int:
        """Skewness of uniform distribution (always 0)."""
        return 0

    def kurt_func(_1: Parametrization, _2: Any, excess: bool = False) -> float:
        """Raw or excess kurtosis of uniform distribution.

        Parameters
        ----------
        _1 : Parametrization
            Needed by architecture parameter
        _2 : Any
            Needed by architecture parameter
        excess : bool
            A value defines if there will be raw or excess kurtosis
            default is False

        Returns
        -------
        float
            Kurtosis value
        """
        if not excess:
            return 1.8
        else:
            return -1.2

    def _uniform_support(parameters: Parametrization) -> ContinuousSupport:
        """Support of uniform distribution"""
        parameters = cast(
            UniformStandardParametrization, parameters.transform_to_base_parametrization()
        )
        return ContinuousSupport(
            left=parameters.lower_bound,
            right=parameters.upper_bound,
            left_closed=True,
            right_closed=True,
        )

    Uniform = ParametricFamily(
        name="ContinuousUniform",
        distr_type=UnivariateContinuous,
        distr_parametrizations=["standard", "meanWidth", "minRange"],
        distr_characteristics={
            PDF: uniform_pdf,
            CDF: uniform_cdf,
            PPF: uniform_ppf,
            CF: uniform_char_func,
            MEAN: mean_func,
            VAR: var_func,
            SKEW: skew_func,
            KURT: kurt_func,
        },
        sampling_strategy=DefaultSamplingUnivariateStrategy(),
        support_by_parametrization=_uniform_support,
    )
    Uniform.__doc__ = UNIFORM_DOC

    parametrization(family=Uniform, name="standard")(UniformStandardParametrization)
    parametrization(family=Uniform, name="meanWidth")(UniformMeanWidthParametrization)
    parametrization(family=Uniform, name="minRange")(UniformMinRangeParametrization)

    ParametricFamilyRegister.register(Uniform)


def reset_families_register() -> None:
    configure_families_register.cache_clear()
    ParametricFamilyRegister._reset()
