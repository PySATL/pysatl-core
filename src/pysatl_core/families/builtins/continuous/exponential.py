"""
Exponential distribution family implementation.

Contains the Exponential family with rate and scale parameterizations.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, cast

import numpy as np

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


def configure_exponential_family() -> None:
    """
    Configure and register the Exponential distribution family.
    """

    if ParametricFamilyRegister.contains(FamilyName.EXPONENTIAL):
        return

    EXPONENTIAL_DOC = """
    Exponential distribution.

    The exponential distribution is a continuous probability distribution that
    describes the time between events in a Poisson process. It has a single
    parameter: rate (λ) or scale (β = 1/λ).

    Probability density function (rate parametrization):
        f(x) = λ * exp(-λ * x) for x ≥ 0

    The exponential distribution is memoryless and is widely used in reliability
    engineering, queuing theory, and survival analysis.
    """

    def pdf(parameters: Parametrization, x: NumericArray) -> NumericArray:
        """
        Probability density function for exponential distribution.

        Parameters
        ----------
        parameters : Parametrization
            Distribution parameters object with fields:
            - lambda_: float (rate parameter)
        x : NumericArray
            Points at which to evaluate the probability density function

        Returns
        -------
        NumericArray
            Probability density values at points x
        """
        parameters = cast(_Rate, parameters)

        lambda_ = parameters.lambda_
        return np.where(x >= 0, lambda_ * np.exp(-lambda_ * x), 0.0)

    def cdf(parameters: Parametrization, x: NumericArray) -> NumericArray:
        """
        Cumulative distribution function for exponential distribution.

        Parameters
        ----------
        parameters : Parametrization
            Distribution parameters object with fields:
            - lambda_: float (rate parameter)
        x : NumericArray
            Points at which to evaluate the cumulative distribution function

        Returns
        -------
        NumericArray
            Probabilities P(X ≤ x) for each point x
        """
        parameters = cast(_Rate, parameters)

        lambda_ = parameters.lambda_
        return np.where(x >= 0, 1.0 - np.exp(-lambda_ * x), 0.0)

    def ppf(parameters: Parametrization, p: NumericArray) -> NumericArray:
        """
        Percent point function (inverse CDF) for exponential distribution.

        Parameters
        ----------
        parameters : Parametrization
            Distribution parameters object with fields:
            - lambda_: float (rate parameter)
        p : NumericArray
            Probability from [0, 1]

        Returns
        -------
        NumericArray
            Quantiles corresponding to probabilities p:
            - For p = 0: returns 0.0
            - For p = 1: returns np.inf
            - For p in (0, 1): returns -ln(1-p)/λ

        Raises
        ------
        ValueError
            If probability is outside [0, 1]
        """
        if np.any((p < 0) | (p > 1)):
            raise ValueError("Probability must be in [0, 1]")

        parameters = cast(_Rate, parameters)
        lambda_ = parameters.lambda_

        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(p < 1.0, -np.log1p(-p) / lambda_, np.inf)

    def char_func(parameters: Parametrization, t: NumericArray) -> ComplexArray:
        """
        Characteristic function of exponential distribution.

        Parameters
        ----------
        parameters : Parametrization
            Distribution parameters object with fields:
            - lambda_: float (rate parameter)
        t : NumericArray
            Points at which to evaluate the characteristic function

        Returns
        -------
        ComplexArray
            Characteristic function values at points t
        """
        CALCULATION_PRECISION = 1e-10

        parameters = cast(_Rate, parameters)
        lambda_ = parameters.lambda_
        t_arr = np.asarray(t, dtype=np.float64)

        denominator = lambda_ - 1j * t_arr
        result = np.where(
            np.abs(t_arr) < CALCULATION_PRECISION,
            1.0 + 0j,
            lambda_ / denominator,
        )
        return cast(ComplexArray, result)

    def mean_func(parameters: Parametrization, _: Any) -> float:
        """Mean of exponential distribution."""
        parameters = cast(_Rate, parameters)
        return 1.0 / parameters.lambda_

    def var_func(parameters: Parametrization, _: Any) -> float:
        """Variance of exponential distribution."""
        parameters = cast(_Rate, parameters)
        return 1.0 / (parameters.lambda_**2)

    def skew_func(_1: Parametrization, _2: Any) -> float:
        """Skewness of exponential distribution (always 2)."""
        return 2.0

    def kurt_func(_1: Parametrization, _2: Any, excess: bool = False) -> float:
        """Raw or excess kurtosis of exponential distribution.

        Parameters
        ----------
        _1 : Parametrization
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
            return 9.0
        else:
            return 6.0

    def _support(_: Parametrization) -> ContinuousSupport:
        """Support of exponential distribution"""
        return ContinuousSupport(left=0.0)

    Exponential = ParametricFamily(
        name=FamilyName.EXPONENTIAL,
        distr_type=UnivariateContinuous,
        distr_parametrizations=["rate", "scale"],
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
    Exponential.__doc__ = EXPONENTIAL_DOC

    @parametrization(family=Exponential, name="rate")
    class _Rate(Parametrization):
        """
        Rate parametrization of exponential distribution.

        Parameters
        ----------
        lambda_ : float
            Rate parameter (λ) of the distribution
        """

        lambda_: float

        @constraint(description="lambda_ > 0")
        def check_lambda_positive(self) -> bool:
            """Check that rate parameter is positive."""
            return self.lambda_ > 0

    @parametrization(family=Exponential, name="scale")
    class _Scale(Parametrization):
        """
        Scale parametrization of exponential distribution.

        Parameters
        ----------
        beta : float
            Scale parameter (β) of the distribution, β = 1/λ
        """

        beta: float

        @constraint(description="beta > 0")
        def check_beta_positive(self) -> bool:
            """Check that scale parameter is positive."""
            return self.beta > 0

        def transform_to_base_parametrization(self) -> Parametrization:
            """
            Transform to Rate parametrization.

            Returns
            -------
            Parametrization
                Rate parametrization instance
            """
            return _Rate(lambda_=1.0 / self.beta)

    ParametricFamilyRegister.register(Exponential)
