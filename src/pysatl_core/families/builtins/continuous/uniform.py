"""
Uniform distribution family implementation.

Contains the Uniform family with multiple parameterizations.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, cast

import numpy as np

from pysatl_core.distributions.strategies import DefaultSamplingUnivariateStrategy
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


def configure_uniform_family() -> None:
    """
    Configure and register the Uniform distribution family.
    """
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

    def pdf(parameters: Parametrization, x: NumericArray) -> NumericArray:
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
        x : NumericArray
            Points at which to evaluate the probability density function

        Returns
        -------
        NumericArray
            Probability density values at points x
        """
        parameters = cast(_Standard, parameters)

        lower_bound = parameters.lower_bound
        upper_bound = parameters.upper_bound

        return np.where(
            (x >= lower_bound) & (x <= upper_bound), 1.0 / (upper_bound - lower_bound), 0.0
        )

    def cdf(parameters: Parametrization, x: NumericArray) -> NumericArray:
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
        x : NumericArray
            Points at which to evaluate the cumulative distribution function

        Returns
        -------
        NumericArray
            Probabilities P(X ≤ x) for each point x
        """
        parameters = cast(_Standard, parameters)

        lower_bound = parameters.lower_bound
        upper_bound = parameters.upper_bound

        return cast(
            NumericArray, np.clip((x - lower_bound) / (upper_bound - lower_bound), 0.0, 1.0)
        )

    def ppf(parameters: Parametrization, p: NumericArray) -> NumericArray:
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
        p : NumericArray
            Probability from [0, 1]

        Returns
        -------
        NumericArray
            Quantiles corresponding to probabilities p

        Raises
        ------
        ValueError
            If probability is outside [0, 1]
        """
        if np.any((p < 0) | (p > 1)):
            raise ValueError("Probability must be in [0, 1]")

        parameters = cast(_Standard, parameters)
        lower_bound = parameters.lower_bound
        upper_bound = parameters.upper_bound

        return cast(NumericArray, lower_bound + p * (upper_bound - lower_bound))

    def char_func(parameters: Parametrization, t: NumericArray) -> ComplexArray:
        """
        Characteristic function of uniform distribution.

        Characteristic function formula for uniform distribution on [lower_bound, upper bound]:
            φ(t) = sinc((upper bound - lower_bound) * t / 2) *
            * exp(i * (lower_bound + upper_bound) * t / 2)
        where sinc(x) = sin(πx)/(πx) as defined by numpy.

        Parameters
        ----------
        parameters : Parametrization
            Distribution parameters object with fields:
            - lower_bound: float (lower bound)
            - upper_bound: float (upper bound)
        t : NumericArray
            Points at which to evaluate the characteristic function

        Returns
        -------
        ComplexArray
            Characteristic function values at points t
        """
        parameters = cast(_Standard, parameters)

        lower_bound = parameters.lower_bound
        upper_bound = parameters.upper_bound

        width = upper_bound - lower_bound
        center = (lower_bound + upper_bound) / 2

        t_arr = np.asarray(t, dtype=np.float64)

        x = width * t_arr / (2 * np.pi)
        sinc_val = np.sinc(x)

        return cast(ComplexArray, sinc_val * np.exp(1j * center * t_arr))

    def mean_func(parameters: Parametrization, _: Any) -> float:
        """Mean of uniform distribution."""
        parameters = cast(_Standard, parameters)
        return (parameters.lower_bound + parameters.upper_bound) / 2

    def var_func(parameters: Parametrization, _: Any) -> float:
        """Variance of uniform distribution."""
        parameters = cast(_Standard, parameters)
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

    def _support(parameters: Parametrization) -> ContinuousSupport:
        """Support of uniform distribution"""
        parameters = cast(_Standard, parameters.transform_to_base_parametrization())
        return ContinuousSupport(
            left=parameters.lower_bound,
            right=parameters.upper_bound,
            left_closed=True,
            right_closed=True,
        )

    Uniform = ParametricFamily(
        name=FamilyName.CONTINUOUS_UNIFORM,
        distr_type=UnivariateContinuous,
        distr_parametrizations=["standard", "meanWidth", "minRange"],
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
        sampling_strategy=DefaultSamplingUnivariateStrategy(),
        support_by_parametrization=_support,
    )
    Uniform.__doc__ = UNIFORM_DOC

    @parametrization(family=Uniform, name="standard")
    class _Standard(Parametrization):
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

    @parametrization(family=Uniform, name="meanWidth")
    class _MeanWidth(Parametrization):
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
            return _Standard(lower_bound=self.mean - half_width, upper_bound=self.mean + half_width)

    @parametrization(family=Uniform, name="minRange")
    class _MinRange(Parametrization):
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
            return _Standard(lower_bound=self.minimum, upper_bound=self.minimum + self.range_val)

    ParametricFamilyRegister.register(Uniform)
