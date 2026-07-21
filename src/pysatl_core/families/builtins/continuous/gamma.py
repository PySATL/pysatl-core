from __future__ import annotations

__author__ = "Elizaveta Bykova"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import cast

import numpy as np
from scipy.special import digamma, gammainc, gammaincinv, gammaln

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


def configure_gamma_family() -> None:
    """
    Configure and register the Gamma distribution family.
    """

    if ParametricFamilyRegister.contains(FamilyName.GAMMA):
        return

    GAMMA_DOC = """
    Gamma distribution.

    The gamma distribution is a two-parameter family of continuous probability
    distributions defined on the non-negative real line [0, +inf). It is
    asymmetric and is characterized by two parameters:
    shape (k) and scale (θ), both of which must be positive.

    Probability density function:
        f(x) = x^(k-1) * exp(-x/θ) / (Γ(k) * θ^k)  for x > 0

    The gamma distribution is widely used in statistics, engineering, and science to model
    waiting times, life testing, and continuous variables that are non-negative and
    right-skewed.
    """

    def pdf(parameters: Parametrization, x: NumericArray) -> NumericArray:
        """
        Probability density function for gamma distribution.

        Parameters
        ----------
        parameters : Parametrization ()
        Distribution parameters object with fields:
            - k: float (shape)
            - theta: float (scale)
        x : NumericArray
            Points at which to evaluate the probability density function

        Returns
        -------
        NumericArray
            Probability density values at points x
        """
        return cast(NumericArray, np.exp(lpdf(parameters, x)))

    def cdf(parameters: Parametrization, x: NumericArray) -> NumericArray:
        """
        Cumulative distribution function for gamma distribution.

        Parameters
        ----------
        parameters : Parametrization
            Distribution parameters object with fields:
            - k: float (shape)
            - theta: float (scale)
        x : NumericArray
            Points at which to evaluate the cumulative distribution function

        Returns
        -------
        NumericArray
            Probabilities P(X ≤ x) for each point x
        """
        parameters = cast(_ShapeScale, parameters)
        k = parameters.k
        theta = parameters.theta

        x_arr = np.asarray(x, dtype=np.float64)
        safe_x = np.maximum(x_arr, 0.0)
        return cast(NumericArray, np.where(x_arr > 0, gammainc(k, safe_x / theta), 0.0))

    def ppf(parameters: Parametrization, u: NumericArray) -> NumericArray:
        """
        Percent point function (inverse CDF) for gamma distribution.

        Parameters
        ----------
        parameters : Parametrization
            Distribution parameters object with fields:
            - k: float (shape)
            - theta: float (scale)
        u : NumericArray
            Probability from [0, 1]

        Returns
        -------
        NumericArray
            Quantiles corresponding to probabilities p
            If p[i] is 0 or 1, then the result[i] is 0.0 and inf correspondingly

        Raises
        ------
        ValueError
            If probability is outside [0, 1]
        """
        if np.any((u < 0) | (u > 1)):
            raise ValueError("Probability must be in [0, 1]")
        parameters = cast(_ShapeScale, parameters)

        k = parameters.k
        theta = parameters.theta
        return cast(NumericArray, theta * gammaincinv(k, u))

    def char_func(parameters: Parametrization, t: NumericArray) -> ComplexArray:
        """
        Characteristic function of gamma distribution.

        Parameters
        ----------
        parameters : Parametrization
            Distribution parameters object with fields:
            - k: float (shape)
            - theta: float (scale)
        x : NumericArray
            Points at which to evaluate the characteristic function

        Returns
        -------
        ComplexArray
            Characteristic function values at points x
        """
        parameters = cast(_ShapeScale, parameters)

        k = parameters.k
        theta = parameters.theta
        return (1 - 1j * theta * t) ** (-k)

    def lpdf(parameters: Parametrization, x: NumericArray) -> NumericArray:
        """
        Logarithm of the probability density function for gamma distribution.

        Parameters
        ----------
        parameters : Parametrization
            Distribution parameters object with fields:
            - k: float (shape)
            - theta: float (scale)
        x : NumericArray
            Points at which to evaluate the log-probability density function

        Returns
        -------
        NumericArray
            Log-probability density values at points x
        """
        parameters = cast(_ShapeScale, parameters)

        k = parameters.k
        theta = parameters.theta

        x_arr = np.asarray(x, dtype=np.float64)
        result = np.full_like(x_arr, -np.inf, dtype=np.float64)

        pos_mask = np.isfinite(x_arr) & (x_arr > 0)
        x_pos = x_arr[pos_mask]
        result[pos_mask] = (k - 1) * np.log(x_pos) - x_pos / theta - k * np.log(theta) - gammaln(k)
        zero_mask = x_arr == 0.0
        if np.any(zero_mask):
            if k > 1.0:
                result[zero_mask] = -np.inf
            elif k == 1.0:
                result[zero_mask] = -np.log(theta)
            else:  # 0 < k < 1.0
                result[zero_mask] = np.inf
        return cast(NumericArray, result)

    def mean_func(parameters: Parametrization) -> float:
        """Mean of gamma distribution."""
        parameters = cast(_ShapeScale, parameters)
        return parameters.k * parameters.theta

    def var_func(parameters: Parametrization) -> float:
        """Variance of gamma distribution."""
        parameters = cast(_ShapeScale, parameters)
        return parameters.k * parameters.theta**2

    def _support(_: Parametrization) -> ContinuousSupport:
        """Support of gamma distribution"""
        return ContinuousSupport(left=0.0)

    def _base_score(parameters: Parametrization, x: NumericArray) -> NumericArray:
        """
        Gradient of log-PDF with respect to base parameters (k, theta).

        Parameters
        ----------
        parameters : Parametrization
            Base parametrization (shapeScale) with fields k and theta.
        x : NumericArray
            Points at which to evaluate the gradient.

        Returns
        -------
        NumericArray
            Gradient array of shape (..., 2) with partial derivatives
            [d lpdf/dk, d lpdf/dtheta].

        Raises
        ------
        ValueError
            If any x <= 0 or any x is non-finite. The score is undefined at
            the x = 0 boundary and outside support.
        """
        parameters = cast(_ShapeScale, parameters)
        x_arr = np.asarray(x, dtype=np.float64)

        if not np.all(np.isfinite(x_arr) & (x_arr > 0)):
            raise ValueError(
                "base_score is undefined at the x = 0 boundary,"
                " outside support, and for non-finite values."
            )

        k = parameters.k
        theta = parameters.theta

        grad_k = np.log(x_arr) - np.log(theta) - digamma(k)
        grad_theta = (x_arr - k * theta) / (theta**2)

        return np.stack([grad_k, grad_theta], axis=-1)

    Gamma = ParametricFamily(
        name=FamilyName.GAMMA,
        distr_type=UnivariateContinuous,
        distr_parametrizations=["shapeScale", "shapeRate", "meanVar"],
        distr_characteristics={
            CharacteristicName.PDF: pdf,
            CharacteristicName.CDF: cdf,
            CharacteristicName.PPF: ppf,
            CharacteristicName.CF: char_func,
            CharacteristicName.LPDF: lpdf,
            CharacteristicName.MEAN: mean_func,
            CharacteristicName.VAR: var_func,
        },
        support_by_parametrization=_support,
        base_score=_base_score,
    )
    Gamma.__doc__ = GAMMA_DOC

    @parametrization(family=Gamma, name="shapeScale")
    class _ShapeScale(Parametrization):
        """
        Standard parametrization of gamma distribution.

        Parameters
        ----------
        k : float
            Shape of the distribution
        theta : float
            Standard scale of the distribution
        """

        k: float
        theta: float

        @constraint(description="k > 0")
        def check_k_positive(self) -> bool:
            """Check that k > 0."""
            return self.k > 0

        @constraint(description="theta > 0")
        def check_theta_positive(self) -> bool:
            """Check that theta > 0."""
            return self.theta > 0

    @parametrization(family=Gamma, name="meanVar")
    class _MeanVar(Parametrization):
        """
        Mean–variance parametrization of gamma distribution.

        Parameters
        ----------
        m : float
            Mean of the distribution
        v : float
            Variance of the distribution
        """

        m: float
        v: float

        @constraint(description="m > 0")
        def check_m_positive(self) -> bool:
            """Check that m is positive."""
            return self.m > 0

        @constraint(description="v > 0")
        def check_v_positive(self) -> bool:
            """Check that var is positive."""
            return self.v > 0

        def transform_to_base_parametrization(self) -> Parametrization:
            """
            Transform to Standard parametrization.

            Returns
            -------
            Parametrization
                Standard parametrization instance
            """
            return _ShapeScale(k=self.m**2 / self.v, theta=self.v / self.m)

        def gradient_transform(self, grad_base: NumericArray) -> NumericArray:
            """
            Transform gradient from base (k, theta) to (m, v).

            k = m^2 / v,  theta = v / m
            Jacobian:
                dk/dm = 2m / v,   dk/dv = -m^2 / v^2
                d theta/dm = -v / m^2,  d theta/dv = 1 / m
            """
            grad_k = grad_base[..., 0]
            grad_theta = grad_base[..., 1]

            dk_dm = 2 * self.m / self.v
            dk_dv = -(self.m**2) / (self.v**2)
            dtheta_dm = -self.v / (self.m**2)
            dtheta_dv = 1.0 / self.m

            grad_m = grad_k * dk_dm + grad_theta * dtheta_dm
            grad_v = grad_k * dk_dv + grad_theta * dtheta_dv

            return np.stack([grad_m, grad_v], axis=-1)

    @parametrization(family=Gamma, name="shapeRate")
    class _ShapeRate(Parametrization):
        """
        Shape-rate parametrization of gamma distribution.

        Parameters
        ----------
        k : float
            Shape of the distribution
        beta : float
            Rate parameter
        """

        k: float
        beta: float

        @constraint(description="k > 0")
        def check_k_positive(self) -> bool:
            """Check that k > 0."""
            return self.k > 0

        @constraint(description="beta > 0")
        def check_beta_positive(self) -> bool:
            """Check that beta > 0."""
            return self.beta > 0

        def transform_to_base_parametrization(self) -> Parametrization:
            """
            Transform to Standard parametrization.

            Returns
            -------
            Parametrization
                Standard parametrization instance
            """
            theta = 1 / self.beta
            return _ShapeScale(k=self.k, theta=theta)

        def gradient_transform(self, grad_base: NumericArray) -> NumericArray:
            """
            Transform gradient from base (k, theta) to (k, beta).

            Since theta = 1/beta, d theta/d beta = -1/beta^2.
            """
            grad_k = grad_base[..., 0]
            grad_theta = grad_base[..., 1]
            grad_beta = grad_theta * (-1.0 / self.beta**2)
            return np.stack([grad_k, grad_beta], axis=-1)

    ParametricFamilyRegister.register(Gamma)
