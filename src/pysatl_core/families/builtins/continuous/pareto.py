"""
Pareto distribution family implementation.

Contains the Pareto (type I) family with the scale-shape parametrization.
"""

from __future__ import annotations

__author__ = "Duan Nguyen"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import math
from typing import cast

import numpy as np
from scipy import integrate

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


def configure_pareto_family() -> None:
    """
    Configure and register the Pareto distribution family.
    """

    if ParametricFamilyRegister.contains(FamilyName.PARETO):
        return

    PARETO_DOC = """
    Pareto distribution (type I).

    The Pareto distribution is a heavy-tailed continuous distribution with
    two parameters: the scale x_m, which is the lower bound of the support,
    and the shape alpha, which is the tail index.

    Probability density function:
        f(x) = alpha * x_m^alpha / x^(alpha + 1) for x >= x_m, 0 otherwise

    The support is [x_m, +inf) and depends on the parameters. The moment of
    order n exists only for alpha > n; a moment that does not exist is +inf.

    The Lomax distribution (Pareto II) and the other Pareto types are separate
    families.
    """

    def lpdf(parameters: Parametrization, x: NumericArray) -> NumericArray:
        """
        Logarithm of the probability density function for Pareto distribution.
            - For x < x_m: returns -inf
            - Otherwise: returns ln(alpha) + alpha * ln(x_m) - (alpha + 1) * ln(x)

        Parameters
        ----------
        parameters : Parametrization
            Distribution parameters object with fields:
            - x_m: float (scale, lower bound of the support)
            - alpha: float (shape, tail index)
        x : NumericArray
            Points at which to evaluate the log-probability density function

        Returns
        -------
        NumericArray
            Log-probability density values at points x
        """
        parameters = cast(_ScaleShape, parameters)
        x_m = parameters.x_m
        alpha = parameters.alpha

        x_arr = np.asarray(x, dtype=np.float64)
        inside = x_arr >= x_m
        # Points outside the support are replaced by x_m, so that no log of a negative
        # number or of zero is evaluated; their result is overwritten below.
        safe_x = np.where(inside, x_arr, x_m)

        with np.errstate(over="ignore", invalid="ignore"):
            value = math.log(alpha) + alpha * math.log(x_m) - (alpha + 1.0) * np.log(safe_x)

        return cast(
            NumericArray,
            np.where(inside, value, np.where(np.isnan(x_arr), np.nan, -np.inf)),
        )

    def pdf(parameters: Parametrization, x: NumericArray) -> NumericArray:
        """
        Probability density function for Pareto distribution.
            - For x < x_m: returns 0
            - Otherwise: returns alpha * x_m^alpha / x^(alpha + 1)

        The density is evaluated as exp(lpdf), so x_m^alpha and x^(alpha + 1)
        are never formed and cannot overflow.

        Parameters
        ----------
        parameters : Parametrization
            Distribution parameters object with fields:
            - x_m: float (scale, lower bound of the support)
            - alpha: float (shape, tail index)
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
        Cumulative distribution function for Pareto distribution.
            - For x < x_m: returns 0
            - Otherwise: returns 1 - (x_m / x)^alpha

        Parameters
        ----------
        parameters : Parametrization
            Distribution parameters object with fields:
            - x_m: float (scale, lower bound of the support)
            - alpha: float (shape, tail index)
        x : NumericArray
            Points at which to evaluate the cumulative distribution function

        Returns
        -------
        NumericArray
            Probabilities P(X <= x) for each point x
        """
        parameters = cast(_ScaleShape, parameters)
        x_m = parameters.x_m
        alpha = parameters.alpha

        x_arr = np.asarray(x, dtype=np.float64)
        inside = x_arr >= x_m
        safe_x = np.where(inside, x_arr, x_m)

        # 1 - (x_m / x)^alpha written with expm1 keeps the precision close to x_m.
        value = -np.expm1(alpha * (math.log(x_m) - np.log(safe_x)))

        return cast(
            NumericArray,
            np.where(inside, value, np.where(np.isnan(x_arr), np.nan, 0.0)),
        )

    def ppf(parameters: Parametrization, p: NumericArray) -> NumericArray:
        """
        Percent point function (inverse CDF) for Pareto distribution.

        For 0 <= p <= 1:
            F^-1(p) = x_m * (1 - p)^(-1 / alpha)
        with F^-1(0) = x_m and F^-1(1) = +inf.

        Parameters
        ----------
        parameters : Parametrization
            Distribution parameters object with fields:
            - x_m: float (scale, lower bound of the support)
            - alpha: float (shape, tail index)
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
        p_arr = np.asarray(p, dtype=np.float64)
        if np.any((p_arr < 0) | (p_arr > 1)):
            raise ValueError("Probability must be in [0, 1]")

        parameters = cast(_ScaleShape, parameters)

        # log1p(-p) is -inf at p = 1, which gives the +inf quantile without a warning.
        with np.errstate(divide="ignore"):
            return cast(
                NumericArray,
                parameters.x_m * np.exp(-np.log1p(-p_arr) / parameters.alpha),
            )

    def char_func(parameters: Parametrization, t: NumericArray) -> ComplexArray:
        """
        Characteristic function of Pareto distribution.

        The characteristic function has no elementary closed form, so it is
        evaluated from the integral definition:
            phi(t) = integral_{x_m}^{inf} exp(i t x) f(x) dx
        which, with u = x / x_m and y = t * x_m, is
            phi(t) = alpha * integral_1^{inf} u^(-alpha - 1) exp(i y u) du.

        The oscillatory integrand is not integrated as it stands. For y > 0 the
        path u = 1 + i s is rotated into the complex plane, which gives an
        absolutely convergent integral with an exponentially decaying factor:
            phi(t) = i * alpha * exp(i y) * integral_0^{inf} (1 + i s)^(-alpha - 1) exp(-y s) ds.
        For large y the variable is rescaled to s = r / y, so that the integrand
        stays smooth. phi(0) = 1 and phi(-t) = conj(phi(t)).

        Parameters
        ----------
        parameters : Parametrization
            Distribution parameters object with fields:
            - x_m: float (scale, lower bound of the support)
            - alpha: float (shape, tail index)
        t : NumericArray
            Points at which to evaluate the characteristic function

        Returns
        -------
        ComplexArray
            Characteristic function values at points t
        """
        parameters = cast(_ScaleShape, parameters)
        x_m = parameters.x_m
        alpha = parameters.alpha

        def scalar_cf(t_value: float) -> complex:
            if t_value == 0.0:
                return complex(1.0, 0.0)
            y = abs(t_value) * x_m
            scale = max(y, 1.0)

            def integrand(r: float) -> complex:
                s = r / scale
                return complex((1.0 + 1j * s) ** (-alpha - 1.0) * math.exp(-y * s) / scale)

            integral, _ = integrate.quad(
                integrand, 0.0, np.inf, complex_func=True, epsabs=1e-12, epsrel=1e-12
            )
            value = 1j * alpha * np.exp(1j * y) * integral
            return complex(value if t_value > 0.0 else value.conjugate())

        t_arr = np.asarray(t, dtype=np.float64)
        result = np.empty(t_arr.shape, dtype=np.complex128)
        for index in np.ndindex(t_arr.shape):
            result[index] = scalar_cf(float(t_arr[index]))
        return cast(ComplexArray, result)

    def mean_func(parameters: Parametrization) -> float:
        """Mean of Pareto distribution: alpha * x_m / (alpha - 1) for alpha > 1, else +inf."""
        parameters = cast(_ScaleShape, parameters)
        if parameters.alpha <= 1.0:
            return math.inf
        return parameters.alpha * parameters.x_m / (parameters.alpha - 1.0)

    def var_func(parameters: Parametrization) -> float:
        """
        Variance of Pareto distribution.

        x_m^2 * alpha / ((alpha - 1)^2 * (alpha - 2)) for alpha > 2, else +inf.
        """
        parameters = cast(_ScaleShape, parameters)
        alpha = parameters.alpha
        if alpha <= 2.0:
            return math.inf
        return parameters.x_m**2 * alpha / ((alpha - 1.0) ** 2 * (alpha - 2.0))

    def _support(parameters: Parametrization) -> ContinuousSupport:
        """Support of Pareto distribution: [x_m, +inf)."""
        parameters = cast(_ScaleShape, parameters.transform_to_base_parametrization())
        return ContinuousSupport(left=parameters.x_m, left_closed=True)

    def _base_score(parameters: Parametrization, x: NumericArray) -> NumericArray:
        """
        Compute the score (gradient of log-PDF) for the base parametrization (x_m, alpha).

        For x >= x_m the log-PDF is:
            ln f(x) = ln(alpha) + alpha * ln(x_m) - (alpha + 1) * ln(x)

        and its gradient is:
            d ln f / d x_m   = alpha / x_m
            d ln f / d alpha = 1 / alpha + ln(x_m) - ln(x)

        Parameters
        ----------
        parameters : Parametrization
            Base parametrization instance (_ScaleShape) with fields x_m and alpha.
        x : NumericArray
            Points at which to evaluate the gradient.

        Returns
        -------
        NumericArray
            Gradient array of shape (..., 2) where the last axis corresponds to
            [d(ln f)/d(x_m), d(ln f)/d(alpha)].

        Raises
        ------
        ValueError
            If any element of x lies outside [x_m, +inf) or is non-finite (inf or nan).
        """
        params = cast(_ScaleShape, parameters)
        x_m = params.x_m
        alpha = params.alpha

        x_arr = np.asarray(x, dtype=np.float64)

        # Reject non-finite values before the support check
        if np.any(~np.isfinite(x_arr)):
            raise ValueError(f"Score is undefined for non-finite x (outside support). Got x = {x}")
        if np.any(x_arr < x_m):
            raise ValueError(f"Score is undefined for x outside support [{x_m}, inf). Got x = {x}")

        grad_x_m = np.full_like(x_arr, alpha / x_m, dtype=np.float64)
        grad_alpha = 1.0 / alpha + math.log(x_m) - np.log(x_arr)
        return np.stack([grad_x_m, grad_alpha], axis=-1)

    Pareto = ParametricFamily(
        name=FamilyName.PARETO,
        distr_type=UnivariateContinuous,
        distr_parametrizations=["scaleShape"],
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
    Pareto.__doc__ = PARETO_DOC

    @parametrization(family=Pareto, name="scaleShape")
    class _ScaleShape(Parametrization):
        """
        Scale-shape parametrization of Pareto distribution.

        Parameters
        ----------
        x_m : float
            Scale: the lower bound of the support
        alpha : float
            Shape: the tail index
        """

        x_m: float
        alpha: float

        @constraint(description="x_m > 0")
        def check_x_m_positive(self) -> bool:
            """Check that the scale is positive."""
            return self.x_m > 0

        @constraint(description="alpha > 0")
        def check_alpha_positive(self) -> bool:
            """Check that the shape is positive."""
            return self.alpha > 0

    ParametricFamilyRegister.register(Pareto)
