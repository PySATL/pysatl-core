"""
Uniform distribution family implementation.

Contains the Uniform family with multiple parameterizations.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping
from typing import Any, cast

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


def configure_uniform_family() -> None:
    """
    Configure and register the Uniform distribution family.
    """

    if ParametricFamilyRegister.contains(FamilyName.CONTINUOUS_UNIFORM):
        return

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

    def lpdf(parameters: Parametrization, x: NumericArray) -> NumericArray:
        """
        Logarithm of the probability density function for uniform distribution.

        Parameters
        ----------
        parameters : Parametrization
            Distribution parameters object with fields:
            - lower_bound: float (lower bound)
            - upper_bound: float (upper bound)
        x : NumericArray
            Points at which to evaluate the log-probability density function

        Returns
        -------
        NumericArray
            Log-probability density values at points x
            For x outside [lower_bound, upper_bound] returns -np.inf
        """
        parameters = cast(_Standard, parameters)
        a = parameters.lower_bound
        b = parameters.upper_bound
        return np.where((x >= a) & (x <= b), -np.log(b - a), -np.inf)

    def mean_func(parameters: Parametrization) -> float:
        """Mean of uniform distribution."""
        parameters = cast(_Standard, parameters)
        return (parameters.lower_bound + parameters.upper_bound) / 2

    def var_func(parameters: Parametrization) -> float:
        """Variance of uniform distribution."""
        parameters = cast(_Standard, parameters)
        width = parameters.upper_bound - parameters.lower_bound
        return width**2 / 12

    def skew_func() -> int:
        """Skewness of uniform distribution (always 0)."""
        return 0

    def kurt_func(*, excess: bool = False) -> float:
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

    def _base_score(parameters: Parametrization, x: NumericArray) -> NumericArray:
        """
        Compute the score (gradient of log‑PDF) for the base
        parametrization (lower_bound, upper_bound).

        For uniform distribution on [a, b] with a < b, log‑PDF is:
            log f(x) = -log(b - a)   for a ≤ x ≤ b, else -∞.

        The gradient with respect to a and b (where defined, i.e., inside the support) is:
            ∂/∂a log f =  1/(b - a)
            ∂/∂b log f = -1/(b - a)

        For points outside the support or non‑finite (inf, nan), raises ValueError.

        Parameters
        ----------
        parameters : Parametrization
            Base parametrization instance (_Standard) with fields lower_bound, upper_bound.
        x : NumericArray
            Points at which to evaluate the gradient.

        Returns
        -------
        NumericArray
            Gradient array of shape (..., 2) where last axis corresponds to
            [d(log f)/d(lower_bound), d(log f)/d(upper_bound)].

        Raises
        ------
        ValueError
            If any element of x lies outside [lower_bound, upper_bound]
            or is non‑finite (inf or nan).
        """
        params = cast(_Standard, parameters)
        a = params.lower_bound
        b = params.upper_bound

        # Reject non‑finite values before support check
        if np.any(~np.isfinite(x)):
            raise ValueError(f"Score is undefined for non‑finite x (outside support). Got x = {x}")
        if np.any((x < a) | (x > b)):
            raise ValueError(f"Score is undefined for x outside support [{a}, {b}]. Got x = {x}")

        width = b - a
        grad_a = np.full_like(x, 1.0 / width, dtype=np.float64)
        grad_b = np.full_like(x, -1.0 / width, dtype=np.float64)
        return np.stack([grad_a, grad_b], axis=-1)

    def _mle_uniform(sample: NumericArray, fixed: Mapping[str, Any]) -> Parametrization | None:
        """
        Closed-form maximum likelihood estimate for the uniform family.

        The estimate is the observed range::

            a_hat = min(x),   b_hat = max(x)

        The likelihood ``(b - a)^-n`` is decreasing in the width, so it is
        maximised by the narrowest interval that still covers every
        observation.  Note that the maximum lies *on the boundary* of the
        admissible region rather than at a stationary point: the score is the
        constant ``[1/(b-a), -1/(b-a)]`` and never vanishes, so there is no
        equation to solve and a gradient method has nothing to converge to.
        That is why this formula is not a convenience but the only reliable
        route for this family.

        With one endpoint fixed through
        :meth:`~pysatl_core.families.parametric_family.ParametricFamily.view`,
        the other is still the corresponding extreme of the sample — provided
        the fixed endpoint does not already exclude part of the data.

        Parameters
        ----------
        sample : NumericArray
            Observed values.
        fixed : Mapping[str, Any]
            Parameters pinned by a view, in the base parametrization.

        Returns
        -------
        Parametrization
            Estimated parameters in the full base parametrization.

        Raises
        ------
        FitDataError
            If a fixed endpoint leaves observations outside the interval. No
            value of the free endpoint can rescue that: the likelihood is zero
            for every candidate.
        """
        from pysatl_core.estimation.errors import FitDataError

        observed_low = float(np.min(sample))
        observed_high = float(np.max(sample))

        fixed_low = fixed.get("lower_bound")
        fixed_high = fixed.get("upper_bound")

        if fixed_low is not None and observed_low < float(fixed_low):
            raise FitDataError(
                f"lower_bound is fixed at {float(fixed_low)}, but the sample reaches down to "
                f"{observed_low}. Those observations lie outside every admissible support, so "
                f"the likelihood is zero for any upper_bound."
            )
        if fixed_high is not None and observed_high > float(fixed_high):
            raise FitDataError(
                f"upper_bound is fixed at {float(fixed_high)}, but the sample reaches up to "
                f"{observed_high}. Those observations lie outside every admissible support, so "
                f"the likelihood is zero for any lower_bound."
            )

        return _Standard(
            lower_bound=observed_low if fixed_low is None else float(fixed_low),
            upper_bound=observed_high if fixed_high is None else float(fixed_high),
        )

    UNIFORM_START_PADDING = 0.05
    """Fraction of the observed range by which the start is widened.

    The start has to cover the sample: an interval that excludes even one
    observation makes the objective a constant wall of penalty, with no
    gradient and no simplex direction leading out of it.  Padding outwards is
    the same trick SciPy plays in ``_fit_loc_scale_support``.
    """

    def _moment_start(sample: NumericArray) -> dict[str, float]:
        """Moment start for the uniform family: the observed range, padded outwards."""
        low = float(sample.min())
        high = float(sample.max())
        span = high - low
        pad = UNIFORM_START_PADDING * span if span > 0.0 else max(abs(low), 1.0)
        return {"lower_bound": low - pad, "upper_bound": high + pad}

    # The key names the method this formula solves; taken from the method
    # itself so that the two spellings cannot drift apart.
    from pysatl_core.estimation.methods.mle import MLE_NAME

    Uniform = ParametricFamily(
        name=FamilyName.CONTINUOUS_UNIFORM,
        distr_type=UnivariateContinuous,
        distr_parametrizations=["standard", "meanWidth", "minRange"],
        distr_characteristics={
            CharacteristicName.PDF: pdf,
            CharacteristicName.CDF: cdf,
            CharacteristicName.PPF: ppf,
            CharacteristicName.CF: char_func,
            CharacteristicName.LPDF: lpdf,
            CharacteristicName.MEAN: mean_func,
            CharacteristicName.VAR: var_func,
            CharacteristicName.SKEW: skew_func,
            CharacteristicName.KURT: kurt_func,
        },
        support_by_parametrization=_support,
        base_score=_base_score,
        closed_forms={MLE_NAME: _mle_uniform},
        moment_start=_moment_start,
        # No 'param_bounds': the only restriction on these parameters is the
        # relation 'lower_bound < upper_bound', and a box of per-parameter
        # bounds cannot express a relation between two parameters. Each
        # endpoint on its own ranges over the whole line. The family is fitted
        # by the closed-form solution above in any case.
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

        def gradient_transform(self, grad_base: NumericArray) -> NumericArray:
            """
            Transform gradient from base parameters (a, b) to (mean, width).

            The transformation is:
                mean = (a + b) / 2
                width = b - a

            The Jacobian matrix:
                d(mean)/da = 0.5,   d(mean)/db = 0.5
                d(width)/da = -1,    d(width)/db = 1

            Hence:
                grad_mean = 0.5 * (grad_a + grad_b)
                grad_width = -grad_a + grad_b

            Parameters
            ----------
            grad_base : NumericArray
                Gradient with respect to (a, b), shape (..., 2).

            Returns
            -------
            NumericArray
                Gradient with respect to (mean, width), shape (..., 2).
            """
            grad_a = grad_base[..., 0]
            grad_b = grad_base[..., 1]
            grad_mean = 0.5 * (grad_a + grad_b)
            grad_width = -grad_a + grad_b
            return np.stack([grad_mean, grad_width], axis=-1)

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

        def gradient_transform(self, grad_base: NumericArray) -> NumericArray:
            """
            Transform gradient from base parameters (a, b) to (minimum, range_val).

            The transformation is:
                minimum = a
                range_val = b - a

            The Jacobian matrix:
                d(minimum)/da = 1,    d(minimum)/db = 0
                d(range_val)/da = -1,  d(range_val)/db = 1

            Hence:
                grad_minimum = grad_a
                grad_range = -grad_a + grad_b

            Parameters
            ----------
            grad_base : NumericArray
                Gradient with respect to (a, b), shape (..., 2).

            Returns
            -------
            NumericArray
                Gradient with respect to (minimum, range_val), shape (..., 2).
            """
            grad_a = grad_base[..., 0]
            grad_b = grad_base[..., 1]
            grad_minimum = grad_a
            grad_range = -grad_a + grad_b
            return np.stack([grad_minimum, grad_range], axis=-1)

    ParametricFamilyRegister.register(Uniform)
