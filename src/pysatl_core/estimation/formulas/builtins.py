"""Built-in estimation formulas, separate from distribution family definitions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pysatl_core.estimation.errors import FitDataError
from pysatl_core.estimation.formulas.registry import AnalyticalEstimate, EstimationFormulaRegistry
from pysatl_core.types import FamilyName

if TYPE_CHECKING:
    from pysatl_core.families.parametric_family import ParametricFamily


FLOAT64_TINY = np.finfo(np.float64).tiny


def _normal_mle(sample: NDArray[np.float64], fixed: Mapping[str, float]) -> dict[str, float]:
    """Mean and population standard deviation, allowing either to be fixed.

    Maximum likelihood divides the variance by ``n``; ``n - 1`` belongs to
    the unbiased variance estimate, which optimizes a different criterion.
    """
    mu = float(fixed["mu"]) if "mu" in fixed else float(np.mean(sample))
    sigma = (
        float(fixed["sigma"]) if "sigma" in fixed else float(np.sqrt(np.mean((sample - mu) ** 2)))
    )
    return {"mu": mu, "sigma": sigma}


def normal_mle_with_log_likelihood(
    sample: NDArray[np.float64], fixed: Mapping[str, float]
) -> AnalyticalEstimate:
    """Return the MLE and its log-likelihood.

    As a performance optimization, evaluate the log-likelihood at the MLE
    using the fitted-variance identity instead of summing log-densities over
    the sample again. This simplification applies when ``sigma`` is estimated;
    with fixed ``sigma`` or extreme floating-point scales, use the full
    expression to preserve numerical behavior.
    """
    parameters = _normal_mle(sample, fixed)
    mu, sigma = parameters["mu"], parameters["sigma"]
    if not np.isfinite(sigma) or sigma <= 0.0:
        return AnalyticalEstimate(parameters, -np.inf)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        variance = sigma**2
        if variance == 0.0:
            return AnalyticalEstimate(parameters, -np.inf)
        denominator = 2.0 * variance
        constant = float(-0.5 * np.log(2.0 * np.pi) - np.log(sigma))
        # The fitted-variance identity loses precision at subnormal scales;
        # an overflowing denominator also changes the provider's arithmetic.
        if "sigma" in fixed or variance < FLOAT64_TINY or not np.isfinite(denominator):
            quadratic_terms = ((sample - mu) ** 2) / denominator
            if not np.all(np.isfinite(quadratic_terms)):
                return AnalyticalEstimate(parameters, -np.inf)
            value = float((constant - quadratic_terms).sum())
        else:
            value = sample.size * (constant - 0.5)
    return AnalyticalEstimate(parameters, float(value))


def normal_moment_start(sample: NDArray[np.float64]) -> dict[str, float]:
    return {"mu": float(sample.mean()), "sigma": float(sample.std())}


def _uniform_mle(sample: NDArray[np.float64], fixed: Mapping[str, float]) -> dict[str, float]:
    """The narrowest interval covering the sample, with fixed-endpoint checks.

    The likelihood maximum lies on the support boundary, so a gradient method
    cannot find it reliably. A fixed endpoint excluding data raises
    ``FitDataError`` because every candidate then has zero likelihood.
    """
    low = float(np.min(sample))
    high = float(np.max(sample))
    fixed_low = fixed.get("lower_bound")
    fixed_high = fixed.get("upper_bound")

    if fixed_low is not None and low < fixed_low:
        raise FitDataError(
            f"lower_bound is fixed at {fixed_low}, but the sample reaches down to {low}. "
            "Those observations lie outside every admissible support, so the likelihood "
            "is zero for any upper_bound."
        )
    if fixed_high is not None and high > fixed_high:
        raise FitDataError(
            f"upper_bound is fixed at {fixed_high}, but the sample reaches up to {high}. "
            "Those observations lie outside every admissible support, so the likelihood "
            "is zero for any lower_bound."
        )
    return {
        "lower_bound": low if fixed_low is None else fixed_low,
        "upper_bound": high if fixed_high is None else fixed_high,
    }


def uniform_mle_with_log_likelihood(
    sample: NDArray[np.float64], fixed: Mapping[str, float]
) -> AnalyticalEstimate:
    """Return the MLE and its log-likelihood.

    As a performance optimization, evaluate the log-likelihood at the MLE
    using the simplified expression ``-n * log(width)``. The constant density
    over the fitted interval avoids evaluating the log-density at each
    observation through the generic likelihood evaluator.
    """
    parameters = _uniform_mle(sample, fixed)
    width = parameters["upper_bound"] - parameters["lower_bound"]
    if not np.isfinite(width) or width <= 0.0:
        return AnalyticalEstimate(parameters, -np.inf)
    return AnalyticalEstimate(parameters, float(-sample.size * np.log(width)))


UNIFORM_START_PADDING = 0.05


def uniform_moment_start(sample: NDArray[np.float64]) -> dict[str, float]:
    """Widen the observed interval so the numerical start covers every value."""
    low = float(sample.min())
    high = float(sample.max())
    span = high - low
    pad = UNIFORM_START_PADDING * span if span > 0.0 else max(abs(low), 1.0)
    return {"lower_bound": low - pad, "upper_bound": high + pad}


def _exponential_mle(
    sample: NDArray[np.float64], fixed: Mapping[str, float]
) -> dict[str, float] | None:
    """The reciprocal sample mean in the rate parametrization."""
    if "lambda_" in fixed:
        return None
    mean = float(np.mean(sample))
    if mean <= 0.0:
        raise FitDataError(
            f"The exponential maximum likelihood estimate is 1 / mean(x), but the "
            f"sample mean is {mean}. Every observation sits at the boundary x = 0, "
            "where the likelihood grows without bound as the rate goes to infinity, "
            "so no finite estimate exists."
        )
    return {"lambda_": 1.0 / mean}


def exponential_mle_with_log_likelihood(
    sample: NDArray[np.float64], fixed: Mapping[str, float]
) -> AnalyticalEstimate | None:
    """Return the MLE and its log-likelihood, or ``None`` for a fixed rate.

    As a performance optimization, evaluate the log-likelihood at the MLE
    using the simplified expression ``n * (log(rate) - 1)``. At the fitted
    rate, ``rate * sum(sample) == n``, so the log-likelihood can reuse the
    estimate without another pass through the sample.
    """
    parameters = _exponential_mle(sample, fixed)
    if parameters is None:
        return None
    rate = parameters["lambda_"]
    if not np.isfinite(rate) or rate <= 0.0:
        return AnalyticalEstimate(parameters, -np.inf)
    return AnalyticalEstimate(parameters, float(sample.size * (np.log(rate) - 1.0)))


def exponential_moment_start(sample: NDArray[np.float64]) -> dict[str, float]:
    mean = float(sample.mean())
    return {"lambda_": 1.0 / mean if mean != 0.0 else 1.0}


def gamma_moment_start(sample: NDArray[np.float64]) -> dict[str, float]:
    """Moment estimates of shape and scale, with a safe degenerate start."""
    mean = float(sample.mean())
    var = float(sample.var())
    if mean <= 0.0 or var <= 0.0:
        return {"k": 1.0, "theta": 1.0}
    return {"k": mean * mean / var, "theta": var / mean}


def register_builtin_estimation_formulas(families: Mapping[FamilyName, ParametricFamily]) -> None:
    """Attach built-in rules to newly created family objects."""
    from pysatl_core.estimation.methods.mle import MLE_NAME

    normal = families.get(FamilyName.NORMAL)
    if normal is not None:
        EstimationFormulaRegistry.register_closed_form(
            normal, MLE_NAME, normal_mle_with_log_likelihood
        )
        EstimationFormulaRegistry.register_moment_start(normal, normal_moment_start)

    uniform = families.get(FamilyName.CONTINUOUS_UNIFORM)
    if uniform is not None:
        EstimationFormulaRegistry.register_closed_form(
            uniform, MLE_NAME, uniform_mle_with_log_likelihood
        )
        EstimationFormulaRegistry.register_moment_start(uniform, uniform_moment_start)

    exponential = families.get(FamilyName.EXPONENTIAL)
    if exponential is not None:
        EstimationFormulaRegistry.register_closed_form(
            exponential, MLE_NAME, exponential_mle_with_log_likelihood
        )
        EstimationFormulaRegistry.register_moment_start(exponential, exponential_moment_start)

    gamma = families.get(FamilyName.GAMMA)
    if gamma is not None:
        EstimationFormulaRegistry.register_moment_start(gamma, gamma_moment_start)


__all__ = ["register_builtin_estimation_formulas"]
