"""
Numerical Fitters for Probability Distribution Conversions
==========================================================

Numerical fitters that construct scalar conversions between univariate
probability distribution characteristics, both continuous and discrete.

Supported characteristics
-------------------------
- PDF (probability density function)
- CDF (cumulative distribution function)
- PPF (percent-point / quantile function)
- PMF (probability mass function)

Continuous conversions
----------------------
- ``pdf → cdf`` via numerical integration;
- ``cdf → pdf`` via numerical differentiation;
- ``cdf → ppf`` via numerical inversion (bracketing + bisection);
- ``ppf → cdf`` via numerical inversion and interpolation;
- ``ppf → pdf`` via implicit differentiation.

Discrete conversions
--------------------
- ``cdf → pmf`` via finite differencing;
- ``pmf → cdf`` via cumulative summation;
- ``pmf → ppf`` and ``ppf → pmf`` via functional composition.

The returned objects are instances of
:class:`~pysatl_core.distributions.computation.FittedComputationMethod`,
wrapping vectorized scalar callables.

Notes
-----
- NumPy is used for vectorization and numerical operations.
- SciPy is used for adaptive numerical integration and interpolation.
- Small numerical artefacts (e.g., negative densities due to finite
  differencing) are clipped where appropriate to preserve validity.
"""

__author__ = "Leonid Elkin, Mikhail Mikhailov, Nikishin Vladimir"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from enum import Enum
from typing import TYPE_CHECKING, Any, TypedDict, Callable
from scipy import interpolate, integrate

import numpy as np

from pysatl_core.distributions.computation import FittedComputationMethod
from pysatl_core.types import GenericCharacteristicName, ScalarFunc

if TYPE_CHECKING:
    from pysatl_core.distributions.distribution import Distribution


# ---------------------------------------------------------------------
# Characteristic names
PDF = "pdf"
CDF = "cdf"
PMF = "pmf"
PPF = "ppf"
SF = "sf"
MEAN = "mean"
CF = "cf"
VAR = "var"
LOG_LIKELIHOOD = "log_likelihood"


# ---------------------------------------------------------------------
# Utility helpers


def _get_scalar_characteristic(
    distribution: "Distribution", name: GenericCharacteristicName
) -> ScalarFunc:
    """
    Retrieve a scalar characteristic function from a distribution.

    Parameters
    ----------
    distribution : Distribution
        Distribution providing a computation strategy.
    name : str
        Name of the requested characteristic (e.g., PDF, CDF, PPF).

    Returns
    -------
    ScalarFunc
        Scalar callable implementing the requested characteristic.

    Raises
    ------
    RuntimeError
        If the distribution does not provide a compatible computation
        strategy.
    """
    try:
        scalar_characteristic = distribution.computation_strategy.query_method(
            name, distribution
        )
    except AttributeError as e:
        raise RuntimeError(
            "Distribution must provide computation_strategy.query_method(name, distribution)."
        ) from e

    def _wrap(x: float) -> float:
        return float(scalar_characteristic(x))

    return _wrap


def apply_scalar_func(func: ScalarFunc, x_arr: np.ndarray) -> np.ndarray:
    """
    Apply a scalar function elementwise to a NumPy array.

    This is a safe wrapper around ``np.vectorize`` that ensures the
    provided callable follows the ``float -> float`` contract.

    Parameters
    ----------
    func : ScalarFunc
        Scalar callable to apply.
    x_arr : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Array of function values with the same shape as ``x_arr``.
    """
    return np.vectorize(func)(x_arr)


class FunctionalDistribution:
    """
    Lightweight distribution wrapper built from scalar characteristic functions.

    This class allows constructing a minimal distribution-like object
    from one or more scalar callables (e.g., CDF, PDF), enabling reuse
    of existing fitters without implementing a full Distribution
    interface.
    """

    def __init__(self, **funcs: ScalarFunc):
        self.computation_strategy = None
        self._funcs = funcs

    def __getattr__(self, name: str):
        if name in self._funcs:
            return self._funcs[name]
        raise AttributeError(name)


# ---------------------------------------------------------------------
# Enums


class IntegrationMethod(Enum):
    QUAD = "quad"  # Adaptive SciPy integration
    LEFT_RECTANGLE = "left_rectangle"  # Left rectangle rule
    RIGHT_RECTANGLE = "right_rectangle"


class DifferentiationMethod(Enum):
    CENTRAL_3POINT = "central_3point"  # Central difference (O(h^2))
    FORWARD_2POINT = "forward_2point"  # Forward difference
    BACKWARD_2POINT = "backward_2point"  # Backward difference


# ---------------------------------------------------------------------
# Parameter schemas


class PDFtoCDFParams(TypedDict, total=False):
    integration_method: IntegrationMethod
    rel_tol: float
    abs_tol: float
    limit: int
    points: list[float]
    step_size: float
    support_cutoff: float


class PDFtoPPFParams(TypedDict, total=False):
    cdf_params: PDFtoCDFParams
    ppf_params: "CDFtoPPFParams"


class CDFtoPDFParams(TypedDict, total=False):
    differentiation_method: DifferentiationMethod
    h: float
    min_h: float
    max_h: float


class CDFtoPMFParams(TypedDict, total=False):
    epsilon: float


class CDFtoPPFParams(TypedDict, total=False):
    abs_tol: float
    max_iter: int
    left_bound: float
    right_bound: float
    expand_iter: int
    expand_factor: float


class PMFtoCDFParams(TypedDict, total=False):
    support_min: int
    support_max: int
    epsilon: float


class PMFtoPPFParams(TypedDict, total=False):
    cdf_params: PMFtoCDFParams
    ppf_params: CDFtoPPFParams


class PPFtoPDFParams(TypedDict, total=False):
    cdf_params: "PPFtoCDFParams"
    h: float
    min_pdf: float


class PPFtoCDFParams(TypedDict, total=False):
    p_min: float
    p_max: float
    initial_samples: int


class PPFtoPMFParams(TypedDict, total=False):
    cdf_params: PPFtoCDFParams
    pmf_params: CDFtoPMFParams


# ---------------------------------------------------------------------
# Fitters


def fit_pdf_to_cdf(
    distribution: "Distribution", params: dict[str, Any]
) -> FittedComputationMethod[float, float]:
    """
    Construct a numerical approximation of the CDF from a PDF.

    The CDF is computed as the integral of the PDF using either adaptive
    SciPy integration or fixed-step rectangle rules.

    Parameters
    ----------
    distribution : Distribution
        Distribution providing a scalar PDF.
    params : dict
        Integration parameters (method, tolerances, support cutoff).

    Returns
    -------
    FittedComputationMethod
        Vectorized callable approximating the CDF.
    """

    pdf_func = _get_scalar_characteristic(distribution, PDF)

    default_params: PDFtoCDFParams = {
        "integration_method": IntegrationMethod.QUAD,
        "rel_tol": 1.49e-8,
        "abs_tol": 1.49e-8,
        "limit": 50,
        "points": [],
        "step_size": 0.01,
        "support_cutoff": 50.0,
    }

    config = {**default_params, **params}

    def compute_cdf(x_arr: np.ndarray) -> np.ndarray:
        method = config["integration_method"]

        if method == IntegrationMethod.QUAD:
            return apply_scalar_func(
                lambda x: _quad_single(pdf_func, x, config),
                x_arr,
            )

        if method in (
            IntegrationMethod.LEFT_RECTANGLE,
            IntegrationMethod.RIGHT_RECTANGLE,
        ):
            left = method == IntegrationMethod.LEFT_RECTANGLE
            return _rectangle_vectorized(pdf_func, x_arr, config, left)

        raise ValueError(f"Unknown integration method: {method}")

    return FittedComputationMethod(target=CDF, source=PDF, func=compute_cdf)


def _quad_single(pdf_func: ScalarFunc, x: float, config: dict) -> float:
    """
    Compute single CDF value using SciPy quad.
    """
    cutoff = config["support_cutoff"]

    if x <= -cutoff:
        return 0.0
    if x >= cutoff:
        return 1.0

    res, _ = integrate.quad(
        pdf_func,
        -np.inf,
        x,
        epsabs=config["abs_tol"],
        epsrel=config["rel_tol"],
        limit=config["limit"],
        points=config["points"],
    )
    return float(np.clip(res, 0.0, 1.0))


def _rectangle_vectorized(
    pdf_func: ScalarFunc,
    x_arr: np.ndarray,
    config: dict,
    left: bool,
) -> np.ndarray:
    """
    Rectangle integration method on a fixed grid.
    """
    cutoff = config["support_cutoff"]
    step = config["step_size"]

    result = np.zeros_like(x_arr, dtype=float)

    mask_low = x_arr <= -cutoff
    mask_high = x_arr >= cutoff
    result[mask_low] = 0.0
    result[mask_high] = 1.0

    mask = ~(mask_low | mask_high)
    if not np.any(mask):
        return result

    grid = np.linspace(-cutoff, cutoff, int(2 * cutoff / step) + 1)
    pdf_vals = apply_scalar_func(pdf_func, grid)

    cdf_vals = np.zeros_like(grid)
    if left:
        cdf_vals[1:] = np.cumsum(pdf_vals[:-1]) * step
    else:
        cdf_vals[1:] = np.cumsum(pdf_vals[1:]) * step

    cdf_vals = np.clip(cdf_vals, 0.0, 1.0)

    interp = interpolate.interp1d(
        grid,
        cdf_vals,
        kind="linear",
        bounds_error=False,
        fill_value=(0.0, 1.0),
        assume_sorted=True,
    )

    result[mask] = interp(x_arr[mask])
    return result


def fit_pdf_to_ppf(
    distribution: "Distribution", params: dict[str, Any]
) -> FittedComputationMethod[float, float]:
    """
    Construct a PPF from a PDF via composition: PDF → CDF → PPF.

    This method first numerically integrates the PDF to obtain a CDF,
    then inverts the CDF to compute quantiles.

    Parameters
    ----------
    distribution : Distribution
        Distribution providing a scalar PDF.
    params : dict
        Parameters for CDF construction and inversion.

    Returns
    -------
    FittedComputationMethod
        Vectorized callable approximating the PPF.
    """

    cdf_comp = fit_pdf_to_cdf(distribution, params.get("cdf_params", {}))

    def cdf_scalar(x: float) -> float:
        return float(cdf_comp.func(np.array([x]))[0])

    temp_dist = FunctionalDistribution(cdf=cdf_scalar)
    ppf_comp = fit_cdf_to_ppf(temp_dist, params.get("ppf_params", {}))

    return FittedComputationMethod(target=PPF, source=PDF, func=ppf_comp.func)


def fit_cdf_to_pdf(
    distribution: "Distribution", params: dict[str, Any]
) -> FittedComputationMethod[float, float]:
    """
    Approximate the PDF by numerically differentiating the CDF.

    Several finite-difference schemes are supported, with configurable
    step size and bounds for numerical stability.

    Parameters
    ----------
    distribution : Distribution
        Distribution providing a scalar CDF.
    params : dict
        Differentiation parameters.

    Returns
    -------
    FittedComputationMethod
        Vectorized callable approximating the PDF.
    """

    cdf_func = _get_scalar_characteristic(distribution, CDF)

    default_params: CDFtoPDFParams = {
        "differentiation_method": DifferentiationMethod.CENTRAL_3POINT,
        "h": 1e-5,
        "min_h": 1e-12,
        "max_h": 1e-3,
    }
    config = {**default_params, **params}

    def compute_pdf(x_arr: np.ndarray) -> np.ndarray:
        h = float(np.clip(config["h"], config["min_h"], config["max_h"]))

        if config["differentiation_method"] == DifferentiationMethod.CENTRAL_3POINT:
            return (
                apply_scalar_func(cdf_func, x_arr + h)
                - apply_scalar_func(cdf_func, x_arr - h)
            ) / (2 * h)

        if config["differentiation_method"] == DifferentiationMethod.FORWARD_2POINT:
            return (
                apply_scalar_func(cdf_func, x_arr + h)
                - apply_scalar_func(cdf_func, x_arr)
            ) / h

        if config["differentiation_method"] == DifferentiationMethod.BACKWARD_2POINT:
            return (
                apply_scalar_func(cdf_func, x_arr)
                - apply_scalar_func(cdf_func, x_arr - h)
            ) / h

        raise ValueError(
            f"Unknown differentiation method: {config['differentiation_method']}"
        )

    return FittedComputationMethod(target=PDF, source=CDF, func=compute_pdf)


def fit_cdf_to_pmf(
    distribution: "Distribution", params: dict[str, Any]
) -> FittedComputationMethod[int, float]:
    """
    Compute a discrete PMF from a CDF using finite differencing.

    The PMF at integer k is approximated as:
        CDF(k + ε) − CDF(k − ε)

    Parameters
    ----------
    distribution : Distribution
        Distribution providing a scalar CDF.
    params : dict
        Finite difference parameters.

    Returns
    -------
    FittedComputationMethod
        Vectorized callable computing the PMF.
    """

    cdf_func = _get_scalar_characteristic(distribution, CDF)

    default_params: CDFtoPMFParams = {
        "epsilon": 1e-8,
    }
    config = {**default_params, **params}
    eps = config["epsilon"]

    def compute_pmf(k_arr: np.ndarray) -> np.ndarray:
        k_arr = k_arr.astype(int)
        return np.array(
            [max(cdf_func(k + eps) - cdf_func(k - eps), 0.0) for k in k_arr]
        )

    return FittedComputationMethod(target=PMF, source=CDF, func=compute_pmf)


def fit_cdf_to_ppf(
    distribution: "Distribution", params: dict[str, Any]
) -> FittedComputationMethod[float, float]:
    """
    Compute the PPF (quantile function) by numerically inverting the CDF.

    For each probability p, the corresponding quantile x is found via
    bracketing and bisection.

    Parameters
    ----------
    distribution : Distribution
        Distribution providing a scalar CDF.
    params : dict
        Root-finding and tolerance parameters.

    Returns
    -------
    FittedComputationMethod
        Vectorized callable approximating the PPF.
    """

    cdf_func = _get_scalar_characteristic(distribution, CDF)

    default_params: CDFtoPPFParams = {
        "abs_tol": 1e-6,
        "max_iter": 100,
        "left_bound": -10.0,
        "right_bound": 10.0,
        "expand_iter": 20,
        "expand_factor": 2.0,
    }
    config = {**default_params, **params}

    def _ppf_single(p: float) -> float:
        if p <= 0.0:
            return -np.inf
        if p >= 1.0:
            return np.inf

        left = config["left_bound"]
        right = config["right_bound"]

        # Expand bounds if needed
        for _ in range(config["expand_iter"]):
            if cdf_func(left) > p:
                right = left
                left *= config["expand_factor"]
            elif cdf_func(right) < p:
                left = right
                right *= config["expand_factor"]
            else:
                break

        for _ in range(config["max_iter"]):
            mid = 0.5 * (left + right)
            val = cdf_func(mid)

            if abs(val - p) < config["abs_tol"]:
                return mid
            if val < p:
                left = mid
            else:
                right = mid

        return 0.5 * (left + right)

    def compute_ppf(p_arr: np.ndarray) -> np.ndarray:
        return apply_scalar_func(_ppf_single, p_arr)

    return FittedComputationMethod(target=PPF, source=CDF, func=compute_ppf)


def fit_pmf_to_cdf(
    distribution: "Distribution", params: dict[str, Any]
) -> FittedComputationMethod[int, float]:
    """
    Construct a CDF from a discrete PMF via cumulative summation.

    The resulting CDF is normalized to ensure the final value equals one.

    Parameters
    ----------
    distribution : Distribution
        Distribution providing a scalar PMF.
    params : dict
        Support bounds and normalization parameters.

    Returns
    -------
    FittedComputationMethod
        Vectorized callable computing the CDF.
    """

    pmf_func = _get_scalar_characteristic(distribution, PMF)

    default_params: PMFtoCDFParams = {
        "support_min": -100,
        "support_max": 100,
        "epsilon": 1e-12,
    }
    config = {**default_params, **params}

    support = np.arange(config["support_min"], config["support_max"] + 1)
    pmf_vals = apply_scalar_func(pmf_func, support)
    cdf_vals = np.cumsum(pmf_vals)
    cdf_vals /= max(cdf_vals[-1], config["epsilon"])

    def compute_cdf(k_arr: np.ndarray) -> np.ndarray:
        k_arr = k_arr.astype(int)
        return np.interp(
            k_arr,
            support,
            cdf_vals,
            left=0.0,
            right=1.0,
        )

    return FittedComputationMethod(target=CDF, source=PMF, func=compute_cdf)


def fit_pmf_to_ppf(
    distribution: "Distribution", params: dict[str, Any]
) -> FittedComputationMethod[float, int]:
    """
    Compute the PPF from a PMF via composition: PMF → CDF → PPF.

    Parameters
    ----------
    distribution : Distribution
        Distribution providing a scalar PMF.
    params : dict
        Parameters for CDF construction and inversion.

    Returns
    -------
    FittedComputationMethod
        Vectorized callable approximating the PPF.
    """

    cdf_comp = fit_pmf_to_cdf(distribution, params.get("cdf_params", {}))

    def cdf_scalar(x: float) -> float:
        return float(cdf_comp.func(np.array([x]))[0])

    temp_dist = FunctionalDistribution(cdf=cdf_scalar)
    ppf_comp = fit_cdf_to_ppf(temp_dist, params.get("ppf_params", {}))

    return FittedComputationMethod(target=PPF, source=PMF, func=ppf_comp.func)


def fit_ppf_to_cdf(
    distribution: "Distribution", params: dict[str, Any]
) -> FittedComputationMethod[float, float]:
    """
    Approximate the CDF from a PPF by numerical inversion.

    The PPF is sampled on a probability grid and inverted using
    interpolation to obtain an approximate CDF.

    Parameters
    ----------
    distribution : Distribution
        Distribution providing a scalar PPF.
    params : dict
        Sampling and interpolation parameters.

    Returns
    -------
    FittedComputationMethod
        Vectorized callable approximating the CDF.
    """

    ppf_func = _get_scalar_characteristic(distribution, PPF)

    default_params: PPFtoCDFParams = {
        "p_min": 1e-4,
        "p_max": 1 - 1e-4,
        "initial_samples": 200,
    }
    config = {**default_params, **params}

    p_grid = np.linspace(
        config["p_min"],
        config["p_max"],
        config["initial_samples"],
    )
    x_grid = apply_scalar_func(ppf_func, p_grid)

    interp = interpolate.interp1d(
        x_grid,
        p_grid,
        bounds_error=False,
        fill_value=(0.0, 1.0),
        assume_sorted=True,
    )

    def compute_cdf(x_arr: np.ndarray) -> np.ndarray:
        return interp(x_arr)

    return FittedComputationMethod(target=CDF, source=PPF, func=compute_cdf)


def fit_ppf_to_pdf(
    distribution: "Distribution", params: dict[str, Any]
) -> FittedComputationMethod[float, float]:
    """
    Approximate the PDF from a PPF via implicit differentiation.

    The PDF is computed as:
        f(x) = 1 / (d/dp PPF(p))  evaluated at p = CDF(x)

    where the CDF is itself obtained numerically from the PPF.

    Parameters
    ----------
    distribution : Distribution
        Distribution providing a scalar PPF.
    params : dict
        Differentiation and stability parameters.

    Returns
    -------
    FittedComputationMethod
        Vectorized callable approximating the PDF.
    """

    # build CDF from PPF
    cdf_comp = fit_ppf_to_cdf(distribution, params.get("cdf_params", {}))

    def cdf_scalar(x: float) -> float:
        return float(cdf_comp.func(np.array([x]))[0])

    # differentiate PPF
    ppf_func = _get_scalar_characteristic(distribution, PPF)

    h = params.get("h", 1e-5)
    min_pdf = params.get("min_pdf", 0.0)

    def _pdf_single(x: float) -> float:
        p = cdf_scalar(x)

        # Guard against numerical issues
        if p <= 0.0 or p >= 1.0:
            return 0.0

        dp = (ppf_func(p + h) - ppf_func(p - h)) / (2 * h)

        if dp <= 0.0:
            return 0.0

        return max(1.0 / dp, min_pdf)

    def compute_pdf(x_arr: np.ndarray) -> np.ndarray:
        return apply_scalar_func(_pdf_single, x_arr)

    return FittedComputationMethod(target=PDF, source=PPF, func=compute_pdf)


def fit_ppf_to_pmf(
    distribution: "Distribution", params: dict[str, Any]
) -> FittedComputationMethod[int, float]:
    """
    Compute a PMF from a PPF via composition: PPF → CDF → PMF.

    This method is intended for discretized distributions derived from
    an underlying quantile function.

    Parameters
    ----------
    distribution : Distribution
        Distribution providing a scalar PPF.
    params : dict
        Parameters for intermediate CDF and PMF construction.

    Returns
    -------
    FittedComputationMethod
        Vectorized callable computing the PMF.
    """

    # PPF → CDF
    cdf_comp = fit_ppf_to_cdf(distribution, params.get("cdf_params", {}))

    def cdf_scalar(x: float) -> float:
        return float(cdf_comp.func(np.array([x]))[0])

    temp_dist = FunctionalDistribution(cdf=cdf_scalar)

    # CDF → PMF
    pmf_comp = fit_cdf_to_pmf(temp_dist, params.get("pmf_params", {}))

    return FittedComputationMethod(target=PMF, source=PPF, func=pmf_comp.func)


def universal_fitter(
    distribution: "Distribution",
    source_characteristic: GenericCharacteristicName,
    target_characteristic: GenericCharacteristicName,
    params: dict[str, Any] | None = None,
) -> FittedComputationMethod[float, float]:
    """
    Dispatch and construct a numerical fitter between two characteristics.

    This function selects the appropriate conversion strategy based on
    source and target characteristics and returns a fitted computation
    method.

    Parameters
    ----------
    distribution : Distribution
        Source distribution.
    source_characteristic : str
        Name of the source characteristic.
    target_characteristic : str
        Name of the target characteristic.
    params : dict, optional
        Parameters for the selected fitter.

    Returns
    -------
    FittedComputationMethod
        Conversion method between the requested characteristics.

    Raises
    ------
    ValueError
        If the requested conversion is not supported.
    """

    if params is None:
        params = {}
    match (source_characteristic, target_characteristic):
        # FROM PDF
        case (PDF, CDF):
            return fit_pdf_to_cdf(distribution, params)  # интегрирование
        case (PDF, PPF):
            return fit_pdf_to_ppf(distribution, params)  # композиция pdf -> cdf -> ppf
        case (PDF, PMF):
            raise ValueError(
                f"Unsupported conversion: {source_characteristic} -> {target_characteristic}"
            )

        # FROM CDF
        case (CDF, PDF):
            return fit_cdf_to_pdf(distribution, params)  # дифференцирование
        case (CDF, PMF):
            return fit_cdf_to_pmf(distribution, params)  # разность значений
        case (CDF, PPF):
            return fit_cdf_to_ppf(distribution, params)  # инверсия

        # FROM PMF
        case (PMF, PDF):
            raise ValueError(
                f"Unsupported conversion: {source_characteristic} -> {target_characteristic}"
            )
        case (PMF, CDF):
            return fit_pmf_to_cdf(distribution, params)  # суммирование
        case (PMF, PPF):
            return fit_pmf_to_ppf(distribution, params)  # композиция pmf -> cdf -> ppf

        # FROM PPF
        case (PPF, PDF):
            return fit_ppf_to_pdf(distribution, params)  # неявное дифференицирование
        case (PPF, CDF):
            return fit_ppf_to_cdf(distribution, params)  # инверсия
        case (PPF, PMF):
            return fit_ppf_to_pmf(distribution, params)  # композиция ppf -> cdf -> pmf

        case _:
            raise ValueError(
                f"Unsupported conversion: {source_characteristic} -> {target_characteristic}"
            )
