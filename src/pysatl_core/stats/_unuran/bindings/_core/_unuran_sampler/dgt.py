from __future__ import annotations

from typing import Any

from .domain import (
    calculate_pmf_sum,
    determine_domain_from_pmf,
    determine_domain_from_support,
)
from .utils import get_unuran_error_message


def _require_attr(lib: Any, attr_name: str) -> None:
    """
    Check if a required UNURAN CFFI attribute is available.
    
    Parameters
    ----------
    lib : Any
        The UNURAN CFFI library object
    attr_name : str
        Name of the required attribute/function
        
    Raises
    ------
    RuntimeError
        If the required attribute is not available
    """
    if not hasattr(lib, attr_name):
        raise RuntimeError(
            f"{attr_name} is not available. "
            "Please recompile CFFI module: cd src/pysatl_core/stats/_unuran/bindings && python3 _cffi_build.py"
        )


def setup_dgt_method(sampler: Any) -> None:
    """
    Set up DGT method specific requirements (domain and probability vector).

    The DGT (Discrete Generation Table) method requires:
    1. A domain to be set for the discrete distribution
    2. A probability vector (PV) to be created from the PMF

    Raises
    ------
    RuntimeError
        If required CFFI functions are not available, if domain setting fails,
        or if PV creation fails.

    Notes
    -----
    - Domain is determined from distribution support if available
    - If support is unbounded or unavailable, domain is determined by
        evaluating PMF until cumulative probability exceeds threshold (0.9999)
    - PMF sum is calculated from the domain if possible, otherwise defaults to 1.0
    - This method should only be called for discrete distributions using
        the DGT method
    """
    lib = sampler._lib
    ffi = sampler._ffi

    _require_attr(lib, "unur_distr_discr_set_domain")
    _require_attr(lib, "unur_distr_discr_make_pv")

    domain_info = determine_domain_from_support(sampler)

    if domain_info is None:
        domain_left, domain_right = determine_domain_from_pmf(sampler)
    else:
        domain_left_candidate, domain_right_candidate = domain_info
        if domain_right_candidate is None:
            domain_left, domain_right = determine_domain_from_pmf(sampler, domain_left_candidate)
        else:
            domain_left = domain_left_candidate
            domain_right = domain_right_candidate

    result = lib.unur_distr_discr_set_domain(sampler._unuran_distr, domain_left, domain_right)
    if result != 0:
        raise RuntimeError(
            f"Failed to set domain for discrete distribution (error code: {result}). "
            f"Tried domain [{domain_left}, {domain_right}]"
        )

    pmf_sum = calculate_pmf_sum(sampler, domain_left, domain_right)
    if hasattr(lib, "unur_distr_discr_set_pmfsum"):
        lib.unur_distr_discr_set_pmfsum(sampler._unuran_distr, pmf_sum)

    pv_length = lib.unur_distr_discr_make_pv(sampler._unuran_distr)
    if pv_length <= 0:
        error_msg = (
            f"Failed to create PV from PMF (returned length: {pv_length}). "
            "DGT method requires PV. "
            "The PMF might not be normalized or domain is too large. "
            f"Domain was set to [{domain_left}, {domain_right}], "
            f"PMF sum calculated as {pmf_sum:.6f}. "
            "Try setting a smaller domain or providing PV directly."
        )
        full_error_msg = get_unuran_error_message(lib, ffi, error_msg)
        raise RuntimeError(full_error_msg)


__all__ = ["setup_dgt_method"]
