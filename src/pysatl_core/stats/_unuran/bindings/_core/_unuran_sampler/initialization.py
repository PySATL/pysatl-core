from __future__ import annotations

from typing import Any

from pysatl_core.stats._unuran.api import UnuranMethod

from .callbacks import setup_continuous_callbacks, setup_discrete_callbacks
from .cleanup import cleanup_unuran_resources
from .dgt import setup_dgt_method
from .utils import get_unuran_error_message


def create_unuran_distribution(sampler: Any) -> None:
    """
    Create UNURAN distribution object.

    Raises
    ------
    RuntimeError
        If the distribution object creation fails (returns NULL pointer).

    Notes
    -----
    Creates either a continuous or discrete UNURAN distribution object
    based on ``_is_continuous`` flag. The created object is stored in
    ``_unuran_distr`` attribute.
    """
    if sampler._is_continuous:
        sampler._unuran_distr = sampler._lib.unur_distr_cont_new()
    else:
        sampler._unuran_distr = sampler._lib.unur_distr_discr_new()

    if sampler._unuran_distr == sampler._ffi.NULL:
        raise RuntimeError("Failed to create UNURAN distribution object")


def create_parameter_object(sampler: Any) -> Any:
    """
    Create UNURAN parameter object based on selected method.

    Returns
    -------
    CFFI pointer
        Pointer to the created UNURAN parameter object, or NULL pointer
        if creation fails.

    Raises
    ------
    ValueError
        If the method is SROU (not available in CFFI bindings) or if the
        method is unsupported.

    Notes
    -----
    Maps UNURAN methods to their corresponding CFFI creation functions:
    - AROU: ``unur_arou_new()``
    - TDR: ``unur_tdr_new()``
    - HINV: ``unur_hinv_new()``
    - PINV: ``unur_pinv_new()``
    - NINV: ``unur_ninv_new()``
    - DGT: ``unur_dgt_new()``

    Note that AROU and TDR require dPDF, which may not be available.
    """
    method = sampler._method
    lib = sampler._lib

    if method == UnuranMethod.AROU:
        return lib.unur_arou_new(sampler._unuran_distr)
    if method == UnuranMethod.TDR:
        return lib.unur_tdr_new(sampler._unuran_distr)
    if method == UnuranMethod.HINV:
        return lib.unur_hinv_new(sampler._unuran_distr)
    if method == UnuranMethod.PINV:
        return lib.unur_pinv_new(sampler._unuran_distr)
    if method == UnuranMethod.NINV:
        return lib.unur_ninv_new(sampler._unuran_distr)
    if method == UnuranMethod.DGT:
        return lib.unur_dgt_new(sampler._unuran_distr)
    if method == UnuranMethod.SROU:
        raise ValueError("Method SROU is not available in CFFI bindings. Use PINV instead.")

    raise ValueError(f"Unsupported UNURAN method: {method}")


def create_and_init_generator(sampler: Any) -> None:
    """
    Create UNURAN parameter object and initialize the generator.

    Raises
    ------
    RuntimeError
        If parameter object creation fails or generator initialization fails.

    Notes
    -----
    The parameter object is created based on the selected method (PINV, HINV,
    DGT, etc.). After successful initialization, the parameter object is
    automatically destroyed by UNURAN, so it should not be freed manually.

    The initialized generator is stored in ``_unuran_gen`` attribute.
    """
    sampler._unuran_par = sampler._create_parameter_object()
    if sampler._unuran_par == sampler._ffi.NULL:
        error_msg = sampler._get_unuran_error_message("Failed to create UNURAN parameter object")
        raise RuntimeError(error_msg)

    sampler._unuran_gen = sampler._lib.unur_init(sampler._unuran_par)
    if sampler._unuran_gen == sampler._ffi.NULL:
        error_msg = sampler._get_unuran_error_message("Failed to initialize UNURAN generator")
        raise RuntimeError(error_msg)


def initialize_unuran_components(sampler: Any, seed: int | None) -> None:
    """
    Initialize all UNURAN components for the sampler.

    Parameters
    ----------
    seed : int, optional
        Random seed for the generator. Currently not used.
    """
    # TODO seed support
    sampler._create_unuran_distribution()

    try:
        if sampler._is_continuous:
            sampler._setup_continuous_callbacks()
        else:
            sampler._setup_discrete_callbacks()
            if sampler._method == UnuranMethod.DGT:
                sampler._setup_dgt_method()

        sampler._create_and_init_generator()
    except Exception:
        sampler._cleanup()
        raise


__all__ = [
    "create_and_init_generator",
    "create_parameter_object",
    "create_unuran_distribution",
    "initialize_unuran_components",
]
