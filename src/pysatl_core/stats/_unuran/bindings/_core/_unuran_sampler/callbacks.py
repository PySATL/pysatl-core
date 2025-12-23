from __future__ import annotations

from typing import Any

from pysatl_core.types import CharacteristicName


def create_pdf_callback(sampler: Any) -> Any | None:
    """
    Create PDF callback function for continuous distributions.

    Returns
    -------
    CFFI callback or None
        CFFI callback function that wraps the distribution's PDF computation,
        or None if the distribution is not continuous or PDF is not available.

    Notes
    -----
    The callback signature matches UNURAN's expected format:
    ``double(double, const struct unur_distr*)``

    The callback is stored in ``_callbacks`` list to prevent garbage collection.
    """
    if not sampler._is_continuous:
        return None

    analytical_comps = sampler.distr.analytical_computations
    if CharacteristicName.PDF not in analytical_comps:
        return None

    pdf_func = analytical_comps[CharacteristicName.PDF]

    def pdf_callback(x: float, distr_ptr: Any) -> float:
        return float(pdf_func(x))

    return sampler._ffi.callback("double(double, const struct unur_distr*)", pdf_callback)


def create_pmf_callback(sampler: Any) -> Any | None:
    """
    Create PMF callback function for discrete distributions.

    Returns
    -------
    CFFI callback or None
        CFFI callback function that wraps the distribution's PMF computation,
        or None if the distribution is not discrete or PMF/PDF is not available.

    Notes
    -----
    The callback signature matches UNURAN's expected format:
    ``double(int, const struct unur_distr*)``

    If PMF is not available, falls back to PDF (some systems use "pdf" for
    discrete distributions too). The callback is stored in ``_callbacks``
    list to prevent garbage collection.
    """
    if sampler._is_continuous:
        return None

    analytical_comps = sampler.distr.analytical_computations

    if CharacteristicName.PMF in analytical_comps:
        pmf_func = analytical_comps[CharacteristicName.PMF]

        def pmf_callback(k: int, distr_ptr: Any) -> float:
            return float(pmf_func(float(k)))

        return sampler._ffi.callback("double(int, const struct unur_distr*)", pmf_callback)

    if CharacteristicName.PDF in analytical_comps:
        pdf_func = analytical_comps[CharacteristicName.PDF]

        def pmf_callback(k: int, distr_ptr: Any) -> float:
            return float(pdf_func(float(k)))

        return sampler._ffi.callback("double(int, const struct unur_distr*)", pmf_callback)

    return None


def create_cdf_callback(sampler: Any) -> Any | None:
    """
    Create PMF callback function for discrete distributions.

    Returns
    -------
    CFFI callback or None
        CFFI callback function that wraps the distribution's PMF computation,
        or None if the distribution is not discrete or PMF/PDF is not available.

    Notes
    -----
    The callback signature matches UNURAN's expected format:
    ``double(int, const struct unur_distr*)``

    If PMF is not available, falls back to PDF (some systems use "pdf" for
    discrete distributions too). The callback is stored in ``_callbacks``
    list to prevent garbage collection.
    """
    analytical_comps = sampler.distr.analytical_computations
    if CharacteristicName.CDF not in analytical_comps:
        return None

    cdf_func = analytical_comps[CharacteristicName.CDF]

    if sampler._is_continuous:

        def cdf_callback_cont(x: float, distr_ptr: Any) -> float:
            return float(cdf_func(x))

        return sampler._ffi.callback("double(double, const struct unur_distr*)", cdf_callback_cont)

    def cdf_callback_discr(k: int, distr_ptr: Any) -> float:
        return float(cdf_func(float(k)))

    return sampler._ffi.callback("double(int, const struct unur_distr*)", cdf_callback_discr)


def create_ppf_callback(sampler: Any) -> Any | None:
    """
    Create PPF (inverse CDF) callback for continuous distributions.

    Returns
    -------
    CFFI callback or None
        Callback wrapping the distribution's PPF computation, or None if the
        distribution is not continuous or lacks a PPF implementation.
    """
    if not sampler._is_continuous:
        return None

    analytical_comps = sampler.distr.analytical_computations
    if CharacteristicName.PPF not in analytical_comps:
        return None

    ppf_func = analytical_comps[CharacteristicName.PPF]

    def ppf_callback(u: float, distr_ptr: Any) -> float:
        return float(ppf_func(u))

    return sampler._ffi.callback("double(double, const struct unur_distr*)", ppf_callback)


def create_dpdf_callback() -> None:
    """
    Create derivative PDF (dPDF) callback function.

    Returns
    -------
    None
        Currently always returns None as derivatives are not computed
        automatically.

    Notes
    -----
    This is a placeholder for future enhancement. Automatic differentiation
    could be used to compute derivatives of PDF functions, which would enable
    additional UNURAN methods like AROU and TDR that require dPDF.
    """
    return None


def setup_continuous_callbacks(sampler: Any) -> None:
    """
    Set up callbacks for continuous distributions.

    Configures PDF, dPDF (if available), and CDF callbacks for the UNURAN
    continuous distribution object.

    Raises
    ------
    RuntimeError
        If setting the PDF callback fails (non-zero return code).

    Notes
    -----
    All created callbacks are appended to ``_callbacks`` list to prevent
    garbage collection. Only available callbacks are set (missing
    characteristics are skipped).
    """
    pdf_callback = sampler._create_pdf_callback()
    if pdf_callback:
        sampler._callbacks.append(pdf_callback)
        result = sampler._lib.unur_distr_cont_set_pdf(sampler._unuran_distr, pdf_callback)
        if result != 0:
            raise RuntimeError(f"Failed to set PDF callback (error code: {result})")

    dpdf_callback = sampler._create_dpdf_callback()
    if dpdf_callback:
        sampler._callbacks.append(dpdf_callback)
        sampler._lib.unur_distr_cont_set_dpdf(sampler._unuran_distr, dpdf_callback)

    cdf_callback = sampler._create_cdf_callback()
    if cdf_callback:
        sampler._callbacks.append(cdf_callback)
        sampler._lib.unur_distr_cont_set_cdf(sampler._unuran_distr, cdf_callback)

    ppf_callback = sampler._create_ppf_callback()
    if ppf_callback and hasattr(sampler._lib, "unur_distr_cont_set_invcdf"):
        sampler._callbacks.append(ppf_callback)
        sampler._lib.unur_distr_cont_set_invcdf(sampler._unuran_distr, ppf_callback)


def setup_discrete_callbacks(sampler: Any) -> None:
    """
    Set up callbacks for discrete distributions.

    Configures PMF and CDF callbacks for the UNURAN discrete distribution
    object.

    Raises
    ------
    RuntimeError
        If setting the PMF callback fails (non-zero return code).

    Notes
    -----
    All created callbacks are appended to ``_callbacks`` list to prevent
    garbage collection. Only available callbacks are set (missing
    characteristics are skipped).
    """
    pmf_callback = sampler._create_pmf_callback()
    if pmf_callback:
        sampler._callbacks.append(pmf_callback)
        result = sampler._lib.unur_distr_discr_set_pmf(sampler._unuran_distr, pmf_callback)
        if result != 0:
            raise RuntimeError(f"Failed to set PMF callback (error code: {result})")

    cdf_callback = sampler._create_cdf_callback()
    if cdf_callback:
        sampler._callbacks.append(cdf_callback)
        sampler._lib.unur_distr_discr_set_cdf(sampler._unuran_distr, cdf_callback)
