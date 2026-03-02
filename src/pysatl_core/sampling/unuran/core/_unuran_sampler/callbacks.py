"""
Callback creation utilities for UNU.RAN sampler bindings.

Provides callback function creation for PDF/PMF evaluation needed during
UNU.RAN distribution setup.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from pysatl_core.types import CharacteristicName, Kind

if TYPE_CHECKING:
    from pysatl_core.types import Method

_CONT_SIG = "double(double, const struct unur_distr*)"
_DISCR_SIG = "double(int, const struct unur_distr*)"


class UnuranCallback:
    """
    Factory and registry for CFFI callbacks wired into a UNU.RAN distribution object.

    Creates closures over distribution characteristic functions (PDF, dPDF, CDF,
    PPF, PMF) and registers them with the corresponding UNU.RAN setter. All live
    callback objects are kept in an internal list so the garbage collector cannot
    reclaim them while the UNU.RAN generator is alive.
    """

    def __init__(
        self,
        unuran_distr: Any,
        kind: Kind,
        lib: Any,
        ffi: Any,
        characteristics: Mapping[CharacteristicName, Method[Any, Any]],
    ) -> None:
        """
        Parameters
        ----------
        unuran_distr : Any
            CFFI pointer to the UNU.RAN distribution object.
        kind : Kind
            Whether the distribution is continuous or discrete.
        lib : Any
            CFFI library handle.
        ffi : Any
            CFFI FFI instance used to create callbacks.
        characteristics : Mapping[CharacteristicName, Method[Any, Any]]
            Map of available characteristic names to their callables.
        """
        self._unuran_distr = unuran_distr
        self._kind = kind
        self._lib = lib
        self._ffi = ffi
        self._characteristics = dict(characteristics)
        self._callbacks: list[Any] = []

    @property
    def callbacks(self) -> list[Any]:
        """Live CFFI callback objects that must remain referenced for UNU.RAN."""
        return self._callbacks

    def _create_callback(self, char_name: CharacteristicName, signature: str) -> Any | None:
        """
        Create a CFFI callback for the given characteristic and UNU.RAN signature.

        The wrapper converts the scalar input to a 1-D ``float64`` numpy array
        before calling the characteristic, then converts the result to ``float``.
        This ensures compatibility with characteristics that use numpy array
        methods internally, while satisfying UNU.RAN's ``double`` return type.

        Parameters
        ----------
        char_name:
            Characteristic to look up in the distribution.
        signature:
            CFFI function type string, e.g. ``"double(double, const struct unur_distr*)"``.

        Returns
        -------
        CFFI callback or None
            None if the characteristic is not available.
        """
        func = self._characteristics.get(char_name)
        if func is None:
            return None

        def cb(x: Any, _: Any) -> float:
            return float(func(np.asarray(x, dtype=float)))

        return self._ffi.callback(signature, cb)

    def setup_callback(self, func: Any | None, cffi_func: Any, error_text: str) -> None:
        """
        Register a single CFFI callback with its UNU.RAN setter.

        Parameters
        ----------
        func : Any or None
            CFFI callback to register. Skipped silently when ``None``.
        cffi_func : Any
            UNU.RAN setter function (e.g. ``unur_distr_cont_set_pdf``).
        error_text : str
            Message template passed to :class:`RuntimeError` on failure; must
            contain one ``{}`` placeholder for the error code.

        Raises
        ------
        RuntimeError
            If the setter returns a non-zero error code.
        """
        if func:
            self._callbacks.append(func)
            result = cffi_func(self._unuran_distr, func)
            if result != 0:
                raise RuntimeError(error_text.format(result))

    def setup_continuous_callbacks(self) -> None:
        """
        Set up callbacks for continuous distributions.

        Configures PDF, dPDF (if available), CDF, and PPF callbacks for the UNURAN
        continuous distribution object.

        Raises
        ------
        RuntimeError
            If setting any callback fails (non-zero return code).

        Notes
        -----
        All created callbacks are appended to ``_callbacks`` list to prevent
        garbage collection. Only available callbacks are set (missing
        characteristics are skipped).
        """
        self.setup_callback(
            self._create_callback(CharacteristicName.PDF, _CONT_SIG),
            self._lib.unur_distr_cont_set_pdf,
            "Failed to set PDF callback (error code: {})",
        )
        self.setup_callback(
            self._create_callback(CharacteristicName.DPDF, _CONT_SIG),
            self._lib.unur_distr_cont_set_dpdf,
            "Failed to set dPDF callback (error code: {})",
        )
        self.setup_callback(
            self._create_callback(CharacteristicName.CDF, _CONT_SIG),
            self._lib.unur_distr_cont_set_cdf,
            "Failed to set CDF callback (error code: {})",
        )
        self.setup_callback(
            self._create_callback(CharacteristicName.PPF, _CONT_SIG),
            self._lib.unur_distr_cont_set_invcdf,
            "Failed to set PPF callback (error code: {})",
        )

    def _create_indexed_pmf_callback(self, points: np.ndarray) -> Any | None:
        """
        Create a CFFI PMF callback that maps integer indices to support values.

        UNU.RAN will call this with indices ``0, 1, ..., n-1``.  The callback
        converts each index to the corresponding support value via ``points[i]``
        before evaluating the PMF, enabling DGT to work with arbitrary
        (non-integer, sparse) supports.

        Parameters
        ----------
        points : np.ndarray
            Array of support values; ``points[i]`` is the actual value for index ``i``.

        Returns
        -------
        CFFI callback or None
            None if PMF is not available.
        """
        func = self._characteristics.get(CharacteristicName.PMF)
        if func is None:
            return None

        def cb(i: Any, _: Any) -> float:
            return float(func(np.asarray(points[i], dtype=float)))

        return self._ffi.callback(_DISCR_SIG, cb)

    def setup_discrete_callbacks(self, index_remap_points: np.ndarray | None = None) -> None:
        """
        Set up callbacks for discrete distributions.

        Configures PMF and CDF callbacks for the UNURAN discrete distribution
        object.

        Parameters
        ----------
        index_remap_points : np.ndarray or None
            When provided, the PMF callback treats its integer argument as an
            index into this array and evaluates the PMF at ``points[i]`` instead
            of at ``i`` directly.  Use this when the UNU.RAN domain is set to
            ``[0, n-1]`` (index space) rather than the actual support values.

        Raises
        ------
        RuntimeError
            If setting any callback fails (non-zero return code).

        Notes
        -----
        All created callbacks are appended to ``_callbacks`` list to prevent
        garbage collection. Only available callbacks are set (missing
        characteristics are skipped).
        """
        if index_remap_points is not None:
            pmf_cb = self._create_indexed_pmf_callback(index_remap_points)
        else:
            pmf_cb = self._create_callback(CharacteristicName.PMF, _DISCR_SIG)

        self.setup_callback(
            pmf_cb,
            self._lib.unur_distr_discr_set_pmf,
            "Failed to set PMF callback (error code: {})",
        )
        self.setup_callback(
            self._create_callback(CharacteristicName.CDF, _DISCR_SIG),
            self._lib.unur_distr_discr_set_cdf,
            "Failed to set CDF callback (error code: {})",
        )
