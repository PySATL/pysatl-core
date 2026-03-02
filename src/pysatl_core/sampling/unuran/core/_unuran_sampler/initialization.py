"""
Initialization helpers for UNU.RAN sampler bindings.

Provides the orchestration logic that builds UNURAN distribution objects,
sets up callbacks/domains, and initializes the generator according to the
selected method for a given distribution.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import contextlib
import math
from typing import TYPE_CHECKING, Any

from pysatl_core.distributions.distribution import Distribution
from pysatl_core.sampling.unuran.core._unuran_sampler.callbacks import UnuranCallback
from pysatl_core.sampling.unuran.core._unuran_sampler.dgt import DGTSetup
from pysatl_core.sampling.unuran.core._unuran_sampler.domain import UnuranDomain
from pysatl_core.sampling.unuran.core._unuran_sampler.utils import get_unuran_error_message
from pysatl_core.sampling.unuran.core.method_requirements import METHOD_CHARACTERISTIC_REQUIREMENTS
from pysatl_core.sampling.unuran.method_config import UnuranMethod
from pysatl_core.types import CharacteristicName, EuclideanDistributionType, Kind

if TYPE_CHECKING:
    from pysatl_core.types import Method


class UnuranSamplerInitializer:
    def __init__(self, distr: Distribution, method: UnuranMethod, lib: Any, ffi: Any):
        self._method = method
        self._query_method = distr.query_method
        self._support = distr.support
        self._lib = lib
        self._ffi = ffi
        self._unuran_gen: Any | None = None
        self._unuran_distr: Any | None = None
        self._unuran_par: Any | None = None
        self._domain = UnuranDomain(self._support)

        distr_type = distr.distribution_type
        if not isinstance(distr_type, EuclideanDistributionType):
            raise RuntimeError(
                f"Unsupported distribution type: {distr_type}. "
                "Only Euclidean distribution types are supported."
            )
        self._kind = distr_type.kind

        if self._kind not in (Kind.CONTINUOUS, Kind.DISCRETE):
            raise ValueError(f"Unsupported distribution kind: {self._kind}")

    def _create_unuran_distribution(self) -> None:
        """
        Create UNURAN distribution object.

        Raises
        ------
        RuntimeError
            If the distribution object creation fails (returns NULL pointer).

        Notes
        -----
        Creates either a continuous or discrete UNURAN distribution object
        based on ``_kind``. The created object is stored in
        ``_unuran_distr`` attribute.
        """
        match self._kind:
            case Kind.CONTINUOUS:
                self._unuran_distr = self._lib.unur_distr_cont_new()
            case Kind.DISCRETE:
                self._unuran_distr = self._lib.unur_distr_discr_new()
            case _:
                raise RuntimeError(f"Unsupported distribution kind: {self._kind}")

        if self._unuran_distr == self._ffi.NULL:
            raise RuntimeError("Failed to create UNURAN distribution object")

    def _create_parameter_object(self) -> Any:
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
        method = self._method
        lib = self._lib
        match method:
            case UnuranMethod.AROU:
                return lib.unur_arou_new(self._unuran_distr)
            case UnuranMethod.TDR:
                return lib.unur_tdr_new(self._unuran_distr)
            case UnuranMethod.HINV:
                return lib.unur_hinv_new(self._unuran_distr)
            case UnuranMethod.PINV:
                return lib.unur_pinv_new(self._unuran_distr)
            case UnuranMethod.NINV:
                return lib.unur_ninv_new(self._unuran_distr)
            case UnuranMethod.DGT:
                return lib.unur_dgt_new(self._unuran_distr)
            case _:
                raise ValueError(f"Unsupported UNURAN method: {method}")

    def _create_and_init_generator(self) -> Any:
        """
        Create UNURAN parameter object and initialize the generator.

        Raises
        ------
        RuntimeError
            If parameter object creation fails or generator initialization fails.

        Returns
        -------
        Any
            Pointer to the initialized UNURAN generator.

        Notes
        -----
        The parameter object is created based on the selected method (PINV, HINV,
        DGT, etc.). After successful initialization, the parameter object is
        automatically destroyed by UNURAN, so it should not be freed manually.

        The initialized generator is stored in ``_unuran_gen`` attribute.
        """
        self._unuran_par = self._create_parameter_object()
        if self._unuran_par == self._ffi.NULL:
            error_msg = get_unuran_error_message(
                self._lib, self._ffi, "Failed to create UNURAN parameter object"
            )
            raise RuntimeError(error_msg)

        unuran_gen = self._lib.unur_init(self._unuran_par)
        if unuran_gen == self._ffi.NULL:
            error_msg = get_unuran_error_message(
                self._lib, self._ffi, "Failed to initialize UNURAN generator"
            )
            raise RuntimeError(error_msg)
        self._unuran_gen = unuran_gen
        return unuran_gen

    def _requires_finite_support(self) -> bool:
        """Return whether the selected UNURAN method mandates bounded support."""
        requirements = METHOD_CHARACTERISTIC_REQUIREMENTS.get(self._method)
        return bool(requirements and requirements.requires_support)

    def _apply_continuous_domain_constraints(self) -> None:
        """Validate and register finite continuous bounds when the method requires them."""
        if self._kind != Kind.CONTINUOUS or not self._requires_finite_support():
            return

        bounds = self._domain.determine_continuous_domain()
        if bounds is None:
            raise RuntimeError(
                f"UNURAN method '{self._method.value}' requires finite support bounds, "
                "but distribution.support is missing or incomplete."
            )

        left, right = bounds
        if not (math.isfinite(left) and math.isfinite(right)):
            raise RuntimeError(
                f"UNURAN method '{self._method.value}' requires finite support bounds, "
                f"got left={left}, right={right}."
            )

        if left >= right:
            raise RuntimeError(
                f"Invalid support bounds for method '{self._method.value}': "
                f"left={left}, right={right}"
            )

        result = self._lib.unur_distr_cont_set_domain(self._unuran_distr, left, right)
        if result != 0:
            raise RuntimeError(
                f"Failed to set continuous domain [{left}, {right}] "
                f"for method '{self._method.value}' (error code: {result})."
            )

    def cleanup(self) -> None:
        """Release UNURAN generator, parameter, and distribution handles in order."""
        gen_freed = False
        if self._unuran_gen and self._unuran_gen is not None:
            if self._unuran_gen != self._ffi.NULL:
                with contextlib.suppress(Exception):
                    self._lib.unur_free(self._unuran_gen)
                    gen_freed = True
            self._unuran_gen = None

        if self._unuran_par and self._unuran_par is not None:
            if not gen_freed and self._unuran_par != self._ffi.NULL:
                with contextlib.suppress(Exception):
                    self._lib.unur_par_free(self._unuran_par)
            self._unuran_par = None

        if self._unuran_distr and self._unuran_distr is not None:
            if self._unuran_distr != self._ffi.NULL:
                with contextlib.suppress(Exception):
                    self._lib.unur_distr_free(self._unuran_distr)
            self._unuran_distr = None

    def initialize_unuran_components(
        self, available_chars: set[CharacteristicName]
    ) -> tuple[Any | None, Any | None, Any | None, list[Any]]:
        """
        Initialize all UNURAN components for the sampler.

        Parameters
        ----------
        available_chars : set[CharacteristicName]
            Characteristics resolved for the distribution (analytical + graph-derived).

        Returns
        -------
        tuple[Any | None, Any | None, Any | None, list[Any]]
            ``(unuran_distr, unuran_par, unuran_gen, callbacks)`` where callbacks keep
            the CFFI handles alive for UNURAN.
        """
        # TODO seed support
        self._create_unuran_distribution()

        distr_characteristic: dict[CharacteristicName, Method[Any, Any]] = {
            characteristic: self._query_method(characteristic) for characteristic in available_chars
        }

        callbacks = UnuranCallback(
            self._unuran_distr, self._kind, self._lib, self._ffi, distr_characteristic
        )
        self._apply_continuous_domain_constraints()

        try:
            match self._kind:
                case Kind.CONTINUOUS:
                    callbacks.setup_continuous_callbacks()
                case Kind.DISCRETE:
                    index_remap_points = self._domain.explicit_table_points()
                    callbacks.setup_discrete_callbacks(index_remap_points=index_remap_points)
                    if self._method == UnuranMethod.DGT:
                        dgt_setup = DGTSetup(
                            self._lib,
                            self._ffi,
                            self._domain,
                            self._unuran_distr,
                            self._support,
                            distr_characteristic.get(CharacteristicName.PMF),
                        )
                        dgt_setup.setup_dgt_method()
                case _:
                    raise RuntimeError(f"Unsupported distribution kind: {self._kind}")

            generator = self._create_and_init_generator()
        except Exception:
            self.cleanup()
            raise

        return self._unuran_distr, self._unuran_par, generator, callbacks.callbacks
