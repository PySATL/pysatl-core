"""
Helper utilities that bridge UNU.RAN's URNG interface with NumPy's RNG.

UNU.RAN expects a default uniform RNG to be configured globally. The vendored
library aborts if ``unur_get_default_urng()`` is called before a generator is
registered. This module exposes :func:`ensure_default_urng` that lazily creates
and registers a URNG backed by ``numpy.random.default_rng``.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

__all__ = ["ensure_default_urng"]

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pysatl_core.sampling.unuran.bindings import _unuran_cffi

if TYPE_CHECKING:
    from numpy.random import Generator

Callback = Callable[[Any], float]

_DEFAULT_URNG: Any | None = None
_DEFAULT_CALLBACK: Callback | None = None
_DEFAULT_STATE: Any | None = None


def ensure_default_urng() -> None:
    """
    Lazily configure the global UNU.RAN URNG backed by NumPy's default RNG.

    Creates a CFFI callback that wraps ``numpy.random.default_rng().random()``
    and registers it as both the primary and auxiliary UNU.RAN URNG via
    ``unur_set_default_urng`` and ``unur_set_default_urng_aux``. Subsequent
    calls are no-ops once the URNG has been registered.

    Raises
    ------
    RuntimeError
        If the CFFI bindings are unavailable or if UNU.RAN fails to create
        the URNG object.
    """

    global _DEFAULT_URNG, _DEFAULT_CALLBACK, _DEFAULT_STATE

    if _DEFAULT_URNG is not None:
        return

    if _unuran_cffi is None:
        raise RuntimeError(
            "UNURAN CFFI bindings are not available. "
            "Please build them via `python -m pysatl_core.sampling.unuran.bindings._cffi_build`."
        )

    ffi = _unuran_cffi.ffi
    lib = _unuran_cffi.lib

    rng = np.random.default_rng()
    state_handle = ffi.new_handle(rng)

    callback_decorator = cast(
        Callable[[Callback], Callback],
        ffi.callback("double(void *)"),
    )

    @callback_decorator
    def _sample_uniform(state: Any) -> float:
        generator = cast("Generator", ffi.from_handle(state))
        return float(generator.random())

    urng = lib.unur_urng_new(_sample_uniform, state_handle)
    if urng == ffi.NULL:
        raise RuntimeError("Failed to create UNU.RAN default URNG")

    lib.unur_set_default_urng(urng)
    lib.unur_set_default_urng_aux(urng)

    _DEFAULT_URNG = urng
    _DEFAULT_CALLBACK = _sample_uniform
    _DEFAULT_STATE = state_handle
