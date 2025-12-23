from __future__ import annotations

import contextlib
from typing import Any


def cleanup_unuran_resources(sampler: Any) -> None:
    """
    Clean up UNURAN resources.

    Frees all allocated UNURAN objects (generator, parameter object,
    distribution object) in the correct order to avoid memory leaks.

    Notes
    -----
    Cleanup order:
    1. Free generator first (this also frees the private copy of distr)
    2. Free parameter object (only if generator was not initialized)
    3. Free the original distribution object

    Important notes:
    - After ``unur_init(par)``, the par object is automatically destroyed
        by UNURAN, so it should NOT be freed after successful initialization
    - UNURAN creates a private copy of distr inside gen, so the original
        distr can be safely freed after freeing gen
    - Callbacks are kept in ``_callbacks`` list to prevent garbage collection
        until the object is deleted
    - This method is idempotent (safe to call multiple times)
    """
    if getattr(sampler, "_cleaned_up", False):
        return

    sampler._cleaned_up = True

    gen_freed = False
    if hasattr(sampler, "_unuran_gen") and sampler._unuran_gen is not None:
        if sampler._unuran_gen != sampler._ffi.NULL:
            with contextlib.suppress(Exception):
                sampler._lib.unur_free(sampler._unuran_gen)
                gen_freed = True
        sampler._unuran_gen = None

    if hasattr(sampler, "_unuran_par") and sampler._unuran_par is not None:
        if not gen_freed and sampler._unuran_par != sampler._ffi.NULL:
            with contextlib.suppress(Exception):
                sampler._lib.unur_par_free(sampler._unuran_par)
        sampler._unuran_par = None

    if hasattr(sampler, "_unuran_distr") and sampler._unuran_distr is not None:
        if sampler._unuran_distr != sampler._ffi.NULL:
            with contextlib.suppress(Exception):
                sampler._lib.unur_distr_free(sampler._unuran_distr)
        sampler._unuran_distr = None

    if hasattr(sampler, "_callbacks"):
        pass


__all__ = ["cleanup_unuran_resources"]
