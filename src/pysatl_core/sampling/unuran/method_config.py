from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class UnuranMethod(StrEnum):
    """
    UNU.RAN sampling methods.

    Methods are categorized by their approach:
    - Inversion methods: use inverse CDF
    - Rejection methods: accept/reject based on envelope
    - Transformation methods: transform from simpler distributions
    - Numerical inversion: approximate inverse CDF
    - Specialized methods: optimized for specific distribution types
    """

    # Automatic method selection
    AUTO = "auto"

    # Inversion methods
    PINV = "pinv"  # Polynomial interpolation inversion
    TDR = "tdr"  # Transformed density rejection

    # Rejection methods
    AROU = "arou"  # Automatic ratio-of-uniforms
    HINV = "hinv"  # Hermite interpolation inversion
    NINV = "ninv"  # Numerical inversion

    # Discrete methods
    DGT = "dgt"  # Discrete guide table method


@dataclass(frozen=True, slots=True)
class UnuranMethodConfig:
    """
    Configuration for a UNU.RAN sampling method.

    Parameters
    ----------
    method : UnuranMethod
        The sampling method to use. If ``UnuranMethod.AUTO``, UNU.RAN will
        automatically select the best method based on available distribution
        characteristics.
    method_params : dict[str, Any], optional
        Method-specific parameters. The exact parameters depend on the chosen
        method. Common parameters include:
        - ``accuracy``: target accuracy for numerical methods
        - ``max_iterations``: maximum iterations for iterative methods
        - ``grid_size``: grid size for interpolation methods
        - ``smooth``: smoothing parameter for kernel methods
    use_ppf : bool, default True
        If ``True``, prefer using PPF (inverse CDF) when available. This is
        typically the fastest method for univariate distributions.
    use_pdf : bool, default True
        If ``True``, allow using PDF for rejection-based methods.
    use_cdf : bool, default True
        If ``True``, allow using CDF for inversion-based methods.
    use_registry_characteristics : bool, default True
        If ``True``, allow using distribution characteristics from the registry
        in addition to those directly available in the Distribution object.
        If ``False``, use only characteristics that are directly available
        in the Distribution object, without querying the registry.
    use_fallback_sampler : bool, default True
        If ``True``, fall back to ``DefaultSamplingUnivariateStrategy`` when
        UNU.RAN initialization fails (e.g. non-integer discrete support).
        If ``False``, the original ``RuntimeError`` is re-raised instead.

    Notes
    -----
    - Method-specific parameters are validated when the sampler is created
    - Some methods require specific characteristics (e.g., rejection methods
      typically need PDF)
    - The ``use_*`` flags control which distribution characteristics can be
      used, but do not guarantee their use
    - When ``use_registry_characteristics`` is ``True``, characteristics may
      be retrieved from the distribution registry if not directly available
      in the Distribution object
    """

    method: UnuranMethod = UnuranMethod.AUTO
    method_params: dict[str, Any] | None = None
    use_ppf: bool = True
    use_pdf: bool = True
    use_cdf: bool = True
    use_registry_characteristics: bool = True
    use_fallback_sampler: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.method_params is None:
            object.__setattr__(self, "method_params", {})
