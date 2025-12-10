
from __future__ import annotations

from pysatl_core.stats._unuran.api import UnuranMethod, UnuranMethodConfig, UnuranSampler

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from pysatl_core.distributions.distribution import Distribution



class DefaultUnuranSampler(UnuranSampler):
    """
    Default UNU.RAN sampler implementation.

    This sampler uses the default UNU.RAN method selection and parameters.
    """

    def __init__(self, distr: Distribution, config: UnuranMethodConfig | None = None, **override_options: Any):
        self.distr = distr
        self.config = config
        self.override_options = override_options

    def sample(self, n: int) -> npt.NDArray[np.float64]:
        ...

    def reset(self, seed: int | None = None) -> None:
        ...

    @property
    def method(self) -> UnuranMethod:
        ...

    @property
    def is_initialized(self) -> bool:
        ...
