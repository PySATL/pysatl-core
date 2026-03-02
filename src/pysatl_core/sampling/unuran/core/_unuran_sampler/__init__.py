"""
PySATL Core — UNU.RAN Sampler Internals
=========================================

Internal sub-package that wires together the two building blocks of the
UNU.RAN sampler:

- ``UnuranSamplerInitializer`` — creates the UNU.RAN distribution object,
  registers distribution callbacks (PDF, CDF, PMF, …) and initialises
  the generator for a pre-selected sampling method (PINV, NINV, DGT, …).
- ``ensure_default_urng`` — registers a NumPy-backed uniform RNG as the
  UNU.RAN default URNG so that seeding and reproducibility work through
  the standard NumPy interface.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .initialization import UnuranSamplerInitializer
from .urng import ensure_default_urng

__all__ = [
    "UnuranSamplerInitializer",
    "ensure_default_urng",
]
