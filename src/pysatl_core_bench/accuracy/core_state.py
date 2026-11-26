"""Core state hygiene for scenario batches.

pysatl-core keeps global singletons (family register, characteristic
registry, fitter registry).  Like the Core test-suite, the benchmark
framework resets them between scenarios so runs never pollute each other;
only the public ``reset_*`` functions are used.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_core.distributions.computations.registry import reset_fitter_registry
from pysatl_core.distributions.registry import reset_characteristic_registry
from pysatl_core.families.configuration import reset_families_register

__all__ = ["reset_core_state"]


def reset_core_state() -> None:
    """Reset all pysatl-core singletons through their public reset hooks."""
    reset_characteristic_registry()
    reset_families_register()
    reset_fitter_registry()
