"""
Distribution Families Configuration
====================================

This module defines and configures parametric distribution families for the PySATL library:

- :class:`Normal Family` — Gaussian distribution with multiple parameterizations.
- :class:`Uniform Family` — Uniform distribution with multiple parameterizations.

Notes
-----
- All families are registered in the global ParametricFamilyRegister.
- Each family supports multiple parameterizations with automatic conversions.
- Analytical implementations are provided where available, with fallbacks to numerical methods.
- Families are designed to be extensible with additional characteristics and parameterizations.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from functools import lru_cache

from pysatl_core.families.builtins import (
    configure_normal_family,
    configure_uniform_family,
)
from pysatl_core.families.registry import ParametricFamilyRegister


@lru_cache(maxsize=1)
def configure_families_register() -> ParametricFamilyRegister:
    """
    Configure and register all distribution families in the global registry.

    This function initializes all parametric families with their respective
    parameterizations, characteristics, and sampling strategies. It should be
    called during application startup to make distributions available.

    Returns
    -------
    ParametricFamilyRegister
        The global registry of parametric families.
    """
    configure_normal_family()
    configure_uniform_family()
    return ParametricFamilyRegister()


def reset_families_register() -> None:
    """
    Reset the cached families registry.
    """
    configure_families_register.cache_clear()
    ParametricFamilyRegister._reset()
