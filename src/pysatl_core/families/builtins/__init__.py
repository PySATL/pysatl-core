"""
Built-in distribution families for PySATL.

This package contains implementations of standard statistical distribution families
that are available by default in PySATL.
"""

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_core.families.builtins.continuous import (
    configure_exponential_family,
    configure_normal_family,
    configure_uniform_family,
)

__all__ = [
    "configure_normal_family",
    "configure_uniform_family",
    "configure_exponential_family",
]
