"""
Built-in continuous distribution families.

This module contains implementations of continuous parametric families.
"""

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_core.families.builtins.continuous.exponential import configure_exponential_family
from pysatl_core.families.builtins.continuous.normal import configure_normal_family
from pysatl_core.families.builtins.continuous.uniform import configure_uniform_family

__all__ = [
    "configure_normal_family",
    "configure_uniform_family",
    "configure_exponential_family",
]
