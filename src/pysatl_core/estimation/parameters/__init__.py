"""
The parameter space: how it is written down, where it is bounded, where to start.

Nothing here knows what is being estimated.  ``vectors`` translates between a
family's named parameters and the flat array an optimizer works on, ``bounds``
turns a family's declaration into the region a search may occupy, and
``start`` answers where that search begins.
"""

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_core.estimation.parameters.bounds import ParameterBox
from pysatl_core.estimation.parameters.start import (
    MomentRule,
    project_onto_base,
    starting_point,
)
from pysatl_core.estimation.parameters.vectors import (
    field_names,
    from_vector,
    satisfies_constraints,
    to_vector,
)

__all__ = [
    "MomentRule",
    "ParameterBox",
    "field_names",
    "from_vector",
    "project_onto_base",
    "satisfies_constraints",
    "starting_point",
    "to_vector",
]
