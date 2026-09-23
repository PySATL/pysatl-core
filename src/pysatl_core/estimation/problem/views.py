"""
How a family's parameters are presented: which are pinned, and in which coordinates.

Both questions are about the *surface* of a family rather than about any
criterion, so both are answered here, once, for every estimation method.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pysatl_core.families.parametric_family import ParametricFamily
    from pysatl_core.families.parametrizations import Parametrization
    from pysatl_core.types import ParametrizationName


@dataclass(frozen=True, slots=True)
class FixedParameters:
    """
    Parameters pinned through ``view``, and the coordinates they were pinned in.

    A named pair rather than a bare tuple: at the call site ``fixed.values`` and
    ``fixed.in_base_parametrization`` say what they are, where the second
    element of a ``tuple[Mapping[str, float], bool]`` said only ``True``.
    """

    values: Mapping[str, float]
    """The fixed values, keyed by parameter name."""

    in_base_parametrization: bool
    """Whether they were fixed in the parent family's *base* parametrization.

    A closed-form rule is written against the base parametrization, so it
    cannot be handed values expressed in any other.
    """


def fixed_parameters(family: ParametricFamily) -> FixedParameters:
    """Report which parameters a view has pinned, and in which coordinates."""
    from pysatl_core.families.parametric_family import PartialParametricFamily

    if not isinstance(family, PartialParametricFamily):
        return FixedParameters(values=MappingProxyType({}), in_base_parametrization=True)
    # A view's own ``base_parametrization_name`` is the parametrization the
    # parameters were fixed in: ``PartialParametricFamily`` registers exactly
    # that one and nothing else.
    return FixedParameters(
        values=family.fixed_parameters,
        in_base_parametrization=(
            family.base_parametrization_name == family.parent_family.base_parametrization_name
        ),
    )


def convert_parametrization(
    family: ParametricFamily,
    params: Parametrization,
    parametrization: ParametrizationName | None,
) -> Parametrization:
    """
    Express the estimate in the parametrization the caller asked for.

    Raises
    ------
    NotImplementedError
        For any parametrization other than the base one.
    """
    if parametrization is None or parametrization == family.base_parametrization_name:
        return params
    raise NotImplementedError(
        f"Cannot return the estimate in parametrization '{parametrization}': converting "
        f"from the base parametrization '{family.base_parametrization_name}' requires an "
        f"inverse transform, and the 'Parametrization' API offers only "
        f"'transform_to_base_parametrization'. Fit in the base parametrization and convert "
        f"the values by hand, or add the inverse transform to the family's "
        f"parametrization class."
    )


__all__ = ["FixedParameters", "convert_parametrization", "fixed_parameters"]
