"""
Global registry for parametric distribution families using singleton pattern.

This module implements a centralized registry that maintains references to all
defined parametric families, enabling easy access and management across the
application.
"""

from __future__ import annotations

from pysatl_core.families.fixed_parametrization_edge import EdgeWithFixedParametrization
from pysatl_core.families.parametrizations import Parametrization

__author__ = "Leonid Elkin, Mikhail Mikhailov, Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import ClassVar

    from pysatl_core.families.parametric_family import ParametricFamily


class ParametricFamilyRegister:
    """
    Singleton registry for parametric distribution families.

    Maintains a global registry of all parametric families, allowing
    them to be accessed by name.
    """

    _instance: ClassVar[ParametricFamilyRegister | None] = None
    _registered_families: dict[str, ParametricFamily]
    _registered_transformations: dict[str, list[EdgeWithFixedParametrization]]

    def __new__(cls) -> ParametricFamilyRegister:
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registered_families = {}
            cls._instance._registered_transformations = {}
        return cls._instance

    @classmethod
    def get_optimal_family(
        cls, name: str, parametrization: Parametrization
    ) -> None | tuple[EdgeWithFixedParametrization, ParametricFamily]:
        self = cls()

        for optimization_edge in self._registered_transformations.get(name, []):
            if (
                optimization_edge.tail_name in self._registered_families
                and optimization_edge.is_transoform_possible(parametrization)
            ):
                return optimization_edge, self._registered_families[optimization_edge.tail_name]

        return None

    @classmethod
    def add_optimization_edge(
        cls,
        head_name: str,
        tail_name: str,
        transform_constraint: Callable[[Parametrization], bool],
        transform_function: Callable[[Parametrization], Parametrization],
    ) -> bool:
        self = cls()

        if head_name in self._registered_families:
            self._registered_transformations.setdefault(head_name, []).append(
                EdgeWithFixedParametrization(tail_name, transform_constraint, transform_function)
            )
            return True

        return False

    @classmethod
    def get(cls, name: str) -> ParametricFamily:
        """
        Retrieve a parametric family by name.

        Parameters
        ----------
        name : str
            Name of the family to retrieve.

        Returns
        -------
        ParametricFamily
            The requested parametric family.

        Raises
        ------
        ValueError
            If no family with the given name exists.
        """
        self = cls()
        if name not in self._registered_families:
            raise ValueError(f"No family {name} found in register")
        return self._registered_families[name]

    @classmethod
    def register(cls, family: ParametricFamily) -> None:
        """
        Register a new parametric family.

        Parameters
        ----------
        family : ParametricFamily
            The family to register.

        Raises
        ------
        ValueError
            If a family with the same name is already registered.
        """
        self = cls()
        if family.name in self._registered_families:
            raise ValueError(f"Family {family.name} already found in register")
        self._registered_families[family.name] = family

    @classmethod
    def contains(cls, name: str) -> bool:
        """
        Check whether a parametric family with the given name
        is registered in the registry.

        Parameters
        ----------
        name : str
            Name of the parametric family.

        Returns
        -------
        bool
            ``True`` if the family is present in the registry,
            ``False`` otherwise.
        """
        self = cls()
        return name in self._registered_families

    @classmethod
    def _reset(cls) -> None:
        """
        Clear the registry (for testing purposes).

        This method removes all registered families and resets the singleton instance.
        It should only be used in tests.
        """
        if cls._instance is not None:
            cls._instance._registered_families.clear()
        cls._instance = None

    @classmethod
    def list_registered_families(cls) -> list[str]:
        if cls._instance is None:
            return []
        return list(cls._instance._registered_families.keys())
