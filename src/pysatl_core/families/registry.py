"""
Global registry for parametric distribution families using singleton pattern.

This module implements a centralized registry that maintains references to all
defined parametric families, enabling easy access and management across the
application. The registry follows the singleton pattern to ensure consistency.
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import ClassVar

    from pysatl_core.families.parametric_family import ParametricFamily


class ParametricFamilyRegister:
    """
    Singleton registry for parametric distribution families.

    This class maintains a global registry of all parametric families,
    allowing them to be accessed by name from anywhere in the codebase.

    Examples
    --------
    >>> registry = ParametricFamilyRegister()
    >>> family = registry.get('Lognormal Family')
    """

    _instance: ClassVar[ParametricFamilyRegister | None] = None
    _registered_families: dict[str, ParametricFamily]

    def __new__(cls) -> ParametricFamilyRegister:
        """
        Create or return the singleton instance.

        Returns
        -------
        ParametricFamiliesRegister
            The singleton registry instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registered_families = {}
        return cls._instance

    @classmethod
    def get(cls, name: str) -> ParametricFamily:
        """
        Retrieve a parametric family by name.

        Parameters
        ----------
        name : str
            The name of the family to retrieve.

        Returns
        -------
        ParametricFamily
            The requested parametric family.

        Raises
        ------
        ValueError
            If no family with the given name exists in the registry.
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
            If family with the same name already was registered
        """
        self = cls()
        if family.name in self._registered_families:
            raise ValueError(f"Family {family.name} already found in register")
        self._registered_families[family.name] = family


def _reset_families_register_for_tests() -> None:
    """Reset the cached distribution type register (test helper)."""
    ParametricFamilyRegister._instance = None
