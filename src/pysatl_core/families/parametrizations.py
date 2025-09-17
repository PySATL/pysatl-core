"""
Parameterization classes and specifications for distribution families.

This module provides the core classes for defining different parameterizations
of statistical distributions, including constraints validation and conversion
between parameterization formats.
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail, Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from pysatl_core.types import ParametrizationName

if TYPE_CHECKING:
    from pysatl_core.families.parametric_family import ParametricFamily


@runtime_checkable
class ParametrizationConstraintProtocol(Protocol):
    @property
    def _is_constraint(self) -> bool: ...
    @property
    def _constraint_description(self) -> str: ...

    def __call__(self, **kwargs: Any) -> bool: ...


@dataclass(slots=True, frozen=True)
class ParametrizationConstraint:
    """
    A constraint on parameter values for a parametrization.

    Attributes
    ----------
    description : str
        Human-readable description of the constraint.
    check : Callable[[Any], bool]
        Function that validates the constraint, returning True if satisfied.
    """

    description: str
    check: Callable[[Any], bool]


class Parametrization(ABC):
    """
    Abstract base class for distribution parametrizations.

    This class defines the interface that all parametrizations must implement,
    including parameter validation and conversion to base parametrization.

    Attributes
    ----------
    constraints : ClassVar[List[ParametrizationConstraint]]
        Class-level list of constraints that apply to this parametrization.
    """

    _constraints: ClassVar[list[ParametrizationConstraint]] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of this parametrization.

        Returns
        -------
        str
            The name of the parametrization.
        """

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """
        Get the parameters as a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping parameter names to values.
        """

    @property
    def constraints(self) -> list[ParametrizationConstraint]:
        """
        Get the constraints for this parametrization.

        Returns
        -------
        List[ParametrizationConstraint]
            List of constraints that apply to this parametrization.
        """
        return self._constraints

    def validate(self) -> None:
        """
        Validate all constraints for this parametrization.

        Raises
        ------
        ValueError
            If any constraint is not satisfied.
        """
        for constraint in self._constraints:
            if not constraint.check(self):
                raise ValueError(f'Constraint "{constraint.description}" does not hold')

    def transform_to_base_parametrization(self) -> Parametrization:
        """
        Convert this parametrization to the base parametrization.

        Returns
        -------
        Parametrization
            The equivalent parameters in the base parametrization.

        Notes
        -----
        The base implementation returns self, assuming this is already
        the base parametrization. Subclasses should override this method
        if they need to convert to a different parametrization.
        """
        return self


class ParametrizationSpec:
    """
    Container for all parametrizations of a distribution family.

    This class manages the collection of parametrizations for a family
    and handles conversions between them.

    Attributes
    ----------
    parametrizations : Dict[ParametrizationName, Type[Parametrization]]
        Mapping from parametrization names to parametrization classes.
    base_parametrization_name : ParametrizationName | None
        Name of the base parametrization, if defined.
    """

    def __init__(self) -> None:
        """Initialize an empty parametrization specification."""
        self.parametrizations: dict[ParametrizationName, type[Parametrization]] = {}
        self.base_parametrization_name: ParametrizationName | None = None

    @property
    def base(self) -> type[Parametrization]:
        """
        Get the base parametrization class.

        Returns
        -------
        Type[Parametrization]
            The base parametrization class.

        Raises
        ------
        ValueError
            If no base parametrization has been defined.
        """
        if self.base_parametrization_name is None:
            raise ValueError("No base parametrization defined")
        return self.parametrizations[self.base_parametrization_name]

    def add_parametrization(
        self,
        name: ParametrizationName,
        parametrization_class: type[Parametrization],
    ) -> None:
        """
        Add a new parametrization to the specification.

        Parameters
        ----------
        name : ParametrizationName
            Name of the parametrization.
        parametrization_class : Type[Parametrization]
            Class implementing the parametrization.
        """
        self.parametrizations[name] = parametrization_class

    def get_base_parameters(self, parameters: Parametrization) -> Parametrization:
        """
        Convert parameters to the base parametrization.

        Parameters
        ----------
        parameters : Parametrization
            Parameters in any parametrization.

        Returns
        -------
        Parametrization
            Equivalent parameters in the base parametrization.
        """
        if parameters.name == self.base_parametrization_name:
            return parameters
        else:
            return parameters.transform_to_base_parametrization()


# Decorators for declarative syntax
def constraint(
    description: str,
) -> Callable[[Callable[[Any], bool]], ParametrizationConstraintProtocol]:
    """
    Decorator to mark a method as a parameter constraint.

    Parameters
    ----------
    description : str
        Human-readable description of the constraint.

    Returns
    -------
    Callable
        Decorator function that marks the method as a constraint.

    Examples
    --------
    >>> @constraint("sigma > 0")
    >>> def check_sigma_positive(self):
    >>>     return self.sigma > 0
    """

    def decorator(func: Callable[[Any], bool]) -> ParametrizationConstraintProtocol:
        @wraps(func)
        def wrapper(*args, **kwargs):  # type: ignore
            return func(*args, **kwargs)

        wrapper._is_constraint = True  # type: ignore
        wrapper._constraint_description = description  # type: ignore
        return wrapper  # type: ignore

    return decorator


def parametrization(
    family: ParametricFamily, name: str
) -> Callable[[type[Parametrization]], type[Parametrization]]:
    """
    Decorator to register a class as a parametrization for a family.

    Parameters
    ----------
    family : ParametricFamily
        The family to register the parametrization with.
    name : str
        Name of the parametrization.
    base : bool, optional
        Whether this is the base parametrization, by default False.

    Returns
    -------
    Callable
        Decorator function that registers the class as a parametrization.

    Examples
    --------
    >>> @parametrization(family=NormalFamily, name='meanvar')
    >>> class MeanVarParametrization:
    >>>     mean: float
    >>>     var: float
    """

    def decorator(cls: type[Parametrization]) -> type[Parametrization]:
        # Convert to dataclass if not already
        if not hasattr(cls, "__dataclass_fields__"):
            cls = dataclass(cls)

        # Add name property
        def name_property(self):  # type: ignore
            return name

        cls.name = property(name_property)  # type: ignore

        # Add parameters property
        def parameters_property(self):  # type: ignore
            return {
                field.name: getattr(self, field.name)
                for field in self.__dataclass_fields__.values()
            }

        cls.parameters = property(parameters_property)  # type: ignore

        # Collect constraints
        constraints = []
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if hasattr(attr, "_is_constraint") and attr._is_constraint:
                constraints.append(
                    ParametrizationConstraint(description=attr._constraint_description, check=attr)
                )
        cls._constraints = constraints

        # Add validate method
        cls.validate = Parametrization.validate  # type: ignore

        # Register with family
        family.parametrizations.add_parametrization(name, cls)

        return cls

    return decorator
