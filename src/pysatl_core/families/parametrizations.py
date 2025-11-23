"""
Parameterization classes and specifications for distribution families.

This module provides the core classes for defining different parameterizations
of statistical distributions, including constraints validation and conversion
between parameterization formats.
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC
from dataclasses import dataclass, is_dataclass
from functools import wraps
from inspect import isfunction
from typing import TYPE_CHECKING, ParamSpec

from pysatl_core.types import ParametrizationName

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, ClassVar

    from pysatl_core.families.parametric_family import ParametricFamily


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

    # These class attributes are set by the @parametrization decorator.
    __family__: ClassVar[ParametricFamily]
    __param_name__: ClassVar[ParametrizationName]

    _constraints: ClassVar[list[ParametrizationConstraint]] = []

    @property
    def name(self) -> str:
        """
        Get the name of this parametrization.

        Returns
        -------
        str
            The name of the parametrization.
        """
        return self.__class__.__param_name__

    @property
    def parameters(self) -> dict[str, Any]:
        """
        Get the parameters as a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping parameter names to values.
        """
        fields = getattr(self, "__dataclass_fields__", None)
        if fields:
            return {f: getattr(self, f) for f in fields}
        ann = getattr(self, "__annotations__", {})
        return {k: getattr(self, k) for k in ann}

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


P = ParamSpec("P")


def constraint(description: str) -> Callable[[Callable[P, bool]], Callable[P, bool]]:
    """
    Decorator to mark an instance method as a parameter constraint.

    The decorated function must be a predicate returning ``bool``. At class
    decoration time it will be discovered and attached as a
    :class:`ParametrizationConstraint`.

    Parameters
    ----------
    description : str
        Human-readable description of the constraint.

    Returns
    -------
    Callable[[Callable[P, bool]], Callable[P, bool]]
        A decorator that returns the function wrapper with two marker
        attributes set on it.

    Notes
    -----
    The following marker attributes are set on the resulting function object:

    * ``__is_constraint`` : ``True``
    * ``__constraint_description`` : ``str``

    Examples
    --------
    >>> class MeanStd(Parametrization):
    ...     mean: float
    ...     sigma: float
    ...
    ...     @constraint("sigma > 0")
    ...     def _c_sigma_positive(self) -> bool:
    ...         return self.sigma > 0
    """

    def decorator(func: Callable[P, bool]) -> Callable[P, bool]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> bool:
            return func(*args, **kwargs)

        setattr(wrapper, "__is_constraint", True)
        setattr(wrapper, "__constraint_description", description)
        return wrapper

    return decorator


def parametrization(
    *,
    family: ParametricFamily,
    name: str,
) -> Callable[[type[Parametrization]], type[Parametrization]]:
    """
    Decorator to register a class as a parametrization for a family.

    Parameters
    ----------
    family : ParametricFamily
        The family to register the parametrization with.
    name : str
        Name of the parametrization.

    Returns
    -------
    Callable[[type[Parametrization]], type[Parametrization]]
        A class decorator that registers the parametrization and returns the class.

    Examples
    --------
    >>> @parametrization(family=normal, name="mean_var")
    ... class MeanVar(Parametrization):
    ...     mean: float
    ...     var: float
    """

    def _collect_constraints(
        cls: type[Parametrization],
    ) -> list[ParametrizationConstraint]:
        """
        Collect constraint methods declared on the class.

        Parameters
        ----------
        cls : type[Parametrization]
            Class being registered as a parametrization.

        Returns
        -------
        list[ParametrizationConstraint]
            Collected constraints in declaration order.

        Raises
        ------
        TypeError
            If a constraint is declared as ``@staticmethod`` or ``@classmethod``.
        """
        constraints: list[ParametrizationConstraint] = []
        for name, attr in cls.__dict__.items():
            if isinstance(attr, staticmethod):
                raise TypeError(
                    f"@constraint '{name}' must be an instance method, not @staticmethod"
                )
            if isinstance(attr, classmethod):
                raise TypeError(
                    f"@constraint '{name}' must be an instance method, not @classmethod"
                )

            func = attr if callable(attr) and isfunction(attr) else None
            if not func:
                continue
            if getattr(func, "__is_constraint", False):
                desc = getattr(func, "__constraint_description", func.__name__)
                constraints.append(ParametrizationConstraint(description=desc, check=func))
        return constraints

    def decorator(cls: type[Parametrization]) -> type[Parametrization]:
        if not is_dataclass(cls):
            cls = dataclass(slots=True, frozen=True)(cls)

        # Attach metadata expected by tooling; declared in base class for mypy.
        cls.__family__ = family
        cls.__param_name__ = name

        # Discover and store constraints.
        constraints = _collect_constraints(cls)
        cls._constraints = constraints

        # Register in the family's spec.
        family.register_parametrization(name, cls)
        return cls

    return decorator
