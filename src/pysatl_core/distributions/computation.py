"""
Computation Primitives and Conversions

Core building blocks for computing distribution characteristics and
conversions between them (e.g., PDF to CDF, CDF to PPF).
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, overload, runtime_checkable

from pysatl_core.types import ComputationFunc

if TYPE_CHECKING:
    from typing import Any

    from mypy_extensions import KwArg

    from pysatl_core.distributions.distribution import Distribution
    from pysatl_core.types import GenericCharacteristicName


@runtime_checkable
class Computation[In, Out](Protocol):
    """
    Protocol for computations that evaluate a single characteristic.

    Attributes
    ----------
    target : str
        Name of the characteristic this computation produces.
    """

    @property
    def target(self) -> GenericCharacteristicName: ...

    @overload
    def __call__(self, **kwargs: Any) -> Out: ...

    @overload
    def __call__(self, x: In, **kwargs: Any) -> Out: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Out: ...


@dataclass(frozen=True, slots=True)
class AnalyticalComputation[In, Out]:
    """
    Analytical computation provided directly by a distribution.

    Parameters
    ----------
    target : str
        Characteristic name (e.g., "pdf", "cdf").
    func : Callable[..., Out]
        Analytical function that computes the characteristic.
    """

    target: GenericCharacteristicName
    func: ComputationFunc[In, Out]

    @overload
    def __call__(self, **options: Any) -> Out: ...

    @overload
    def __call__(self, data: In, **options: Any) -> Out: ...

    def __call__(self, *args: Any, **options: Any) -> Out:
        """Evaluate the analytical function."""
        return self.func(*args, **options)


@dataclass(frozen=True, slots=True)
class FittedComputationMethod[In, Out]:
    """
    Fitted conversion method ready for use.

    Parameters
    ----------
    target : str
        Destination characteristic name.
    sources : Sequence[str]
        Source characteristic names (typically length 1 for unary conversions).
    func : Callable[..., Out]
        Callable implementing the fitted conversion.
    """

    target: GenericCharacteristicName
    sources: Sequence[GenericCharacteristicName]
    func: ComputationFunc[In, Out]

    @overload
    def __call__(self, **options: Any) -> Out: ...

    @overload
    def __call__(self, data: In, **options: Any) -> Out: ...

    def __call__(self, *args: Any, **options: Any) -> Out:
        """Evaluate the fitted conversion."""
        return self.func(*args, **options)


@dataclass(frozen=True, slots=True)
class ComputationMethod[In, Out]:
    """
    Factory for creating fitted computation methods.

    This class represents a conversion method that needs to be fitted to a
    specific distribution before it can be used.

    Parameters
    ----------
    target : str
        Destination characteristic name.
    sources : Sequence[str]
        Source characteristic names (typically length 1 for unary conversions).
    fitter : Callable[[Distribution, **options], FittedComputationMethod]
        Function that fits the computation method to a distribution.
    """

    target: GenericCharacteristicName
    sources: Sequence[GenericCharacteristicName]
    fitter: Callable[[Distribution, KwArg(Any)], FittedComputationMethod[In, Out]]

    def fit(self, distribution: Distribution, **options: Any) -> FittedComputationMethod[In, Out]:
        """
        Fit the computation method to a specific distribution.

        Parameters
        ----------
        distribution : Distribution
            Distribution to fit the computation method to.
        **options : Any
            Additional options passed to the fitter.

        Returns
        -------
        FittedComputationMethod
            Fitted method ready for evaluation.
        """
        return self.fitter(distribution, **options)


type Method[In, Out] = AnalyticalComputation[In, Out] | FittedComputationMethod[In, Out]
