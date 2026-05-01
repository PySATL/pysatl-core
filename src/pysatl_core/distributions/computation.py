"""
Computation Primitives and Conversions (backward-compatibility shim).

This module re-exports from the new ``computations`` subpackage.
New code should import directly from ``pysatl_core.distributions.computations.computation``.
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, overload, runtime_checkable

from pysatl_core.distributions.computations.computation import (
    ComputationMethodUnion,
    EvaluatorMethod,
    FittedComputationMethod,
    FitterMethod,
)
from pysatl_core.types import ComputationFunc

if TYPE_CHECKING:
    from typing import Any

    from mypy_extensions import KwArg

    from pysatl_core.distributions.distribution import Distribution
    from pysatl_core.types import GenericCharacteristicName


type Fitter[In, Out] = Callable[[Distribution, KwArg(Any)], FittedComputationMethod[In, Out]]
type Evaluator[In, Out] = (
    Callable[[Distribution, KwArg(Any)], Out] | Callable[[Distribution, In, KwArg(Any)], Out]
)


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
    func : ComputationFunc[In, Out]
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


# Re-export ComputationMethod as a backward-compatible alias.
# New code should use FitterMethod or EvaluatorMethod directly.
ComputationMethod = FitterMethod

type Method[In, Out] = AnalyticalComputation[In, Out] | FittedComputationMethod[In, Out]

__all__ = [
    "AnalyticalComputation",
    "Computation",
    "ComputationMethod",
    "ComputationMethodUnion",
    "EvaluatorMethod",
    "FittedComputationMethod",
    "FitterMethod",
    "Method",
]
