"""
Computation Primitives and Conversions

Core building blocks for computing distribution characteristics and
conversions between them (e.g., PDF to CDF, CDF to PPF).

This module provides:
- ``FittedComputationMethod``: fitted conversion method ready for use
- ``FitterMethod``: cacheable computation that performs expensive precomputation
- ``EvaluatorMethod``: lightweight direct computation called on every query
- ``AnalyticalComputation``: analytical computation provided directly by a distribution
- ``Computation``: protocol for computations that evaluate a single characteristic
"""

from __future__ import annotations

__author__ = "Irina Sergeeva, Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, overload, runtime_checkable

from pysatl_core.types import ComputationFunc, NumericArray

if TYPE_CHECKING:
    from mypy_extensions import KwArg

    from pysatl_core.distributions.distribution import Distribution
    from pysatl_core.types import (
        EvaluatorFunc,
        FitterFunc,
        GenericCharacteristicName,
    )


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
    func : ComputationFunc[In, Out]
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
class FitterMethod:
    """
    Cacheable computation method that performs expensive precomputation.

    A fitter is called once per distribution to produce a
    ``FittedComputationMethod`` that can be cached and reused for
    subsequent evaluations.

    Parameters
    ----------
    target : str
        Destination characteristic name.
    sources : Sequence[str]
        Source characteristic names (typically length 1 for unary conversions).
    fitter : FitterFunc
        Function that fits the computation method to a distribution.
    """

    target: GenericCharacteristicName
    sources: Sequence[GenericCharacteristicName]
    fitter: FitterFunc

    @property
    def cacheable(self) -> bool:
        """Whether it makes sense to cache the prepared method at strategy level."""
        return True

    def fit(
        self, distribution: Distribution, **options: Any
    ) -> FittedComputationMethod[NumericArray, NumericArray]:
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

    def prepare(
        self, distribution: Distribution, **options: Any
    ) -> FittedComputationMethod[NumericArray, NumericArray]:
        """Alias for :meth:`fit`."""
        return self.fit(distribution, **options)


@dataclass(frozen=True, slots=True)
class EvaluatorMethod:
    """
    Lightweight direct computation method called on every query.

    An evaluator does not perform expensive precomputation and returns
    the computed value directly rather than a ``FittedComputationMethod``.

    Parameters
    ----------
    target : str
        Destination characteristic name.
    sources : Sequence[str]
        Source characteristic names (typically length 1 for unary conversions).
    evaluator : EvaluatorFunc
        Direct evaluator callable.
    """

    target: GenericCharacteristicName
    sources: Sequence[GenericCharacteristicName]
    evaluator: EvaluatorFunc

    @property
    def cacheable(self) -> bool:
        """Evaluators are not cacheable."""
        return False

    @overload
    def evaluate(self, distribution: Distribution, **options: Any) -> NumericArray: ...

    @overload
    def evaluate(
        self, distribution: Distribution, data: NumericArray, **options: Any
    ) -> NumericArray: ...

    def evaluate(self, distribution: Distribution, *args: Any, **options: Any) -> NumericArray:
        """
        Evaluate the computation directly.

        Parameters
        ----------
        distribution : Distribution
            Distribution to evaluate for.
        *args : Any
            Optional positional data argument.
        **options : Any
            Additional options.

        Returns
        -------
        NumericArray
            Computed result.
        """
        return self.evaluator(distribution, *args, **options)

    def prepare(
        self, distribution: Distribution, **options: Any
    ) -> FittedComputationMethod[NumericArray, NumericArray]:
        """
        Create a lightweight fitted wrapper that binds the distribution.

        This allows evaluator-based methods to be used in the same way
        as fitter-based methods when needed.
        """

        def _bound(*args: Any, **kwargs: Any) -> NumericArray:
            return self.evaluator(distribution, *args, **{**options, **kwargs})

        return FittedComputationMethod[NumericArray, NumericArray](
            target=self.target,
            sources=list(self.sources),
            func=_bound,
        )


type ComputationMethodUnion = FitterMethod | EvaluatorMethod
"""Union type for computation methods (fitter or evaluator)."""

if TYPE_CHECKING:
    type Fitter[In, Out] = Callable[[Distribution, KwArg(Any)], FittedComputationMethod[In, Out]]
    type Evaluator[In, Out] = (
        Callable[[Distribution, KwArg(Any)], Out] | Callable[[Distribution, In, KwArg(Any)], Out]
    )
else:
    type Fitter[In, Out] = Callable[..., Any]
    type Evaluator[In, Out] = Callable[..., Any]


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


type Method[In, Out] = AnalyticalComputation[In, Out] | FittedComputationMethod[In, Out]


__all__ = [
    "AnalyticalComputation",
    "Computation",
    "ComputationMethodUnion",
    "Evaluator",
    "EvaluatorMethod",
    "Fitter",
    "FittedComputationMethod",
    "FitterMethod",
    "Method",
]
