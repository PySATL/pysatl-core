"""
Computation Primitives and Conversions

Core building blocks for computing distribution characteristics and
conversions between them (e.g., PDF to CDF, CDF to PPF).

This module provides:
- ``FitterMethod``: cacheable computation that performs expensive precomputation
- ``EvaluatorMethod``: lightweight direct computation called on every query
"""

from __future__ import annotations

__author__ = "Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, overload

from pysatl_core.types import ComputationFunc

if TYPE_CHECKING:
    from pysatl_core.distributions.distribution import Distribution
    from pysatl_core.types import (
        EvaluatorFunc,
        FitterFunc,
        GenericCharacteristicName,
        NumericArray,
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
            return self.evaluator(distribution, *args, **kwargs)

        return FittedComputationMethod[NumericArray, NumericArray](
            target=self.target,
            sources=list(self.sources),
            func=_bound,
        )


type ComputationMethodUnion = FitterMethod | EvaluatorMethod
"""Union type for computation methods (fitter or evaluator)."""


__all__ = [
    "FittedComputationMethod",
    "FitterMethod",
    "EvaluatorMethod",
    "ComputationMethodUnion",
]
