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
from typing import TYPE_CHECKING, Protocol, cast, overload, runtime_checkable

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
    fitter : Fitter[In, Out] | None
        Function that fits the computation method to a distribution.
        If provided, the method is considered *cacheable* (fitting may perform
        expensive precomputation).
    evaluator : Evaluator[In, Out] | None
        Direct evaluator that performs the computation in one step, without
        a separate fitting stage. If provided, the method is considered
        *non-cacheable* at the strategy level.
    """

    target: GenericCharacteristicName
    sources: Sequence[GenericCharacteristicName]
    fitter: Fitter[In, Out] | None = None
    evaluator: Evaluator[In, Out] | None = None

    def __post_init__(self) -> None:
        has_fitter = self.fitter is not None
        has_eval = self.evaluator is not None
        if has_fitter == has_eval:
            raise ValueError(
                "ComputationMethod must define exactly one of 'fitter' or 'evaluator'."
            )

    @property
    def cacheable(self) -> bool:
        """Whether it makes sense to cache the prepared method at strategy level."""
        return self.fitter is not None

    def prepare(
        self, distribution: Distribution, **options: Any
    ) -> FittedComputationMethod[In, Out]:
        """Prepare a callable method for a specific distribution.

        - If ``fitter`` is provided, run the fitting stage and return the fitted method.
        - If ``evaluator`` is provided, bind the distribution and return a lightweight
          fitted wrapper.
        """
        if self.fitter is not None:
            return self.fitter(distribution, **options)

        def _bound(*args: Any, **kwargs: Any) -> Out:
            return cast(Evaluator[In, Out], self.evaluator)(distribution, *args, **kwargs)

        return FittedComputationMethod[In, Out](
            target=self.target,
            sources=self.sources,
            func=_bound,
        )

    @overload
    def evaluate(self, distribution: Distribution, **options: Any) -> Out: ...

    @overload
    def evaluate(self, distribution: Distribution, data: In, **options: Any) -> Out: ...

    def evaluate(self, distribution: Distribution, *args: Any, **options: Any) -> Out:
        """Evaluate *direct* computation methods.

        This is only available for methods defined via ``evaluator``.
        """
        if self.evaluator is None:
            raise RuntimeError(
                "This ComputationMethod requires fitting. "
                "Call .fit(...) / .prepare(...) to obtain a callable."
            )
        return self.evaluator(distribution, *args, **options)

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
        if self.fitter is None:
            raise RuntimeError(
                "This ComputationMethod is evaluator-based and does not support .fit(). "
                "Use .prepare(...) or call the method directly."
            )
        return self.fitter(distribution, **options)

    @overload
    def __call__(self, distribution: Distribution, **options: Any) -> Out: ...

    @overload
    def __call__(self, distribution: Distribution, data: In, **options: Any) -> Out: ...

    def __call__(self, distribution: Distribution, *args: Any, **options: Any) -> Out:
        """Fit if possible and then evaluate"""
        return self.prepare(distribution, **options)(*args)


type Method[In, Out] = AnalyticalComputation[In, Out] | FittedComputationMethod[In, Out]
