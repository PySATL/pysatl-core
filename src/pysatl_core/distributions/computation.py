"""
Computation Primitives and Conversions
======================================

This module defines the core building blocks used to compute distribution
characteristics:

- :class:`Computation` — callable (analytical or fitted) for a single
  characteristic.
- :class:`FittedComputationMethod` — a fitted conversion method
  (e.g., from CDF to PDF) ready to be called.
- :class:`ComputationMethod` — a factory that *fits* a conversion given a
  distribution and returns :class:`FittedComputationMethod`.
- :class:`AnalyticalComputation` — an analytical callable provided by a
  distribution directly.

It also exposes canonical univariate continuous conversions:

- ``pdf_to_cdf_1C``
- ``cdf_to_pdf_1C``
- ``cdf_to_ppf_1C``
- ``ppf_to_cdf_1C``

Notes
-----
- All callables are intentionally **scalar** (``float -> float``) in the
  univariate case. Vectorization, if needed, should be handled outside or by
  the caller.
- ``**options`` in fitters are free-form and may contain numeric tolerances,
  disambiguation flags (e.g., ``most_left``), etc.
"""

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from mypy_extensions import KwArg

from pysatl_core.types import (
    GenericCharacteristicName,
)

if TYPE_CHECKING:
    from pysatl_core.distributions.distribution import Distribution


@runtime_checkable
class Computation[In, Out](Protocol):
    """Callable for a single characteristic.

    Attributes
    ----------
    target : str
        The characteristic name this computation represents.

    Methods
    -------
    __call__(data, **options)
        Evaluate the characteristic at ``data``.
    """

    @property
    def target(self) -> GenericCharacteristicName: ...
    def __call__(self, data: In, **options: Any) -> Out: ...


@runtime_checkable
class FittedComputationMethodProtocol[In, Out](Protocol):
    """Protocol for a fitted conversion method.

    Attributes
    ----------
    target : str
        Destination characteristic.
    sources : Sequence[str]
        Source characteristics this method depends on.
    """

    @property
    def target(self) -> GenericCharacteristicName: ...
    @property
    def sources(self) -> Sequence[GenericCharacteristicName]: ...
    def __call__(self, data: In, **options: Any) -> Out: ...


@runtime_checkable
class ComputationMethodProtocol[In, Out](Protocol):
    """Protocol for a conversion method factory (to be fitted)."""

    @property
    def target(self) -> GenericCharacteristicName: ...
    @property
    def sources(self) -> Sequence[GenericCharacteristicName]: ...
    def fit(
        self, distribution: "Distribution", **options: Any
    ) -> FittedComputationMethodProtocol[In, Out]: ...


@dataclass(frozen=True, slots=True)
class AnalyticalComputation[In, Out]:
    """Analytical computation provided directly by the distribution.

    Parameters
    ----------
    target : str
        Characteristic name (e.g., ``"pdf"``).
    func : Callable[[In, KwArg(Any)], Out]
        Analytical callable.

    Notes
    -----
    The callable is expected to be scalar for univariate distributions.
    """

    target: GenericCharacteristicName
    func: Callable[[In, KwArg(Any)], Out]

    def __call__(self, data: In, **options: Any) -> Out:
        """Evaluate the analytical function."""
        return self.func(data, **options)


@dataclass(frozen=True, slots=True)
class FittedComputationMethod[In, Out]:
    """Fitted conversion method (ready-to-use).

    Parameters
    ----------
    target : str
        Destination characteristic name.
    sources : Sequence[str]
        Source characteristic names (unary conversions use length 1).
    func : Callable[[In, KwArg(Any)], Out]
        Callable implementing the fitted conversion.
    """

    target: GenericCharacteristicName
    sources: Sequence[GenericCharacteristicName]
    func: Callable[[In, KwArg(Any)], Out]

    def __call__(self, data: In, **options: Any) -> Out:
        """Evaluate the fitted conversion."""
        return self.func(data, **options)


@dataclass(frozen=True, slots=True)
class ComputationMethod[In, Out]:
    """Conversion method factory (to be fitted).

    Parameters
    ----------
    target : str
        Destination characteristic name.
    sources : Sequence[str]
        Source characteristic names (unary for current graph edges).
    fitter : Callable[[Distribution, KwArg(Any)], FittedComputationMethod]
        Fitter that prepares a callable conversion for the given distribution.

    Methods
    -------
    fit(distribution, **options)
        Fit and return a :class:`FittedComputationMethod`.
    """

    target: GenericCharacteristicName
    sources: Sequence[GenericCharacteristicName]
    fitter: Callable[["Distribution", KwArg(Any)], FittedComputationMethod[In, Out]]

    def fit(self, distribution: "Distribution", **options: Any) -> FittedComputationMethod[In, Out]:
        """Fit and return a :class:`FittedComputationMethod`."""
        return self.fitter(distribution, **options)
