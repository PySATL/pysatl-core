"""
Base classes for transformed distributions.

This module introduces the first architectural layer for derived
probability distributions produced by transformations. The goal is to
keep them fully compatible with the existing :class:`Distribution`
protocol and computation graph while still preserving transformation
metadata.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping
from numbers import Real
from types import NotImplementedType
from typing import TYPE_CHECKING, Any

from pysatl_core.distributions.computation import AnalyticalComputation
from pysatl_core.distributions.distribution import _KEEP, Distribution
from pysatl_core.distributions.strategies import (
    ComputationStrategy,
    DefaultComputationStrategy,
    DefaultSamplingUnivariateStrategy,
    SamplingStrategy,
)
from pysatl_core.types import (
    ApproximationName,
    DistributionType,
    GenericCharacteristicName,
    ParentRole,
    TransformationName,
)

if TYPE_CHECKING:
    from pysatl_core.distributions.support import Support
    from pysatl_core.transformations.approximations.approximation import DistributionApproximator
    from pysatl_core.transformations.operations.affine import AffineDistribution


class DerivedDistribution(Distribution):
    """
    Base class for distributions obtained from one or more parents.

    Parameters
    ----------
    distribution_type : DistributionType
        Type descriptor of the derived distribution.
    bases : Mapping[ParentRole, Distribution]
        Parent distributions participating in the transformation.
    analytical_computations : Mapping[GenericCharacteristicName, AnalyticalComputation[Any, Any]]
        Direct analytical computations attached to the derived distribution.
    transformation_name : TransformationName
        Logical name of the transformation.
    support : Support | None, optional
        Support of the transformed distribution.
    sampling_strategy : SamplingStrategy | None, optional
        Strategy used to generate random samples. If omitted, a default
        univariate inverse-transform strategy is installed when the derived
        distribution provides an analytical PPF. Otherwise a placeholder
        strategy is used.
    computation_strategy : ComputationStrategy | None, optional
        Strategy used to resolve characteristics.
    """

    def __init__(
        self,
        *,
        distribution_type: DistributionType,
        bases: Mapping[ParentRole, Distribution],
        analytical_computations: Mapping[
            GenericCharacteristicName, AnalyticalComputation[Any, Any]
        ],
        transformation_name: TransformationName,
        support: Support | None = None,
        sampling_strategy: SamplingStrategy | None = None,
        computation_strategy: ComputationStrategy | None = None,
    ) -> None:
        self._distribution_type = distribution_type
        self._bases = dict(bases)
        self._analytical_computations = dict(analytical_computations)
        self._transformation_name = transformation_name
        self._support = support
        self._sampling_strategy = sampling_strategy or DefaultSamplingUnivariateStrategy()
        self._computation_strategy = computation_strategy or DefaultComputationStrategy()

    @property
    def distribution_type(self) -> DistributionType:
        """Get the type descriptor of the derived distribution."""
        return self._distribution_type

    @property
    def bases(self) -> Mapping[ParentRole, Distribution]:
        """Get parent distributions grouped by their logical roles."""
        return self._bases

    @property
    def transformation_name(self) -> TransformationName:
        """Get the logical name of the transformation."""
        return self._transformation_name

    @property
    def analytical_computations(
        self,
    ) -> Mapping[GenericCharacteristicName, AnalyticalComputation[Any, Any]]:
        """Get direct analytical computations of the derived distribution."""
        return self._analytical_computations

    @property
    def sampling_strategy(self) -> SamplingStrategy:
        """Get the sampling strategy."""
        return self._sampling_strategy

    @property
    def computation_strategy(self) -> ComputationStrategy:
        """Get the characteristic resolution strategy."""
        return self._computation_strategy

    @property
    def support(self) -> Support | None:
        """Get the support of the derived distribution."""
        return self._support

    def approximate(
        self,
        approximator: DistributionApproximator,
        **options: Any,
    ) -> ApproximatedDistribution:
        """
        Approximate the current derivation tree.

        Parameters
        ----------
        approximator : DistributionApproximator
            External approximation object responsible for creating the new
            distribution instance.
        **options : Any
            Extra options forwarded to the approximator.

        Returns
        -------
        ApproximatedDistribution
            Approximated representation of the current derived distribution.
        """
        return approximator.approximate(self, **options)

    def _clone_with_strategies(
        self,
        *,
        sampling_strategy: SamplingStrategy | None | object = _KEEP,
        computation_strategy: ComputationStrategy | None | object = _KEEP,
    ) -> DerivedDistribution:
        """
        Return a copy of the derived distribution with updated strategies.

        Subclasses representing concrete transformations should normally
        override this method to preserve their own constructor parameters.
        """
        return DerivedDistribution(
            distribution_type=self.distribution_type,
            bases=self.bases,
            analytical_computations=self.analytical_computations,
            transformation_name=self.transformation_name,
            support=self.support,
            sampling_strategy=self._new_sampling_strategy(sampling_strategy),
            computation_strategy=self._new_computation_strategy(computation_strategy),
        )

    def _affine(self, scale: float, shift: float) -> AffineDistribution:
        """
        Build an affine transformation of the current distribution.

        Parameters
        ----------
        scale : float
            Multiplicative coefficient.
        shift : float
            Additive coefficient.

        Returns
        -------
        AffineDistribution
            Derived distribution representing ``scale * X + shift``.
        """
        from pysatl_core.transformations.operations.affine import AffineDistribution

        return AffineDistribution(self, scale=scale, shift=shift)

    def __add__(self, other: object) -> DerivedDistribution | NotImplementedType:
        """Return ``self + scalar`` as an affine transformation."""
        if isinstance(other, Real):
            return self._affine(scale=1.0, shift=float(other))
        return NotImplemented

    def __radd__(self, other: object) -> DerivedDistribution | NotImplementedType:
        """Return ``scalar + self`` as an affine transformation."""
        if isinstance(other, Real):
            return self._affine(scale=1.0, shift=float(other))
        return NotImplemented

    def __sub__(self, other: object) -> DerivedDistribution | NotImplementedType:
        """Return ``self - scalar`` as an affine transformation."""
        if isinstance(other, Real):
            return self._affine(scale=1.0, shift=-float(other))
        return NotImplemented

    def __rsub__(self, other: object) -> DerivedDistribution | NotImplementedType:
        """Return ``scalar - self`` as an affine transformation."""
        if isinstance(other, Real):
            return self._affine(scale=-1.0, shift=float(other))
        return NotImplemented

    def __mul__(self, other: object) -> DerivedDistribution | NotImplementedType:
        """Return ``self * scalar`` as an affine transformation."""
        if isinstance(other, Real):
            return self._affine(scale=float(other), shift=0.0)
        return NotImplemented

    def __rmul__(self, other: object) -> DerivedDistribution | NotImplementedType:
        """Return ``scalar * self`` as an affine transformation."""
        if isinstance(other, Real):
            return self._affine(scale=float(other), shift=0.0)
        return NotImplemented

    def __truediv__(self, other: object) -> DerivedDistribution | NotImplementedType:
        """Return ``self / scalar`` as an affine transformation."""
        if isinstance(other, Real):
            divisor = float(other)
            if divisor == 0.0:
                raise ZeroDivisionError("Cannot divide a distribution by zero.")
            return self._affine(scale=1.0 / divisor, shift=0.0)
        return NotImplemented

    def __neg__(self) -> DerivedDistribution:
        """Return ``-self`` as an affine transformation."""
        return self._affine(scale=-1.0, shift=0.0)


class ApproximatedDistribution(DerivedDistribution):
    """
    Derived distribution whose analytical computations were materialized by an
    external approximator.

    Parameters
    ----------
    source_distribution : DerivedDistribution
        Original distribution being approximated.
    approximation_name : ApproximationName
        Name of the approximation procedure.
    distribution_type : DistributionType
        Type descriptor of the approximated distribution.
    bases : Mapping[ParentRole, Distribution]
        Parent distributions preserved by the approximated distribution.
    analytical_computations : Mapping[GenericCharacteristicName, AnalyticalComputation[Any, Any]]
        Materialized analytical computations produced by the approximator.
    support : Support | None, optional
        Support of the approximated distribution.
    sampling_strategy : SamplingStrategy | None, optional
        Sampling strategy to expose.
    computation_strategy : ComputationStrategy | None, optional
        Characteristic resolution strategy.
    """

    def __init__(
        self,
        *,
        source_distribution: DerivedDistribution,
        approximation_name: ApproximationName,
        distribution_type: DistributionType,
        bases: Mapping[ParentRole, Distribution],
        analytical_computations: Mapping[
            GenericCharacteristicName, AnalyticalComputation[Any, Any]
        ],
        support: Support | None = None,
        sampling_strategy: SamplingStrategy | None = None,
        computation_strategy: ComputationStrategy | None = None,
    ) -> None:
        super().__init__(
            distribution_type=distribution_type,
            bases=bases,
            analytical_computations=analytical_computations,
            transformation_name=TransformationName.APPROXIMATION,
            support=support,
            sampling_strategy=sampling_strategy,
            computation_strategy=computation_strategy,
        )
        self._source_distribution = source_distribution
        self._approximation_name = approximation_name

    @property
    def source_distribution(self) -> DerivedDistribution:
        """Get the original non-approximated distribution."""
        return self._source_distribution

    @property
    def approximation_name(self) -> ApproximationName:
        """Get the name of the approximation procedure."""
        return self._approximation_name

    def _clone_with_strategies(
        self,
        *,
        sampling_strategy: SamplingStrategy | None | object = _KEEP,
        computation_strategy: ComputationStrategy | None | object = _KEEP,
    ) -> ApproximatedDistribution:
        """Return a copy of the approximated distribution with updated strategies."""
        return ApproximatedDistribution(
            source_distribution=self._source_distribution,
            approximation_name=self.approximation_name,
            distribution_type=self.distribution_type,
            bases=self.bases,
            analytical_computations=self.analytical_computations,
            support=self.support,
            sampling_strategy=self._new_sampling_strategy(sampling_strategy),
            computation_strategy=self._new_computation_strategy(computation_strategy),
        )


__all__ = [
    "ApproximatedDistribution",
    "DerivedDistribution",
]
