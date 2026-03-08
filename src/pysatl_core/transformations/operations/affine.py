"""
Affine transformation for probability distributions.

This module implements the transformation ``Y = aX + b``. The
transformation is represented as a derived distribution with analytical
computations built from parent methods resolved through ``query_method()``.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from pysatl_core.distributions.computation import Method
from pysatl_core.distributions.distribution import Distribution
from pysatl_core.distributions.support import (
    ContinuousSupport,
    ExplicitTableDiscreteSupport,
    Support,
)
from pysatl_core.transformations.distribution import DerivedDistribution
from pysatl_core.transformations.transformation_method import (
    ResolvedSourceMethods,
    SourceRequirements,
    TransformationMethod,
)
from pysatl_core.types import (
    CharacteristicName,
    ComplexArray,
    ComputationFunc,
    DistributionType,
    GenericCharacteristicName,
    Kind,
    NumericArray,
    ParentRole,
    TransformationName,
)

_BASE_ROLE: ParentRole = "base"


class AffineDistribution(DerivedDistribution):
    """
    Distribution obtained from the affine transformation ``Y = aX + b``.

    Parameters
    ----------
    base_distribution : Distribution
        Source distribution being transformed.
    scale : float
        Multiplicative coefficient ``a``.
    shift : float, default=0.0
        Additive coefficient ``b``.

    Notes
    -----
    The current implementation focuses on one-dimensional continuous and
    discrete distributions.
    """

    def __init__(
        self,
        base_distribution: Distribution,
        *,
        scale: float,
        shift: float = 0.0,
    ) -> None:
        if scale == 0.0:
            raise ValueError("scale must be non-zero for an affine transformation.")

        self._base_distribution = base_distribution
        self._scale = float(scale)
        self._shift = float(shift)
        distribution_type = self._validate_distribution_type(base_distribution.distribution_type)
        bases: dict[ParentRole, Distribution] = {_BASE_ROLE: base_distribution}

        super().__init__(
            distribution_type=distribution_type,
            bases=bases,
            analytical_computations=self._build_analytical_computations(
                distribution_type=distribution_type,
                bases=bases,
            ),
            transformation_name=TransformationName.AFFINE,
            support=self._transform_support(base_distribution.support),
        )

    @property
    def base_distribution(self) -> Distribution:
        """Get the source distribution."""
        return self._base_distribution

    @property
    def scale(self) -> float:
        """Get the multiplicative coefficient ``a``."""
        return self._scale

    @property
    def shift(self) -> float:
        """Get the additive coefficient ``b``."""
        return self._shift

    def _validate_distribution_type(self, distribution_type: DistributionType) -> DistributionType:
        """
        Validate that the affine transformation can be applied.

        Parameters
        ----------
        distribution_type : DistributionType
            Distribution type descriptor of the parent distribution.

        Returns
        -------
        DistributionType
            The validated distribution type.

        Raises
        ------
        TypeError
            If the distribution is not one-dimensional continuous or discrete.
        """
        dimension = getattr(distribution_type, "dimension", None)
        kind = getattr(distribution_type, "kind", None)

        if dimension != 1:
            raise TypeError(
                "AffineDistribution currently supports only one-dimensional distributions."
            )

        if kind not in {Kind.CONTINUOUS, Kind.DISCRETE}:
            raise TypeError("Unsupported distribution kind for affine transformation.")

        return distribution_type

    def _build_analytical_computations(
        self,
        *,
        distribution_type: DistributionType,
        bases: Mapping[ParentRole, Distribution],
    ) -> Mapping[GenericCharacteristicName, TransformationMethod[Any, Any]]:
        """
        Build analytical computations for the affine transformation.

        Parameters
        ----------
        distribution_type : DistributionType
            Type descriptor of the transformed distribution.
        bases : Mapping[ParentRole, Distribution]
            Parent distributions grouped by logical role.

        Returns
        -------
        Mapping[GenericCharacteristicName, TransformationMethod[Any, Any]]
            Analytical computations exposed by the transformed distribution.

        Raises
        ------
        TypeError
            If the distribution kind is not supported.
        """
        computations: dict[GenericCharacteristicName, TransformationMethod[Any, Any]] = {}
        kind = getattr(distribution_type, "kind", None)

        computations[CharacteristicName.CF] = TransformationMethod.from_parents(
            target=CharacteristicName.CF,
            transformation=TransformationName.AFFINE,
            bases=bases,
            source_requirements=self._requirements(CharacteristicName.CF),
            evaluator=self._make_cf,
        )
        computations[CharacteristicName.MEAN] = TransformationMethod.from_parents(
            target=CharacteristicName.MEAN,
            transformation=TransformationName.AFFINE,
            bases=bases,
            source_requirements=self._requirements(CharacteristicName.MEAN),
            evaluator=self._make_mean,
        )
        computations[CharacteristicName.VAR] = TransformationMethod.from_parents(
            target=CharacteristicName.VAR,
            transformation=TransformationName.AFFINE,
            bases=bases,
            source_requirements=self._requirements(CharacteristicName.VAR),
            evaluator=self._make_var,
        )
        computations[CharacteristicName.SKEW] = TransformationMethod.from_parents(
            target=CharacteristicName.SKEW,
            transformation=TransformationName.AFFINE,
            bases=bases,
            source_requirements=self._requirements(CharacteristicName.SKEW),
            evaluator=self._make_skew,
        )
        computations[CharacteristicName.KURT] = TransformationMethod.from_parents(
            target=CharacteristicName.KURT,
            transformation=TransformationName.AFFINE,
            bases=bases,
            source_requirements=self._requirements(CharacteristicName.KURT),
            evaluator=self._make_kurt,
        )

        if kind == Kind.CONTINUOUS:
            computations[CharacteristicName.CDF] = TransformationMethod.from_parents(
                target=CharacteristicName.CDF,
                transformation=TransformationName.AFFINE,
                bases=bases,
                source_requirements=self._requirements(CharacteristicName.CDF),
                evaluator=self._make_continuous_cdf,
            )
            computations[CharacteristicName.PDF] = TransformationMethod.from_parents(
                target=CharacteristicName.PDF,
                transformation=TransformationName.AFFINE,
                bases=bases,
                source_requirements=self._requirements(CharacteristicName.PDF),
                evaluator=self._make_continuous_pdf,
            )
            computations[CharacteristicName.PPF] = TransformationMethod.from_parents(
                target=CharacteristicName.PPF,
                transformation=TransformationName.AFFINE,
                bases=bases,
                source_requirements=self._requirements(CharacteristicName.PPF),
                evaluator=self._make_continuous_ppf,
            )
            return computations

        if kind == Kind.DISCRETE:
            computations[CharacteristicName.PMF] = TransformationMethod.from_parents(
                target=CharacteristicName.PMF,
                transformation=TransformationName.AFFINE,
                bases=bases,
                source_requirements=self._requirements(CharacteristicName.PMF),
                evaluator=self._make_discrete_pmf,
            )
            computations[CharacteristicName.CDF] = TransformationMethod.from_parents(
                target=CharacteristicName.CDF,
                transformation=TransformationName.AFFINE,
                bases=bases,
                source_requirements=(
                    self._requirements(CharacteristicName.CDF)
                    if self.scale > 0.0
                    else self._requirements(CharacteristicName.CDF, CharacteristicName.PMF)
                ),
                evaluator=(
                    self._make_discrete_cdf
                    if self.scale > 0.0
                    else self._make_discrete_cdf_negative_scale
                ),
            )
            computations[CharacteristicName.PPF] = TransformationMethod.from_parents(
                target=CharacteristicName.PPF,
                transformation=TransformationName.AFFINE,
                bases=bases,
                source_requirements=self._requirements(CharacteristicName.PPF),
                evaluator=self._make_discrete_ppf,
            )
            return computations

        raise TypeError("Unsupported distribution kind for affine transformation.")

    def _requirements(
        self,
        *characteristics: GenericCharacteristicName,
    ) -> SourceRequirements:
        """
        Build source requirements for the base distribution.

        Parameters
        ----------
        *characteristics : GenericCharacteristicName
            Parent characteristics required to evaluate a transformed one.

        Returns
        -------
        SourceRequirements
            Requirements grouped by the single base role.
        """
        return {_BASE_ROLE: tuple(characteristics)}

    def _make_continuous_cdf(
        self,
        sources: ResolvedSourceMethods,
    ) -> ComputationFunc[NumericArray, NumericArray]:
        """
        Build the transformed CDF for a continuous base distribution.

        For ``Y = aX + b`` the formula is
        ``F_Y(y) = F_X((y - b) / a)`` for ``a > 0`` and
        ``F_Y(y) = 1 - F_X((y - b) / a)`` for ``a < 0``.
        """
        base_cdf = cast(
            Method[NumericArray, NumericArray], sources[_BASE_ROLE][CharacteristicName.CDF]
        )

        def _cdf(data: NumericArray, **options: Any) -> NumericArray:
            values = base_cdf((data - self.shift) / self.scale, **options)
            return 1.0 - values if self.scale < 0.0 else values

        return cast(ComputationFunc[NumericArray, NumericArray], _cdf)

    def _make_continuous_pdf(
        self,
        sources: ResolvedSourceMethods,
    ) -> ComputationFunc[NumericArray, NumericArray]:
        """
        Build the transformed PDF for a continuous base distribution.

        The affine density transformation is
        ``f_Y(y) = f_X((y - b) / a) / |a|``.
        """
        base_pdf = cast(
            Method[NumericArray, NumericArray], sources[_BASE_ROLE][CharacteristicName.PDF]
        )

        def _pdf(data: NumericArray, **options: Any) -> NumericArray:
            return cast(
                NumericArray,
                base_pdf((data - self.shift) / self.scale, **options) / abs(self.scale),
            )

        return cast(ComputationFunc[NumericArray, NumericArray], _pdf)

    def _make_continuous_ppf(
        self,
        sources: ResolvedSourceMethods,
    ) -> ComputationFunc[NumericArray, NumericArray]:
        """
        Build the transformed PPF for a continuous base distribution.

        For positive scale the quantiles are transformed directly. For
        negative scale the probabilities are mirrored as ``1 - p``.
        """
        base_ppf = cast(
            Method[NumericArray, NumericArray], sources[_BASE_ROLE][CharacteristicName.PPF]
        )

        def _ppf(data: NumericArray, **options: Any) -> NumericArray:
            probabilities = data if self.scale > 0.0 else 1.0 - data
            return self.scale * base_ppf(probabilities, **options) + self.shift

        return cast(ComputationFunc[NumericArray, NumericArray], _ppf)

    def _make_discrete_cdf(
        self,
        sources: ResolvedSourceMethods,
    ) -> ComputationFunc[NumericArray, NumericArray]:
        """
        Build the transformed CDF for a discrete base distribution with ``a > 0``.
        """
        base_cdf = cast(
            Method[NumericArray, NumericArray], sources[_BASE_ROLE][CharacteristicName.CDF]
        )

        def _cdf(data: NumericArray, **options: Any) -> NumericArray:
            return base_cdf((data - self.shift) / self.scale, **options)

        return cast(ComputationFunc[NumericArray, NumericArray], _cdf)

    def _make_discrete_cdf_negative_scale(
        self,
        sources: ResolvedSourceMethods,
    ) -> ComputationFunc[NumericArray, NumericArray]:
        """
        Build the transformed CDF for a discrete base distribution with ``a < 0``.

        For a decreasing affine map the lower tail of ``Y`` becomes the upper
        tail of ``X``. Using both CDF and PMF avoids support-specific logic and
        preserves jump values exactly.
        """
        base_cdf = cast(
            Method[NumericArray, NumericArray], sources[_BASE_ROLE][CharacteristicName.CDF]
        )
        base_pmf = cast(
            Method[NumericArray, NumericArray], sources[_BASE_ROLE][CharacteristicName.PMF]
        )

        def _cdf(data: NumericArray, **options: Any) -> NumericArray:
            x = (data - self.shift) / self.scale
            return np.asarray(1.0 - base_cdf(x, **options) + base_pmf(x, **options))

        return cast(ComputationFunc[NumericArray, NumericArray], _cdf)

    def _make_discrete_pmf(
        self,
        sources: ResolvedSourceMethods,
    ) -> ComputationFunc[NumericArray, NumericArray]:
        """
        Build the transformed PMF for a discrete base distribution.

        Since the affine map is bijective for ``a != 0``, the probability mass
        is preserved without any Jacobian factor.
        """
        base_pmf = cast(
            Method[NumericArray, NumericArray], sources[_BASE_ROLE][CharacteristicName.PMF]
        )

        def _pmf(data: NumericArray, **options: Any) -> NumericArray:
            return base_pmf((data - self.shift) / self.scale, **options)

        return cast(ComputationFunc[NumericArray, NumericArray], _pmf)

    def _make_discrete_ppf(
        self,
        sources: ResolvedSourceMethods,
    ) -> ComputationFunc[NumericArray, NumericArray]:
        """
        Build the transformed PPF for a discrete base distribution.

        For positive scale the quantiles are transformed directly. For
        negative scale the lower-tail quantile of ``Y`` corresponds to the
        strict upper-tail quantile of ``X``, implemented through
        ``nextafter(1 - p, 1)``.
        """
        base_ppf = cast(
            Method[NumericArray, NumericArray], sources[_BASE_ROLE][CharacteristicName.PPF]
        )

        def _ppf(data: NumericArray, **options: Any) -> NumericArray:
            x = data if self.scale > 0.0 else np.nextafter(1.0 - data, 1.0)
            return self.scale * base_ppf(x, **options) + self.shift

        return cast(ComputationFunc[NumericArray, NumericArray], _ppf)

    def _make_cf(
        self,
        sources: ResolvedSourceMethods,
    ) -> ComputationFunc[NumericArray, ComplexArray]:
        """
        Build the characteristic function of the affine transform.

        The formula is ``phi_Y(t) = exp(i b t) * phi_X(a t)``.
        """
        base_cf = cast(
            Method[NumericArray, ComplexArray], sources[_BASE_ROLE][CharacteristicName.CF]
        )

        def _cf(data: NumericArray, **options: Any) -> ComplexArray:
            return cast(
                ComplexArray, np.exp(1j * self.shift * data) * base_cf(self.scale * data, **options)
            )

        return cast(ComputationFunc[NumericArray, ComplexArray], _cf)

    def _make_mean(self, sources: ResolvedSourceMethods) -> ComputationFunc[Any, float]:
        """Build the transformed mean."""
        base_mean = cast(Method[Any, float], sources[_BASE_ROLE][CharacteristicName.MEAN])

        def _mean(**options: Any) -> float:
            return self.scale * base_mean(**options) + self.shift

        return _mean

    def _make_var(self, sources: ResolvedSourceMethods) -> ComputationFunc[Any, float]:
        """Build the transformed variance."""
        base_var = cast(Method[Any, float], sources[_BASE_ROLE][CharacteristicName.VAR])

        def _var(**options: Any) -> float:
            return self.scale**2 * base_var(**options)

        return _var

    def _make_skew(self, sources: ResolvedSourceMethods) -> ComputationFunc[Any, float]:
        """
        Build the transformed skewness.

        Multiplication by a negative constant flips the sign of skewness.
        """
        base_skew = cast(Method[Any, float], sources[_BASE_ROLE][CharacteristicName.SKEW])

        def _skew(**options: Any) -> float:
            sign = -1.0 if self.scale < 0.0 else 1.0
            return sign * base_skew(**options)

        return _skew

    def _make_kurt(self, sources: ResolvedSourceMethods) -> ComputationFunc[Any, float]:
        """
        Build the transformed kurtosis.

        Kurtosis is invariant under affine transformations with non-zero scale.
        """
        base_kurt = cast(Method[Any, float], sources[_BASE_ROLE][CharacteristicName.KURT])

        def _kurt(**options: Any) -> float:
            return base_kurt(**options)

        return _kurt

    def _transform_support(self, support: Support | None) -> Support | None:
        """
        Transform the parent support when its structure is known.

        Notes
        -----
        Some support types are intentionally left unhandled for now.
        In such cases the transformed distribution exposes ``None``.
        """
        if support is None:
            return None

        if isinstance(support, ContinuousSupport):
            return self._transform_continuous_support(support)

        if isinstance(support, ExplicitTableDiscreteSupport):
            transformed_points = np.asarray(support.points, dtype=float) * self.scale + self.shift
            return ExplicitTableDiscreteSupport(points=transformed_points, assume_sorted=False)

        return None

    def _transform_continuous_support(self, support: ContinuousSupport) -> ContinuousSupport:
        """
        Transform a continuous interval support under the affine map.
        """
        left = float(self.scale * support.left + self.shift)
        right = float(self.scale * support.right + self.shift)

        if self.scale > 0.0:
            return ContinuousSupport(
                left=left,
                right=right,
                left_closed=support.left_closed,
                right_closed=support.right_closed,
            )

        return ContinuousSupport(
            left=right,
            right=left,
            left_closed=support.right_closed,
            right_closed=support.left_closed,
        )


def affine(
    distribution: Distribution,
    *,
    scale: float,
    shift: float = 0.0,
) -> AffineDistribution:
    """
    Apply the affine transformation ``Y = aX + b`` to a distribution.

    Parameters
    ----------
    distribution : Distribution
        Source distribution.
    scale : float
        Multiplicative coefficient ``a``.
    shift : float, default=0.0
        Additive coefficient ``b``.

    Returns
    -------
    AffineDistribution
        Derived distribution representing the transformed random variable.
    """
    return AffineDistribution(distribution, scale=scale, shift=shift)


__all__ = [
    "AffineDistribution",
    "affine",
]
