"""
Finite weighted mixture transformation for probability distributions.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping, Sequence
from math import sqrt
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pysatl_core.distributions.computation import Method
from pysatl_core.distributions.distribution import _KEEP, Distribution
from pysatl_core.distributions.registry import characteristic_registry
from pysatl_core.distributions.support import (
    ContinuousSupport,
    ExplicitTableDiscreteSupport,
    Support,
)
from pysatl_core.transformations.distribution import DerivedDistribution
from pysatl_core.transformations.lightweight_distribution import LightweightDistribution
from pysatl_core.transformations.transformation_method import (
    ResolvedSourceMethods,
    SourceRequirements,
    TransformationEvaluator,
    TransformationMethod,
)
from pysatl_core.types import (
    DEFAULT_ANALYTICAL_COMPUTATION_LABEL,
    CharacteristicName,
    ComplexArray,
    ComputationFunc,
    DistributionType,
    GenericCharacteristicName,
    Kind,
    LabelName,
    NumericArray,
    ParentRole,
    TransformationName,
)

if TYPE_CHECKING:
    from pysatl_core.distributions.strategies import (
        ComputationStrategy,
        SamplingStrategy,
    )

_COMPONENT_ROLE_PREFIX = "component_"
_MAX_MOMENT_ORDER = 4


def _component_role(index: int) -> ParentRole:
    """Build deterministic parent role for a mixture component."""
    return cast(ParentRole, f"{_COMPONENT_ROLE_PREFIX}{index}")


class FiniteMixtureDistribution(DerivedDistribution):
    """
    Distribution obtained as a finite weighted mixture of components.

    Parameters
    ----------
    weighted_components : Sequence[tuple[float, Distribution]]
        Ordered pairs ``(weight, distribution)``.
        Weights must be finite, non-negative and sum to one.
    sampling_strategy : SamplingStrategy | None, optional
        Sampling strategy exposed by the transformed distribution.
    computation_strategy : ComputationStrategy | None, optional
        Computation strategy exposed by the transformed distribution.
    """

    def __init__(
        self,
        weighted_components: Sequence[tuple[float, Distribution]],
        *,
        sampling_strategy: SamplingStrategy | None = None,
        computation_strategy: ComputationStrategy | None = None,
    ) -> None:
        if not weighted_components:
            raise ValueError("Finite mixture requires at least one component distribution.")

        components = tuple(component for _, component in weighted_components)
        weights = [float(weight) for weight, _ in weighted_components]

        component_snapshots = tuple(
            LightweightDistribution.from_distribution(component) for component in components
        )
        validated_weights = self._validate_weights(weights)
        roles = tuple(_component_role(index) for index in range(len(component_snapshots)))

        self._components = component_snapshots
        self._weights = validated_weights
        self._roles = roles
        self._role_to_index = {role: index for index, role in enumerate(roles)}
        self._cached_discrete_mass_table: tuple[NumericArray, NumericArray, NumericArray] | None = (
            None
        )

        distribution_type = self._validate_distribution_types(
            [component.distribution_type for component in component_snapshots]
        )
        self._discrete_support = self._build_discrete_support()
        self._continuous_support = self._build_continuous_support()

        self._discrete_points = (
            cast(NumericArray, np.asarray(self._discrete_support.points, dtype=float))
            if self._discrete_support is not None
            else None
        )

        bases: dict[ParentRole, LightweightDistribution] = dict(
            zip(roles, component_snapshots, strict=True)
        )
        analytical_computations, loop_analytical_flags = self._build_analytical_computations(
            distribution_type=distribution_type,
            bases=bases,
        )

        super().__init__(
            distribution_type=distribution_type,
            bases=bases,
            analytical_computations=analytical_computations,
            transformation_name=TransformationName.FINITE_MIXTURE,
            support=self._transform_support(distribution_type),
            sampling_strategy=sampling_strategy,
            computation_strategy=computation_strategy,
            loop_analytical_flags=loop_analytical_flags,
        )

    @property
    def components(self) -> tuple[LightweightDistribution, ...]:
        """Get the lightweight component snapshots."""
        return self._components

    @property
    def weights(self) -> NumericArray:
        """Get validated component weights."""
        return cast(NumericArray, np.array(self._weights, copy=True))

    def sample(self, n: int, **options: Any) -> NumericArray:
        """
        Generate mixture samples by sampling selected components.

        A component index is sampled for each draw according to the
        validated mixture weights, then each component is sampled only
        for the number of selected draws.
        """
        rng = np.random.default_rng()
        component_indices = np.asarray(rng.choice(len(self._components), size=n, p=self._weights))
        samples = np.empty(component_indices.shape, dtype=float)

        for index, component in enumerate(self.components):
            selected_positions = np.nonzero(component_indices == index)[0]
            selected_count = int(selected_positions.size)
            if selected_count == 0:
                continue

            component_samples = np.asarray(
                self.sampling_strategy.sample(selected_count, distr=component, **options),
                dtype=float,
            ).reshape(-1)
            if component_samples.size != selected_count:
                raise RuntimeError(
                    "Component sampler returned incompatible sample shape for finite mixture."
                )
            samples[selected_positions] = component_samples

        return cast(NumericArray, samples)

    def _clone_with_strategies(
        self,
        *,
        sampling_strategy: SamplingStrategy | None | object = _KEEP,
        computation_strategy: ComputationStrategy | None | object = _KEEP,
    ) -> FiniteMixtureDistribution:
        """Return a copy of the mixture distribution with updated strategies."""
        return FiniteMixtureDistribution(
            weighted_components=[
                (float(weight), component)
                for weight, component in zip(self._weights, self.components, strict=True)
            ],
            sampling_strategy=self._new_sampling_strategy(sampling_strategy),
            computation_strategy=self._new_computation_strategy(computation_strategy),
        )

    @staticmethod
    def _validate_weights(weights: Sequence[float]) -> NumericArray:
        """Validate mixture weights."""
        validated = np.asarray(weights, dtype=float)
        if validated.ndim != 1:
            raise ValueError("Mixture weights must be a one-dimensional sequence.")
        if np.any(~np.isfinite(validated)):
            raise ValueError("Mixture weights must be finite numbers.")
        if np.any(validated < 0.0):
            raise ValueError("Mixture weights must be non-negative.")

        total = float(np.sum(validated))
        if not np.isclose(total, 1.0, rtol=1e-12, atol=1e-12):
            raise ValueError(f"Sum of mixture weights must be equal to 1.0, got {total}.")
        return cast(NumericArray, np.array(validated, copy=True))

    @staticmethod
    def _validate_distribution_types(
        distribution_types: Sequence[DistributionType],
    ) -> DistributionType:
        """Validate component distribution type compatibility."""
        first = distribution_types[0]
        first_dimension = getattr(first, "dimension", None)
        first_kind = getattr(first, "kind", None)

        if first_dimension != 1:
            raise TypeError("Finite mixture currently supports only one-dimensional distributions.")
        if first_kind not in {Kind.CONTINUOUS, Kind.DISCRETE}:
            raise TypeError("Unsupported distribution kind for finite mixture.")

        for distribution_type in distribution_types[1:]:
            dimension = getattr(distribution_type, "dimension", None)
            kind = getattr(distribution_type, "kind", None)
            if dimension != first_dimension:
                raise TypeError("Finite mixture requires components with equal dimension.")
            if kind != first_kind:
                raise TypeError("Finite mixture requires components of the same distribution kind.")

        return first

    def _build_discrete_support(self) -> ExplicitTableDiscreteSupport | None:
        """Build explicit discrete support union, if available."""
        points_blocks: list[NumericArray] = []
        for component in self.components:
            support = component.support
            if not isinstance(support, ExplicitTableDiscreteSupport):
                return None
            points_blocks.append(cast(NumericArray, np.asarray(support.points, dtype=float)))

        if not points_blocks:
            return None
        merged = np.unique(np.concatenate(points_blocks)).tolist()
        return ExplicitTableDiscreteSupport(points=merged, assume_sorted=True)

    def _build_continuous_support(self) -> ContinuousSupport | None:
        """Build continuous support envelope, if available."""
        supports: list[ContinuousSupport] = []
        for component in self.components:
            support = component.support
            if not isinstance(support, ContinuousSupport):
                return None
            supports.append(support)

        left = min(float(support.left) for support in supports)
        right = max(float(support.right) for support in supports)
        left_closed = any(
            float(support.left) == left and support.left_closed for support in supports
        )
        right_closed = any(
            float(support.right) == right and support.right_closed for support in supports
        )
        return ContinuousSupport(
            left=left,
            right=right,
            left_closed=left_closed,
            right_closed=right_closed,
        )

    def _transform_support(self, distribution_type: DistributionType) -> Support | None:
        """Transform support metadata for the mixture."""
        kind = getattr(distribution_type, "kind", None)

        if kind == Kind.CONTINUOUS:
            return self._continuous_support

        if kind == Kind.DISCRETE:
            return self._discrete_support

        return None

    def _requirements_for_all(
        self,
        *characteristics: GenericCharacteristicName,
    ) -> SourceRequirements:
        """Build source requirements for all component roles."""
        return {role: tuple(characteristics) for role in self._roles}

    @staticmethod
    def _validate_moment_order(max_order: int) -> int:
        """Validate and normalize required moment order."""
        if not 1 <= max_order <= _MAX_MOMENT_ORDER:
            raise ValueError(f"Moment order must be in [1, {_MAX_MOMENT_ORDER}], got {max_order}.")
        return max_order

    @classmethod
    def _statistics_for_moment_order(
        cls,
        max_order: int,
    ) -> tuple[GenericCharacteristicName, ...]:
        """Return component statistics needed to recover raw moments up to ``max_order``."""
        order = cls._validate_moment_order(max_order)
        if order == 1:
            return (CharacteristicName.MEAN,)
        if order == 2:
            return CharacteristicName.MEAN, CharacteristicName.VAR
        if order == 3:
            return (
                CharacteristicName.MEAN,
                CharacteristicName.VAR,
                CharacteristicName.SKEW,
            )
        return (
            CharacteristicName.MEAN,
            CharacteristicName.VAR,
            CharacteristicName.SKEW,
            CharacteristicName.KURT,
        )

    def _build_analytical_computations(
        self,
        *,
        distribution_type: DistributionType,
        bases: Mapping[ParentRole, LightweightDistribution],
    ) -> tuple[
        Mapping[GenericCharacteristicName, Mapping[LabelName, TransformationMethod[Any, Any]]],
        Mapping[GenericCharacteristicName, Mapping[LabelName, bool]],
    ]:
        """Build analytical computations for finite mixture transformation."""
        computations: dict[
            GenericCharacteristicName, dict[LabelName, TransformationMethod[Any, Any]]
        ] = {}
        loop_analytical_flags: dict[GenericCharacteristicName, dict[LabelName, bool]] = {}
        kind = getattr(distribution_type, "kind", None)
        declared_registry_characteristics = characteristic_registry().declared_characteristics

        def _register(
            target: GenericCharacteristicName,
            source_requirements: SourceRequirements,
            evaluator: TransformationEvaluator[Any, Any],
        ) -> None:
            # Non-registry characteristics cannot be resolved via graph fitters.
            # For them we require direct component presence for each required source.
            for role, characteristics in source_requirements.items():
                base = bases[role]
                for characteristic in characteristics:
                    if characteristic in declared_registry_characteristics:
                        continue
                    if characteristic not in base.analytical_computations:
                        return

            method, is_analytical, has_any_present_source = TransformationMethod.try_from_parents(
                target=target,
                transformation=TransformationName.FINITE_MIXTURE,
                bases=bases,
                source_requirements=source_requirements,
                evaluator=evaluator,
            )
            if method is None or not has_any_present_source:
                return
            computations[target] = {DEFAULT_ANALYTICAL_COMPUTATION_LABEL: method}
            loop_analytical_flags[target] = {DEFAULT_ANALYTICAL_COMPUTATION_LABEL: is_analytical}

        _register(
            CharacteristicName.CF,
            self._requirements_for_all(CharacteristicName.CF),
            self._make_cf,
        )
        _register(
            CharacteristicName.MEAN,
            self._requirements_for_all(*self._statistics_for_moment_order(1)),
            self._make_mean,
        )
        _register(
            CharacteristicName.VAR,
            self._requirements_for_all(*self._statistics_for_moment_order(2)),
            self._make_var,
        )
        _register(
            CharacteristicName.SKEW,
            self._requirements_for_all(*self._statistics_for_moment_order(3)),
            self._make_skew,
        )
        _register(
            CharacteristicName.KURT,
            self._requirements_for_all(*self._statistics_for_moment_order(4)),
            self._make_kurt,
        )

        if kind == Kind.CONTINUOUS:
            _register(
                CharacteristicName.CDF,
                self._requirements_for_all(CharacteristicName.CDF),
                self._make_continuous_cdf,
            )
            _register(
                CharacteristicName.PDF,
                self._requirements_for_all(CharacteristicName.PDF),
                self._make_continuous_pdf,
            )
        elif kind == Kind.DISCRETE:
            _register(
                CharacteristicName.PMF,
                self._requirements_for_all(CharacteristicName.PMF),
                self._make_discrete_pmf,
            )
            _register(
                CharacteristicName.CDF,
                self._requirements_for_all(CharacteristicName.PMF),
                self._make_discrete_cdf,
            )
            _register(
                CharacteristicName.PPF,
                self._requirements_for_all(CharacteristicName.PMF),
                self._make_discrete_ppf,
            )
        else:
            raise TypeError("Unsupported distribution kind for finite mixture.")

        if computations:
            return computations, loop_analytical_flags

        raise RuntimeError(
            "Finite mixture produced no analytical computations. "
            "At least one source characteristic must be present."
        )

    @staticmethod
    def _eval_nullary_scalar(method: Method[Any, Any], **options: Any) -> float:
        """Evaluate nullary method and cast to scalar float."""
        return float(np.asarray(method(**options), dtype=float))

    @staticmethod
    def _kurt_raw_from_method(method: Method[Any, Any], **options: Any) -> float:
        """Evaluate raw kurtosis from a method with optional ``excess`` support."""
        try:
            value = method(excess=False, **options)
        except TypeError:
            value = method(**options)
        return float(np.asarray(value, dtype=float))

    @staticmethod
    def _raw_moments_from_statistics(
        mean: float,
        variance: float,
        skewness: float,
        raw_kurtosis: float,
    ) -> tuple[float, float, float, float]:
        """Convert mean/variance/skewness/kurtosis to raw moments up to order four."""
        variance_safe = max(variance, 0.0)
        std = sqrt(variance_safe)
        mu3 = skewness * std**3
        mu4 = raw_kurtosis * variance_safe**2

        m1 = mean
        m2 = variance_safe + mean**2
        m3 = mu3 + 3.0 * mean * variance_safe + mean**3
        m4 = mu4 + 4.0 * mean * mu3 + 6.0 * mean**2 * variance_safe + mean**4
        return m1, m2, m3, m4

    @staticmethod
    def _central_moments_from_raw(
        m1: float,
        m2: float,
        m3: float,
        m4: float,
    ) -> tuple[float, float, float]:
        """Convert raw moments to central moments ``(var, mu3, mu4)``."""
        variance = max(m2 - m1**2, 0.0)
        mu3 = m3 - 3.0 * m1 * m2 + 2.0 * m1**3
        mu4 = m4 - 4.0 * m1 * m3 + 6.0 * m1**2 * m2 - 3.0 * m1**4
        return variance, mu3, mu4

    def _result_raw_moments(
        self,
        sources: ResolvedSourceMethods,
        *,
        max_order: int = _MAX_MOMENT_ORDER,
        **options: Any,
    ) -> tuple[float, float, float, float]:
        """Compute mixture raw moments up to ``max_order``."""
        order = self._validate_moment_order(max_order)
        m1_total = 0.0
        m2_total = 0.0
        m3_total = 0.0
        m4_total = 0.0

        for index, role in enumerate(self._roles):
            methods = sources[role]
            mean = self._eval_nullary_scalar(methods[CharacteristicName.MEAN], **options)
            m1 = mean
            m2 = 0.0
            m3 = 0.0
            m4 = 0.0

            if order >= 2:
                variance = self._eval_nullary_scalar(methods[CharacteristicName.VAR], **options)
                m2 = max(variance, 0.0) + mean**2
            else:
                variance = 0.0

            skewness = 0.0
            if order >= 3:
                skewness = self._eval_nullary_scalar(methods[CharacteristicName.SKEW], **options)
                _, _, m3, _ = self._raw_moments_from_statistics(mean, variance, skewness, 3.0)

            if order == 4:
                raw_kurtosis = self._kurt_raw_from_method(
                    methods[CharacteristicName.KURT],
                    **options,
                )
                _, _, _, m4 = self._raw_moments_from_statistics(
                    mean,
                    variance,
                    skewness,
                    raw_kurtosis,
                )

            weight = float(self._weights[index])
            m1_total += weight * m1
            if order >= 2:
                m2_total += weight * m2
            if order >= 3:
                m3_total += weight * m3
            if order == 4:
                m4_total += weight * m4

        return m1_total, m2_total, m3_total, m4_total

    def _make_mean(self, sources: ResolvedSourceMethods) -> ComputationFunc[Any, float]:
        """Build mixture mean."""

        def _mean(**options: Any) -> float:
            m1, _, _, _ = self._result_raw_moments(sources, max_order=1, **options)
            return m1

        return _mean

    def _make_var(self, sources: ResolvedSourceMethods) -> ComputationFunc[Any, float]:
        """Build mixture variance."""

        def _var(**options: Any) -> float:
            m1, m2, _, _ = self._result_raw_moments(sources, max_order=2, **options)
            return max(m2 - m1**2, 0.0)

        return _var

    def _make_skew(self, sources: ResolvedSourceMethods) -> ComputationFunc[Any, float]:
        """Build mixture skewness."""

        def _skew(**options: Any) -> float:
            m1, m2, m3, _ = self._result_raw_moments(sources, max_order=3, **options)
            variance, mu3, _ = self._central_moments_from_raw(m1, m2, m3, 0.0)
            if variance <= 0.0:
                return 0.0
            return float(mu3 / variance**1.5)

        return _skew

    def _make_kurt(self, sources: ResolvedSourceMethods) -> ComputationFunc[Any, float]:
        """Build mixture raw or excess kurtosis."""

        def _kurt(*, excess: bool = False, **options: Any) -> float:
            m1, m2, m3, m4 = self._result_raw_moments(sources, max_order=4, **options)
            variance, _, mu4 = self._central_moments_from_raw(m1, m2, m3, m4)
            raw = 3.0 if variance <= 0.0 else mu4 / variance**2
            return raw - 3.0 if excess else raw

        return _kurt

    def _make_cf(
        self,
        sources: ResolvedSourceMethods,
    ) -> ComputationFunc[NumericArray, ComplexArray]:
        """Build mixture characteristic function."""
        methods = [
            cast(Method[NumericArray, ComplexArray], sources[role][CharacteristicName.CF])
            for role in self._roles
        ]

        def _cf(data: NumericArray, **options: Any) -> ComplexArray:
            array = np.asarray(data, dtype=float)
            result = np.zeros(array.shape, dtype=complex)
            for weight, method in zip(self._weights, methods, strict=True):
                result += float(weight) * np.asarray(method(array, **options), dtype=complex)
            return cast(ComplexArray, result)

        return cast(ComputationFunc[NumericArray, ComplexArray], _cf)

    def _make_continuous_pdf(
        self,
        sources: ResolvedSourceMethods,
    ) -> ComputationFunc[NumericArray, NumericArray]:
        """Build mixture PDF for continuous components."""
        methods = [
            cast(Method[NumericArray, NumericArray], sources[role][CharacteristicName.PDF])
            for role in self._roles
        ]

        def _pdf(data: NumericArray, **options: Any) -> NumericArray:
            array = np.asarray(data, dtype=float)
            result = np.zeros(array.shape, dtype=float)
            for weight, method in zip(self._weights, methods, strict=True):
                result += float(weight) * np.asarray(method(array, **options), dtype=float)
            return cast(NumericArray, result)

        return cast(ComputationFunc[NumericArray, NumericArray], _pdf)

    def _make_continuous_cdf(
        self,
        sources: ResolvedSourceMethods,
    ) -> ComputationFunc[NumericArray, NumericArray]:
        """Build mixture CDF for continuous components."""
        methods = [
            cast(Method[NumericArray, NumericArray], sources[role][CharacteristicName.CDF])
            for role in self._roles
        ]

        def _cdf(data: NumericArray, **options: Any) -> NumericArray:
            array = np.asarray(data, dtype=float)
            result = np.zeros(array.shape, dtype=float)
            for weight, method in zip(self._weights, methods, strict=True):
                result += float(weight) * np.asarray(method(array, **options), dtype=float)
            return cast(NumericArray, np.clip(result, 0.0, 1.0))

        return cast(ComputationFunc[NumericArray, NumericArray], _cdf)

    def _discrete_mass_table(
        self,
        sources: ResolvedSourceMethods,
        **options: Any,
    ) -> tuple[NumericArray, NumericArray, NumericArray]:
        """Build normalized finite PMF table for discrete mixture."""
        if not options and self._cached_discrete_mass_table is not None:
            return self._cached_discrete_mass_table

        if self._discrete_points is None:
            raise RuntimeError(
                "Discrete finite mixture requires ExplicitTableDiscreteSupport "
                "for every component."
            )

        methods = [
            cast(Method[NumericArray, NumericArray], sources[role][CharacteristicName.PMF])
            for role in self._roles
        ]
        points = self._discrete_points
        masses = np.zeros(points.shape, dtype=float)
        for weight, method in zip(self._weights, methods, strict=True):
            masses += float(weight) * np.asarray(method(points, **options), dtype=float)

        masses = np.clip(masses, 0.0, None)
        total = float(np.sum(masses))
        if total <= 0.0:
            raise RuntimeError("Discrete finite mixture produced non-positive total mass.")

        masses = cast(NumericArray, masses / total)
        cdf_values = cast(NumericArray, np.cumsum(masses, dtype=float))
        cdf_values[-1] = 1.0
        output = (points, masses, cdf_values)
        if not options:
            self._cached_discrete_mass_table = output
        return output

    def _make_discrete_pmf(
        self,
        sources: ResolvedSourceMethods,
    ) -> ComputationFunc[NumericArray, NumericArray]:
        """Build mixture PMF for discrete components."""

        def _pmf(data: NumericArray, **options: Any) -> NumericArray:
            points, masses, _ = self._discrete_mass_table(sources, **options)
            array = np.asarray(data, dtype=float)
            flat = array.reshape(-1)

            indices = np.searchsorted(points, flat, side="left")
            values = np.zeros_like(flat, dtype=float)

            in_range = indices < points.size
            if np.any(in_range):
                in_range_indices = indices[in_range]
                close = np.isclose(points[in_range_indices], flat[in_range], atol=1e-12, rtol=0.0)
                if np.any(close):
                    values[np.where(in_range)[0][close]] = masses[in_range_indices[close]]

            return cast(NumericArray, values.reshape(array.shape))

        return cast(ComputationFunc[NumericArray, NumericArray], _pmf)

    def _make_discrete_cdf(
        self,
        sources: ResolvedSourceMethods,
    ) -> ComputationFunc[NumericArray, NumericArray]:
        """Build mixture CDF for discrete components."""

        def _cdf(data: NumericArray, **options: Any) -> NumericArray:
            points, _, cdf_values = self._discrete_mass_table(sources, **options)
            array = np.asarray(data, dtype=float)
            flat = array.reshape(-1)

            indices = np.searchsorted(points, flat, side="right") - 1
            values = np.zeros_like(flat, dtype=float)

            valid = indices >= 0
            if np.any(valid):
                clipped = np.minimum(indices[valid], cdf_values.size - 1)
                values[valid] = cdf_values[clipped]

            return cast(NumericArray, values.reshape(array.shape))

        return cast(ComputationFunc[NumericArray, NumericArray], _cdf)

    def _make_discrete_ppf(
        self,
        sources: ResolvedSourceMethods,
    ) -> ComputationFunc[NumericArray, NumericArray]:
        """Build mixture PPF for discrete components."""

        def _ppf(data: NumericArray, **options: Any) -> NumericArray:
            points, _, cdf_values = self._discrete_mass_table(sources, **options)
            array = np.asarray(data, dtype=float)
            flat = array.reshape(-1)
            if np.any((flat < 0.0) | (flat > 1.0)):
                raise ValueError("PPF input must be in [0, 1].")

            indices = np.searchsorted(cdf_values, flat, side="left")
            clipped = np.clip(indices, 0, points.size - 1)
            values = points[clipped]
            return cast(NumericArray, values.reshape(array.shape))

        return cast(ComputationFunc[NumericArray, NumericArray], _ppf)


def finite_mixture(
    weighted_components: Sequence[tuple[float, Distribution]],
) -> FiniteMixtureDistribution:
    """
    Build a finite weighted mixture distribution.

    Parameters
    ----------
    weighted_components : Sequence[tuple[float, Distribution]]
        Ordered component pairs ``(weight, distribution)``.

    Returns
    -------
    FiniteMixtureDistribution
        Mixture distribution.
    """
    return FiniteMixtureDistribution(
        weighted_components=weighted_components,
    )


def discrete_mixture(
    weighted_components: Sequence[tuple[float, Distribution]],
) -> FiniteMixtureDistribution:
    """
    Build a finite mixture with a discrete set of component weights.

    This is an alias of :func:`finite_mixture`.
    """
    return finite_mixture(
        weighted_components=weighted_components,
    )


__all__ = [
    "FiniteMixtureDistribution",
    "discrete_mixture",
    "finite_mixture",
]
