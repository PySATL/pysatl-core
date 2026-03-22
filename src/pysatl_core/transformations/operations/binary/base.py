"""
Base abstractions for binary transformations over distributions.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from math import inf, isfinite, sqrt
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from pysatl_core.distributions.computation import Method
from pysatl_core.distributions.distribution import Distribution
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
    BinaryOperationName,
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

_LEFT_ROLE: ParentRole = "left"
_RIGHT_ROLE: ParentRole = "right"
_MAX_MOMENT_ORDER = 4


class BinaryDistribution(DerivedDistribution, ABC):
    """
    Base class for binary transformations over two parent distributions.

    Parameters
    ----------
    left_distribution : Distribution
        Left parent distribution.
    right_distribution : Distribution
        Right parent distribution.
    operation : BinaryOperationName
        Binary operation identifier.
    sampling_strategy : SamplingStrategy | None, optional
        Sampling strategy exposed by the transformed distribution.
    computation_strategy : ComputationStrategy | None, optional
        Computation strategy exposed by the transformed distribution.
    """

    def __init__(
        self,
        left_distribution: Distribution,
        right_distribution: Distribution,
        *,
        operation: BinaryOperationName,
        sampling_strategy: SamplingStrategy | None = None,
        computation_strategy: ComputationStrategy | None = None,
    ) -> None:
        self._operation = operation

        left_snapshot = LightweightDistribution.from_distribution(left_distribution)
        right_snapshot = LightweightDistribution.from_distribution(right_distribution)

        self._left_distribution = left_snapshot
        self._right_distribution = right_snapshot
        self._cached_discrete_mass_table: tuple[NumericArray, NumericArray, NumericArray] | None = (
            None
        )

        distribution_type = self._validate_distribution_types(
            left_snapshot.distribution_type,
            right_snapshot.distribution_type,
        )
        bases: dict[ParentRole, LightweightDistribution] = {
            _LEFT_ROLE: left_snapshot,
            _RIGHT_ROLE: right_snapshot,
        }
        transformed_support = self._transform_support(left_snapshot.support, right_snapshot.support)
        self._precomputed_support = transformed_support
        analytical_computations, loop_analytical_flags = self._build_analytical_computations(
            distribution_type=distribution_type,
            bases=bases,
        )

        super().__init__(
            distribution_type=distribution_type,
            bases=bases,
            analytical_computations=analytical_computations,
            transformation_name=TransformationName.BINARY,
            support=transformed_support,
            sampling_strategy=sampling_strategy,
            computation_strategy=computation_strategy,
            loop_analytical_flags=loop_analytical_flags,
        )

    @property
    def left_distribution(self) -> LightweightDistribution:
        """Get the lightweight snapshot of the left parent distribution."""
        return self._left_distribution

    @property
    def right_distribution(self) -> LightweightDistribution:
        """Get the lightweight snapshot of the right parent distribution."""
        return self._right_distribution

    @property
    def operation(self) -> BinaryOperationName:
        """Get the binary operation name."""
        return self._operation

    def sample(self, n: int, **options: Any) -> NumericArray:
        """
        Generate transformed samples from two parent-distribution samples.

        The method samples both parents using the currently attached
        sampling strategy and applies the concrete binary operation
        element-wise.
        """
        left_samples = np.asarray(
            self.sampling_strategy.sample(n, distr=self.left_distribution, **options),
            dtype=float,
        )
        right_samples = np.asarray(
            self.sampling_strategy.sample(n, distr=self.right_distribution, **options),
            dtype=float,
        )
        transformed_samples = self._operation_value_array(left_samples, right_samples)
        return cast(NumericArray, np.asarray(transformed_samples, dtype=float))

    @abstractmethod
    def _characteristic_specs(
        self, *, kind: Kind | None
    ) -> tuple[
        tuple[
            GenericCharacteristicName,
            SourceRequirements,
            TransformationEvaluator[Any, Any],
        ],
        ...,
    ]:
        """Return characteristic registration specs for the concrete operation."""

    @abstractmethod
    def _operation_value(
        self,
        left: float | NumericArray,
        right: float | NumericArray,
    ) -> float | NumericArray:
        """Apply the concrete binary operation in scalar or array semantics."""

    def _operation_value_array(self, left: NumericArray, right: NumericArray) -> NumericArray:
        """Apply binary operation to broadcast-compatible arrays."""
        return cast(NumericArray, np.asarray(self._operation_value(left, right), dtype=float))

    @abstractmethod
    def _result_raw_moments(
        self,
        sources: ResolvedSourceMethods,
        *,
        max_order: int = _MAX_MOMENT_ORDER,
        **options: Any,
    ) -> tuple[float, float, float, float]:
        """Compute transformed raw moments up to ``max_order`` (1..4)."""

    @abstractmethod
    def _make_cf(
        self,
        sources: ResolvedSourceMethods,
    ) -> ComputationFunc[NumericArray, ComplexArray]:
        """Build transformed characteristic function."""

    @abstractmethod
    def _make_continuous_pdf(
        self,
        sources: ResolvedSourceMethods,
    ) -> ComputationFunc[NumericArray, NumericArray]:
        """Build transformed continuous PDF."""

    @abstractmethod
    def _make_continuous_cdf(
        self,
        sources: ResolvedSourceMethods,
    ) -> ComputationFunc[NumericArray, NumericArray]:
        """Build transformed continuous CDF."""

    def _build_analytical_computations(
        self,
        *,
        distribution_type: DistributionType,
        bases: Mapping[ParentRole, LightweightDistribution],
    ) -> tuple[
        Mapping[GenericCharacteristicName, Mapping[LabelName, TransformationMethod[Any, Any]]],
        Mapping[GenericCharacteristicName, Mapping[LabelName, bool]],
    ]:
        """
        Build analytical computations for the concrete binary transformation.
        """
        computations: dict[
            GenericCharacteristicName, dict[LabelName, TransformationMethod[Any, Any]]
        ] = {}
        loop_analytical_flags: dict[GenericCharacteristicName, dict[LabelName, bool]] = {}
        kind = cast(Kind | None, getattr(distribution_type, "kind", None))
        declared_registry_characteristics = characteristic_registry().declared_characteristics

        def _register(
            target: GenericCharacteristicName,
            source_requirements: SourceRequirements,
            evaluator: TransformationEvaluator[Any, Any],
        ) -> None:
            # Non-registry characteristics cannot be resolved via graph fitters.
            # For them we require direct parent presence for each required source.
            for role, characteristics in source_requirements.items():
                base = bases[role]
                for characteristic in characteristics:
                    if characteristic in declared_registry_characteristics:
                        continue
                    if characteristic not in base.analytical_computations:
                        return

            method, is_analytical, has_any_present_source = TransformationMethod.try_from_parents(
                target=target,
                transformation=TransformationName.BINARY,
                bases=bases,
                source_requirements=source_requirements,
                evaluator=evaluator,
            )
            if method is None or not has_any_present_source:
                return
            computations[target] = {DEFAULT_ANALYTICAL_COMPUTATION_LABEL: method}
            loop_analytical_flags[target] = {DEFAULT_ANALYTICAL_COMPUTATION_LABEL: is_analytical}

        for target, source_requirements, evaluator in self._characteristic_specs(kind=kind):
            _register(target, source_requirements, evaluator)

        if computations:
            return computations, loop_analytical_flags

        raise RuntimeError(
            "Binary transformation produced no analytical computations. "
            "At least one source characteristic must be present."
        )

    @staticmethod
    def _validate_distribution_types(
        left_type: DistributionType,
        right_type: DistributionType,
    ) -> DistributionType:
        """
        Validate compatibility of parent distribution types.

        Parameters
        ----------
        left_type : DistributionType
            Type descriptor of the left parent.
        right_type : DistributionType
            Type descriptor of the right parent.

        Returns
        -------
        DistributionType
            Type descriptor used by the transformed distribution.
        """
        left_dimension = getattr(left_type, "dimension", None)
        right_dimension = getattr(right_type, "dimension", None)
        left_kind = getattr(left_type, "kind", None)
        right_kind = getattr(right_type, "kind", None)

        if left_dimension != 1 or right_dimension != 1:
            raise TypeError(
                "BinaryDistribution currently supports only one-dimensional distributions."
            )
        if left_kind not in {Kind.CONTINUOUS, Kind.DISCRETE}:
            raise TypeError("Unsupported distribution kind for binary transformation.")
        if left_kind != right_kind:
            raise TypeError(
                "BinaryDistribution currently requires both parents to have the same kind."
            )
        return left_type

    @staticmethod
    def _requirements_both(
        *characteristics: GenericCharacteristicName,
    ) -> SourceRequirements:
        """Build source requirements for both parent roles."""
        return {
            _LEFT_ROLE: tuple(characteristics),
            _RIGHT_ROLE: tuple(characteristics),
        }

    @staticmethod
    def _density_characteristic(kind: Kind | None) -> GenericCharacteristicName:
        """Resolve density characteristic for the given kind."""
        return CharacteristicName.PDF if kind == Kind.CONTINUOUS else CharacteristicName.PMF

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
        """Return parent statistics needed to recover raw moments up to ``max_order``."""
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

    @staticmethod
    def _eval_method_scalar(
        method: Method[Any, Any],
        argument: float,
        **options: Any,
    ) -> float:
        """Evaluate a scalar-valued method at one point and cast to ``float``."""
        return float(np.asarray(method(argument, **options), dtype=float))

    @staticmethod
    def _eval_method_scalar_complex(
        method: Method[Any, Any],
        argument: float,
        **options: Any,
    ) -> complex:
        """Evaluate a complex-valued method at one point and cast to ``complex``."""
        return complex(np.asarray(method(argument, **options), dtype=complex))

    @staticmethod
    def _eval_nullary_scalar(
        method: Method[Any, Any],
        **options: Any,
    ) -> float:
        """Evaluate a nullary method and cast to ``float``."""
        return float(np.asarray(method(**options), dtype=float))

    @staticmethod
    def _map_scalar_real(
        data: NumericArray,
        scalar_func: Callable[[float], float],
    ) -> NumericArray:
        """Apply a scalar real function to scalar or vector input."""
        array = np.asarray(data, dtype=float)
        flat = array.reshape(-1)
        mapped = np.fromiter((scalar_func(float(x)) for x in flat), dtype=float, count=flat.size)
        return cast(NumericArray, mapped.reshape(array.shape))

    @staticmethod
    def _map_scalar_complex(
        data: NumericArray,
        scalar_func: Callable[[float], complex],
    ) -> ComplexArray:
        """Apply a scalar complex function to scalar or vector input."""
        array = np.asarray(data, dtype=float)
        flat = array.reshape(-1)
        mapped = np.fromiter((scalar_func(float(x)) for x in flat), dtype=complex, count=flat.size)
        return cast(ComplexArray, mapped.reshape(array.shape))

    @staticmethod
    def _quad_real(
        integrand: Callable[[float], float],
        left: float,
        right: float,
    ) -> float:
        """Integrate a real-valued function with SciPy ``quad``."""
        value, _ = quad(integrand, left, right, limit=300)
        return float(value)

    @classmethod
    def _integrate_real(
        cls,
        integrand: Callable[[float], float],
        left: float,
        right: float,
        *,
        split_at_zero: bool = False,
    ) -> float:
        """Integrate a real function and optionally split around zero."""
        if left >= right:
            return 0.0

        if split_at_zero and left < 0.0 < right:
            eps_pos = float(np.nextafter(0.0, 1.0))
            eps_neg = float(np.nextafter(0.0, -1.0))
            return cls._quad_real(integrand, left, eps_neg) + cls._quad_real(
                integrand, eps_pos, right
            )

        return cls._quad_real(integrand, left, right)

    @classmethod
    def _integrate_complex(
        cls,
        integrand: Callable[[float], complex],
        left: float,
        right: float,
        *,
        split_at_zero: bool = False,
    ) -> complex:
        """Integrate a complex function by integrating real and imaginary parts."""
        real_part = cls._integrate_real(
            lambda x: float(np.real(integrand(x))),
            left,
            right,
            split_at_zero=split_at_zero,
        )
        imag_part = cls._integrate_real(
            lambda x: float(np.imag(integrand(x))),
            left,
            right,
            split_at_zero=split_at_zero,
        )
        return complex(real_part, imag_part)

    def _continuous_bounds_for_role(self, role: ParentRole) -> tuple[float, float]:
        """Get integration bounds from continuous support or fallback to real line."""
        support = (
            self.left_distribution.support
            if role == _LEFT_ROLE
            else self.right_distribution.support
        )
        if isinstance(support, ContinuousSupport):
            return float(support.left), float(support.right)
        return -inf, inf

    def _discrete_points_for_role(self, role: ParentRole) -> NumericArray:
        """Get explicit discrete support points for one parent role."""
        support = (
            self.left_distribution.support
            if role == _LEFT_ROLE
            else self.right_distribution.support
        )
        if not isinstance(support, ExplicitTableDiscreteSupport):
            raise RuntimeError(
                "Binary discrete computations require ExplicitTableDiscreteSupport "
                "for both parents."
            )
        return cast(NumericArray, np.asarray(support.points, dtype=float))

    @staticmethod
    def _raw_moments_from_statistics(
        mean: float,
        variance: float,
        skewness: float,
        raw_kurtosis: float,
    ) -> tuple[float, float, float, float]:
        """Convert mean/variance/skewness/kurtosis to raw moments up to order 4."""
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

    @staticmethod
    def _kurt_raw_from_method(
        method: Method[Any, Any],
        **options: Any,
    ) -> float:
        """Evaluate raw kurtosis from a method with optional ``excess`` support."""
        try:
            value = method(excess=False, **options)
        except TypeError:
            value = method(**options)
        return float(np.asarray(value, dtype=float))

    def _parent_raw_moments(
        self,
        sources: ResolvedSourceMethods,
        role: ParentRole,
        *,
        max_order: int = _MAX_MOMENT_ORDER,
        **options: Any,
    ) -> tuple[float, float, float, float]:
        """Get parent raw moments up to ``max_order`` from statistical characteristics."""
        order = self._validate_moment_order(max_order)
        mean_method = sources[role][CharacteristicName.MEAN]
        mean = self._eval_nullary_scalar(mean_method, **options)
        if order == 1:
            return mean, 0.0, 0.0, 0.0

        var_method = sources[role][CharacteristicName.VAR]
        variance = self._eval_nullary_scalar(var_method, **options)
        m1 = mean
        m2 = max(variance, 0.0) + mean**2
        if order == 2:
            return m1, m2, 0.0, 0.0

        skew_method = sources[role][CharacteristicName.SKEW]
        skewness = self._eval_nullary_scalar(skew_method, **options)
        _, _, m3, _ = self._raw_moments_from_statistics(mean, variance, skewness, 3.0)
        if order == 3:
            return m1, m2, m3, 0.0

        kurt_method = sources[role][CharacteristicName.KURT]
        raw_kurtosis = self._kurt_raw_from_method(kurt_method, **options)
        return self._raw_moments_from_statistics(mean, variance, skewness, raw_kurtosis)

    def _make_mean(
        self,
        sources: ResolvedSourceMethods,
    ) -> ComputationFunc[Any, float]:
        """Build transformed mean."""

        def _mean(**options: Any) -> float:
            m1, _, _, _ = self._result_raw_moments(sources, max_order=1, **options)
            return m1

        return _mean

    def _make_var(
        self,
        sources: ResolvedSourceMethods,
    ) -> ComputationFunc[Any, float]:
        """Build transformed variance."""

        def _var(**options: Any) -> float:
            m1, m2, _, _ = self._result_raw_moments(sources, max_order=2, **options)
            return max(m2 - m1**2, 0.0)

        return _var

    def _make_skew(
        self,
        sources: ResolvedSourceMethods,
    ) -> ComputationFunc[Any, float]:
        """Build transformed skewness."""

        def _skew(**options: Any) -> float:
            m1, m2, m3, _ = self._result_raw_moments(sources, max_order=3, **options)
            variance, mu3, _ = self._central_moments_from_raw(m1, m2, m3, 0.0)
            if variance <= 0.0:
                return 0.0
            return float(mu3 / variance**1.5)

        return _skew

    def _make_kurt(
        self,
        sources: ResolvedSourceMethods,
    ) -> ComputationFunc[Any, float]:
        """Build transformed raw or excess kurtosis."""

        def _kurt(*, excess: bool = False, **options: Any) -> float:
            m1, m2, m3, m4 = self._result_raw_moments(sources, max_order=4, **options)
            variance, _, mu4 = self._central_moments_from_raw(m1, m2, m3, m4)
            raw = 3.0 if variance <= 0.0 else mu4 / variance**2
            return raw - 3.0 if excess else raw

        return _kurt

    def _make_continuous_ppf(
        self,
        sources: ResolvedSourceMethods,
    ) -> ComputationFunc[NumericArray, NumericArray]:
        """Build transformed continuous PPF via numerical inversion of transformed CDF."""
        cdf_method = cast(Method[float, float], self._make_continuous_cdf(sources))
        support = self._precomputed_support
        support_left = float(support.left) if isinstance(support, ContinuousSupport) else -inf
        support_right = float(support.right) if isinstance(support, ContinuousSupport) else inf

        def _solve_single_quantile(q: float, **options: Any) -> float:
            def _f(x: float) -> float:
                return float(cdf_method(x, **options) - q)

            def _f_brentq(
                x: np.float64,
                *_args: Any,
                **_kwargs: Any,
            ) -> np.float64:
                return np.float64(_f(float(x)))

            left = support_left
            right = support_right
            if not (isfinite(left) and isfinite(right) and left < right):
                left = -1.0
                right = 1.0
                for _ in range(80):
                    fl = _f(left)
                    fr = _f(right)
                    if fl <= 0.0 <= fr:
                        break
                    left *= 2.0
                    right *= 2.0
                else:
                    raise RuntimeError(
                        "Could not bracket transformed PPF root for binary continuous operation."
                    )
            return float(brentq(cast(Any, _f_brentq), left, right, xtol=1e-10, maxiter=200))

        def _ppf(data: NumericArray, **options: Any) -> NumericArray:
            probabilities = np.asarray(data, dtype=float)
            flat_probabilities = probabilities.reshape(-1)

            if np.any((flat_probabilities < 0.0) | (flat_probabilities > 1.0)):
                raise ValueError("PPF input must be in [0, 1].")

            results = np.empty_like(flat_probabilities, dtype=float)
            is_zero = flat_probabilities == 0.0
            is_one = flat_probabilities == 1.0
            is_interior = ~(is_zero | is_one)

            results[is_zero] = support_left
            results[is_one] = support_right

            interior_indices = np.nonzero(is_interior)[0]
            for idx in interior_indices:
                results[idx] = _solve_single_quantile(float(flat_probabilities[idx]), **options)

            return cast(NumericArray, results.reshape(probabilities.shape))

        return cast(ComputationFunc[NumericArray, NumericArray], _ppf)

    def _discrete_mass_table(
        self,
        sources: ResolvedSourceMethods,
        **options: Any,
    ) -> tuple[NumericArray, NumericArray, NumericArray]:
        """Build transformed finite PMF table for discrete parent supports."""
        if not options and self._cached_discrete_mass_table is not None:
            return self._cached_discrete_mass_table

        left_pmf = sources[_LEFT_ROLE][CharacteristicName.PMF]
        right_pmf = sources[_RIGHT_ROLE][CharacteristicName.PMF]
        left_points = self._discrete_points_for_role(_LEFT_ROLE)
        right_points = self._discrete_points_for_role(_RIGHT_ROLE)
        left_masses = np.asarray(
            [self._eval_method_scalar(left_pmf, float(x), **options) for x in left_points],
            dtype=float,
        )
        right_masses = np.asarray(
            [self._eval_method_scalar(right_pmf, float(y), **options) for y in right_points],
            dtype=float,
        )

        positive_left = np.nonzero(left_masses > 0.0)[0]
        positive_right = np.nonzero(right_masses > 0.0)[0]
        left_values = left_points[positive_left]
        left_weights = left_masses[positive_left]
        right_values = right_points[positive_right]
        right_weights = right_masses[positive_right]

        if self.operation == BinaryOperationName.DIV:
            nonzero_mask = ~np.isclose(right_values, 0.0, atol=0.0, rtol=0.0)
            right_values = right_values[nonzero_mask]
            right_weights = right_weights[nonzero_mask]

        if left_values.size == 0 or right_values.size == 0:
            raise RuntimeError("Binary discrete transformation produced an empty PMF table.")

        transformed = self._operation_value_array(
            left_values[:, None], right_values[None, :]
        ).reshape(-1)
        pair_weights = (left_weights[:, None] * right_weights[None, :]).reshape(-1)

        rounded_points = np.round(transformed, 12)
        unique_points, inverse = np.unique(rounded_points, return_inverse=True)
        accumulated_masses = np.zeros_like(unique_points, dtype=float)
        np.add.at(accumulated_masses, inverse, pair_weights)

        positive_mass_mask = accumulated_masses > 0.0
        points = cast(NumericArray, unique_points[positive_mass_mask])
        masses = cast(NumericArray, accumulated_masses[positive_mass_mask])
        total = float(np.sum(masses))
        if total <= 0.0:
            raise RuntimeError("Binary discrete transformation produced non-positive total mass.")
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
        """Build transformed discrete PMF."""

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
        """Build transformed discrete CDF."""

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
        """Build transformed discrete PPF."""

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

    def _transform_support(
        self,
        left_support: Support | None,
        right_support: Support | None,
    ) -> Support | None:
        """Transform support metadata when both parent supports are explicit enough."""
        if isinstance(left_support, ContinuousSupport) and isinstance(
            right_support, ContinuousSupport
        ):
            finite_bounds = all(
                isfinite(value)
                for value in (
                    float(left_support.left),
                    float(left_support.right),
                    float(right_support.left),
                    float(right_support.right),
                )
            )
            if not finite_bounds:
                return None

            left_bounds = (float(left_support.left), float(left_support.right))
            right_bounds = (float(right_support.left), float(right_support.right))
            if (
                self.operation == BinaryOperationName.DIV
                and right_bounds[0] <= 0.0 <= right_bounds[1]
            ):
                return None

            left_values = np.asarray(
                [left_bounds[0], left_bounds[0], left_bounds[1], left_bounds[1]],
                dtype=float,
            )
            right_values = np.asarray(
                [right_bounds[0], right_bounds[1], right_bounds[0], right_bounds[1]],
                dtype=float,
            )
            values = self._operation_value_array(left_values, right_values)
            closed = (
                left_support.left_closed
                and left_support.right_closed
                and right_support.left_closed
                and right_support.right_closed
            )
            return ContinuousSupport(
                left=float(np.min(values)),
                right=float(np.max(values)),
                left_closed=closed,
                right_closed=closed,
            )

        if isinstance(left_support, ExplicitTableDiscreteSupport) and isinstance(
            right_support, ExplicitTableDiscreteSupport
        ):
            left_points = np.asarray(left_support.points, dtype=float)
            right_points = np.asarray(right_support.points, dtype=float)
            if self.operation == BinaryOperationName.DIV:
                right_points = right_points[~np.isclose(right_points, 0.0, atol=0.0, rtol=0.0)]

            if left_points.size == 0 or right_points.size == 0:
                return None

            transformed = self._operation_value_array(
                left_points[:, None],
                right_points[None, :],
            ).reshape(-1)
            return ExplicitTableDiscreteSupport(points=transformed.tolist(), assume_sorted=False)

        return None


_SUPPORTED_BINARY_OPERATIONS: frozenset[BinaryOperationName] = frozenset(BinaryOperationName)


def binary(
    left_distribution: Distribution,
    right_distribution: Distribution,
    *,
    operation: BinaryOperationName,
) -> BinaryDistribution:
    """
    Apply a binary operation to two distributions.

    Parameters
    ----------
    left_distribution : Distribution
        Left parent distribution.
    right_distribution : Distribution
        Right parent distribution.
    operation : BinaryOperationName
        Binary operation to apply.

    Returns
    -------
    BinaryDistribution
        Derived distribution representing the binary transformation.
    """
    from .division import DivisionBinaryDistribution
    from .linear import LinearBinaryDistribution
    from .multiplication import MultiplicationBinaryDistribution

    if operation in {BinaryOperationName.ADD, BinaryOperationName.SUB}:
        return LinearBinaryDistribution(
            left_distribution=left_distribution,
            right_distribution=right_distribution,
            operation=operation,
        )
    if operation == BinaryOperationName.MUL:
        return MultiplicationBinaryDistribution(
            left_distribution=left_distribution,
            right_distribution=right_distribution,
        )
    if operation == BinaryOperationName.DIV:
        return DivisionBinaryDistribution(
            left_distribution=left_distribution,
            right_distribution=right_distribution,
        )
    raise ValueError(
        f"Unsupported binary operation '{operation}'. "
        f"Supported operations: {', '.join(sorted(_SUPPORTED_BINARY_OPERATIONS))}."
    )


__all__ = [
    "BinaryDistribution",
    "BinaryOperationName",
    "_LEFT_ROLE",
    "_RIGHT_ROLE",
    "binary",
]
