from __future__ import annotations

import gc
import weakref
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pytest

from pysatl_core.distributions.computations.computation import AnalyticalComputation
from pysatl_core.distributions.support import ExplicitTableDiscreteSupport
from pysatl_core.transformations.operations.distributions.binary import binary
from pysatl_core.transformations.operations.distributions.binary.base import BinaryDistribution
from pysatl_core.transformations.operations.distributions.mixture import (
    FiniteMixtureDistribution,
    finite_mixture,
)
from pysatl_core.types import (
    BinaryOperationName,
    CharacteristicName,
    ComputationFunc,
    Kind,
    Method,
    NumericArray,
    ResolvedSourceMethods,
)
from tests.utils.mocks import StandaloneEuclideanUnivariateDistribution


@dataclass
class _CountingPMF:
    # These methods are unhashable and compare equal even when their masses differ.
    masses: tuple[float, ...] = field(compare=False)
    calls: int = field(default=0, init=False, compare=False)
    target: str = field(default=CharacteristicName.PMF, init=False)

    def __call__(self, data: NumericArray, **options: Any) -> NumericArray:
        self.calls += 1
        array = np.asarray(data, dtype=float)
        shift = float(options.get("shift", 0.0))
        values = np.zeros_like(array)
        for point, mass in enumerate(self.masses):
            values += mass * (array == point + shift)
        return cast(NumericArray, values)


def _parent(method: _CountingPMF) -> StandaloneEuclideanUnivariateDistribution:
    return StandaloneEuclideanUnivariateDistribution(
        kind=Kind.DISCRETE,
        analytical_computations={
            CharacteristicName.PMF: AnalyticalComputation[NumericArray, NumericArray](
                target=CharacteristicName.PMF,
                func=cast(ComputationFunc[NumericArray, NumericArray], method),
            ),
        },
        support=ExplicitTableDiscreteSupport([0.0, 1.0, 2.0]),
    )


@dataclass
class _CacheCase:
    kind: str
    distribution: BinaryDistribution | FiniteMixtureDistribution
    methods: tuple[_CountingPMF, _CountingPMF]

    def sources(self, *methods: _CountingPMF) -> ResolvedSourceMethods:
        return {
            role: {CharacteristicName.PMF: cast(Method[Any, Any], method)}
            for role, method in zip(
                self.distribution.parent_roles, methods or self.methods, strict=True
            )
        }

    def point_masses(self, left: float, right: float) -> dict[float, float]:
        if self.kind == "mixture":
            masses = {left: 0.25}
            masses[right] = masses.get(right, 0.0) + 0.75
            return masses
        point = left + right if self.kind == "sum" else left - right
        return {point: 1.0}


def _make_case(
    kind: str,
    left_masses: tuple[float, ...] = (1.0, 0.0, 0.0),
    right_masses: tuple[float, ...] = (1.0, 0.0, 0.0),
) -> _CacheCase:
    methods = (_CountingPMF(left_masses), _CountingPMF(right_masses))
    left, right = map(_parent, methods)
    distribution: BinaryDistribution | FiniteMixtureDistribution
    if kind == "mixture":
        distribution = finite_mixture([(0.25, left), (0.75, right)])
    else:
        operation = BinaryOperationName.ADD if kind == "sum" else BinaryOperationName.SUB
        distribution = binary(left, right, operation=operation)
    return _CacheCase(kind, distribution, methods)


@pytest.fixture(params=["sum", "difference", "mixture"])
def cache_case(request: pytest.FixtureRequest) -> _CacheCase:
    return _make_case(request.param)


def _assert_table(
    table: tuple[NumericArray, NumericArray, NumericArray],
    expected: dict[float, float],
) -> None:
    points, masses, cdf = table
    assert set(expected).issubset(set(points))
    np.testing.assert_allclose(masses, [expected.get(float(point), 0.0) for point in points])
    assert np.all(masses >= 0.0)
    assert float(np.sum(masses)) == pytest.approx(1.0)
    np.testing.assert_allclose(cdf, np.cumsum(masses))
    assert cdf[-1] == 1.0


def test_equivalent_sources_reuse_table(cache_case: _CacheCase) -> None:
    case = cache_case
    first = case.distribution._discrete_mass_table(case.sources())
    counts = tuple(method.calls for method in case.methods)
    assert all(count > 0 for count in counts)

    # Both mapping levels are new, and the role insertion order is reversed.
    equivalent = dict(reversed(list(case.sources().items())))
    repeated = case.distribution._discrete_mass_table(equivalent)
    assert repeated is first
    assert tuple(method.calls for method in case.methods) == counts
    _assert_table(repeated, case.point_masses(0.0, 0.0))


@pytest.mark.parametrize("role_index", [0, 1])
def test_changed_parent_method_recomputes_and_can_be_restored(
    cache_case: _CacheCase, role_index: int
) -> None:
    case = cache_case
    sources = case.sources()
    original = case.distribution._discrete_mass_table(sources)
    original_counts = tuple(method.calls for method in case.methods)
    replacement = _CountingPMF((0.0, 1.0, 0.0))
    assert replacement == case.methods[role_index]
    role = case.distribution.parent_roles[role_index]

    # Mutating the original container must not mutate the cached dependency snapshot.
    sources[role][CharacteristicName.PMF] = cast(Method[Any, Any], replacement)
    changed = case.distribution._discrete_mass_table(sources)
    assert changed is not original
    assert replacement.calls > 0
    assert case.methods[1 - role_index].calls > original_counts[1 - role_index]
    _assert_table(changed, case.point_masses(float(role_index == 0), float(role_index == 1)))

    selected = list(case.methods)
    selected[role_index] = replacement
    counts = tuple(method.calls for method in selected)
    assert case.distribution._discrete_mass_table(case.sources(*selected)) is changed
    assert tuple(method.calls for method in selected) == counts

    restored = case.distribution._discrete_mass_table(case.sources())
    assert restored is not changed
    assert case.methods[role_index].calls > original_counts[role_index]
    _assert_table(restored, case.point_masses(0.0, 0.0))
    assert case.distribution._discrete_mass_table(case.sources()) is restored


def test_method_roles_are_part_of_dependencies(cache_case: _CacheCase) -> None:
    case = cache_case
    zero, one = _CountingPMF((1.0, 0.0, 0.0)), _CountingPMF((0.0, 1.0, 0.0))
    first = case.distribution._discrete_mass_table(case.sources(zero, one))
    counts = (zero.calls, one.calls)
    _assert_table(first, case.point_masses(0.0, 1.0))

    swapped = case.distribution._discrete_mass_table(case.sources(one, zero))
    assert swapped is not first
    assert zero.calls > counts[0]
    assert one.calls > counts[1]
    _assert_table(swapped, case.point_masses(1.0, 0.0))


def test_options_bypass_cache_without_replacing_default_table(cache_case: _CacheCase) -> None:
    case = cache_case
    default = case.distribution._discrete_mass_table(case.sources())
    for shift in [0.0, 1.0, 0.0]:
        counts = tuple(method.calls for method in case.methods)
        # Arbitrary options can contain unhashable values; bypass them safely.
        shifted = case.distribution._discrete_mass_table(case.sources(), shift=np.asarray(shift))
        assert shifted is not default
        assert all(method.calls > count for method, count in zip(case.methods, counts, strict=True))
        _assert_table(shifted, case.point_masses(shift, shift))

    replacement = _CountingPMF((0.0, 1.0, 0.0))
    alternative = case.distribution._discrete_mass_table(
        case.sources(replacement, case.methods[1]), shift=1.0
    )
    _assert_table(alternative, case.point_masses(2.0, 1.0))
    counts = tuple(method.calls for method in case.methods)
    assert case.distribution._discrete_mass_table(case.sources()) is default
    assert tuple(method.calls for method in case.methods) == counts
    _assert_table(default, case.point_masses(0.0, 0.0))


def test_cached_dependencies_are_retained_only_until_replacement(cache_case: _CacheCase) -> None:
    case = cache_case
    temporary = (_CountingPMF((0.0, 1.0, 0.0)), _CountingPMF((0.0, 0.0, 1.0)))
    references = tuple(weakref.ref(method) for method in temporary)
    sources = case.sources(*temporary)
    case.distribution._discrete_mass_table(sources)
    del sources, temporary
    gc.collect()
    assert all(reference() is not None for reference in references)

    case.distribution._discrete_mass_table(case.sources())
    gc.collect()
    assert all(reference() is None for reference in references)


def test_public_pmf_cdf_ppf_share_a_normalized_table(cache_case: _CacheCase) -> None:
    case = _make_case(cache_case.kind, (0.25, 0.75, 0.0), (0.6, 0.4, 0.0))
    expected = {
        "sum": {0.0: 0.15, 1.0: 0.55, 2.0: 0.3},
        "difference": {-1.0: 0.1, 0.0: 0.45, 1.0: 0.45},
        "mixture": {0.0: 0.5125, 1.0: 0.4875},
    }[case.kind]
    points = np.asarray(sorted(expected), dtype=float)
    masses = np.asarray([expected[point] for point in points])
    cumulative = np.cumsum(masses)
    probabilities = cumulative - 0.5 * masses
    pmf = case.distribution.query_method(CharacteristicName.PMF)
    cdf = case.distribution.query_method(CharacteristicName.CDF)
    ppf = case.distribution.query_method(CharacteristicName.PPF)

    np.testing.assert_allclose(pmf(points), masses)
    assert float(np.sum(pmf(points))) == pytest.approx(1.0)
    counts = tuple(method.calls for method in case.methods)
    assert all(count > 0 for count in counts)
    np.testing.assert_allclose(cdf(points), cumulative)
    np.testing.assert_allclose(ppf(probabilities), points)
    assert float(cdf(points[0] - 1.0)) == 0.0
    assert float(cdf(points[-1] + 1.0)) == 1.0
    assert float(pmf(points[-1] + 1.0)) == 0.0
    assert float(ppf(1.0)) == points[-1]
    assert tuple(method.calls for method in case.methods) == counts

    offset = 2.0 if case.kind == "sum" else 1.0 if case.kind == "mixture" else 0.0
    np.testing.assert_allclose(pmf(points + offset, shift=1.0), masses)
    np.testing.assert_allclose(cdf(points + offset, shift=1.0), cumulative)
    np.testing.assert_allclose(ppf(probabilities, shift=1.0), points + offset)
    np.testing.assert_allclose(pmf(points), masses)


@pytest.mark.parametrize("role_index", [0, 1, 2])
def test_mixture_tracks_every_component_method(role_index: int) -> None:
    methods = [_CountingPMF((1.0, 0.0, 0.0)) for _ in range(3)]
    weights = [0.2, 0.3, 0.5]
    mixture = finite_mixture(list(zip(weights, map(_parent, methods), strict=True)))
    sources: ResolvedSourceMethods = {
        role: {CharacteristicName.PMF: cast(Method[Any, Any], method)}
        for role, method in zip(mixture.parent_roles, methods, strict=True)
    }
    original = mixture._discrete_mass_table(sources)
    counts = [method.calls for method in methods]
    replacement = _CountingPMF((0.0, 1.0, 0.0))
    sources[mixture.parent_roles[role_index]][CharacteristicName.PMF] = cast(
        Method[Any, Any], replacement
    )
    changed = mixture._discrete_mass_table(sources)
    assert changed is not original
    assert replacement.calls > 0
    assert all(method.calls > counts[i] for i, method in enumerate(methods) if i != role_index)
    _assert_table(changed, {0.0: 1.0 - weights[role_index], 1.0: weights[role_index]})
    assert mixture._discrete_mass_table(sources) is changed
