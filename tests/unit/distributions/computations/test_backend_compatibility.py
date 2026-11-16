"""Regression tests for preparation on a non-NumPy array backend."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest

from pysatl_core.arrays import (
    ArrayBackend,
    ArrayContext,
    Backend,
    Device,
    DType,
    ValueKind,
    ValueProfile,
    array_operation,
    initialize_backend,
)
from pysatl_core.arrays.state import _reset_backend
from pysatl_core.distributions import AnalyticalComputation, strategies as strategies_module
from pysatl_core.distributions.computations.continuous import _build_cdf_to_pdf_1C
from pysatl_core.distributions.registry import CharacteristicRegistry
from pysatl_core.types import CharacteristicName, ComputationFunc, Kind
from tests.utils.mocks import StandaloneEuclideanUnivariateDistribution


@dataclass
class _NativeArray:
    values: np.ndarray[Any, Any]

    @property
    def ndim(self) -> int:
        return self.values.ndim

    def __array__(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("Implicit conversion of native arrays to NumPy is forbidden.")


def test_cdf_to_pdf_prepares_and_runs_on_native_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = ArrayBackend("native")
    backend.register(lambda device: device is Device.CPU, name="supports_device")
    backend.register(str, name="normalize_native_dtype")
    backend.register(lambda: {DType.FLOAT64: "float64"}, name="dtype_mappings")
    inspected: list[_NativeArray] = []

    @array_operation(backend, "is_native")
    def is_native(value: object) -> bool:
        return isinstance(value, _NativeArray)

    @array_operation(backend, "inspect")
    def inspect(value: _NativeArray, *, context: ArrayContext) -> ValueProfile:
        inspected.append(value)
        return ValueProfile(
            kind=ValueKind.SCALAR if value.ndim == 0 else ValueKind.ARRAY,
            dtype=context.normalize_dtype(value.values.dtype),
            device=Device.CPU,
            shape=value.values.shape,
            ndim=value.ndim,
            size=value.values.size,
        )

    def unwrap(value: Any) -> Any:
        return value.values if isinstance(value, _NativeArray) else value

    @array_operation(backend, "asarray")
    def asarray(value: Any, dtype: Any = None) -> _NativeArray:
        return _NativeArray(np.asarray(unwrap(value), dtype=dtype))

    @array_operation(backend, "atleast_1d")
    def atleast_1d(value: _NativeArray) -> _NativeArray:
        return _NativeArray(np.atleast_1d(value.values))

    def register_binary(name: str, operation: Callable[..., Any]) -> None:
        @array_operation(backend, name)
        def binary(left: Any, right: Any) -> _NativeArray:
            return _NativeArray(operation(unwrap(left), unwrap(right)))

    for name, operation in (
        ("add", np.add),
        ("subtract", np.subtract),
        ("multiply", np.multiply),
        ("divide", np.divide),
    ):
        register_binary(name, operation)

    @array_operation(backend, "clip")
    def clip(value: _NativeArray, minimum: Any, maximum: Any) -> _NativeArray:
        return _NativeArray(np.clip(value.values, minimum, maximum))

    def cdf(value: Any, **options: Any) -> Any:
        if not isinstance(value, _NativeArray):
            raise TypeError("CDF accepts only this backend's native arrays.")
        assert any(previous is value for previous in inspected)
        return _NativeArray(1.0 / (1.0 + np.exp(-value.values)))

    registry = CharacteristicRegistry()
    registry.add_characteristic(CharacteristicName.CDF, is_definitive=True)
    registry.add_characteristic(CharacteristicName.PDF, is_definitive=False)
    registry.add_computation(_build_cdf_to_pdf_1C().to_computation_method())
    monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: registry)
    distribution = StandaloneEuclideanUnivariateDistribution(
        kind=Kind.CONTINUOUS,
        analytical_computations={
            CharacteristicName.CDF: AnalyticalComputation[Any, Any](
                target=CharacteristicName.CDF, func=cast(ComputationFunc[Any, Any], cdf)
            )
        },
    )

    _reset_backend()
    try:
        initialize_backend(Backend("native", backend))
        plan = distribution.explain_computation_path(CharacteristicName.PDF)
        assert plan.steps[-1].target == CharacteristicName.PDF

        pdf = distribution.query_method(CharacteristicName.PDF)
        result = pdf(_NativeArray(np.array([0.0])))

        assert isinstance(result, _NativeArray)
        assert len(inspected) == 6
        np.testing.assert_allclose(result.values, [0.25], atol=1e-8)
    finally:
        _reset_backend()
