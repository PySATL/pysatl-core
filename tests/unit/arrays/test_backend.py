"""Tests for array backend registration and NumPy dtype mapping."""

from collections.abc import Hashable, Iterator
from typing import cast

import numpy as np
import pytest

from pysatl_core.arrays import (
    ArrayBackend,
    ArrayBackendFrozenError,
    ArrayContext,
    ArrayValidator,
    Backend,
    BackendAlreadyInitializedError,
    Device,
    DeviceMismatchError,
    DType,
    ForeignArrayError,
    NativeArray,
    UnsupportedArrayOperationError,
    UnsupportedDeviceError,
    UnsupportedDTypeError,
    ValueContract,
    ValueKind,
    ValueProfile,
    add,
    array_operation,
    asarray,
    initialize_backend,
    linspace,
    nan_to_num,
    normalize_value,
    resolve_array_backend,
    resolve_array_context,
    resolve_backend,
    scalar,
    zeros,
)
from pysatl_core.arrays.numpy import numpy_array_backend, numpy_backend
from pysatl_core.arrays.state import _reset_backend


def _identity_dtype(value: object) -> Hashable:
    if not isinstance(value, Hashable):
        raise TypeError
    return value


def _configure_dtypes(arrays: ArrayBackend, native_dtypes: dict[DType, Hashable]) -> None:
    arrays.register(_identity_dtype, name="normalize_native_dtype")
    arrays.register(lambda: native_dtypes, name="dtype_mappings")


@pytest.fixture(autouse=True)
def reset_backend() -> Iterator[None]:
    _reset_backend()
    try:
        yield
    finally:
        _reset_backend()


def test_decorator_registers_operation() -> None:
    backend = ArrayBackend("test")

    @array_operation(backend)
    def add(left: int, right: int) -> int:
        return left + right

    assert backend.operation_names == frozenset({"add"})
    assert backend.call("add", 2, 3) == 5
    assert add(2, 3) == 5


def test_duplicate_and_missing_operations_fail_explicitly() -> None:
    backend = ArrayBackend("test")

    @array_operation(backend, "value")
    def first() -> int:
        return 1

    def second() -> int:
        return 2

    with pytest.raises(ValueError, match="already registered"):
        array_operation(backend, "value")(second)

    with pytest.raises(UnsupportedArrayOperationError, match="missing"):
        backend.call("missing")


def test_numpy_dtype_conversion_uses_shared_types() -> None:
    context = resolve_array_context()
    assert context.normalize_dtype(np.int64) is DType.INT64
    assert context.normalize_dtype(np.dtype("float32")) is DType.FLOAT32
    assert context.native_dtype(DType.COMPLEX128) == np.dtype(np.complex128)


@pytest.mark.parametrize("value", [True, 1, 1.5, 1j, np.array(2.0)])
def test_scalar_creation_returns_zero_dimensional_native_value(value: object) -> None:
    result = scalar(value)

    assert isinstance(result, np.generic)
    assert result.ndim == 0
    assert result == value


@pytest.mark.parametrize("value", [[1, 2], np.array([1]), np.empty(0)])
def test_scalar_creation_rejects_arrays(value: object) -> None:
    with pytest.raises(ValueError, match="zero-dimensional"):
        scalar(value)


def test_unregistered_shared_dtype_is_rejected() -> None:
    backend = ArrayBackend("test")

    context = ArrayContext(Backend("test", backend))
    with pytest.raises(UnsupportedDTypeError):
        context.normalize_dtype(DType.FLOAT64)


def test_conflicting_dtype_mapping_is_rejected() -> None:
    arrays = ArrayBackend("test")
    arrays.register(_identity_dtype, name="normalize_native_dtype")

    with pytest.raises(ValueError, match="already mapped"):
        ArrayContext(
            Backend("test", arrays),
            native_dtypes={DType.INT64: "int64", DType.INT32: "int64"},
        )


def test_context_owns_immutable_copy_of_native_dtypes() -> None:
    arrays = ArrayBackend("test")
    arrays.register(_identity_dtype, name="normalize_native_dtype")
    native_dtypes: dict[DType, Hashable] = {DType.FLOAT64: "float64"}
    context = ArrayContext(Backend("test", arrays), native_dtypes=native_dtypes)

    native_dtypes[DType.FLOAT64] = "float32"
    assert context.native_dtype(DType.FLOAT64) == "float64"
    assert context.normalize_dtype("float64") is DType.FLOAT64
    with pytest.raises(TypeError):
        cast(dict[DType, Hashable], context.native_dtypes)[DType.FLOAT64] = "float32"


def test_inspection_uses_validator_context_dtype_mapping() -> None:
    context = ArrayContext(numpy_backend, native_dtypes={DType.FLOAT32: np.dtype(np.float32)})
    validator = ArrayValidator(context)

    assert validator.inspect(np.float32(1.0)).dtype is DType.FLOAT32
    with pytest.raises(UnsupportedDTypeError):
        validator.inspect(np.float64(1.0))


def test_missing_dtype_provider_does_not_initialize_or_freeze_backend() -> None:
    arrays = ArrayBackend("test")
    arrays.register(lambda device: device is Device.CPU, name="supports_device")
    backend = Backend("test", arrays)

    with pytest.raises(UnsupportedArrayOperationError, match="dtype_mappings"):
        initialize_backend(backend)

    _configure_dtypes(arrays, {DType.FLOAT64: "float64"})
    initialize_backend(backend)
    assert resolve_array_context().native_dtype(DType.FLOAT64) == "float64"


def test_numpy_operations_are_registered_and_use_shared_dtype() -> None:
    result = numpy_array_backend.call("zeros", 3, dtype=DType.INT16)

    assert result.dtype == np.dtype(np.int16)
    np.testing.assert_array_equal(result, np.zeros(3, dtype=np.int16))
    assert {"zeros", "exp", "reshape", "sum"}.issubset(numpy_array_backend.operation_names)


def test_process_backend_is_initialized_once() -> None:
    arrays = ArrayBackend("test")

    @array_operation(arrays, "supports_device")
    def supports_device(device: Device) -> bool:
        return device is Device.CPU

    _configure_dtypes(arrays, {DType.FLOAT64: "float64"})
    backend = Backend(identity="test", arrays=arrays)
    initialize_backend(backend, device=Device.CPU)

    assert resolve_backend() is backend
    assert resolve_array_backend() is arrays
    assert resolve_array_context().backend is backend
    assert resolve_array_context().device is Device.CPU
    assert resolve_array_context().default_dtype is DType.FLOAT64

    with pytest.raises(BackendAlreadyInitializedError):
        initialize_backend(numpy_backend)

    def late_operation() -> None:
        pass

    with pytest.raises(ArrayBackendFrozenError):
        array_operation(arrays)(late_operation)

    with pytest.raises(TypeError):
        cast(dict[DType, Hashable], resolve_array_context().native_dtypes)[DType.INT64] = "int64"


def test_initialization_rejects_unmapped_default_dtype() -> None:
    arrays = ArrayBackend("test")
    _configure_dtypes(arrays, {})
    arrays.register(lambda device: device is Device.CPU, name="supports_device")
    backend = Backend(identity="test", arrays=arrays)

    with pytest.raises(UnsupportedDTypeError):
        initialize_backend(backend)


def test_unsupported_device_does_not_freeze_or_initialize_backend() -> None:
    arrays = ArrayBackend("test")
    arrays.register(lambda device: False, name="supports_device")
    _configure_dtypes(arrays, {DType.FLOAT64: "float64"})
    backend = Backend("test", arrays)

    with pytest.raises(UnsupportedDeviceError, match="device 'cpu'"):
        initialize_backend(backend, device=Device.CPU)

    arrays.register(lambda: 1, name="value")
    initialize_backend(numpy_backend)
    assert resolve_array_context().device is Device.CPU
    assert resolve_backend() is numpy_backend


def test_string_device_does_not_initialize_backend() -> None:
    with pytest.raises(TypeError, match="Device enum"):
        initialize_backend(numpy_backend, device=cast(Device, "cpu"))

    result = zeros(2)
    assert normalize_value(result) is result
    assert resolve_array_context().device is Device.CPU


def test_nan_to_num_defaults_replace_nan_and_infinities() -> None:
    original = np.array([np.nan, np.inf, -np.inf, 1.0])
    value = asarray(original)
    np.testing.assert_array_equal(nan_to_num(value), np.nan_to_num(original))


def test_nan_to_num_respects_explicit_replacements() -> None:
    value = asarray([np.nan, np.inf, -np.inf])
    np.testing.assert_array_equal(
        nan_to_num(value, nan=-1.0, posinf=2.0, neginf=-2.0), [-1.0, 2.0, -2.0]
    )


def test_linspace_infers_complex_dtype_from_endpoints() -> None:
    result = linspace(1j, 1 + 2j, 3)
    np.testing.assert_array_equal(result, np.linspace(1j, 1 + 2j, 3))
    assert result.dtype == np.dtype(np.complex128)


def test_linspace_respects_explicit_dtype_and_real_default() -> None:
    assert linspace(0, 1, 3).dtype == np.dtype(np.float64)
    assert linspace(0, 1, 3, dtype=DType.FLOAT32).dtype == np.dtype(np.float32)


def test_array_function_uses_default_numpy_backend() -> None:
    result = asarray([1, 2], dtype=DType.FLOAT64)

    assert resolve_backend() is numpy_backend
    assert resolve_array_backend() is numpy_array_backend
    assert resolve_array_context().device is Device.CPU
    assert resolve_array_context().default_dtype is DType.FLOAT64
    assert result.dtype == np.dtype(np.float64)


def test_array_function_uses_initialized_backend() -> None:
    arrays = ArrayBackend("test")
    arrays.register(lambda device: device is Device.CPU, name="supports_device")
    _configure_dtypes(arrays, {DType.FLOAT64: "float64"})

    @array_operation(arrays, "add")
    def add_values(left: int, right: int) -> int:
        return left + right

    backend = Backend(identity="test", arrays=arrays)
    initialize_backend(backend)

    assert add(2, 3) == 5
    assert resolve_backend() is backend


def test_missing_device_operation_does_not_freeze_or_initialize_backend() -> None:
    arrays = ArrayBackend("test")
    _configure_dtypes(arrays, {DType.FLOAT64: "float64"})
    backend = Backend("test", arrays)

    with pytest.raises(UnsupportedArrayOperationError, match="supports_device"):
        initialize_backend(backend)

    arrays.register(lambda device: device is Device.CPU, name="supports_device")
    initialize_backend(backend)
    assert resolve_backend() is backend


def test_validator_checks_selected_device_through_registry() -> None:
    arrays = ArrayBackend("test")
    requested: list[Device] = []

    @array_operation(arrays, "supports_device")
    def supports_device(device: Device) -> bool:
        requested.append(device)
        return device is Device.CPU

    context = ArrayContext(Backend("test", arrays), device=Device.CPU)
    ArrayValidator(context).validate_device()

    assert requested == [Device.CPU]


def test_context_exposes_capabilities_and_native_type_promotion() -> None:
    context = resolve_array_context()

    assert DType.FLOAT64 in context.supported_dtypes
    assert context.supports({"add", "zeros"}, {DType.FLOAT32})
    assert context.result_type(DType.INT16, DType.FLOAT32) is DType.FLOAT32


def test_public_creation_returns_native_array() -> None:
    result = zeros(3)

    assert isinstance(result, NativeArray)
    assert result.shape == (3,)


def test_validator_inspects_and_normalizes_public_values() -> None:
    scalar_value = normalize_value(1.5)
    array_value = normalize_value([1, 2, 3])
    validator = ArrayValidator(resolve_array_context())

    scalar_profile = validator.validate(scalar_value)
    array_profile = validator.validate(
        array_value, ValueContract(allowed_dtypes=frozenset({DType.INT64}))
    )

    assert scalar_profile.kind is ValueKind.SCALAR
    assert scalar_profile.dtype is DType.FLOAT64
    assert scalar_profile.shape == ()
    assert array_profile.kind is ValueKind.ARRAY
    assert array_profile.shape == (3,)


def test_validator_rejects_tuple_foreign_value_and_disallowed_dtype() -> None:
    validator = ArrayValidator(resolve_array_context())

    with pytest.raises(ForeignArrayError, match="Tuple inputs"):
        normalize_value((1, 2))
    with pytest.raises(ForeignArrayError, match="cannot be normalized"):
        normalize_value(object())
    with pytest.raises(TypeError, match="not allowed"):
        validator.validate(
            asarray([1, 2]), ValueContract(allowed_dtypes=frozenset({DType.FLOAT64}))
        )


def test_validator_rejects_non_enum_device_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    context = resolve_array_context()
    validator = ArrayValidator(context)
    invalid_profile = ValueProfile(
        kind=ValueKind.SCALAR,
        dtype=DType.FLOAT64,
        device=cast(Device, "cpu"),
        shape=(),
        ndim=0,
        size=1,
    )
    monkeypatch.setattr(validator, "inspect", lambda value: invalid_profile)

    with pytest.raises(DeviceMismatchError, match="active device"):
        validator.validate(np.float64(1.0))
