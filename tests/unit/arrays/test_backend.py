"""Tests for array backend registration and NumPy dtype mapping."""

from collections.abc import Hashable, Iterator

import numpy as np
import pytest

from pysatl_core.arrays import (
    ArrayBackend,
    ArrayBackendFrozenError,
    ArrayShapeError,
    ArrayValidator,
    Backend,
    BackendAlreadyInitializedError,
    DeviceMismatchError,
    DType,
    ForeignArrayError,
    NativeArray,
    UnsupportedArrayOperationError,
    UnsupportedDTypeError,
    ValueContract,
    ValueKind,
    add,
    array_operation,
    asarray,
    initialize_backend,
    normalize_value,
    resolve_array_backend,
    resolve_array_context,
    resolve_backend,
    zeros,
)
from pysatl_core.arrays.numpy import numpy_array_backend, numpy_backend
from pysatl_core.arrays.state import _reset_backend


def _identity_dtype(value: object) -> Hashable:
    if not isinstance(value, Hashable):
        raise TypeError
    return value


@pytest.fixture(autouse=True)
def reset_backend() -> Iterator[None]:
    _reset_backend()
    try:
        yield
    finally:
        _reset_backend()


def test_decorator_registers_operation() -> None:
    backend = ArrayBackend("test", dtype_normalizer=_identity_dtype)

    @array_operation(backend)
    def add(left: int, right: int) -> int:
        return left + right

    assert backend.operation_names == frozenset({"add"})
    assert backend.call("add", 2, 3) == 5
    assert add(2, 3) == 5


def test_duplicate_and_missing_operations_fail_explicitly() -> None:
    backend = ArrayBackend("test", dtype_normalizer=_identity_dtype)

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
    assert numpy_array_backend.normalize_dtype(np.int64) is DType.INT64
    assert numpy_array_backend.normalize_dtype(np.dtype("float32")) is DType.FLOAT32
    assert numpy_array_backend.native_dtype(DType.COMPLEX128) == np.dtype(np.complex128)


def test_unregistered_shared_dtype_is_rejected() -> None:
    backend = ArrayBackend("test", dtype_normalizer=_identity_dtype)

    with pytest.raises(UnsupportedDTypeError):
        backend.normalize_dtype(DType.FLOAT64)


def test_conflicting_dtype_mapping_is_rejected() -> None:
    backend = ArrayBackend("test", dtype_normalizer=_identity_dtype)
    backend.register_dtype(DType.INT64, "int64")

    with pytest.raises(ValueError, match="already mapped"):
        backend.register_dtype(DType.INT64, "long")
    with pytest.raises(ValueError, match="already mapped"):
        backend.register_dtype(DType.INT32, "int64")


def test_numpy_operations_are_registered_and_use_shared_dtype() -> None:
    result = numpy_array_backend.call("zeros", 3, dtype=DType.INT16)

    assert result.dtype == np.dtype(np.int16)
    np.testing.assert_array_equal(result, np.zeros(3, dtype=np.int16))
    assert {"zeros", "exp", "reshape", "sum"}.issubset(numpy_array_backend.operation_names)


def test_process_backend_is_initialized_once() -> None:
    arrays = ArrayBackend("test", dtype_normalizer=_identity_dtype)
    arrays.register_dtype(DType.FLOAT64, "float64")
    backend = Backend(identity="test", arrays=arrays)
    initialize_backend(backend, device="cuda")

    assert resolve_backend() is backend
    assert resolve_array_backend() is arrays
    assert resolve_array_context().backend is backend
    assert resolve_array_context().device == "cuda"
    assert resolve_array_context().default_dtype is DType.FLOAT64

    with pytest.raises(BackendAlreadyInitializedError):
        initialize_backend(numpy_backend)

    def late_operation() -> None:
        pass

    with pytest.raises(ArrayBackendFrozenError):
        array_operation(arrays)(late_operation)

    with pytest.raises(ArrayBackendFrozenError):
        arrays.register_dtype(DType.INT64, "int64")


def test_initialization_rejects_unmapped_default_dtype() -> None:
    arrays = ArrayBackend("test", dtype_normalizer=_identity_dtype)
    backend = Backend(identity="test", arrays=arrays)

    with pytest.raises(UnsupportedDTypeError):
        initialize_backend(backend)


def test_array_function_uses_default_numpy_backend() -> None:
    result = asarray([1, 2], dtype=DType.FLOAT64)

    assert resolve_backend() is numpy_backend
    assert resolve_array_backend() is numpy_array_backend
    assert resolve_array_context().device == "cpu"
    assert resolve_array_context().default_dtype is DType.FLOAT64
    assert result.dtype == np.dtype(np.float64)


def test_array_function_uses_initialized_backend() -> None:
    arrays = ArrayBackend("test", dtype_normalizer=_identity_dtype)
    arrays.register_dtype(DType.FLOAT64, "float64")

    @array_operation(arrays, "add")
    def add_values(left: int, right: int) -> int:
        return left + right

    backend = Backend(identity="test", arrays=arrays)
    initialize_backend(backend)

    assert add(2, 3) == 5
    assert resolve_backend() is backend


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
    array_profile = validator.validate(array_value, ValueContract(min_ndim=1, max_ndim=1))

    assert scalar_profile.kind is ValueKind.SCALAR
    assert scalar_profile.dtype is DType.FLOAT64
    assert scalar_profile.shape == ()
    assert array_profile.kind is ValueKind.ARRAY
    assert array_profile.shape == (3,)


def test_validator_rejects_tuple_foreign_value_and_invalid_shape() -> None:
    validator = ArrayValidator(resolve_array_context())

    with pytest.raises(ForeignArrayError, match="Tuple inputs"):
        normalize_value((1, 2))
    with pytest.raises(ForeignArrayError, match="cannot be normalized"):
        normalize_value(object())
    with pytest.raises(ArrayShapeError, match="at least 2"):
        validator.validate(asarray([1, 2]), ValueContract(min_ndim=2))


def test_validator_rejects_value_from_another_device() -> None:
    context = resolve_array_context()
    cuda_context = type(context)(backend=context.backend, device="cuda")
    validator = ArrayValidator(cuda_context)

    with pytest.raises(DeviceMismatchError, match="active device is 'cuda'"):
        validator.validate(asarray([1, 2]))
