"""Validation and normalization of native backend values."""

from __future__ import annotations

from collections.abc import Iterable

from pysatl_core.arrays.backend import UnsupportedDeviceError
from pysatl_core.arrays.context import ArrayContext
from pysatl_core.arrays.creation import asarray, scalar
from pysatl_core.arrays.profiles import ValueContract, ValueProfile
from pysatl_core.arrays.state import resolve_array_context


class ForeignArrayError(TypeError):
    """Raised when a value does not belong to the active backend."""


class DeviceMismatchError(ValueError):
    """Raised when a native value is stored on another device."""


class ArrayValidator:
    """Inspect and validate native values of one process context."""

    def __init__(self, context: ArrayContext) -> None:
        self._context = context

    def missing_operations(self, required: Iterable[str]) -> frozenset[str]:
        """Return required operations absent from the active backend."""
        return self._context.arrays.missing_operations(required)

    def validate_device(self) -> None:
        """Check that the backend supports the device selected in this context."""
        if not self._context.arrays.call("supports_device", self._context.device):
            raise UnsupportedDeviceError(
                f"Array backend '{self._context.backend.identity}' does not support "
                f"device '{self._context.device}'."
            )

    def is_native(self, value: object) -> bool:
        """Return whether a value belongs to the active array backend."""
        return bool(self._context.arrays.call("is_native", value))

    def inspect(self, value: object) -> ValueProfile:
        """Build a backend-independent profile for a native value."""
        if not self.is_native(value):
            raise ForeignArrayError(
                f"Value of type {type(value).__name__!r} does not belong to "
                f"array backend '{self._context.backend.identity}'."
            )

        profile = self._context.arrays.call("inspect", value, context=self._context)
        if not isinstance(profile, ValueProfile):
            raise TypeError("Array backend returned an invalid value profile.")

        return profile

    def validate(
        self,
        value: object,
        contract: ValueContract | None = None,
    ) -> ValueProfile:
        """Validate a native value and return its profile."""
        profile = self.inspect(value)

        if profile.device is not self._context.device:
            raise DeviceMismatchError(
                f"Value is on device '{profile.device}', active device is '{self._context.device}'."
            )

        self._context.native_dtype(profile.dtype)

        if contract is None:
            return profile

        if contract.allowed_dtypes is not None and profile.dtype not in contract.allowed_dtypes:
            raise TypeError(f"Data type {profile.dtype!r} is not allowed by the value contract.")

        return profile


def resolve_array_validator() -> ArrayValidator:
    """Create a validator for the active process context."""
    return ArrayValidator(resolve_array_context())


def normalize_value(value: object) -> object:
    """Normalize one public input to a native scalar or array."""
    validator = resolve_array_validator()

    if validator.is_native(value):
        validator.validate(value)
        return value

    if isinstance(value, tuple):
        raise ForeignArrayError("Tuple inputs are not supported; pass a list or native array.")

    if isinstance(value, list):
        normalized = asarray(value)
    elif isinstance(value, bool | int | float | complex):
        normalized = scalar(value)
    else:
        raise ForeignArrayError(
            f"Value of type {type(value).__name__!r} cannot be normalized by the active backend."
        )

    validator.validate(normalized)
    return normalized


__all__ = [
    "ArrayValidator",
    "DeviceMismatchError",
    "ForeignArrayError",
    "normalize_value",
    "resolve_array_validator",
]
