"""
Operator mixin for transformation-enabled distributions.

This mixin provides arithmetic operators implemented through
transformation primitives.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from numbers import Real
from types import NotImplementedType
from typing import TYPE_CHECKING, cast

from pysatl_core.types import BinaryOperationName

if TYPE_CHECKING:
    from pysatl_core.distributions.distribution import Distribution


class TransformationOperatorsMixin:
    """
    Mixin adding affine and binary arithmetic operators to distributions.
    """

    def _affine_transform(self, *, scale: float, shift: float) -> Distribution:
        """
        Apply affine transformation ``Y = scale * X + shift``.
        """
        from pysatl_core.distributions.distribution import Distribution
        from pysatl_core.transformations.operations.affine import affine

        return affine(cast(Distribution, self), scale=scale, shift=shift)

    def _binary_transform(
        self,
        other: Distribution,
        *,
        operation: BinaryOperationName,
    ) -> Distribution:
        """
        Apply binary transformation between two distributions.
        """
        from pysatl_core.distributions.distribution import Distribution
        from pysatl_core.transformations.operations.binary import binary

        return binary(cast(Distribution, self), other, operation=operation)

    def __add__(self, other: object) -> Distribution | NotImplementedType:
        """Return ``self + other`` for scalar or distribution operands."""
        from pysatl_core.distributions.distribution import Distribution

        if isinstance(other, Real):
            return self._affine_transform(scale=1.0, shift=float(other))
        if isinstance(other, Distribution):
            return self._binary_transform(other, operation=BinaryOperationName.ADD)
        return NotImplemented

    def __radd__(self, other: object) -> Distribution | NotImplementedType:
        """Return ``other + self`` for scalar or distribution operands."""
        from pysatl_core.distributions.distribution import Distribution
        from pysatl_core.transformations.operations.binary import binary

        if isinstance(other, Real):
            return self._affine_transform(scale=1.0, shift=float(other))
        if isinstance(other, Distribution):
            return binary(other, cast(Distribution, self), operation=BinaryOperationName.ADD)
        return NotImplemented

    def __sub__(self, other: object) -> Distribution | NotImplementedType:
        """Return ``self - other`` for scalar or distribution operands."""
        from pysatl_core.distributions.distribution import Distribution

        if isinstance(other, Real):
            return self._affine_transform(scale=1.0, shift=-float(other))
        if isinstance(other, Distribution):
            return self._binary_transform(other, operation=BinaryOperationName.SUB)
        return NotImplemented

    def __rsub__(self, other: object) -> Distribution | NotImplementedType:
        """Return ``other - self`` for scalar or distribution operands."""
        from pysatl_core.distributions.distribution import Distribution
        from pysatl_core.transformations.operations.binary import binary

        if isinstance(other, Real):
            return self._affine_transform(scale=-1.0, shift=float(other))
        if isinstance(other, Distribution):
            return binary(other, cast(Distribution, self), operation=BinaryOperationName.SUB)
        return NotImplemented

    def __mul__(self, other: object) -> Distribution | NotImplementedType:
        """Return ``self * other`` for scalar or distribution operands."""
        from pysatl_core.distributions.distribution import Distribution

        if isinstance(other, Real):
            return self._affine_transform(scale=float(other), shift=0.0)
        if isinstance(other, Distribution):
            return self._binary_transform(other, operation=BinaryOperationName.MUL)
        return NotImplemented

    def __rmul__(self, other: object) -> Distribution | NotImplementedType:
        """Return ``other * self`` for scalar or distribution operands."""
        from pysatl_core.distributions.distribution import Distribution
        from pysatl_core.transformations.operations.binary import binary

        if isinstance(other, Real):
            return self._affine_transform(scale=float(other), shift=0.0)
        if isinstance(other, Distribution):
            return binary(other, cast(Distribution, self), operation=BinaryOperationName.MUL)
        return NotImplemented

    def __truediv__(self, other: object) -> Distribution | NotImplementedType:
        """Return ``self / other`` for scalar or distribution operands."""
        from pysatl_core.distributions.distribution import Distribution

        if isinstance(other, Real):
            divisor = float(other)
            if divisor == 0.0:
                raise ZeroDivisionError("Cannot divide a distribution by zero.")
            return self._affine_transform(scale=1.0 / divisor, shift=0.0)
        if isinstance(other, Distribution):
            return self._binary_transform(other, operation=BinaryOperationName.DIV)
        return NotImplemented

    def __rtruediv__(self, other: object) -> Distribution | NotImplementedType:
        """Return ``other / self`` for distribution operands."""
        from pysatl_core.distributions.distribution import Distribution
        from pysatl_core.transformations.operations.binary import binary

        if isinstance(other, Distribution):
            return binary(other, cast(Distribution, self), operation=BinaryOperationName.DIV)
        return NotImplemented

    def __neg__(self) -> Distribution:
        """Return ``-self`` as an affine transformation."""
        return self._affine_transform(scale=-1.0, shift=0.0)


__all__ = [
    "TransformationOperatorsMixin",
]
