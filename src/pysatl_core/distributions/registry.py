from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, ClassVar, Self

from pysatl_core.distributions.computation import ComputationMethod
from pysatl_core.types import (
    Dimension,
    GenericCharacteristicName,
    Kind,
)


@dataclass
class GenericCharacteristicRegister:
    dimension: Dimension
    kind: Kind

    __registered_indefinitive_chars: dict[
        GenericCharacteristicName,
        list[ComputationMethod[Any, Any]],
    ] = field(default_factory=dict)

    __registered_definitive_chars: dict[
        GenericCharacteristicName, list[ComputationMethod[Any, Any]]
    ] = field(default_factory=dict)

    def register_indefinitive_characteristic(
        self,
        default_computation_method: ComputationMethod[Any, Any],
    ) -> None:
        self.__registered_indefinitive_chars[default_computation_method.target].append(
            default_computation_method
        )

    def register_definitive_characteristic(
        self,
        default_computation_method: ComputationMethod[Any, Any],
        default_inversion_method: ComputationMethod[Any, Any],
    ) -> None:
        if (
            default_computation_method.target not in default_inversion_method.sources
            or default_inversion_method.target not in default_computation_method.sources
        ):
            raise AttributeError(
                "New characteristic always must be in targets for default "
                "computation method and in sources for default inversion method"
            )
        self.__registered_definitive_chars[default_computation_method.target].append(
            default_computation_method
        )
        self.__registered_definitive_chars[default_inversion_method.target].append(
            default_inversion_method
        )

    def get_available_indefinitive_characteristics(
        self,
    ) -> dict[GenericCharacteristicName, list[ComputationMethod[Any, Any]]]:
        return self.__registered_indefinitive_chars

    def get_available_definitive_characteristics(
        self,
    ) -> dict[GenericCharacteristicName, list[ComputationMethod[Any, Any]]]:
        return self.__registered_definitive_chars

    def get_all_available_characteristics_keys(self) -> list[GenericCharacteristicName]:
        return list(self.__registered_definitive_chars.keys()) + list(
            self.__registered_indefinitive_chars.keys()
        )


class DistributionTypeRegister:
    _instance: ClassVar[Self | None] = None
    _register_kinds: dict[tuple[Dimension, Kind], GenericCharacteristicRegister]

    def __new__(cls) -> Self:
        if cls._instance is None:
            self = super().__new__(cls)
            self._register_kinds = {}
            cls._instance = self
        return cls._instance

    def get(self, dimension: Dimension, kind: Kind) -> GenericCharacteristicRegister:
        key = (dimension, kind)
        reg = self._register_kinds.get(key)
        if reg is None:
            reg = GenericCharacteristicRegister(dimension=dimension, kind=kind)
            self._register_kinds[key] = reg
        if reg.dimension != dimension or reg.kind != kind:
            raise TypeError(
                f"Inconsistent registry under key ({dimension}, {kind}): "
                f"got ({reg.dimension}, {reg.kind}) inside"
            )
        return reg

    __call__ = get


def _configure(reg: DistributionTypeRegister) -> None:
    reg.get(1, Kind.DISCRETE)
    reg.get(1, Kind.CONTINUOUS)

    # Тут много заполнений дефолтами


@lru_cache(maxsize=1)
def distribution_type_register() -> DistributionTypeRegister:
    reg = DistributionTypeRegister()
    _configure(reg)
    return reg


def _reset_distribution_type_register_for_tests() -> None:
    distribution_type_register.cache_clear()
