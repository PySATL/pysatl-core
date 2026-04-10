from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov, Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from mypy_extensions import KwArg

    from pysatl_core.distributions.computation import (
        ComputationMethod,
        FittedComputationMethod,
    )
    from pysatl_core.distributions.distribution import Distribution
    from pysatl_core.types import GenericCharacteristicName, NumericArray

    type FitterFunc = Callable[
        [Distribution, KwArg(Any)],
        FittedComputationMethod[NumericArray, NumericArray],
    ]


@dataclass(frozen=True, slots=True)
class FitterOption:
    """
    Structured descriptor for a single fitter option.

    Attributes
    ----------
    name : str
        Option name as it appears in ``**kwargs``.
    type : type
        Expected Python type (``int``, ``float``, …).
    default : Any
        Default value used when the caller does not supply the option.
    description : str
        Human-readable description shown in documentation / introspection.
    validate : Callable[[Any], bool] | None
        Optional predicate.  When not ``None``, the option value is rejected
        (``ValueError``) if ``validate(value)`` returns ``False``.
    """

    name: str
    type: type
    default: Any
    description: str = ""
    validate: Callable[[Any], bool] | None = None


    def resolve(self, kwargs: dict[str, Any]) -> Any:
        """
        Extract and validate the option from *kwargs*.

        Parameters
        ----------
        kwargs : dict[str, Any]
            Caller-supplied keyword arguments.  The key matching
            :pyattr:`name` is consumed (popped) if present.

        Returns
        -------
        Any
            Resolved value (caller-supplied or default), cast to
            :pyattr:`type`.

        Raises
        ------
        ValueError
            If the resolved value fails the :pyattr:`validate` predicate.
        TypeError
            If the value cannot be cast to :pyattr:`type`.
        """
        raw = kwargs.pop(self.name, self.default)
        try:
            value = self.type(raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Option '{self.name}': cannot convert {raw!r} to {self.type.__name__}"
            ) from exc
        if self.validate is not None and not self.validate(value):
            raise ValueError(
                f"Option '{self.name}': value {value!r} failed validation."
            )
        return value


@dataclass(frozen=True, slots=True)
class FitterDescriptor:
    """
    Complete metadata for a single fitter or evaluator.

    Parameters
    ----------
    name : str
        Unique human-readable identifier (e.g. ``"pdf_to_cdf_1C"``).
    target : GenericCharacteristicName
        Characteristic produced by this fitter.
    sources : Sequence[GenericCharacteristicName]
        Characteristics consumed by this fitter (typically length 1).
    fitter : FitterFunc
        The actual fitting / evaluating callable.
    cacheable : bool
        ``True`` (default) — the callable is a **fitter** (``fit_`` prefix).
        It performs expensive precomputation and returns a
        ``FittedComputationMethod``.  The strategy layer caches the result.

        ``False`` — the callable is a **direct evaluator** (``eval_`` prefix).
        It is lightweight and called on every query without caching.

        This flag determines which slot (``fitter`` vs ``evaluator``) is
        populated when :meth:`to_computation_method` builds a
        :class:`ComputationMethod`.
    options : tuple[FitterOption, ...]
        Structured option schema.
    tags : frozenset[str]
        Constraint tags used for matching (e.g. ``{"continuous", "univariate"}``).
    priority : int
        Higher priority wins when multiple fitters match.  Default is 0.
    description : str
        Human-readable summary of what the fitter does.
    """

    name: str
    target: GenericCharacteristicName
    sources: Sequence[GenericCharacteristicName]
    fitter: FitterFunc
    cacheable: bool = True
    options: tuple[FitterOption, ...] = ()
    tags: frozenset[str] = field(default_factory=frozenset)
    priority: int = 0
    description: str = ""


    def to_computation_method(self) -> ComputationMethod[NumericArray, NumericArray]:
        """
        Build a :class:`ComputationMethod` from this descriptor.

        When :pyattr:`cacheable` is ``True``, the callable is placed in the
        ``fitter`` slot of :class:`ComputationMethod` — the strategy layer
        will call it once and cache the resulting ``FittedComputationMethod``.

        When :pyattr:`cacheable` is ``False``, the callable is placed in the
        ``evaluator`` slot — the strategy layer calls it directly on every
        query without caching.

        Returns
        -------
        ComputationMethod[NumericArray, NumericArray]
        """
        from pysatl_core.distributions.computation import ComputationMethod

        if self.cacheable:
            return ComputationMethod(
                target=self.target,
                sources=list(self.sources),
                fitter=self.fitter,
            )
        return ComputationMethod(
            target=self.target,
            sources=list(self.sources),
            evaluator=self.fitter,  # type: ignore[arg-type]
        )

    def resolve_options(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Resolve all declared options from *kwargs*.

        Consumes recognised keys from *kwargs* and returns a dict of
        ``{option_name: resolved_value}``.  Unrecognised keys are left
        in *kwargs* untouched.

        Parameters
        ----------
        kwargs : dict[str, Any]
            Mutable keyword-argument dict from the caller.

        Returns
        -------
        dict[str, Any]
            Mapping from option name to resolved (validated, typed) value.
        """
        return {opt.name: opt.resolve(kwargs) for opt in self.options}

    def option_names(self) -> tuple[str, ...]:
        """Return the names of all declared options."""
        return tuple(opt.name for opt in self.options)

    def option_defaults(self) -> dict[str, Any]:
        """Return ``{name: default}`` for every declared option."""
        return {opt.name: opt.default for opt in self.options}
