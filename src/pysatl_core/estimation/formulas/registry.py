"""Family-specific formulas used by parameter estimation methods.

The registry is a singleton, like the family registry. Keys are family objects,
not names, so unrelated families cannot share rules by accident. Views resolve
to their original family.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from numpy.typing import NDArray

from pysatl_core.families.registry import ParametricFamilyRegister

if TYPE_CHECKING:
    from pysatl_core.families.parametric_family import ParametricFamily
    from pysatl_core.types import EstimatorName


type MomentRule = Callable[[NDArray[np.float64]], Mapping[str, float]]


@dataclass(frozen=True, slots=True)
class AnalyticalEstimate:
    """Closed-form parameters and an optional, penalty-free log-likelihood.

    If supplied, the log-likelihood must describe these parameters and the
    entire input sample; MLE uses it directly without constructing a generic
    likelihood evaluator. If it is ``None``, MLE computes it from the family's
    log-density at the estimated parameters.
    """

    parameters: Mapping[str, float]
    log_likelihood: float | None = None


type AnalyticalFormula = Callable[
    [NDArray[np.float64], Mapping[str, float]], AnalyticalEstimate | None
]


def _original_family(family: ParametricFamily) -> ParametricFamily:
    from pysatl_core.families.parametric_family import PartialParametricFamily

    return family.parent_family if isinstance(family, PartialParametricFamily) else family


class EstimationFormulaRegistry:
    """Store analytical estimates and moment-based starting rules by family.

    A closed-form rule receives a validated sample and parameters fixed in the
    original family's base coordinates. It returns an :class:`AnalyticalEstimate`
    containing all base parameters and an optional log-likelihood, or ``None``
    when it does not cover the case. A moment rule returns named starting
    values. Registration is explicit and duplicate keys are rejected
    unless ``replace=True`` is requested. Every construction returns the same
    registry for the current family registry generation.
    """

    _instance: ClassVar[EstimationFormulaRegistry | None] = None
    _families: ClassVar[ParametricFamilyRegister | None] = None

    _closed_forms: dict[ParametricFamily, dict[EstimatorName, AnalyticalFormula]]
    _moment_starts: dict[ParametricFamily, MomentRule]

    def __new__(cls) -> EstimationFormulaRegistry:
        """Create a new singleton when the family registry has been reset."""
        families = ParametricFamilyRegister()
        if cls._instance is None or cls._families is not families:
            instance = super().__new__(cls)
            instance._closed_forms = {}
            instance._moment_starts = {}
            cls._instance = instance
            cls._families = families
        return cls._instance

    @classmethod
    def register_closed_form(
        cls,
        family: ParametricFamily,
        method: EstimatorName,
        formula: AnalyticalFormula,
        *,
        replace: bool = False,
    ) -> None:
        """Register one direct estimate for a family and estimation method."""
        self = cls()
        key = _original_family(family)
        formulas = self._closed_forms.setdefault(key, {})
        if method in formulas and not replace:
            raise ValueError(
                f"A closed-form {method!r} rule is already registered for {key.name!r}."
            )
        formulas[method] = formula

    @classmethod
    def closed_form_for(
        cls, family: ParametricFamily, method: EstimatorName
    ) -> AnalyticalFormula | None:
        """Look up a direct estimate; a missing formula is an ordinary result."""
        self = cls()
        return self._closed_forms.get(_original_family(family), {}).get(method)

    @classmethod
    def register_moment_start(
        cls, family: ParametricFamily, rule: MomentRule, *, replace: bool = False
    ) -> None:
        """Register the family's moment-based numerical starting rule."""
        self = cls()
        key = _original_family(family)
        if key in self._moment_starts and not replace:
            raise ValueError(f"A moment starting rule is already registered for {key.name!r}.")
        self._moment_starts[key] = rule

    @classmethod
    def moment_start_for(cls, family: ParametricFamily) -> MomentRule | None:
        """Look up a starting rule, including through a partial-family view."""
        self = cls()
        return self._moment_starts.get(_original_family(family))

    @classmethod
    def _reset(cls) -> None:
        """Clear formulas for tests; normally the family registry owns the lifecycle."""
        if cls._instance is not None:
            cls._instance._closed_forms.clear()
            cls._instance._moment_starts.clear()
        cls._instance = None
        cls._families = None


__all__ = [
    "AnalyticalEstimate",
    "AnalyticalFormula",
    "EstimationFormulaRegistry",
    "MomentRule",
]
