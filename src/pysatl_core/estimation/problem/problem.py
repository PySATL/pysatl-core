"""
One family, one sample, and everything every method derives from the pair.

:class:`FitProblem` is the object the steps in the neighbouring modules hang
off.  It is what an estimation method builds first and then asks, instead of
calling four functions in the right order and carrying the results around by
hand.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from pysatl_core.estimation._attribution import warn_at_caller
from pysatl_core.estimation.errors import EstimationError
from pysatl_core.estimation.parameters.bounds import ParameterBox
from pysatl_core.estimation.parameters.start import project_onto_base, starting_point
from pysatl_core.estimation.parameters.vectors import field_names
from pysatl_core.estimation.problem.support import SupportInfo, cached_support_info
from pysatl_core.estimation.problem.validation import check_resolved_fixed_support, validate_sample

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from numpy.typing import NDArray

    from pysatl_core.families.parametric_family import ParametricFamily
    from pysatl_core.families.parametrizations import Parametrization
    from pysatl_core.types import ParametrizationName


@dataclass(frozen=True, eq=False)
class FitProblem:
    """
    One family, one sample, and everything every method derives from the pair.

    The steps below — validating the sample, finding a plausible point of the
    parameter space, deciding whether the support moves, resolving the bounds —
    are what *any* estimation method needs
    before it can do anything of its own.  Written as free functions they had
    to be called in the right order by every method, and the results carried
    around by hand; here they are named fields of one object.

    Everything but the sample is computed **on first access and never again**.
    Laziness is not an optimisation detail: a method of moments has no use for
    :attr:`support_moves`, and for it :attr:`probe` *is* the answer rather than
    a starting point, so computing both up front would do work half the methods
    throw away.

    Building the object is not obligatory: an estimator takes a family and a
    sample, as :class:`~pysatl_core.estimation.estimator.Estimator` says, and
    reaches for this only because it is convenient.  Making the protocol take a
    ``FitProblem`` would have made the preparation part of the contract, and
    with it the assumption that every method wants the same preparation.

    Not a ``slots`` dataclass, unlike everything else here: ``cached_property``
    stores what it computed in the instance dictionary, which ``slots`` removes.
    Not comparable either — ``eq=False`` — because a field holding a NumPy array
    has no sensible ``==``.

    Attributes
    ----------
    family : ParametricFamily
        Family being fitted.
    sample : NDArray[np.float64]
        The validated sample: 1-D, finite, long enough.
    """

    family: ParametricFamily
    sample: NDArray[np.float64]

    @classmethod
    def prepare(cls, family: ParametricFamily, sample: npt.ArrayLike) -> FitProblem:
        """
        Validate *sample* against *family* and hold the two together.

        This is the only eager step, and it is eager because everything else
        would be meaningless on a sample that cannot carry a fit at all.

        Raises
        ------
        ValueError
            If the sample is not 1-D or not finite.
        InsufficientDataError
            If it holds fewer observations than there are free parameters.
        """
        return cls(family=family, sample=validate_sample(family, sample))

    @cached_property
    def probe(self) -> Parametrization:
        """A plausible point of the parameter space, from the registered moment rule."""
        return probe_params(self.family, self.sample)

    @cached_property
    def support_info(self) -> SupportInfo:
        """Shared classification and fixed support, resolved once per family."""
        return cached_support_info(self.family, lambda: self.probe)

    @cached_property
    def support_moves(self) -> bool:
        """Whether the family's support depends on its parameters, shared across fits."""
        return self.support_info.moves

    @cached_property
    def box(self) -> ParameterBox:
        """The region of the parameter space a search may occupy."""
        return ParameterBox.of(self.family)

    def reject_data_outside_a_fixed_support(self) -> None:
        """
        Stop the fit if the data lie outside a support no parameter can move.

        A no-op when the support moves with the parameters: such a point is
        then a symptom of the current iterate rather than a fact about the
        data, and the objective charges it a penalty instead.

        Raises
        ------
        FitDataError
            If the support is fixed and some observation falls outside it.
        """
        info = self.support_info
        if info.moves:
            return
        check_resolved_fixed_support(self.family, self.sample, info.fixed_support)

    def report_in(
        self, estimated_params: Parametrization, parametrization: ParametrizationName | None
    ) -> Parametrization:
        """Express an estimate in the requested parametrization.

        Only the family's base parametrization is supported: converting to
        another one requires an inverse transform that the parametrization API
        does not provide.
        """
        base = self.family.base_parametrization_name
        if parametrization is None or parametrization == base:
            return estimated_params
        raise NotImplementedError(
            f"Cannot return the estimate in parametrization '{parametrization}': converting "
            f"from the base parametrization '{base}' requires an "
            f"inverse transform, and the 'Parametrization' API offers only "
            f"'transform_to_base_parametrization'. Fit in the base parametrization and convert "
            f"the values by hand, or add the inverse transform to the family's "
            f"parametrization class."
        )


def probe_params(
    family: ParametricFamily,
    sample: NDArray[np.float64],
) -> Parametrization:
    """
    A plausible point of the parameter space, obtained without raising.

    The caught exceptions are the ones a *moment rule* can legitimately produce
    on awkward data: a value it refuses (``ValueError``), arithmetic that does
    not work out (``ArithmeticError`` and its three subclasses), or a start
    that cannot be assembled at all (``EstimationError``).  Anything else — a
    ``TypeError``, a ``KeyError`` — is a bug in the rule rather than a verdict
    about the data, and is left to surface.

    Either way the substitution is announced: a rule that fails every time is a
    defect worth seeing, and swallowing it in silence is what would really blur
    the line between "the rule declined" and "the rule is broken".

    Parameters
    ----------
    family : ParametricFamily
        Family being fitted.
    sample : NDArray[np.float64]
        The validated sample.
    """
    try:
        return starting_point(family, sample)
    except (EstimationError, ValueError, ArithmeticError) as exc:
        ones = project_onto_base(family, dict.fromkeys(field_names(family.base), 1.0))
        if ones is None:  # pragma: no cover - ones covers every field by construction
            raise
        warn_at_caller(
            f"the method-of-moments starting rule for family '{family.name}' failed "
            f"({type(exc).__name__}: {exc}); starting from a default probe instead. The fit "
            f"continues, but it starts further from the answer than it needs to."
        )

        return ParameterBox.of(family).clip(ones)


__all__ = ["FitProblem", "probe_params"]
