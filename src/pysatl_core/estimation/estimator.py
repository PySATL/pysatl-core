"""
What an estimation method is, expressed as a type.

An estimator is an object, not a string.  It carries its own configuration in
its own declared fields — ``MLE`` carries an optimizer and the options for it,
a method of moments would carry how many moments to match — and it answers one
question, :meth:`Estimator.estimate`.

Why an object.  The alternative, ``fit(sample, method="mle", **options)``,
cannot be typed honestly: the options of different methods have nothing in
common, so ``**options`` degrades to ``Any`` or the signature grows one
``@overload`` per method.  It is also closed: every new method means editing
``ParametricFamily.fit``, which puts methods defined elsewhere — in another
PySATL package, or by a user — out of reach.  With an object, the result type
follows from the estimator's own type, an option that does not apply to a
method is a type error where it is written, and a new method is a new class
that this package never has to hear about.

Because an estimator is a value, several of them go in a list and a comparison
across methods is a loop::

    for estimator in (MLE(), MLE(optimizer="Powell")):
        print(estimator.name, estimator.estimate(family, sample).params)

Configuration is separate from data on purpose: an estimator is built once and
applied to many samples or many families.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pysatl_core.estimation.result import FitResult
from pysatl_core.families.parametrizations import Parametrization

if TYPE_CHECKING:
    import numpy.typing as npt

    from pysatl_core.families.parametric_family import ParametricFamily
    from pysatl_core.types import EstimatorName, ParametrizationName


type AnyFitResult = FitResult[Parametrization]
"""The widest result an estimator may return.

``Parametrization`` rather than a concrete class because a family is bound to
its parametrization classes at runtime, by the ``@parametrization`` decorator,
which runs after the family object exists — so no static type can name the
class a given family will register.  It is what the bound of
:class:`Estimator` is written against.
"""


@runtime_checkable
class Estimator[R: AnyFitResult](Protocol):
    """
    A parameter estimation method.

    Generic in the result it produces, which is what makes the return type of
    a fit precise without a single ``@overload``: ``MLE`` is an
    ``Estimator[MLEResult[Parametrization]]``, so ``family.fit(sample,
    estimator=MLE())`` is statically an ``MLEResult`` and
    ``result.log_likelihood`` checks, while a method that reports no
    likelihood returns its own result type and the same attribute is an error.

    A ``Protocol`` rather than a base class: an estimator needs nothing from
    this package to *be* one, and the steps it does share with the others —
    validating the sample, probing the parameter space, checking a fixed
    support, converting the estimate to the requested parametrization — are
    free functions in :mod:`pysatl_core.estimation.problem.problem`, which it calls rather
    than inherits.  Composition leaves an implementation free to skip a step it
    does not need; inheritance would have made the order of the steps part of
    the contract.

    It is ``runtime_checkable`` so that
    :meth:`~pysatl_core.families.parametric_family.ParametricFamily.fit` can
    refuse a non-estimator with a message naming what is missing, instead of
    failing later with an ``AttributeError`` from inside the call.
    """

    @property
    def name(self) -> EstimatorName:
        """
        Short name of the method, recorded in :attr:`FitResult.estimator`.

        A property rather than a field so that an implementation which is a
        frozen dataclass — the usual shape — does not have to accept its own
        name as a constructor argument.
        """
        ...

    def estimate(
        self,
        family: ParametricFamily,
        sample: npt.ArrayLike,
        *,
        parametrization: ParametrizationName | None = None,
    ) -> R:
        """
        Estimate the parameters of *family* from *sample*.

        The sample arrives raw: an estimator validates it itself, through
        :func:`~pysatl_core.estimation.problem.problem.validate_sample`, so that
        ``estimator.estimate(family, data)`` is a complete entry point and not
        only something ``ParametricFamily.fit`` may call.

        Estimation is an operation on a *family*, never on a distribution: a
        distribution already has its parameters pinned, so there is nothing in
        it left to estimate.

        Parameters
        ----------
        family : ParametricFamily
            Family to fit.  A view produced by
            :meth:`~pysatl_core.families.parametric_family.ParametricFamily.view`
            works unchanged: it *is* a family, whose base parametrization holds
            only the free parameters.
        sample : array_like
            Observed values.
        parametrization : ParametrizationName or None, optional
            Parametrization the estimate should be reported in.  ``None`` means
            the family's base parametrization.

        Returns
        -------
        R
            The estimate and whatever diagnostics this method reports.
        """
        ...


__all__ = ["AnyFitResult", "Estimator"]
