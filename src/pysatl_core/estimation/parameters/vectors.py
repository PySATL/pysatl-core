"""
Parameters as a flat vector, and back.

An optimizer works on a plain array; a family speaks in named parameters with
constraints attached.  Translating between the two is the whole of this module,
and it is the one place in the package where a flat vector exists at all.

Nothing here knows what is being estimated.  Every numerical method needs these
four functions, which is why they sit below the methods rather than inside the
one that happened to need them first.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import fields
from typing import TYPE_CHECKING

import numpy as np

from pysatl_core.families.parametrizations import Parametrization

if TYPE_CHECKING:
    from numpy.typing import NDArray


def field_names(params_or_class: Parametrization | type[Parametrization]) -> tuple[str, ...]:
    """
    List a parametrization's fields, in declaration order.

    ``Parametrization`` is an ABC that the ``@parametrization`` decorator turns
    into a dataclass, and views get a class synthesised at runtime.  That
    promise is now declared on the base class itself (``Parametrization.
    __dataclass_fields__``), so the fields are read through
    ``dataclasses.fields`` rather than probed by name.

    Parameters
    ----------
    params_or_class : Parametrization or type[Parametrization]
        A parametrization instance or class.

    Returns
    -------
    tuple[str, ...]
        Field names in declaration order. For a view this is the free
        parameters only.

    Raises
    ------
    TypeError
        If the argument is not a parametrization that the decorator has turned
        into a dataclass.  Previously such an object silently yielded an empty
        tuple, which reads downstream as "a family with no free parameters" —
        a different situation entirely.
    """
    return tuple(f.name for f in fields(params_or_class))


def to_vector(params: Parametrization) -> NDArray[np.float64]:
    """
    Flatten a parametrization into the vector the optimizer works on.

    Parameters
    ----------
    params : Parametrization
        Parameters to flatten.

    Returns
    -------
    NDArray[np.float64]
        Values ordered by ``params.__dataclass_fields__``, that is, by
        declaration order of the parametrization's fields.
    """
    values = params.parameters
    return np.array([values[name] for name in field_names(params)], dtype=np.float64)


def from_vector[P: Parametrization](param_cls: type[P], vec: NDArray[np.float64]) -> P:
    """
    Rebuild a parametrization from a flat vector.

    Parameters
    ----------
    param_cls : type[P]
        Parametrization class to instantiate.  For a view this is the
        lightweight class holding only the free parameters.
    vec : NDArray[np.float64]
        Values in the order of ``param_cls.__dataclass_fields__``.

    Returns
    -------
    P
        Instance of exactly the class that was passed in.  It is *not*
        validated: rejecting inadmissible parameters is the objective
        function's job, and it does so with a value rather than an exception.
    """
    names = field_names(param_cls)
    return param_cls(**{name: float(value) for name, value in zip(names, vec, strict=True)})


def satisfies_constraints(params: Parametrization) -> bool:
    """
    Check a parametrization against its own ``@constraint`` predicates.

    The predicates are evaluated directly rather than through
    ``Parametrization.validate`` (which raises) or ``family.distribution``
    (which raises *and* costs about fifteen times more): inside an optimisation
    loop the answer has to be a value.

    Parameters
    ----------
    params : Parametrization
        Candidate parameters.

    Returns
    -------
    bool
        ``True`` if every constraint holds.  A predicate that raises — which
        happens when the optimizer probes values a family's constraint was
        never written to handle, such as ``nan`` — counts as not holding.
    """
    for constraint in params.constraints:
        try:
            if not constraint.check(params):
                return False
        except (ValueError, ArithmeticError, TypeError):
            return False
    return True


__all__ = [
    "field_names",
    "from_vector",
    "satisfies_constraints",
    "to_vector",
]
