"""
Characteristics API
===================

Lightweight wrapper for calling a distribution's characteristic (e.g., ``pdf``,
``cdf``, ``ppf``) resolved by the current computation strategy.

This module exposes a single generic helper, :class:`GenericCharacteristic`,
that delegates the actual computation to the distribution's
:class:`~pysatl_core.distributions.strategies.ComputationStrategy`.

Notes
-----
- The characteristic name controls *what* to compute (e.g., "pdf").
- ``**options`` control *how* to compute it (e.g., numeric parameters for
  fitters or disambiguation flags).
"""

__author__ = "Leonid Elkin, Mikhail, Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from pysatl_core.distributions.strategies import Method
from pysatl_core.types import (
    GenericCharacteristicName,
)

if TYPE_CHECKING:
    from pysatl_core.distributions.distribution import Distribution


@dataclass(slots=True, frozen=True)
class GenericCharacteristic[In, Out]:
    """
    Callable characteristic descriptor.

    Parameters
    ----------
    name : str
        Characteristic identifier (e.g., ``"pdf"``, ``"cdf"`` or ``"ppf"``).

    Notes
    -----
    This object does not implement the characteristic itself. It resolves and
    calls either an analytical function or a fitted method via the
    active :class:`~pysatl_core.distributions.strategies.ComputationStrategy`.

    Examples
    --------
    >>> from pysatl_core.distributions.characteristics import GenericCharacteristic
    >>> PDF = GenericCharacteristic[float, float]("pdf")
    >>> # Later:
    >>> # value = PDF(dist, 0.0)  # resolves dist's pdf(0.0)
    """

    name: GenericCharacteristicName

    # NOTICE: options контролирует математическую формулу характеристики
    def __call__(self, distribution: "Distribution", data: In, **options: Any) -> Out:
        """
        Evaluate the characteristic on the given data.

        Parameters
        ----------
        distribution : Distribution
            Distribution instance providing computation and sampling strategies.
        data : Any
            Input value for the characteristic (scalar for univariate case).
        **options
            Strategy- and fitter-specific options.

        Returns
        -------
        Any
            Characteristic value at ``data``.
        """
        method = cast(
            Method[In, Out],
            distribution.computation_strategy.query_method(self.name, distribution, **options),
        )
        return method(data)
