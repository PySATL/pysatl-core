"""
Computation strategy specialised for :class:`EmpiricalDistribution`.

Adds estimator-identity tracking on top of :class:`DefaultComputationStrategy`:
fitted methods derived from a previous underlying estimator (e.g. KDE fit on
the sample) are dropped automatically when the empirical method is swapped,
so the strategy never returns a CDF/PPF derived from a stale PDF.

The swap is detected through the distribution's public ``estimator``
property, which
:class:`~pysatl_core.distributions.empirical.distribution.EmpiricalDistribution`
documents as an identity-comparable handle on the current fit.  Any
distribution willing to expose the same property gets the same invalidation
for free; distributions without it are served as plain Default.

The strategy does **not** override characteristic resolution.  The fast,
vectorised PPF used by empirical distributions is a regular ``cdf -> ppf``
edge in the characteristic graph (``cdf_to_ppf_tabulated_1C``), selected by
the graph itself because :class:`EmpiricalDistribution` declares a
``tabulation_domain``.  Keeping it in the graph is what makes
``explain_computation_path`` and ``query_method`` agree and lets the standard
option machinery reach the fitter.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, Any

from pysatl_core.distributions.strategies import DefaultComputationStrategy

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pysatl_core.distributions.computations.options import StepOptions
    from pysatl_core.distributions.distribution import Distribution
    from pysatl_core.distributions.strategies import ComputationPlan
    from pysatl_core.types import GenericCharacteristicName, Method


class EmpiricalComputationStrategy(DefaultComputationStrategy):
    """
    Computation strategy used by :class:`EmpiricalDistribution`.

    Behaves like :class:`DefaultComputationStrategy`, except the fitted-method
    cache is implicitly keyed by the identity of the distribution's underlying
    estimator, read from its public ``estimator`` property.  Whenever the
    strategy notices that the estimator has changed since the last query, it
    clears the cache so that any fitted CDF/PPF/etc. previously derived from
    the old estimator is rebuilt against the new one.

    This means callers do **not** need to call an explicit ``invalidate()``
    after swapping the empirical method on the distribution — the strategy
    detects the swap on the next characteristic query.

    The ``estimator`` property is a duck-typed hook, matching how the graph
    reads ``tabulation_domain``: a distribution that does not expose one is
    treated as having no estimator, and the strategy degenerates to plain
    Default behaviour.

    Parameters
    ----------
    enable_caching : bool, default True
        Whether to cache fitted computation methods.  Defaults to ``True``
        here (unlike the base strategy): deriving a CDF or PPF from a kernel
        estimator is expensive, and the estimator-identity tracking above
        already guarantees stale fits are dropped.
    computation_defaults : Mapping[str, Any] | None, default None
        Strategy-level defaults for computation options, forwarded verbatim
        to :class:`DefaultComputationStrategy`.
    """

    def __init__(
        self,
        enable_caching: bool = True,
        computation_defaults: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            enable_caching=enable_caching,
            computation_defaults=computation_defaults,
        )
        # We hold a strong reference to the last seen estimator. Two reasons:
        # (1) `is`-comparison is unambiguous (id() can be recycled by GC);
        # (2) the small memory cost of keeping one extra reference is
        # negligible compared to the cost of silently caching a stale fit.
        self._tracked_estimator: object | None = None

    def _maybe_invalidate(self, distr: Distribution) -> None:
        """
        Drop the cache if the distribution's estimator changed since last query.

        Distributions that expose no ``estimator`` property (or whose estimator
        is ``None``) are treated as "no estimator"; the cache is reset only on
        transitions between distinct estimator objects, not on every query.
        """
        current = getattr(distr, "estimator", None)
        if current is not self._tracked_estimator:
            self._cache.clear()
            self._tracked_estimator = current

    def invalidate(self) -> None:
        """Drop every cached fit and forget the tracked estimator."""
        self._cache.clear()
        self._tracked_estimator = None

    def explain_computation_path(
        self, state: GenericCharacteristicName, distr: Distribution
    ) -> ComputationPlan:
        self._maybe_invalidate(distr)
        return super().explain_computation_path(state, distr)

    def query_method(
        self,
        state: GenericCharacteristicName,
        distr: Distribution,
        options: StepOptions | None = None,
        *,
        characteristic_options: Mapping[str, Any] | None = None,
        computation_defaults: Mapping[str, Any] | None = None,
    ) -> Method[Any, Any]:
        self._maybe_invalidate(distr)
        return super().query_method(
            state,
            distr,
            options,
            characteristic_options=characteristic_options,
            computation_defaults=computation_defaults,
        )


__all__ = ["EmpiricalComputationStrategy"]
