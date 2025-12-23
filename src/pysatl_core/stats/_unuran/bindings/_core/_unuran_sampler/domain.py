from __future__ import annotations

from typing import Any

import numpy as np

from pysatl_core.types import CharacteristicName


def determine_domain_from_support(sampler: Any) -> tuple[int, int | None] | None:
    """
    Determine domain boundaries from distribution support if available.

    Returns
    -------
    tuple[int, int | None] or None
        Domain as (left, right) if support is bounded, or (left, None) if only
        left boundary is known, or None if support is unavailable/unbounded.
    """
    support = sampler.distr.support
    if support is None:
        return None

    from pysatl_core.distributions.support import (
        ExplicitTableDiscreteSupport,
        IntegerLatticeDiscreteSupport,
    )

    if isinstance(support, ExplicitTableDiscreteSupport):
        points = support.points
        if points.size == 0:
            return None
        left = int(np.floor(points[0]))
        right = int(np.ceil(points[-1]))
        return (left, right)

    if isinstance(support, IntegerLatticeDiscreteSupport):
        first = support.first()
        last = support.last()

        if first is not None and last is not None:
            return (first, last)

        if first is not None:
            return (first, None)

        return None

    try:
        first = support.first()  # type: ignore[attr-defined]
        last = support.last()  # type: ignore[attr-defined]
        if first is not None and last is not None:
            return (int(first), int(last))
    except (AttributeError, TypeError):
        pass

    return None


def determine_domain_from_pmf(
    sampler: Any,
    domain_left: int | None = None,
) -> tuple[int, int]:
    """
    Determine domain boundaries by evaluating PMF until probability becomes negligible.

    This heuristic evaluates PMF starting from domain_left (or 0 if None)
    and finds the right boundary where cumulative probability exceeds threshold.

    Parameters
    ----------
    domain_left : int, optional
        Left boundary of domain. If None, starts from 0.

    Returns
    -------
    tuple[int, int]
        Domain as (left, right).
    """
    analytical_comps = sampler.distr.analytical_computations

    if CharacteristicName.PMF in analytical_comps:
        pmf_func = analytical_comps[CharacteristicName.PMF]
    elif CharacteristicName.PDF in analytical_comps:
        pmf_func = analytical_comps[CharacteristicName.PDF]
    else:
        raise RuntimeError("PMF or PDF is required for domain determination")

    if domain_left is None:
        start_k = 0
        for k in range(-10, 0):
            try:
                p = float(pmf_func(float(k)))
                if p > 1e-10:
                    start_k = k
                    break
            except (ValueError, TypeError):
                break
        domain_left = start_k
    else:
        domain_left = int(domain_left)

    cumulative_prob = 0.0
    threshold = 0.9999
    max_iterations = 10000
    domain_right = domain_left

    for k in range(domain_left, domain_left + max_iterations):
        try:
            p = float(pmf_func(float(k)))
            if p < 0 or np.isnan(p) or np.isinf(p):
                break
            cumulative_prob += p
            domain_right = k

            if cumulative_prob >= threshold:
                break
            if k > domain_left + 100 and p < 1e-10:
                break
        except (ValueError, TypeError):
            break

    return (domain_left, domain_right)


def calculate_pmf_sum(sampler: Any, domain_left: int, domain_right: int) -> float:
    """
    Calculate the sum of PMF over the specified domain.

    Parameters
    ----------
    domain_left : int
        Left boundary of domain.
    domain_right : int
        Right boundary of domain.

    Returns
    -------
    float
        Sum of PMF values over the domain.
    """
    analytical_comps = sampler.distr.analytical_computations

    if CharacteristicName.PMF in analytical_comps:
        pmf_func = analytical_comps[CharacteristicName.PMF]
    elif CharacteristicName.PDF in analytical_comps:
        pmf_func = analytical_comps[CharacteristicName.PDF]
    else:
        return 1.0

    total = 0.0
    for k in range(domain_left, domain_right + 1):
        try:
            p = float(pmf_func(float(k)))
            if p >= 0 and not (np.isnan(p) or np.isinf(p)):
                total += p
        except (ValueError, TypeError):
            continue

    return total


__all__ = [
    "calculate_pmf_sum",
    "determine_domain_from_pmf",
    "determine_domain_from_support",
]
