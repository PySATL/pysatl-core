"""Generate production reference tables with mpmath (high precision).

The tables are loaded by the accuracy framework as ``TableLookupSource``
services over ``PrecomputedTable`` data: ``library``/``library_version``/``dps``/
``table_hash`` become the
provenance of the reference side, and ``coverage_tol = 0`` means every
grid point must be matched exactly.

Grid lattice.  The framework resolves an unbounded region side against an
infinite support with ``TAIL_EXTENSION = 10`` (``scenarios.grid``), so the
shipped scenarios (regions ``[-3, 3]`` plus tails) produce
``numpy.linspace`` grids inside ``[-13, 13]``.  With ``n_points = 2^k + 1``
their step is ``length / 2^k``, i.e. every point is an exact multiple of
``2^-7`` on this range.  The default lattice therefore is
``x = m * 2**-7`` for ``|x| <= 13``; a table built on it covers those
grids *exactly* (``coverage_tol = 0`` is honest, nearest-neighbor
interpolation is never involved).

Usage (from the repository root, after ``poetry install --with dev`` —
both packages, ``pysatl_core`` and ``pysatl_core_bench``, are listed in
``tool.poetry.packages`` and land in the venv):

    poetry run python benchmarks/reference_tables/generate_mpmath_tables.py
    poetry run python benchmarks/reference_tables/generate_mpmath_tables.py \
        --characteristics pdf,cdf,sf --dps 50

Values are computed with mpmath at ``--dps`` significant digits and
rounded once to float64 (the best representable reference); the script
prints the max deviation against scipy.stats.norm as a sanity check.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from pysatl_core_bench import PrecomputedTable
from pysatl_core_bench.accuracy.errors import FrameworkDefect

mp: Any = importlib.import_module("mpmath")  # mpmath ships no py.typed

_FORMULAS: dict[str, str] = {
    "pdf": "exp(-z**2 / 2) / (sigma * sqrt(2*pi)), z = (x - mu)/sigma",
    "cdf": "erfc(-z / sqrt(2)) / 2, z = (x - mu)/sigma",
    "sf": "erfc(z / sqrt(2)) / 2, z = (x - mu)/sigma",
}

_SPACES: dict[str, str] = {
    "pdf": "probability",
    "cdf": "probability",
    "sf": "survival",
}


def mpmath_value(characteristic: str, x: mp.mpf, mu: mp.mpf, sigma: mp.mpf) -> mp.mpf:
    """Normal(mu, sigma) characteristic at ``x`` with arbitrary precision."""
    z = (x - mu) / sigma
    if characteristic == "pdf":
        return mp.e ** (-z * z / 2) / (sigma * mp.sqrt(2 * mp.pi))
    if characteristic == "cdf":
        return mp.erfc(-z / mp.sqrt(2)) / 2
    if characteristic == "sf":
        return mp.erfc(z / mp.sqrt(2)) / 2
    msg = f"unsupported characteristic: {characteristic!r}"
    raise ValueError(msg)


def build_table(
    characteristic: str, mu: float, sigma: float, dps: int, half_width: float, step_exp: int
) -> PrecomputedTable:
    """Compute the table on the ``m * 2**-step_exp`` lattice."""
    step = 2.0**-step_exp
    m_max = int(round(half_width / step))
    points = np.arange(-m_max, m_max + 1, dtype=np.float64) * step
    mp_mu, mp_sigma = mp.mpf(str(mu)), mp.mpf(str(sigma))
    # REV-5: scoped precision — the global mp.dps is not mutated.
    with mp.workdps(dps):
        values = np.asarray(
            [
                float(mpmath_value(characteristic, mp.mpf(float(x)), mp_mu, mp_sigma))
                for x in points
            ],
            dtype=np.float64,
        )
    return PrecomputedTable(
        characteristic=characteristic,
        space=_SPACES[characteristic],
        points=points,
        values=values,
        library="mpmath",
        library_version=mp.__version__,
        family="Normal",
        parametrization="meanStd",
        parameters={"mu": mu, "sigma": sigma},
        dps=dps,
        coverage_tol=0.0,
    )


def sanity_against_scipy(table: PrecomputedTable) -> float:
    """Max |table value - scipy.stats.norm| over the lattice (sanity only)."""
    reference = (
        stats.norm.pdf(table.points)
        if table.characteristic == "pdf"
        else stats.norm.cdf(table.points)
        if table.characteristic == "cdf"
        else stats.norm.sf(table.points)
    )
    return float(np.max(np.abs(table.values - reference)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--dps", type=int, default=50)
    parser.add_argument("--half-width", type=float, default=13.0)
    parser.add_argument("--step-exp", type=int, default=7, help="lattice step 2**-STEP_EXP")
    parser.add_argument(
        "--characteristics", default="pdf,cdf", help="comma-separated subset of pdf,cdf,sf"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="output directory (default: next to this script)",
    )
    args = parser.parse_args()

    if not args.sigma > 0.0:
        msg = "sigma must be positive"
        raise ValueError(msg)
    characteristics = [c.strip() for c in args.characteristics.split(",") if c.strip()]
    unknown = [c for c in characteristics if c not in _FORMULAS]
    if unknown:
        msg = f"unknown characteristics {unknown}; supported: {sorted(_FORMULAS)}"
        raise ValueError(msg)

    args.out.mkdir(parents=True, exist_ok=True)
    for characteristic in characteristics:
        table = build_table(
            characteristic, args.mu, args.sigma, args.dps, args.half_width, args.step_exp
        )
        path = args.out / f"normal_{characteristic}_mu{args.mu:g}_sigma{args.sigma:g}.json"
        path.write_text(
            json.dumps(table.to_json_dict(), separators=(",", ":"), ensure_ascii=True),
            encoding="utf-8",
        )
        # round-trip guard: the file must load back to the same table_hash
        reloaded = PrecomputedTable.from_json(path)
        if reloaded.table_hash != table.table_hash:  # REV-5: survives python -O
            msg = f"table round-trip hash mismatch: {path}"
            raise FrameworkDefect(msg)
        deviation = sanity_against_scipy(reloaded)
        print(
            f"{path.name}: {table.points.size} points, dps={args.dps}, "
            f"hash={table.table_hash[:24]}..., max|value - scipy| = {deviation:.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
