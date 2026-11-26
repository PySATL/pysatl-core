"""Run the example accuracy scenario (``benchmark_scenario.json``) from Python.

This script is the executable companion of ``examples/benchmark_scenario.json``:
it builds a source registry (pysatl-core as the measured side, a small
scipy.stats.norm wrapper as the reference), loads the scenario file, and runs
the accuracy pipeline through the public API of ``pysatl_core_bench``.

Usage (from the repository root, or anywhere with both packages installed):

    python examples/run_benchmark_scenario.py

The script persists nothing unless a ``--out DIR`` argument is given:

    python examples/run_benchmark_scenario.py --out /tmp/example-out
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from scipy import __version__ as scipy_version, stats as sps

from pysatl_core.types import CharacteristicName, NumericArray
from pysatl_core_bench import (
    AccuracyQuery,
    AccuracyRunner,
    DistributionSpec,
    ExternalFunctionSource,
    ParticipantSpec,
    Provenance,
    PySATLAdapter,
    ResultsStore,
    ScenarioExpander,
    ScenarioLoader,
    SourceRegistry,
    SupportBounds,
)
from pysatl_core_bench.accuracy import reset_core_state

SCENARIO_FILE = Path(__file__).resolve().parent / "benchmark_scenario.json"


def scipy_norm_source(source_id: str = "scipy-norm") -> ExternalFunctionSource:
    """A minimal reference source over scipy.stats.norm (Normal/meanStd only).

    The wrapper serves exactly the identity it declares: a scenario with a
    different family, parametrization or parameter set is an honest refusal
    (``NotImplementedError``), never a silent substitution by N(0, 1).
    """

    def evaluate(
        distribution: DistributionSpec,
        participant: ParticipantSpec,  # noqa: ARG001 - adapter knobs unused here
        query: AccuracyQuery,
        points: NumericArray,
    ) -> ArrayLike:
        family = distribution.family
        parametrization = distribution.parametrization
        parameters: Mapping[str, object] = distribution.parameters
        if family != "Normal" or parametrization != "meanStd":
            msg = (
                "scipy.stats.norm serves Normal/meanStd only; the scenario declares "
                f"family={family!r}, parametrization={parametrization!r}"
            )
            raise NotImplementedError(msg)
        if "mu" not in parameters or "sigma" not in parameters:
            msg = f"scipy.stats.norm requires explicit 'mu' and 'sigma'; got {sorted(parameters)}"
            raise NotImplementedError(msg)
        mu = float(parameters["mu"])
        sigma = float(parameters["sigma"])
        arr = np.asarray(points, dtype=np.float64)
        name = query.characteristic
        if name is CharacteristicName.PDF:
            return sps.norm.pdf(arr, loc=mu, scale=sigma)
        if name is CharacteristicName.CDF:
            return sps.norm.cdf(arr, loc=mu, scale=sigma)
        if name is CharacteristicName.PPF:
            return sps.norm.ppf(arr, loc=mu, scale=sigma)
        if name is CharacteristicName.LPDF:
            return sps.norm.logpdf(arr, loc=mu, scale=sigma)
        msg = f"scipy.stats.norm does not serve characteristic {str(name)!r}"
        raise NotImplementedError(msg)

    return ExternalFunctionSource(
        source_id,
        evaluate,
        provenance=Provenance(library="scipy", library_version=scipy_version),
        characteristics=(
            CharacteristicName.PDF,
            CharacteristicName.CDF,
            CharacteristicName.PPF,
            CharacteristicName.LPDF,
        ),
        support=SupportBounds(float("-inf"), float("inf"), False, False),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=None,
        help="directory to persist results (default: in-memory store, nothing written)",
    )
    args = parser.parse_args()

    registry = SourceRegistry()
    registry.register(PySATLAdapter())  # the measured side
    registry.register(scipy_norm_source())  # the reference side

    store = ResultsStore(root=args.out)  # root=None -> in-memory
    raw = json.loads(SCENARIO_FILE.read_text(encoding="utf-8"))

    for variant in ScenarioExpander().expand(raw):
        reset_core_state()  # core singletons per scenario
        spec = ScenarioLoader().load(variant)  # canonical spec + spec_hash
        results = AccuracyRunner(registry, store).run(spec)
        for result in results:
            print(
                f"{result.scenario_id}: {result.subject_id} vs {result.reference} "
                f"-> {result.status.value}"
            )
            for record in result.metrics:
                region = record.region
                abs_err = record.metrics.get("abs_error", float("nan"))
                lre = record.metrics.get("lre", float("nan"))
                print(f"    region {region}: abs_err={abs_err:.3e}  LRE={lre:.2f}")

    report = store.read_report("example-normal-accuracy")
    if report is not None:
        print("\n--- group report ---")
        print(report.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
