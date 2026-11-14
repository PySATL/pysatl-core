# Accuracy benchmarks for pysatl-core

Scenario files and a runnable entry point for the accuracy model of the
`pysatl_core_bench` framework (issue #124: accuracy only; time/memory belong to
a separate future model).

## Layout

- `scenarios/` — JSON scenarios in the grammar of `pysatl_core_bench.accuracy`
  (`schema_version 1.0`); `sweep_spec`/`chain_spec` files are expanded into
  plain scenarios automatically.
- `reference_sources.py` — example *external* reference wrappers built by
  the bench consumer (the framework itself never ships library wrappers):
  - `scipy_norm_source()` — a runnable compute wrapper (generic
    `ExternalFunctionSource`; its callable takes the query points as a
    pysatl-core `NumericArray`, one wrapper serves any library);
  - `mpmath_table_sources()` — the production reference: mpmath-generated
    tables from `reference_tables/` loaded as lookup-only
    `TableLookupSource` services over `PrecomputedTable` data (library, version, dps and `table_hash`
    travel with the data; `coverage_tol = 0` demands an exact point
    match, no interpolation).
- `reference_tables/generate_mpmath_tables.py` — regenerates the tables
  with mpmath at high dps (see the script docstring for the lattice
  choice); the `*.json` artifacts are committed and checked by tests.
- `run_accuracy.py` — expands, loads, runs and prints the report for
  every scenario file.

Package layout of the framework itself: `src/pysatl_core_bench/accuracy`
is split into three work folders — `computations/` (value sources,
spaces, metrics), `scenarios/` (spec, loader/expander/validator, grids,
special points, invariants), `reporting/` (store, comparator, reporter) —
plus the kernel (`runner.py`, `dto.py`, `types.py`, `errors.py`,
`canonical.py`, `core_state.py`); see the `accuracy` package docstring.

## Running a JSON scenario

Commands follow the upstream README style. `pyproject.toml` lists both
packages (`pysatl_core` and `pysatl_core_bench`) in `tool.poetry.packages`, so
`poetry install` puts both into the venv and no `PYTHONPATH` hack is needed
(the full command reference lives in `COMMANDS.md` at the repository root):

```bash
poetry install --with dev                            # once; repeat after pyproject changes
poetry run pytest tests/bench                        # framework tests
poetry run python benchmarks/run_accuracy.py --smoke   # smoke: set + expected outcomes
poetry run python benchmarks/run_accuracy.py --all     # every scenario file
poetry run python benchmarks/run_accuracy.py --all S1.4       # + substring filter
poetry run python benchmarks/run_accuracy.py --smoke --out out/ # persist artifacts
```

`--smoke` runs the canonical smoke set (the repository scenarios plus the
drop-in smoke files `S1.0`, `S1.6`, `S1.7`, `S3.1`, `NEG1–NEG5`, all shipped
in `benchmarks/scenarios/`) and verifies every file against its expected
outcome: positives must be `measured` with per-region metrics, each negative
must show exactly its honest status (NEG2 also carries the
`degenerate_comparison` note). Exit code 0 — pipeline healthy; 2 — an
expectation is violated. `--all` prints the reports and applies the А-15
exit rule (`failed`/`invalid_reference` → 2); for the full set exit code 2
is the *normal* outcome (NEG4 is part of the set on purpose). A persisted
measurement store is created only by `--out <dir>`; the repository and the
deliverable archives ship without one — pass an empty directory.

Python API — the same pipeline by hand:

```python
from pysatl_core_bench import (
    AccuracyRunner,
    OnDemandComputer,
    PySATLAdapter,
    ResultsStore,
    ScenarioExpander,
    ScenarioLoader,
    SourceRegistry,
)
from pysatl_core_bench.accuracy import reset_core_state

registry = SourceRegistry()
registry.register(PySATLAdapter())  # measured side
registry.register(OnDemandComputer(my_source()))  # your reference side

store = ResultsStore(root="out/")  # omit root for in-memory
for raw in ScenarioExpander().expand(scenario_dict):
    reset_core_state()  # core singletons per scenario
    spec = ScenarioLoader().load(raw)  # canonical + spec_hash
    results = AccuracyRunner(registry, store).run(spec)
    report = store.read_report(spec.comparison_group)
    print(report.text)
```

A scenario dict is one JSON file parsed with `json.loads`. Expansion is
skipped automatically when the file has no `sweep_spec`/`chain_spec`.

Artifacts written when `--out` is given:

- `out/<comparison_group>/result-<scenario>-<pair>.json` — one result per
  pair (status, explanation, metrics, provenance with the plan dump);
- `out/report-<comparison_group>.json` — rendered report (flat table plus
  the sweep curve when the group is a sweep);
- `out/sidecar-*.json` — content-addressed pointwise values.

## Scenarios

| File | What it measures | Smoke expectation |
|---|---|---|
| `S1.0_minimal.json` | **minimal file**: participants + one `characteristic`; the file stem becomes the comparison group | `measured` |
| `S1.1_normal_pdf.json` | Normal(0,1) pdf vs scipy reference over bulk + both tails | `measured` |
| `S1.2_normal_cdf.json` | Normal(0,1) cdf; right tail measured in survival space (S = 1 − F) | `measured` |
| `S1.3_normal_pdf_from_cf.json` | Normal(0,1) pdf computed **from the characteristic function** (`ForcedGraphStrategy` with `from: "cf"`, route `cf → cf_to_pdf_1C`) vs reference | `measured` |
| `S1.4_normal_pdf_mpmath_table.json` | Normal(0,1) pdf vs the **production reference**: mpmath-generated `PrecomputedTable` data served by `TableLookupSource` (dps 50, exact lattice coverage) | `measured` |
| `S1.5_normal_cdf_mpmath_table.json` | Normal(0,1) cdf vs the mpmath table; right tail in survival space (D8 conversion facts in provenance) | `measured` |
| `S1.6_normal_ppf.json` | ppf: quantile-level regions; e1/e2_true metrics (+ `e2_self_consistency` diagnostics, rev. 46) | `measured` |
| `S1.7_normal_cdf_survival.json` | survival-space query, `S = 1 − F` conversion facts in provenance | `measured` |
| `S3.1_symmetric_agreement.json` | 3 participants (Core default, Core from=cf, scipy), `pair_policy "symmetric"` — agreement | `measured` |
| `sweep_normal_cdf_npoints.json` | sweep over `grid_hint.n_points` (33/65/129); the report renders the curve over `sort_key` | `measured` |
| `chain_normal_cdf_sigma.json` | chain of stages core(sigma) vs reference(sigma) for sigma in {0.5, 1.0, 2.0} | `measured` |
| `NEG1_unknown_strategy.json` | unknown strategy — no silent fallback | `unsupported_configuration` |
| `NEG2_degenerate_selfpair.json` | identical configurations — comparison still runs | `measured` + `degenerate_comparison` note |
| `NEG3_table_wrong_characteristic.json` | table of the wrong characteristic | `not_applicable` |
| `NEG4_missing_reference.json` | reference not registered | `invalid_reference` |
| `NEG5_subject_unregistered.json` | subject not registered — per-pair status, no crash | `unsupported_configuration` |

Note: a reference participant carries its own `params`; sweeping the
distribution shape therefore goes through `chain_spec` (stage <-> its
reference), while single-axis sweeps (grid resolution, strategy options)
use `sweep_spec`.

## Strategies (М-series)

`participant.strategy` selects the computation route. Names in JSON match
the adapter catalog; absence means the library default first-match.

```jsonc
"strategy": "DefaultComputationStrategy"
// or
"strategy": {"name": "ForcedGraphStrategy", "params": {"from": "cf"}}
```

- `ForcedGraphStrategy` forces the route to start at `params.from` (М-2):
  analytical loop at that characteristic + graph path to the target. A
  start that is not a graph node (today `cf` — the characteristic-graph
  invariants forbid one-way source nodes) is served by a direct
  fitter-registry lookup of the `from -> target` edge. The library graph
  ships no `cf -> pdf` edge; the bench provides its own Gil-Pelaez fitter
  and registers it through the public `fitter_registry().register` API —
  no pysatl-core file is modified. The plan dump in provenance is the fact
  (М-7) either way.
- Validation happens at *assess*: unknown name, bad params, inapplicable
  start or a missing route produce `unsupported_configuration` — never a
  silent fallback (М-6). Lookup sources (tables) refuse strategies with
  `not_applicable`.

## Reference tables (production reference)

`reference_tables/normal_{pdf,cdf}_mu0_sigma1.json` hold 3329 points on
the lattice `x = m·2⁻⁷` over `[−13, 13]`, computed with mpmath at
dps 50 and rounded once to float64 (max deviation against scipy:
≈ 1.1e-16 — float64 rounding itself). The lattice covers every
`numpy.linspace` grid the shipped scenarios produce (their steps are
multiples of `2⁻⁷` on this range), so `coverage_tol = 0` is honest:
lookup is exact, never interpolated.

Regenerate and verify from the repository root:

```bash
poetry run python benchmarks/reference_tables/generate_mpmath_tables.py
```

The generator round-trips every written file through
`PrecomputedTable.from_json` and asserts the `table_hash` is stable.

## Consuming the benchmark code

No build tooling of the upstream repository is modified beyond listing the
second package in `tool.poetry.packages`. Downstream consumers either
`poetry install` (both packages land in the venv), or copy
`src/pysatl_core_bench` (pure Python, `py.typed`-marked) into their own
packaging — the package has no dependencies beyond pysatl-core, numpy
and scipy.
