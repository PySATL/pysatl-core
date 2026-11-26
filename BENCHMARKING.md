# BENCHMARKING — the pysatl-core accuracy benchmark

How to measure the numerical accuracy of pysatl-core with the benchmark
framework shipped in this repository (`src/pysatl_core_bench`, driven by
`benchmarks/run_accuracy.py`), and how to write a scenario JSON for it.
Environment setup (clone, submodules, `poetry install`) is described in
the repository README.

The model: the library under test is a black box behind a registry of
*value sources*.  A scenario declares **what** is measured (a
characteristic of a distribution, on chosen points), **where** (regions of
the argument axis), and **who measures whom** (a measured subject, one or
more references).  The framework runs both sides on the same points and
reports honest per-region error metrics — or an honest refusal — for every
participant pair.  It never silently substitutes a missing value.

> Scenario grammar version: `schema_version` `1.2` (files written in
> `1.1` stay valid; the 1.2 generation is purely additive).

---

## 1. Result statuses

Every result row carries exactly one of five statuses, frozen by the
accuracy model:

| Status | Meaning |
| ------ | ------- |
| `measured` | values compared; per-region metrics present (partial coverage is counted in notes) |
| `not_applicable` | the source cannot answer this query by design (e.g. a table for another characteristic) |
| `unsupported_configuration` | the configuration is known but unsupported (unknown strategy, unregistered source, wrong parametrization) |
| `failed` | the measured side crashed |
| `invalid_reference` | the reference side is unusable (crash, empty coverage, disproved by an invariant) |

Every non-`measured` outcome carries a human-readable explanation;
validator notes (`degenerate_comparison`, `explicit_points_out_of_support`,
`explicit_points_deduplicated`, `tail_extension_fallback`,
`invariant_violation`, …) explain every transformation the framework
applied.  Nothing is ever silently substituted.

---

## 2. Commands

### 2.1 Framework test suite

```bash
poetry run pytest tests/bench    # the accuracy-framework suite
poetry run pytest tests/unit     # the upstream pysatl-core suite
poetry run pytest                # everything
```

### 2.2 Smoke mode — pipeline health check

```bash
poetry run python benchmarks/run_accuracy.py --smoke
```

Runs the canonical smoke set (every scenario file shipped under
`benchmarks/scenarios/`: S1.0–S1.11, S3.1, the sweep and chain files,
NEG1–NEG5) and **verifies each file against its expected outcome**:
positive files must come out `measured` with per-region metrics, and every
NEG file must show exactly its honest status (NEG1/NEG5 →
`unsupported_configuration`, NEG3 → `not_applicable`, NEG4 →
`invalid_reference`, NEG2 → `measured` with the `degenerate_comparison`
note).

Exit codes: **0** — pipeline healthy; **2** — an expectation is violated
(a measurement is broken); a file rejected by the loader is also a smoke
failure.

### 2.3 Full run of every scenario file

```bash
poetry run python benchmarks/run_accuracy.py --all
```

Runs every `*.json` in `benchmarks/scenarios/` and prints each group's
report.  The exit rule (`A-15`): `failed`/`invalid_reference` → exit 2,
honest outcomes → exit 0.  For the shipped set the exit code **2 is
normal** — NEG4 (`invalid_reference`) is part of the set on purpose.

### 2.4 Running one specific scenario file (or a subset)

The CLI takes a **substring filter on file names**; it works with both
modes:

```bash
poetry run python benchmarks/run_accuracy.py --all S1.9        # one file by name
poetry run python benchmarks/run_accuracy.py --smoke S1.10     # one file, with expectations
poetry run python benchmarks/run_accuracy.py --all mpmath      # every file matching "mpmath"
poetry run python benchmarks/run_accuracy.py --list            # what would run (and exits)
poetry run python -m benchmarks.run_accuracy --smoke           # module invocation, same thing
```

`--smoke`/`--all`/`--clean` are mutually exclusive; the filter combines
with both run modes.  Exit code 1 means the filter matched no files.

### 2.5 Where the artifacts go

Every run persists its artifacts; the default storage is
`benchmark_results/run-<timestamp>/` (created automatically, accumulates
history, kept out of git).  An explicit `--out DIR` is used as-is.

```bash
poetry run python benchmarks/run_accuracy.py --smoke --out out/
```

Storage layout:

- `SUMMARY.md` — the human-readable entry point of the run: one
  column-aligned table per scenario file, one row per participant pair
  (status, worst-region `abs_err`, `LRE`, validator notes) plus the list
  of rejected files;
- `<group>/result-<scenario>-<pair>.json` — one result per pair (status,
  explanation, metrics, provenance, predicted special points,
  `excluded_points`);
- `report-<group>.json` — the rendered report of the group (flat table,
  plus the sweep curve for sweep/chain files);
- `sidecar-*.json` — content-addressed pointwise values
  (reference/subject per grid hash).

Because the storage accumulates history, there is an explicit command to
wipe it — the default `benchmark_results/`, or the directory given with
`--out`:

```bash
poetry run python benchmarks/run_accuracy.py --clean             # wipe the default storage
poetry run python benchmarks/run_accuracy.py --clean --out out/  # wipe an explicit storage
```

### 2.6 Reference tables

The production tables under `benchmarks/reference_tables/` are generated
by mpmath (50 decimal digits).  Regenerate them with:

```bash
poetry run python benchmarks/reference_tables/generate_mpmath_tables.py
```

---

## 3. The scenario JSON

A scenario file is one JSON object.  The minimal legal file is three
questions short — distribution, participants, one query
(`benchmarks/scenarios/S1.0_minimal.json`):

```json
{
  "distribution": {
    "family": "Normal",
    "parametrization": "meanStd",
    "parameters": {"mu": 0.0, "sigma": 1.0}
  },
  "subject":    {"source_id": "pysatl-core"},
  "references": [{"source_id": "scipy-norm"}],
  "queries": [{"characteristic": "pdf"}]
}
```

Every optional field has a documented default — the Default column of the
field tables below; the loader materializes every default before
executing, so the canonical spec always shows the effective values.

### 3.1 Top-level fields

| Field               | Type | Default | Meaning |
| ------------------- | ---- | ------- | ------- |
| `schema_version`    | `"1.1"` \| `"1.2"` | the current generation (`1.2`) | grammar generation of the file; any other value is rejected |
| `scenario_id`       | string | derived: `comparison_group`, or `comparison_group#<sort_key>` for expanded sweep/chain variants | identity of one measured scenario |
| `comparison_group`  | string | `scenario_id`; if **neither** is given, `run_accuracy.py` injects the file stem as the group | artifact/report grouping; a sweep/chain file's variants share one group |
| `distribution`      | object | **required** | the measured distribution, scenario-level (§3.2) |
| `participants`      | list of objects | — **or** the `subject`/`references` shorthand | who participates (§3.3) |
| `pair_policy`       | `"subject"` \| `"symmetric"` \| `[[i, j], ...]` | `"subject"` | who is compared with whom (§3.4) |
| `queries`           | list of objects | **required** | what is measured (§3.5) |
| `regions`           | list of objects | `[{"name": "bulk", "interval": [null, null], "output_scale": "probability"}]` | where on the axis (§3.6) |
| `invariant_check_on`| list of strings | `[]` (no checks) | participants to post-check mathematically (§3.7) |
| `sweep_spec`        | `{param, values}` | — | sweep expansion (§3.8); mutually exclusive with `chain_spec` |
| `chain_spec`        | `{stages: [...]}` | — | chain expansion (§3.8); mutually exclusive with `sweep_spec` |
| `sort_key`          | number \| string | — (set by the expander) | curve axis of expanded variants |

Unknown top-level fields are rejected (a typo must fail loudly, not
measure a guess).

### 3.2 `distribution` — the measured distribution

```json
{"family": "Exponential", "parametrization": "rate", "parameters": {"lambda_": 0.5}}
```

| Field | Type | Default | Meaning |
| ----- | ---- | ------- | ------- |
| `family` | string | **required** | `Normal`, `ContinuousUniform`, `Exponential`, `Gamma` are built-in |
| `parametrization` | string \| null | grammar-level: the family's registered default | e.g. `meanStd`/`meanVar`/`meanPrec` for Normal; the shipped v1 pysatl-core adapter requires an **explicit** parametrization — an omitted one is an honest `unsupported_configuration` |
| `parameters` | object | `{}` (the family's defaults) | parameter values; ints are canonicalized to floats; the shipped v1 adapter requires **explicit** parameters for the measured side |

The distribution is **scenario-level**: one declaration serves every
participant; adapters encode it their own way (`ParametricFamilyRegister`
for pysatl-core, `loc/scale` for scipy wrappers, an identity check for
lookup tables).

### 3.3 Participants — two equivalent forms

Shorthand (one subject + references):

```json
"subject":    {"source_id": "pysatl-core"},
"references": [{"source_id": "scipy-norm"}]
```

Explicit list — with `pair_policy: "symmetric"` the redundant `role`
fields may be omitted (§3.4):

```json
"participants": [
  {"source_id": "pysatl-core"},
  {"source_id": "pysatl-core",
   "strategy": {"name": "ForcedGraphStrategy", "params": {"from": "cf"}}}
]
```

Outside `symmetric`, every list entry must carry an explicit `role`:

```json
"participants": [
  {"role": "subject", "source_id": "pysatl-core"},
  {"role": "reference", "source_id": "scipy-norm"}
]
```

| Field | Type | Default | Meaning |
| ----- | ---- | ------- | ------- |
| `role` | `"subject"` \| `"reference"` \| `"peer"` | **required in the list form, except under `pair_policy: "symmetric"`** — there every participant is a peer, so the field may be omitted (an explicit `null` behaves the same) | role inside the scenario |
| `source_id` | string | **required** | the registry key of the value source |
| `adapter_config` | object | `{}` | source-specific knobs, validated by the source, not the loader |
| `strategy` | string \| `{name, params}` | `null` — the source's default computation route | computation strategy of the source (§3.9) |

Both forms (list vs. shorthand) are mutually exclusive.  A participant's
canonical identity is `(source_id, adapter_config, strategy)` — two
participants with identical identities make a comparison *degenerate by
design* (it runs and is marked with the `degenerate_comparison` validator
note).

### 3.4 `pair_policy` — who measures whom

The three forms are mutually exclusive: a scenario compares pairs under
exactly one policy.

| Value | Meaning |
| ----- | ------- |
| `"subject"` (default) | the single `subject` is measured against every `reference` |
| `"symmetric"` | every unordered pair of participants measures both directions; the roles are redundant here, so `role` may be omitted in the participants list |
| `[[0, 1], [0, 2]]` | explicit pairs of participant indices; pairs are normalized (sorted, deduplicated) |

### 3.5 `queries` — what is measured

```json
{"characteristic": "cdf", "output_scale": "survival", "grid_hint": {"n_points": 33}}
```

| Field | Type | Default | Meaning |
| ----- | ---- | ------- | ------- |
| `characteristic` | string | **required** | `pdf`, `cdf`, `ppf`, `lpdf` are served by the v1 adapter; scalar characteristics (`mean`, `var`, …) are honestly refused (`unsupported_configuration`) |
| `output_scale` | `"probability"` \| `"survival"` | `"probability"` | the axis the compared values live on; conversions are recorded, never silent |
| `grid_hint` | object | `null` — documented default grid | steers the generated grid (§3.6.1) |
| `points` | list of numbers | `null` | the user's own coordinates (§3.6.2) |

One `(characteristic, output_scale)` pair may appear at most once per
scenario.

### 3.6 Where to measure — regions, grid hints, user points

The measurement set of a query is the **user's decision**:

- the user lists their own `points` — the framework measures exactly
  those coordinates (order preserved, no densification, no reordering);
- and/or the user declares regions with a `grid_hint` — the framework
  generates the requested grid over them;
- the two may coexist in one query: the grid and the user's points are
  then measured together (grid first, user order preserved).

What the framework *offers on top* is special-point handling: it predicts
likely special points of the distribution (support boundaries, curated
per-family rules) and densifies the *generated* part around them — the
extremes of the error usually live there.  The detection is best-effort
**and carries no completeness guarantee**: a point the framework missed is
a measurement blind spot, not a proof of smoothness.  Nothing is ever
placed outside the distribution support; user points that fall outside are
removed and reported as a count (they are never enumerated), while
inside-support points that produced no comparable value are enumerated in
the result's `excluded_points` with their coordinate and a reason tag.

#### 3.6.1 `grid_hint` — the generated grid

| Field | Type | Default | Meaning |
| ----- | ---- | ------- | ------- |
| `n_points` | int ≥ 4 | `129` (odd on purpose: the linspace gets a midpoint) | point count of the generated linspace per region |
| `tail_extension` | positive number | resolved **from the distribution parameters**: Normal `8·σ`, Exponential/Gamma `40·scale`; unknown family/parametrization → constant `10.0` (the fallback is reported as a validator note) | length of the measured segment on an unbounded region side of an infinite support |
| `densify` | boolean | `true` | `false` opts out of special-point densification — the generated part is exactly the uniform linspace |

A query with `points` but **no** `grid_hint` measures exactly those
points — no default grid is added.

#### 3.6.2 `points` — the user's coordinates

```json
{"characteristic": "pdf", "grid_hint": {"n_points": 41}, "points": [-0.5, 0.0, 25.0]}
```

The array is the user's contract "measure exactly these coordinates":
the loader validates the numbers (finite, at least one) and does **not**
normalize the content (no sorting, no deduplication, no clipping).  The
transformations the framework applies later are reported, never silent:

- duplicates inside the array are measured once (first occurrence wins,
  `explicit_points_deduplicated` note);
- non-ppf points outside the runtime support are removed
  (`explicit_points_out_of_support` note, as a count);
- a point falling into no declared region is a file defect — rejected at
  load, with its coordinate;
- `ppf` levels must lie strictly inside (0, 1) — levels at exactly 0 or 1
  are infinite quantiles.

### 3.7 `invariant_check_on` — mathematical post-checks

```json
"invariant_check_on": ["subject"]
```

Names the participants whose values are post-checked: `"subject"` or one
of the participant's `source_id`/`participant_id`.  v1 rules: finiteness
for everything, monotonicity of `cdf`/`ppf`, `pdf ≥ 0` (checked on
*native* values, before any scale conversion).  A violation disproves the
participant's values and spreads over the group (`measured` →
`invalid_reference`); characteristics without rules produce an
`invariant_check_skipped` note.

### 3.8 `sweep_spec` / `chain_spec` — one file, many scenarios

A sweep varies one nested field (any dotted path, including distribution
parameters and `grid_hint`):

```json
"sweep_spec": {"param": "queries.0.grid_hint.n_points", "values": [33, 65, 129]}
```

A chain walks over per-stage distributions/participants:

```json
"chain_spec": {"stages": [
  {"id": "sigma-0.5",
   "participant": {"source_id": "pysatl-core"},
   "reference":   {"source_id": "on-demand-scipy-norm"},
   "distribution": {"family": "Normal", "parametrization": "meanStd",
                    "parameters": {"mu": 0.0, "sigma": 0.5}}}
]}
```

Both expand into ordinary scenarios sharing one `comparison_group`; every
variant carries a `sort_key` (the sweep value or the stage index), which
the report turns into a curve/slice.  Values of a sweep axis must be
scalars; the two specs are mutually exclusive.

### 3.9 `strategy` — computation strategies

Two forms: a bare name string or `{"name": ..., "params": {...}}`.  The
v1 pysatl-core adapter knows `DefaultComputationStrategy` (no params) and
`ForcedGraphStrategy` (`{"from": "cf"}`, optionally `labels` — route
preferences).  An unknown strategy or an inapplicable forced route is an
explicit `unsupported_configuration`, never a silent fallback.

---

## 4. Value sources shipped in this repository

Registered by `benchmarks/run_accuracy.py`:

| `source_id` | Kind | Serves |
| ----------- | ---- | ------ |
| `pysatl-core` | measured subject (adapter) | the library under test |
| `scipy-norm` | external function (scipy.stats.norm) | Normal/meanStd: pdf, cdf, ppf, lpdf |
| `scipy-exp` | external function (scipy.stats.expon) | Exponential rate/scale: pdf, cdf |
| `on-demand-scipy-norm` | caching wrapper of `scipy-norm` | same, shared cache per distribution fingerprint |
| `mpmath-normal-pdf`, `mpmath-normal-cdf` | lookup tables (50-dps mpmath) | exact points of the shipped tables, `coverage_tol = 0` |

A scenario may only reference registered sources; an unknown `source_id`
is the honest `unsupported_configuration` (NEG5), not a crash.

---

## 5. Example scenario files

Every grammar feature is exercised by at least one shipped file — reading
them together is a worked tutorial:

| File | Demonstrates |
| ---- | ------------ |
| `benchmarks/scenarios/S1.0_minimal.json` | the minimal file; naming by file stem |
| `S1.1_normal_pdf.json` | `schema_version`, ids, regions with tails, `grid_hint.n_points` |
| `S1.2_normal_cdf.json` | cdf |
| `S1.3_normal_pdf_from_cf.json` | `strategy` object form, forced graph route |
| `S1.4/S1.5_*.json` | mpmath lookup tables as references |
| `S1.6_normal_ppf.json` | ppf on the probability level axis |
| `S1.7_normal_cdf_survival.json` | `output_scale: "survival"` |
| `S1.8_normal_lpdf.json` | lpdf through the graph |
| `S1.9_explicit_points.json` | user `points` + grid coexisting |
| `S1.10_tail_policy.json` | `grid_hint.tail_extension`, `densify: false`, a point outside the support |
| `S1.11_invariant_check.json` | `invariant_check_on: ["subject"]`, `strategy` string form |
| `S3.1_symmetric_agreement.json` | `participants` list without `role` fields, `pair_policy: symmetric` |
| `sweep_normal_cdf_npoints.json` | `sweep_spec` over a nested path |
| `chain_normal_cdf_sigma.json` | `chain_spec` with per-stage distributions |
| `NEG1…NEG5.json` | every honest refusal outcome |
| `examples/benchmark_scenario.json` | a copy-ready minimal example; `examples/run_benchmark_scenario.py` runs it from Python |
