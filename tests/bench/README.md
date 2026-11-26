# tests/bench — accuracy-framework test suite

The folder mirrors the layout of the benchmark module
(`src/pysatl_core_bench/accuracy/`), so a failing test names the part of
the framework it belongs to:

| Folder | Tests | Module under test |
| ------ | ----- | ----------------- |
| `scenarios/` | loader, expander, validator, grid generation, tail policy, invariant checker | `accuracy/scenarios/` |
| `computations/` | pysatl-core adapter, value sources, registry, metrics, scale conversion | `accuracy/computations/` |
| `reporting/` | results store, `SUMMARY.md` rendering | `accuracy/reporting/` |
| (root) | the kernel and the CLI: runner orchestration, canonical forms, DTO shapes, cross-cutting regressions, the smoke command | `runner.py`, `canonical.py`, `dto.py`, `benchmarks/run_accuracy.py` |

Shared fixtures and fake value sources live in `conftest.py` (it applies
to every subfolder).  The suite never tests pysatl-core internals
directly: the library is exercised through its public API, exactly as the
benchmark measures it.

Conventions: every test states the contract it pins in its docstring; no
mocks — uncooperative sources are real `ValueSource` implementations that
fail honestly.
