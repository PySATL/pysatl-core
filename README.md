# PySATL Core

[status-shield]: https://img.shields.io/github/actions/workflow/status/PySATL/pysatl-core/ci.yml?branch=main&event=push&style=for-the-badge&label=CI
[status-url]: https://github.com/PySATL/pysatl-core/actions/workflows/ci.yml
[license-shield]: https://img.shields.io/github/license/PySATL/pysatl-core.svg?style=for-the-badge&color=blue
[license-url]: LICENSE

[![CI][status-shield]][status-url]
[![MIT License][license-shield]][license-url]

**PySATL Core** is a minimal kernel for probability distributions: a distribution protocol, computation strategies, numeric converters between characteristics, and a dependency graph of characteristics. The kernel is designed as a foundation for other PySATL modules (CPD, etc.), with a focus on strict typing, extensibility, and reproducible computations.

## Features

- `Distribution` protocol with the minimal implementation `StandaloneEuclideanUnivariateDistribution`.
- Analytical and fitted computations for characteristics (`pdf`, `cdf`, `ppf`, etc.).
- Conversions between characteristics: `ppf↔cdf`, `cdf↔pdf` and their compositions.
- Characteristic registry as a directed graph with invariants and path-based planning.
- Sampling strategies (by default sampling via `ppf`).
- Strict typing (`mypy --strict`), modern Python 3.12+.

---

## Requirements

- Python **3.12+**
- NumPy **2.x**
- SciPy **1.13+**
- Poetry (for development)

## Installation

Clone the repository:

```bash
git clone https://github.com/PySATL/pysatl-core.git
cd pysatl-core
```

Install dependencies (via Poetry):

```bash
poetry install
```

Or install the package locally with `pip` (editable):

```bash
pip install -e .
```

---

## Quickstart

Below is a minimal example: define an analytical `ppf`, draw a sample, compute `cdf` and `pdf` through the kernel’s converters, and evaluate log-likelihood for a simple uniform case.

```python
from pysatl_core.types import Kind
from pysatl_core.distributions import (
    AnalyticalComputation,
    StandaloneEuclideanUnivariateDistribution,
)
from pysatl_core.distributions.characteristics import GenericCharacteristic

# Characteristic names
PDF = "pdf"
CDF = "cdf"
PPF = "ppf"

# 1) A distribution with analytical PPF (identity on [0,1] for demo)
dist_ppf = StandaloneEuclideanUnivariateDistribution(
    kind=Kind.CONTINUOUS,
    analytical_computations=[
        AnalyticalComputation[float, float](PPF, lambda q: q)  # x := ppf(q)
    ],
)

# 2) Sample 5 points (shape (n, d) = (5, 1))
sample = dist_ppf.sample(5)
print(sample.shape)  # -> (5, 1)

# 3) Compute characteristics via generic dispatch:
CDF_ = GenericCharacteristic[float, float](CDF)
PDF_ = GenericCharacteristic[float, float](PDF)

x = 0.5
print("cdf(0.5) =", CDF_(dist_ppf, x))  # cdf reconstructed from ppf
print("pdf(0.5) =", PDF_(dist_ppf, x))  # pdf reconstructed from cdf

# 4) Log-likelihood: define an analytical uniform pdf on [0,1]
dist_uniform_pdf = StandaloneEuclideanUnivariateDistribution(
    kind=Kind.CONTINUOUS,
    analytical_computations=[
        AnalyticalComputation[float, float](PDF, lambda t: 1.0 if 0.0 <= t <= 1.0 else 0.0)
    ],
)

ll = dist_uniform_pdf.log_likelihood(sample)
print("log-likelihood =", ll)  # for uniform on [0,1]: log(1) == 0
```

The idea is simple:
- The **what** is defined by the characteristic name (`"pdf"`, `"cdf"`, `"ppf"`).
- The **how** is decided by the computation strategy: it either picks an analytical implementation or reconstructs it from other characteristics using the registry graph (with optional caching).

---

## Concepts & Design

- **Distribution protocol.** A minimal stable interface for downstream PySATL modules.
- **Analytical vs Fitted.** You can provide analytical functions; otherwise the kernel composes fitted methods from other available characteristics.
- **Characteristic graph.** A directed graph over characteristic names for a fixed distribution type; path search yields a pipeline of converters.
- **Sampling via PPF.** In the 1D Euclidean case, default sampling uses `ppf` with i.i.d. `U(0,1)`.
- **Strict typing.** The codebase targets `mypy --strict` and static analysis.

---

## Development

Set up the dev environment:

```bash
poetry install --with dev
```

Run tests:

```bash
poetry run pytest -q
```

Coverage:

```bash
poetry run pytest --cov=pysatl_core --cov-report=term-missing
```

Static checks and style:

```bash
poetry run ruff check .
poetry run mypy src
```

### Pre-commit

Install hooks:

```bash
poetry run pre-commit install
```

Run manually:

```bash
poetry run pre-commit run --all-files --color always --verbose --show-diff-on-failure
```

---

## Roadmap

- Multivariate Euclidean distributions and sampling strategies.
- Expanded converter registry and stronger invariants in the characteristic graph.
- Mixtures and compositional operations in the core.

---

## License

Distributed under the **MIT** License. See [LICENSE](LICENSE).
