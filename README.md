# PySATL Core

[status-shield]: https://github.com/PySATL/pysatl-core/actions/workflows/ci.yml/badge.svg?branch=main&event=push
[status-url]: https://github.com/PySATL/pysatl-core/actions/workflows/ci.yml?query=branch%3Amain+event%3Apush
[license-shield]: https://img.shields.io/github/license/PySATL/pysatl-core.svg?style=for-the-badge&color=blue
[license-url]: LICENSE

[![CI][status-shield]][status-url]
[![MIT License][license-shield]][license-url]

**PySATL Core** is the computational core of the PySATL project, providing abstractions and infrastructure for probability distributions, parametric families, characteristic-based computations, transformations, and sampling.

The library is designed as a **foundational kernel** rather than a ready-to-use end-user package. Its primary goals are explicit probabilistic structure, extensibility, and suitability as a basis for further stochastic and statistical tooling.

> **Project status**  
> PySATL Core is currently in **early alpha**.  
> The public API may change between alpha releases.

---

## ✨ Key features

- **Parametric families of distributions** with multiple parametrizations  
  (e.g. Normal: `meanStd`, `meanPrec`).
- A global **family registry** for configuring, querying, and extending available distribution families.
- **Characteristic computation graph** (`CharacteristicRegistry`) that allows computing
  arbitrary characteristics by specifying only a minimal analytical subset.
- **Transformations module** for derived distributions:
  affine transformations (`aX + b`), binary operations (`X ± Y`, `X * Y`, `X / Y`),
  finite weighted mixtures, and characteristic-level approximations.
- Distribution objects exposing common probabilistic operations
  (sampling, analytical and fitted computations).
- Clear separation between *distribution definitions*, *parametrizations*,
  *computation strategies*, and *characteristics*.
- Modern Python with strict static typing (PEP 695).

---

## Requirements

- Python **3.12+** (the project relies on **PEP 695** syntax)
- NumPy **2.x**
- SciPy **1.13+**
- A C toolchain when building from source or from an sdist:
  - **Linux/macOS:** GCC (or Clang) plus standard build utilities.
  - **Windows:** Microsoft Visual C++ Build Tools (MSVC) from Visual Studio or the standalone Build Tools installer.
- Poetry (recommended for development)

---

## Installation

Install from PyPI:

```bash
pip install pysatl-core
```

Linux wheels are not published yet. On Linux, `pip` builds the package from the
source distribution and requires a working C toolchain.

For development, install from source.

Clone the repository:

```bash
git clone https://github.com/PySATL/pysatl-core.git
cd pysatl-core
git submodule update --init --recursive
```

### Development install

```bash
poetry install --with dev,docs
```

### Editable runtime install

```bash
pip install -e .
```

---

## 🚀 Quickstart

Below is a compact example demonstrating the use of a **built-in Normal distribution**.
It mirrors the example shown in the documentation (`examples/overview.ipynb`).

```python
from pysatl_core import (
    FamilyName,
    ParametricFamilyRegister,
    configure_normal_family,
)

configure_normal_family()
normal_family = ParametricFamilyRegister.get(FamilyName.NORMAL)

normal = normal_family.distribution(
    parametrization_name="meanStd",
    mu=0.0,
    sigma=1.0,
)

normal_alt = normal_family.distribution(
    parametrization_name="meanPrec",
    mu=0.0,
    tau=1.0,
)

samples = normal.sample(n=10_000)
print(samples[:5])

mean = normal.query_method("mean")()
variance = normal.query_method("variance")()

print(mean, variance)
```

This example uses a **predefined family** and **predefined parametrizations**.
PySATL Core also supports defining custom families, parametrizations,
and characteristic graphs.

For transformation workflows, see `examples/transformations_overview.ipynb`.

---

## 📓 Notebooks

- `examples/overview.ipynb` — base walkthrough for families, parametrizations, and characteristic queries.
- `examples/transformations_overview.ipynb` — affine, binary, finite-mixture, and approximation workflows.
- `examples/example_sampling_methods.ipynb` — sampling backends and UNURAN-oriented scenarios.

---

## 📖 Documentation

👉 **Online documentation:**  
https://pysatl.github.io/pysatl-core/

Documentation previews are automatically generated for pull requests and can be inspected via CI artifacts.

---

## 🧠 Concepts & design

- **ParametricFamily** — a family of distributions sharing a common mathematical form.
- **Parametrization** — a concrete coordinate system for a family.
- **Distribution** — a probabilistic object instantiated from a family and parametrization.
- **Characteristic graph** — a directed graph describing relationships between computable characteristics.
- **Registries** — explicit global registries enabling controlled extensibility.

---

## 🛠 Development

```bash
poetry install --with dev
poetry run pytest
poetry run pre-commit run --all-files
```

### Package sanity check

```bash
poetry run python -m build
poetry run twine check dist/*
```

These commands build the local distribution artifacts and validate their metadata.
They do not publish anything.

---

## 🗺 Roadmap

- Extension of the transformations module (functional transforms, performance improvements).
- Extension of characteristic graphs.
- Stabilization of APIs and **publishing PySATL Core as an installable package**.

---

## License

Distributed under the **MIT License**. See [LICENSE](LICENSE).
