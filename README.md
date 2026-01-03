# PySATL Core

[status-shield]: https://img.shields.io/github/actions/workflow/status/PySATL/pysatl-core/ci.yml?branch=main&event=push&style=for-the-badge&label=CI
[status-url]: https://github.com/PySATL/pysatl-core/actions/workflows/ci.yml
[license-shield]: https://img.shields.io/github/license/PySATL/pysatl-core.svg?style=for-the-badge&color=blue
[license-url]: LICENSE

[![CI][status-shield]][status-url]
[![MIT License][license-shield]][license-url]

**PySATL Core** is the computational core of the PySATL project, providing abstractions and infrastructure for probability distributions, parametric families, characteristic-based computations, and sampling.

The library is designed as a **foundational kernel** rather than a ready-to-use end-user package. Its primary goals are explicit probabilistic structure, extensibility, and suitability as a basis for further stochastic and statistical tooling.

> **Project status**  
> PySATL Core is currently in **early alpha**.  
> It is **not published** to package managers such as `pip` yet.  
> To experiment with the library, clone the repository and work with it locally.

---

## ✨ Key features

- **Parametric families of distributions** with multiple parametrizations  
  (e.g. Normal: `meanStd`, `meanPrec`).
- A global **family registry** for configuring, querying, and extending available distribution families.
- **Characteristic computation graph** (`CharacteristicRegistry`) that allows computing
  arbitrary characteristics by specifying only a minimal analytical subset.
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
- Poetry (recommended for development)

---

## Installation (from source)

Clone the repository:

```bash
git clone https://github.com/PySATL/pysatl-core.git
cd pysatl-core
```

### Using Poetry (recommended)

```bash
poetry install --with docs
```

### Using pip (editable install)

```bash
pip install -e ".[docs]"
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

---

## 🗺 Roadmap

- **Transformations module** for mixtures and distribution transformations.
- Extension of characteristic graphs.
- Stabilization of APIs and **publishing PySATL Core as an installable package**.

---

## License

Distributed under the **MIT License**. See [LICENSE](LICENSE).
