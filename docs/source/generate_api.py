#!/usr/bin/env python3
"""Generate API documentation with a package-like tree.

This script scans the repository's ``src/`` directory and generates ``.rst`` files
under ``docs/source/api/pysatl_core`` mirroring the Python package structure.

Design goals:
 - Sidebar navigation should reflect *module/package tree*, not a flat list of members.
 - Package ``index.rst`` pages contain only a toctree (and optional package docstring),
  while *module* pages contain ``automodule`` with members.

Run:
    python docs/source/generate_api.py
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
DOCS_SOURCE = Path(__file__).resolve().parent
API_ROOT = DOCS_SOURCE / "api"
PKG_NAME = "pysatl_core"
PKG_DIR = SRC_ROOT / PKG_NAME

GENERATED_ROOT = API_ROOT / PKG_NAME


@dataclass(frozen=True)
class ModuleDoc:
    dotted: str  # e.g. pysatl_core.distributions.registry.graph
    rst_path: Path  # relative to DOCS_SOURCE
    title: str  # display title


def _is_package_dir(path: Path) -> bool:
    return path.is_dir() and (path / "__init__.py").exists()


def _copy_user_guide_notebooks(repo_root: Path, docs_source_dir: Path) -> None:
    src_nb = repo_root / "examples" / "overview.ipynb"
    dst_dir = docs_source_dir / "examples"
    dst_nb = dst_dir / "overview.ipynb"

    if not src_nb.exists():
        raise FileNotFoundError(f"Notebook not found: {src_nb}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy only if missing or source is newer
    if not dst_nb.exists() or src_nb.stat().st_mtime > dst_nb.stat().st_mtime:
        shutil.copy2(src_nb, dst_nb)


def _iter_packages_and_modules(pkg_dir: Path) -> Iterable[tuple[Path, list[Path], list[Path]]]:
    """Yield (package_dir, subpackage_dirs, module_files) recursively."""
    for current, dirs, files in os.walk(pkg_dir):
        current_path = Path(current)

        # Only traverse Python packages (directories with __init__.py)
        if not _is_package_dir(current_path):
            # prevent descending into non-packages
            dirs[:] = []
            continue

        subpackages = []
        for d in list(dirs):
            dp = current_path / d
            if _is_package_dir(dp):
                subpackages.append(dp)
            else:
                # don't descend into non-packages
                dirs.remove(d)

        modules = []
        for f in files:
            if not f.endswith(".py"):
                continue
            if f == "__init__.py":
                continue
            if f.startswith("_"):
                # keep private modules out of API by default
                continue
            modules.append(current_path / f)

        yield current_path, sorted(subpackages), sorted(modules)


def _to_dotted(module_path: Path) -> str:
    rel = module_path.relative_to(SRC_ROOT)
    parts = list(rel.with_suffix("").parts)
    return ".".join(parts)


def _rst_title(name: str) -> str:
    # For files, strip .py; for directories, keep as is
    return name.replace("_", " ")


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _package_index_rst(package_dotted: str, title: str, children: list[Path]) -> str:
    """Create index.rst for a package.

    Important: Do NOT include :members: here to avoid the navigation becoming a flat list.
    """
    lines = [
        title,
        "=" * len(title),
        "",
        f".. automodule:: {package_dotted}",
        "   :no-index:",
        "",
    ]
    if children:
        lines += [
            ".. toctree::",
            "   :maxdepth: 2",
            "",
        ]
        for child in children:
            # paths are relative to api/<PKG_NAME>/.../index.rst location
            lines.append(f"   {child.as_posix()}")
        lines.append("")
    return "\n".join(lines)


def _module_page_rst(module_dotted: str, title: str) -> str:
    return "\n".join(
        [
            title,
            "=" * len(title),
            "",
            f".. automodule:: {module_dotted}",
            "   :members:",
            "   :undoc-members:",
            "   :show-inheritance:",
            "   :member-order: bysource",
            "",
        ]
    )


def main() -> None:
    docs_source_dir = Path(__file__).resolve().parent
    repo_root = docs_source_dir.parents[1]

    _copy_user_guide_notebooks(repo_root=repo_root, docs_source_dir=docs_source_dir)

    if not PKG_DIR.exists():
        raise SystemExit(f"Package directory not found: {PKG_DIR}")

    # Clean generated root
    if GENERATED_ROOT.exists():
        shutil.rmtree(GENERATED_ROOT)
    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)

    # Generate tree
    for pkg_path, subpackages, modules in _iter_packages_and_modules(PKG_DIR):
        pkg_dotted = _to_dotted(pkg_path / "__init__.py")[:-9]  # strip .__init__
        rel_pkg = pkg_path.relative_to(PKG_DIR)  # path inside pysatl_core

        out_dir = GENERATED_ROOT / rel_pkg
        index_path = out_dir / "index.rst"

        # children docs (relative to this index)
        children: list[Path] = []
        # subpackage indices
        for sp in subpackages:
            sp_rel = sp.relative_to(pkg_path)
            children.append(Path(sp_rel.as_posix()) / "index")
        # modules
        for mod in modules:
            mod_rel = mod.relative_to(pkg_path)
            children.append(Path(mod_rel.with_suffix("").as_posix()))

        title = PKG_NAME if rel_pkg == Path(".") else rel_pkg.name
        _write(index_path, _package_index_rst(pkg_dotted, title, children))

        # module pages
        for mod in modules:
            mod_dotted = _to_dotted(mod)
            mod_rel = mod.relative_to(PKG_DIR).with_suffix("")  # inside pysatl_core
            mod_out = GENERATED_ROOT / mod_rel
            _write(mod_out.with_suffix(".rst"), _module_page_rst(mod_dotted, mod_rel.name))

    # Also ensure api/index.rst points at the generated root index
    api_index = API_ROOT / "index.rst"
    if not api_index.exists():
        _write(
            api_index,
            "\n".join(
                [
                    "API Reference",
                    "=============",
                    "",
                    "This section contains the complete API documentation for the",
                    f":mod:`{PKG_NAME}` package — the core computation and distribution",
                    "infrastructure of PySATL.",
                    "",
                    "The documentation is organized to reflect the internal package and",
                    "module structure.",
                    "",
                    ".. toctree::",
                    "   :maxdepth: 1",
                    "",
                    f"   {PKG_NAME}/index",
                    "",
                ]
            ),
        )


if __name__ == "__main__":
    main()
