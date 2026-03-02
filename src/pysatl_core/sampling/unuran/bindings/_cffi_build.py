"""
CFFI build script that links against an already installed UNU.RAN library.

Uses the helper from find.py to locate the headers and shared library.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Final

from cffi import FFI  # type: ignore[import-untyped]

MODULE_NAME = "pysatl_core.sampling.unuran.bindings._unuran_cffi"
UNURAN_DIR_NAME = "unuran-pysatl"
_CDEF_FILE = Path(__file__).with_name("cffi_unuran.h")
UNURAN_CDEF: Final = _CDEF_FILE.read_text()

LOGGER = logging.getLogger(__name__)

ffi = FFI()

ffi.cdef(UNURAN_CDEF)


def _configure_logging(verbose: bool, log_file: str | None = None) -> None:
    """Initialize logging with INFO when verbose and WARNING otherwise.

    Parameters
    ----------
    verbose : bool
        Enable INFO-level logging; WARNING level is used otherwise.
    log_file : str | None
        Path to a file where log output will be written. If ``None``, logs
        are written to stderr only.
    """
    level = logging.INFO if verbose else logging.WARNING
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s", handlers=handlers)


def _get_project_root() -> Path:
    """Ascend parent directories to locate the repository containing pyproject.toml."""
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent

    while current_dir != current_dir.parent:
        if (current_dir / "pyproject.toml").exists():
            return current_dir
        current_dir = current_dir.parent

    raise RuntimeError(
        f"Could not find project root (pyproject.toml) starting from {current_file.parent}"
    )


def _extract_library_name(library_path: Path) -> str:
    """Return the bare library name stripping common prefixes/suffixes."""
    name = library_path.name
    if name.startswith("lib"):
        name = name[3:]

    for marker in (".so", ".dylib", ".a", ".dll"):
        if marker in name:
            name = name.split(marker, 1)[0]
            break

    return name or library_path.stem


def _configure_from_paths(include_file: Path, library_file: Path) -> None:
    """Configure the CFFI module depending on whether the target is static or shared."""
    include_dir = include_file.parent

    if library_file.suffix == ".a":
        LOGGER.info("Linking against static library %s", library_file)
        ffi.set_source(
            MODULE_NAME,
            '#include "unuran.h"',
            include_dirs=[str(include_dir)],
            extra_objects=[str(library_file)],
            extra_compile_args=["-std=c99", "-O2"],
        )
        return

    library_name = _extract_library_name(library_file)
    library_dir = library_file.parent

    LOGGER.info("Linking against shared library %s", library_file)
    ffi.set_source(
        MODULE_NAME,
        '#include "unuran.h"',
        include_dirs=[str(include_dir)],
        libraries=[library_name],
        library_dirs=[str(library_dir)],
        extra_compile_args=["-std=c99"],
    )


def find_unuran(unuran_dir: Path, raise_on_error: bool = True) -> dict[str, Path | None]:
    """
    Locate the UNU.RAN library and its header file within the submodules tree.

    Parameters
    ----------
    unuran_dir : Path
        Root of the vendored UNU.RAN source tree.
    raise_on_error : bool, default True
        When ``True``, raise :exc:`ImportError` if either the header or the
        library file cannot be found. When ``False``, return ``None`` values
        for missing paths without raising.

    Returns
    -------
    dict[str, Path | None]
        Dictionary with keys ``"include_path"`` and ``"library_path"``,
        each holding a resolved :class:`Path` or ``None`` if not found.

    Raises
    ------
    ImportError
        If ``raise_on_error`` is ``True`` and the library or header is missing.
    """
    results: dict[str, Path | None] = {
        "include_path": None,
        "library_path": None,
    }

    possible_lib_paths = [
        unuran_dir / "out" / "libunuran.a",  # Windows + Linux
        unuran_dir / "out" / "libunuran.so",  # Linux
        unuran_dir / "out" / "libunuran.dylib",  # macOS
    ]

    for path in possible_lib_paths:
        if path and path.exists():
            results["library_path"] = path.resolve()
            break

    include_path = unuran_dir / "unuran" / "src" / "unuran.h"

    if include_path.exists():
        results["include_path"] = include_path

    if not results["library_path"] and raise_on_error:
        raise ImportError("libunuran.so not found")

    if not results["include_path"] and raise_on_error:
        raise ImportError("Header unuran.h not found")

    return results


def build_unuran(unuran_dir: Path) -> None:
    """
    Build the UNU.RAN C library from source if it has not been built yet.

    Skips the build when both the header and library file are already present.
    Otherwise invokes the vendored ``build_unuran.py`` script via the current
    Python interpreter.

    Parameters
    ----------
    unuran_dir : Path
        Root of the vendored UNU.RAN source tree (must contain ``build_unuran.py``).

    Raises
    ------
    RuntimeError
        If ``build_unuran.py`` does not exist in ``unuran_dir``.
    subprocess.CalledProcessError
        If the build script exits with a non-zero status.
    """
    if all(find_unuran(unuran_dir, False).values()):
        return

    build_script = unuran_dir / "build_unuran.py"
    if not build_script.exists():
        raise RuntimeError(f"Build script {build_script} not found")

    subprocess.run([sys.executable, str(build_script)], check=True)


def main() -> None:
    """
    Entry point for the CFFI build: locate UNU.RAN, configure the extension,
    and compile the ``_unuran_cffi`` module into ``src/``.

    Raises
    ------
    RuntimeError
        If UNU.RAN cannot be found after attempting the build.
    """
    project_root = _get_project_root()
    unuran_dir = project_root / "subprojects" / UNURAN_DIR_NAME
    build_unuran(unuran_dir)
    paths = find_unuran(unuran_dir)

    if not (paths["include_path"] and paths["library_path"]):
        raise RuntimeError("UNU.RAN not found")

    include_file = Path(paths["include_path"]).expanduser().resolve()
    library_file = Path(paths["library_path"]).expanduser().resolve()

    LOGGER.info("Found UNU.RAN include at %s", include_file)
    LOGGER.info("Found UNU.RAN library at %s", library_file)

    _configure_from_paths(include_file, library_file)

    build_output_dir = project_root / "src"
    previous_cwd = Path.cwd()

    try:
        os.chdir(build_output_dir)
        ffi.compile(verbose=True)
    finally:
        os.chdir(previous_cwd)

    LOGGER.info("CFFI bindings compiled successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build CFFI bindings for the vendored UNU.RAN library."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable INFO-level logging during the build.",
    )
    parser.add_argument(
        "--log-file",
        metavar="PATH",
        default=None,
        help="Write log output to PATH in addition to stderr.",
    )
    args = parser.parse_args()
    _configure_logging(args.verbose, args.log_file)
    main()
