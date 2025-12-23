"""
CFFI build script for UNURAN.

Compiles UNURAN from sources in vendor/unuran-1.11.0/ and creates Python bindings.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from cffi import FFI  # type: ignore[import-untyped]

# TODO: Replace with logging
ENABLE_PRINTS = False

MODULE_NAME = "pysatl_core.stats._unuran.bindings._unuran_cffi"
WINDOWS_PREBUILT_DIRNAME = "unuran-win32"
WINDOWS_LIB_CANDIDATES = ("libunuran.lib", "unuran.lib")
WINDOWS_DLL_CANDIDATES = ("libunuran.dll", "unuran.dll")


def _print(*args: Any, **kwargs: Any) -> None:
    """Conditionally print based on ENABLE_PRINTS flag."""
    if ENABLE_PRINTS:
        print(*args, **kwargs)


ffi = FFI()

ffi.cdef("""
    struct unur_distr;
    struct unur_gen;
    struct unur_par;

    typedef struct unur_distr* UNUR_DISTR;
    typedef struct unur_gen* UNUR_GEN;
    typedef struct unur_par* UNUR_PAR;

    UNUR_DISTR unur_distr_cont_new(void);
    UNUR_DISTR unur_distr_discr_new(void);

    int unur_distr_cont_set_pdf(UNUR_DISTR distribution,
                               double (*pdf)(double, const struct unur_distr*));
    int unur_distr_cont_set_dpdf(UNUR_DISTR distribution,
                                double (*dpdf)(double, const struct unur_distr*));
    int unur_distr_cont_set_cdf(UNUR_DISTR distribution,
                               double (*cdf)(double, const struct unur_distr*));
    int unur_distr_cont_set_invcdf(UNUR_DISTR distribution,
                                   double (*invcdf)(double, const struct unur_distr*));
    int unur_distr_cont_set_domain(UNUR_DISTR distribution, double left, double right);
    int unur_distr_cont_set_mode(UNUR_DISTR distribution, double mode);
    int unur_distr_cont_set_pdfparams(UNUR_DISTR distribution, const double* params, int n_params);

    int unur_distr_discr_set_pmf(UNUR_DISTR distribution,
                                double (*pmf)(int, const struct unur_distr*));
    int unur_distr_discr_set_cdf(UNUR_DISTR distribution,
                                double (*cdf)(int, const struct unur_distr*));
    int unur_distr_discr_set_pv(UNUR_DISTR distribution, const double* pv, int n_pv);
    int unur_distr_discr_set_pmfparams(UNUR_DISTR distribution, const double* params, int n_params);
    int unur_distr_discr_set_domain(UNUR_DISTR distribution, int left, int right);
    int unur_distr_discr_set_pmfsum(UNUR_DISTR distribution, double sum);
    int unur_distr_discr_make_pv(UNUR_DISTR distribution);

    UNUR_PAR unur_arou_new(const UNUR_DISTR distribution);
    UNUR_PAR unur_tdr_new(const UNUR_DISTR distribution);
    UNUR_PAR unur_hinv_new(const UNUR_DISTR distribution);
    UNUR_PAR unur_pinv_new(const UNUR_DISTR distribution);
    UNUR_PAR unur_ninv_new(const UNUR_DISTR distribution);
    UNUR_PAR unur_dgt_new(const UNUR_DISTR distribution);

    UNUR_GEN unur_init(UNUR_PAR parameters);

    double unur_sample_cont(UNUR_GEN generator);
    int unur_sample_discr(UNUR_GEN generator);
    int unur_sample_vec(UNUR_GEN generator, double* vector);

    double unur_quantile(UNUR_GEN generator, double U);

    void unur_free(UNUR_GEN generator);
    void unur_distr_free(UNUR_DISTR distribution);
    void unur_par_free(UNUR_PAR par);

    const char* unur_get_strerror(const int errnocode);
    int unur_get_errno(void);
    const char* unur_gen_info(UNUR_GEN generator, int help);
""")


def _get_project_root() -> Path:
    """Finds the project root by searching for pyproject.toml marker."""
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent

    while current_dir != current_dir.parent:  # Stop at filesystem root
        if (current_dir / "pyproject.toml").exists():
            return current_dir
        current_dir = current_dir.parent

    raise RuntimeError(
        f"Could not find project root (pyproject.toml) starting from {current_file.parent}"
    )


def _get_unuran_paths() -> tuple[Path, Path]:
    """Determines paths to UNURAN sources."""
    project_root = _get_project_root()
    unuran_dir = project_root / "vendor" / "unuran-1.11.0"
    unuran_src = unuran_dir / "src"

    if not unuran_src.exists():
        raise FileNotFoundError(
            f"UNURAN sources not found at {unuran_src}\n"
            f"Expected location: {project_root}/vendor/unuran-1.11.0/src/"
        )

    return unuran_dir, unuran_src


def _build_unuran_library(unuran_dir: Path, build_dir: Path) -> tuple[Path | None, Path]:
    """
    Builds UNURAN library via autotools.

    Returns:
        (lib_path, include_dir): Path to library and directory with headers
    """
    build_dir.mkdir(parents=True, exist_ok=True)

    configure_script = unuran_dir / "configure"
    if not configure_script.exists():
        _print("Warning: configure script not found, will try to link with system library")
        return None, unuran_dir / "src"

    _print("Configuring UNURAN...")
    try:
        env = os.environ.copy()
        env["CFLAGS"] = "-fPIC " + env.get("CFLAGS", "")
        subprocess.run(
            [
                str(configure_script),
                f"--prefix={build_dir}",
                "--enable-static",
                "--disable-shared",
            ],
            cwd=str(unuran_dir),
            check=True,
            capture_output=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else "Unknown error"
        _print(f"Warning: configure failed: {error_msg}")
        _print("Will try to link with system library")
        return None, unuran_dir / "src"

    _print("Cleaning previous build (if any)...")
    subprocess.run(
        ["make", "clean"],
        cwd=str(unuran_dir),
        capture_output=True,
    )

    _print("Building UNURAN...")
    try:
        env = os.environ.copy()
        env["CFLAGS"] = "-fPIC " + env.get("CFLAGS", "")
        subprocess.run(
            ["make", "-j"],
            cwd=str(unuran_dir),
            check=True,
            capture_output=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else "Unknown error"
        _print(f"Warning: make failed: {error_msg}")
        _print("Will try to link with system library")
        return None, unuran_dir / "src"

    lib_paths = [
        unuran_dir / "src" / ".libs" / "libunuran.a",
        unuran_dir / ".libs" / "libunuran.a",
        build_dir / "lib" / "libunuran.a",
    ]

    for lib_path in lib_paths:
        if lib_path.exists():
            _print(f"Found UNURAN library at {lib_path}")
            return lib_path, unuran_dir / "src"

    _print("Warning: Built library not found, will try to link with system library")
    return None, unuran_dir / "src"


def _setup_static_source(static_lib: Path, include_dir: Path) -> None:
    """Scenario 1: Link with the freshly built static library."""
    _print(f"Using static library: {static_lib}")
    ffi.set_source(
        MODULE_NAME,
        '#include "unuran.h"',
        extra_objects=[str(static_lib)],
        include_dirs=[str(include_dir)],
        libraries=[],
        extra_compile_args=["-std=c99", "-O2"],
    )


def _setup_system_source(include_dir: Path, system_lib_path: str) -> None:
    """Scenario 2: Link with a library installed in the system."""
    _print(f"Using system library: {system_lib_path}")
    ffi.set_source(
        MODULE_NAME,
        '#include "unuran.h"',
        libraries=["unuran"],
        include_dirs=[str(include_dir)],
        library_dirs=[os.path.dirname(system_lib_path)],
    )


def _setup_fallback_source(include_dir: Path) -> None:
    """Scenario 3: Attempt direct compilation (fallback if no library found)."""
    _print("Warning: No pre-built library found, attempting direct source compilation")
    _print("This may fail due to complex dependencies. Consider building UNURAN first.")
    ffi.set_source(
        MODULE_NAME,
        '#include "unuran.h"',
        include_dirs=[str(include_dir)],
        libraries=["unuran"],
        extra_compile_args=["-std=c99"],
    )


def _find_windows_prebuilt_artifacts(
    unuran_dir: Path,
) -> tuple[Path, Path | None, Path]:
    """
    Locate Windows binaries placed under vendor/unuran-1.11.0/unuran-win32.
    Returns (lib_path, dll_path, include_dir).
    """
    win_dir = unuran_dir / WINDOWS_PREBUILT_DIRNAME
    if not win_dir.exists():
        raise FileNotFoundError(
            "Windows binaries not found. Download UNU.RAN for Windows and extract it to "
            f"{win_dir} (see vendor/unuran-1.11.0/README.win32)."
        )

    lib_path = next(
        (win_dir / name for name in WINDOWS_LIB_CANDIDATES if (win_dir / name).exists()), None
    )
    if lib_path is None:
        raise FileNotFoundError(
            f"UNURAN import library (*.lib) not found in {win_dir}. "
            "Expected one of: " + ", ".join(WINDOWS_LIB_CANDIDATES)
        )

    dll_path = next(
        (win_dir / name for name in WINDOWS_DLL_CANDIDATES if (win_dir / name).exists()), None
    )
    include_dir = win_dir if (win_dir / "unuran.h").exists() else unuran_dir / "src"
    return lib_path, dll_path, include_dir


def _setup_windows_prebuilt_source(lib_path: Path, include_dir: Path) -> None:
    """Configure ffi to link against the prebuilt Windows import library."""
    _print(f"Using Windows prebuilt library: {lib_path}")
    ffi.set_source(
        MODULE_NAME,
        '#include "unuran.h"',
        include_dirs=[str(include_dir)],
        libraries=["unuran"],
        library_dirs=[str(lib_path.parent)],
        extra_compile_args=["/O2"],
    )


def _maybe_copy_windows_runtime(dll_path: Path | None, destination_dir: Path) -> None:
    """Copy libunuran.dll next to the compiled extension so wheels include it."""
    if dll_path is None or not dll_path.exists():
        return
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / dll_path.name
    shutil.copy2(dll_path, target)


def main() -> None:
    project_root = _get_project_root()
    unuran_dir, unuran_src = _get_unuran_paths()
    build_dir = unuran_dir.parent / "unuran-build"

    windows_dll: Path | None = None

    if os.name == "nt":
        lib_path, windows_dll, include_dir = _find_windows_prebuilt_artifacts(unuran_dir)
        _setup_windows_prebuilt_source(lib_path, include_dir)
    else:
        static_lib, include_dir = _build_unuran_library(unuran_dir, build_dir)

        if static_lib and static_lib.exists():
            _setup_static_source(static_lib, include_dir)
        else:
            import ctypes.util

            system_lib = ctypes.util.find_library("unuran")

            if system_lib:
                _setup_system_source(include_dir, system_lib)
            else:
                _setup_fallback_source(include_dir)

    _print("Compiling CFFI bindings for UNURAN...")
    _print(f"UNURAN source directory: {unuran_src}")
    _print(f"Include directory: {include_dir}")
    build_output_dir = project_root / "src"
    previous_cwd = Path.cwd()
    try:
        os.chdir(build_output_dir)
        ffi.compile(verbose=True)
    finally:
        os.chdir(previous_cwd)

    bindings_dir = project_root / "src" / "pysatl_core" / "stats" / "_unuran" / "bindings"
    _maybe_copy_windows_runtime(windows_dll, bindings_dir)
    _print("Compilation complete!")


if __name__ == "__main__":
    main()
