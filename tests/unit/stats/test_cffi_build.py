from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from pysatl_core.stats._unuran.bindings import _cffi_build as cffi_build


@pytest.fixture
def dummy_ffi(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Capture calls to ffi.set_source/compile without running a real build."""
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class DummyFFI(SimpleNamespace):
        def set_source(self, *args: object, **kwargs: object) -> None:
            calls.append(("set_source", args, kwargs))

        def compile(self, verbose: bool = False) -> None:
            calls.append(("compile", (verbose,), {}))

    dummy = DummyFFI()
    monkeypatch.setattr(cffi_build, "ffi", dummy)
    dummy.calls = calls
    return dummy


class TestCffiBuild:
    """Tests for `_cffi_build` helper functions and main routine."""

    def test_get_project_root_returns_directory_with_pyproject(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ensures _get_project_root finds the directory containing pyproject.toml."""
        project_root = tmp_path / "project"
        package_dir = project_root / "src" / "pkg"
        package_dir.mkdir(parents=True)
        (project_root / "pyproject.toml").write_text("[build-system]\n")
        dummy_file = package_dir / "module.py"
        dummy_file.write_text("# test")
        monkeypatch.setattr(cffi_build, "__file__", str(dummy_file))

        assert cffi_build._get_project_root() == project_root

    def test_get_project_root_raises_when_missing_pyproject(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verifies _get_project_root raises when no pyproject marker is found."""
        orphan_dir = tmp_path / "isolated"
        nested = orphan_dir / "pkg"
        nested.mkdir(parents=True)
        dummy_file = nested / "module.py"
        dummy_file.write_text("# test")
        monkeypatch.setattr(cffi_build, "__file__", str(dummy_file))

        with pytest.raises(RuntimeError):
            cffi_build._get_project_root()

    def test_get_unuran_paths_returns_vendor_src(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checks _get_unuran_paths returns vendor/unuran-1.11.0/src when present."""
        project_root = tmp_path / "project"
        vendor_src = project_root / "vendor" / "unuran-1.11.0" / "src"
        vendor_src.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(cffi_build, "_get_project_root", lambda: project_root)

        unuran_dir, unuran_src = cffi_build._get_unuran_paths()
        assert unuran_dir == vendor_src.parent
        assert unuran_src == vendor_src

    def test_get_unuran_paths_raises_when_src_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Asserts _get_unuran_paths fails if vendor sources are absent."""
        project_root = tmp_path / "project"
        project_root.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(cffi_build, "_get_project_root", lambda: project_root)

        with pytest.raises(FileNotFoundError):
            cffi_build._get_unuran_paths()

    def test_build_unuran_library_returns_static_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulates successful configure/make and expects static lib path."""
        unuran_dir = tmp_path / "unuran"
        src_dir = unuran_dir / "src"
        libs_dir = src_dir / ".libs"
        libs_dir.mkdir(parents=True, exist_ok=True)
        lib_path = libs_dir / "libunuran.a"
        lib_path.write_bytes(b"0")
        configure_script = unuran_dir / "configure"
        configure_script.parent.mkdir(parents=True, exist_ok=True)
        configure_script.write_text("#!/bin/sh\n")

        build_dir = tmp_path / "build"
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        static_lib, include_dir = cffi_build._build_unuran_library(unuran_dir, build_dir)

        assert static_lib == lib_path
        assert include_dir == src_dir
        assert calls[0][0] == str(configure_script)
        assert any("make" in call[0] for call in calls)

    def test_build_unuran_library_handles_configure_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Confirms configure failure triggers fallback (None library path)."""
        unuran_dir = tmp_path / "unuran"
        src_dir = unuran_dir / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        configure_script = unuran_dir / "configure"
        configure_script.parent.mkdir(parents=True, exist_ok=True)
        configure_script.write_text("#!/bin/sh\n")

        def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            raise subprocess.CalledProcessError(1, cmd, stderr=b"boom")

        monkeypatch.setattr(subprocess, "run", fake_run)

        static_lib, include_dir = cffi_build._build_unuran_library(unuran_dir, tmp_path / "build")

        assert static_lib is None
        assert include_dir == src_dir

    def test_setup_static_source_configures_ffi(
        self, tmp_path: Path, dummy_ffi: SimpleNamespace
    ) -> None:
        """Checks _setup_static_source wires ffi.set_source with extra_objects."""
        static_lib = tmp_path / "libunuran.a"
        static_lib.write_text("")
        include_dir = tmp_path / "include"
        include_dir.mkdir()

        cffi_build._setup_static_source(static_lib, include_dir)

        call = dummy_ffi.calls[0]
        assert call[0] == "set_source"
        assert call[1][0] == "_unuran_cffi"
        assert call[2]["extra_objects"] == [str(static_lib)]
        assert call[2]["include_dirs"] == [str(include_dir)]

    def test_setup_system_source_configures_ffi(
        self, tmp_path: Path, dummy_ffi: SimpleNamespace
    ) -> None:
        """Ensures _setup_system_source sets libraries/library_dirs/include_dirs."""
        include_dir = tmp_path / "include"
        include_dir.mkdir()

        cffi_build._setup_system_source(include_dir, "/usr/lib/libunuran.so")

        call = dummy_ffi.calls[0]
        assert call[0] == "set_source"
        assert call[2]["libraries"] == ["unuran"]
        assert call[2]["library_dirs"] == ["/usr/lib"]
        assert call[2]["include_dirs"] == [str(include_dir)]

    def test_setup_fallback_source_configures_ffi(
        self, tmp_path: Path, dummy_ffi: SimpleNamespace
    ) -> None:
        """Validates fallback source sets libraries and c99 compile flag."""
        include_dir = tmp_path / "include"
        include_dir.mkdir()

        cffi_build._setup_fallback_source(include_dir)

        call = dummy_ffi.calls[0]
        assert call[0] == "set_source"
        assert call[2]["libraries"] == ["unuran"]
        assert "-std=c99" in call[2]["extra_compile_args"]

    def test_main_prefers_static_library(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Expects main() to use static lib when available and call compile()."""
        unuran_dir = tmp_path / "vendor" / "unuran-1.11.0"
        unuran_dir.mkdir(parents=True)
        src_dir = unuran_dir / "src"
        src_dir.mkdir()
        static_lib = tmp_path / "libunuran.a"
        static_lib.write_text("")
        include_dir = tmp_path / "include"
        include_dir.mkdir()

        monkeypatch.setattr(cffi_build, "_get_unuran_paths", lambda: (unuran_dir, src_dir))
        monkeypatch.setattr(
            cffi_build, "_build_unuran_library", lambda *_: (static_lib, include_dir)
        )

        static_called = SimpleNamespace(count=0)

        def fake_setup_static(path: Path, inc: Path) -> None:
            static_called.count += 1
            assert path == static_lib
            assert inc == include_dir

        monkeypatch.setattr(cffi_build, "_setup_static_source", fake_setup_static)
        monkeypatch.setattr(
            cffi_build,
            "_setup_system_source",
            lambda *_, **__: (_ for _ in ()).throw(AssertionError),
        )
        monkeypatch.setattr(
            cffi_build,
            "_setup_fallback_source",
            lambda *_, **__: (_ for _ in ()).throw(AssertionError),
        )

        compile_called = SimpleNamespace(count=0)

        class DummyFFI(SimpleNamespace):
            def compile(self, verbose: bool = False) -> None:
                compile_called.count += 1
                assert verbose is True

        monkeypatch.setattr(cffi_build, "ffi", DummyFFI())

        cffi_build.main()

        assert static_called.count == 1
        assert compile_called.count == 1

    def test_main_falls_back_to_system_library(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Confirms main() selects system library when static lib missing."""
        unuran_dir = tmp_path / "vendor" / "unuran-1.11.0"
        unuran_dir.mkdir(parents=True)
        src_dir = unuran_dir / "src"
        src_dir.mkdir()
        include_dir = tmp_path / "include"
        include_dir.mkdir()

        monkeypatch.setattr(cffi_build, "_get_unuran_paths", lambda: (unuran_dir, src_dir))
        monkeypatch.setattr(cffi_build, "_build_unuran_library", lambda *_: (None, include_dir))

        system_called = SimpleNamespace(count=0)

        def fake_setup_system(inc: Path, system_path: str) -> None:
            system_called.count += 1
            assert inc == include_dir
            assert system_path == "/usr/lib/libunuran.so"

        monkeypatch.setattr(
            cffi_build,
            "_setup_static_source",
            lambda *_, **__: (_ for _ in ()).throw(AssertionError),
        )
        monkeypatch.setattr(cffi_build, "_setup_system_source", fake_setup_system)
        monkeypatch.setattr(
            cffi_build,
            "_setup_fallback_source",
            lambda *_, **__: (_ for _ in ()).throw(AssertionError),
        )

        monkeypatch.setattr("ctypes.util.find_library", lambda name: f"/usr/lib/lib{name}.so")

        compile_called = SimpleNamespace(count=0)

        class DummyFFI(SimpleNamespace):
            def compile(self, verbose: bool = False) -> None:
                compile_called.count += 1

        monkeypatch.setattr(cffi_build, "ffi", DummyFFI())

        cffi_build.main()

        assert system_called.count == 1
        assert compile_called.count == 1

    def test_main_uses_fallback_when_no_library_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checks final fallback path is used when neither static nor system lib exist."""
        unuran_dir = tmp_path / "vendor" / "unuran-1.11.0"
        unuran_dir.mkdir(parents=True)
        src_dir = unuran_dir / "src"
        src_dir.mkdir()
        include_dir = tmp_path / "include"
        include_dir.mkdir()

        monkeypatch.setattr(cffi_build, "_get_unuran_paths", lambda: (unuran_dir, src_dir))
        monkeypatch.setattr(cffi_build, "_build_unuran_library", lambda *_: (None, include_dir))
        monkeypatch.setattr("ctypes.util.find_library", lambda *_: None)

        fallback_called = SimpleNamespace(count=0)

        def fake_setup_fallback(inc: Path) -> None:
            fallback_called.count += 1
            assert inc == include_dir

        monkeypatch.setattr(
            cffi_build,
            "_setup_static_source",
            lambda *_, **__: (_ for _ in ()).throw(AssertionError),
        )
        monkeypatch.setattr(
            cffi_build,
            "_setup_system_source",
            lambda *_, **__: (_ for _ in ()).throw(AssertionError),
        )
        monkeypatch.setattr(cffi_build, "_setup_fallback_source", fake_setup_fallback)

        compile_called = SimpleNamespace(count=0)

        class DummyFFI(SimpleNamespace):
            def compile(self, verbose: bool = False) -> None:
                compile_called.count += 1

        monkeypatch.setattr(cffi_build, "ffi", DummyFFI())

        cffi_build.main()

        assert fallback_called.count == 1
        assert compile_called.count == 1
