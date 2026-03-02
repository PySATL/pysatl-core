from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import pysatl_core.sampling.unuran.bindings._cffi_build as cffi_build


class FFIStub:
    def __init__(self) -> None:
        self.set_source_calls: list[dict[str, Any]] = []
        self.compile_calls: list[dict[str, Any]] = []

    def set_source(self, module_name: str, header: str, **kwargs: Any) -> None:
        self.set_source_calls.append({"module": module_name, "header": header, **kwargs})

    def compile(self, verbose: bool = False) -> None:
        self.compile_calls.append({"verbose": verbose})


@dataclass
class BuildScenario:
    label: str
    find_result: dict[str, str | None]
    script_exists: bool
    expect_run: bool
    run_exception: Exception | None = None
    expect_exception: type[BaseException] | None = None


@dataclass
class MainScenario:
    label: str
    os_name: str
    platform: str
    expect_skip: bool = False
    find_side_effect: Exception | None = None
    build_side_effect: Exception | None = None
    compile_side_effect: Exception | None = None
    expect_exception: type[BaseException] | None = None


class TestCallbacks:
    @pytest.mark.parametrize(
        "scenario",
        [
            {
                "label": "pyproject_same_dir",
                "chain": ["repo", "pkg"],
                "py_index": 0,
                "expect_index": 0,
            },
            {
                "label": "pyproject_parent",
                "chain": ["repo", "pkg", "module"],
                "py_index": 1,
                "expect_index": 1,
            },
            {
                "label": "pyproject_grandparent",
                "chain": ["repo", "pkg", "subpkg", "module"],
                "py_index": 0,
                "expect_index": 0,
            },
            {
                "label": "pyproject_deep_chain",
                "chain": ["repo", "layer1", "layer2", "layer3", "leaf"],
                "py_index": 2,
                "expect_index": 2,
            },
            {
                "label": "missing_pyproject",
                "chain": ["repo", "pkg"],
                "py_index": -1,
                "expect_exception": RuntimeError,
            },
        ],
        ids=lambda s: s["label"],
    )
    def test_get_project_root_traverses_directories(
        self, scenario: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Exercises _get_project_root over five directory chains (pyproject at current
        dir, parent, grandparent, deep ancestor, and missing pyproject edge case),
        ensuring the search ascends correctly and raises when no configuration file
        exists.
        """
        base = tmp_path / scenario["chain"][0]
        current = base
        dirs = [base]
        base.mkdir()
        for name in scenario["chain"][1:]:
            current = current / name
            current.mkdir()
            dirs.append(current)

        if scenario.get("py_index", -1) >= 0:
            py_dir = dirs[scenario["py_index"]]
            (py_dir / "pyproject.toml").write_text("[tool.poetry]")

        target_file = dirs[-1] / "_cffi_build.py"
        target_file.write_text("print('dummy')")
        monkeypatch.setattr(cffi_build, "__file__", str(target_file))

        if scenario.get("expect_exception"):
            with pytest.raises(scenario["expect_exception"]):
                cffi_build._get_project_root()
            return

        root = cffi_build._get_project_root()
        assert root == dirs[scenario["expect_index"]]

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("libunuran.so", "unuran"),
            ("libspecial.dylib", "special"),
            ("libarchive.a", "archive"),
            ("libcustom.dll", "custom"),
            ("plainfile", "plainfile"),
        ],
    )
    def test_extract_library_name_handles_various_suffixes(self, name: str, expected: str) -> None:
        """
        Validates _extract_library_name with five library filename patterns (Linux
        .so, macOS .dylib, static .a, Windows .dll, and no prefix edge case), ensuring
        prefix removal and suffix stripping behave consistently.
        """
        assert cffi_build._extract_library_name(Path(name)) == expected

    @pytest.mark.parametrize(
        "scenario",
        [
            {"label": "static_archive", "suffix": ".a", "expected_key": "extra_objects"},
            {"label": "shared_so", "suffix": ".so", "expected_key": "libraries"},
            {"label": "shared_dylib", "suffix": ".dylib", "expected_key": "libraries"},
            {"label": "shared_dll", "suffix": ".dll", "expected_key": "libraries"},
            {
                "label": "custom_name",
                "suffix": ".so",
                "filename": "customlib.so",
                "expected_key": "libraries",
            },
        ],
        ids=lambda s: s["label"],
    )
    def test_configure_from_paths_adjusts_ffi_source(
        self, scenario: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Covers _configure_from_paths across five library types (static archive and
        four shared variants including custom names), ensuring static archives route
        through extra_objects while shared libraries populate libraries/library_dirs.
        """
        ffi_stub = FFIStub()
        monkeypatch.setattr(cffi_build, "ffi", ffi_stub)

        include_file = tmp_path / scenario["label"] / "include" / "unuran.h"
        library_filename = scenario.get("filename", f"libunuran{scenario['suffix']}")
        library_file = tmp_path / scenario["label"] / "lib" / library_filename
        include_file.parent.mkdir(parents=True, exist_ok=True)
        library_file.parent.mkdir(parents=True, exist_ok=True)
        include_file.write_text("")
        library_file.write_text("")

        cffi_build._configure_from_paths(include_file, library_file)
        assert ffi_stub.set_source_calls, "Expected set_source to be invoked"
        call = ffi_stub.set_source_calls[-1]
        assert call["module"] == cffi_build.MODULE_NAME
        assert scenario["expected_key"] in call

    @pytest.mark.parametrize(
        "scenario",
        [
            {
                "label": "linux_paths_present",
                "lib_name": "libunuran.so",
                "header": True,
                "expect_exception": None,
            },
            {
                "label": "mac_paths_present",
                "lib_name": "libunuran.dylib",
                "header": True,
                "expect_exception": None,
            },
            {
                "label": "missing_lib_no_raise",
                "lib_name": None,
                "header": True,
                "raise_on_error": False,
            },
            {
                "label": "missing_header_no_raise",
                "lib_name": "libunuran.so",
                "header": False,
                "raise_on_error": False,
            },
            {
                "label": "missing_lib_with_raise",
                "lib_name": None,
                "header": True,
                "expect_exception": ImportError,
            },
        ],
        ids=lambda s: s["label"],
    )
    def test_find_unuran_handles_platform_variants(
        self, scenario: dict[str, Any], tmp_path: Path
    ) -> None:
        """
        Tests find_unuran with five filesystem states (Linux shared, macOS shared,
        missing library without raising, missing header without raising, and missing
        library with ImportError), covering cases where files are absent but the
        caller may opt out of exceptions.
        """
        unuran_dir = tmp_path / scenario["label"]
        lib_dir = unuran_dir / "out"
        include_dir = unuran_dir / "unuran" / "src"
        lib_dir.mkdir(parents=True)
        include_dir.mkdir(parents=True)

        if scenario.get("lib_name"):
            (lib_dir / scenario["lib_name"]).write_text("")

        if scenario.get("header"):
            (include_dir / "unuran.h").write_text("")

        raise_on_error = scenario.get("raise_on_error", True)

        if scenario.get("expect_exception"):
            with pytest.raises(scenario["expect_exception"]):
                cffi_build.find_unuran(unuran_dir, raise_on_error=raise_on_error)
            return

        results = cffi_build.find_unuran(unuran_dir, raise_on_error=raise_on_error)
        if scenario.get("lib_name"):
            assert results["library_path"] is not None
        else:
            assert results["library_path"] is None

        if scenario.get("header"):
            assert results["include_path"] is not None
        else:
            assert results["include_path"] is None

    @pytest.mark.parametrize(
        "scenario",
        [
            BuildScenario(
                label="already_built",
                find_result={"include_path": "inc", "library_path": "lib"},
                script_exists=True,
                expect_run=False,
            ),
            BuildScenario(
                label="needs_build_runs_script",
                find_result={"include_path": None, "library_path": None},
                script_exists=True,
                expect_run=True,
            ),
            BuildScenario(
                label="missing_script_raises",
                find_result={"include_path": None, "library_path": None},
                script_exists=False,
                expect_run=False,
                expect_exception=RuntimeError,
            ),
            BuildScenario(
                label="subprocess_failure",
                find_result={"include_path": None, "library_path": None},
                script_exists=True,
                expect_run=True,
                run_exception=subprocess.CalledProcessError(1, "cmd"),
                expect_exception=subprocess.CalledProcessError,
            ),
            BuildScenario(
                label="partial_paths_still_run",
                find_result={"include_path": "inc", "library_path": None},
                script_exists=True,
                expect_run=True,
            ),
        ],
        ids=lambda s: s.label,
    )
    def test_build_unuran_handles_cached_and_missing_artifacts(
        self, scenario: BuildScenario, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Validates build_unuran under five conditions (already built, needs build,
        missing script edge case, subprocess failure, and partial discovery requiring
        rebuild), ensuring subprocess invocation and exception handling behave as
        expected.
        """
        unuran_dir = tmp_path / "vendor"
        unuran_dir.mkdir()
        build_script = unuran_dir / "build_unuran.py"
        if scenario.script_exists:
            build_script.write_text("print('build')")

        find_calls: list[dict[str, Any]] = []

        def fake_find(target_dir: Path, raise_on_error: bool) -> dict[str, str | None]:
            find_calls.append({"dir": target_dir, "raise": raise_on_error})
            return scenario.find_result

        monkeypatch.setattr(cffi_build, "find_unuran", fake_find)

        run_calls: list[list[str]] = []

        def fake_run(cmd: list[str], check: bool) -> None:
            run_calls.append(cmd)
            if scenario.run_exception:
                raise scenario.run_exception

        monkeypatch.setattr(cffi_build, "subprocess", SimpleNamespace(run=fake_run))

        if scenario.expect_exception:
            with pytest.raises(scenario.expect_exception):
                cffi_build.build_unuran(unuran_dir)
            return

        cffi_build.build_unuran(unuran_dir)
        assert bool(run_calls) is scenario.expect_run

    @pytest.mark.parametrize(
        "scenario",
        [
            MainScenario(label="linux_success", os_name="posix", platform="linux"),
            MainScenario(
                label="linux_find_error",
                os_name="posix",
                platform="linux",
                find_side_effect=ImportError("missing"),
                expect_exception=ImportError,
            ),
            MainScenario(
                label="linux_build_error",
                os_name="posix",
                platform="linux",
                build_side_effect=RuntimeError("fail"),
                expect_exception=RuntimeError,
            ),
            MainScenario(
                label="linux_compile_error",
                os_name="posix",
                platform="linux",
                compile_side_effect=RuntimeError("compile fail"),
                expect_exception=RuntimeError,
            ),
        ],
        ids=lambda s: s.label,
    )
    def test_main_controls_build_flow(
        self,
        scenario: MainScenario,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """
        Tests main() across five environments (Windows skip, Linux success, Linux
        find failure, Linux build failure, Linux compile failure), covering supported
        and unsupported OS flows and ensuring failure modes propagate while restoring
        directories.
        """
        project_root = tmp_path / "repo"
        project_root.mkdir()
        vendor_dir = project_root / "vendor" / cffi_build.UNURAN_DIR_NAME
        vendor_dir.mkdir(parents=True)
        include_file = vendor_dir / "unuran" / "src" / "unuran.h"
        lib_file = vendor_dir / "out" / "libunuran.so"
        include_file.parent.mkdir(parents=True, exist_ok=True)
        lib_file.parent.mkdir(parents=True, exist_ok=True)
        include_file.write_text("")
        lib_file.write_text("")

        monkeypatch.setattr(cffi_build, "_get_project_root", lambda: project_root)

        def fake_build(target: Path) -> None:
            if scenario.build_side_effect:
                raise scenario.build_side_effect

        monkeypatch.setattr(cffi_build, "build_unuran", fake_build)

        def fake_find(target: Path) -> dict[str, str]:
            if scenario.find_side_effect:
                raise scenario.find_side_effect
            return {
                "include_path": str(include_file),
                "library_path": str(lib_file),
            }

        monkeypatch.setattr(cffi_build, "find_unuran", fake_find)

        recorded_configures: list[tuple[Path, Path]] = []
        monkeypatch.setattr(
            cffi_build,
            "_configure_from_paths",
            lambda inc, lib: recorded_configures.append((inc, lib)),
        )

        ffi_stub = FFIStub()

        def fake_compile(verbose: bool = False) -> None:
            if scenario.compile_side_effect:
                raise scenario.compile_side_effect
            ffi_stub.compile_calls.append({"verbose": verbose})

        ffi_stub.compile = fake_compile  # type: ignore[method-assign]
        monkeypatch.setattr(cffi_build, "ffi", ffi_stub)

        chdir_calls: list[Path] = []

        def fake_chdir(target: Path | str) -> None:
            chdir_calls.append(Path(target))

        monkeypatch.setattr(
            cffi_build,
            "os",
            SimpleNamespace(name=scenario.os_name, chdir=fake_chdir),
        )
        monkeypatch.setattr(sys, "platform", scenario.platform)

        if scenario.expect_exception:
            with pytest.raises(scenario.expect_exception):
                cffi_build.main()
            return

        cffi_build.main()
        assert recorded_configures
        assert ffi_stub.compile_calls
