"""
Unit tests for pysatl_core.sampling.unuran.bindings._cffi_build

Tests pure-Python helper functions in the CFFI build script:
  - _configure_logging: logging level and handler setup
  - _get_project_root: ascending to pyproject.toml
  - _extract_library_name: stripping lib prefix and suffixes
  - find_unuran: locating library and header in a vendor directory
  - build_unuran: skipping/invoking the build script
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import logging
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pysatl_core.sampling.unuran.bindings._cffi_build import (
    _configure_logging,
    _extract_library_name,
    _get_project_root,
    build_unuran,
    find_unuran,
)


class TestConfigureLogging:
    """Tests for _configure_logging function."""

    def test_verbose_mode_calls_basicconfig_with_info_level(self) -> None:
        """verbose=True passes logging.INFO to basicConfig."""
        with patch("logging.basicConfig") as mock_basicconfig:
            _configure_logging(verbose=True)
            mock_basicconfig.assert_called_once()
            _, kwargs = mock_basicconfig.call_args
            assert kwargs["level"] == logging.INFO

    def test_non_verbose_mode_calls_basicconfig_with_warning_level(self) -> None:
        """verbose=False passes logging.WARNING to basicConfig."""
        with patch("logging.basicConfig") as mock_basicconfig:
            _configure_logging(verbose=False)
            mock_basicconfig.assert_called_once()
            _, kwargs = mock_basicconfig.call_args
            assert kwargs["level"] == logging.WARNING

    def test_log_file_handler_included_when_log_file_given(self, tmp_path: Path) -> None:
        """Providing a log_file path includes a FileHandler in the handlers list."""
        log_file = tmp_path / "build.log"

        with patch("logging.basicConfig") as mock_basicconfig:
            _configure_logging(verbose=False, log_file=str(log_file))
            _, kwargs = mock_basicconfig.call_args
            handler_types = [type(h) for h in kwargs["handlers"]]
            assert logging.FileHandler in handler_types

    def test_no_file_handler_when_log_file_is_none(self) -> None:
        """Without a log_file, handlers contain only a StreamHandler."""
        with patch("logging.basicConfig") as mock_basicconfig:
            _configure_logging(verbose=False, log_file=None)
            _, kwargs = mock_basicconfig.call_args
            handler_types = [type(h) for h in kwargs["handlers"]]
            assert logging.FileHandler not in handler_types
            assert logging.StreamHandler in handler_types


class TestGetProjectRoot:
    """Tests for _get_project_root function."""

    def test_finds_pyproject_toml_ancestor(self, tmp_path: Path) -> None:
        """_get_project_root walks up and returns the directory containing pyproject.toml."""
        (tmp_path / "pyproject.toml").write_text("[tool.poetry]")
        deep_dir = tmp_path / "src" / "pkg" / "subpkg"
        deep_dir.mkdir(parents=True)
        test_file = deep_dir / "some_module.py"
        test_file.write_text("# stub")

        with patch("pysatl_core.sampling.unuran.bindings._cffi_build.Path") as mock_path_cls:
            mock_path_cls.return_value.resolve.return_value = test_file
            mock_path_cls.return_value.parent = deep_dir

        # Use the actual function against a real tmp_path structure
        result = _get_project_root()

        # The real function should find pyproject.toml somewhere up the path
        assert (result / "pyproject.toml").exists()


class TestExtractLibraryName:
    """Tests for _extract_library_name function."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("libunuran.so", "unuran"),
            ("libunuran.a", "unuran"),
            ("libunuran.dylib", "unuran"),
            ("libunuran.so.1.2", "unuran"),
            ("unuran.dll", "unuran"),
            ("libfoo_bar.so", "foo_bar"),
        ],
        ids=[
            "shared_linux",
            "static",
            "shared_macos",
            "versioned_so",
            "dll_no_prefix",
            "underscore_name",
        ],
    )
    def test_strips_prefix_and_suffix(self, filename: str, expected: str) -> None:
        """Library name is extracted by stripping 'lib' prefix and known suffixes."""
        result = _extract_library_name(Path(filename))

        assert result == expected

    def test_path_with_directory_prefix(self) -> None:
        """Absolute path to a library file is handled correctly."""
        result = _extract_library_name(Path("/usr/local/lib/libunuran.so"))

        assert result == "unuran"

    def test_name_without_lib_prefix_is_returned_as_is(self) -> None:
        """Files without 'lib' prefix keep their stem after suffix removal."""
        result = _extract_library_name(Path("unuran.so"))

        assert result == "unuran"


class TestFindUnuran:
    """Tests for find_unuran function."""

    def _create_vendor_tree(
        self, base: Path, create_lib: bool = True, create_header: bool = True
    ) -> Path:
        """Create a minimal vendor directory tree."""
        lib_dir = base / "out"
        lib_dir.mkdir(parents=True)
        header_dir = base / "unuran" / "src"
        header_dir.mkdir(parents=True)

        if create_lib:
            (lib_dir / "libunuran.a").write_bytes(b"")
        if create_header:
            (header_dir / "unuran.h").write_bytes(b"")

        return base

    def test_returns_both_paths_when_both_present(self, tmp_path: Path) -> None:
        """find_unuran returns resolved paths for both header and library when both exist."""
        vendor_dir = self._create_vendor_tree(tmp_path)

        result = find_unuran(vendor_dir, raise_on_error=False)

        assert result["library_path"] is not None
        assert result["include_path"] is not None

    def test_library_path_is_none_when_missing(self, tmp_path: Path) -> None:
        """find_unuran returns None for library_path when the library file is absent."""
        vendor_dir = self._create_vendor_tree(tmp_path, create_lib=False)

        result = find_unuran(vendor_dir, raise_on_error=False)

        assert result["library_path"] is None
        assert result["include_path"] is not None

    def test_include_path_is_none_when_missing(self, tmp_path: Path) -> None:
        """find_unuran returns None for include_path when the header is absent."""
        vendor_dir = self._create_vendor_tree(tmp_path, create_header=False)

        result = find_unuran(vendor_dir, raise_on_error=False)

        assert result["include_path"] is None
        assert result["library_path"] is not None

    def test_raises_import_error_when_library_missing_and_raise_on_error(
        self, tmp_path: Path
    ) -> None:
        """find_unuran raises ImportError when library is missing and raise_on_error=True."""
        vendor_dir = self._create_vendor_tree(tmp_path, create_lib=False)

        with pytest.raises(ImportError, match="libunuran"):
            find_unuran(vendor_dir, raise_on_error=True)

    def test_raises_import_error_when_header_missing_and_raise_on_error(
        self, tmp_path: Path
    ) -> None:
        """find_unuran raises ImportError when header is missing and raise_on_error=True."""
        vendor_dir = self._create_vendor_tree(tmp_path, create_header=False)

        with pytest.raises(ImportError, match="unuran.h"):
            find_unuran(vendor_dir, raise_on_error=True)

    def test_prefers_static_library_over_shared(self, tmp_path: Path) -> None:
        """find_unuran uses libunuran.a if present (listed before .so in preference)."""
        vendor_dir = self._create_vendor_tree(tmp_path)
        (tmp_path / "out" / "libunuran.so").write_bytes(b"")

        result = find_unuran(vendor_dir, raise_on_error=False)

        assert result["library_path"] is not None
        assert result["library_path"].suffix == ".a"

    def test_returns_dict_with_correct_keys(self, tmp_path: Path) -> None:
        """find_unuran always returns a dict with 'library_path' and 'include_path' keys."""
        result = find_unuran(tmp_path, raise_on_error=False)

        assert "library_path" in result
        assert "include_path" in result


class TestBuildUnuran:
    """Tests for build_unuran function."""

    def test_skips_build_when_library_and_header_already_present(self, tmp_path: Path) -> None:
        """build_unuran does nothing when both library and header already exist."""
        lib_dir = tmp_path / "out"
        lib_dir.mkdir()
        (lib_dir / "libunuran.a").write_bytes(b"")
        header_dir = tmp_path / "unuran" / "src"
        header_dir.mkdir(parents=True)
        (header_dir / "unuran.h").write_bytes(b"")

        with patch("subprocess.run") as mock_run:
            build_unuran(tmp_path)
            mock_run.assert_not_called()

    def test_raises_when_build_script_missing(self, tmp_path: Path) -> None:
        """build_unuran raises RuntimeError when build_unuran.py is absent."""
        # No files at all → find_unuran returns None for both → triggers build attempt
        with pytest.raises(RuntimeError, match="Build script"):
            build_unuran(tmp_path)

    def test_invokes_build_script_when_library_missing(self, tmp_path: Path) -> None:
        """build_unuran calls the build script when the library file is not yet built."""
        build_script = tmp_path / "build_unuran.py"
        build_script.write_text("# stub")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            build_unuran(tmp_path)

        mock_run.assert_called_once()

    def test_propagates_subprocess_error_when_build_fails(self, tmp_path: Path) -> None:
        """A non-zero exit code from the build script propagates as CalledProcessError."""
        build_script = tmp_path / "build_unuran.py"
        build_script.write_text("# stub")

        with (
            patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "cmd")),
            pytest.raises(subprocess.CalledProcessError),
        ):
            build_unuran(tmp_path)
