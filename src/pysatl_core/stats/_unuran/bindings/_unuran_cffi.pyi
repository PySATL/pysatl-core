from typing import Any

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

# TODO problems with 'from cffi import FFI' (mypy)
ffi: Any
lib: Any
__all__ = ["ffi", "lib"]
