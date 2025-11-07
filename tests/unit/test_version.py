from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import re

from pysatl_core import __version__


def test_version_pep440() -> None:
    assert re.match(
        r"^\d+!\d+(\.\d+)*([abc]|rc)?\d*(\.post\d+)?(\.dev\d+)?$|^\d+(\.\d+)*([abc]|rc)?\d*(\.post\d+)?(\.dev\d+)?$",
        __version__,
    )
