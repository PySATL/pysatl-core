"""
Charging a warning to the line that asked for the fit.

A warning about a family or a sample is the caller's business, so it has to
point at the line where the fit was requested rather than at some frame inside
this package: ``simplefilter("error")``, filtering by module and
``pytest.warns`` all key off the frame a warning is charged to, and at the
default ``stacklevel`` every one of them would stop inside ``pysatl_core``.

``warnings.warn`` expresses that as a count of frames to skip, which means the
number has to be maintained by hand at every warning site and re-derived
whenever the call chain changes.  It was five in three places here, and none of
the three stayed right for long: adding one object between the estimator and
the step that warns moved all of them at once, and the count then depended on
which of two lazily computed fields happened to trigger the computation first
— an internal detail no constant should encode.

:func:`warn_at_caller` counts the frames itself instead, by walking out of the
package.  It cannot drift, because there is nothing left to keep in sync.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import inspect
import warnings
from pathlib import Path
from typing import Final

_PACKAGE_ROOT: Final[str] = str(Path(__file__).resolve().parent.parent)
"""Directory of ``pysatl_core``: a frame under it is ours, one outside is the caller's."""

_FALLBACK_STACKLEVEL: Final[int] = 2
"""Used where frames cannot be inspected at all, on a Python without them.

Two, not one, so that the warning at least leaves :func:`warn_at_caller`.
``inspect.currentframe`` is documented to return ``None`` on implementations
without Python stack frame support, and that branch has to mean something.
"""


def warn_at_caller(message: str, category: type[Warning] = UserWarning) -> None:
    """
    Warn, charging it to the nearest frame outside ``pysatl_core``.

    Parameters
    ----------
    message : str
        What to say.
    category : type[Warning], optional
        Warning class; ``UserWarning`` by default.

    Notes
    -----
    The boundary is the frame just past the **outermost** frame of this
    package, not the first frame that is not ours.  The two differ whenever
    something foreign sits in the middle of the chain, and something usually
    does: a ``functools.cached_property`` access puts a ``functools`` frame
    between two of ours, and stopping at the first alien frame would charge the
    warning to the standard library.  Scanning to the end and remembering the
    last frame that *was* ours cannot be fooled that way.

    The same rule decides what happens when this package calls code the user
    wrote — a moment rule, say — and that code calls back in: the walk passes
    over the user's rule and lands on their outermost call, which is the line
    that started the fit.  That is the line a caller can act on.
    """
    frame = inspect.currentframe()
    if frame is None:  # pragma: no cover - CPython always provides frames
        warnings.warn(message, category, stacklevel=_FALLBACK_STACKLEVEL)
        return

    depth = 1
    outermost_ours = 0
    caller = frame.f_back
    while caller is not None:
        if caller.f_code.co_filename.startswith(_PACKAGE_ROOT):
            outermost_ours = depth
        caller = caller.f_back
        depth += 1
    warnings.warn(message, category, stacklevel=outermost_ours + 2)


__all__ = ["warn_at_caller"]
