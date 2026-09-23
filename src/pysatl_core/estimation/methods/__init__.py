"""
The estimation methods this package implements.

One directory — or, for a small method, one module — per method, and nothing
here is shared between them.  Everything a method builds on lives one level up:
:mod:`~pysatl_core.estimation.problem` (the family and the sample),
:mod:`~pysatl_core.estimation.parameters` (the parameter space) and
:mod:`~pysatl_core.estimation.optimizers` (the search).  A method that finds
itself importing from a sibling is a sign the shared part was put in the wrong
place, not a reason to import it.

Kept apart from those three so that the directory listing of ``estimation``
answers one question rather than two: what methods exist, and separately what
they are built from.

A method starts life as a single module here and becomes a package when it
grows a second subject of its own — the policy of the fit, the criterion it
minimises, and the result it reports being the usual three.
"""

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_core.estimation.methods.mle import MLE, MLEResult

__all__ = ["MLE", "MLEResult"]
