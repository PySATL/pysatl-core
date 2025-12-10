"""
Setup script for pysatl-core with CFFI support.

This file is needed to integrate CFFI modules with Poetry.
"""
from setuptools import setup

# CFFI modules will be compiled during installation
setup(
    cffi_modules=[
        "src/pysatl_core/stats/_unuran/bindings/_cffi_build.py:ffi",
    ],
)

