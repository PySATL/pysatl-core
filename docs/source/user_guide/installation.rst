Installation
============

.. note::

   PySATL Core is currently in an early alpha stage. The public API may change
   between alpha releases.

From PyPI
---------

Install the package with:

.. code-block:: bash

   pip install pysatl-core

Linux wheels are not published yet. On Linux, ``pip`` builds the package from
the source distribution and requires a working C toolchain.

Clone the repository
--------------------

.. code-block:: bash

   git clone https://github.com/PySATL/pysatl-core.git
   cd pysatl-core
   git submodule update --init --recursive

Using Poetry
------------

For development, install runtime, dev, and documentation dependencies with:

.. code-block:: bash

   poetry install --with dev,docs

Using pip
---------

If you prefer plain pip, make sure you are on **Python 3.12+** (the project
uses PEP 695 syntax), then install the package in editable mode:

.. code-block:: bash

   pip install -e .
