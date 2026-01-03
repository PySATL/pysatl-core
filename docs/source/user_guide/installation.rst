Installation
============

.. note::

   PySATL Core is currently in an early alpha stage and is not distributed as a
   package via common managers such as pip. To try it out, you need to clone
   the repository and work with it locally.

Using Poetry
------------

If you use Poetry, install documentation dependencies with:

.. code-block:: bash

   poetry install --with docs

Using pip
---------

If you prefer plain pip, make sure you are on **Python 3.12+** (the project
uses PEP 695 syntax), then install docs requirements:

.. code-block:: bash

   pip install -e ".[docs]"
