.. _dev-style-guide:

===========
Style Guide
===========

pulse2percept follows `PEP 8`_ with a small number of project-specific
conventions. The repository's lint configuration is authoritative.

General Conventions
-------------------

* Use four spaces for indentation.
* Keep lines within the configured line-length limit.
* Use two blank lines around top-level functions and classes.
* Use blank lines within functions only to separate logical sections.
* Follow standard spacing for keyword arguments:

  .. code-block:: python

      def complex(real, image=0.0):
          return magic(r=real, i=image)

Imports
-------

Use the conventional package aliases used throughout the codebase:

.. code-block:: python

    import numpy as np
    import numpy.testing as npt
    import scipy as sp
    import pandas as pd

    import pulse2percept as p2p

Line Breaks Around Operators
----------------------------

pulse2percept breaks long expressions **after** binary operators:

.. code-block:: python

    income = (gross_wages +
              taxable_interest -
              student_loan_interest)

This differs from the current PEP 8 preference, but is retained for consistency
with the existing codebase. In flake8 terminology, avoid W503-style line
breaks.

.. _PEP 8: https://peps.python.org/pep-0008/