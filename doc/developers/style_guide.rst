.. _dev-style_guide:

==================
Coding style guide
==================

The main principles behind pulse2percept development are:

-  **Robustness**: The results of a piece of code must be verified
   systematically, enhancing stability and reducing redundancies.

-  **Readability**: The code is read much more frequently than it is written,
   and should be easy to understand by other developers.
   Documentation is essential.

-  **Consistency**: Following these guidelines will ease reading the code and
   will make it less error-prone.

Coding style
============

pulse2percept uses the standard Python `PEP8`_ style guide to ensure
readibility and consistency of the code base.

.. note::

    There are `tools <https://pypi.org/project/pep8>`_ that allow you to
    automatically check your Python code against `PEP8`_ style conventions.

    For example, if you work with Sublime, you can use the `PEP8 Autoformat`_
    package. For convenience, make sure the setting ``autoformat_on_save`` is
    set to ``true``.

A few general guidelines:

-  Use 4 spaces instead of tab.
-  Limit all lines to a maximum of 79 characters.
-  Write code for Python 3.5+.

.. _PEP8: https://www.python.org/dev/peps/pep-0008
.. _PEP8 Autoformat: https://packagecontrol.io/packages/Python%20PEP8%20Autoformat

Imports
-------

pulse2percept recommends using the following package shorthands to increase
consistency and readibility across the library:

.. code-block:: python

    import numpy as np
    import numpy.testing as npt
    import scipy as sp
    import pandas as pd

    import pulse2percept as p2p

Whitespace and blank lines
--------------------------

Don't use spaces around the ``=`` sign to indicate a keyword argument:

.. code-block:: python

    # Good:
    def complex(real, image=0.0):
        return magic(r=real, i=imag)

    # Bad:
    def complex(real, image = 0.0):
        return magic(r = real, i = imag)

           
Surround top-level function and class definitions with two blank lines.
Use blank lines in functions, sparingly, to indicate logical sections.

Breaking long lines (W503 vs W504)
----------------------------------

The recommended style is to break a line after binary operators:

.. code-block:: python

    # Good:
    income = (gross_wages +
              taxable_interest -
              student_loan_interest)

    # Bad:
    income = (gross_wages
              + taxable_interest
              - student_loan_interest)

Yes, we are aware of the ongoing `W503 vs W504 discussion <https://www.python.org/dev/peps/pep-0008/#should-a-line-break-before-or-after-a-binary-operator>`_.
W504 would improve readibility.
However, for now, we want to ensure consistency with the existing code.

Class layout (slots, properties)
--------------------------------

Classes can have ``__slots__`` that list all class attributes.
This can have several advantages:

-  Users cannot add new class attributes on-the-fly. This is desirable for
   models (e.g., ``AxonMapModel``), which has a predefined number of
   attributes.
-  Slots reduce memory usage. This is desirable for stimuli (e.g.,
   ``TimeSeries``), for which a large number of instances are created.

Subclasses that inherit from classes with ``__slots__`` must themselves
implement slots. The top-level class must inherit from ``object``, and
attributes cannot be listed more than once:

.. code-block:: python

    class Vehicle(object):

        __slots__ = ('owner')

        def __init__(self, owner):
            self.owner = owner


    class Car(Vehicle):

        __slots__ = ('n_doors')

        def __init__(self, owner, n_doors):
            self.owner = owner
            self.n_doors = n_doors

If you did it right, then neither ``Vehicle`` nor ``Car`` should have a
``__dict__`` attribute:

.. code-block:: python

	car = Car('myself', 4)
	assert hasattr(car, '__slots__')
    assert not hasattr(car, '__dict__')

*This guide was inspired by the `DIPY style guide <https://dipy.org/documentation/1.1.1./devel/coding_style_guideline>`_.*