.. _users-faq:

==========================
Frequently Asked Questions
==========================

.. note ::

    Don't see your question here? Please `open an issue`_ on GitHub and ask
    your question there.

Theoretical
===========

How are retinal coordinates mapped to visual field coordinates?
---------------------------------------------------------------

Studies often assume a linear mapping between retinal and visual field
coordinates (e.g., [Hayes2003]_, [Thompson2003]_).
A more exact transformation is given in [Watson2014]_, which corresponding code
in the :py:mod:`~pulse2percept.models.watson2014` submodule.
In any case, note that stimulation of the inferior (superior) retina leads to
phosphenes appearing in the upper (lower) visual field.
This is why visualization functions such as
:py:meth:`~pulse2percept.viz.plot_fundus` provide an option to flip the image
upside down.

Practical
=========

Why Python?
-----------

Python is free, well-designed, painless to read, and easy to use.
True, sometimes Python can be slow, but that is why we use `Cython`_ under the
hood, which takes execution up to C speed.

How can I contribute to pulse2percept?
--------------------------------------

We appreciate all contributions to pulse2percept, but those accepted fastest
will follow a workflow similar to the one described in our
:ref:`Contribution Guidelines <dev-contributing>`.

.. _open an issue: https://github.com/uwescience/pulse2percept/issues
.. _Cython: http://cython.org
