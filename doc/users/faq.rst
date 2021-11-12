.. _users-faq:

==========================
Frequently Asked Questions
==========================

.. note::

    Don't see your question here? Please `open an issue`_ on GitHub and ask
    your question there.

.. _open an issue: https://github.com/pulse2percept/pulse2percept/issues

Theoretical
===========

How are retinal coordinates mapped to visual field coordinates?
---------------------------------------------------------------

Studies often assume a linear mapping between retinal and visual field
coordinates (e.g., [Hayes2003]_, [Thompson2003]_), based on the work by
[Curcio1990]_
(see :py:class:`~pulse2percept.utils.Curcio1990Map`).

A more exact transformation is given in [Watson2014]_
(see :py:class:`~pulse2percept.utils.Watson2014Map`
and :py:class:`~pulse2percept.utils.Watson2014DisplaceMap`).

You can also write your own
:py:class:`~pulse2percept.utils.VisualFieldMap`.

In any case, note that stimulation of the inferior (superior) retina leads to
phosphenes appearing in the upper (lower) visual field.

Practical
=========

Why Python?
-----------

Python is free, well-designed, painless to read, and easy to use.
True, sometimes Python can be slow, but that is why we use `Cython`_ under the
hood, which takes execution up to C speed.
A GPU back end is planned for a future release.

.. _Cython: http://cython.org

How can I contribute to pulse2percept?
--------------------------------------

If you found a bug or want to request a feature, simply open an issue in our
`Issue Tracker`_ on GitHub. Make sure to
:ref:`label your issue appropriately <dev-contributing-issue-labels>`.

If you would like to contribute some code, great!
We appreciate all contributions, but those accepted fastest will follow a
workflow similar to the one described in our
:ref:`Contribution Guidelines <dev-contributing-workflow>`.

.. _Issue Tracker: https://github.com/pulse2percept/pulse2percept/issues

The code I downloaded does not match the documentation. What gives?
-------------------------------------------------------------------

Make sure you are reading the right version of the documentation:

*  If you installed pulse2percept :ref:`with pip <install-release>`, you are
   using the stable release, for which you can find documentation at
   `pulse2percept.readthedocs.io/en/stable`_.

*  If you installed pulse2percept from source, you are using the
   :ref:`bleeding-edge version <install-source>`, for which you can find
   documentation at `pulse2percept.readthedocs.io/en/latest`_.

*  Unfortunately, pulse2percept < 0.5 is incompatible with ReadTheDocs.
   Please refer to the :ref:`Installation Guide <install-upgrade>` for
   information on how to upgrade your code to the latest version.

.. _pulse2percept.readthedocs.io/en/stable: https://pulse2percept.readthedocs.io/en/stable/index.html
.. _pulse2percept.readthedocs.io/en/latest: https://pulse2percept.readthedocs.io/en/latest/index.html
