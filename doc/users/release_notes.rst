.. _users-release-notes:

=============
Release Notes
=============

.. important::

    pulse2percept 0.4.3 is the last release to support Python 2.7 and 3.4.
    pulse2percept 0.5+ require **Python 3.5 or newer**.

v0.6.0 (2020, planned)
----------------------

Highlights
~~~~~~~~~~

*   New API (:pull:`96`)
*   New models: :py:class:`~pulse2percept.models.ScoreboardModel` (:pull:`96`),
    :py:class:`~pulse2percept.models.AxonMapModel` (:pull:`96`).
*   New build process: Compile code and run tests via ``Makefile``
    (:pull:`96`).
*   Documentation now includes a tutorial, user guide, developer's guide, and
    a gallery.
*   Python 2.7 and 3.4 are no longer supported (:pull:`96`).

New features
~~~~~~~~~~~~

*   A visual prosthesis is now considered a
    :py:class:`~pulse2percept.implants.ProsthesisSystem` consisting of an
    :py:class:`~pulse2percept.implants.ElectrodeArray` and optionally a
    :py:class:`~pulse2percept.stimuli.Stimulus` (:pull:`96`).
*   A :py:class:`~pulse2percept.stimuli.Stimulus` can be created from various
    source types, such as scalars, NumPy arrays, dictionaries, and
    :py:class:`~pulse2percept.stimuli.TimeSeries` objects.
*   :py:class:`~pulse2percept.implants.ElectrodeArray` now stores electrodes in
    a dictionary (:issue:`74`).
*   :py:class:`~pulse2percept.implants.ElectrodeGrid` can be used to create
    electrodes on a rectangular (:pull:`150`) or hexagonal grid (:pull:`160`).

API changes
~~~~~~~~~~~

Backward-incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*   The ``Simulation`` object has been removed. Please directly
    :ref:`instantiate a model <topics-models>` instead.
*   ``pulse2percept.retina``: use :py:mod:`~pulse2percept.models` instead
*   ``pulse2percept.files``: use :py:mod:`~pulse2percept.io` instead

v0.5.2 (2020-02-25)
-------------------

Bug fixes
~~~~~~~~~

*   ``pulse2percept.retina.Nanduri2012``: improved Cython implementation

v0.5.1 (2020-02-05)
-------------------

Bug fixes
~~~~~~~~~

*   ``pulse2percept.retina.Nanduri2012``: allow switch between FFT/Cython
*   ``pulse2percept.retina.Horsager2009``: respect ``use_jit`` option
*   ``pulse2percept.utils.center_vector``: "cannot determine Numba type"

v0.5.0 (2019-11-29)
-------------------

Highlights
~~~~~~~~~~

*   New :py:mod:`pulse2percept.viz` module (:pull:`84`)
*   Support for the :py:class:`~pulse2percept.implants.AlphaIMS` implant
    (:pull:`87`)
*   Automated wheelhouse build (:pull:`130`)
*   New contribution guidelines (:pull:`92`)
*   New issue templates (:pull:`93`)
*   New code of conduct (:pull:`95`)
*   Host documentation on
    `pulse2percept.readthedocs.io <https://pulse2percept.readthedocs.io>`_.

Bug fixes
~~~~~~~~~

*   Fix nasal/temporal labeling for left eyes (:commit:`9c3bddc`)
*   Fix :py:meth:`~pulse2percept.viz.plot_fundus` for left eyes
    (:commit:`a6ffdbc`)
*   Fix ``scipy.special.factorial`` (:commit:`c9631ae`)

v0.4.3 (2018-05-21)
-------------------

Highlights
~~~~~~~~~~

*   Cython integration:

    * The model described in Nanduri et al. (2012) now uses a finite difference
      method implemented in Cython as opposed to FFT-based convolutions
      (:pull:`83`)

    * Single-core benchmarks show a 200x speedup over a pure-Python
      implementation.

Bug fixes
~~~~~~~~~

*   Python 2.7 unpacking error in :py:meth:`~pulse2percept.viz.plot_fundus`
    (:commit:`3dd9d1e`)

.. _0.4.3-deprecation-removals:

Deprecation removals
~~~~~~~~~~~~~~~~~~~~

* ``pulse2percept.files.savemoviefiles``
* ``pulse2percept.files.npy2movie``
* ``pulse2percept.files.scale``
* ``pulse2percept.stimuli.Movie2Pulsetrain``
* ``pulse2percept.stimuli.retinalmovie2electrodtimeseries``
* ``pulse2percept.utils.Parameters``
* ``pulse2percept.utils.mov2npy``

v0.3.0 (2018-02-20)
-------------------

Highlights
~~~~~~~~~~

*   New, faster axon map calculation
*   Better plotting
*   Support for left/right eye
