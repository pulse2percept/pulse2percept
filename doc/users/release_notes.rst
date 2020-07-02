.. _users-release-notes:

=============
Release Notes
=============

.. important::

    pulse2percept 0.4.3 is the last release to support Python 2.7 and 3.4.
    pulse2percept 0.5+ requires **Python 3.5 or newer**.

v0.7.0 (2020, planned)
----------------------

Highlights
~~~~~~~~~~

*  New implants: :py:class:`~pulse2percept.implants.PRIMA`, 
   :py:class:`~pulse2percept.implants.PRIMA75`,
   :py:class:`~pulse2percept.implants.PRIMA55`, 
   :py:class:`~pulse2percept.implants.PRIMA40` (:pull:`188`)
*  New electrodes: :py:class:`~pulse2percept.implants.SquareElectrode`,
   :py:class:`~pulse2percept.implants.HexElectrode`,
   :py:class:`~pulse2percept.implants.PhotovoltaicPixel` (:pull:`188`, 
   :pull:`193`)
*  New stimuli: :py:class:`~pulse2percept.stimuli.ImageStimulus` and
   :py:class:`~pulse2percept.stimuli.VideoStimulus` (:pull:`196`, :pull:`220`,
   :pull:`221`)
*  Computational cost and memory usage of
   :py:class:`~pulse2percept.models.AxonMapModel` have been drastically reduced
   (:pull:`215`)

New features
~~~~~~~~~~~~

*  Percepts can be animated directly in IPython / Jupyter Notebook, and saved
   as a movie file (:pull:`196`, :pull:`226`)
*  Electrodes, electrode arrays, and prosthesis systems now have their own
   plot method (:pull:`188`, :pull:`195`, :pull:`222`)

API changes
~~~~~~~~~~~

Backward-incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*  pulse2percept now requires Matplotlib 3.0.2 or newer (:pull:`223`)
*  FFMPEG and scikit-video dependencies have been removed (:pull:`196`)
*  ``TimeSeries`` has been removed. Please use
   :py:class:`~pulse2percept.stimuli.Stimulus` instead
*  ``LegacyMonophasicPulse``, ``LegacyBiphasicPulse`` and ``LegacyPulseTrain``
   have been removed. Use their equivalents without the "Legacy" prefix.

Deprecations
^^^^^^^^^^^^

*  ``plot_axon_map``: Use :py:meth`pulse2percept.models.AxonMapModel.plot`
*  ``plot_implant_on_axon_map``: Use
   :py:meth:`pulse2percept.implants.ProsthesisSystem.plot` on top of
   :py:meth`pulse2percept.models.AxonMapModel.plot`

Bug fixes
~~~~~~~~~

*  :py:class:`~pulse2percept.utils.Grid2D`: Grid now produces correct step size
   even when range is not divisible by step (:pull:`201`)
*  :py:class:`~pulse2percept.implants.AlphaIMS`: Implant now uses
   :py:class:`~pulse2percept.implants.SquareElectrode` objects and has exactly
   1500 electrodes (:pull:`193`)
*  :py:class:`~pulse2percept.implants.ElectrodeGrid`: Alphabetic names now
   follow A-Z, AA-AZ, BA-BZ, etc. (:pull:`192`)
*  :py:class:`~pulse2percept.implants.BVA24`: Setting a stimulus in the
   constructor now has the desired effect (:pull:`186`)

v0.6.0 (2020-05-05)
----------------------

Highlights
~~~~~~~~~~

*   New API (:pull:`96`, :pull:`174`, :pull:`178`)
*   New implants: :py:class:`~pulse2percept.implants.BVA24` (:pull:`161`)
*   New models: :py:class:`~pulse2percept.models.ScoreboardModel` (:pull:`96`),
    :py:class:`~pulse2percept.models.AxonMapModel` (:pull:`96`),
    :py:class:`~pulse2percept.models.Nanduri2012Model` (:pull:`168`),
    :py:class:`~pulse2percept.models.Horsager2009Model` (:pull:`180`)
*   New stimuli: :py:class:`~pulse2percept.stimuli.BiphasicPulseTrain`,
    :py:class:`~pulse2percept.stimuli.AsymmetricBiphasicPulse`,
    :py:class:`~pulse2percept.stimuli.AsymmetricBiphasicPulseTrain`
    (:pull:`178`)
*   New :py:mod:`~pulse2percept.percepts` subpackage (:pull:`174`)
*   New :py:mod:`~pulse2percept.datasets` subpackage (:pull:`167`)
*   New build process: Compile code and run tests via ``Makefile``
    (:pull:`96`)
*   Documentation now includes a tutorial, user guide, developer's guide, and
    a gallery
*   Python 2.7 and 3.4 are no longer supported (:pull:`96`)

New features
~~~~~~~~~~~~

*   A visual prosthesis is now considered a
    :py:class:`~pulse2percept.implants.ProsthesisSystem` consisting of an
    :py:class:`~pulse2percept.implants.ElectrodeArray` and optionally a
    :py:class:`~pulse2percept.stimuli.Stimulus` (:pull:`96`).
*   A :py:class:`~pulse2percept.models.Model` can be built by mix-and-matching
    spatial and temporal models from different publications (:pull:`174`).
*   A :py:class:`~pulse2percept.stimuli.Stimulus` can be created from various
    source types, such as scalars, NumPy arrays, lists, and dictionaries.
    There are also a variety of built-in pulses and pulse trains
    (e.g., :py:class:`~pulse2percept.stimuli.BiphasicPulseTrain`).
*   :py:class:`~pulse2percept.implants.ElectrodeArray` now stores electrodes in
    a dictionary (:issue:`74`).
*   :py:class:`~pulse2percept.implants.ElectrodeGrid` can be used to create
    electrodes on a rectangular (:pull:`150`) or hexagonal grid (:pull:`160`).

API changes
~~~~~~~~~~~

Backward-incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*  Times are now specified in milliseconds.
*  The ``Simulation`` object has been removed. Please directly
   :ref:`instantiate a model <topics-models>` instead.
*  ``pulse2percept.retina``: use :py:mod:`~pulse2percept.models` instead
*  ``pulse2percept.files``: use :py:mod:`~pulse2percept.io` instead

Deprecations
^^^^^^^^^^^^

*  ``TimeSeries``: use :py:class:`~pulse2percept.stimuli.Stimulus` instead
*  Old pulses got renamed to ``LegacyMonophasicPulse``, ``LegacyBiphasicPulse``
   and ``LegacyPulseTrain``

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
