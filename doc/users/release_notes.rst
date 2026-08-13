.. _users-release-notes:

=============
Release Notes
=============

v0.10.0 Encoders (unreleased)
-----------------------------

Highlights:

*  New :py:class:`~pulse2percept.stimuli.Encoder` classes translate images and
   videos into electrical stimulation. Amplitude and frequency modulation are
   supported, including stimulator timing, input resolution, and electrode
   multiplexing. Pulse timing is independent of the video frame rate: a frame
   says when the modulation parameters change, and the requested ``freq`` is
   the rate actually delivered.

*  New :py:class:`~pulse2percept.implants.Raster` classes describe how
   stimulators multiplex electrodes that cannot be driven simultaneously. Each
   raster group gets a slot in a repeating cycle and pulse periods are whole
   multiples of that cycle, so two groups provably never pulse at the same
   instant and the instantaneous current limit holds by construction.

*  :py:class:`~pulse2percept.models.FadingTemporal` now half-wave rectifies its
   drive, so it can see a charge-balanced pulse. Previously the anodic phase of
   a biphasic pulse undid exactly what the cathodic phase had done, and a pulse
   train produced sub-millisecond transients around zero rather than a percept
   that persisted between pulses.

* :py:meth:`~pulse2percept.percepts.Percept.play` and
  :py:meth:`~pulse2percept.stimuli.VideoStimulus.play` are roughly 100x faster
  and produce much smaller notebooks and documentation pages.

* Python 3.14 is now supported. Python 3.11 and NumPy 2 are now required.

API changes:

* :py:class:`~pulse2percept.models.FadingTemporal` is now driven by
  :math:`\max(-A, 0)` rather than :math:`-A`: anodic current no longer reduces
  brightness, it is ignored. A stimulus that is purely cathodic is unaffected;
  a charge-balanced one now delivers net charge instead of integrating to
  nothing. At 20 Hz and ``tau=100``, brightness between pulses used to fall to
  0.6% of the peak it had just reached (99% ripple) and now holds at 61%.
  Raising ``tau`` used to make the percept dimmer rather than steadier, because
  the peak scales as :math:`1/\tau` while the floor between pulses does not.

* Temporal models gained a ``reduce`` parameter. When
  ``predict_percept`` picks the output times itself (``t_percept=None``), each
  point now reports the peak brightness reached over the interval leading up to
  it rather than the brightness at the instant it ends. Naming ``t_percept``
  still asks for those instants. Pass ``reduce='last'`` for the old reporting.

  Electrical stimulation is pulsatile, so an instant sampled from a percept is
  usually an instant between pulses, and which pulses a frame catches drifts
  when the frame rate and the pulse rate are incommensurate. Under a raster
  that showed up as groups appearing in the wrong order or not at all: driving
  Argus II with ``SequentialRaster(6)`` at 20 Hz against a 29.97 fps video,
  25 of 94 percept frames came out entirely dark and two of the six rows never
  appeared at all. :py:class:`~pulse2percept.models.FadingTemporal` tracks the
  peak inside its integrator, so it is exact at any output rate.

* :py:meth:`~pulse2percept.stimuli.ImageStimulus.encode` and
  :py:meth:`~pulse2percept.stimuli.VideoStimulus.encode` now use
  :py:class:`~pulse2percept.stimuli.AmplitudeEncoder`. Most importantly, gray
  levels map to ``amp_range`` absolutely rather than being stretched to fill
  it, and each frame now produces a pulse train rather than a single pulse.
  Pass ``stretch=True`` for the old gray-level mapping.

* Encoders can operate directly at implant resolution and account for electrode
  multiplexing through the implant's raster.

* :py:class:`~pulse2percept.implants.ProsthesisSystem` now exposes ``raster``
  and ``max_current``.

* ``play`` gained a ``fmt`` argument.
  :py:meth:`~pulse2percept.stimuli.VideoStimulus.play` defaults to JPEG, which
  is substantially smaller and faster to build for color video;
  :py:meth:`~pulse2percept.percepts.Percept.play` defaults to PNG, which is
  pixel-exact and nearly as compact for scalar data. Pass ``fmt`` to override.

* Minimum dependency versions were raised for NumPy 2 compatibility. NumPy 1.x
  users should remain on v0.9.1.

## Bug fixes

* :py:class:`~pulse2percept.stimuli.PulseTrain` no longer ends on partial,
  unbalanced pulses when its frequency does not divide ``stim_dur``.

* Stimulus time axes now use float64 precision, fixing false
  :py:attr:`~pulse2percept.stimuli.Stimulus.is_charge_balanced` failures for
  longer pulse trains and improving frequency-modulation accuracy.

* :py:class:`~pulse2percept.implants.EnsembleImplant` now merges nearly
  identical time points using the same tolerance as
  :py:class:`~pulse2percept.stimuli.Stimulus`.

* Visual-field-map equality now handles array-valued attributes correctly,
  maps are hashable again, and maps of different classes no longer compare
  equal.

* Temporal models now warn when stimulus polarity produces an all-zero percept
  instead of silently returning a blank result.

* :py:meth:`~pulse2percept.stimuli.ImageStimulus.encode` and
  :py:meth:`~pulse2percept.stimuli.VideoStimulus.encode` no longer modify their
  inputs in place.

* :py:meth:`~pulse2percept.percepts.Percept.play`,
  :py:meth:`~pulse2percept.stimuli.VideoStimulus.play`, and
  :py:meth:`~pulse2percept.percepts.Percept.save` now handle single-frame
  inputs correctly and give a useful error for nonuniform time axes.

* :py:meth:`~pulse2percept.stimuli.VideoStimulus.encode` now correctly infers
  frame duration from the time axis when frame-rate metadata is absent.

* :py:attr:`~pulse2percept.stimuli.VideoStimulus.vid_shape` now reports the
  number of frames the stimulus actually has, rather than the number the source
  video had before ``compress=True`` dropped the redundant time points. This
  fixes :py:meth:`~pulse2percept.stimuli.VideoStimulus.play` and every other
  operation that reshapes ``data`` back into frames. Playing a video that was
  compressed in *space* raises an explanatory error instead of a reshape error.

* A four-channel :py:class:`~pulse2percept.stimuli.VideoStimulus` is played
  back as RGBA, as documented: the alpha channel is preserved for ``fmt='png'``
  and composited onto the axes background for ``fmt='jpg'``, which cannot carry
  it. Previously the four channels were reinterpreted as RGB, which sheared the
  color channels across every row.

* The per-frame title in the HTML player no longer piles up on itself or on a
  title that was already on the axes.

v0.9.1 (2026-08-06)
-------------------

Highlights:

*  Python 3.13 support; the minimum supported Python is now 3.10 (:pull:`649`)
*  NumPy 2.x support: wheels import under both NumPy 1.x and 2.x, so installing
   pulse2percept no longer downgrades NumPy in environments that ship it, such
   as Google Colab (:pull:`635`, :pull:`736`)
*  Removed the jax engine and ``predict_percept_batched`` from
   :py:class:`~pulse2percept.models.BiphasicAxonMapModel`; the ``engine``
   argument of the effect models and the ``pad`` argument of
   :py:meth:`~pulse2percept.models.AxonMapSpatial.calc_axon_sensitivity` are
   deprecated, and will be removed in v0.10.0 (:pull:`788`)
*  Removed the ``model_selection`` module (:pull:`685`) and support for the
   joblib and dask parallel backends (:pull:`686`); the ``engine`` and
   ``scheduler`` model parameters are deprecated, and will be removed in
   v0.10.0 (:pull:`788`)
*  ``n_jobs`` is now an alias for ``n_threads``: either name sets the number
   of OpenMP threads, and ``None`` or ``-1`` uses every core (:pull:`788`)
*  More robust dataset downloads, with updated OSF endpoints (:pull:`754`)
*  Smarter ``reshape_stim`` for implants (:pull:`680`)
*  Installation is now tested inside the real Google Colab runtime (:pull:`777`)
*  Various bug fixes (:pull:`682`, :pull:`700`, :pull:`732`, :pull:`776`)

v0.9.0 Cortex (2025-02-17)
--------------------------

Highlights:

*  Cortical implants: :py:class:`~pulse2percept.implants.cortex.Cortivis`
   [Fernandez2017]_ (:pull:`525`),
   :py:class:`~pulse2percept.implants.cortex.ICVP` [Troyk2003]_ (:pull:`542`),
   :py:class:`~pulse2percept.implants.cortex.Neuralink` [Musk2019]_
   (:pull:`597`)
*  Cortical models: :py:class:`~pulse2percept.models.cortex.ScoreboardModel`
   (:pull:`533`), :py:class:`~pulse2percept.models.cortex.DynaphosModel`
   [vanderGrinten2023]_ (:pull:`547`)
*  Cortical maps: :py:class:`~pulse2percept.topography.Polimeni2006Map`
   (:pull:`509`), :py:class:`~pulse2percept.topography.NeuropythyMap` 
   (:pull:`597`)
*  Other new implants: :py:class:`~pulse2percept.implants.IMIE` [Xu2021]_
   (:pull:`492`), :py:class:`~pulse2percept.implants.EnsembleImplant` 
   (:pull:`537`), :py:class:`~pulse2percept.implants.RectangleImplant`
   (:pull:`631`)
*  New datasets: :py:class:`~pulse2percept.datasets.fetch_han2021` 
   [Han2021]_ (:pull:`494`)
*  Torch and CUDA support (:pull:`633`)
*  Python 3.11 and 3.12 support
*  Various bug fixes

v0.8.0 Retina (2022-05-05)
--------------------------

Highlights:

*  New implants: :py:class:`~pulse2percept.implants.BVT44` [Petoe2021]_
   (:pull:`465`)
*  New models: :py:class:`~pulse2percept.models.BiphasicAxonMapModel`
   [Granley2021]_ (:pull:`398`) and
   :py:class:`~pulse2percept.models.Thompson2003Model` [Thompson2003]_
   (:pull:`448`)
*  New datasets: :py:func:`~pulse2percept.datasets.load_greenwald2009`
   [Greenwald2009]_ (:pull:`459`) and
   :py:func:`~pulse2percept.datasets.load_perezfornos2012`
   [PerezFornos2012]_ (:pull:`457`)
*  New stimuli: :py:class:`~pulse2percept.stimuli.BarStimulus`,
   :py:class:`~pulse2percept.stimuli.GratingStimulus` (:pull:`310`)
*  Python 3.10 support (:pull:`479`)
*  Various bug fixes

v0.7.1 (2021-06-21)
-------------------

Highlights:

*  Add :py:class:`~pulse2percept.models.FadingTemporal`, a generic phosphene fading model (:pull:`378`)
*  Various implant usability and speed upgrades (:pull:`375`, :pull:`382`, :pull:`383`, :pull:`386`)
*  Various stimulus usability and speed upgrades (:pull:`382`, :pull:`383`, :pull:`384`, :pull:`385`)
*  Improve documentation and usability of various :py:class:`~pulse2percept.models.AxonMapModel` methods (:pull:`370`)

v0.7.0 Implants (2021-04-04)
----------------------------

Highlights:

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
   :pull:`221`, :pull:`356`), :py:class:`~pulse2percept.stimuli.BarStimulus`
   and :py:class:`~pulse2percept.stimuli.GratingStimulus` (:pull:`323`)
*  New datasets: :py:class:`~pulse2percept.datasets.load_nanduri2012`
   (:pull:`250`)
*  New model selection subpackage (:pull:`311`)
*  100x speedup of building :py:class:`~pulse2percept.models.AxonMapModel` (:pull:`331`)
*  OpenMP support (:pull:`260`)
*  Python 3.9 support (:pull:`348`)
*  Various usability upgrades
*  Various bug fixes

v0.6.0 API (2020-05-05)
-----------------------

Highlights:

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
*   Various bug fixes

v0.5.2 (2020-02-25)
-------------------

Bug fix:

*   ``pulse2percept.retina.Nanduri2012``: improved Cython implementation

v0.5.1 (2020-02-05)
-------------------

Bug fixes:

*   ``pulse2percept.retina.Nanduri2012``: allow switch between FFT/Cython
*   ``pulse2percept.retina.Horsager2009``: respect ``use_jit`` option
*   ``pulse2percept.utils.center_vector``: "cannot determine Numba type"

v0.5.0 Community (2019-11-29)
-----------------------------

*   New :py:mod:`pulse2percept.viz` module (:pull:`84`)
*   Support for the :py:class:`~pulse2percept.implants.AlphaIMS` implant
    (:pull:`87`)
*   Automated wheelhouse build (:pull:`130`)
*   New contribution guidelines (:pull:`92`)
*   New issue templates (:pull:`93`)
*   New code of conduct (:pull:`95`)
*   Host documentation on
    `pulse2percept.readthedocs.io <https://pulse2percept.readthedocs.io>`_.

v0.4.3 Cython (2018-05-21)
--------------------------

Highlights:

*   Cython integration:

    * The model described in Nanduri et al. (2012) now uses a finite difference
      method implemented in Cython as opposed to FFT-based convolutions
      (:pull:`83`)

    * Single-core benchmarks show a 200x speedup over a pure-Python
      implementation.


v0.3.0 Baby Steps (2018-02-20)
------------------------------

*   New, faster axon map calculation
*   Better plotting
*   Support for left/right eye
