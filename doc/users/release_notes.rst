.. _users-release-notes:

=============
Release Notes
=============

v0.11.0 Residual Vision (unreleased)
------------------------------------

Highlights:

* **New backwards-incompatible API**: Models are now bound to the implant
  they describe, they build automatically, and ``predict_percept`` takes a
  stimulus directly:

  .. code-block:: python

      implant = p2p.implants.ArgusII()
      model = p2p.models.AxonMapModel(implant=implant)
      percept = model.predict_percept(stim)

* New :py:mod:`pulse2percept.vision` introduces
  :py:class:`~pulse2percept.vision.Scene` and
  :py:class:`~pulse2percept.vision.Scotoma` for gaze-aware simulation of
  residual vision and retinal prostheses (:pull:`854`).

API changes:

* :py:class:`~pulse2percept.implants.ProsthesisSystem` adds descriptive
  ``placement``, ``technology`` and ``family`` class attributes (:pull:`865`).

* :py:class:`~pulse2percept.implants.RectangleImplant` is deprecated in favor
  of :py:class:`~pulse2percept.implants.GridImplant` (:pull:`859`)

* New :py:class:`~pulse2percept.stimuli.PRIMAEncoder` maps images and videos
  to 880 nm irradiance for :py:class:`~pulse2percept.implants.PRIMAPivotal`,
  using the pivotal system's 30 Hz projector and 14 pulse-duration levels
  (0.7--9.8 ms). ``PRIMAPivotal`` uses this encoder by default, and its
  ``safe_mode`` checks the documented projector operating envelope.

* New :py:class:`~pulse2percept.stimuli.Encoder` base class, so that
  ``implant.encoder`` accepts non-electrical encoders.
  :py:class:`~pulse2percept.stimuli.StimulusEncoder` inherits from it.

* New ``W``, ``mW`` and ``uW`` units, from which irradiance follows as
  ``mW / mm ** 2``.

* :py:class:`~pulse2percept.models.ScoreboardSpatial` can also read the
  normalized dimensionless drive produced by photovoltaic encoders such as
  :py:class:`~pulse2percept.stimuli.PRIMAEncoder`; arbitrary dimensionless
  stimuli such as image gray levels remain invalid.

* New :py:class:`~pulse2percept.implants.Ho2019FlatArray` models the 55
  and 40 um flat arrays of [Ho2019]_; new
  :py:class:`~pulse2percept.implants.Huang2021Array` models the 55, 40, 30 and
  20 um vertical-junction arrays of [Huang2021]_ (:pull:`865`).

* ``PRIMA`` is renamed
  :py:class:`~pulse2percept.implants.PRIMAPivotal`, ``PRIMA75`` becomes
  :py:class:`~pulse2percept.implants.Lorach2015Array`, and ``PRIMA55``/
  ``PRIMA40`` become ``Ho2019FlatArray(55)``/``Ho2019FlatArray(40)``.
  The old names are deprecated until 0.12.0 (:pull:`865`).

* New ``deg`` and ``rad`` units for ordinary geometric angle, accepted
  wherever p2p already took an angle in degrees.

* ``find_threshold`` has been removed. Threshold searches are
  experiment-specific optimizations over stimulus parameters and should be
  implemented at that level.

* The PRIMA implants replace ``trench`` with ``pixel_width`` (flat-to-flat
  width of the hexagonal pixel body) and ``gap`` (open gap between pixel
  bodies), plus a derived ``row_spacing``. ``spacing`` keeps its meaning:
  nearest-neighbor center-to-center distance (:pull:`865`).

* :py:class:`~pulse2percept.implants.HexElectrode` takes ``orientation`` and
  ``rot``, so hexagonal bodies face the nearest-neighbor axis of the lattice
  they sit on and turn with it (:pull:`865`).

* PRIMA photovoltaic arrays now plot their substrate and clip rim pixels
  to the substrate boundary (:pull:`865`).

Bug fixes:

* :py:class:`~pulse2percept.models.BiphasicAxonMapSpatial` ignored ``n_gray``
  and ``noise``, and stored only the stimulus metadata (not the stimulus) under
  ``percept.metadata['stim']`` (:issue:`864`).

* A hexagonal :py:class:`~pulse2percept.implants.ElectrodeGrid` with a single
  row (or, in vertical orientation, a single column) is now centered on
  ``(x, y)`` rather than offset by a quarter of the electrode spacing
  (:pull:`859`)

* :py:class:`~pulse2percept.implants.HexElectrode` sized its hexagon by
  passing the apothem ``a`` to Matplotlib as a circumradius, drawing every
  hexagonal pixel 13% too small (:pull:`865`)

* :py:class:`~pulse2percept.implants.PRIMAPivotal` pixels are corrected
  from 85 to 100 um wide; pixel centers are unchanged (:pull:`865`).

* ``PRIMA55`` and ``PRIMA40`` (now
  :py:class:`~pulse2percept.implants.Ho2019FlatArray`) model the F55 and F40
  arrays of [Ho2019]_: 250 and 502 pixels (was 273 and 532), as wide as their
  center spacing with no open gap, and 14 um and 10 um active electrodes (was
  16 um). Correcting the layouts changes which electrode names these two
  devices have (:pull:`865`)

v0.10.0 Encoders (2026-08-23)
-----------------------------

Highlights:

* New stimulus encoders support amplitude and frequency modulation of images and
  videos, with implant-specific raster strategies for multiplexed stimulation
  (:pull:`810`, :pull:`820`, :pull:`833`)

* Stimuli now retain structured pulse and encoder representations and generate
  waveform samples lazily, substantially reducing construction time and memory
  use (:pull:`842`)

* New :py:mod:`~pulse2percept.units` support for dimension-checked physical
  quantities throughout the public API (:pull:`828`)

* Major performance improvements for stimulus handling, spatial models, temporal
  models, and large-array simulations (:pull:`800`, :pull:`805`, :pull:`808`,
  :pull:`821`, :pull:`850`)

* :py:meth:`~pulse2percept.percepts.Percept.play` and
  :py:meth:`~pulse2percept.stimuli.VideoStimulus.play` are substantially faster,
  and percept playback now supports irregular time axes (:pull:`809`,
  :pull:`834`)

* New :py:class:`~pulse2percept.models.AlphaTemporal` and much faster
  :py:class:`~pulse2percept.models.FadingTemporal` temporal models
  (:pull:`849`)

* New :py:meth:`~pulse2percept.percepts.Percept.load` reads percepts from image
  and video files (:pull:`835`)

* Python 3.14 is supported. Python 3.11 and NumPy 2 are now required
  (:pull:`790`)


API changes:

* ``Stimulus.data``, ``time``, ``electrodes``, and pulse parameters are now
  read-only. New :py:meth:`~pulse2percept.stimuli.Stimulus.pad` and
  :py:meth:`~pulse2percept.stimuli.Stimulus.shift` methods provide explicit
  time-axis transformations (:pull:`837`, :pull:`842`)

* Image and video stimuli now use grid-style electrode names such as ``'A1'``
  and ``'C12_G'`` instead of integer pixel indices (:pull:`805`)

* :py:class:`~pulse2percept.implants.ProsthesisSystem` now supports stimulus
  encoders, raster strategies, and maximum-current limits. Dimensionless
  image/video stimuli can be encoded automatically when assigned to an implant
  (:pull:`810`, :pull:`833`)

* Temporal models gained a ``reduce`` parameter for summarizing automatically
  selected output intervals. :py:class:`~pulse2percept.models.FadingTemporal`
  now responds only to cathodic current (:pull:`818`)

* :py:class:`~pulse2percept.models.BiphasicAxonMapModel` now distinguishes
  physical current from threshold-relative amplitude via the new ``xTh`` unit
  (:pull:`848`)

* :py:class:`~pulse2percept.models.BiphasicAxonMapSpatial` can now be composed
  with temporal models using a space-time-separable approximation (:pull:`847`)

* Model parameter ``xystep`` was renamed to ``step`` and ``axlambda`` to
  ``lam``; the old names are deprecated until v0.11 (:pull:`824`, :pull:`830`)

* Axon-map and cortical scoreboard models can smooth predictions across
  visual-field meridians (:pull:`838`)

* Multi-electrode stimuli now plot as electrode-by-time heatmaps by default;
  percept playback and saving gained ``vmin``/``vmax`` controls
  (:pull:`835`, :pull:`841`)


Bug fixes:

* Fixed stimulus timing, metadata propagation, pulse scheduling, and structured
  stimulus handling across implants and models (:pull:`804`, :pull:`810`,
  :pull:`818`, :pull:`825`, :pull:`846`)

* Fixed several image and video issues, including argument forwarding,
  centering, single-frame playback, irregular timing, and efficient partial
  video loading (:pull:`815`, :pull:`822`, :pull:`834`, :pull:`844`)

* Fixed several Neuropythy and cortical-map issues involving shapes, NaNs,
  scalar inputs, mesh vertices, subject IDs, and coordinate units
  (:pull:`826`, :pull:`836`, :pull:`845`)

* Various smaller correctness fixes (:pull:`814`)

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
