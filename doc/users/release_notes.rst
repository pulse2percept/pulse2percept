.. _users-release-notes:

=============
Release Notes
=============

v0.11.0 Residual Vision (unreleased)
====================================

Highlights
----------

* **New backwards-incompatible model API:** models are now bound to their
  implant and built automatically; ``predict_percept`` takes a stimulus
  directly (:pull:`862`):

  .. code-block:: python

      implant = p2p.implants.retina.ArgusII()
      model = p2p.models.retina.AxonMapModel(implant)
      percept = model.predict_percept(stim)

* New :py:mod:`pulse2percept.vision` module with
  :py:class:`~pulse2percept.vision.Scene` and
  :py:class:`~pulse2percept.vision.Scotoma` for gaze-aware simulation of
  residual vision and retinal prostheses (:pull:`854`, :pull:`871`, 
  :pull:`883`).

* New photovoltaic stimulation pipeline for
  :py:class:`~pulse2percept.implants.retina.PRIMAPivotal`, from image encoding to
  irradiance-based model input (:pull:`868`).


API changes and improvements
----------------------------

Stimuli and encoding
~~~~~~~~~~~~~~~~~~~~

* New :py:class:`~pulse2percept.stimuli.Encoder` framework supports electrical
  and non-electrical stimulation. New
  :py:class:`~pulse2percept.stimuli.PRIMAEncoder` implements photovoltaic
  image/video encoding, while
  :py:class:`~pulse2percept.stimuli.AmplitudeEncoder` supports threshold-based
  amplitude ranges such as ``(0 * xTh, 3 * xTh)`` (:pull:`868`,
  :pull:`869`).

* Added optical power units (``W``, ``mW``, ``uW``), proper geometric-angle
  units (``deg``, ``rad``), and ``pathlib.Path`` support for image and video
  stimuli (:pull:`855`, :pull:`868`, :pull:`880`).


Implants
~~~~~~~~

* Implants moved into anatomical subpackages:
  :py:mod:`pulse2percept.implants` now holds only generic device machinery
  (electrodes, electrode arrays, rasters,
  :py:class:`~pulse2percept.implants.Implant`,
  :py:class:`~pulse2percept.implants.GridImplant`,
  :py:class:`~pulse2percept.implants.EnsembleImplant`), with the devices
  themselves under ``implants.retina`` and ``implants.cortex``. The old flat
  paths were removed, not deprecated (:pull:`887`):

  .. code-block:: python

      # before
      from pulse2percept.implants import ArgusII
      p2p.implants.PRIMAPivotal()

      # v0.11
      from pulse2percept.implants.retina import ArgusII
      p2p.implants.retina.PRIMAPivotal()

  Implementation modules moved with them, so ``pulse2percept.implants.argus``
  is now ``pulse2percept.implants.retina.argus``. The deprecated ``PRIMA``,
  ``PRIMA75``, ``PRIMA55`` and ``PRIMA40`` spellings live in
  ``implants.retina``.

* ``eye`` moved off :py:class:`~pulse2percept.implants.Implant` and
  :py:class:`~pulse2percept.implants.GridImplant` onto the new
  :py:class:`~pulse2percept.implants.retina.RetinalImplant`, which every
  retinal device now derives from. A generic implant or grid carries no
  laterality; wrap an :py:class:`~pulse2percept.implants.ElectrodeGrid` in a
  ``RetinalImplant`` where a custom grid relied on the old implicit
  ``eye='RE'``. :py:class:`~pulse2percept.models.retina.AxonMapModel` mirrors
  the optic disc onto the implanted eye and says so if handed an implant
  without one (:pull:`887`).

* New :py:class:`~pulse2percept.implants.cortex.CorticalImplant` gives
  cortical devices ``hemisphere`` (``'LH'``, ``'RH'``, or ``None``, the
  default), as does
  :py:class:`~pulse2percept.implants.cortex.Neuralink`. It is device metadata:
  electrode coordinates and the model's ``implant_position`` still place the
  array (:pull:`887`).

* ``EnsembleImplant.from_cortical_map`` becomes
  :py:meth:`~pulse2percept.implants.EnsembleImplant.from_visual_field_map`,
  which accepts any 2D :py:class:`~pulse2percept.topography.VisualFieldMap`
  rather than only a cortical one, and rejects 3D maps. ``region`` defaults to
  the map's only region instead of ``'v1'``, so a multi-region map now
  requires it (:pull:`887`).

* ``ProsthesisSystem`` is renamed
  :py:class:`~pulse2percept.implants.Implant`; the old name remains as a
  deprecated alias until 0.12.0 (:pull:`876`).

* The PRIMA family was reorganized around the published devices:
  ``PRIMA`` becomes :py:class:`~pulse2percept.implants.retina.PRIMAPivotal`,
  ``PRIMA75`` becomes :py:class:`~pulse2percept.implants.retina.Lorach2015Array`,
  and new :py:class:`~pulse2percept.implants.retina.Ho2019FlatArray` and
  :py:class:`~pulse2percept.implants.retina.Huang2021Array` classes capture other
  photovoltaic designs. Device geometry and pixel dimensions were corrected
  accordingly (:pull:`865`).

* ``RectangleImplant`` was removed in favor of
  :py:class:`~pulse2percept.implants.GridImplant`, which is not a drop-in
  replacement: pass ``electrode_type=DiskElectrode, radius=75,
  preprocess=True`` to keep the old defaults, wrap it in
  :py:class:`~pulse2percept.implants.retina.RetinalImplant` for an ``eye``,
  and note that a left-eye grid keeps the column names of a right-eye one
  (:pull:`859`).

* Implant and electrode APIs were cleaned up for consistency: descriptive
  names replace abbreviations such as ``earray``, ``vfmap``, ``etype``, ``r``,
  and ``a``; ``grid_type`` now uses ``'rect'``/``'hex'`` throughout;
  ``plot3D()`` becomes ``plot3d()``; and implants/electrode arrays follow
  normal Python container behavior (:pull:`880`).

* :py:class:`~pulse2percept.implants.Implant` now exposes device
  ``placement``, ``technology``, and ``family`` metadata and supports
  per-electrode thresholds (:pull:`865`, :pull:`869`).


Models
~~~~~~

* Models moved into anatomical subpackages:
  :py:mod:`pulse2percept.models` now holds only the abstract model classes and
  the generic temporal models, with the models themselves under
  ``models.retina`` and ``models.cortex``. The old flat paths were removed,
  not deprecated (:pull:`887`):

  .. code-block:: python

      # before
      from pulse2percept.models import AxonMapModel
      p2p.models.Nanduri2012Model(implant)

      # v0.11
      from pulse2percept.models.retina import AxonMapModel
      p2p.models.retina.Nanduri2012Model(implant)

  Implementation modules moved with them, so
  ``pulse2percept.models.beyeler2019`` is now
  ``pulse2percept.models.retina.beyeler2019``. The cortical scoreboard moved
  out of ``models.cortex.base`` into ``models.cortex.scoreboard``, and both
  scoreboards now share the same Gaussian spread kernels in
  ``pulse2percept.models._scoreboard``.

* New :py:class:`~pulse2percept.models.retina.RetinalSpatial` owns the retinal
  half of the old :py:class:`~pulse2percept.models.SpatialModel`: the default
  retinotopic map, reading a physical retinal extent as shorthand for
  ``xrange``/``yrange``, and registering a scene against the electrodes.
  :py:class:`~pulse2percept.models.SpatialModel` is now anatomy-neutral: it
  raises if built without a ``visual_field_map``, and predicting from a
  :py:class:`~pulse2percept.vision.Scene` on a model that does not implement
  the registration (a cortical one, for now) raises ``NotImplementedError``
  rather than ``ValueError`` (:pull:`887`).

* Model constructors now expose their supported parameters explicitly instead
  of accepting arbitrary ``**params``. Composite models no longer forward
  component attributes: spatial and temporal parameters are accessed through
  ``model.spatial`` and ``model.temporal`` (:pull:`879`).

* New :py:class:`~pulse2percept.models.retina.BiphasicScoreboardModel` and
  :py:class:`~pulse2percept.models.retina.BiphasicScoreboardSpatial` apply the
  [Granley2021]_ brightness and size fits to round, electrode-centered
  phosphenes, without the axonal streak (:pull:`886`).

* :py:class:`~pulse2percept.models.retina.ScoreboardSpatial` can consume normalized
  photovoltaic drive, and
  :py:class:`~pulse2percept.models.retina.BiphasicAxonMapModel` can predict directly
  from still images encoded with the standard biphasic pipeline
  (:pull:`868`, :pull:`869`).

* ``find_threshold`` has been removed; threshold searches belong at the
  experiment level rather than in the model API (:pull:`862`).

* Implant placement moved from named implants to models. Named implants
  now use device-local coordinates; use ``implant_position``,
  ``implant_rotation`` and ``implant_depth`` to place them in tissue.
  ``implant_position`` may also be given in dva for a single-region 2D field
  map. 3D maps such as ``NeuropythyMap`` support only identity model-side
  placement. Named retinal arrays now default to ``z=0``; fixed device
  geometry such as shank depth remains on the implant.
  :py:meth:`~pulse2percept.implants.EnsembleImplant.from_coords` replaces
  positioning named implants through their constructors (:pull:`884`).

* The generic ``noise`` parameter has been removed from
  :py:class:`~pulse2percept.percepts.Percept` and all models (:pull:`885`).

* New ``location_noise`` parameter displaces each electrode's phosphene by a
  fixed, subject-specific offset in the visual field (dva) rather than at the
  location the ``visual_field_map`` gives it. Requires a 2D, invertible map
  (:pull:`881`).

* ``implant.plot()`` shows device-local geometry;
  ``model.plot(show_implant=True)`` shows model-side placement. The Argus
  phosphene plotting helpers likewise accept ``implant_position`` and
  ``implant_rotation`` (:pull:`884`).


Topography
~~~~~~~~~~

* Visual field maps moved into anatomical subpackages:
  :py:mod:`pulse2percept.topography` now holds only
  :py:class:`~pulse2percept.topography.Grid2D` and
  :py:class:`~pulse2percept.topography.VisualFieldMap`, with the maps
  themselves under ``topography.retina`` and ``topography.cortex``. The old
  flat paths were removed, not deprecated (:pull:`887`):

  .. code-block:: python

      # before
      p2p.topography.Watson2014Map()
      p2p.topography.Polimeni2006Map()
      p2p.topography.NeuropythyMap()

      # v0.11
      p2p.topography.retina.Watson2014Map()
      p2p.topography.cortex.Polimeni2006Map()
      p2p.topography.cortex.NeuropythyMap()


Residual vision
~~~~~~~~~~~~~~~

* :py:class:`~pulse2percept.vision.Scene` adds softened scotoma boundaries,
  optional inpainting, configurable backgrounds, and eccentricity rings in
  both static and animated views (:pull:`871`).

* Prosthetic composition now rejects ``scotoma_fill='inpaint'``, whose
  interaction with prosthetic brightness is not modeled (:pull:`884`).


Plotting
~~~~~~~~

* New :py:mod:`pulse2percept.plotting` module provides combined stimulus/percept
  figures and animations. :py:mod:`pulse2percept.viz` is deprecated until
  0.12.0 (:issue:`872`).


Bug fixes
---------

* Fixed centering of single-row and single-column hexagonal
  :py:class:`~pulse2percept.implants.ElectrodeGrid` instances (:pull:`859`).

* Corrected PRIMA-family pixel dimensions and layouts (:pull:`865`).

* :py:class:`~pulse2percept.models.retina.BiphasicAxonMapSpatial` now respects
  ``n_gray`` and preserves the full stimulus in percept metadata
  (:pull:`869`).


v0.10.0 Encoders (2026-08-23)
=============================

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

* ``ProsthesisSystem`` (now
  :py:class:`~pulse2percept.implants.Implant`) supports stimulus
  encoders, raster strategies, and maximum-current limits. Dimensionless
  image/video stimuli can be encoded automatically when assigned to an implant
  (:pull:`810`, :pull:`833`)

* Temporal models gained a ``reduce`` parameter for summarizing automatically
  selected output intervals. :py:class:`~pulse2percept.models.FadingTemporal`
  now responds only to cathodic current (:pull:`818`)

* :py:class:`~pulse2percept.models.retina.BiphasicAxonMapModel` now distinguishes
  physical current from threshold-relative amplitude via the new ``xTh`` unit
  (:pull:`848`)

* :py:class:`~pulse2percept.models.retina.BiphasicAxonMapSpatial` can now be composed
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
===================

Highlights:

*  Python 3.13 support; the minimum supported Python is now 3.10 (:pull:`649`)
*  NumPy 2.x support: wheels import under both NumPy 1.x and 2.x, so installing
   pulse2percept no longer downgrades NumPy in environments that ship it, such
   as Google Colab (:pull:`635`, :pull:`736`)
*  Removed the jax engine and ``predict_percept_batched`` from
   :py:class:`~pulse2percept.models.retina.BiphasicAxonMapModel`; the ``engine``
   argument of the effect models and the ``pad`` argument of
   :py:meth:`~pulse2percept.models.retina.AxonMapSpatial.calc_axon_sensitivity` are
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
==========================

Highlights:

*  Cortical implants: :py:class:`~pulse2percept.implants.cortex.Cortivis`
   [Fernandez2017]_ (:pull:`525`),
   :py:class:`~pulse2percept.implants.cortex.ICVP` [Troyk2003]_ (:pull:`542`),
   :py:class:`~pulse2percept.implants.cortex.Neuralink` [Musk2019]_
   (:pull:`597`)
*  Cortical models: :py:class:`~pulse2percept.models.cortex.ScoreboardModel`
   (:pull:`533`), :py:class:`~pulse2percept.models.cortex.DynaphosModel`
   [vanderGrinten2023]_ (:pull:`547`)
*  Cortical maps: :py:class:`~pulse2percept.topography.cortex.Polimeni2006Map`
   (:pull:`509`), :py:class:`~pulse2percept.topography.cortex.NeuropythyMap` 
   (:pull:`597`)
*  Other new implants: :py:class:`~pulse2percept.implants.retina.IMIE` [Xu2021]_
   (:pull:`492`), :py:class:`~pulse2percept.implants.EnsembleImplant` 
   (:pull:`537`), ``RectangleImplant`` (:pull:`631`)
*  New datasets: :py:class:`~pulse2percept.datasets.fetch_han2021` 
   [Han2021]_ (:pull:`494`)
*  Torch and CUDA support (:pull:`633`)
*  Python 3.11 and 3.12 support
*  Various bug fixes

v0.8.0 Retina (2022-05-05)
==========================

Highlights:

*  New implants: :py:class:`~pulse2percept.implants.retina.BVT44` [Petoe2021]_
   (:pull:`465`)
*  New models: :py:class:`~pulse2percept.models.retina.BiphasicAxonMapModel`
   [Granley2021]_ (:pull:`398`) and
   :py:class:`~pulse2percept.models.retina.Thompson2003Model` [Thompson2003]_
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
===================

Highlights:

*  Add :py:class:`~pulse2percept.models.FadingTemporal`, a generic phosphene fading model (:pull:`378`)
*  Various implant usability and speed upgrades (:pull:`375`, :pull:`382`, :pull:`383`, :pull:`386`)
*  Various stimulus usability and speed upgrades (:pull:`382`, :pull:`383`, :pull:`384`, :pull:`385`)
*  Improve documentation and usability of various :py:class:`~pulse2percept.models.retina.AxonMapModel` methods (:pull:`370`)

v0.7.0 Implants (2021-04-04)
============================

Highlights:

*  New implants: :py:class:`~pulse2percept.implants.retina.PRIMA`, 
   :py:class:`~pulse2percept.implants.retina.PRIMA75`,
   :py:class:`~pulse2percept.implants.retina.PRIMA55`, 
   :py:class:`~pulse2percept.implants.retina.PRIMA40` (:pull:`188`)
*  New electrodes: :py:class:`~pulse2percept.implants.SquareElectrode`,
   :py:class:`~pulse2percept.implants.HexElectrode`,
   :py:class:`~pulse2percept.implants.retina.PhotovoltaicPixel` (:pull:`188`, 
   :pull:`193`)
*  New stimuli: :py:class:`~pulse2percept.stimuli.ImageStimulus` and
   :py:class:`~pulse2percept.stimuli.VideoStimulus` (:pull:`196`, :pull:`220`,
   :pull:`221`, :pull:`356`), :py:class:`~pulse2percept.stimuli.BarStimulus`
   and :py:class:`~pulse2percept.stimuli.GratingStimulus` (:pull:`323`)
*  New datasets: :py:class:`~pulse2percept.datasets.load_nanduri2012`
   (:pull:`250`)
*  New model selection subpackage (:pull:`311`)
*  100x speedup of building :py:class:`~pulse2percept.models.retina.AxonMapModel` (:pull:`331`)
*  OpenMP support (:pull:`260`)
*  Python 3.9 support (:pull:`348`)
*  Various usability upgrades
*  Various bug fixes

v0.6.0 API (2020-05-05)
=======================

Highlights:

*   New API (:pull:`96`, :pull:`174`, :pull:`178`)
*   New implants: ``BVA24``, since renamed
    :py:class:`~pulse2percept.implants.retina.BVT24` (:pull:`161`)
*   New models: :py:class:`~pulse2percept.models.retina.ScoreboardModel` (:pull:`96`),
    :py:class:`~pulse2percept.models.retina.AxonMapModel` (:pull:`96`),
    :py:class:`~pulse2percept.models.retina.Nanduri2012Model` (:pull:`168`),
    :py:class:`~pulse2percept.models.retina.Horsager2009Model` (:pull:`180`)
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
===================

Bug fix:

*   ``pulse2percept.retina.Nanduri2012``: improved Cython implementation

v0.5.1 (2020-02-05)
===================

Bug fixes:

*   ``pulse2percept.retina.Nanduri2012``: allow switch between FFT/Cython
*   ``pulse2percept.retina.Horsager2009``: respect ``use_jit`` option
*   ``pulse2percept.utils.center_vector``: "cannot determine Numba type"

v0.5.0 Community (2019-11-29)
=============================

*   New :py:mod:`pulse2percept.viz` module (:pull:`84`)
*   Support for the :py:class:`~pulse2percept.implants.retina.AlphaIMS` implant
    (:pull:`87`)
*   Automated wheelhouse build (:pull:`130`)
*   New contribution guidelines (:pull:`92`)
*   New issue templates (:pull:`93`)
*   New code of conduct (:pull:`95`)
*   Host documentation on
    `pulse2percept.readthedocs.io <https://pulse2percept.readthedocs.io>`_.

v0.4.3 Cython (2018-05-21)
==========================

Highlights:

*   Cython integration:

    * The model described in Nanduri et al. (2012) now uses a finite difference
      method implemented in Cython as opposed to FFT-based convolutions
      (:pull:`83`)

    * Single-core benchmarks show a 200x speedup over a pure-Python
      implementation.


v0.3.0 Baby Steps (2018-02-20)
==============================

*   New, faster axon map calculation
*   Better plotting
*   Support for left/right eye
