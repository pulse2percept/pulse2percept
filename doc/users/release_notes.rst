.. _users-release-notes:

=============
Release Notes
=============

v0.9.2 (unreleased)
-------------------

Highlights:

*  New :py:class:`~pulse2percept.stimuli.Encoder` classes, which translate the
   gray levels of an image or a video into the electrical stimulus an implant
   would actually deliver: every frame becomes a train of biphasic pulses that
   lasts one frame period.
   :py:class:`~pulse2percept.stimuli.AmplitudeEncoder` maps the gray level of a
   pixel onto the amplitude of its pulses at a fixed frequency;
   :py:class:`~pulse2percept.stimuli.FrequencyEncoder` maps it onto how often
   they come, at a fixed amplitude. Passing either an ``implant`` samples the
   source at the electrode locations *before* building the pulse trains, rather
   than after: encoding the 94-frame ``BostonTrain`` for Argus II now allocates
   0.2 MB and takes 50 ms, where encoding it at pixel resolution allocated
   308 MB and took 720 ms
*  Encoders model two properties of real stimulators that also decide how
   expensive the resulting stimulus is to simulate: ``clock``, the period of
   the device's time base, onto which pulse periods are rounded; and
   ``n_levels``, the resolution of its input stage. They matter most for
   frequency modulation, where electrodes pulsing at different rates otherwise
   need a time point wherever any of them has a pulse edge -- ``BostonTrain``
   at frequencies in (0, 300] Hz needs 121,494 time points unquantized, but
   18,056 on a 1 ms clock
*  A frame now delivers only pulses it can finish. Previously a pulse train
   whose period did not divide the frame duration ended with a truncated pulse,
   which injects net charge; a 30 Hz train on a 29.97 fps video did this on
   every frame
*  New :py:class:`~pulse2percept.implants.Raster` classes describe how a
   stimulator takes turns between electrodes it cannot drive at the same time.
   :py:class:`~pulse2percept.implants.SequentialRaster` splits electrodes into
   groups by position -- on the 6x10
   :py:class:`~pulse2percept.implants.ArgusII` grid, ``SequentialRaster(6)`` is
   a line raster -- and :py:class:`~pulse2percept.implants.CustomRaster`
   assigns them by name. An encoder staggers each group's pulses accordingly,
   taking the raster from its own ``raster`` argument or from the implant's
*  :py:class:`~pulse2percept.implants.ProsthesisSystem` gained a ``raster``
   attribute and a ``max_current`` one, the total current the stimulator can
   source at any instant. When ``max_current`` is set, assigning a stimulus
   that exceeds it raises, the way ``safe_mode`` does for charge balance.
   Encoding ``BostonTrain`` for Argus II at 0-50 uA draws 1310 uA at once
   without a raster and 298 uA with a six-group line raster, at the cost of a
   time axis roughly ``n_groups`` times longer
*  Python 3.14 support; the minimum supported Python is now 3.11
*  NumPy 2 is now required (``numpy>=2,<3``). If you are pinned to NumPy 
   1.x, stay on v0.9.1 -- ``pip`` will select it for you
*  Minimum dependency versions now match the oldest release of each that
   supports NumPy 2: SciPy 1.13, scikit-image 0.24, Matplotlib 3.9 and
   pandas 2.2.2
*  ``pulse2percept.topography`` no longer imports from
   ``pulse2percept.models``, removing a circular dependency between the two
   subpackages that had to be worked around with function-local imports.
   This is achieved with a new :py:class:`~pulse2percept.utils.Parametrized`
   base class, from which both 
   :py:class:`~pulse2percept.topography.VisualFieldMap` and
   :py:class:`~pulse2percept.models.BaseModel` inherit.
*  Visual-field-map equality now compares array-valued attributes elementwise
   instead of raising ``ValueError``, and maps are hashable again. Maps of
   different classes no longer compare equal, so
   ``Watson2014DisplaceMap() == Watson2014Map()`` is now ``False``
*  Merging the time axes of a multi-electrode
   :py:class:`~pulse2percept.stimuli.Stimulus` now compares time points with a
   tolerance that scales with their magnitude so that time points stay
   strictly monotonically increasing even after merging two stimuli.
*  ``Percept.play`` and ``VideoStimulus.play`` are roughly 100x faster and
   produce much smaller notebooks and doc pages. Instead of re-rendering the
   whole figure once per frame and embedding every frame as its own PNG (as
   Matplotlib's ``to_jshtml`` does), the new
   :py:class:`~pulse2percept.utils.HTMLAnimation` renders the figure once and
   ships all frames as a single color-mapped sprite sheet that is played back
   by a small, self-contained JavaScript player. Animating a 65x97x94 percept
   now takes 0.06s instead of 25s, and long videos are no longer silently
   truncated at Matplotlib's 20 MB embed limit
*  ``play`` gained a ``fmt`` parameter that chooses how the frames are
   embedded. The new default, ``fmt='jpg'``, halves the size of the resulting
   page again (a 94-frame percept goes from 3.7 MB to 0.22 MB); pass
   ``fmt='png'`` if you need the frames to be pixel-exact
*  ``Percept.play``, ``VideoStimulus.play``, and ``Percept.save`` no longer
   raise ``IndexError`` on a single-frame percept or video, which has no frame
   rate of its own. Inferring the frame rate from a non-homogeneous time axis
   now raises ``NotImplementedError`` with an explanation instead of a bare
   traceback. Both are handled by the new
   :py:func:`~pulse2percept.utils.frame_interval`

API changes:

*  :py:meth:`~pulse2percept.stimuli.ImageStimulus.encode` and
   :py:meth:`~pulse2percept.stimuli.VideoStimulus.encode` are now shorthands
   for :py:class:`~pulse2percept.stimuli.AmplitudeEncoder`, and their behavior
   changed in four ways:

   -  Gray levels map onto ``amp_range`` **absolutely**: a gray level of 0.5
      always encodes to the middle of the range. Previously they were stretched
      to fill it, which made the encoding depend on the content of the source
      and silently encoded a uniform image as zero amplitude everywhere. Pass
      ``stretch=True`` for the old behavior.
   -  Each frame now receives a pulse *train* (new ``freq`` argument,
      defaulting to 20 Hz) rather than a single pulse.
   -  The ``pulse`` argument now takes a single pulse to repeat, whose
      amplitude is normalized away, and is no longer modified in place.
   -  The new ``implant`` argument encodes at electrode rather than pixel
      resolution, and is strongly recommended for videos.

*  :py:meth:`~pulse2percept.stimuli.VideoStimulus.encode` used to infer a frame
   duration of ``1000 / dt`` ms from a video whose metadata carried no frame
   rate, where ``dt`` is the spacing of its time axis in ms. It now uses ``dt``
   itself, so such a video no longer comes back a factor of ``1000 / dt**2``
   too long.

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
