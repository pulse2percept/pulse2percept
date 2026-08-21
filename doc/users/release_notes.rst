.. _users-release-notes:

=============
Release Notes
=============

v0.10.0 Encoders (unreleased)
-----------------------------

Highlights:

*  New :py:class:`~pulse2percept.stimuli.StimulusEncoder` classes translate
   images and videos into electrical stimulation using amplitude or frequency
   modulation.

*  New :py:class:`~pulse2percept.implants.Raster` classes describe how
   stimulators multiplex electrodes that cannot be driven simultaneously.
   :py:class:`~pulse2percept.implants.CheckerboardRaster` implements the
   checkerboard pattern of [Kasowski2025]_ in a generalized form.

*  New :py:mod:`~pulse2percept.units` module support using physical quantities
   (``50 * uA``, ``450 * us``, ``15 * mm``, ``2 * dva``), which are
   dimension-checked and converted to the unit the code expects. Bare numbers
   keep working and keep their documented meaning everywhere.
   See :ref:`Physical Units <topics-units>`.

* :py:meth:`~pulse2percept.percepts.Percept.play` and
  :py:meth:`~pulse2percept.stimuli.VideoStimulus.play` are roughly 100x faster
  and produce much smaller notebooks and documentation pages.

* New :py:meth:`~pulse2percept.percepts.Percept.load` reads a percept back
  from an image, GIF, or movie file.

* Python 3.14 is now supported. Python 3.11 and NumPy 2 are now required.

API changes:

* :py:class:`~pulse2percept.models.FadingTemporal` is now driven by
  :math:`\max(-A, 0)` rather than :math:`-A`: anodic current no longer reduces
  brightness, it is ignored. A stimulus that is purely cathodic is unaffected.
  In addition, ``FadingTemporal`` now enforces ``tau >= dt``.

* The grid-spacing parameter ``xystep`` was renamed to ``step`` across spatial
  models and implant-grid factory methods. The axon-map parameter ``axlambda``
  was similarly renamed to ``lam``.

* Temporal models gained a ``reduce`` parameter. When ``predict_percept`` picks
  the output times itself (``t_percept=None``), ``reduce='peak'`` makes each
  point report the highest brightness reached over the interval leading up to
  it rather than the brightness at the instant it ends.

* :py:class:`~pulse2percept.implants.ProsthesisSystem` now exposes ``encoder``,
  ``raster`` and ``max_current``. A raster is bound to the implant it is
  assigned to (:py:meth:`~pulse2percept.implants.Raster.bind`).

* :py:meth:`~pulse2percept.percepts.Percept.play` and
  :py:meth:`~pulse2percept.percepts.Percept.save` gained ``vmin`` and
  ``vmax``. ``play`` also gained a ``fmt`` argument (defaulting to JPEG in
  :py:meth:`~pulse2percept.stimuli.VideoStimulus.play` and PNG in
  :py:meth:`~pulse2percept.percepts.Percept.play`).

* :py:class:`~pulse2percept.models.AxonMapSpatial` now smooths across the
  horizontal meridian by default (``meridian_blend=1`` dva), and
  :py:class:`~pulse2percept.models.cortex.ScoreboardSpatial` smooths across
  the vertical meridian (``meridian_blend=0.1`` dva). Set
  ``meridian_blend=0`` to recover the previous behavior.

* ``predict_percept`` now raises ``DimensionMismatchError`` when the stimulus
  is not the physical quantity the model reads. Assigning an
  :py:class:`~pulse2percept.stimuli.ImageStimulus` or
  :py:class:`~pulse2percept.stimuli.VideoStimulus` straight to ``implant.stim``
  previously had its gray levels silently treated as microamps. The same check
  guards the ``safe_mode`` and ``max_current`` safety checks.

* Minimum dependency versions were raised for NumPy 2 compatibility. NumPy 1.x
  users should remain on v0.9.1.

Bug fixes:

* Stimulus time axes now use float64 precision, fixing false
  :py:attr:`~pulse2percept.stimuli.Stimulus.is_charge_balanced` failures for
  longer pulse trains and improving frequency-modulation accuracy.

* Stimulus metadata now survives ``predict_percept`` and other transformations.

* Various fixes for :py:class:`~pulse2percept.stimuli.ImageStimulus`:
  keyword arguments are passed on to scikit-image; inputs are no longer
  modified in place; :py:meth:`~pulse2percept.stimuli.ImageStimulus.center`
  honors its ``loc`` argument instead of always centering on the middle of
  the image.

* :py:meth:`~pulse2percept.percepts.Percept.play`,
  :py:meth:`~pulse2percept.stimuli.VideoStimulus.play`, and
  :py:meth:`~pulse2percept.percepts.Percept.save` now handle single-frame
  inputs correctly and give a useful error for nonuniform time axes.

* :py:class:`~pulse2percept.implants.EnsembleImplant` now merges nearly
  identical time points using the same tolerance as
  :py:class:`~pulse2percept.stimuli.Stimulus`.

* Visual-field-map equality now handles array-valued attributes correctly,
  maps are hashable again, and maps of different classes no longer compare
  equal.

* :py:meth:`~pulse2percept.topography.NeuropythyMap.cortex_to_dva` (and the
  ``v1_to_dva``, ``v2_to_dva``, ``v3_to_dva`` methods that call it) now returns
  coordinates with the same shape as its input. Cortical points that land
  exactly on a mesh vertex now map to that vertex instead of dividing by zero
  and returning NaN. The docstrings said the cortical coordinates were in mm;
  they are in um, as the code always assumed.

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
