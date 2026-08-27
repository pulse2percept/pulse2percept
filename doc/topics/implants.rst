.. _topics-implants:

=================
Visual Prostheses
=================

An implant describes the device: its electrodes, their geometry and location,
and how a source stimulus becomes the current those electrodes deliver. The
implant does not predict vision; that is the model's job.

All implants derive from
:py:class:`~pulse2percept.implants.ProsthesisSystem`. The attributes used most
often are:

``earray``
    The :py:class:`~pulse2percept.implants.ElectrodeArray`.

``eye``
    The implanted eye for retinal systems.

``placement``
    Where the device sits relative to the tissue it stimulates
    (``'epiretinal'``, ``'subretinal'``, ``'suprachoroidal'``,
    ``'epicortical'``, ``'intracortical'``), or ``None`` for a generic array.

``encoder`` and ``raster``
    Optional device behavior used when visual input is converted to electrical
    stimulation. These are covered later in :ref:`topics-encoders` and
    :ref:`topics-rasters`.

An implant holds no stimulus. What it delivers is derived from a source, on
demand, by :py:meth:`~pulse2percept.implants.ProsthesisSystem.prepare_stim`.

Basic use
---------

Electrodes can be accessed by name or index, and the array can be plotted
directly:

.. code-block:: python

    import pulse2percept as p2p

    implant = p2p.implants.ArgusII()

    implant['A8']
    implant[0]
    implant.electrode_names
    implant.earray.coordinates()
    implant.plot()

Preparing a stimulus
--------------------

:py:meth:`~pulse2percept.implants.ProsthesisSystem.prepare_stim` converts a
source into stimulation the device can deliver:

.. code-block:: python

    delivered = implant.prepare_stim({'A8': 30})
    delivered = implant.prepare_stim(p2p.stimuli.BostonTrain())

Preparation includes preprocessing, image/video encoding, resampling onto the
electrode array, raster scheduling, threshold calibration, and safety checks.
The result is returned as a :py:class:`~pulse2percept.stimuli.Stimulus` and is
not stored on the implant.

Models call ``prepare_stim`` internally. Call it directly when the delivered
stimulation itself is of interest:

.. code-block:: python

    implant.prepare_stim(source).plot()
    implant.plot(stim=source, stim_cmap=True)

Available implants
------------------

pulse2percept includes software representations of several published visual
prostheses:

.. list-table::
   :header-rows: 1

   * - Location
     - Implants
   * - Epiretinal
     - :py:class:`~pulse2percept.implants.ArgusI`,
       :py:class:`~pulse2percept.implants.ArgusII`,
       :py:class:`~pulse2percept.implants.IMIE`
   * - Subretinal
     - :py:class:`~pulse2percept.implants.AlphaIMS`,
       :py:class:`~pulse2percept.implants.AlphaAMS`,
       :py:class:`~pulse2percept.implants.PRIMA`,
       :py:class:`~pulse2percept.implants.PRIMA75`,
       :py:class:`~pulse2percept.implants.PRIMA55`,
       :py:class:`~pulse2percept.implants.PRIMA40`
   * - Suprachoroidal
     - :py:class:`~pulse2percept.implants.BVT24`,
       :py:class:`~pulse2percept.implants.BVT44`
   * - Cortical
     - :py:class:`~pulse2percept.implants.cortex.Orion`,
       :py:class:`~pulse2percept.implants.cortex.Cortivis`,
       :py:class:`~pulse2percept.implants.cortex.ICVP`,
       :py:class:`~pulse2percept.implants.cortex.Neuralink`

These classes are research-software representations based on published device
descriptions, not manufacturer-validated simulators. See each class's API
documentation for geometry-specific assumptions.

Coordinate systems
------------------

Retinal implants are centered on the fovea and store distances in microns.
Positive ``x`` points toward nasal retina, positive ``y`` toward superior
retina, and positive ``z`` away from the retina into the vitreous. ``eye``
handles left- versus right-eye geometry where needed.

Cortical implants use physical cortical coordinates. A cortical model combines
those coordinates with a
:py:class:`~pulse2percept.topography.VisualFieldMap` to place stimulation in
the visual field.

Custom implants
---------------

A custom array usually does not need a new implant class. For a regular grid,
use :py:class:`~pulse2percept.implants.GridImplant`:

.. code-block:: python

    import pulse2percept as p2p

    implant = p2p.implants.GridImplant(shape=(10, 10), spacing=500)

Grids can also be hexagonal:

.. code-block:: python

    implant = p2p.implants.GridImplant(shape=(20, 20), spacing=400, type='hex')

By default the electrodes are point sources. Pass an ``etype`` and its
arguments for electrodes with a physical extent:

.. code-block:: python

    implant = p2p.implants.GridImplant(shape=(20, 20), spacing=400, type='hex',
                                       etype=p2p.implants.DiskElectrode, r=75)

:py:class:`~pulse2percept.implants.GridImplant` is a convenience only:
:py:class:`~pulse2percept.implants.ElectrodeGrid` describes the geometry,
:py:class:`~pulse2percept.implants.ProsthesisSystem` describes the device, and
the two can still be combined by hand. Do that for an irregular array, built
from individual electrodes:

.. code-block:: python

    from pulse2percept.implants import ElectrodeArray, ProsthesisSystem

    earray = ElectrodeArray(...)
    implant = ProsthesisSystem(earray)

:py:class:`~pulse2percept.implants.EnsembleImplant` combines multiple implants
into one system.
