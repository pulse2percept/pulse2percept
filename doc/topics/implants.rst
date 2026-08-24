.. _topics-implants:

=================
Visual Prostheses
=================

An implant describes where stimulation is delivered: its electrodes, their
geometry and location, and the stimulus assigned to them. The implant does not
predict vision; that is the model's job.

All implants derive from
:py:class:`~pulse2percept.implants.ProsthesisSystem`. The attributes used most
often are:

``earray``
    The :py:class:`~pulse2percept.implants.ElectrodeArray`.

``stim``
    The :py:class:`~pulse2percept.stimuli.Stimulus` assigned to the implant.

``eye``
    The implanted eye for retinal systems.

``encoder`` and ``raster``
    Optional device behavior used when visual input is converted to electrical
    stimulation. These are covered later in :ref:`topics-encoders` and
    :ref:`topics-rasters`.

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

A custom array usually does not need a new implant class. Wrap an
:py:class:`~pulse2percept.implants.ElectrodeGrid` in a
:py:class:`~pulse2percept.implants.ProsthesisSystem`:

.. code-block:: python

    from pulse2percept.implants import ElectrodeGrid, ProsthesisSystem

    earray = ElectrodeGrid(shape=(10, 10), spacing=500)
    implant = ProsthesisSystem(earray=earray)

For irregular arrays, build an
:py:class:`~pulse2percept.implants.ElectrodeArray` from individual electrodes.
:py:class:`~pulse2percept.implants.EnsembleImplant` combines multiple implants
into one system.
