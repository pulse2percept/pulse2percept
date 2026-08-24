.. _topics-rasters:

=================
Raster Strategies
=================

.. versionadded:: 0.10.0

Some stimulators cannot drive every electrode simultaneously. A
:py:class:`~pulse2percept.implants.Raster` divides an array into groups that
take turns. It is a scheduling constraint used by an encoder, not a stimulus by
itself.

Basic usage
-----------

Rasters are attached to an implant:

.. code-block:: python

    import pulse2percept as p2p

    implant = p2p.implants.ArgusII()
    implant.raster = p2p.implants.CheckerboardRaster(n_groups=5)
    implant.encoder = p2p.stimuli.AmplitudeEncoder(
        amp_range=(0, 50),
        freq=20,
    )
    implant.stim = p2p.stimuli.BostonTrain()

Electrodes in one group may pulse together; different groups occupy different
time slots.

Built-in strategies
-------------------

.. list-table::
   :header-rows: 1

   * - Raster
     - Grouping
   * - :py:class:`~pulse2percept.implants.SequentialRaster`
     - Sequential or interleaved groups
   * - :py:class:`~pulse2percept.implants.CheckerboardRaster`
     - Spatially distributed grid groups
   * - :py:class:`~pulse2percept.implants.CustomRaster`
     - Explicit user-defined groups

For example:

.. code-block:: python

    implant.raster = p2p.implants.SequentialRaster(n_groups=6)
    implant.raster = p2p.implants.SequentialRaster(
        n_groups=6, interleave=True
    )
    implant.raster = p2p.implants.CheckerboardRaster(n_groups=5)

A CustomRaster is useful when the hardware already defines the groups. Every
electrode must belong to exactly one group.

Use ``raster.plot()`` to inspect the pattern and ``raster.members(...)`` to
retrieve the electrodes in a group.

Timing
------

Groups fire in order. ``group_dur`` sets the spacing between group starts. If
it is ``None``, the encoder spreads the groups across the pulse period. An
explicit value fixes the raster sweep duration:

.. code-block:: python

    implant.raster = p2p.implants.SequentialRaster(
        n_groups=6,
        group_dur=1,
    )

The slot must be long enough for a pulse, and the full sweep must fit within the
relevant pulse period.

With amplitude encoding, all electrodes share a pulse period, so rastering
only offsets the groups. With frequency encoding, electrodes may request
different periods; those schedules are constrained to whole raster sweeps and
may therefore run more slowly than requested, never faster.

If rastering is not part of the device or question being modeled, leave
``implant.raster`` unset.
