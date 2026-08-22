.. _topics-rasters:

=================
Raster Strategies
=================

.. versionadded:: 0.10.0

A stimulator may not be able to drive every electrode at the same time. A
:py:class:`~pulse2percept.implants.Raster` splits the electrode array into
groups that take turns, limiting how much current has to be delivered at any
one instant.

A raster is a **scheduling constraint**, not a stimulus by itself:

::

    image / video -> Encoder -> electrical Stimulus
                         ^
                         |
                       Raster

The raster says **which electrodes may pulse together**. The
:py:class:`~pulse2percept.stimuli.StimulusEncoder` decides when the pulses occur and
what their amplitudes or frequencies are.

The usual workflow
------------------

Raster strategies are attached to an implant and are picked up automatically
by an encoder:

.. code-block:: python

    import pulse2percept as p2p
    from pulse2percept.units import uA, Hz

    implant = p2p.implants.ArgusII()
    implant.raster = p2p.implants.CheckerboardRaster(n_groups=5)

    encoder = p2p.stimuli.AmplitudeEncoder(
        amp_range=(0, 50 * uA),
        freq=20 * Hz,
    )
    implant.stim = encoder.encode(
        p2p.stimuli.BostonTrain(),
        implant=implant,
    )

Here the 60 electrodes are split into five groups. Electrodes within one group
may pulse together, while different groups receive different time slots.

You can inspect the grouping directly:

.. code-block:: python

    implant.raster.plot()
    implant.raster.members(implant.electrode_names, 0)

Built-in strategies
-------------------

pulse2percept provides three raster strategies:

.. list-table::
   :header-rows: 1

   * - Raster
     - Strategy
   * - :py:class:`~pulse2percept.implants.SequentialRaster`
     - Split electrodes into sequential groups. On a regular grid this can
       reproduce a row or line raster.
   * - :py:class:`~pulse2percept.implants.CheckerboardRaster`
     - Spread electrodes in each group as far apart as possible across a
       regular grid.
   * - :py:class:`~pulse2percept.implants.CustomRaster`
     - Assign electrodes to groups explicitly.

A sequential raster is the simplest:

.. code-block:: python

    implant.raster = p2p.implants.SequentialRaster(n_groups=6)

For Argus II, whose electrodes are ordered row by row,
``SequentialRaster(6)`` puts one row in each group. Setting
``interleave=True`` instead distributes consecutive electrodes across
different groups.

A checkerboard raster is usually more spatially distributed:

.. code-block:: python

    implant.raster = p2p.implants.CheckerboardRaster(n_groups=5)

It derives the grouping from the electrode locations, so it works with square,
rectangular, rotated, and hexagonal grids. The array must actually lie on a
regular grid, and not every number of groups is possible for every geometry.

For complete control, specify the groups yourself:

.. code-block:: python

    corners = ['A1', 'A10', 'F1', 'F10']
    rest = [e for e in implant.electrode_names if e not in corners]

    implant.raster = p2p.implants.CustomRaster([corners, rest])

Every electrode must belong to exactly one group.

How the timing works
--------------------

Groups take their turns one after another. The spacing between turns is the
raster's ``group_dur``.

By default, ``group_dur=None``. The encoder then spreads the groups evenly
across the pulse period. For example, six groups driven at 20 Hz share the
50 ms period, so their slots begin one-sixth of a period apart.

You can instead specify the slot duration explicitly:

.. code-block:: python

    from pulse2percept.units import ms

    raster = p2p.implants.SequentialRaster(
        n_groups=6, group_dur=1 * ms
    )

This gives a 6 ms raster sweep: group 0 starts at 0 ms, group 1 at 1 ms, and so
on.

The slot must be long enough to contain a pulse, and the whole sweep must fit
within the relevant pulse period.

Amplitude versus frequency encoding
-----------------------------------

Rastering behaves differently depending on what the encoder modulates.

With :py:class:`~pulse2percept.stimuli.AmplitudeEncoder`, every electrode has
the same pulse period. Their group offsets therefore remain fixed and cannot
drift into one another. **Rastering does not lower the requested pulse
frequency in this case.**

With :py:class:`~pulse2percept.stimuli.FrequencyEncoder`, electrodes can have
different pulse periods. Those schedules would eventually drift into one
another, so the encoder constrains differing periods to whole raster sweeps.
Periods are always rounded **up**, never down, so rastering may make an
electrode pulse more slowly than requested but never faster.

A shorter explicit ``group_dur`` produces a shorter sweep and therefore finer
frequency resolution when this matters.

Choosing a strategy
-------------------

For most simulations:

* use :py:class:`~pulse2percept.implants.SequentialRaster` when you want a
  simple line, block, or interleaved schedule;
* use :py:class:`~pulse2percept.implants.CheckerboardRaster` when you want
  simultaneously active electrodes spread across a regular array;
* use :py:class:`~pulse2percept.implants.CustomRaster` when the hardware
  already defines the groups or you need a specific pattern.

If rastering is not part of the question you are studying, you can leave it
unset. The encoder will then stimulate all electrodes on the same schedule.

Physical units
--------------

``group_dur`` accepts either a bare number in milliseconds or a unitful
quantity:

.. code-block:: python

    from pulse2percept.units import us

    raster = p2p.implants.SequentialRaster(
        n_groups=6, group_dur=1000 * us
    )

See :ref:`topics-units` for the full units convention.

.. seealso::

    * :py:class:`pulse2percept.implants.Raster`
    * :py:class:`pulse2percept.implants.SequentialRaster`
    * :py:class:`pulse2percept.implants.CheckerboardRaster`
    * :py:class:`pulse2percept.implants.CustomRaster`
    * :ref:`Stimulus Encoders <topics-encoders>`
    * :ref:`Electrical Stimuli <topics-stimuli>`
    * :ref:`Physical Units <topics-units>`
