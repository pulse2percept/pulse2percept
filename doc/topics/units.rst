.. _topics-units:

==============
Physical Units
==============

.. versionadded:: 0.10.0

The :py:mod:`~pulse2percept.units` module adds dimensional checking and
conversion to physical parameters. Bare numbers remain valid and use the
canonical unit below.

.. code-block:: python

    from pulse2percept.stimuli import BiphasicPulse
    from pulse2percept.units import mA, us

    BiphasicPulse(50, 0.45)
    BiphasicPulse(0.05 * mA, 450 * us)  # equivalent

Canonical units
---------------

.. list-table::
   :header-rows: 1

   * - Quantity
     - Bare number means
   * - stimulus current
     - microamps (``uA``)
   * - stimulus and percept time
     - milliseconds (``ms``)
   * - electrode and tissue geometry
     - microns (``um``)
   * - visual-field coordinates
     - degrees of visual angle (``dva``)
   * - geometric angle
     - degrees (``deg``)
   * - frequency
     - hertz (``Hz``)
   * - image and video intensity
     - dimensionless

Objects that store a different unit (e.g. a Percept with another time base)
record it in an attribute such as ``time_unit``.

Quantities and conversion
-------------------------

Multiply a number or array by a unit to create a
:py:class:`~pulse2percept.units.Quantity`:

.. doctest::

    >>> from pulse2percept.units import mA, uA
    >>> q = 500 * uA
    >>> q.to(mA)
    0.5 mA
    >>> q.to_value(mA)
    0.5

``to`` returns another Quantity; ``to_value`` returns a plain number or array.
Compatible quantities can be added, multiplied, divided, and raised to powers.

Dimensional boundaries
----------------------

Some quantities never convert into each other, and mixing them raises a
``DimensionMismatchError``:

*  ``dva`` and ``um``: convert through a
   :py:class:`~pulse2percept.topography.VisualFieldMap`
   (see :ref:`topics-coordinates`).
*  Gray levels and current: images and videos are dimensionless; an encoder
   converts them (see :ref:`topics-encoders`).
*  ``dva`` and ``deg``: see below.

Geometric angle
---------------

``deg`` and ``rad`` measure geometric angle: implant and image rotation,
grating direction and phase, axon polar angle. They convert into each other;
bare numbers mean degrees:

.. code-block:: python

    import numpy as np
    from pulse2percept.implants import ElectrodeGrid
    from pulse2percept.units import deg, rad

    ElectrodeGrid((6, 10), 575, rot=45)               # degrees
    ElectrodeGrid((6, 10), 575, rot=45 * deg)         # equivalent
    ElectrodeGrid((6, 10), 575, rot=np.pi / 4 * rad)  # equivalent

``dva`` is a separate dimension: a position or extent in the visual field,
not a rotation.

Threshold-relative amplitude
----------------------------

``xTh`` is a multiple of perceptual threshold. Some models (e.g.
:py:class:`~pulse2percept.models.retina.BiphasicAxonMapModel`) take amplitude
in ``xTh``. With ``implant.thresholds`` set, ``xTh`` amplitudes are converted
to uA when stimulation is prepared:

.. code-block:: python

    from pulse2percept.stimuli import BiphasicPulseTrain
    from pulse2percept.units import uA, xTh

    train = BiphasicPulseTrain(20, 2 * xTh, 0.45)

    implant.thresholds = {'A4': 80 * uA}
    implant.prepare_stim({'A4': train})  # calibrated to 160 uA

Without a threshold, the amplitude stays in ``xTh``. Current-based models
and safety checks require a threshold.

Documented shorthands
---------------------

A few parameters accept a second dimension where the meaning is
unambiguous. These are not general conversions:

*  Retinal-model ``xrange`` and ``yrange`` accept retinal lengths (converted
   to dva through the model's map).
*  Frame-rate arguments such as ``fps`` accept frequencies (``30 * Hz``).

Inspecting units
----------------

Objects expose the units of their stored numbers and return them in any
compatible unit:

.. code-block:: python

    stim.unit
    stim.values(mA)
    stim.time_unit
    stim.time_quantity

    implant.electrode_array.coordinates()
    implant.electrode_array.coordinates(mm)

    percept.time_unit
    percept.times(s)

Model parameter units are available through
:py:meth:`~pulse2percept.utils.Parametrized.get_param_units`.

The unit system is small: no unit registry, no string parsing, no automatic
NumPy propagation. Cython and Torch kernels receive plain numbers.
