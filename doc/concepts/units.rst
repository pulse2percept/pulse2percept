.. _topics-units:

==============
Physical Units
==============

.. versionadded:: 0.10.0

:mod:`pulse2percept.units` adds dimensional checking and unit conversion to
physical parameters.

Bare numbers remain valid. Each quantity has a canonical unit, so these two
calls are equivalent:

.. code-block:: python

    from pulse2percept.stimuli import BiphasicPulse
    from pulse2percept.units import mA, us

    BiphasicPulse(50, 0.45)
    BiphasicPulse(0.05 * mA, 450 * us)

When in doubt, use units.


Canonical Units
===============

.. list-table::
   :header-rows: 1
   :widths: 44 28 28

   * - Quantity
     - Bare number means
     - Unit
   * - Stimulus current
     - microamps
     - ``uA``
   * - Stimulus and percept time
     - milliseconds
     - ``ms``
   * - Electrode and tissue geometry
     - microns
     - ``um``
   * - Visual-field coordinates
     - degrees of visual angle
     - ``dva``
   * - Geometric angle
     - degrees
     - ``deg``
   * - Frequency
     - hertz
     - ``Hz``
   * - Image and video intensity
     - dimensionless
     - none

Objects that store values in another compatible unit record that unit
explicitly, for example through ``time_unit`` on a
:py:class:`~pulse2percept.percepts.Percept`.


Quantities and Conversion
=========================

Multiply a scalar or array by a unit to create a
:py:class:`~pulse2percept.units.Quantity`:

.. doctest::

    >>> from pulse2percept.units import mA, uA
    >>> q = 500 * uA
    >>> q.to(mA)
    0.5 mA
    >>> q.to_value(mA)
    0.5

The two conversion methods differ only in their return type:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Method
     - Returns
   * - ``q.to(unit)``
     - Another :py:class:`~pulse2percept.units.Quantity`
   * - ``q.to_value(unit)``
     - A plain scalar or NumPy array

Compatible quantities can be added, multiplied, divided, and raised to
powers.


Dimensions
==========

Units convert only within the same physical dimension.

For example:

.. list-table::
   :header-rows: 1
   :widths: 24 24 52

   * - Quantity 1
     - Quantity 2
     - Relationship
   * - ``uA``
     - ``mA``
     - Same dimension; conversion is direct
   * - ``ms``
     - ``s``
     - Same dimension; conversion is direct
   * - ``deg``
     - ``rad``
     - Same geometric-angle dimension
   * - ``dva``
     - ``um``
     - Different dimensions; conversion requires a
       :py:class:`~pulse2percept.topography.VisualFieldMap`
   * - Gray level
     - current
     - Different dimensions; conversion is performed by an encoder
   * - ``dva``
     - ``deg``
     - Different dimensions; visual position is not geometric rotation

Combining incompatible dimensions raises
:py:class:`~pulse2percept.units.DimensionMismatchError`.

See :ref:`topics-coordinates` for retinal/cortical geometry and
:ref:`topics-encoders` for conversion from visual intensity to stimulation.


Geometric Angle
===============

``deg`` and ``rad`` represent geometric rotation: implant orientation, image
rotation, grating direction and phase, and axon polar angle.

Bare angular values are interpreted as degrees:

.. code-block:: python

    import numpy as np

    from pulse2percept.implants import ElectrodeGrid
    from pulse2percept.units import deg, rad

    ElectrodeGrid((6, 10), 575, rot=45)
    ElectrodeGrid((6, 10), 575, rot=45 * deg)
    ElectrodeGrid((6, 10), 575, rot=np.pi / 4 * rad)

All three specify the same rotation.

``dva`` is intentionally separate. It describes position or extent in the
visual field, not rotation.


Threshold-Relative Amplitude
============================

``xTh`` represents multiples of perceptual threshold.

Some models, including
:py:class:`~pulse2percept.models.retina.BiphasicAxonMapModel`, use
threshold-relative amplitude rather than absolute current.

.. code-block:: python

    from pulse2percept.stimuli import BiphasicPulseTrain
    from pulse2percept.units import uA, xTh

    train = BiphasicPulseTrain(
        20,
        2 * xTh,
        0.45,
    )

    implant.thresholds = {
        'A4': 80 * uA,
    }

    implant.prepare_stim({
        'A4': train,
    })  # calibrated to 160 uA

When ``implant.thresholds`` is available, threshold-relative amplitudes are
converted to current during ``prepare_stim``.

Without a threshold value, the amplitude remains in ``xTh``. Models and safety
checks that require absolute current therefore also require threshold
calibration.


Accepted Shorthand Units
========================

A few parameters accept an additional dimension when the intended conversion
is defined by the surrounding object. These are API conveniences, not general
unit conversions.

.. list-table::
   :header-rows: 1
   :widths: 36 26 38

   * - Parameter
     - Also accepts
     - Conversion
   * - Retinal-model ``xrange`` and ``yrange``
     - retinal length
     - Converted to dva through the model's visual-field map
   * - Frame-rate arguments such as ``fps``
     - frequency
     - Converted to the corresponding frame rate

In particular, the unit system does not define a general ``um <-> dva``
conversion independently of a visual-field map.


Inspecting Units
================

Objects expose both their stored units and convenience methods for requesting
compatible units:

.. code-block:: python

    stim.unit
    stim.values(mA)
    stim.time_unit
    stim.time_quantity

    implant.electrode_array.coordinates()
    implant.electrode_array.coordinates(mm)

    percept.time_unit
    percept.times(s)

Parameter units defined by models and other parametrized objects are available
through
:py:meth:`~pulse2percept.utils.Parametrized.get_param_units`.
