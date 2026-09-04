.. _topics-units:

==============
Physical Units
==============

.. versionadded:: 0.10.0

Many pulse2percept parameters represent physical quantities. The
:py:mod:`~pulse2percept.units` module adds dimensional checking and automatic
conversion while keeping bare numbers fully supported and consistent with
older p2p versions.

.. code-block:: python

    from pulse2percept.stimuli import BiphasicPulse
    from pulse2percept.units import mA, us

    BiphasicPulse(50, 0.45)
    BiphasicPulse(0.05 * mA, 450 * us)  # equivalent

Bare numbers use the canonical unit documented by the API.

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

Objects that use a different unit, such as a Percept with its own time base,
record that unit explicitly.

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

Units prevent physically different quantities from being confused. For
example, retinal distance is not visual angle, and image intensity is not
current.

``dva`` therefore does not convert directly to ``um``. Use a
:py:class:`~pulse2percept.topography.VisualFieldMap` when mapping between
visual-field and tissue coordinates:

.. code-block:: python

    from pulse2percept.topography import Watson2014Map
    from pulse2percept.units import dva

    x_um, y_um = Watson2014Map().dva_to_ret(2 * dva, 3 * dva)

Likewise, :py:class:`~pulse2percept.stimuli.ImageStimulus` and
:py:class:`~pulse2percept.stimuli.VideoStimulus` are dimensionless. A
:py:class:`~pulse2percept.stimuli.StimulusEncoder` defines how their gray
levels become electrical stimulation.

Geometric angle
---------------

``deg`` and ``rad`` measure ordinary geometric angle: implant rotation, image
rotation, grating direction and phase, and axon polar angle. They convert into
each other freely, and bare numbers still mean degrees:

.. code-block:: python

    import numpy as np
    from pulse2percept.implants import ArgusII
    from pulse2percept.units import deg, rad

    ElectrodeGrid((6, 10), 575, rot=45)              # degrees, as before
    ElectrodeGrid((6, 10), 575, rot=45 * deg)        # equivalent
    ElectrodeGrid((6, 10), 575, rot=np.pi / 4 * rad) # equivalent

``dva`` is deliberately a different dimension: it describes coordinates or
extent in the visual field, whereas ``deg`` and ``rad`` describe geometric
rotation.
A visual field map converts visual-field coordinates to retinal or cortical
positions; it does not make ``dva`` interchangeable with geometric angle.

Threshold-relative amplitude
----------------------------

Some models (e.g., :py:class:`~pulse2percept.models.BiphasicAxonMapModel`)
operate on ``xTh``, which means a multiple of perceptual threshold, rather
than raw current.

Thresholds can be stored in ``implant.thresholds``. When they are set,
pulses expressed in ``xTh`` will be automatically converted to microamps:

.. code-block:: python

    from pulse2percept.stimuli import BiphasicPulseTrain
    from pulse2percept.units import uA, xTh

    train = BiphasicPulseTrain(20, 2 * xTh, 0.45)

    implant.thresholds = {'A4': 80 * uA}
    implant.prepare_stim({'A4': train})  # calibrated to 160 uA

Without a threshold, the train remains in ``xTh``. Current-based safety checks
and models require calibration to a physical current.

Documented shorthands
---------------------

A few APIs accept another dimension when there is one unambiguous physical
interpretation. In particular:

* retinal-model ``xrange`` and ``yrange`` may be specified as retinal lengths;
  the model's visual-field map converts the endpoints to visual angle;
* frame-rate arguments such as ``fps`` accept frequency quantities such as
  ``30 * Hz``.

These are API-level interpretations, not general unit conversions.

Inspecting units
----------------

Objects expose the units of their stored numbers and methods for reading them
in another compatible unit:

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

The unit system is intentionally small. It has no unit registry, string
parsing, or automatic NumPy propagation, and physical units do not enter
Cython or Torch kernels.
