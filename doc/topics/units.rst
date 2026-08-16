.. _topics-units:

==============
Physical Units
==============

.. versionadded:: 0.10.0

Every number in pulse2percept stands for a physical quantity: a current, a
duration, a distance on the retina, a position in the visual field. The
:py:mod:`~pulse2percept.units` module lets you say which, so that the library
can check you meant it:

.. code-block:: python

    from pulse2percept.stimuli import BiphasicPulse
    from pulse2percept.units import mA, us

    pulse = BiphasicPulse(50, 0.45)             # microamps and milliseconds
    pulse = BiphasicPulse(0.05 * mA, 450 * us)  # exactly the same pulse

Both lines mean the same thing and produce the same numbers. **Units are
optional.** Existing code that passes bare numbers keeps working and keeps its
documented meaning, and pulse2percept never warns about it.

What units buy you is that a mistake of *kind* becomes an error instead of a
result:

.. doctest::

    >>> from pulse2percept.stimuli import BiphasicPulse
    >>> from pulse2percept.units import ms, uA
    >>> BiphasicPulse(0.45 * ms, 50 * uA)  # arguments swapped
    ... # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      ...
    DimensionMismatchError: Parameter 'amp' expects electric current (uA), got time (ms).

They do not check magnitudes: ``450 * ms`` where you meant ``450 * us`` is
still a valid duration, and pulse2percept will happily build it.

What bare numbers mean
----------------------

Each domain has one canonical unit. A bare number is read as being in it, and
a unitful value is converted into it:

=============================  =====================================
Quantity                       A bare number means
-----------------------------  -------------------------------------
stimulus current               microamps (µA)
stimulus and percept time      milliseconds (ms) [#f1]_
electrode and tissue geometry  microns (µm)
visual field coordinates       degrees of visual angle (dva)
frequency                      hertz (Hz)
image and video intensity      dimensionless
rotation of an implant         plain degrees [#f2]_
=============================  =====================================

.. [#f1] Unless the object says otherwise. A
   :py:class:`~pulse2percept.percepts.Percept` records its own
   :py:attr:`~pulse2percept.percepts.Percept.time_unit`, and a model declares
   the unit its kernels count in; see `For model authors`_.

.. [#f2] ``rot`` is the angle an array is rotated by in its own plane. It is
   not a visual angle, so ``dva`` is refused there.

Working with quantities
-----------------------

Multiply a number by a unit to get a
:py:class:`~pulse2percept.units.Quantity`:

.. doctest::

    >>> from pulse2percept.units import uA, mA
    >>> q = 500 * uA
    >>> q
    500 uA
    >>> q.to(mA)
    0.5 mA
    >>> q.to_value(mA)
    0.5

:py:meth:`~pulse2percept.units.Quantity.to` gives you another quantity;
:py:meth:`~pulse2percept.units.Quantity.to_value` gives you a plain number and
is the explicit way to remove a unit. Nothing removes one implicitly.

Lists and arrays work too, and quantities do arithmetic:

.. doctest::

    >>> from pulse2percept.units import mA, ms, nC, s, uA
    >>> amps = [10, 20, 50] * uA
    >>> amps.to_value(mA)
    array([0.01, 0.02, 0.05])
    >>> 20 * ms + 0.03 * s
    50.0 ms
    >>> (3 * uA * (2 * ms)).to(nC)
    6 nC

Units compose, so a quantity the vocabulary does not name directly can still
be built out of the ones it does:

.. doctest::

    >>> from pulse2percept.units import uA, mm
    >>> 675 * uA / mm ** 2
    675 uA/mm^2

Boundaries you cannot cross
---------------------------

Dimensions are checked, so a quantity of the wrong kind is refused rather than
reinterpreted:

.. doctest::

    >>> from pulse2percept.implants import DiskElectrode
    >>> from pulse2percept.units import dva
    >>> DiskElectrode(2 * dva, 0, 0, 100)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      ...
    DimensionMismatchError: Parameter 'x' expects length (um), got visual angle (dva).

Two of these boundaries are worth stating outright, because they are the ones
that look like unit conversions and are not.

**Visual angle is not a length.** ``dva`` has its own dimension. There is no
factor that turns degrees of visual angle into microns of tissue, because the
relationship is not a constant: it depends on where in the visual field you
are, and on which map of the visual system you believe. That is what a
:py:class:`~pulse2percept.topography.VisualFieldMap` is for:

.. code-block:: python

    from pulse2percept.topography import Watson2014Map
    from pulse2percept.units import dva

    x_um, y_um = Watson2014Map().dva_to_ret(2 * dva, 3 * dva)

**Gray levels are not small currents.** An image or a video is dimensionless,
and a model that stimulates tissue will refuse one. An
:py:class:`~pulse2percept.stimuli.Encoder` is what says how much current a
gray level stands for:

.. code-block:: python

    from pulse2percept.stimuli import AmplitudeEncoder, ImageStimulus
    from pulse2percept.implants import ArgusII
    from pulse2percept.models import ScoreboardModel

    img = ImageStimulus('path-to-image.png')
    stim = AmplitudeEncoder(ArgusII(), amp_range=(0, 50)).encode(img)

    model = ScoreboardModel().build()
    percept = model.predict_percept(ArgusII(stim=stim))

Asking what an object's numbers mean
------------------------------------

Objects store plain numbers, and separately record what those numbers are.
Reading them back in another unit never changes what is stored:

.. code-block:: python

    stim.unit                            # uA
    stim.values(mA)                      # the data, in milliamps
    stim.time_unit                       # ms
    stim.time_quantity                   # the time axis, with its unit

    implant.earray.coordinates()         # (n, 3) array of microns
    implant.earray.coordinates(mm)       # the same array, in millimeters

    percept.time_unit                    # the model's own time unit
    percept.times(s)                     # the time axis, in seconds

Model parameters answer the same question through
:py:meth:`~pulse2percept.utils.Parametrized.get_param_units`:

.. doctest::

    >>> from pulse2percept.models import ScoreboardModel
    >>> ScoreboardModel().get_param_units()['rho']
    um

.. _units-for-model-authors:

For model authors
-----------------

A model declares the units its numerical implementation works in:

.. code-block:: python

    from pulse2percept.models import SpatialModel
    from pulse2percept.units import ms, uA, um

    class MyModel(SpatialModel):
        stimulus_unit = uA
        space_unit = um
        time_unit = ms

These are not decoration. Ask for what you consume, and the conversion happens
for you:

.. code-block:: python

    def _predict_spatial(self, earray, stim):
        x, y, z = self._electrode_coords(earray, stim)  # in space_unit
        amp = self._stim_values(stim)                   # in stimulus_unit
        t = self._stim_times(stim)                      # in time_unit
        return my_kernel(amp, x, y, z, t)               # plain float arrays

If your model has parameters of its own, declare them in
``get_param_units()`` and they will be normalized on assignment, whether
they arrive through the constructor, ``build()``, or a plain attribute set.

The rule for everything else: **convert at the Python boundary and keep
numerical kernels unitless.** By the time an array reaches Cython, NumPy or
Torch it is ordinary contiguous numeric data, exactly as it always has been.

Deliberately not supported
--------------------------

.. important::

    This is a small, fixed unit system, not a general one. It does not have:

    *  a unit registry or user-defined units — the vocabulary is the handful
       of units that appear in pulse2percept's own APIs, plus whatever you
       build out of them with ``*``, ``/`` and ``**``;
    *  string parsing — ``"5 mA"`` is a string, ``5 * mA`` is a quantity;
    *  automatic propagation through NumPy — a
       :py:class:`~pulse2percept.units.Quantity` is not an ``ndarray`` and
       does not survive ``np.mean`` or ``np.concatenate``;
    *  conversion between visual angle and length — that is a visual field
       map, not a unit;
    *  quantities inside Cython or Torch kernels;
    *  any requirement that you use units at all.

.. seealso::

    *  :py:mod:`pulse2percept.units` for the full API
    *  :ref:`Electrical Stimuli <topics-stimuli>`
    *  :ref:`Computational Models <topics-models>`
