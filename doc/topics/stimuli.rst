.. _topics-stimuli:

==================
Electrical Stimuli
==================

.. ipython:: python
    :suppress:

    # Use defaults so we don't get gridlines in generated docs
    import matplotlib as mpl

    mpl.rcdefaults()

The :py:class:`~pulse2percept.stimuli.Stimulus` object defines a common
interface for all electrical stimuli.
It provides a 2-D data array with labeled axes, where rows denote electrodes
and columns denote points in time.
Stimuli can be assigned to electrodes of a
:py:class:`~pulse2percept.implants.ProsthesisSystem` object, who will deliver
them to the retina.

A stimulus can be created from a variety of source types, such as the
following:

* Scalar value: interpreted as the current amplitude delivered to a single
  electrode (no time component).
* NumPy array:
   * Nx1 array: interpreted as N current amplitudes delivered to N
     electrodes (no time component).
   * NxM array: interpreted as N electrodes each receiving M current
     amplitudes in time.
* :py:class:`~pulse2percept.stimuli.TimeSeries`: interpreted as the stimulus
  in time for a single electrode (e.g.,
  :py:class:`~pulse2percept.stimuli.PulseTrain`).

In addition, you can also pass a collection of source types (e.g., list,
dictionary).

See the :py:class:`~pulse2percept.stimuli.Stimulus` API documentation for a
full list.

.. note::
   Depending on the source type, a stimulus might have a time component or not.

Single-electrode stimuli
------------------------

The easiest way to create a stimulus is to specify the current amplitude (uA)
to be delivered to an electrode:

.. ipython:: python

    from pulse2percept.stimuli import Stimulus

    # Stimulate an unnamed electrode with -14uA:
    Stimulus(-14)

You can also specify the name of the electrode to be stimulated:

.. ipython:: python

    # Stimulate Electrode 'B7' with -14uA:
    Stimulus(-14, electrodes='B7')

By default, this stimulus will not have a time component
(``stim.time`` is None).
Some models, such as
:py:class:`~pulse2percept.models.ScoreboardModel`, cannot handle stimuli in
time.

To create stimuli in time, you can pass a
:py:class:`~pulse2percept.stimuli.TimeSeries` object, such as a
:py:class:`~pulse2percept.stimuli.BiphasicPulse` or a
:py:class:`~pulse2percept.stimuli.PulseTrain`:

.. ipython:: python

    # Stimulate Electrode 'A001' with a cathodic-first 20Hz pulse train
    # with 10uA amplitude, lasting for 0.5s, sampled at 0.1ms:
    from pulse2percept.stimuli import PulseTrain
    pt = PulseTrain(0.0001, freq=20, amp=10, dur=0.5)
    stim = Stimulus(pt)
    stim

    # This stimulus has a time component:
    stim.time

You can specify not only the name of the electrode but also the time steps to
be used:

.. ipython:: python

   # Stimulate Electrode 'C7' with int time steps:
   Stimulus(pt, electrodes='C7', time=np.arange(pt.shape[-1]))

Creating multi-electrode stimuli
--------------------------------

Stimuli can also be created from a list or dictionary of source types:

.. ipython:: python

    # Stimulate three unnamed electrodes with -2uA, 14uA, and -100uA,
    # respectively:
    Stimulus([-2, 14, -100])

Electrode names can be passed in a list:

.. ipython:: python

    Stimulus([-2, 14, -100], electrodes=['A1', 'B1', 'C1'])

Alternatively, stimuli can be created from a dictionary:

.. ipython:: python

    # Equivalent to the previous one:
    Stimulus({'A1': -2, 'B1': 14, 'C1': -100})

The same is true for a dictionary of pulse trains:

.. ipython:: python

    # Sending the same pulse train to three specific electrodes:
    Stimulus({'A1': pt, 'B1': pt, 'C1': pt})

Assigning new coordinates to an existing stimulus
-------------------------------------------------

You can change the coordinates of an existing
:py:class:`~pulse2percept.stimuli.Stimulus` object, but retain all its data,
as follows:

.. ipython:: python

    # Say you have a Stimulus object with unlabeled axes:
    stim = Stimulus(np.ones((2, 5)))
    stim

    # You can create a new object from it with named electrodes:
    Stimulus(stim, electrodes=['A1', 'F10'])

    # Same goes for time points:
    Stimulus(stim, time=[0, 0.1, 0.2, 0.3, 0.4])

Compressing a stimulus
----------------------

The :py:meth:`~pulse2percept.stimuli.Stimulus.compress` method automatically
compresses the data in two ways:

* Removes electrodes with all-zero activation.
* Retains only the time points at which the stimulus changes.

For example, only the signal edges of a pulse train are saved.
That is, rather than saving the current amplitude at every 0.1ms time step,
only the non-redundant values are retained.
This drastically reduces the memory footprint of the stimulus.
You can convince yourself of that by inspecting the size of a Stimulus object
before and after compression:

.. ipython:: python

    # An uncompressed stimulus:
    stim = Stimulus(PulseTrain(0.0001, freq=10), compress=False)
    stim

    # Now compress the data:
    stim.compress()

    # Notice how the stimulus shape and time axis have changed:
    stim

Interpolating stimulus values
-----------------------------

The :py:meth:`~pulse2percept.stimuli.Stimulus.interp` method interpolates
stimulus values at time points that are not explicitly provided:

.. ipython:: python

    # A single-electrode ramp stimulus:
    stim = Stimulus(np.arange(10).reshape((1, -1)))

    # Interpolate stimulus at a single time point:
    stim.interp(time=3.45)

    # Interpolate stimulus at multiple time points:
    stim.interp(time=[3.45, 6.78])

    # You can also extrapolate values outside the provided data range:
    stim.interp(time=123.45)

For a multi-electrode stimulus, the stimulus values at time t are returned
for all electrodes:

.. ipython:: python

    # Multi-electrode stimulus
    stim = Stimulus(np.arange(100).reshape((5, 20)))

    # Interpolate:
    stim.interp(time=4.5)

You can choose different interpolation methods, as long as
`scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_ accepts them.
For example, the 'nearest' method will return the value of the nearest
data point:

.. ipython:: python

    # A single-electrode ramp stimulus:
    stim = Stimulus(np.arange(10).reshape((1, -1)), interp_method='nearest')

    # Interpolate:
    stim.interp(time=3.45)

    # Outside the data range:
    stim.interp(time=12.2)
