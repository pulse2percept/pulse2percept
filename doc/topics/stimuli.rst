.. _topics-stimuli:

==================
Electrical Stimuli
==================

The :py:mod:`~pulse2percept.stimuli` module provides a number of common
electrical stimulus types, which can be assigned to electrodes of a
:py:class:`~pulse2percept.implants.ProsthesisSystem` object:

================================  ==========================================
Stimulus                          Description
--------------------------------  ------------------------------------------
`MonophasicPulse`                 single phase: cathodic or anodic
`BiphasicPulse`                   biphasic: cathodic + anodic
`AsymmetricBiphasicPulse`         biphasic with unequal amplitude/duration
`PulseTrain`                      combine any Stimulus into a pulse train
`BiphasicPulseTrain`              series of (symmetric) biphasic pulses
`AsymmetricBiphasicPulseTrain`    series of asymmetric biphasic pulses
`BiphasicTripletTrain`            series of biphasic pulse triplets
================================  ==========================================

In addition, pulse2percept provides convenience functions to convert
images and videos into :py:class:`~pulse2percept.stimuli.Stimulus` objects
(see the :py:mod:`~pulse2percept.io` module).

.. important ::

    Stimuli specify electrical currents in microamps (uA) and time in
    milliseconds (ms). When in doubt, check the docstring of the function or
    class you are trying to use.

Understanding the Stimulus class
---------------------------------

The :py:class:`~pulse2percept.stimuli.Stimulus` object defines a common
interface for all electrical stimuli, consisting of a 2D data array with 
labeled axes, where rows denote electrodes and columns denote points in time.

A stimulus can be created from a variety of source types.
The number of electrodes and time points will be automatically extracted from
the source type:

================  ==========  ======
Source type       electrodes  time
----------------  ----------  ------
Scalar value      1           None
Nx1 NumPy array   N           None
NxM NumPy array   N           M
================  ==========  ======

In addition, you can also pass a collection of source types (e.g., list,
dictionary).

.. note::
   Depending on the source type, a stimulus might have a time component or not.

.. ipython:: python
    :suppress:

    # Use defaults so we don't get gridlines in generated docs
    import matplotlib as mpl
    mpl.rcdefaults()
    mpl.use('TkAgg')
    
Single-electrode stimuli
^^^^^^^^^^^^^^^^^^^^^^^^

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

To create stimuli in time, you can use one of the above mentioned stimulus
types, such as :py:class:`~pulse2percept.stimuli.MonophasicPulse` or
:py:class:`~pulse2percept.stimuli.BiphasicPulseTrain`:

.. ipython:: python

    # Stimulate Electrode 'A001' with a 20Hz pulse train lasting 0.5s
    # (pulses: cathodic-first, 10uA amplitude, 0.45ms phase duration):
    from pulse2percept.stimuli import BiphasicPulseTrain
    pt = BiphasicPulseTrain(20, 10, 0.45, stim_dur=500)
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

    from pulse2percept.stimuli import BiphasicPulse
    Stimulus({'A1': BiphasicPulse(10, 0.45, stim_dur=100),
              'C9': BiphasicPulse(-30, 1, delay_dur=10, stim_dur=100)})

Plotting stimuli
----------------

The easiest way to visualize a stimulus is to use the built-in
:py:meth:`~pulse2percept.stimuli.Stimulus.plot` method:

.. ipython:: python

    from pulse2percept.stimuli import Stimulus, BiphasicPulseTrain

    # Create a multi-electrode stimulus
    stim = Stimulus({'E%d' % i: BiphasicPulseTrain(i, 10, 0.45)
                     for i in np.arange(5)})
    # Plot it:
    stim.plot()

You can also select individual electrodes, or specify a range of time points:

.. ipython:: python

    # Plot two electrodes with available time points in the range t=[0, 0.5]:
    stim.plot(electrodes=['E2', 'E4'], time=(0, 0.5))

Interacting with stimuli
------------------------

Accessing individual data points
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can directly index into the :py:class:`~pulse2percept.stimuli.Stimulus`
object to retrieve individual data points: ``stim[item]``.
``item`` can be an integer, string, slice, or tuple.

For example, to retrieve all data points of the first electrode in a
multi-electrode stimulus, use the following:

.. ipython:: python

    stim = Stimulus(np.arange(10).reshape((2, 5)))
    stim[0]

Here ``0`` is a valid electrode index, because we did not specify an electrode
name. Analogously:

.. ipython:: python

    stim = Stimulus(np.arange(10).reshape((2, 5)), electrodes=['B1', 'C2'])
    stim['B1']

Similarly, you can retrieve all data points at a particular time:

.. ipython:: python

    stim = Stimulus(np.arange(10).reshape((2, 5)))
    stim[:, 3]

.. important ::

    The second index or slice into ``stim`` is not a column index into
    ``stim.data``, but an exact time specified in ms!
    For example, ``stim[:, 3]`` translates to "retrieve all data points at
    time = 3 ms", not "retrieve stim.data[:, 3]".

This works even when the specified time is not explicitly provided in the
stimulus!
In that case, the value is automatically interpolated (using SciPy's 
``interp1d``):

.. ipython:: python

    # A single-electrode ramp stimulus:
    stim = Stimulus(np.arange(10).reshape((1, -1)))
    stim

    # Retrieve stimulus at t=3:
    stim[0, 3]

    # Time point 3.45 is not in the data provided above, but can be
    # interpolated as follows:
    stim[0, 3.45]

    # This also works for multiple time points:
    stim[0, [3.45, 6.78]]
    
    # Extrapolating is disabled by default, but you can enable it:
    stim = Stimulus(np.arange(10).reshape((1, -1)), extrapolate=True)
    stim[0, 123.45]

You can choose different interpolation methods, as long as
`scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_ accepts them.
For example, the 'nearest' method will return the value of the nearest
data point:

.. ipython:: python

    # A single-electrode ramp stimulus:
    stim = Stimulus(np.arange(10).reshape((1, -1)), interp_method='nearest',
                    extrapolate=True)

    # Interpolate:
    stim[0, 3.45]

    # Outside the data range:
    stim[0, 12.2]

Accessing the raw data
^^^^^^^^^^^^^^^^^^^^^^

The raw data is accessible as a 2D NumPy array (electrodes x time) stored in
the ``data`` container of a Stimulus:

.. ipython:: python

    stim = Stimulus(np.arange(10).reshape((2, 5)))
    stim.data

You can index and slice the ``data`` container like any NumPy array.

Assigning new coordinates to an existing stimulus
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can change the coordinates of an existing
:py:class:`~pulse2percept.stimuli.Stimulus` object, but retain all its data,
by wrapping it in a second Stimulus object:

.. ipython:: python

    # Say you have a Stimulus object with unlabeled axes:
    stim = Stimulus(np.ones((2, 5)))
    stim

    # You can create a new object from it with named electrodes:
    Stimulus(stim, electrodes=['A1', 'F10'])

    # Same goes for time points:
    Stimulus(stim, time=[0, 0.1, 0.2, 0.3, 0.4])

Compressing a stimulus
^^^^^^^^^^^^^^^^^^^^^^

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
    stim = Stimulus([[0, 0, 0, 1, 2, 0, 0, 0]], time=[0, 1, 2, 3, 4, 5, 6, 7])
    stim

    # Now compress the data:
    stim.compress()

    # Notice how the time axis have changed:
    stim

