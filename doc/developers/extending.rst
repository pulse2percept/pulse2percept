.. _dev-extending:

=======================
Extending pulse2percept
=======================

pulse2percept is designed to allow for implants, models, and stimuli to be
customized through `class inheritance`_, a concept from
`object-oriented programming`_.

.. note::

	If you are unfamiliar with these concepts, have a look at this `tutorial`_.

.. _class inheritance: https://docs.python.org/3/tutorial/classes.html#inheritance
.. _object-oriented programming: https://en.wikipedia.org/wiki/Object-oriented_programming
.. _tutorial: https://www.pythonforthelab.com/blog/a-primer-on-classes-in-python/

Creating your own implant
=========================

Objects in the :py:mod:`~pulse2percept.implants` module are organized into the
following categories:

*  **Electrodes** are objects whose behavior is dictated by the
   :py:class:`~pulse2percept.implants.Electrode` base class.

   The base class provides:

   *  the 3D coordinates of the center of the electrode.

   In addition, a custom electrode object must implement:

   *  a method called
      :py:meth:`~pulse2percept.implants.Electrode.electric_potential` that
      returns the electric potential at a point (x, y, z).

   A small working example:

   .. code-block:: python

       class MyElectrode(Electrode):
       	   """Named electrode with electric potential 0 everywhere"""

       	   def __init__(self, x, y, z, name):
       	       # Note: If you don't plan on adding any new variables, you can
       	       # omit the constructor entirely. In that case, your object will
       	       # inherit the constructor of the base class.
       	       self.x = x
       	       self.y = y
       	       self.z = z
       	       self.name = name

           def electric_potential(self, x, y, z):
               return 0.0

   .. seealso::

       *  :py:class:`~pulse2percept.implants.PointSource`
       *  :py:class:`~pulse2percept.implants.DiskElectrode`

*  **Electrode arrays** are collections of
   :py:class:`~pulse2percept.implants.Electrode` objects whose behavior is
   dictated by the :py:class:`~pulse2percept.implants.ElectrodeArray` base
   class.

   The base class provides:

   *  :py:attr:`~pulse2percept.implants.ElectrodeArray.electrodes`: an ordered
      dictionary of electrode objects (meaning it will remember the order in
      which electrodes were added),
   *  :py:attr:`~pulse2percept.implants.ElectrodeArray.n_electrodes`: a property
      returning the number of electrodes in the array.
   *  :py:meth:`~pulse2percept.implants.ElectrodeArray.add_electrode`: a method
      to add a single electrode to the collection,
   *  :py:meth:`~pulse2percept.implants.ElectrodeArray.add_electrodes`: a method
      to add a multiple electrodes to the collection at once,
   *  a way to access a single electrode either by index or by name,
   *  a way to iterate over all electrodes in the array.

   A small working example:

   .. code-block:: python

       class MyElectrodeArray(ElectrodeArray):
           """Array with a single disk electrode"""

           def __init__(self, name):
               self.electrodes = coll.OrderedDict()
               self.add_electrode(name, DiskElectrode(0, 0, 0, 100))

   .. seealso::

       *  :py:class:`~pulse2percept.implants.ElectrodeGrid`

*  **Prosthesis systems** ("retinal implants") are comprised of an
   :py:class:`~pulse2percept.implants.ElectrodeArray` object and (optionally)
   a :py:class:`~pulse2percept.stimuli.Stimulus`. Their behavior is dictated
   by the :py:class:`~pulse2percept.implants.ProsthesisSystem` base class.

   The base class provides:

   *  :py:class:`~pulse2percept.implants.ElectrodeArray`: as described above,
   *  :py:class:`~pulse2percept.stimuli.Stimulus`: as described above,
   *  :py:class:`~pulse2percept.implants.ProsthesisSystem.check_stim`: a method
      that quality-checks the stimulus. By default this method does nothing,
      but its behavior might depend on the actual system, such as
      :py:class:`~pulse2percept.implants.ArgusII` or
      :py:class:`~pulse2percept.implants.AlphaIMS`,
   *  :py:attr:`~pulse2percept.implants.ProsthesisSystem.eye`: a string
      indicating whether the system is implanted in the left or right eye,
   *  a means to access and iterate over electrodes in the array, as described
      above.

   A small working example:

   .. code-block:: python

       class MyFovealArgusII(ProsthesisSystem):
           """An Argus II implant centered over the fovea"""

           def __init__(self, stim=None):
               self.earray = ElectrodeGrid((6, 10), x=0, y=0, z=0, rot=0,
                                           r=100, spacing=525,
                                           names=('A', '1'))
               self.stim = stim

   .. seealso::

   	   *  :py:class:`~pulse2percept.implants.ArgusI`
   	   *  :py:class:`~pulse2percept.implants.ArgusII`
   	   *  :py:class:`~pulse2percept.implants.AlphaIMS`
   	   *  :py:class:`~pulse2percept.implants.AlphaAMS`

Creating your own stimulus
==========================

All stimuli described in the :py:mod:`~pulse2percept.stimuli` inherit their
functionality from the :py:class:`~pulse2percept.stimuli.Stimulus` base class.

The base class provides:

*  :py:attr:`~pulse2percept.stimuli.Stimulus.data`: A 2-D NumPy array,
   where the rows denote electrodes and the columns denote points in time,
*  :py:attr:`~pulse2percept.stimuli.Stimulus.shape`: the shape of the data
   array,
*  :py:meth:`~pulse2percept.stimuli.Stimulus.compress`: a method to compress
   the data container so that only nonredundant values are kept.
*  a means to compare two stimuli with the ``==`` or ``!=`` operators

A small working example:

.. code-block:: python

    class MyMonophasicPulse(Stimulus):
        """A monophasic pulse applied to a single electrode"""

        def __init__(self, pulse_amp, pulse_dur, stim_dur, time_step):
            # Provide only the time steps at which the signal changes. All
            # other time steps will be interpolated:
        	time = [0, pulse_dur, pulse_dur + time_step, stim_dur]
        	data = [pulse_amp, pulse_amp, 0, 0]
        	# Call the constructor of the base class:
        	super().__init__(self, data, time=time)

.. seealso:

    *  :py:class:`~pulse2percept.stimuli.TimeSeries`
    *  :py:class:`~pulse2percept.stimuli.MonophasicPulse`
    *  :py:class:`~pulse2percept.stimuli.BiphasicPulse`
    *  :py:class:`~pulse2percept.stimuli.PulseTrain`

Creating your own model
=======================

TODO after :pull:`96` is merged. API will change.
