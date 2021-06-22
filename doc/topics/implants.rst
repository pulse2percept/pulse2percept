.. _topics-implants:

=================
Visual Prostheses
=================

Objects in the :py:mod:`~pulse2percept.implants` module are organized into the
following categories:

*  :ref:`Electrodes <topics-implants-electrode>` are objects whose behavior
   is dictated by the :py:class:`~pulse2percept.implants.Electrode` base class.

*  :ref:`Electrode arrays <topics-implants-electrode-array>` are
   collections of :py:class:`~pulse2percept.implants.Electrode` objects whose
   behavior is dictated by the
   :py:class:`~pulse2percept.implants.ElectrodeArray` class.

*  :ref:`Prosthesis systems <topics-implants-prosthesis-system>` (aka
   'retinal implants', aka 'bionic eye') are comprised of an
   :py:class:`~pulse2percept.implants.ElectrodeArray` object and (optionally) a
   :py:class:`~pulse2percept.stimuli.Stimulus` object. Their behavior is
   dictated by the :py:class:`~pulse2percept.implants.ProsthesisSystem` base
   class.

.. _topics-implants-prosthesis-system:

Prosthesis systems
------------------

pulse2percept provides the following prosthesis systems (aka 'retinal
implants', 'bionic eyes'):

==========  ==============  ==============  =================================
Implant     Location        Num Electrodes  Manufacturer
----------  --------------  --------------  ---------------------------------
`ArgusI`    epiretinal      16              Second Sight Medical Products Inc
`ArgusII`   epiretinal      60              Second Sight Medical Products Inc
`AlphaIMS`  subretinal      1500            Retina Implant AG
`AlphaAMS`  subretinal      1600            Retina Implant AG
`PRIMA`     subretinal      378             Pixium Vision SA
`PRIMA75`   subretinal      142             Pixium Vision SA
`PRIMA55`   subretinal      273(?)          Pixium Vision SA
`PRIMA40`   subretinal      532(?)          Pixium Vision SA
`BVA24`     suprachoroidal  24              Bionic Vision Technologies
==========  ==============  ==============  =================================

Stimuli can be assigned to the various electrodes in the electrode array,
who will deliver them to the retina
(see :ref:`Electrical Stimuli <topics-stimuli>`).
A mathematical model is then used to compute the neural stimulus response and
predict the resulting visual percept
(see :ref:`Computational Models <topics-models>`).

Understanding the coordinate system
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to understand the coordinate system is to look at the
organization of the optic fiber layer:

.. ipython:: python

    from pulse2percept.models import AxonMapModel
    AxonMapModel(eye='RE').plot()

Here you can see that:

*  the coordinate system is centered on the fovea
*  in a right eye, positive :math:`x` values correspond to the nasal retina
*  in a right eye, positive :math:`y` values correspond to the superior retina

Positive :math:`z` values move an electrode away from the retina into the
vitreous humor (:math:`z` is sometimes called electrode-retina distance).
Analogously, negative :math:`z` values move an electrode through the different
retinal layers towards the outer retina.

Understanding the ProsthesisSystem class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:class:`~pulse2percept.implants.ProsthesisSystem` base class provides
a template for all prosthesis systems. It is comprised of:

*  :py:class:`~pulse2percept.implants.ElectrodeArray`: as mentioned above,
*  :py:class:`~pulse2percept.stimuli.Stimulus`: as mentioned above,
*  :py:class:`~pulse2percept.implants.ProsthesisSystem.check_stim`: a method
   that quality-checks the stimulus. By default this method does nothing,
   but its behavior might depend on the actual system, such as
   :py:class:`~pulse2percept.implants.ArgusII` or
   :py:class:`~pulse2percept.implants.AlphaIMS`,
*  :py:attr:`~pulse2percept.implants.ProsthesisSystem.eye`: a string
   indicating whether the system is implanted in the left or right eye,
*  a means to access and iterate over electrodes in the array.

Accessing electrodes
^^^^^^^^^^^^^^^^^^^^

You can access individual electrodes in a prosthesis system either by integer
index or by electrode name. For example, the first electrode in
:py:class:`~pulse2percept.implants.AlphaAMS` can be accessed as follows:

.. ipython:: python

    from pulse2percept.implants import AlphaAMS
    implant = AlphaAMS()
    # Access by index:
    implant[0]

    # Access by name:
    implant['A1']

The simplest way to iterate over all electrodes is to pretend that the
prosthesis system is a Python dictionary:

.. ipython:: python

    from pulse2percept.implants import ArgusI
    for name, electrode in ArgusI().electrodes.items():
        print(name, electrode)


Creating your own prosthesis system
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can quickly create a prosthesis system from an
:py:class:`~pulse2percept.implants.ElectrodeArray` (or even a single
:py:class:`~pulse2percept.implants.Electrode`) by wrapping it in a
:py:class:`~pulse2percept.implants.ProsthesisSystem` container:

.. ipython:: python

    from pulse2percept.implants import ElectrodeGrid, ProsthesisSystem
    ProsthesisSystem(earray=ElectrodeGrid((10, 10), 200))

To create a more advanced prosthesis system, you will need to subclass the base
class:

.. code-block:: python

    import numpy as np
    from pulse2percept.implants import ElectrodeGrid, ProsthesisSystem

    class MyFovealElectrodeGrid(ProsthesisSystem):
        """An ElectrodeGrid implant centered over the fovea"""

        def __init__(self, stim=None, eye='RE'):
            self.earray = ElectrodeGrid((3, 3), x=0, y=0, z=0, rot=0,
                                        r=100, spacing=500,
                                        names=('A', '1'))
            self.stim = stim
            self.eye = eye

        def check_stim(self, stim):
            """Make sure the stimulus is charge-balanced"""
            if stim.time is not None:
                for s in stim:
                    assert np.isclose(np.sum(s), 0)

.. minigallery:: pulse2percept.implants.ProsthesisSystem
    :add-heading: Examples using ``ProsthesisSystem``
    :heading-level: ~

.. _topics-implants-electrode-array:

Electrode arrays
----------------

**Electrode arrays** are collections of
:py:class:`~pulse2percept.implants.Electrode` objects whose behavior is
dictated by the :py:class:`~pulse2percept.implants.ElectrodeArray` base class.

.. seealso::

    *  :py:class:`~pulse2percept.implants.ElectrodeGrid`

Understanding the ElectrodeArray class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:class:`~pulse2percept.implants.ElectrodeArray` base provides:

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

Accessing electrodes
^^^^^^^^^^^^^^^^^^^^

You can access individual electrodes in an electrode array either by integer
index or by electrode name. The syntax is exactly the same as for the
prosthesis system.

Creating your own electrode array
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can create your own electrode array by starting with an empty
:py:class:`~pulse2percept.implants.ElectrodeArray`, and adding the desired
electrodes one by one:

.. ipython:: python

    from pulse2percept.implants import DiskElectrode, ElectrodeArray
    earray = ElectrodeArray([])
    earray.add_electrode(0, DiskElectrode(0, 0, 0, 50))
    earray.add_electrode(1, DiskElectrode(100, 100, 0, 150))
    earray

To create a more advanced electrode array, you will need to subclass the base
class. In the constructor, make sure to initialize ``self.electrodes`` with an
ordered dictionary (``OrderedDict``):

.. code-block:: python

    from collections import OrderedDict
    from pulse2percept.implants import ElectrodeArray

    class MyElectrodeArray(ElectrodeArray):
        """Array with a single disk electrode"""

        def __init__(self, name):
            self.electrodes = OrderedDict()
            self.add_electrode(name, DiskElectrode(0, 0, 0, 100))

.. minigallery:: pulse2percept.implants.ElectrodeArray
    :add-heading: Examples using ``ElectrodeArray``
    :heading-level: ~

.. _topics-implants-electrode:

Electrodes
----------

**Electrodes** are objects whose behavior is dictated by the
:py:class:`~pulse2percept.implants.Electrode` base class.
They are located at a particular 3D location and provide a method to calculate
the electric potential at arbitrary 3D locations.

.. seealso::

   *  :py:class:`~pulse2percept.implants.PointSource`
   *  :py:class:`~pulse2percept.implants.DiskElectrode`
   *  :py:class:`~pulse2percept.implants.SquareElectrode`
   *  :py:class:`~pulse2percept.implants.HexElectrode`

Understanding the Electrode class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The base class provides:

*  the 3D coordinates of the center of the electrode.

In addition, a custom electrode object must implement:

*  a method called
   :py:meth:`~pulse2percept.implants.Electrode.electric_potential` that
   returns the electric potential at a point (x, y, z).

Creating your own electrode
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To create a new electrode type, you will need to subclass the base class.
Make sure to specify an ``electric_potential`` method for your class:

.. code-block:: python

    from pulse2percept.implants import Electrode

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

.. minigallery:: pulse2percept.implants.Electrode
    :add-heading: Examples using ``Electrode``
    :heading-level: ~
