"""
============================================================================
Creating your own electrode array
============================================================================

This example shows how to create a new
:py:class:`~pulse2percept.implants.ElectrodeArray` object.

As the base class for all electrode arrays in pulse2percept, the
:py:class:`~pulse2percept.implants.ElectrodeArray` class provides a blue print
for the functionality that every electrode array should have.

First and foremost, an :py:class:`~pulse2percept.implants.ElectrodeArray`
contains a collection of :py:class:`~pulse2percept.implants.Electrode` objects,
and new electrodes can be added via the
:py:func:`~pulse2percept.implants.ElectrodeArray.add_electrodes` method.

In addition, individual electrodes in the array can be accessed by indexing
using either their pre-assigned names (a string) or their place in the array
(integer).

Arranging electrodes in a circle
--------------------------------

In this example, we want to build a new type of electrode array that arranges
all of its electrodes in a circle.

To do this, we need to create a new class ``CircleElectrodeArray`` that is
a child of :py:class:`~pulse2percept.implants.ElectrodeArray`:
"""

##############################################################################
# .. code-block:: python
#
#     class CircleElectrodeArray(ElectrodeArray):
#         """Electrodes arranged in a circle"""
#         ...
#
# This way, the ``CircleElectrodeArray`` class can access all public methods
# of :py:class:`~pulse2percept.implants.ElectrodeArray`.
#
# The constructor then has the job of creating all electrodes in the array
# and placing them at the appropriate location; for example, by using the
# :py:func:`~pulse2percept.implants.ElectrodeArray.add_electrodes` method.
#
# The constructor of the class should accept a number of arguments:
#
# - ``n_electrodes``: how many electrodes to arrange in a circle
# - ``radius``: the radius of the circle
# - ``x_center``: the x-coordinate of the center of the circle
# - ``y_center``: the y-coordinate of the center of the circle
#
# For simplicity, we will use :py:class:`~pulse2percept.implants.DiskElectrode`
# objects of a given radius (100um), although it would be relatively straightforward
# to allow the user to choose the electrode type.

from pulse2percept.implants import ElectrodeArray, DiskElectrode
import collections as coll
import numpy as np


class CircleElectrodeArray(ElectrodeArray):

    def __init__(self, n_electrodes, radius, x_center, y_center):
        """Electrodes arranged in a circle

        Electrodes will be named 'A0', 'A1', ...

        Parameters
        ----------
        n_electrodes : int
            how many electrodes to arrange in a circle
        radius : float
            the radius of the circle (microns)
        x_center, y_center : float
            the x,y coordinates of the center of the circle (microns),
            where (0,0) is the center of the fovea
        """
        # The job of the constructor is to create the electrodes. We start
        # with an empty collection:
        self.electrodes = coll.OrderedDict()
        # We then generate a number `n_electrodes` of electrodes, arranged on
        # the circumference of a circle:
        for n in range(n_electrodes):
            # Angular position of the electrode:
            ang = 2.0 * np.pi / n_electrodes * n
            # Create the disk electrode:
            electrode = DiskElectrode(x_center + np.cos(ang) * radius,
                                      y_center + np.sin(ang) * radius, 0, 100)
            # Add the electrode to the collection:
            self.add_electrode('A' + str(n), electrode)

##############################################################################
# Using the CircleElectrodeArray class
# ------------------------------------
#
# To use the new class, we need to specify all input arguments and pass them
# to the constructor:

n_electrodes = 10
radius = 1000  # radius in microns
x_center = 0  # x-coordinate of circle center (microns)
y_center = 0  # y-coordinate of circle center (microns)

# Create a new instance of type CircleElectrodeArray:
earray = CircleElectrodeArray(n_electrodes, radius, x_center, y_center)
print(earray)

##############################################################################
# Individual electrodes can be accessed by their name or integer index:

earray[0]

earray['A0']

earray[0] == earray['A0']

##############################################################################
# Visualizing the electrode array
# -------------------------------
#
# :py:func:`~pulse2percept.viz.plot_implant_on_axon_map` allows us to plot the
# location of a :py:class:`~pulse2percept.implants.ProsthesisSystem` with
# respect to the axon map in a simulated eye.
#
# Please note that an electrode array is only one part of a
# :py:class:`~pulse2percept.implants.ProsthesisSystem`. Other parts include the
# stimulus to be applied to each electrode, stimulus preprocessing methods,
# and safety checks.
#
# However, for the purpose of this tutorial, all we need to do is wrap the
# electrode array in a prosthesis system:
from pulse2percept.implants import ProsthesisSystem
implant = ProsthesisSystem(earray)

##############################################################################
# We can then visualize the array as follows:

import matplotlib.pyplot as plt
from pulse2percept.viz import plot_implant_on_axon_map
plot_implant_on_axon_map(implant)

##############################################################################
# Extending the CircleElectrodeArray class
# ----------------------------------------
#
# Similar to extending :py:class:`~pulse2percept.implants.ElectrodeArray` for
# our purposes, we can extend ``CircleElectrodeArray``.
#
# To add new functionality, we could simply edit the above constructor.
# However, nobody stops us from creating our own hierarchy of classes.
#
# For example, we could build a ``FlexibleCircleElectrodeArray`` that allows us to
# remove individual electrodes from the array:


class FlexibleCircleElectrodeArray(CircleElectrodeArray):

    def remove(self, name):
        """Deletean electrode from the array

        Parameters
        ----------
        name : int, string
            the name of the electrode to be removed
        """
        del self.electrodes[name]

##############################################################################
# Note how we didn't even specify a constructor.
# By default, the class inherits all (public) functionality from its parent,
# including its constructor. So the following line will create the same
# electrode array as above:

flex_earray = FlexibleCircleElectrodeArray(
    n_electrodes, radius, x_center, y_center)
print(flex_earray)

##############################################################################
# A single electrode can be removed by passing its name to the ``remove``
# method:

# Remove electrode 'A1'
flex_earray.remove('A1')

# Replot the implant on the axon map:
plot_implant_on_axon_map(ProsthesisSystem(flex_earray))
