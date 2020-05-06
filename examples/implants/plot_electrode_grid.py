"""
============================================================================
Creating a grid of electrodes
============================================================================

This example shows how to use
:py:class:`~pulse2percept.implants.ElectrodeGrid`.

Most current electrode arrays arrange their electrodes in a 2D grid.

Creating a rectangular grid
---------------------------

To create a rectangular grid, we need to specify:

-  The ``shape`` of the grid, passed as a tuple containing the desired number
   of rows and columns
-  The electrode-to-electrode ``spacing`` in microns
-  The (``x``, ``y``) location of the center of the array
-  The rotation angle ``rot`` of the grid in radians, where positive angles
   rotate all electrodes in the array in a counter-clockwise fashion on the
   retinal surface.

We can also specify:

-  A naming convention ``names`` for the rows and columns in the grid.
   For example, to label rows alphabetically and columns numerically, we would
   pass a tuple ``('A', '1')``. To label both alphabetically, we would pass
   ``('A', 'A')``.
-  An electrode type ``etype``, which must be a subclass of
   :py:class:`~pulse2percept.implants.Electrode`. By default,
   :py:class:`~pulse2percept.implants.PointSource` is chosen.
-  Any additional parameters that should be passed to the
   :py:class:`~pulse2percept.implants.Electrode` constructor, such as a radius
   ``r`` for :py:class:`~pulse2percept.implants.DiskElectrode`.

Let's say we want to create a 2x3 rectangular grid of
:py:class:`~pulse2percept.implants.PointSource` objects, each electrode spaced
500 microns apart, and the whole grid should be centered over the fovea:

"""
# sphinx_gallery_thumbnail_number = 3
from pulse2percept.implants import ElectrodeGrid

grid = ElectrodeGrid((2, 3), 500)

##############################################################################
# We can access individual electrodes by indexing into ``grid``:
#
# The first electrode:

grid[0]

##############################################################################
# The first electrode by name:

grid['A1']

##############################################################################
# Accessing the x-coordinate of the first electrode:

grid[0].x

##############################################################################
# Showing all electrodes:

grid[:]

##############################################################################
# We can iterate over all electrodes as if we were dealing with a dictionary:

for name, electrode in grid.items():
    print(name, electrode)

##############################################################################
# To make a grid of :py:class:`~pulse2percept.implants.DiskElectrode` objects,
# we need to explicitly specify the electrode type (``etype``) and the radius
# to use (``r``):

from pulse2percept.implants import DiskElectrode

disk_grid = ElectrodeGrid((3, 5), 800, etype=DiskElectrode, r=100)

disk_grid[:]

##############################################################################
# .. note::
#
#     You can also specify a list of radii, one value for each electrode in
#     the grid.
#
# We can visualize the grid by wrapping it in a
# :py:class:`~pulse2percept.implants.ProsthesisSystem` object and passing it
# to :py:func:`~pulse2percept.viz.plot_implant_on_axon_map`:

from pulse2percept.implants import ProsthesisSystem
from pulse2percept.viz import plot_implant_on_axon_map

plot_implant_on_axon_map(ProsthesisSystem(disk_grid))

##############################################################################
# Creating a hexagonal grid
# -------------------------
#
# To create a hexagonal grid instead, all we need to do is change the grid type
# from 'rect' (default) to 'hex':

hex_grid = ElectrodeGrid((3, 5), 800, type='hex', etype=DiskElectrode, r=100)

plot_implant_on_axon_map(ProsthesisSystem(hex_grid))

##############################################################################
# The following example centers the grid on (x,y) = (-1000um, 200 um),
# z=150um away from the retinal surface, and rotates it clockwise by 45 degrees
# (note the minus sign):

from numpy import pi
offset_grid = ElectrodeGrid((3, 5), 800, type='hex', x=-1000, y=200, z=150,
                            rot=-pi / 4, etype=DiskElectrode, r=100)

plot_implant_on_axon_map(ProsthesisSystem(offset_grid))

##############################################################################
# .. note::
#
#     Clockwise/counter-clockwise rotations refer to rotations on the retinal
#     surface (that is, as if seen on a fundus photograph).
#
#     Since the inferior retina maps to the upper visual field, you would have
#     to flip the image upside-down to see how electrode location maps to
#     the location of a visual percept. You can do this by passing
#     ``upside_down=True``
#     to :py:func:`~pulse2percept.viz.plot_implant_on_axon_map`
