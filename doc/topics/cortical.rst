.. _topics-cortical:

==========================
Cortical Visual Prostheses
==========================
Visual prostheses can either be implanted on the retina or the cortex.  
pulse2percept was originally designed for retinal implants, but now 
supports cortical implants.  In this section, we will cover cortical 
topography, implants, and models, and provide code examples of using them.

.. _topics-cortical-topography:

Topography
----------
The visual cortex is the part of our brain that processes visual information.
It is located at the back of our brain, and is split into two hemispheres:
left and right.  The visual cortex is divided into multiple regions, including
v1, v2, and v3, with each region performing a different function required
to process visual information.

Each region processes an aspect (such as color or motion) of the entire visual
field.  Within a region, different parts of the visual field are processed by
different neurons.  We can define a mapping between locations in the visual field
and locations in the cortex.  This mapping is called a visual field map, or
topography.

Model Plotting
^^^^^^^^^^^^^^
One way to visualize the mapping between the visual field and the cortex is
to plot a spatial model.  A spatial model consists of a set of points in the
visual field and the corresponding points in the cortex (using a visual field
map).  The plot of a model shows all of these points, either in the visual
field or on the cortex depending on the parameters used to create the plot.

The first step is to create a model, for example
:py:class:`~pulse2percept.models.cortex.ScoreboardModel`.  We can create the
model in regions v1, v2, and v3 as follows:

.. ipython:: python

    from pulse2percept.models.cortex import ScoreboardModel
    import matplotlib.pyplot as plt
    model = ScoreboardModel(regions=["v1", "v2", "v3"]).build()

Note the `model.build()` call.  This must be called before we can plot the
model.


If we want to plot the model in the visual field, we can do so by setting
`use_dva=True`.  If we use the style `"scatter"`, then we will be able to see
the points in the visual field.  The points in the visual field are evenly
spaced, and are represented by `+` symbols.

.. ipython:: python 
    :okwarning:

    @savefig score.png align=center
    model.plot(style="scatter", use_dva=True)

If we don't set `use_dva=True`, then the visual field mapping will be applied
to the points in the visual field, and the points on the cortex will be
plotted instead.  Because we created a model with three regions, each point
in the visual field will be transformed to three new points: one in v1, one
in v2, and one in v3.

The cortex is split into left and right hemispheres, with each side being
responsible for processing information from one eye.  In reality, the left
and right hemispheres of our brain are disconnected, but to simplify the
code, pulse2percept represents them as one continuous space.  The origin
of both hemispheres corresponds to the center of our visual field, as defined
by the visual field map.  However, because we represent both hemispheres as
one continuous space, this would result in both hemispheres overlapping
around the origin.

To avoid this, the left hemisphere are offset by 20mm, meaning the origin
of the left hemisphere is (-20, 0).  In addition, cortical visual field maps
have a `split_map` attribute set to `True`, which means that no current will
be allowed to cross between the hemispheres.

.. ipython:: python
    :okwarning:

    @savefig model_scatter.png align=center
    model.plot(style="scatter")

One effect that can be seen in the plot is that around the origins of each
hemisphere, the points are less dense.  This is because an area at the
center of our visual field is represented by a larger area on the cortex than
equally sized area at the periphery of our visual field, an effect called
cortical magnification.

Another style option for the plot is `"hull"`` (the default):

.. ipython:: python
    :okwarning:

    @savefig model_hull.png align=center
    model.plot(style="hull")

And the last style is `"cell"`:

.. ipython:: python
    :okwarning:

    @savefig model_cell.png align=center
    model.plot(style="cell")

Visual Field Mapping Plotting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can also directly plot visual field maps, such as
:py:class:`~pulse2percept.topography.Polimeni2006Map`, which is a cortical
map.  The origin corresponds to the fovea (center of our visual field).  The
units of the plot are in mm.  The plot also shows what part of the visual
field is represented by different areas along the cortex in dva.  This
shows the cortical magnification effect mentioned above, since for a given
area of the cortex near the fovea, a larger area of the visual field is
represented than the same area of the cortex near the periphery of the
visual field.

.. ipython:: python
    :okwarning:

    from pulse2percept.topography import Polimeni2006Map
    map = Polimeni2006Map()
    @savefig polimeni.png align=center
    map.plot()


.. _topics-cortical-implants:

Cortical Implants
-----------------

:py:class:`~pulse2percept.implants.cortex.Orion`, 
:py:class:`~pulse2percept.implants.cortex.Cortivis`, 
and :py:class:`~pulse2percept.implants.cortex.ICVP`  are cortical implants.
This tutorial will show you how to create and plot these implants.  Setting
`annotate=True` will show the implant names for each electrode.  The 
electrode names are useful if you want to add a stimulus to specific
electrodes.  For more information about these implants, see the documentation
for each specific implant.

Orion 
^^^^^

:py:class:`~pulse2percept.implants.cortex.Orion` is an implant with 60 
electrodes in a hex shaped grid.

.. ipython:: python

    from pulse2percept.implants.cortex import Orion

    orion = Orion()
    @savefig orion.png align=center
    orion.plot(annotate=True)

Cortivis
^^^^^^^^

:py:class:`~pulse2percept.implants.cortex.Cortivis` is an implant with 96 
electrodes in a square shaped grid.

.. ipython:: python

    from pulse2percept.implants.cortex import Cortivis

    cortivis = Cortivis()
    @savefig cortivis.png align=center
    cortivis.plot(annotate=True)

ICVP
^^^^

:py:class:`~pulse2percept.implants.cortex.ICVP` is an implant with 16 
primary electrodes in a hex shaped grid, along with 2 additional "reference" 
and "counter" electrodes.

.. ipython:: python

    from pulse2percept.implants.cortex import ICVP

    icvp = ICVP()
    @savefig icvp.png align=center
    icvp.plot(annotate=True)

.. _topics-ensemble-implant:

Ensemble Implants
-----------------

:py:class:`~pulse2percept.implants.EnsembleImplant` is a new class which
allows the user to use multiple implants in tandem. It can be used with any 
implant type, but was made for use with small implants meant to be used together,
such as :py:class:`~pulse2percept.implants.cortex.Cortivis`. This tutorial will 
demonstrate how to create an :py:class:`~pulse2percept.implants.EnsembleImplant`,
to combine multiple :py:class:`~pulse2percept.implants.cortex.Cortivis` objects.

The first step is to create the individual implants that will be combined.

.. ipython:: python
    :okwarning:

    i1 = Cortivis(x=15000,y=0)
    i2 = Cortivis(x=20000,y=0)
    i1.plot(annotate=True)
    i2.plot(annotate=True)
    @savefig cortivis_multiple.png align=center
    plt.show()

Then, we can create an EnsembleImplant using these two implants. 

.. ipython:: python

    from pulse2percept.implants import EnsembleImplant

    ensemble = EnsembleImplant(implants=[i1,i2])
    _,ax = plt.subplots(1, 1, figsize=(12,7))
    @savefig ensemble.png align=center
    ensemble.plot(annotate=True, ax=ax)

Note that electrodes are renamed, with the pattern `index-electrode` where `index`
is the index of the implant in the constructor list. Implants can also be passed using
a dictionary, in which case the naming pattern is `key-electrode` where `key` is the
electrode's dictionary key.

.. _topics-cortical-models:

Models
------

This example shows how to apply the
:py:class:`~pulse2percept.models.cortex.ScoreboardModel` to an
:py:class:`~pulse2percept.implants.cortex.Cortivis` implant.

First, we create the model and build it:

.. ipython:: python

    from pulse2percept.models.cortex import ScoreboardModel

    model = ScoreboardModel(rho=1000).build()

Next, we can create the implant:

.. ipython:: python

    from pulse2percept.implants.cortex import Cortivis

    implant = Cortivis()

Now, we can plot the model and implant together to see where the implant is
(by default, Cortivis is centered at (15,0))

.. ipython:: python
    :okwarning:

    model.plot()
    implant.plot()
    @savefig model_implant_cortivis.png align=center
    plt.show()

After that, we can add a stimulus to the implant.  One simple way to do this
is to create an array of the same shape as the implant (which has 96
electrodes), where each value in the array represents the current to apply
to the corresponding electrode.  For example, if we want to apply no current
to the first 32 electrodes, 1 microamp of current to the next 32 electrodes,
and 2 microamps of current to the last 32 electrodes, we can do the
following:

.. ipython:: python

    import numpy as np
    implant.stim = np.concatenate(
        (
            np.zeros(32),
            np.zeros(32) + 1,
            np.zeros(32) + 2,
        )
    )
    @savefig model_stim.png align=center
    implant.plot(stim_cmap=True)

In the implant plots, darker colors indicate low current and lighter colors
indicate high current (relative to the other currents).
Alternatively, we can set the current for specific electrodes by passing in
a dictionary, where the keys are the electrode names and the values are the
current to apply to that electrode.  For example, if we want to apply 1
microamp of current to the electrode named "15", 1.5 microamps of current
to the electrode named "37", and 0.5 microamps of current to the electrode
named "61", we can do the following:

.. ipython:: python

    implant.stim = {"15": 1, "37": 1.5, "61": 0.5}
    @savefig model_stim_specific.png align=center
    implant.plot(stim_cmap=True)

In order to make the stimulus more visible, we can use the larger
:py:class:`~pulse2percept.implants.cortex.Orion` implant instead.
We can add a current to the top 30 electrodes as follows:

.. ipython:: python

    from pulse2percept.implants.cortex import Orion

    implant = Orion()
    implant.stim = np.concatenate(
        (
            np.zeros(30),
            np.zeros(30) + 1,
        )
    )
    @savefig model_implant_orion.png align=center
    implant.plot(stim_cmap=True)

The final step is to run the model using `predict_percept`.  This will return
the calculated brightness at each location in the grid.  We can then plot
the brightness using the `plot` function:

.. ipython:: python

    percept = model.predict_percept(implant)
    @savefig model_percept.png align=center
    percept.plot()

The plot shows that the top half of the visual field has brightness.  If we
instead stimulate the bottom 30 electrodes:

.. ipython:: python

    implant.stim = np.concatenate(
        (
            np.zeros(30) + 1,
            np.zeros(30),
        )
    )
    @savefig model_stim_bottom.png align=center
    implant.plot(stim_cmap=True)

Then we will see that the bottom half of the visual field has brightness
instead.

.. ipython:: python

    percept = model.predict_percept(implant)
    @savefig model_percept_bottom.png align=center
    percept.plot()

If we move the implant closer to the periphery of the visual field, we can
see that the predicted percept is now larger due to cortical magnification:

.. ipython:: python

    implant = Orion(x=25000)
    implant.stim = np.concatenate(
        (
            np.zeros(30) + 1,
            np.zeros(30),
        )
    )
    percept = model.predict_percept(implant)
    @savefig model_stim_periphery.png align=center
    percept.plot()


.. _topics-cortical-developers:

For Developers
--------------

In this section we will discuss some of the changes made under the hood
accomadate cortical features, as well as some important notes for developers
to keep in mind.

Grid Plotting
^^^^^^^^^^^^^
Previously, the grid's plot function took in an optional `transform`
function that would be applied to all of the points.  This parameter has been
removed, and instead the plot function will automatically apply all of the
transforms in the grid's `retinotopy` attribute.  This is a dictionary of
different transforms that can be applied to the grid, such as dva to retinal
coordinates or dva to cortical coordinates.  If you want to plot the grid
without transformed points, you can pass in `use_dva=True`.

Units
^^^^^
Keep in mind that pulse2percept uses units of microns for length, microamps
for current, and milliseconds for time.

Topography
^^^^^^^^^^
Mappings from the visual field to cortical coordinates are implemented
as a subclass of :py:class:`~pulse2percept.topography.CorticalMap`,
such as :py:class:`~pulse2percept.topography.Polimeni2006Map`.  These
classes have a `split_map` attribute, which is set to `True` by default,
meaning that no current will be allowed to cross between the hemispheres.
These classes also have a `left_offset` attribute, which is set to 20mm by
default, meaning that the origin of the left hemisphere is (-20, 0) to
avoid overlapping with the right hemisphere.  This is visualized above in
the model plotting section.

In order to create your own visual field map, you must create a subclass of
:py:class:`~pulse2percept.topography.CorticalMap`, and implement the `dva_to_v1`
method.  In addition, if your map also maps to v2 and/or v3, you must also
implement the `dva_to_v2` and/or `dva_to_v3` methods. Optinally, you can also
implement `v1_to_dva`, `v2_to_dva`, and/or `v3_to_dva` methods.

For example, if you wanted to create a map that mapped `(x, y)` in dva to
`(x, y)` in v1, `(2x, 2y)` in v2, and `(3x, 3y)` in v3, you would do the
following (note that this is not a real map, and is only used for demonstration
purposes).  See 
:py:class:`~pulse2percept.topography.CorticalMap` for an example of a real map:

.. code-block:: python

    from pulse2percept.topography import CorticalMap
    import numpy as np

    class TestMap(CorticalMap):
        # Maps an array of points x, y in dva to an array of points x, y in v1
        def dva_to_v1(self, x, y):
            return x, y
        
        # Maps an array of points x, y in dva to an array of points x, y in v2
        def dva_to_v2(self, x, y):
            return 2 * x, 2 * y
        
        # Maps an array of points x, y in dva to an array of points x, y in v3
        def dva_to_v3(self, x, y):
            return 3 * x, 3 * y

    map = TestMap(regions=["v1", "v2", "v3"])

    points_dva_x = np.array([0, 1, 2])
    points_dva_y = np.array([3, 4, 5])

    points_v1 = map.from_dva()["v1"](points_dva_x, points_dva_y)
    points_v2 = map.from_dva()["v2"](points_dva_x, points_dva_y)
    points_v3 = map.from_dva()["v3"](points_dva_x, points_dva_y)

    print(f"Points in v1: {points_v1}")
    print(f"Points in v2: {points_v2}")
    print(f"Points in v3: {points_v3}")

Points in v1: (array([0, 1, 2]), array([3, 4, 5]))


Points in v2: (array([0, 2, 4]), array([ 6,  8, 10]))


Points in v3: (array([0, 3, 6]), array([ 9, 12, 15]))