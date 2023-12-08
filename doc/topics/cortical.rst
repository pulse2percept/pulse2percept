.. _topics-cortical:

==========================
Cortical Visual Prostheses
==========================

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

    @savefig score.png align=center
    model.plot(style="scatter", use_dva=True)