# -*- coding: utf-8 -*-
"""
===============================================================================
Retinotopy: Predicting the perceptual effects of different visual field maps
===============================================================================

Every computational model needs to assume a mapping between retinal and visual
field coordinates. A number of these visual field maps are provided in the
:py:mod:`~pulse2percept.topography.geometry` module of the utilities subpackage:

*  :py:class:`~pulse2percept.topography.Curcio1990Map`: The [Curcio1990]_ model
   simply assumes that one degree of visual angle (dva) is equal to 280 um on
   the retina.
*  :py:class:`~pulse2percept.topography.Watson2014Map`: The [Watson2014]_ model
   extends [Curcio1990]_ by recognizing that the transformation between dva and
   retinal eccentricity is not linear (see Eq. A5 in [Watson2014]_). However,
   within 40 degrees of eccentricity, the transform is virtually
   indistuingishable from [Curcio1990]_.
*  :py:class:`~pulse2percept.topography.Watson2014DisplaceMap`: [Watson2014]_ also
   describes the retinal ganglion cell (RGC) density at different retinal
   eccentricities. In specific, there is a central retinal zone where RGC bodies
   are displaced centrifugally some distance from the inner segments of the
   cones to which they are connected through the bipolar cells, and thus from
   their receptive field (see Eq. 5 [Watson2014]_).

All of these visual field maps follow the
:py:class:`~pulse2percept.topography.VisualFieldMap` template.
This means that they have to specify a ``dva_to_ret`` method, which transforms
visual field coordinates into retinal coordinates, and a complementary 
``ret_to_dva`` method.

Visual field maps
-----------------

To appreciate the difference between the available visual field maps, let us
look at a rectangular grid in visual field coordinates:

"""
# sphinx_gallery_thumbnail_number = 3

import pulse2percept as p2p
import matplotlib.pyplot as plt

grid = p2p.topography.Grid2D((-50, 50), (-50, 50), step=5)
grid.plot(style='scatter', use_dva=True)
plt.xlabel('x (degrees of visual angle)')
plt.ylabel('y (degrees of visual angle)')
plt.axis('square')

###############################################################################
# Such a grid is typically created during a model's ``build`` process and
# defines at which (x,y) locations the percept is to be evaluated.
#
# However, these visual field coordinates are mapped onto different retinal
# coordinates under the three visual field maps:

transforms = [p2p.topography.Curcio1990Map(),
              p2p.topography.Watson2014Map(),
              p2p.topography.Watson2014DisplaceMap()]
fig, axes = plt.subplots(ncols=3, sharey=True, figsize=(13, 4))
for ax, transform in zip(axes, transforms):
    grid.build(transform)
    grid.plot(style='cell', ax=ax)
    ax.set_title(transform.__class__.__name__)
    ax.set_xlabel('x (microns)')
    ax.set_ylabel('y (microns)')
    ax.axis('equal')

###############################################################################
# Whereas the [Curcio1990]_ map applies a simple scaling factor to the visual
# field coordinates, [Watson2014]_ uses a nonlinear transform.
# One thing to note is the RGC displacement zone in the third panel, which might
# lead to distortions in the fovea.
#
# Perceptual distortions
# ----------------------
#
# The perceptual consequences of these visual field maps become apparent when
# used in combination with an implant.
#
# For this purpose, let us create an :py:class:`~pulse2percept.models.AlphaAMS`
# device on the fovea and feed it a suitable stimulus:

implant = p2p.implants.AlphaAMS(stim=p2p.stimuli.LogoUCSB())
implant.stim

###############################################################################
# We can easily switch out the visual field maps by passing a ``retinotopy``
# attribute to :py:class:`~pulse2percept.models.ScoreboardModel` (by default,
# the scoreboard model will use [Curcio1990]_):

fig, axes = plt.subplots(ncols=3, sharey=True, figsize=(13, 4))
for ax, transform in zip(axes, transforms):
    model = p2p.models.ScoreboardModel(xrange=(-6, 6), yrange=(-6, 6),
                                       retinotopy=transform)
    model.build()
    model.predict_percept(implant).plot(ax=ax)
    ax.set_title(transform.__class__.__name__)

###############################################################################
# Whereas the left and center panel look virtually identical, the rightmost
# panel predicts a rather striking perceptual effect of the RGC displacement
# zone.
#
# Creating your own visual field map
# ----------------------------------
#
# To create your own visual field map, you need to subclass the
# :py:class:`~pulse2percept.topography.RetinalMap` template and provide your own
# ``dva_to_ret`` and ``ret_to_dva`` methods.
# For example, the following class would (wrongly) assume that retinal
# coordinates are identical to visual field coordinates:
#
# .. code-block:: python
#
#     class MyVisualFieldMap(p2p.topography.VisualFieldMap):
#
#         def dva_to_ret(self, xdva, ydva):
#             return xdva, ydva
#
#         def ret_to_dva(self, xret, yret):
#             return xret, yret
#
# To use it with a model, you need to pass ``retinotopy=MyVisualFieldMap()``
# to the model's constructor.
