# -*- coding: utf-8 -*-
"""
============================================================================
Beyeler et al. (2019): Axonal streaks with the axon map model
============================================================================

This example shows how to apply the
:py:class:`~pulse2percept.models.AxonMapModel` to an
:py:class:`~pulse2percept.implants.ArgusII` implant.

The axon map model assumes that electrical stimulation leads to percepts that
are elongated along the direction of the underlying nerve fiber bundle
trajectory. Because the layout of nerve fiber bundles in the human retina is
highly stereotyped [Jansonius2009]_, percept shape is predictable based on
(but also highly variable depending on) the location of the stimulating
electrode.

An axon's sensitivity to electrical stimulation is assumed to decay
exponentially:

*  with distance from the soma :math:`(x_{soma}, y_{soma})`, with spatial decay
   constant :math:`\\lambda`,
*  with distance from the stimulated retinal location
   :math:`(x_{stim}, y_{stim})`, with spatial decay constant :math:`\\rho`:

.. math::

    I_{axon}(x,y; \\rho, \\lambda) =& \\exp \\Big(
    -\\frac{(x-x_{stim})^2 + (y-y_{stim})^2}{2 \\rho^2} \\Big) \\\\
                                    & \\exp \\Big(
    -\\frac{(x-x_{soma})^2 + (y-y_{soma})^2}{2 \\lambda^2} \\Big).

The axon map model can be instantiated and run in three steps.

Choosing an implant
-------------------

A model predicts what a *particular* device produces, so the first step is to
specify a visual prosthesis from the :py:mod:`~pulse2percept.implants` module.

In the following, we will use an
:py:class:`~pulse2percept.implants.ArgusII` implant. By default, the implant
will be centered over the fovea (at x=0, y=0) and aligned with the horizontal
meridian (rot=0):

"""
# sphinx_gallery_thumbnail_number = 2

import numpy as np
from pulse2percept.implants import ArgusII
from pulse2percept.models import AxonMapModel

implant = ArgusII()

##############################################################################
# Creating the model
# ------------------
#
# The second step is to instantiate the
# :py:class:`~pulse2percept.models.AxonMapModel` class, bound to that implant.
# The two most important parameters to set are ``rho`` and ``lam`` from
# the equation above (here set to 150 micrometers and 500 micrometers,
# respectively):

model = AxonMapModel(implant=implant, rho=150, lam=500)

##############################################################################
# Parameters you don't specify will take on default values. You can inspect
# all current model parameters as follows:

print(model)

##############################################################################
# This reveals a number of other parameters to set, such as:
#
# * ``xrange``, ``yrange``: the extent of the visual field to be simulated,
#   specified as a range of x and y coordinates (in degrees of visual angle,
#   or dva). For example, we are currently sampling x values between -20 dva
#   and +20dva, and y values between -15 dva and +15 dva.
# * ``step``: The resolution (in dva) at which to sample the visual field.
#   For example, we are currently sampling at 0.25 dva in both x and y
#   direction.
# * ``loc_od_x``, ``loc_od_y``: the location of the center of the optic disc
#   (in dva)
# * ``thresh_percept``: You can also define a brightness threshold, below which
#   the predicted output brightness will be zero. It is currently set to
#   ``1/sqrt(e)``, because that will make the radius of the predicted percept
#   equal to ``rho``.
#
# A number of parameters control the amount of detail used when generating the
# axon map:
#
# * ``n_axons``: the number of axons to generate
# * ``axons_range``: the range of angles (in degrees) to use at which axon
#   trajectories emanate from the center of the optic disc
# * ``n_ax_segments``: the number of segments each generated axon should have
# * ``n_ax_segments_range``: the range of distances (in dva) to use, measured
#   from the center of the optic disc, at which axon segments should be placed
# * ``axons_pickle``: path to a pickle file where previously generated axon
#   maps are stored
#
# To change parameter values, either pass them directly to the constructor
# above or set them by hand.
#
# Before it can predict anything, the model performs a number of expensive
# setup computations (building the axon map, laying out the simulation grid).
# That happens automatically the first time you ask for a percept, and again
# whenever you change a model parameter. You can also trigger it yourself with
# ``model.build()`` -- useful when you would rather pay the cost at a moment of
# your choosing, or want to inspect the built axon map first.
#
# You can inspect the location of the implant with respect to the underlying
# nerve fiber bundles using the built-in plot methods:

model.plot()
implant.plot()


##############################################################################
# By default, the plots will be added to the current Axes object.
# Alternatively, you can pass ``ax=`` to specify in which Axes to plot.
#
# Predicting the percept
# ----------------------
# The third step is to hand the model a stimulus. The easiest kind is a NumPy
# array that specifies the current amplitude to be applied to every electrode
# in the implant.
#
# For example, the following sends 1 microamp to all 60 electrodes and predicts
# the resulting percept. Note that this may take some time on your machine:

percept = model.predict_percept(np.ones(60))

##############################################################################
# The resulting percept is stored in a
# :py:class:`~pulse2percept.percepts.Percept` object, which is similar in
# organization to the :py:class:`~pulse2percept.stimuli.Stimulus` object:
# the ``data`` container is a 3D NumPy array (Y, X, T) with labeled axes
# ``xdva``, ``ydva``, and ``time``.
#
# The percept can be plotted as follows:

ax = percept.plot()
ax.set_title('Predicted percept')

##############################################################################
# A major prediction of the axon map model is that the percept changes
# depending on the location of the implant. You can convince yourself of that
# by re-running the model on an implant shifted and rotated across the retina:

model.implant = ArgusII(x=-50, y=50, rot=-45)
model.plot()
model.implant.plot()

##############################################################################
# The resulting percepts should look very different from the previous example.
# Rebinding the implant invalidated the build, so the model rebuilds itself
# here:

percept = model.predict_percept(np.ones(60))
ax = percept.plot()
ax.set_title('Predicted percept')

##############################################################################
# .. important::
#
#     When specifying the rotation of the implant, positive angles will result
#     in counterclockwise rotations **on the retinal surface**.
#
#     However, because the superior (inferior) retina is mapped onto the lower
#     (upper) visual field, a counterclockwise orientation on the retina is
#     equivalent to a clockwise orientation of the percept in visual field
#     coordinates.

##############################################################################
# You can also use the axon map model to imitate
# :py:class:`~pulse2percept.models.ScoreboardModel` by setting lambda to a small
# value.
# However, you may have to increase the number of axons and number of segments
# per axon to get a smooth percept out:

model = AxonMapModel(implant=model.implant, rho=200, lam=10, n_axons=3000,
                     n_ax_segments=3000)
percept = model.predict_percept(np.ones(60))
ax = percept.plot()
ax.set_title('Predicted percept')

##############################################################################
# This is of course not very computationally efficient, because the model is
# still performing all the axon map calculations.
# In this case, you might be better off using
# :py:class:`~pulse2percept.models.ScoreboardModel`.
