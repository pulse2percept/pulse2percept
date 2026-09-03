# -*- coding: utf-8 -*-
"""
===============================================================================
Visual field maps and phosphene locations
===============================================================================

A :py:class:`~pulse2percept.topography.VisualFieldMap` describes how locations
in the visual field map onto retinal or cortical tissue. pulse2percept includes
several built-in maps.

Retinal maps derive from :py:class:`~pulse2percept.topography.RetinalMap`
and include:

* :py:class:`~pulse2percept.topography.Curcio1990Map` uses a linear retinal
  scaling of 280 microns per degree of visual angle (dva).
* :py:class:`~pulse2percept.topography.Watson2014Map` uses the nonlinear
  retinal magnification model from [Watson2014]_.
* :py:class:`~pulse2percept.topography.Watson2014DisplaceMap` additionally
  accounts for retinal ganglion-cell displacement near the fovea.

Cortical maps derive from :py:class:`~pulse2percept.topography.CorticalMap`
and include:

* :py:class:`~pulse2percept.topography.Polimeni2006Map` maps visual field
  locations onto V1, V2, and V3 using the wedge-dipole model from
  [Polimeni2006]_.
* :py:class:`~pulse2percept.topography.NeuropythyMap` uses Neuropythy to
  estimate subject-specific cortical maps from MRI data [Benson2018]_.

Visual field maps
-----------------

To see how the retinal maps differ, start with a regular grid in the visual
field:
"""

# sphinx_gallery_thumbnail_number = 4

import matplotlib.pyplot as plt
import numpy as np

import pulse2percept as p2p


grid = p2p.topography.Grid2D((-50, 50), (-50, 50), step=5)
grid.plot(style='scatter', use_dva=True)
plt.xlabel('x (dva)')
plt.ylabel('y (dva)')
plt.axis('square')

###############################################################################
# A model creates a similar grid during ``build`` and maps it onto the tissue
# coordinates required by its spatial model. The same visual-field grid looks
# different under the available retinal maps:

transforms = [
    p2p.topography.Curcio1990Map(),
    p2p.topography.Watson2014Map(),
    p2p.topography.Watson2014DisplaceMap(),
]

fig, axes = plt.subplots(ncols=3, sharey=True, figsize=(13, 4))
for ax, transform in zip(axes, transforms):
    grid.build(transform)
    grid.plot(style='cell', ax=ax)
    ax.set_title(transform.__class__.__name__)
    ax.set_xlabel('x (microns)')
    ax.set_ylabel('y (microns)')
    ax.axis('equal')

###############################################################################
# ``Curcio1990Map`` is a simple scaling, whereas ``Watson2014Map`` is
# nonlinear. ``Watson2014DisplaceMap`` adds the prominent foveal distortion
# caused by retinal ganglion-cell displacement.
#
# Perceptual consequences
# -----------------------
#
# These differences matter because the visual-field map determines where an
# implant lies relative to the simulated visual field. Here we stimulate an
# :py:class:`~pulse2percept.implants.AlphaAMS` with the same encoded image under
# each retinal map:

implant = p2p.implants.AlphaAMS()
stim = p2p.stimuli.LogoUCSB().encode(implant=implant)

fig, axes = plt.subplots(ncols=3, sharey=True, figsize=(13, 4))
for ax, transform in zip(axes, transforms):
    model = p2p.models.ScoreboardModel(
        implant=implant,
        xrange=(-6, 6),
        yrange=(-6, 6),
        visual_field_map=transform,
    )
    model.predict_percept(stim).plot(ax=ax)
    ax.set_title(transform.__class__.__name__)

###############################################################################
# The first two maps give very similar percepts at this scale. The RGC
# displacement map produces a much stronger foveal distortion.
#
# Subject-specific phosphene locations
# ------------------------------------
#
# A visual-field map is still only a canonical mapping. In an individual
# implant user, a phosphene may appear somewhere else than the map predicts.
# ``location_noise`` models this subject-specific variability without changing
# the underlying ``visual_field_map``.
#
# For electrode :math:`i`, pulse2percept draws a fixed visual-field offset
#
# .. math::
#
#    \mathbf{p}'_i = \mathbf{p}_i + \boldsymbol{\epsilon}_i,
#    \qquad
#    \boldsymbol{\epsilon}_i \sim \mathcal{N}(0, \sigma^2 I),
#
# where :math:`\mathbf{p}_i` is the canonical phosphene location and
# :math:`\sigma` is ``location_noise`` in dva. The offsets are fixed for a
# model instance, so repeated stimulation of the same simulated subject gives
# the same spatial distortion.

np.random.seed(1)

fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(9, 4))
for ax, noise, title in zip(
        axes,
        [None, 1],
        ['Canonical map', 'Subject-specific variation']):
    model = p2p.models.ScoreboardModel(
        implant=implant,
        xrange=(-6, 6),
        yrange=(-6, 6),
        visual_field_map=p2p.topography.Watson2014Map(),
        location_noise=noise,
    )
    model.predict_percept(stim).plot(ax=ax)
    ax.set_title(title)

###############################################################################
# ``location_noise`` changes where the percept appears, not the physical
# electrode locations or the canonical map itself. This makes it useful for
# simulating individual phosphene-location variability while keeping the same
# anatomical model.
#
# Cortical visual field maps
# --------------------------
#
# Cortical models use a
# :py:class:`~pulse2percept.topography.CorticalMap`. The standard choice is
# :py:class:`~pulse2percept.topography.Polimeni2006Map`, which maps the visual
# field onto V1, V2, and V3:

fig, axes = plt.subplots(ncols=2, figsize=(9, 4))

visual_field_map = p2p.topography.Polimeni2006Map(
    regions=['v1', 'v2', 'v3'])
model = p2p.models.cortex.ScoreboardModel(
    implant=p2p.implants.cortex.Orion(),
    visual_field_map=visual_field_map,
)
model.build()

visual_field_map.plot(ax=axes[0])
axes[0].set_title('Polimeni map')
model.plot(ax=axes[1])
axes[1].set_title('Model grid')
plt.show()

###############################################################################
# ``Polimeni2006Map`` has six parameters: the global scale ``k``, wedge-dipole
# parameters ``a`` and ``b``, and azimuthal shear parameters ``alpha1``,
# ``alpha2``, and ``alpha3`` for V1--V3. The defaults come from
# [Polimeni2006]_, but cortical retinotopy varies substantially across people.
# When subject-specific anatomy is available, a
# :py:class:`~pulse2percept.topography.NeuropythyMap` can provide a more
# individualized mapping.
#
# Creating your own visual field map
# ----------------------------------
#
# Custom retinal maps subclass
# :py:class:`~pulse2percept.topography.RetinalMap` and implement ``dva_to_ret``.
# An inverse ``ret_to_dva`` should also be provided when the mapping can be
# inverted. For example:
#
# .. code-block:: python
#
#     class MyVisualFieldMap(p2p.topography.RetinalMap):
#
#         def dva_to_ret(self, xdva, ydva):
#             return xdva, ydva
#
#         def ret_to_dva(self, xret, yret):
#             return xret, yret
#
# Pass the map to a model with
# ``visual_field_map=MyVisualFieldMap()``.
