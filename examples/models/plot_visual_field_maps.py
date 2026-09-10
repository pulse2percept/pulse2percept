# -*- coding: utf-8 -*-
"""
===============================================================================
Visual field maps and phosphene locations
===============================================================================

A :py:class:`~pulse2percept.topography.VisualFieldMap` describes how locations
in the visual field map onto retinal or cortical tissue. pulse2percept includes
several built-in maps.

Retinal maps derive from :py:class:`~pulse2percept.topography.retina.RetinalMap`
and include:

* :py:class:`~pulse2percept.topography.retina.Curcio1990Map`, which uses a linear
  retinal scaling of 280 microns per degree of visual angle (dva).
* :py:class:`~pulse2percept.topography.retina.Watson2014Map`, which uses the nonlinear
  retinal magnification model from [Watson2014]_.
* :py:class:`~pulse2percept.topography.retina.Montesano2020Map`, which adds a
  two-dimensional, meridian-dependent retinal ganglion-cell displacement field
  from [Montesano2020]_ on top of Watson's retinal magnification.

Cortical maps derive from :py:class:`~pulse2percept.topography.cortex.CorticalMap`
and include:

* :py:class:`~pulse2percept.topography.cortex.Polimeni2006Map`, which maps the visual
  field onto V1, V2, and V3 using the wedge-dipole model from [Polimeni2006]_.
* :py:class:`~pulse2percept.topography.cortex.NeuropythyMap`, which uses Neuropythy to
  estimate subject-specific cortical maps from MRI data [Benson2018]_.

Retinal visual field maps
-------------------------

To see how the retinal maps differ, start with a regular grid in the visual
field:
"""

# sphinx_gallery_thumbnail_number = 6

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
    p2p.topography.retina.Curcio1990Map(),
    p2p.topography.retina.Watson2014Map(),
    p2p.topography.retina.Montesano2020Map(eye='right'),
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
# nonlinear. ``Montesano2020Map`` keeps Watson's magnification and adds the
# foveal distortion caused by retinal ganglion-cell displacement: the cell
# bodies sit farther from the fovea than the receptive fields they serve.
#
# That displacement is meridian-dependent, which is easiest to see on the
# cardinal meridians. The displacement zone reaches 14.1 dva on the temporal
# and superior retina, but only 9.5 dva nasally and 10.5 dva inferiorly, so
# the four directions do not stop bending at the same eccentricity:

vfmap = p2p.topography.retina.Montesano2020Map(eye='right')
plain = p2p.topography.retina.Watson2014Map()
radius = np.linspace(0, 20, 400)

fig, ax = plt.subplots(figsize=(6, 4))
for angle, label in [(0, 'nasal'), (90, 'superior'),
                     (180, 'temporal'), (270, 'inferior')]:
    # The visual field mirrors the retina: an anatomical meridian of a right
    # eye sits at the negated visual-field polar angle.
    theta = np.deg2rad(-angle)
    x, y = radius * np.cos(theta), radius * np.sin(theta)
    displaced = np.hypot(*vfmap.dva_to_ret(x, y))
    ax.plot(radius, displaced - np.hypot(*plain.dva_to_ret(x, y)),
            label=label)
ax.set_xlabel('receptive-field eccentricity (dva)')
ax.set_ylabel('RGC displacement (microns)')
ax.legend(title='retinal meridian')

###############################################################################
# ``eye`` is what decides which side of the visual field is nasal retina and
# which is temporal, so a left eye is the horizontal mirror of a right eye.
# The vertical direction is the same in both.
#
# The field is population reference anatomy reconstructed from
# [Montesano2020]_ and [Curcio1990]_ histology, not subject-specific: how far
# an individual's ganglion cells are displaced, and where their fovea sits,
# both vary. ``Watson2014DisplaceMap`` remains available for reproducing
# earlier results, but it fits the horizontal meridian only and has no
# inverse; it is deprecated in favor of ``Montesano2020Map``.
#
# Cortical visual field maps
# --------------------------
#
# Cortical models use a
# :py:class:`~pulse2percept.topography.cortex.CorticalMap`. The standard choice is
# :py:class:`~pulse2percept.topography.cortex.Polimeni2006Map`, which maps the visual
# field onto V1, V2, and V3:

fig, axes = plt.subplots(ncols=2, figsize=(9, 4))

visual_field_map = p2p.topography.cortex.Polimeni2006Map(
    regions=['v1', 'v2', 'v3'])
model = p2p.models.cortex.ScoreboardModel(
    implant=p2p.implants.cortex.Orion(),
    implant_position=(15, 0) * p2p.units.mm,
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
# When subject-specific anatomy is available,
# :py:class:`~pulse2percept.topography.cortex.NeuropythyMap` can provide an
# individualized mapping.
#
# Subject-specific phosphene locations
# ------------------------------------
#
# Retinal and cortical maps describe a canonical relationship between tissue
# and visual-field location. An individual phosphene may appear somewhere else
# than that canonical map predicts. ``location_noise`` models this variability
# without changing the underlying ``visual_field_map``.
#
# For electrode :math:`i`, pulse2percept draws a fixed visual-field offset:
#
# .. math::
#
#    \mathbf{p}'_i = \mathbf{p}_i + \boldsymbol{\epsilon}_i,
#    \qquad
#    \boldsymbol{\epsilon}_i \sim \mathcal{N}(0, \sigma^2 I),
#
# where :math:`\mathbf{p}_i` is the canonical phosphene location and
# :math:`\sigma` is ``location_noise`` in dva. The offsets remain fixed for a
# model instance.
#
# A sparse retinal pattern makes the effect easiest to see. With the linear
# ``Curcio1990Map``, stimulating a regular subset of Argus II electrodes
# produces an approximately regular grid of phosphenes. Adding
# ``location_noise`` moves those phosphenes off the grid:

implant = p2p.implants.retina.ArgusII(raster=None)
stim = {
    electrode: 50
    for electrode in ['A1', 'A3', 'A5',
                      'C1', 'C3', 'C5',
                      'E1', 'E3', 'E5']
}

fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(9, 4))
for ax, noise, title in zip(
        axes,
        [None, 1],
        ['Canonical locations', 'Subject-specific locations']):
    np.random.seed(1)
    model = p2p.models.retina.ScoreboardModel(
        implant=implant,
        xrange=(-12, 2),
        yrange=(-7, 7),
        visual_field_map=p2p.topography.retina.Curcio1990Map(),
        location_noise=noise,
    )
    model.predict_percept(stim).plot(ax=ax)
    ax.set_title(title)

###############################################################################
# The phosphenes move; they do not change shape. Each one is still the circular
# Gaussian the Scoreboard model draws, sitting somewhere else in the visual
# field. On cortex the same is true of coherence but not of size: a displaced
# electrode samples a different cortical magnification, so its phosphene
# covers a different extent of the visual field.
#
# The same mechanism applies to cortical stimulation. Here a sparse subset of
# CORTIVIS electrodes is mapped through V1 with ``Polimeni2006Map``. The
# cortical implant and retinotopic map are unchanged; only the predicted
# phosphene locations differ:

implant_cortex = p2p.implants.cortex.Cortivis()
implant_position = (20, -5) * p2p.units.mm
cortex_coords = implant_cortex.electrode_array.coordinates(p2p.units.um)
stim_cortex = {
    electrode: 100
    for electrode, (x, y) in zip(implant_cortex.electrode_names,
                                 cortex_coords[:, :2])
    if round(x) in (-1400, 200, 1400) and round(y) in (-1400, 200, 1400)
}

fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(9, 4))
for ax, noise, title in zip(
        axes,
        [None, 0.5],
        ['Canonical locations', 'Subject-specific locations']):
    np.random.seed(2)
    model = p2p.models.cortex.ScoreboardModel(
        implant=implant_cortex,
        implant_position=implant_position,
        regions=['v1'],
        rho=300,
        xrange=(-4, 0.5),
        yrange=(-1, 3.5),
        step=0.02,
        location_noise=noise,
    )
    model.predict_percept(stim_cortex).plot(ax=ax)
    ax.set_title(title)

###############################################################################
# With a dense stimulus, the same per-electrode offsets scramble the percept as
# a whole. Here the retinal implant sees the same encoded UCSB logo with and
# without subject-specific phosphene locations:

implant = p2p.implants.retina.AlphaAMS()
stim = p2p.stimuli.samples.logo_ucsb().encode(implant=implant)

fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(9, 4))
for ax, noise, title in zip(
        axes,
        [None, 0.3],
        ['Canonical locations', 'Subject-specific locations']):
    np.random.seed(3)
    model = p2p.models.retina.ScoreboardModel(
        implant=implant,
        xrange=(-6, 6),
        yrange=(-6, 6),
        visual_field_map=p2p.topography.retina.Watson2014Map(),
        location_noise=noise,
    )
    model.predict_percept(stim).plot(ax=ax)
    ax.set_title(title)

###############################################################################
# The logo stays legible but its mosaic is spatially jumbled: the phosphenes
# remain coherent, but appear at subject-specific locations.
#
# ``location_noise`` changes the predicted percept, not the physical electrode
# locations or the canonical visual-field map. It therefore captures
# subject-specific phosphene-location variability while leaving the anatomical
# model intact. It requires an invertible map, so one without an inverse (such
# as the deprecated ``Watson2014DisplaceMap``) is not supported.
#
# Creating your own visual field map
# ----------------------------------
#
# Custom retinal maps subclass
# :py:class:`~pulse2percept.topography.retina.RetinalMap` and implement ``dva_to_ret``.
# An inverse ``ret_to_dva`` should also be provided when the mapping can be
# inverted. For example:
#
# .. code-block:: python
#
#     class MyVisualFieldMap(p2p.topography.retina.RetinalMap):
#
#         def dva_to_ret(self, xdva, ydva):
#             return xdva, ydva
#
#         def ret_to_dva(self, xret, yret):
#             return xret, yret
#
# Pass the map to a model with
# ``visual_field_map=MyVisualFieldMap()``.
