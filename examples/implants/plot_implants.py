"""
===============================================================================
Retinal implant gallery
===============================================================================

pulse2percept supports the following implants:

Argus Retinal Prosthesis System (Second Sight Medical Products Inc.)
--------------------------------------------------------------------

:py:class:`~pulse2percept.implants.ArgusI` and
:py:class:`~pulse2percept.implants.ArgusII` are epiretinal implants
developed at the University of Southern California (USC) and commercialized
by Second Sight. The devices were used in several clinical trials, including
`NCT00279500`_ and `NCT00407602`_.

Argus I is a modified cochlear implant containing 16 electrodes in a 4x4
array with a center-to-center separation of 800 um, and two electrode
diameters (250 um and 500 um) arranged in a checkerboard pattern [Yue2020]_.

Argus II contains 60 electrodes of 225 um diameter arranged in a 6 x 10
grid (575 um center-to-center separation) [Yue2020]_.

.. _NCT00279500: https://clinicaltrials.gov/ct2/show/NCT00279500
.. _NCT00407602: https://www.clinicaltrials.gov/ct2/show/NCT00407602

"""
# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
from pulse2percept.implants import *
from pulse2percept.models import AxonMapModel

fig, ax = plt.subplots(ncols=2, figsize=(10, 6))

# Argus I and II are typically implanted at a 30-45deg angle. For illustrative
# purpose, also show the map of fiber bundles in the optic fiber layer -- an
# axon map is grown for a particular eye, so each model names its implant:
for axis, implant, title in [(ax[0], ArgusI(rot=-30), 'Argus I'),
                             (ax[1], ArgusII(rot=-30), 'Argus II')]:
    AxonMapModel(implant=implant).plot(ax=axis)
    implant.plot(ax=axis, annotate=title == 'Argus I')
    axis.set_title(title)

###############################################################################
# PRIMA Bionic Vision System (Pixium Vision SA)
# ----------------------------------------------
#
# :py:class:`~pulse2percept.implants.PRIMA` is a subretinal device developed
# at Stanford University and commercialized by Pixium Vision.
#
# There are several versions of the PRIMA device. Each is a hexagonal array
# of photovoltaic pixels, described here by three numbers: the *pixel width*
# (flat-to-flat width of the hexagonal pixel body), the *center spacing*
# between nearest-neighbor pixels, and the *row spacing* between adjacent
# rows, which follows from the other two as ``spacing * sqrt(3) / 2``.
#
# The device used in clinical trial `NCT03392324`_ has 378 pixels 100um wide
# on a 100um center spacing (86.6um row spacing), on a 2 x 2 mm substrate, and
# is available in pulse2percept simply as
# :py:class:`~pulse2percept.implants.PRIMA` [Palanker2020]_.
#
# :py:class:`~pulse2percept.implants.PRIMA75` is an earlier 75um prototype,
# with 142 pixels 70um wide on a 75um center spacing, leaving a 5um open
# trench between pixel bodies, on a 1 mm substrate [Lorach2015]_. Its 65um
# row spacing is what [Lorach2015]_ calls the "pixel pitch".
#
# .. _NCT03392324: https://www.clinicaltrials.gov/ct2/show/NCT03392324

fig, ax = plt.subplots(ncols=2, figsize=(10, 6))

PRIMA().plot(ax=ax[0])
ax[0].set_title('PRIMA-100')

PRIMA75().plot(ax=ax[1])
ax[1].set_title('PRIMA-75')

###############################################################################
# In addition, the developers are working on miniaturizing the device, in two
# distinct device families. :py:class:`~pulse2percept.implants.Ho2019FlatArray`
# models the experimental flat F55 and F40 arrays of [Ho2019]_, which have 250
# and 502 pixels on a 1 mm circular substrate. (``PRIMA55`` and ``PRIMA40``
# are deprecated wrappers for ``Ho2019FlatArray(55)`` and
# ``Ho2019FlatArray(40)``: both families contain a 55um and a 40um device, so
# the old names are ambiguous. [Ho2019]_ also describes pillar arrays Pil55
# and Pil40, which pulse2percept does not model.)
#
# Unlike PRIMA-75, these arrays have no open trench between pixels: their
# 1um isolation trenches are covered by the shared return electrode, so the
# pixel bodies tile the array and the pixel width equals the center spacing,
# 55um and 40um (47.6um and 34.6um row spacing, which [Ho2019]_ calls the
# "pixel pitch"). Their active electrodes are 14um and 10um in diameter.
#
# [Ho2019]_ gives the pixel count and the substrate diameter, and the F55
# outline is reconstructed from its Fig. 2(a). The F40 outline is not
# published, so pulse2percept places its 502 pixels on the lattice sites
# nearest the center of the substrate.

fig, ax = plt.subplots(ncols=2, figsize=(10, 6))

Ho2019FlatArray(55).plot(ax=ax[0])
ax[0].set_title('Ho2019FlatArray(55) - F55')

Ho2019FlatArray(40).plot(ax=ax[1])
ax[1].set_title('Ho2019FlatArray(40) - F40')

###############################################################################
# :py:class:`~pulse2percept.implants.Huang2021Array` models the later
# vertical-junction arrays of [Huang2021]_, which come in 55um, 40um, 30um and
# 20um pixel sizes on a 1.5 mm circular substrate. Pixel bodies again tile the
# lattice with no open gap, and the active electrode at the center of each
# pixel is 40% of the pixel size across (22um, 16um, 12um and 8um).
#
# ``n_electrodes`` counts only the exposed, stimulating pixels: 421, 821, 1388
# and 2806. The die carries more pixels than that (526, 1027, 1735 and 3508),
# but a peripheral common return electrode covers roughly 20% of them. Those
# covered pixels are not exposed as independently stimulating pixels, so they
# are not modeled as individually addressable electrodes; ``n_total_pixels``
# reports the fabricated count.
#
# The outlines are reconstructed from Fig. 7 of [Huang2021]_, registered to
# the triangular lattice, and constrained to reproduce the published
# exposed-pixel counts. Three rim pixels of the 20um layout fall outside the
# published quadrant and are inferred; only one of the three is meaningfully
# ambiguous.

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

for axis, pixel_size in zip(ax.ravel(), [55, 40, 30, 20]):
    implant = Huang2021Array(pixel_size)
    implant.plot(ax=axis)
    axis.set_title(f'Huang2021Array({pixel_size}) - '
                   f'{implant.n_electrodes} of {implant.n_total_pixels} '
                   f'pixels exposed')

###############################################################################
# BVT Bionic Eye System (Bionic Vision Technologies)
# --------------------------------------------------
#
# :py:class:`~pulse2percept.implants.BVT24` is a 24-channel suprachoroidal
# retinal prosthesis [Layton2014]_, which was developed by the Bionic Vision
# Australia Consortium and commercialized by Bionic Vision Technologies (BVT).
#
# Note that the array actually consists of a total of 35 electrodes:
#
# -  33 platinum stimulating electrodes:
#
#    -  30 electrodes with 600um diameter (Electrodes 1-20 except
#       9, 17, 19; and Electrodes 21a-m),
#
#    -  3 electrodes with 400um diameter (Electrodes 9, 17, 19)
#
# -  2 return electrodes with 2000um diameter (Electrodes 22, 23)
#
# However, Electrodes 21a-m are typically ganged together to provide an
# external ring for common ground. Not counting the two large return electrodes
# leaves 24 stimulating electrodes.

fig, ax = plt.subplots(figsize=(10, 6))

BVT24().plot(ax=ax, annotate=True)
ax.set_title('BVT-24')

###############################################################################
# Alpha-IMS and Alpha-AMS Retinal Implant System (Retina Implant AG)
# ------------------------------------------------------------------
#
# :py:class:`~pulse2percept.implants.AlphaIMS` and
# :py:class:`~pulse2percept.implants.AlphaAMS` are subretinal implants
# developed at the University of Tuebingen, Germany and commercialized by
# Retina Implant AG.
#
# Alpha-IMS consists of 1500 50um-wide square pixels, arranged on a 39x39
# rectangular grid with 72um pixel pitch [Stingl2013]_.
#
# Alpha-AMS is the second generation device, consisting 1600 30um-wide round
# pixels, arranged on a 40x40 rectangular grid with 70um pixel pitch
# [Stingl2017]_.

fig, ax = plt.subplots(ncols=2, figsize=(10, 6))

AlphaIMS().plot(ax=ax[0])
ax[0].set_title('Alpha-IMS')

AlphaAMS().plot(ax=ax[1])
ax[1].set_title('Alpha-AMS')

###############################################################################
# Intelligent Micro Implant Eye epiretinal prosthesis system (IMIE)
# ------------------------------------------------------------------
#
# :py:class:`~pulse2percept.implants.IMIE` is an epiretinal implant co-developed
# by Golden Eye Bionic, LLC (Pasadena CA) and IntelliMicro Medical Co., Ltd. 
# (Changsha, Hunan Province, China) and is manufactured by IntelliMicro. 
#
# IMIE consists of 248 large disc-shaped electrodes (210 µm in diameter) and 8 
# smaller disc-shaped electrodes (160 µm in diameter), arranged on an area of 
# 4.75 mm × 6.50 mm. The center-to-center pitch is 350 µm for the large electrodes
# and 300 µm for the small electrodes. [Xu2021]_.
#

fig, ax = plt.subplots(figsize=(10, 6))

IMIE().plot(ax=ax, annotate=True)
ax.set_title('IMIE')