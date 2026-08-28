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
# PRIMA photovoltaic implants
# ---------------------------
#
# PRIMA is a subretinal photovoltaic prosthesis developed at Stanford.
# Pixium Vision developed the clinical system; `Science Corporation`_ acquired
# Pixium's PRIMA assets and IP in 2024.
#
# :py:class:`~pulse2percept.implants.PRIMAPivotal` models the 378-pixel,
# 100um array used in the pivotal PRIMAvera trial `NCT04676854`_ [Holz2026]_.
# The same configuration was used in the earlier US feasibility study
# `NCT03392324`_ [Palanker2020]_.
#
# :py:class:`~pulse2percept.implants.Lorach2015Array` models the earlier
# 142-pixel, 70um research array of [Lorach2015]_.
#
# .. _Science Corporation: https://science.xyz/news/pixium-vision-acquisition/
# .. _NCT04676854: https://clinicaltrials.gov/study/NCT04676854
# .. _NCT03392324: https://clinicaltrials.gov/study/NCT03392324

fig, ax = plt.subplots(ncols=2, figsize=(10, 6))

PRIMAPivotal().plot(ax=ax[0])
ax[0].set_title('PRIMA (pivotal)')

Lorach2015Array().plot(ax=ax[1])
ax[1].set_title('Lorach et al. (2015)')

###############################################################################
# :py:class:`~pulse2percept.implants.Ho2019FlatArray` models the flat F55 and
# F40 research arrays of [Ho2019]_, with 250 and 502 pixels on a 1 mm
# substrate. The F55 outline is reconstructed from Fig. 2(a); the F40 outline
# was not published and is approximated by the 502 lattice sites nearest the
# substrate center.

fig, ax = plt.subplots(ncols=2, figsize=(10, 6))

Ho2019FlatArray(55).plot(ax=ax[0])
ax[0].set_title('Ho et al. F55')

Ho2019FlatArray(40).plot(ax=ax[1])
ax[1].set_title('Ho et al. F40')

###############################################################################
# :py:class:`~pulse2percept.implants.Huang2021Array` models the 1.5 mm
# vertical-junction arrays of [Huang2021]_. Only exposed pixels are modeled as
# electrodes; ``n_total_pixels`` gives the total number fabricated on the die.

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

for axis, pixel_size in zip(ax.ravel(), [55, 40, 30, 20]):
    implant = Huang2021Array(pixel_size)
    implant.plot(ax=axis)
    axis.set_title(f'{pixel_size} um ({implant.n_electrodes} exposed)')

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