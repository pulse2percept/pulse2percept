"""
===============================================================================
Retinal implant gallery
===============================================================================

pulse2percept supports the following implants:

Argus Retinal Prosthesis System (Second Sight Medical Products Inc.)
---------------------------------------------------------------------

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

# For illustrative purpose, also show the map of fiber
# bundles in the optic fiber layer:
model = AxonMapModel()
model.plot(ax=ax[0])
# Argus I is typically implanted at a 30-45deg angle:
ArgusI(rot=-0.52).plot(ax=ax[0], annotate=True)
ax[0].set_title('Argus I')

model.plot(ax=ax[1])
# Argus II is typically implanted at a 30-45deg angle:
ArgusII(rot=-0.52).plot(ax=ax[1], annotate=False)
ax[1].set_title('Argus II')

###############################################################################
# PRIMA Bionic Vision System (Pixium Vision SA)
# ----------------------------------------------
#
# :py:class:`~pulse2percept.implants.PRIMA` is a subretinal device developed
# at Stanford University and commercialized by Pixium Vision.
#
# There are several versions of the PRIMA device.
# The device used in clinical trial `NCT03392324`_ consists of 378 85um-wide
# pixels separated by 15um trenches (i.e., 100um pixel pitch), arranged in a
# 2-mm wide hexagonal pattern, and is available in pulse2percept simply as
# :py:class:`~pulse2percept.implants.PRIMA` [Palanker2020]_.
#
# :py:class:`~pulse2percept.implants.PRIMA75` is a newer version of the device,
# consisting of 142 70um-wide pixels separated by 5um trenches (i.e., 75um
# pixel pitch), arranged in a 1-mm wide hexagonal pattern [Lorach2015]_.
#
# .. _NCT03392324: https://www.clinicaltrials.gov/ct2/show/NCT03392324

fig, ax = plt.subplots(ncols=2, figsize=(10, 6))

PRIMA().plot(ax=ax[0])
ax[0].set_title('PRIMA-100')

PRIMA75().plot(ax=ax[1])
ax[1].set_title('PRIMA-75')

###############################################################################
# In addition, the developers are working on miniaturizing the device. At least
# two other prototypes are currently in development:
#
# :py:class:`~pulse2percept.implants.PRIMA55` consists of 50um-wide pixels
# separated by 5um trenches (i.e., 55um pixel pitch), whereas
# :py:class:`~pulse2percept.implants.PRIMA40` consists of 35um-wide pixels
# separated by 5um trenches (i.e., 40um pixel pitch).
#
# The exact geometric arrangement of these two prototypes have not been
# published yet. The devices available in pulse2percept assume that the arrays
# fit on a circular 1mm-diameter substrate, which yields 273 electrodes for
# PRIMA-55 and 532 electrodes for PRIMA-40.
# These prototypes will be updated once more information about them is
# available.

fig, ax = plt.subplots(ncols=2, figsize=(10, 6))

PRIMA55().plot(ax=ax[0])
ax[0].set_title('PRIMA-55')

PRIMA40().plot(ax=ax[1])
ax[1].set_title('PRIMA-40')

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
