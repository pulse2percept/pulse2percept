"""
===============================================================================
Simulating PRIMA
===============================================================================

Background
----------

:py:class:`~pulse2percept.implants.PRIMAPivotal` is the subretinal photovoltaic
array used in the PRIMAvera pivotal trial [Holz2026]_ and in the earlier
first-in-human study [Palanker2020]_. Its 378 pixels sit on a 100 um hexagonal
grid on a 2 x 2 mm substrate.

PRIMA is not driven by a current source. A goggle-mounted projector paints an
880 nm near-infrared image onto the implant, and each photovoltaic pixel turns
the light it receives into local electrical stimulation. The projector runs at
a fixed frame rate (30 Hz) and a fixed peak irradiance (3.5 mW/mm^2), and
encodes intensity in *how long* a pixel is lit -- 0.7 to 9.8 ms per frame --
rather than in how brightly.

:py:class:`~pulse2percept.stimuli.PRIMAEncoder` implements that projector, and
:py:class:`~pulse2percept.implants.PRIMAPivotal` brings one along, so a picture
can be presented directly:
"""
# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import numpy as np
import pulse2percept as p2p

implant = p2p.implants.PRIMAPivotal()
image = p2p.stimuli.LogoBVL()
stim = implant.prepare_stim(image)
print(stim.unit)

###############################################################################
# What comes back is 880 nm optical irradiance versus time at each pixel, not
# electrical current. Turning light into retinal current depends on photodiode
# behavior, electrode capacitance, and the surrounding tissue, none of which
# pulse2percept models yet.
#
# One projector period looks the same at every lit pixel: irradiance rises to
# 3.5 mW/mm^2 for the pixel's ON duration and is zero for the rest of the
# 33.3 ms frame.

lit = np.flatnonzero(stim.data.max(axis=1) > 0)[0]
fig, ax = plt.subplots(figsize=(8, 2.5))
ax.plot(stim.time, stim.data[lit], lw=1.5)
ax.set_xlim(0, 100)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Irradiance (mW/mm$^2$)')
ax.set_title(f'Pixel {stim.electrodes[lit]}: 9.8 ms on, 33.3 ms period')
fig.tight_layout()

###############################################################################
# Visualizing the stimulation pattern
# -----------------------------------
#
# :py:class:`~pulse2percept.models.ScoreboardModel` places a Gaussian blob at
# every pixel. For a photovoltaic implant it weights each blob by *normalized
# optical drive*: irradiance times ON duration times frame rate, divided by the
# largest drive the pivotal device is documented to produce. A dark pixel is 0,
# a pixel at the full documented drive is 1.
#
# This is a simple visualization of implant geometry and stimulation pattern.
# It does not model photodiode transduction, retinal electric fields,
# bipolar-cell activation, electrode-retina distance, or temporal retinal
# dynamics.

model = p2p.models.ScoreboardModel(implant=implant, rho=100)

###############################################################################
# Image processing is not PRIMA-specific
# --------------------------------------
#
# The encoder binarizes at ``threshold=0.5`` by default, which is
# pulse2percept's approximation of an unpublished clinical algorithm. Contrast
# inversion and edge enhancement are *not* intrinsic to the device, so they are
# ordinary :py:class:`~pulse2percept.stimuli.ImageStimulus` operations applied
# before encoding:

sources = [('As is', image),
           ('Inverted', image.invert()),
           ('Canny edges', image.rgb2gray().filter('canny'))]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(11, 6))
for (title, source), top, bottom in zip(sources, axes[0], axes[1]):
    source.plot(ax=top)
    top.set_title(title)
    model.predict_percept(source).plot(ax=bottom)
fig.tight_layout()

###############################################################################
# Grayscale mode
# --------------
#
# Current clinical operation is effectively binary, which is why that is the
# default. The projector hardware does support several pulse durations, though,
# and ``grayscale=True`` pulse-width modulates gray levels onto that 0.7 ms
# grid. Peak irradiance stays fixed either way:

implant.encoder = p2p.stimuli.PRIMAEncoder(grayscale=True)
gray = implant.prepare_stim(image)
print(np.unique(np.round(gray.pulse_dur, 3)))
