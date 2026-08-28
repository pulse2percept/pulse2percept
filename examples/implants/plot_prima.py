"""
===============================================================================
Simulating PRIMA: from image to optical drive
===============================================================================

:class:`~pulse2percept.implants.PRIMAPivotal` models the 378-pixel
photovoltaic array used in the PRIMAvera trial [Holz2025]_. Unlike other
implants, PRIMA is not driven by injected current. A near-infrared projector
illuminates the array at 880 nm, and image intensity is encoded by how long
each pixel is illuminated during a projector frame.

This example follows the path currently implemented in pulse2percept:

``image -> preprocessing -> implant sampling -> optical drive -> percept``

The last step uses :class:`~pulse2percept.models.ScoreboardModel` only as a
visualization of the stimulation pattern. pulse2percept does not yet model
photodiode transduction or the retinal response to PRIMA.
"""
# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import binary_dilation, disk
import pulse2percept as p2p

implant = p2p.implants.PRIMAPivotal()
encoder = implant.encoder

###############################################################################
# From gray level to optical stimulation
# ---------------------------------------
#
# PRIMA uses a fixed peak irradiance and encodes intensity using pulse
# duration. The default encoder maps normalized image intensity onto the 14
# nonzero durations supported by the pivotal projector.

levels = np.array([0.25, 0.5, 1.0], dtype=np.float32)
probe = p2p.stimuli.ImageStimulus(levels[np.newaxis, :])
encoded = encoder.encode(probe)

fig, ax = plt.subplots(figsize=(7, 2.5))
for idx, gray in enumerate(levels):
    dur = encoded.pulse_dur[idx, 0]
    ax.plot(encoded.time, encoded.data[idx],
            label=f'gray={gray:g}  ({dur:g} ms)')
ax.set_xlim(0, encoder.period)
ax.set_xlabel('Time (ms)')
ax.set_ylabel(r'Irradiance (mW/mm$^2$)')
ax.legend(frameon=False)
fig.tight_layout()

###############################################################################
# Peak irradiance is the same for every illuminated pixel. A brighter image
# pixel stays on longer, so it delivers more optical energy per projector
# frame. For the spatial visualization below, pulse2percept normalizes the
# time-averaged optical drive so that 0 is dark and 1 is the largest documented
# drive of the pivotal system. At the default projector settings, this is simply
# ``pulse_dur / 9.8 ms``.
#
# From an image to a percept
# --------------------------
#
# Preprocessing happens before optical encoding. We convert the logo to
# grayscale before comparing different transforms. For edge extraction, the
# image is first resized to roughly twice the nominal implant-grid resolution.
# Thin edges otherwise tend to disappear when sampled onto the 378 PRIMA
# pixels. A one-pixel dilation makes the edge map a little more robust to that
# sampling step.

image = p2p.stimuli.LogoBVL().rgb2gray()
edge_shape = tuple(2 * n for n in implant.shape)
edges = image.resize(edge_shape).filter('canny', sigma=1.0)
edges = edges.apply(binary_dilation, footprint=disk(1))

sources = [
    ('Grayscale', image),
    ('Inverted grayscale', image.invert()),
    ('Edge-enhanced', edges),
]

model = p2p.models.ScoreboardModel(
    implant=implant,
    rho=100,
    xrange=(-4, 4),
    yrange=(-4, 4),
    step=0.05,
)


def normalized_drive(stim):
    """Normalized time-averaged optical drive for a static PRIMA image."""
    return (stim.irradiance * stim.duty_cycle[:, 0]
            / implant.encoder.ref_drive)


xy = implant.earray.coordinates()
fig, axes = plt.subplots(3, 3, figsize=(10, 9), constrained_layout=True)

for col, (title, source) in enumerate(sources):
    # Image entering the PRIMA encoder:
    ax = axes[0, col]
    ax.imshow(source.data.reshape(source.img_shape), cmap='gray',
              vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis('off')

    # Normalized optical drive on the actual 378-pixel array:
    stim = implant.prepare_stim(source)
    drive = normalized_drive(stim)
    ax = axes[1, col]
    ax.scatter(xy[:, 0], xy[:, 1], c=drive, cmap='gray',
               vmin=0, vmax=1, marker='h', s=55, linewidths=0)
    ax.set_aspect('equal')
    ax.axis('off')

    # Scoreboard visualization of that drive:
    model.predict_percept(source).plot(ax=axes[2, col], cmap='gray')

axes[0, 0].text(-0.10, 0.5, 'Source', rotation=90,
                va='center', ha='right', transform=axes[0, 0].transAxes,
                fontsize=12)
axes[1, 0].text(-0.10, 0.5, 'Optical drive', rotation=90,
                va='center', ha='right', transform=axes[1, 0].transAxes,
                fontsize=12)
axes[2, 0].text(-0.18, 0.5, 'Scoreboard percept', rotation=90,
                va='center', ha='right', transform=axes[2, 0].transAxes,
                fontsize=12)

###############################################################################
# How optical drive becomes percept brightness
# --------------------------------------------
#
# ``ScoreboardModel`` places a Gaussian at each implant pixel. For PRIMA, the
# Gaussian is weighted by the normalized optical drive shown in the middle row.
# A longer optical pulse therefore produces a brighter Gaussian in this
# visualization. ``rho=100`` controls how broadly neighboring pixels overlap
# and is approximately one PRIMA pixel pitch.
#
# This is not a biological PRIMA brightness model. Real percepts depend on
# photovoltaic conversion, electrode-retina distance, retinal current spread,
# neural activation and temporal dynamics. Those steps require a PRIMA-specific
# retinal model that is not yet implemented.
#
# The three preprocessing columns are examples, not a reconstruction of the
# clinical image-processing pipeline. The clinical system performs its own
# preprocessing before projection, but it is not specified in enough detail to
# reproduce here. Keeping preprocessing explicit makes its effect on implant
# sampling easy to inspect.
