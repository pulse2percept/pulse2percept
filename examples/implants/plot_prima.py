"""
===============================================================================
Simulating PRIMA: from image to optical drive
===============================================================================

:class:`~pulse2percept.implants.PRIMAPivotal` models the 378-pixel array used
in the PRIMAvera trial [Holz2025]_. PRIMA uses 880 nm illumination rather than
injected current, with image intensity encoded by pulse duration.

This example shows image preprocessing, optical drive on the array, and a
:class:`~pulse2percept.models.ScoreboardModel` visualization. The Scoreboard
output is not a PRIMA retinal-response model.
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
# PRIMA uses fixed peak irradiance; gray level controls pulse duration.
# The default encoder maps [0, 1] onto 14 nonzero duration levels.

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
# Normalized optical drive is time-averaged irradiance relative to the
# documented maximum. At the default settings it is ``pulse_dur / 9.8 ms``.
#
# From an image to a percept
# --------------------------
#
# Convert to grayscale before preprocessing. Canny edges are computed at
# about twice the nominal array resolution and dilated once so thin edges
# survive sampling onto the implant.

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
    # Source image
    ax = axes[0, col]
    ax.imshow(source.data.reshape(source.img_shape), cmap='gray',
              vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis('off')

    # Drive after sampling onto PRIMA pixels
    stim = implant.prepare_stim(source)
    drive = normalized_drive(stim)
    ax = axes[1, col]
    ax.scatter(xy[:, 0], xy[:, 1], c=drive, cmap='gray',
               vmin=0, vmax=1, marker='h', s=55, linewidths=0)
    ax.set_aspect('equal')
    ax.axis('off')

    # Scoreboard visualization
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
# ``ScoreboardModel`` weights one Gaussian per pixel by normalized optical
# drive. ``rho=100`` sets the spatial spread, approximately one PRIMA pixel
# pitch.
#
# This is not a PRIMA brightness model: photovoltaic conversion and retinal
# activation are not modeled. The preprocessing examples are illustrative and
# do not reproduce the clinical image-processing pipeline.
