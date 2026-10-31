# -*- coding: utf-8 -*-
"""
===============================================================================
Optimizing stimuli through a phosphene model
===============================================================================

:py:func:`~pulse2percept.to_torch` turns a configured model into a
:py:class:`torch.nn.Module` that maps a batch of inputs to a batch of
predicted percepts, with gradients. Stimuli can then be optimized for a
desired percept by gradient descent through the fixed model, as in the hybrid
neural autoencoder of Granley et al. (NeurIPS 2022).

This example uses MNIST digits as target percepts:

1. **Image input** to an axon map model [Beyeler2019]_: a small fully
   connected network learns which image to hand the implant, which samples
   and encodes it as usual.
2. **Electrode input** to the [Granley2023]_ model: the amplitude of every
   electrode is optimized directly, for a few digits.

Only the network or the amplitudes change; the phosphene models stay fixed.
Reconstruction MSE is a convenient training signal, not a validated measure
of perceptual quality, and a few epochs on a CPU give only a modest
improvement over the naive encoding.
"""
# sphinx_gallery_thumbnail_number = 1

import gzip
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import pulse2percept as p2p

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

###############################################################################
# MNIST digits
# ------------
#
# The MNIST training images (10 MB) are downloaded once to the pulse2percept
# data directory. A few thousand digits suffice here.

url = ('https://storage.googleapis.com/cvdf-datasets/mnist/'
       'train-images-idx3-ubyte.gz')
path = os.path.join(p2p.datasets.get_data_dir(), 'mnist-train-images.gz')
if not os.path.isfile(path):
    p2p.datasets.fetch_url(url, path, remote_checksum=(
        '440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609'))
with gzip.open(path) as f:
    digits = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
# Gray levels in [0, 1], as (B, C, H, W):
images = torch.tensor(digits[:2200] / 255, dtype=torch.float32)[:, None]
train, test = images[:2000].to(device), images[2000:].to(device)

###############################################################################
# Implant and model
# -----------------
#
# A 16 x 16 grid of electrodes, 300 um apart, in a right eye. Axonal streaks
# (``lam``, um) distort the naive encoding, which drives each electrode with
# the gray level beneath it. Without an implant encoder, a spatial-only axon
# map model reads gray levels as relative drive.

implant = p2p.implants.retina.RetinalImplant(
    p2p.implants.ElectrodeGrid((16, 16), 300))
model = p2p.models.retina.AxonMapModel(implant, rho=150, lam=500,
                                       xrange=(-8, 8), yrange=(-8, 8),
                                       step=0.5)
forward = p2p.to_torch(model, input_shape=(28, 28)).to(device)

###############################################################################
# The image spans the electrode bounding box. The target percept is the digit
# where the implant samples it, on the model's grid in degrees of visual
# angle: each grid point is mapped to the retina and looked up in the image.
# This keeps target and percept registered, including the up-down flip of the
# retinotopic map.

x, y = implant.electrode_array.coordinates(p2p.units.um)[:, :2].T
grid = model.spatial.grid
# Retinal position (um) of each grid point, as image coordinates in [-1, 1]:
u = 2 * (grid.ret.x - x.min()) / np.ptp(x) - 1
v = 2 * (grid.ret.y - y.min()) / np.ptp(y) - 1
lookup = torch.tensor(np.stack([u, v], axis=-1), dtype=torch.float32,
                      device=device)[None]


def target(images):
    return F.grid_sample(images, lookup.expand(len(images), -1, -1, -1),
                         align_corners=True)


def show(rows):
    fig, axes = plt.subplots(len(rows), 6, figsize=(9, 1.6 * len(rows)))
    for ax_row, (name, row) in zip(axes, rows.items()):
        ax_row[0].set_ylabel(name)
        for ax, img in zip(ax_row, row.cpu()):
            ax.imshow(img[0], cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()


###############################################################################
# Image input: AxonMapModel
# -------------------------
#
# Percepts are raw model output. Dividing them by the peak response to an
# all-white image puts targets and percepts on a shared [0, 1] scale.
#
# The network maps a digit to the image the implant should see. Its output
# bias starts at -3, so training starts from a dark image rather than from
# every electrode at half drive.

with torch.no_grad():
    white = forward(torch.ones((1, 1, 28, 28), device=device)).max()

encoder = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 256),
    nn.ReLU(),
    nn.Linear(256, 28 * 28),
    nn.Sigmoid(),
    nn.Unflatten(1, (1, 28, 28)),
).to(device)
nn.init.constant_(encoder[3].bias, -3)

optimizer = torch.optim.Adam(encoder.parameters(), lr=3e-3)
for epoch in range(10):
    for batch in train[torch.randperm(len(train))].split(64):
        loss = F.mse_loss(forward(encoder(batch)) / white, target(batch))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

with torch.no_grad():
    naive_mse = F.mse_loss(forward(test) / white, target(test))
    learned_mse = F.mse_loss(forward(encoder(test)) / white, target(test))
print(f"Test MSE: naive {naive_mse:.4f}, learned {learned_mse:.4f}")

shown = test[:6]
with torch.no_grad():
    show({'Target': target(shown),
          'Naive': forward(shown) / white,
          'Learned': forward(encoder(shown)) / white})

###############################################################################
# Electrode input: Granley2023Model
# ---------------------------------
#
# Here the amplitudes of all electrodes are the free parameters, bypassing
# image sampling and encoding. [Granley2023]_ reads pulse parameters:
# amplitude in multiples of threshold (here 0-3 xTh), and a fixed 20 Hz pulse
# train of 0.45 ms phases. Its ``rho`` is a phosphene area in output pixels.
# Rows follow ``forward_el.electrode_names``, the order of
# ``implant.electrode_names``.

granley = p2p.models.retina.Granley2023Model(implant, xrange=(-8, 8),
                                             yrange=(-8, 8), step=0.5)
forward_el = p2p.to_torch(granley, input_kind='electrodes', freq=20,
                          phase_dur=0.45).to(device)
with torch.no_grad():
    white_el = forward_el(torch.full((1, implant.n_electrodes), 3.0,
                                     device=device)).max()

###############################################################################
# The naive encoding drives each electrode with 3 xTh times the gray level
# beneath it. Optimization starts there. Amplitudes at or below the model's
# ``amp_cutoff`` (0.25 xTh) produce no phosphene and receive no gradient, so
# an electrode pushed below it stays off.

naive = np.array([implant.reshape_stim(p2p.stimuli.ImageStimulus(img)).data
                  for img in shown[:, 0].cpu().numpy()])
naive = 3 * torch.tensor(naive[..., 0], dtype=torch.float32, device=device)
amp = naive.clone().requires_grad_()
optimizer = torch.optim.Adam([amp], lr=0.1)
for step in range(100):
    loss = F.mse_loss(forward_el(amp) / white_el, target(shown))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        # Stay within 0-3 xTh:
        amp.clamp_(0, 3)

with torch.no_grad():
    naive_mse = F.mse_loss(forward_el(naive) / white_el, target(shown))
    optimized_mse = F.mse_loss(forward_el(amp) / white_el, target(shown))
    print(f"MSE: naive {naive_mse:.4f}, optimized {optimized_mse:.4f}")
    show({'Target': target(shown),
          'Naive': forward_el(naive) / white_el,
          'Optimized': forward_el(amp) / white_el})
