"""Torch modules for exact ONNX export of spatial models.

Each module reproduces a configured model with fixed geometry, so ONNX
receives precomputed weights instead of the code that computes them. See
:py:func:`~pulse2percept.export_onnx`.
"""
import torch
from torch import nn

from .base import _blend_geometry, _blend_operator


def _register_list(module, name, tensors):
    """Register ``tensors`` as buffers ``name0``, ``name1``, ..."""
    for i, tensor in enumerate(tensors):
        module.register_buffer(f'{name}{i}', tensor)
    return len(tensors)


class _MeridianBlend(nn.Module):
    """``_blend_meridian`` of a flat ``(P, 1)`` response, then threshold."""

    def __init__(self, shape, axis, blur, weight, thresh):
        super().__init__()
        self.shape = tuple(shape)
        self.axis = axis
        # Both axes become a plain matrix product with a (Y, X) response:
        blur = torch.as_tensor(blur, dtype=torch.float32)
        self.register_buffer('blur',
                             blur if axis == 0 else blur.T.contiguous())
        self.register_buffer('weight',
                             torch.as_tensor(weight, dtype=torch.float32))
        self.thresh = float(thresh)

    def forward(self, resp):
        work = resp.reshape(self.shape)
        blurred = self.blur @ work if self.axis == 0 else work @ self.blur
        out = torch.lerp(work, blurred, self.weight).reshape(-1, 1)
        return torch.where(out.abs() < self.thresh, 0.0, out)


def _meridian_blend(spatial, meridian, width):
    """Return the blend module of a built model, or None for a no-op."""
    geometry = _blend_geometry(spatial.grid, meridian, width)
    if geometry is None:
        return None
    blur, weight = _blend_operator(*geometry)
    return _MeridianBlend(spatial.grid.x.shape, geometry[1], blur, weight,
                          spatial.thresh_percept)


class _Scoreboard(nn.Module):
    """Scoreboard response ``(P, 1)`` to drive ``(E, 1)``.

    ``weights`` holds one float32 ``(P, E)`` matrix per region. Each region
    is thresholded before the sum, as in ``_scoreboard_response``.
    """

    def __init__(self, weights, thresh, blend=None):
        super().__init__()
        self.n_regions = _register_list(self, 'weights', weights)
        self.thresh = float(thresh)
        self.blend = blend

    def forward(self, drive):
        resp = 0
        for i in range(self.n_regions):
            region = getattr(self, f'weights{i}') @ drive
            # `+ 0.0` turns -0.0 into 0.0, as in `_scoreboard_response`:
            resp = resp + (torch.where(region.abs() < self.thresh, 0.0,
                                       region) + 0.0)
        if self.blend is not None:
            resp = self.blend(resp)
        return resp


class _AxonMap(nn.Module):
    """AxonMap response ``(P, 1)`` to drive ``(E, 1)``.

    ``weights`` is the float32 ``(S + 1, E)`` electrode weight of each
    retained segment, plus a zero last row. Each bucket ``(n_pixels, L)``
    lists the segments of its pixels in axon order, padded with the zero
    row. ``order`` maps concatenated bucket outputs to grid order.
    """

    def __init__(self, weights, buckets, order, thresh, blend=None):
        super().__init__()
        self.register_buffer('weights', weights)
        self.shapes = [tuple(b.shape) for b in buckets]
        _register_list(self, 'bucket', [b.reshape(-1) for b in buckets])
        self.register_buffer('order', order)
        self.thresh = float(thresh)
        self.blend = blend

    def forward(self, drive):
        seg = (self.weights @ drive).reshape(-1)
        out = []
        for i, shape in enumerate(self.shapes):
            packed = seg.index_select(0, getattr(self, f'bucket{i}'))
            packed = packed.reshape(shape)
            # First largest |response|; the winner keeps its sign:
            best = packed.abs().argmax(dim=1, keepdim=True)
            out.append(packed.gather(1, best))
        resp = torch.cat(out).index_select(0, self.order)
        resp = torch.where(resp.abs() >= self.thresh, resp, 0.0) + 0.0
        if self.blend is not None:
            resp = self.blend(resp)
        return resp


class _ImagePercept(nn.Module):
    """Raw percept ``(1, 1, Hp, Wp)`` of a gray image ``(1, 1, H, W)``.

    ``pixel_index`` and ``pixel_weight`` ``(E * 4,)`` are the bilinear
    sampler of the flattened image at every electrode. ``modulate`` maps gray
    levels to amplitude; ``on`` ``(E, 1)`` marks electrodes that deliver it.
    """

    def __init__(self, pixel_index, pixel_weight, modulate, on, spatial,
                 shape):
        super().__init__()
        self.register_buffer('pixel_index', pixel_index)
        self.register_buffer('pixel_weight', pixel_weight)
        self.modulate = modulate
        self.register_buffer('on', on)
        self.spatial = spatial
        self.shape = tuple(shape)

    def drive(self, image):
        """Return the ``(E, 1)`` frame-level drive (uA) of ``image``."""
        pixels = image.reshape(-1).index_select(0, self.pixel_index)
        gray = (pixels * self.pixel_weight).reshape(-1, 4).sum(dim=1,
                                                               keepdim=True)
        amp = self.modulate(gray.clamp(0, 1)).to(torch.float32)
        return torch.where(self.on, amp, 0.0)

    def forward(self, image):
        return self.spatial(self.drive(image)).reshape((1, 1) + self.shape)
