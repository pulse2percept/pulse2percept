"""Torch modules with fixed geometry, shared by
:py:func:`~pulse2percept.to_torch` and :py:func:`~pulse2percept.export_onnx`.

Each module reproduces a configured model stage from precomputed weights
instead of the code that computes them. Spatial modules map drive ``(E, N)``
to a flat response ``(P, N)``; columns are independent (e.g., time points or
batch elements).
"""
import numpy as np
import torch
from torch import nn

from .base import _ModelResponse, _blend_geometry, _blend_operator
from .retina.granley2023 import _mvg_response
from ..units import ms
from ..utils.constants import MIN_AMP


def _register_list(module, name, tensors):
    """Register ``tensors`` as buffers ``name0``, ``name1``, ..."""
    for i, tensor in enumerate(tensors):
        module.register_buffer(f'{name}{i}', tensor)
    return len(tensors)


#: `skimage.color.rgb2gray` weights, as in `Implant.reshape_stim`:
_RGB_WEIGHTS = (0.2125, 0.7154, 0.0721)


class _ImageSampler(nn.Module):
    """Gray level ``(B, E, F)`` at every electrode of images ``(B, C, H, W)``
    or videos ``(B, C, F, H, W)``.

    ``pixel_index`` and ``pixel_weight`` ``(E * 4,)`` are the bilinear
    sampler of the flattened image. RGB (``C=3``) is sampled per channel,
    then converted to gray as ``Implant.reshape_stim`` does.
    """

    def __init__(self, pixel_index, pixel_weight, shape, video=False):
        super().__init__()
        self.register_buffer('pixel_index', pixel_index)
        self.register_buffer('pixel_weight', pixel_weight)
        self.register_buffer('rgb', torch.tensor(_RGB_WEIGHTS))
        self.shape = tuple(shape)
        self.video = video

    def forward(self, x):
        ndim = 5 if self.video else 4
        if (x.ndim != ndim or tuple(x.shape[-2:]) != self.shape or
                x.shape[1] not in (1, 3)):
            layout = '(B, C, F, H, W)' if self.video else '(B, C, H, W)'
            raise ValueError(f"Input must have shape {layout} with C=1 (gray) "
                             f"or C=3 (RGB) and (H, W)={self.shape}, not "
                             f"{tuple(x.shape)}.")
        if not x.is_floating_point():
            raise TypeError(f"Input must hold float gray levels in [0, 1], "
                            f"not {x.dtype}; divide 8-bit images by 255.")
        # Eager only: export cannot trace a data-dependent check.
        if not torch.compiler.is_compiling() and not bool(
                torch.isfinite(x).all()):
            raise ValueError("Input must be finite, but contains NaN or Inf.")
        pixels = x.flatten(-2).index_select(-1, self.pixel_index)
        gray = (pixels * self.pixel_weight).unflatten(-1, (-1, 4)).sum(-1)
        if x.shape[1] == 3:
            gray = torch.tensordot(self.rgb.to(gray.dtype), gray, ([0], [1]))
        else:
            gray = gray[:, 0]
        # (B, [F,] E) -> (B, E, F):
        return gray.transpose(1, 2) if self.video else gray[..., None]


class _MeridianBlend(nn.Module):
    """``_blend_meridian`` of a flat ``(P, N)`` response, then threshold."""

    def __init__(self, shape, axis, blur, weight, thresh):
        super().__init__()
        self.shape = tuple(shape)
        self.axis = axis
        # Both axes become a plain matrix product:
        blur = torch.as_tensor(blur, dtype=torch.float32)
        self.register_buffer('blur',
                             blur if axis == 0 else blur.T.contiguous())
        self.register_buffer('weight', torch.as_tensor(
            weight, dtype=torch.float32)[..., None])
        self.thresh = float(thresh)

    def forward(self, resp):
        work = resp.reshape(self.shape + (-1,))
        if self.axis == 0:
            blurred = (self.blur @ work.flatten(1)).reshape(work.shape)
        else:
            blurred = (work.transpose(1, 2) @ self.blur).transpose(1, 2)
        out = torch.lerp(work, blurred, self.weight).reshape(resp.shape)
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
    """Scoreboard response ``(P, N)`` to drive ``(E, N)``.

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


#: Approximate working-memory target (bytes) per pixel chunk in `_AxonMap`.
_AXON_CHUNK_BYTES = 16 * 2 ** 20


class _AxonMap(nn.Module):
    """AxonMap response ``(P, N)`` to drive ``(E, N)``.

    Bucket ``i`` holds the float32 weights ``(n_pixels * L, E)`` of its
    pixels' retained segments, ``L`` rows per pixel in axon order, padded
    with zero rows. Columns follow ``electrodes``, the drive rows in
    summation order. ``order`` maps concatenated bucket outputs to grid
    order. Pixels are processed in chunks of about ``_AXON_CHUNK_BYTES``.
    """

    def __init__(self, buckets, lengths, electrodes, order, thresh,
                 blend=None):
        super().__init__()
        _register_list(self, 'bucket', buckets)
        self.lengths = [int(n) for n in lengths]
        self.register_buffer('electrodes', electrodes)
        self.register_buffer('order', order)
        self.thresh = float(thresh)
        self.blend = blend

    def forward(self, drive):
        drive = drive.index_select(0, self.electrodes)
        n_col = drive.shape[1]
        # The response, its `abs`, and the gathered winners:
        per_row = 3 * n_col * drive.element_size()
        out = []
        for i, length in enumerate(self.lengths):
            rows = max(1, _AXON_CHUNK_BYTES // (length * per_row)) * length
            for weights in getattr(self, f'bucket{i}').split(rows):
                seg = (weights @ drive).reshape(-1, length, n_col)
                # First largest |response|; the winner keeps its sign:
                best = seg.abs().argmax(dim=1, keepdim=True)
                out.append(seg.gather(1, best).squeeze(1))
        resp = torch.cat(out).index_select(0, self.order)
        resp = torch.where(resp.abs() >= self.thresh, resp, 0.0) + 0.0
        if self.blend is not None:
            resp = self.blend(resp)
        return resp


class _Granley2023(nn.Module):
    """Granley2023 response ``(B, P)`` to pulse parameters ``(B, E)``.

    Takes frequency (Hz), amplitude (xTh), phase duration (ms), and a bool
    mask of electrodes that deliver pulses. Geometry is that of the built
    ``spatial`` stage.
    """

    def __init__(self, spatial):
        super().__init__()
        for name in ('px', 'py', 'el_x', 'el_y'):
            self.register_buffer(name, torch.as_tensor(
                getattr(spatial, '_' + name)))
        # float32, as in `_Granley2023Spatial._predict_spatial`:
        self.register_buffer('theta', torch.as_tensor(
            spatial._el_theta * np.float32(spatial.orient_scale)))
        self.params = {name: float(getattr(spatial, name)) for name in (
            'rho', 'lam', 'a0', 'a1', 'a2', 'a3', 'a4', 'amp_cutoff',
            'thresh_percept')}

    def forward(self, freq, amp, pdur, active):
        return _mvg_response(freq, amp, pdur, self.px, self.py, self.el_x,
                             self.el_y, self.theta, active=active,
                             **self.params)


class _SpatialKernel(nn.Module):
    """Spatial response ``(P, N)`` to drive ``(E, N)`` from the
    ``_predict_tensor`` kernel of a model snapshot.

    For models without a precomputed module: geometry is recomputed on every
    call, on the device of the drive.
    """

    def __init__(self, spatial):
        super().__init__()
        self.spatial = spatial

    def forward(self, drive):
        return self.spatial._predict_tensor(drive, None).data


class _Temporal(nn.Module):
    """Temporal response ``(R, T_out)`` of rows ``(R, T)`` sampled at
    ``time`` (ms), from a model snapshot. Rows are independent."""

    def __init__(self, temporal, time, frame_clock, t_percept):
        super().__init__()
        self.temporal = temporal
        self.time = time
        self.frame_clock = frame_clock
        self.t_percept = t_percept

    def forward(self, resp):
        resp = _ModelResponse(resp, self.time, ms, (resp.shape[0],),
                              frame_clock=self.frame_clock)
        return self.temporal._predict_response(resp, self.t_percept)


def _check_delivered(waveform, time, safe_mode, max_current):
    """Run the ``Implant.check_stim`` checks on delivered current.

    ``waveform`` is ``(B, E, T)`` in uA, sampled at ``time`` (ms), or
    ``(B, E, 1)`` with ``time=None`` for a static drive.
    """
    with torch.no_grad():
        if safe_mode:
            # `Stimulus.is_charge_balanced`, per batch element:
            if time is None:
                charge = waveform.double()
            else:
                charge = torch.trapezoid(waveform.double(), x=torch.as_tensor(
                    time, dtype=torch.float64, device=waveform.device))
            if bool((charge.abs() > MIN_AMP).any()):
                raise ValueError("Safety check: Stimulus must be "
                                 "charge-balanced.")
        if max_current is not None:
            # Instantaneous load is the sum of |current| over all electrodes:
            total = waveform.abs().sum(dim=1)
            peak = float(total.max())
            if peak > max_current:
                b, t = np.unravel_index(int(total.argmax()), total.shape)
                n_active = int((waveform[b, :, t] != 0).sum())
                raise ValueError(
                    f"Safety check: stimulus draws {peak:.1f} uA at once "
                    f"({n_active} electrodes active), which exceeds "
                    f"max_current={max_current:.1f} uA. Give the implant a "
                    f"'raster' so that fewer electrodes fire at the same "
                    f"time, or lower the amplitude.")


class _Pulses(nn.Module):
    """AmplitudeEncoder stage with a fixed pulse schedule.

    Maps gray levels ``(B, E, F)`` to delivered amplitude ``(B, E, F)`` in
    the order of ``prepare_stim``: gray levels clipped to [0, 1],
    ``amp_lo + gray * span``, deactivated electrodes (``on``) removed, and
    threshold multiples converted to uA by ``scale`` ``(E,)`` if given. Then
    runs the implant's safety checks on the delivered pulse train.

    The schedule does not depend on amplitude. ``firing`` ``(E, F)`` marks
    running pulse clocks. ``frame`` and ``wave`` ``(E, U)`` hold the distinct
    waveform columns, ``columns`` ``(T,)`` the column of each sample at
    ``time`` (ms).
    """

    def __init__(self, amp_lo, span, on, scale, firing, frame, wave, columns,
                 time, frame_clock, safe_mode=False, max_current=None):
        super().__init__()
        self.amp_lo, self.span = float(amp_lo), float(span)
        self.register_buffer('on', torch.as_tensor(on)[:, None])
        self.register_buffer('scale', None if scale is None else
                             torch.as_tensor(scale)[:, None])
        self.register_buffer('firing', torch.as_tensor(firing))
        self.register_buffer('frame', torch.as_tensor(frame))
        self.register_buffer('wave', torch.as_tensor(wave,
                                                     dtype=torch.float32))
        self.register_buffer('columns', torch.as_tensor(columns))
        self.time = time
        self.frame_clock = frame_clock
        self.safe_mode = bool(safe_mode)
        self.max_current = max_current

    def forward(self, gray):
        # Keep the one-sided derivative at the valid gray-level endpoints:
        gray = torch.where(gray < 0, 0, torch.where(gray > 1, 1, gray))
        amp = torch.where(self.on, self.amp_lo + gray * self.span, 0.0)
        if self.scale is not None:
            amp = amp * self.scale
        if self.safe_mode or self.max_current is not None:
            _check_delivered(self.waveform(amp).index_select(-1, self.columns),
                             self.time, self.safe_mode, self.max_current)
        return amp

    def drive(self, amp):
        """Return the frame-level drive of spatial-only models."""
        return torch.where(self.firing, amp, 0.0)

    def waveform(self, amp):
        """Return the distinct waveform columns ``(B, E, U)``."""
        frame = self.frame.expand(amp.shape[:-1] + self.frame.shape[-1:])
        return amp.gather(-1, frame) * self.wave


class _Workflow(nn.Module):
    """Batched input-to-percept workflow; see ``to_torch``."""

    def __init__(self, *, shape, electrode_names, grid, sampler=None,
                 pulses=None, on=None, checks=None, spatial=None,
                 temporal=None, granley=None, xth_scale=None, freq=None,
                 phase_dur=None, time=None):
        super().__init__()
        self.sampler = sampler
        self.pulses = pulses
        self.spatial = spatial
        self.temporal = temporal
        self.granley = granley
        self.register_buffer('on', None if on is None else
                             torch.as_tensor(on)[:, None])
        self.checks = checks
        # Granley2023 only: thresholds (uA) of calibrated amplitudes, and
        # default pulse parameters:
        self.register_buffer('xth_scale', xth_scale)
        self.register_buffer('freq', freq)
        self.register_buffer('phase_dur', phase_dur)
        self.shape = tuple(shape)
        self.electrode_names = list(electrode_names)
        self.grid = grid
        self.time = time
        self.t_percept = None

    def forward(self, x, freq=None, phase_dur=None):
        if (freq is not None or phase_dur is not None) and (
                self.granley is None or self.sampler is not None):
            raise TypeError("'freq' and 'phase_dur' are inputs only of a "
                            "Granley2023Model with input_kind='electrodes'.")
        if self.sampler is not None:
            drive = self.sampler(x)
        else:
            drive = self._electrodes(x)
        if self.pulses is not None:
            drive = self.pulses(drive)
        else:
            drive = torch.where(self.on, drive, 0.0)
            if self.checks is not None:
                _check_delivered(drive, self.time, *self.checks)
        if self.granley is not None:
            resp = self._granley(drive[..., 0], freq, phase_dur)
            return resp.unflatten(-1, self.shape)[:, None]
        if self.temporal is None:
            if self.pulses is not None:
                drive = self.pulses.drive(drive)
            resp = self._spatial(drive)
            if self.t_percept is None:
                return resp[..., 0].unflatten(-1, self.shape)[:, None]
        else:
            if self.pulses is not None:
                resp = self._spatial(self.pulses.waveform(drive))
                resp = resp.index_select(-1, self.pulses.columns)
            else:
                resp = self._spatial(drive)
            n_batch = resp.shape[0]
            resp = self.temporal(resp.flatten(0, 1)).data
            resp = resp.unflatten(0, (n_batch, -1))
        # (B, P, T) -> (B, 1, T, Hp, Wp):
        return resp.transpose(1, 2).unflatten(-1, self.shape)[:, None]

    def _electrodes(self, x):
        """Return validated electrode input as ``(B, E, T)``."""
        n_el = len(self.electrode_names)
        n_time = None if self.time is None else len(self.time)
        expected = (n_el,) if n_time is None else (n_el, n_time)
        if x.ndim != len(expected) + 1 or tuple(x.shape[1:]) != expected:
            layout = '(B, E)' if n_time is None else '(B, E, T)'
            raise ValueError(f"Electrode input must have shape {layout} = "
                             f"(B, {', '.join(map(str, expected))}), not "
                             f"{tuple(x.shape)}.")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("Electrode input must be finite, but contains "
                             "NaN or Inf.")
        return x if n_time is not None else x[..., None]

    def _spatial(self, drive):
        """Return the spatial response ``(B, P, N)`` to ``(B, E, N)``."""
        n_batch, n_el, n_col = drive.shape
        resp = self.spatial(drive.transpose(0, 1).reshape(n_el, -1))
        return resp.unflatten(1, (n_batch, n_col)).transpose(0, 1)

    def _granley(self, amp, freq, pdur):
        """Return the Granley2023 response ``(B, P)`` to delivered ``amp``."""
        params = {}
        for name, value in (('freq', freq), ('phase_dur', pdur)):
            value = getattr(self, name) if value is None else value
            if value is None:
                raise ValueError(f"'{name}' is required: pass it to "
                                 f"to_torch or to this call.")
            value = torch.as_tensor(value, dtype=amp.dtype, device=amp.device)
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"'{name}' must be finite.")
            params[name] = value
        freq, pdur = params['freq'], params['phase_dur']
        if not bool((pdur > 0).all()):
            raise ValueError("'phase_dur' must be positive.")
        if self.xth_scale is not None:
            # Calibrated current back to threshold multiples:
            amp = amp / self.xth_scale
        active = self.on[:, 0] & (amp != 0) & (freq > 0)
        return self.granley(freq, amp, pdur, active)


def _bucket_axons(segments, n_pad):
    """Return padded per-pixel segment rows, grouped by length.

    ``segments`` lists the weight rows of each pixel's segments in axon
    order; ``n_pad`` is the zero row. Bucket lengths grow by 25%, which
    bounds padding at 25%. Returns the ``(n_pixels, L)`` row index of each
    bucket and the pixel order of their concatenation.
    """
    counts = np.array([s.size for s in segments])
    ladder = [1]
    while ladder[-1] < counts.max(initial=0):
        ladder.append(max(ladder[-1] + 1, int(np.ceil(ladder[-1] * 1.25))))
    length = np.asarray(ladder)[np.searchsorted(ladder, counts)]
    buckets, pixels = [], []
    for size in np.unique(length):
        members = np.flatnonzero(length == size)
        index = np.full((members.size, size), n_pad)
        for i, p in enumerate(members):
            index[i, :counts[p]] = segments[p]
        buckets.append(index)
        pixels.append(members)
    return buckets, np.argsort(np.concatenate(pixels))
