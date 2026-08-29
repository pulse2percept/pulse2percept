""":py:class:`~pulse2percept.implants.PhotovoltaicPixel`,
   :py:class:`~pulse2percept.implants.PRIMAPivotal`,
   :py:class:`~pulse2percept.implants.Lorach2015Array`,
   :py:class:`~pulse2percept.implants.Ho2019FlatArray`,
   :py:class:`~pulse2percept.implants.Huang2021Array`"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, RegularPolygon

import numpy as np
from collections.abc import Sequence

from .base import ProsthesisSystem
from .electrodes import HexElectrode
from .electrode_arrays import ElectrodeGrid
from ..stimuli import PRIMAEncoder
from ..stimuli.base import _describe_unit
from ..stimuli.encoders import _OpticalStimulus
from ..units import DimensionMismatchError, as_value, mW, mm, um
from ..utils import deprecated
from ..utils.constants import MS_PER_S, ZORDER

# Distinguish the default PRIMAEncoder from ``encoder=None``.
_DEVICE_DEFAULT = object()


def _projector(stim):
    """Return ``stim`` if it retains a PRIMA projector schedule."""
    return stim if isinstance(stim, _OpticalStimulus) else None


#: F55 layout reconstructed from Fig. 2(a) of [Ho2019]_.
_HO2019_F55_AXIAL_SPANS = {
    -9: (3, 8), -8: (1, 9), -7: (-1, 9), -6: (-3, 9), -5: (-4, 9),
    -4: (-5, 9), -3: (-6, 9), -2: (-6, 8), -1: (-7, 8), 0: (-7, 7),
    1: (-8, 7), 2: (-8, 6), 3: (-9, 6), 4: (-9, 5), 5: (-9, 4),
    6: (-9, 3), 7: (-9, 2), 8: (-9, 1), 9: (-8, -1),
}

#: Exposed-pixel layouts reconstructed from Fig. 7 of [Huang2021]_.
_HUANG2021_AXIAL_SPANS = {
    55: {
        -12: (3, 9), -11: (1, 10), -10: (-1, 11), -9: (-3, 12), -8: (-4, 12),
        -7: (-5, 12), -6: (-6, 12), -5: (-7, 12), -4: (-8, 12), -3: (-8, 12),
        -2: (-9, 11), -1: (-10, 11), 0: (-10, 10), 1: (-11, 10), 2: (-11, 9),
        3: (-11, 8), 4: (-12, 8), 5: (-12, 7), 6: (-12, 6), 7: (-12, 5),
        8: (-11, 4), 9: (-11, 2), 10: (-10, 1), 11: (-9, -1), 12: (-7, -5),
    },
    40: {
        -17: (6, 11), -16: (3, 13), -15: (0, 14), -14: (-1, 15), -13: (-3, 16),
        -12: (-4, 17), -11: (-5, 17), -10: (-6, 17), -9: (-7, 17),
        -8: (-8, 17), -7: (-9, 17), -6: (-10, 17), -5: (-11, 17),
        -4: (-11, 16), -3: (-12, 16), -2: (-12, 15), -1: (-13, 15),
        0: (-13, 14), 1: (-14, 14), 2: (-14, 13), 3: (-15, 13), 4: (-15, 12),
        5: (-16, 12), 6: (-16, 11), 7: (-16, 10), 8: (-17, 9), 9: (-17, 8),
        10: (-17, 7), 11: (-17, 6), 12: (-17, 5), 13: (-16, 4), 14: (-16, 3),
        15: (-15, 1), 16: (-14, -1), 17: (-13, -3),
    },
    30: {
        -23: (8, 13), -22: (4, 16), -21: (1, 18), -20: (-1, 19), -19: (-3, 20),
        -18: (-4, 20), -17: (-6, 21), -16: (-7, 21), -15: (-8, 21),
        -14: (-9, 21), -13: (-10, 21), -12: (-11, 21), -11: (-12, 21),
        -10: (-13, 21), -9: (-14, 21), -8: (-15, 21), -7: (-16, 21),
        -6: (-17, 21), -5: (-17, 20), -4: (-18, 20), -3: (-18, 19),
        -2: (-19, 19), -1: (-20, 19), 0: (-20, 18), 1: (-20, 18), 2: (-21, 17),
        3: (-21, 16), 4: (-22, 16), 5: (-22, 15), 6: (-22, 14), 7: (-22, 13),
        8: (-23, 13), 9: (-23, 12), 10: (-23, 11), 11: (-23, 10), 12: (-23, 9),
        13: (-23, 8), 14: (-22, 6), 15: (-22, 5), 16: (-22, 4), 17: (-21, 2),
        18: (-20, 0), 19: (-19, -1), 20: (-18, -4), 21: (-16, -7),
    },
    # Three rim pixels are inferred to match the published count; (0, 27)
    # is ambiguous with (-9, 31).
    20: {
        -34: (11, 21), -33: (8, 24), -32: (5, 26), -31: (3, 27), -30: (1, 28),
        -29: (-1, 29), -28: (-3, 30), -27: (-5, 30), -26: (-6, 31),
        -25: (-8, 31), -24: (-9, 31), -23: (-10, 32), -22: (-11, 32),
        -21: (-13, 32), -20: (-14, 32), -19: (-15, 32), -18: (-16, 32),
        -17: (-17, 32), -16: (-17, 32), -15: (-18, 32), -14: (-19, 32),
        -13: (-20, 32), -12: (-21, 31), -11: (-22, 31), -10: (-22, 31),
        -9: (-23, 30), -8: (-23, 30), -7: (-24, 30), -6: (-24, 29),
        -5: (-25, 29), -4: (-25, 29), -3: (-26, 28), -2: (-26, 28),
        -1: (-27, 27), 0: (-28, 27), 1: (-28, 26), 2: (-29, 25), 3: (-29, 25),
        4: (-29, 24), 5: (-30, 23), 6: (-30, 23), 7: (-30, 22), 8: (-31, 21),
        9: (-31, 20), 10: (-31, 19), 11: (-31, 19), 12: (-31, 18),
        13: (-31, 17), 14: (-31, 16), 15: (-31, 15), 16: (-31, 14),
        17: (-31, 12), 18: (-31, 11), 19: (-30, 10), 20: (-30, 9),
        21: (-30, 7), 22: (-29, 6), 23: (-29, 4), 24: (-28, 3), 25: (-27, 1),
        26: (-26, -1), 27: (-25, -4), 28: (-23, -6), 29: (-20, -10),
    },
}

#: Total fabricated pixels in each [Huang2021]_ array.
_HUANG2021_TOTAL_PIXELS = {55: 526, 40: 1027, 30: 1735, 20: 3508}


def _axial_rows(spans):
    """Convert axial-coordinate spans to grid row/column indices."""
    q0 = min(spans)
    ij = [(r + ((q - q0) + 1) // 2, q - q0)
          for q, (r_lo, r_hi) in spans.items()
          for r in range(r_lo, r_hi + 1)]
    lo = min(i for i, _ in ij)
    return [(i - lo, j) for i, j in ij]


def _axial_mask_shape(spans):
    """Return the smallest grid that contains an axial mask."""
    ij = _axial_rows(spans)
    return max(i for i, _ in ij) + 1, max(spans) - min(spans) + 1


#: Flat-array parameters from [Ho2019]_.
_HO2019_VARIANTS = {
    55: {'elec_radius': 7, 'n_pixels': 250,
         'shape': _axial_mask_shape(_HO2019_F55_AXIAL_SPANS),
         'spans': _HO2019_F55_AXIAL_SPANS},
    # Smallest grid that holds the 502 nearest sites without the trim having
    # to split a ring of equidistant ones:
    40: {'elec_radius': 5, 'n_pixels': 502, 'shape': (26, 27), 'spans': None},
}


def _pixel_size_um(pixel_size, supported, cls_name):
    """Validate and normalize a supported pixel size."""
    value = as_value(pixel_size, um, 'pixel_size')
    if isinstance(value, (Sequence, np.ndarray)):
        raise TypeError(f"'pixel_size' must be a scalar, not "
                        f"{type(pixel_size)}.")
    for size in supported:
        # Allow round-off from unit conversion:
        if abs(value - size) < 1e-6:
            return size
    sizes = ', '.join(str(s) for s in sorted(supported))
    raise ValueError(f"{cls_name} does not model a {value} um pixel. "
                     f"Supported pixel sizes (um): {sizes}.")


def _device_frame(earray, center):
    """Return coordinates in the unrotated device frame."""
    c, s = np.cos(np.radians(earray.rot)), np.sin(np.radians(earray.rot))
    return ((earray.coordinates()[:, :2] - np.asarray(center, dtype=float))
            @ np.array([[c, -s], [s, c]]))


def _recenter(earray, center):
    """Center a trimmed array on ``center``."""
    xy = _device_frame(earray, center)
    off = -0.5 * (xy.min(axis=0) + xy.max(axis=0))
    c, s = np.cos(np.radians(earray.rot)), np.sin(np.radians(earray.rot))
    dx, dy = off[0] * c - off[1] * s, off[0] * s + off[1] * c
    for elec in earray.electrode_objects:
        elec.x += dx
        elec.y += dy


def _trim_to_disc(earray, n_pixels, center):
    """Keep the ``n_pixels`` lattice sites nearest ``center``."""
    xy = _device_frame(earray, center)
    # Round to keep ties stable under rotation:
    r = np.round(np.hypot(*xy.T), 6)
    ang = np.round(np.arctan2(xy[:, 1], xy[:, 0]), 6)
    names = np.asarray(list(earray.electrodes))
    for name in names[np.lexsort((ang, r))[n_pixels:]]:
        earray.remove_electrode(name)


def _trim_to_axial_mask(earray, spans, center):
    """Trim a grid to an axial-coordinate mask."""
    cols = max(spans) - min(spans) + 1
    keep = {i * cols + j for i, j in _axial_rows(spans)}
    for idx, name in enumerate(list(earray.electrodes)):
        if idx not in keep:
            earray.remove_electrode(name)
    # Center reconstructed masks on the substrate:
    _recenter(earray, center)


def _plot_substrate(ax, center, rot, radius=None, side=None):
    """Draw the implant substrate."""
    if ax is None:
        ax = plt.gca()
    style = {'fc': (0.92, 0.92, 0.92, 1), 'ec': (0.6, 0.6, 0.6, 1),
             'lw': 1, 'zorder': ZORDER['background']}
    if radius is not None:
        # A disc is its own rotation, so `rot` does not enter here:
        patch = Circle(center, radius=radius, **style)
    else:
        th = np.radians(rot) + np.radians(45 + 90 * np.arange(4))
        corner = side / np.sqrt(2) * np.column_stack([np.cos(th), np.sin(th)])
        patch = Polygon(np.asarray(center, dtype=float) + corner, closed=True,
                        **style)
    ax.add_patch(patch)
    return ax, patch


def _clip_pixels(ax, substrate, drawn_before):
    """Clip plotted pixels to the substrate."""
    for coll in ax.collections:
        if coll not in drawn_before:
            coll.set_clip_path(substrate)


class PhotovoltaicPixel(HexElectrode):
    """Photovoltaic pixel
    
    A hexagonal pixel body with a circular active electrode at its center.
    
    .. versionadded:: 0.7
    
    Parameters
    ----------
    x/y/z : double
        3D location of the electrode.
        The coordinate system is centered over the fovea.
        Positive ``x`` values move the electrode into the nasal retina.
        Positive ``y`` values move the electrode into the superior retina.
        Positive ``z`` values move the electrode away from the retina into the
        vitreous humor (sometimes called electrode-retina distance).
    r : double
        Radius (um) of the active electrode.
    a : double
        Apothem (um) of the hexagonal pixel body.
    activated : bool
        To deactivate, set to ``False``. Deactivated electrodes cannot receive
        stimuli.
    orientation : {'horizontal', 'vertical'}, optional
        Pixel orientation.
    
        .. versionadded:: 0.11.0
    rot : double, optional
        Rotation angle (deg, positive counter-clockwise).
    
        .. versionadded:: 0.11.0
    
    Notes
    -----
    *  Lengths may be given as plain numbers of microns or as unitful quantities.
       See :py:mod:`pulse2percept.units`.
    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('r',)

    def __init__(self, x, y, z, r, a, name=None, activated=True,
                 orientation='vertical', rot=0):
        super(PhotovoltaicPixel, self).__init__(x, y, z, a, name=name,
                                                activated=activated,
                                                orientation=orientation,
                                                rot=rot)
        r = as_value(r, um, 'r')
        if isinstance(r, (Sequence, np.ndarray)):
            raise TypeError("Radius of the active electrode must be a scalar.")
        if r <= 0:
            raise ValueError("Radius of the active electrode must be > 0, not "
                             "{r}.")
        self.r = r
        # Plot the pixel body and active electrode:
        hex_kwargs = self._hex_patch_kwargs()
        self.plot_patch = [RegularPolygon, Circle]
        self.plot_kwargs = [{**hex_kwargs, 'alpha': 0.2, 'fc': 'k', 'ec': 'k'},
                            {'radius': r, 'linewidth': 0, 'color': 'k',
                             'alpha': 0.5}]
        self.plot_deactivated_kwargs = [{**hex_kwargs, 'alpha': 0.1,
                                         'fc': 'k', 'ec': 'k'},
                                        {'radius': r, 'linewidth': 0,
                                         'color': 'k', 'alpha': 0.2}]

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'r': self.r, 'a': self.a})
        return params

    def electric_potential(self, x, y, z, v0):
        raise NotImplementedError


class PRIMAPivotal(ProsthesisSystem):
    """Create the PRIMA array used in the pivotal PRIMAvera trial
    
    The implant has 378 photovoltaic pixels, each 100 um wide on a 100 um
    hexagonal grid, with a 28 um active electrode, on a 2 x 2 mm substrate
    [Holz2026]_. The same configuration was used in the earlier first-in-human
    study [Palanker2020]_.
    
    .. versionadded:: 0.7
    
    .. versionchanged:: 0.11.0
        Pixel width corrected from 85 to 100 um; pixel centers are unchanged.
    
    .. versionchanged:: 0.11.0
        Renamed from ``PRIMA`` to distinguish this pivotal-trial configuration
        from future commercial hardware.
    
    Parameters
    ----------
    x/y/z : double
        3D location (um) of the center of the electrode array.
        The coordinate system is centered over the fovea.
        Positive ``x`` values move the electrode into the nasal retina.
        Positive ``y`` values move the electrode into the superior retina.
        Positive ``z`` values move the electrode away from the retina into the
        vitreous humor (sometimes called electrode-retina distance).
        ``z`` can either be a list with 378 entries or a scalar that is applied
        to all electrodes.
        May be given as unitful quantities (e.g. ``z=100 * um``); see
        :py:mod:`pulse2percept.units`.
    rot : float or Quantity, optional
        Rotation angle of the array (deg). Positive values denote
        counter-clock-wise (CCW) rotations in the retinal coordinate system.
    eye : {'RE', 'LE'}, optional
        Eye in which array is implanted.
    preprocess : bool or callable, optional
        Either True/False to indicate whether to execute the implant's default
        preprocessing method whenever a stimulus is prepared, or a custom
        function (callable).
    safe_mode : bool, optional
        Enforces the documented projector operating envelope: irradiance
        <= 3.5 mW/mm^2, ON durations on the 0.7 ms grid and <= 9.8 ms,
        duty cycle <= 0.294, and frame rate <= 30 Hz. All 378 pixels may be
        illuminated simultaneously. This is not a biological safety limit or a
        demonstrated hardware maximum.

        .. versionchanged:: 0.11.0
            Checks the optical envelope instead of electrical charge balance.
    encoder : :py:class:`~pulse2percept.stimuli.Encoder`, optional
        Image/video encoder. Defaults to
        :py:class:`~pulse2percept.stimuli.PRIMAEncoder`. Pass ``None`` to
        disable automatic encoding.

        .. versionadded:: 0.11.0

    Notes
    -----
    *  The active-electrode diameter was estimated from Fig. 1 of
       [Palanker2020]_.
    *  :py:meth:`~pulse2percept.implants.ProsthesisSystem.prepare_stim`
       returns optical irradiance (``mW/mm^2``). Photovoltaic conversion is not
       modeled.
    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape', 'spacing', 'pixel_width', 'gap',
                 '_substrate_center')

    placement = 'subretinal'
    technology = 'photovoltaic'
    family = 'PRIMA'

    #: The device is illuminated, not driven by a current source.
    stimulus_unit = mW / mm ** 2

    def __init__(self, x=0, y=0, z=-100, rot=0, eye='RE', preprocess=False,
                 safe_mode=False, encoder=_DEVICE_DEFAULT):
        self.spacing = 100  # um, nearest-neighbor center-to-center
        self.pixel_width = 100  # um, flat-to-flat
        self.gap = self.spacing - self.pixel_width  # um, open inter-pixel gap
        elec_radius = 14  # um
        # Roughly a 19x22 grid, but edges are trimmed off:
        self.shape = (19, 22)
        self.eye = eye
        self.preprocess = preprocess
        self.safe_mode = safe_mode
        # Do not share mutable encoder state between implant instances.
        self.encoder = (PRIMAEncoder() if encoder is _DEVICE_DEFAULT
                        else encoder)

        # Normalized here rather than in ElectrodeGrid, because a
        # per-electrode list of heights never reaches the grid at all -- it is
        # written onto the electrodes further down:
        z = as_value(z, um, 'z')
        # Center the substrate at the requested position:
        self._substrate_center = (as_value(x, um, 'x'), as_value(y, um, 'y'))

        # Assign per-electrode z values after trimming:
        overwrite_z = isinstance(z, (list, np.ndarray))
        zarr = -100 if overwrite_z else z

        self.earray = ElectrodeGrid(self.shape, self.spacing, x=x, y=y,
                                    z=zarr, rot=rot, type='hex',
                                    orientation='vertical',
                                    etype=PhotovoltaicPixel, r=elec_radius,
                                    a=self.pixel_width / 2)

        # Remove extra electrodes to fit the actual implant:
        extra_elecs = ['A1', 'A2', 'A3', 'A4', 'A14', 'A16', 'A17',
                       'A18', 'A19', 'A20', 'A21', 'A22', 'B1',
                       'B2', 'B18', 'B19', 'B20', 'B21', 'B22',
                       'C1', 'C20', 'C21', 'C22', 'D22', 'E22', 'P1',
                       'Q1', 'Q22', 'R1', 'R2', 'R21', 'R22', 'S1',
                       'S2', 'S3', 'S5', 'S19', 'S20', 'S21', 'S22']
        for elec in extra_elecs:
            self.earray.remove_electrode(elec)

        if overwrite_z:
            z_arr = np.asarray(z).flatten()
            if z_arr.size != self.n_electrodes:
                raise ValueError(f"If `z` is a list, it must have "
                                 f"{self.n_electrodes} entries, not "
                                 f"{z_arr.size}.")
            for elec, z_elec in zip(self.earray.electrode_objects, z):
                elec.z = z_elec

    def _require_physical_light(self, stim):
        """Reject negative or nonfinite irradiance."""
        if stim.unit.dimension != self.stimulus_unit.dimension:
            return
        projector = _projector(stim)
        if projector is not None:
            # Read peak irradiance directly from the projector schedule.
            values = np.array([projector.irradiance], dtype=np.float64)
        else:
            values = np.asarray(stim.data, dtype=np.float64)
        if values.size == 0:
            return
        if not np.all(np.isfinite(values)):
            raise ValueError("Optical stimulus has non-finite irradiance.")
        if values.min() < 0:
            raise ValueError(
                f"Optical stimulus asks for {values.min():.3f} mW/mm^2. "
                f"Irradiance is a power density and cannot be negative; "
                f"a dark pixel is zero.")

    def _require_within_optical_envelope(self, stim):
        """Check the documented PRIMA projector operating envelope."""
        if stim.unit.dimension != self.stimulus_unit.dimension:
            raise DimensionMismatchError(
                f"Safety check 'safe_mode' needs an optical stimulus to "
                f"check, and this one is measured in "
                f"{_describe_unit(stim.unit)}. Give the implant a "
                f"PRIMAEncoder so that image or video input is encoded into "
                f"irradiance first.")
        projector = _projector(stim)
        if projector is None:
            # Duty cycle requires the projector schedule, not waveform samples.
            raise ValueError(
                "Safety check: this stimulus no longer describes a projector "
                "(irradiance, frame rate, per-pixel ON durations), so its "
                "duty cycle cannot be verified. Build it with a PRIMAEncoder "
                "and keep it intact, or set safe_mode=False and check the "
                "device envelope yourself.")
        irradiance, freq = projector.irradiance, projector.freq
        dur = np.asarray(projector.pulse_dur, dtype=np.float64)
        step = PRIMAEncoder.pulse_step
        if irradiance > PRIMAEncoder.max_irradiance + 1e-9:
            raise ValueError(
                f"Safety check: stimulus projects {irradiance:.3f} mW/mm^2, "
                f"which exceeds the {PRIMAEncoder.max_irradiance} mW/mm^2 the "
                f"device delivers.")
        longest = float(dur.max(initial=0.0))
        if longest > PRIMAEncoder.max_pulse_dur + 1e-9:
            raise ValueError(
                f"Safety check: stimulus lights a pixel for {longest:.3f} ms, "
                f"which exceeds the longest documented ON duration of "
                f"{PRIMAEncoder.max_pulse_dur} ms.")
        # Require 0.7 ms ON-duration steps.
        steps = dur[dur > 0] / step
        if steps.size and np.any(np.abs(steps - np.round(steps)) > 1e-6):
            raise ValueError(
                f"Safety check: ON durations must be whole multiples of "
                f"{step} ms, the step the projector modulates in.")
        # Check duty cycle before frame rate to catch combined violations.
        duty = freq * longest / MS_PER_S
        if duty > PRIMAEncoder.max_duty_cycle + 1e-9:
            raise ValueError(
                f"Safety check: stimulus asks for a duty cycle of "
                f"{duty:.3f} ({freq:g} Hz x {longest:.3f} ms), which exceeds "
                f"the {PRIMAEncoder.max_duty_cycle:.3f} the device delivers. "
                f"Lower 'freq' or shorten 'pulse_dur'.")
        if freq > PRIMAEncoder.max_freq + 1e-9:
            raise ValueError(
                f"Safety check: stimulus runs the projector at {freq:g} Hz, "
                f"and the pivotal system is reported to run at "
                f"{PRIMAEncoder.max_freq:g} Hz. Set safe_mode=False to "
                f"explore other frame rates.")

    def check_stim(self, stim):
        """Validate optical stimulation and, in safe mode, projector limits.

        .. versionadded:: 0.11.0
        """
        self._require_physical_light(stim)
        if self.safe_mode:
            self._require_within_optical_envelope(stim)
        if self.max_current is not None:
            # The inherited current-limit check rejects optical units.
            self._require_within_current_limit(stim)

    def plot(self, annotate=False, autoscale=True, ax=None, stim=None,
             stim_cmap=False):
        """Plot the implant and its 2 x 2 mm substrate.
        
        .. versionadded:: 0.11.0
        """
        ax, substrate = _plot_substrate(ax, self._substrate_center,
                                        self.earray.rot, side=2000)
        drawn = list(ax.collections)
        ax = super().plot(annotate=annotate, autoscale=autoscale, ax=ax,
                          stim=stim, stim_cmap=stim_cmap)
        _clip_pixels(ax, substrate, drawn)
        return ax

    @property
    def row_spacing(self):
        """Distance (um) between adjacent pixel rows."""
        return self.spacing * np.sqrt(3) / 2


class Lorach2015Array(ProsthesisSystem):
    """Create the 70 um photovoltaic array of [Lorach2015]_
    
    The array has 142 pixels, each 70 um wide on a 75 um hexagonal grid, with a
    20 um active electrode, on a nominal 1 mm substrate.
    
    .. versionadded:: 0.7
    
    .. versionchanged:: 0.11.0
        Renamed from ``PRIMA75``; that name was pulse2percept shorthand.
    
    Parameters
    ----------
    x/y/z : double
        3D location (um) of the center of the electrode array.
        The coordinate system is centered over the fovea.
        Positive ``x`` values move the electrode into the nasal retina.
        Positive ``y`` values move the electrode into the superior retina.
        Positive ``z`` values move the electrode away from the retina into the
        vitreous humor (sometimes called electrode-retina distance).
        ``z`` can either be a list with 142 entries or a scalar that is applied
        to all electrodes.
        May be given as unitful quantities (e.g. ``z=100 * um``); see
        :py:mod:`pulse2percept.units`.
    rot : float or Quantity, optional
        Rotation angle of the array (deg). Positive values denote
        counter-clock-wise (CCW) rotations in the retinal coordinate system.
    eye : {'RE', 'LE'}, optional
        Eye in which array is implanted.
    preprocess : bool or callable, optional
        Either True/False to indicate whether to execute the implant's default
        preprocessing method whenever a stimulus is prepared, or a custom
        function (callable).
    safe_mode : bool, optional
        If safe mode is enabled, only charge-balanced stimuli are allowed.
    
    Notes
    -----
    *  [Lorach2015]_ reports the 65 um row spacing as the "pixel pitch".
    *  Seven rim pixels extend beyond the nominal 1 mm substrate and are clipped
       when plotted.
    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape', 'spacing', 'pixel_width', 'gap',
                 '_substrate_center')

    placement = 'subretinal'
    technology = 'photovoltaic'

    def __init__(self, x=0, y=0, z=-100, rot=0, eye='RE', preprocess=False,
                 safe_mode=False):
        self.spacing = 75  # um, nearest-neighbor center-to-center
        self.pixel_width = 70  # um, flat-to-flat
        self.gap = self.spacing - self.pixel_width  # um, open inter-pixel gap
        elec_radius = 10  # um
        # Roughly a 12x15 grid, but edges are trimmed off:
        self.shape = (12, 15)
        self.eye = eye
        self.preprocess = preprocess
        self.safe_mode = safe_mode

        # Normalized here rather than in ElectrodeGrid, because a
        # per-electrode list of heights never reaches the grid at all -- it is
        # written onto the electrodes further down:
        z = as_value(z, um, 'z')
        # Center the substrate at the requested position:
        self._substrate_center = (as_value(x, um, 'x'), as_value(y, um, 'y'))

        # Assign per-electrode z values after trimming:
        overwrite_z = isinstance(z, (list, np.ndarray))
        zarr = -100 if overwrite_z else z

        self.earray = ElectrodeGrid(self.shape, self.spacing, x=x, y=y,
                                    z=zarr, rot=rot, type='hex',
                                    orientation='vertical',
                                    etype=PhotovoltaicPixel, r=elec_radius,
                                    a=self.pixel_width / 2)

        # Remove extra electrodes to fit the actual implant:
        extra_elecs = ['A1', 'B1', 'C1', 'D1', 'E1', 'I1', 'J1', 'K1', 'L1',
                       'A2', 'B2', 'C2', 'D2', 'K2', 'L2',
                       'A3', 'B3', 'L3',
                       'A4',
                       'A12',
                       'A13', 'K13', 'L13',
                       'A14', 'B14', 'C14', 'J14', 'K14', 'L14',
                       'A15', 'B15', 'C15', 'D15', 'H15', 'I15', 'J15', 'K15',
                       'L15']
        for elec in extra_elecs:
            self.earray.remove_electrode(elec)

        if overwrite_z:
            z_arr = np.asarray(z).flatten()
            if z_arr.size != self.n_electrodes:
                raise ValueError(f"If `z` is a list, it must have "
                                 f"{self.n_electrodes} entries, not "
                                 f"{z_arr.size}.")
            for elec, z_elec in zip(self.earray.electrode_objects, z):
                elec.z = z_elec

    def plot(self, annotate=False, autoscale=True, ax=None, stim=None,
             stim_cmap=False):
        """Plot the implant and its 1 mm substrate.
        
        .. versionadded:: 0.11.0
        """
        ax, substrate = _plot_substrate(ax, self._substrate_center,
                                        self.earray.rot, radius=500)
        drawn = list(ax.collections)
        ax = super().plot(annotate=annotate, autoscale=autoscale, ax=ax,
                          stim=stim, stim_cmap=stim_cmap)
        _clip_pixels(ax, substrate, drawn)
        return ax

    @property
    def row_spacing(self):
        """Distance (um) between adjacent pixel rows."""
        return self.spacing * np.sqrt(3) / 2


class Ho2019FlatArray(ProsthesisSystem):
    """Create a flat photovoltaic array of [Ho2019]_
    
    Supports the F55 and F40 arrays on a 1 mm substrate:
    
    ===================  ======  ===========  ================
    ``pixel_size`` (um)  pixels  row spacing  active electrode
    ===================  ======  ===========  ================
    55 (F55)             250     47.6 um      14 um diameter
    40 (F40)             502     34.6 um      10 um diameter
    ===================  ======  ===========  ================
    
    .. versionadded:: 0.11.0
    
    Parameters
    ----------
    pixel_size : {55, 40}
        Pixel width (um), which selects the device variant.
    x/y/z : double
        3D location (um) of the center of the electrode array.
        The coordinate system is centered over the fovea.
        Positive ``x`` values move the electrode into the nasal retina.
        Positive ``y`` values move the electrode into the superior retina.
        Positive ``z`` values move the electrode away from the retina into the
        vitreous humor (sometimes called electrode-retina distance).
        ``z`` may be a scalar or one value per pixel.
        May be given as unitful quantities; see :py:mod:`pulse2percept.units`.
    rot : float or Quantity, optional
        Rotation angle of the array (deg). Positive values denote
        counter-clock-wise (CCW) rotations in the retinal coordinate system.
    eye : {'RE', 'LE'}, optional
        Eye in which array is implanted.
    preprocess : bool or callable, optional
        Either True/False to indicate whether to execute the implant's default
        preprocessing method whenever a stimulus is prepared, or a custom
        function (callable).
    safe_mode : bool, optional
        If safe mode is enabled, only charge-balanced stimuli are allowed.
    
    Notes
    -----
    *  [Ho2019]_ also describes pillar arrays Pil55 and Pil40, which are not
       modeled here.
    *  The F55 layout is reconstructed from Fig. 2(a). The F40 outline is not
       published; its 502 pixels are taken as the nearest lattice sites to the
       substrate center.
    *  The 1 um isolation trenches are covered by the shared return electrode, so
       ``gap`` is 0.
    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('pixel_size', 'shape', 'spacing', 'pixel_width', 'gap',
                 '_substrate_center')

    placement = 'subretinal'
    technology = 'photovoltaic'

    def __init__(self, pixel_size, x=0, y=0, z=-100, rot=0, eye='RE',
                 preprocess=False, safe_mode=False):
        self.pixel_size = _pixel_size_um(pixel_size, _HO2019_VARIANTS,
                                         'Ho2019FlatArray')
        spec = _HO2019_VARIANTS[self.pixel_size]
        self.spacing = self.pixel_size  # um, nearest-neighbor center-to-center
        self.pixel_width = self.pixel_size  # um, flat-to-flat
        self.gap = self.spacing - self.pixel_width  # um, open inter-pixel gap
        self.shape = spec['shape']
        self.eye = eye
        self.preprocess = preprocess
        self.safe_mode = safe_mode

        # Normalized here rather than in ElectrodeGrid, because a
        # per-electrode list of heights never reaches the grid at all -- it is
        # written onto the electrodes further down:
        z = as_value(z, um, 'z')
        # Center the substrate at the requested position:
        self._substrate_center = (as_value(x, um, 'x'), as_value(y, um, 'y'))

        # Assign per-electrode z values after trimming:
        overwrite_z = isinstance(z, (list, np.ndarray))
        zarr = -100 if overwrite_z else z

        self.earray = ElectrodeGrid(self.shape, self.spacing, x=x, y=y,
                                    z=zarr, rot=rot, type='hex',
                                    orientation='vertical',
                                    etype=PhotovoltaicPixel,
                                    r=spec['elec_radius'],
                                    a=self.pixel_width / 2)
        if spec['spans'] is not None:
            _trim_to_axial_mask(self.earray, spec['spans'],
                                self._substrate_center)
        else:
            _trim_to_disc(self.earray, spec['n_pixels'],
                          self._substrate_center)

        if overwrite_z:
            z_arr = np.asarray(z).flatten()
            if z_arr.size != self.n_electrodes:
                raise ValueError(f"If `z` is a list, it must have "
                                 f"{self.n_electrodes} entries, not "
                                 f"{z_arr.size}.")
            for elec, z_elec in zip(self.earray.electrode_objects, z):
                elec.z = z_elec

    def plot(self, annotate=False, autoscale=True, ax=None, stim=None,
             stim_cmap=False):
        """Plot the implant and its 1 mm substrate.
        
        .. versionadded:: 0.11.0
        """
        ax, substrate = _plot_substrate(ax, self._substrate_center,
                                        self.earray.rot, radius=500)
        drawn = list(ax.collections)
        ax = super().plot(annotate=annotate, autoscale=autoscale, ax=ax,
                          stim=stim, stim_cmap=stim_cmap)
        _clip_pixels(ax, substrate, drawn)
        return ax

    @property
    def row_spacing(self):
        """Distance (um) between adjacent pixel rows."""
        return self.spacing * np.sqrt(3) / 2


class Huang2021Array(ProsthesisSystem):
    """Create a vertical-junction photovoltaic array of [Huang2021]_
    
    Supports four arrays on a 1.5 mm substrate. Only exposed pixels are modeled
    as electrodes:
    
    ===================  ================  =================  ================
    ``pixel_size`` (um)  ``n_electrodes``  fabricated pixels  active electrode
    ===================  ================  =================  ================
    55                   421               526                22 um diameter
    40                   821               1027               16 um diameter
    30                   1388              1735               12 um diameter
    20                   2806              3508               8 um diameter
    ===================  ================  =================  ================
    
    .. versionadded:: 0.11.0
    
    Parameters
    ----------
    pixel_size : {55, 40, 30, 20}
        Pixel width (um), which selects the device variant.
    x/y/z : double
        3D location (um) of the center of the electrode array.
        The coordinate system is centered over the fovea.
        Positive ``x`` values move the electrode into the nasal retina.
        Positive ``y`` values move the electrode into the superior retina.
        Positive ``z`` values move the electrode away from the retina into the
        vitreous humor (sometimes called electrode-retina distance).
        ``z`` may be a scalar or one value per exposed pixel.
        May be given as unitful quantities; see :py:mod:`pulse2percept.units`.
    rot : float or Quantity, optional
        Rotation angle of the array (deg). Positive values denote
        counter-clock-wise (CCW) rotations in the retinal coordinate system.
    eye : {'RE', 'LE'}, optional
        Eye in which array is implanted.
    preprocess : bool or callable, optional
        Either True/False to indicate whether to execute the implant's default
        preprocessing method whenever a stimulus is prepared, or a custom
        function (callable).
    safe_mode : bool, optional
        If safe mode is enabled, only charge-balanced stimuli are allowed.
    
    Notes
    -----
    *  ``n_total_pixels`` gives the fabricated pixel count; peripheral pixels
       covered by the common return are not modeled as electrodes.
    *  Exposed-pixel layouts are reconstructed from Fig. 7 of [Huang2021]_.
       Three rim pixels of the 20 um array are inferred; one is ambiguous.
    *  Isolation trenches are covered by the shared return electrode, so
       ``gap`` is 0.
    *  The substrate is modeled as a 1.5 mm circle; the small die notch is omitted.
    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('pixel_size', 'n_total_pixels', 'shape', 'spacing',
                 'pixel_width', 'gap', '_substrate_center')

    placement = 'subretinal'
    technology = 'photovoltaic'

    def __init__(self, pixel_size, x=0, y=0, z=-100, rot=0, eye='RE',
                 preprocess=False, safe_mode=False):
        self.pixel_size = _pixel_size_um(pixel_size, _HUANG2021_AXIAL_SPANS,
                                         'Huang2021Array')
        spans = _HUANG2021_AXIAL_SPANS[self.pixel_size]
        self.n_total_pixels = _HUANG2021_TOTAL_PIXELS[self.pixel_size]
        self.spacing = self.pixel_size  # um, nearest-neighbor center-to-center
        self.pixel_width = self.pixel_size  # um, flat-to-flat
        self.gap = self.spacing - self.pixel_width  # um, open inter-pixel gap
        # Active-electrode diameter is 40% of the pixel size:
        elec_radius = self.pixel_size * 0.2
        self.shape = _axial_mask_shape(spans)
        self.eye = eye
        self.preprocess = preprocess
        self.safe_mode = safe_mode

        # Normalized here rather than in ElectrodeGrid, because a
        # per-electrode list of heights never reaches the grid at all -- it is
        # written onto the electrodes further down:
        z = as_value(z, um, 'z')
        # Center the substrate at the requested position:
        self._substrate_center = (as_value(x, um, 'x'), as_value(y, um, 'y'))

        # Assign per-electrode z values after trimming:
        overwrite_z = isinstance(z, (list, np.ndarray))
        zarr = -100 if overwrite_z else z

        self.earray = ElectrodeGrid(self.shape, self.spacing, x=x, y=y,
                                    z=zarr, rot=rot, type='hex',
                                    orientation='vertical',
                                    etype=PhotovoltaicPixel, r=elec_radius,
                                    a=self.pixel_width / 2)
        _trim_to_axial_mask(self.earray, spans, self._substrate_center)

        if overwrite_z:
            z_arr = np.asarray(z).flatten()
            if z_arr.size != self.n_electrodes:
                raise ValueError(f"If `z` is a list, it must have "
                                 f"{self.n_electrodes} entries, not "
                                 f"{z_arr.size}.")
            for elec, z_elec in zip(self.earray.electrode_objects, z):
                elec.z = z_elec

    def plot(self, annotate=False, autoscale=True, ax=None, stim=None,
             stim_cmap=False):
        """Plot the implant and its 1.5 mm substrate.
        
        .. versionadded:: 0.11.0
        """
        ax, substrate = _plot_substrate(ax, self._substrate_center,
                                        self.earray.rot, radius=750)
        drawn = list(ax.collections)
        ax = super().plot(annotate=annotate, autoscale=autoscale, ax=ax,
                          stim=stim, stim_cmap=stim_cmap)
        _clip_pixels(ax, substrate, drawn)
        return ax

    @property
    def row_spacing(self):
        """Distance (um) between adjacent pixel rows."""
        return self.spacing * np.sqrt(3) / 2


@deprecated(alt_func='Ho2019FlatArray(55)', deprecated_version='0.11.0',
            removed_version='0.12.0',
            extra_msg='The name is ambiguous: 55 um arrays appear in two '
                      'device families. This one is the flat F55 array of '
                      'Ho et al. (2019); the 1.5 mm vertical-junction array '
                      'of Huang et al. (2021) is ``Huang2021Array(55)``.')
class PRIMA55(Ho2019FlatArray):
    """Deprecated name for the F55 array of [Ho2019]_.
    
    .. deprecated:: 0.11.0
        Use ``Ho2019FlatArray(55)`` instead.
    """
    __slots__ = ()

    def __init__(self, x=0, y=0, z=-100, rot=0, eye='RE', preprocess=False,
                 safe_mode=False):
        super().__init__(55, x=x, y=y, z=z, rot=rot, eye=eye,
                         preprocess=preprocess, safe_mode=safe_mode)


@deprecated(alt_func='Ho2019FlatArray(40)', deprecated_version='0.11.0',
            removed_version='0.12.0',
            extra_msg='The name is ambiguous: 40 um arrays appear in two '
                      'device families. This one is the flat F40 array of '
                      'Ho et al. (2019); the 1.5 mm vertical-junction array '
                      'of Huang et al. (2021) is ``Huang2021Array(40)``.')
class PRIMA40(Ho2019FlatArray):
    """Deprecated name for the F40 array of [Ho2019]_.
    
    .. deprecated:: 0.11.0
        Use ``Ho2019FlatArray(40)`` instead.
    """
    __slots__ = ()

    def __init__(self, x=0, y=0, z=-100, rot=0, eye='RE', preprocess=False,
                 safe_mode=False):
        super().__init__(40, x=x, y=y, z=z, rot=rot, eye=eye,
                         preprocess=preprocess, safe_mode=safe_mode)


@deprecated(alt_func='PRIMAPivotal', deprecated_version='0.11.0',
            removed_version='0.12.0',
            extra_msg='``PRIMA`` is reserved for the eventual commercial '
                      'device, whose specifications may differ from the '
                      'pivotal-trial configuration this class models.')
class PRIMA(PRIMAPivotal):
    """Deprecated name for the pivotal-trial PRIMA array.
    
    .. deprecated:: 0.11.0
        Use :py:class:`~pulse2percept.implants.PRIMAPivotal` instead.
    """
    __slots__ = ()


@deprecated(alt_func='Lorach2015Array', deprecated_version='0.11.0',
            removed_version='0.12.0',
            extra_msg='``PRIMA75`` was pulse2percept shorthand, not an '
                      'official device name.')
class PRIMA75(Lorach2015Array):
    """Deprecated name for the 70 um array of [Lorach2015]_.
    
    .. deprecated:: 0.11.0
        Use :py:class:`~pulse2percept.implants.Lorach2015Array` instead.
    """
    __slots__ = ()
