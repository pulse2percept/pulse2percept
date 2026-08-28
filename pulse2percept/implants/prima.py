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
from ..units import as_value, um
from ..utils import deprecated
from ..utils.constants import ZORDER


#: Layout of the F55 array of [Ho2019]_ in axial hex coordinates: for each
#: column ``q``, the inclusive range of rows ``r`` that carries a pixel.
#: Recovered from Fig. 2(a) of [Ho2019]_; exactly 250 pixels.
_HO2019_F55_AXIAL_SPANS = {
    -9: (3, 8), -8: (1, 9), -7: (-1, 9), -6: (-3, 9), -5: (-4, 9),
    -4: (-5, 9), -3: (-6, 9), -2: (-6, 8), -1: (-7, 8), 0: (-7, 7),
    1: (-8, 7), 2: (-8, 6), 3: (-9, 6), 4: (-9, 5), 5: (-9, 4),
    6: (-9, 3), 7: (-9, 2), 8: (-9, 1), 9: (-8, -1),
}

#: Exposed (stimulating) pixels of the vertical-junction arrays of
#: [Huang2021]_ in axial hex coordinates: for each pixel size (um) and column
#: ``q``, the inclusive range of rows ``r`` that carries an exposed pixel.
#: Reconstructed from Fig. 7 of [Huang2021]_, registered to the triangular
#: lattice, and constrained to the published exposed-pixel counts.
#: The peripheral pixels covered by the common return electrode are not in
#: here; see ``_HUANG2021_TOTAL_PIXELS`` for the fabricated totals.
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
    # Three of the 2806 exposed pixels -- (11, 19), (24, 3) and (0, 27) -- lie
    # outside the quadrant Fig. 7 of [Huang2021]_ shows for this variant and
    # are inferred: they are the sites the published total requires, at the
    # nearest admissible rim positions. Only (0, 27) is meaningfully ambiguous;
    # (-9, 31) is an almost exact geometric tie, and (0, 27) wins by sitting
    # fractionally closer to the fitted center of the active region. Full-
    # device imagery or CAD would settle that one site.
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

#: Pixels fabricated on each [Huang2021]_ die, keyed by pixel size (um). Larger
#: than the exposed counts in ``_HUANG2021_AXIAL_SPANS``: the peripheral common
#: return electrode covers roughly 20% of the pixels, which are therefore not
#: exposed as independently stimulating pixels and are not modeled as
#: electrodes.
_HUANG2021_TOTAL_PIXELS = {55: 526, 40: 1027, 30: 1735, 20: 3508}


def _axial_rows(spans):
    """Grid row index of every ``(q, r)`` in an axial-coordinate mask

    On an ``orientation='vertical'`` hex grid, axial column ``q`` is grid
    column ``j = q - min(spans)``, and axial row ``r`` is grid row
    ``r + (j + 1) // 2``, offset so the lowest one is row 0. The ``(j + 1)
    // 2`` term undoes the half-spacing stagger the grid applies to its
    even-numbered columns.
    """
    q0 = min(spans)
    ij = [(r + ((q - q0) + 1) // 2, q - q0)
          for q, (r_lo, r_hi) in spans.items()
          for r in range(r_lo, r_hi + 1)]
    lo = min(i for i, _ in ij)
    return [(i - lo, j) for i, j in ij]


def _axial_mask_shape(spans):
    """Smallest ``(rows, cols)`` hex grid holding an axial-coordinate mask"""
    ij = _axial_rows(spans)
    return max(i for i, _ in ij) + 1, max(spans) - min(spans) + 1


#: The flat arrays of [Ho2019]_, keyed by pixel size (um): the radius (um) of
#: the active electrode, the hex grid the layout is cut from, and how the
#: layout is defined -- an image-derived axial mask for F55, the ``n_pixels``
#: sites nearest the substrate center for F40, whose outline is unpublished.
_HO2019_VARIANTS = {
    55: {'elec_radius': 7, 'n_pixels': 250,
         'shape': _axial_mask_shape(_HO2019_F55_AXIAL_SPANS),
         'spans': _HO2019_F55_AXIAL_SPANS},
    # Smallest grid that holds the 502 nearest sites without the trim having
    # to split a ring of equidistant ones:
    40: {'elec_radius': 5, 'n_pixels': 502, 'shape': (26, 27), 'spans': None},
}


def _pixel_size_um(pixel_size, supported, cls_name):
    """Validated pixel size (um), keyed into a variant table

    Takes a bare number of microns or a length quantity, and requires one of
    the ``supported`` sizes: each layout is a per-device reconstruction, not a
    formula that could be evaluated at an arbitrary size.
    """
    value = as_value(pixel_size, um, 'pixel_size')
    if isinstance(value, (Sequence, np.ndarray)):
        raise TypeError(f"'pixel_size' must be a scalar, not "
                        f"{type(pixel_size)}.")
    for size in supported:
        # Loose enough to absorb the round-off a unit conversion leaves behind
        # (0.055 * mm is 55.00000000000001 um), far tighter than the gap
        # between two variants:
        if abs(value - size) < 1e-6:
            return size
    sizes = ', '.join(str(s) for s in sorted(supported))
    raise ValueError(f"{cls_name} does not model a {value} um pixel. "
                     f"Supported pixel sizes (um): {sizes}.")


def _device_frame(earray, center):
    """Electrode coordinates relative to ``center``, with the grid's rotation
    undone

    ``center`` is in microns and the rotation comes from ``earray``, which
    stores it normalized, so a unitful ``PRIMA55(x=1 * mm, rot=30 * deg)``
    trims exactly like the bare-number spelling.
    """
    c, s = np.cos(np.radians(earray.rot)), np.sin(np.radians(earray.rot))
    return ((earray.coordinates()[:, :2] - np.asarray(center, dtype=float))
            @ np.array([[c, -s], [s, c]]))


def _recenter(earray, center):
    """Shift a trimmed array so its footprint is centered on ``center``

    For a layout whose registration against the substrate is not published:
    :py:class:`~pulse2percept.implants.ElectrodeGrid` centers the untrimmed
    lattice, so whichever pixels survive a trim generally sit a fraction of
    the spacing off that center. The correction is computed in the unrotated
    device frame, so a rotated device is the unrotated one turned about
    ``center`` rather than a differently trimmed array.

    Do not apply this to a layout that is *defined* relative to the substrate
    center, such as one from :py:func:`_trim_to_disc`: shifting those pixels
    changes the lattice phase and they are no longer the sites the rule
    selected.
    """
    xy = _device_frame(earray, center)
    off = -0.5 * (xy.min(axis=0) + xy.max(axis=0))
    c, s = np.cos(np.radians(earray.rot)), np.sin(np.radians(earray.rot))
    dx, dy = off[0] * c - off[1] * s, off[0] * s + off[1] * c
    for elec in earray.electrode_objects:
        elec.x += dx
        elec.y += dy


def _trim_to_disc(earray, n_pixels, center):
    """Trim a hex grid down to the ``n_pixels`` pixels nearest ``center``

    For a device whose pixel count and substrate diameter are published but
    whose outline is not: the pixels kept are the lattice sites closest to the
    center of the substrate, which is the most circular layout with that
    count. It is not the fabrication mask. The sites are kept where the
    lattice puts them; their bounding box can sit up to half a spacing off
    center, which is what a discrete lattice on a round substrate does.

    Distances and the tie-breaking angle are measured in the unrotated device
    frame, so ``rot`` turns the device without changing which pixels it has.
    """
    xy = _device_frame(earray, center)
    # Rounded, so that round-off from the rotation cannot reorder two sites
    # that are the same distance out:
    r = np.round(np.hypot(*xy.T), 6)
    ang = np.round(np.arctan2(xy[:, 1], xy[:, 0]), 6)
    names = np.asarray(list(earray.electrodes))
    for name in names[np.lexsort((ang, r))[n_pixels:]]:
        earray.remove_electrode(name)


def _trim_to_axial_mask(earray, spans, center):
    """Trim a hex grid down to the pixels named by an axial-coordinate mask"""
    cols = max(spans) - min(spans) + 1
    keep = {i * cols + j for i, j in _axial_rows(spans)}
    for idx, name in enumerate(list(earray.electrodes)):
        if idx not in keep:
            earray.remove_electrode(name)
    # The mask fixes which pixels exist, but not where the reconstructed
    # outline sits on the substrate; centering it there is the modeling
    # choice:
    _recenter(earray, center)


def _plot_substrate(ax, center, rot, radius=None, side=None):
    """Draw a PRIMA substrate outline and return ``(ax, patch)``

    The silicon die, not the pixel array: give either a ``radius`` for a round
    substrate or a ``side`` for a square one. The patch goes in at background
    z-order so the pixels stay on top however the caller layers its own
    drawing, and it enters the data limits, so ``autoscale`` sees the chip and
    not just the pixels.
    """
    if ax is None:
        ax = plt.gca()
    # Deliberately low-contrast: the substrate says where the chip ends, and
    # should not compete with the pixels drawn on it.
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
    """Clip the pixels an implant just drew to its substrate outline

    A pixel at the rim of a round die is diced through: the lattice site and
    its stimulation are unaffected, but the silicon of the hexagon is cut off
    at the edge of the chip. p2p draws every pixel as a whole hexagon, so the
    truncation is applied here, to the drawing.
    """
    for coll in ax.collections:
        if coll not in drawn_before:
            coll.set_clip_path(substrate)


class PhotovoltaicPixel(HexElectrode):
    """Photovoltaic pixel

    A hexagonal pixel body with a circular active electrode at its center, as
    used by the subretinal photovoltaic arrays modeled in this module.

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
        Radius (um) of the circular active electrode in the x,y plane.
    a : double
        Apothem (um) of the hexagonal pixel body: half its flat-to-flat width.
    activated : bool
        To deactivate, set to ``False``. Deactivated electrodes cannot receive
        stimuli.
    orientation : {'horizontal', 'vertical'}, optional
        Which way the pixel body's flats face; see
        :py:class:`~pulse2percept.implants.HexElectrode`.

        .. versionadded:: 0.11.0
    rot : double, optional
        Rotation of the pixel body (deg, counter-clockwise).

        .. versionadded:: 0.11.0

    Notes
    -----
    *  Lengths may be given as plain numbers of microns or as unitful
       quantities (e.g. ``14 * um``). See :py:mod:`pulse2percept.units`.

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
        # Plot two objects: hex pixel body and circular active electrode. The
        # body reuses HexElectrode's geometry so the two cannot drift apart:
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
    """Create a PRIMA array as used in the pivotal trial

    This class creates the PRIMA array with 378 photovoltaic pixels used in the
    pivotal PRIMAvera clinical trial [Holz2026]_, and places it in the
    subretinal space such that the center of the array is located at 3D
    location (x,y,z), given in microns, and the array is rotated by rotation
    angle ``rot``, given in degrees. The same 378-pixel, 100 um configuration
    was described earlier in the first-in-human study [Palanker2020]_.

    Each hexagonal pixel is 100 um wide (flat-to-flat), and neighboring pixel
    centers are 100 um apart, on a 2 x 2 mm substrate. Adjacent rows are
    therefore separated by ``100 * sqrt(3) / 2`` = 86.6 um, which sets the
    sampling limit of the hexagonal array. The active electrode at the center
    of each pixel is a disk 28 um in diameter.

    .. versionadded:: 0.7

    .. versionchanged:: 0.11.0
        Pixels are 100 um wide, matching the pixel size reported throughout
        the clinical PRIMA literature. Earlier versions drew 85 um pixels
        separated by 15 um trenches, an interpretation no primary source
        supports. Pixel centers are unchanged.

    .. versionchanged:: 0.11.0
        Was called ``PRIMA``. The qualified name identifies this one fixed
        published hardware configuration, and leaves ``PRIMA`` free for the
        eventual commercial device, whose specifications may differ.

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
        counter-clock-wise (CCW) rotations in the retinal coordinate
        system.
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
    *  The diameter of the active electrode was estimated from Fig. 1 in
       [Palanker2020]_.

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape', 'spacing', 'pixel_width', 'gap',
                 '_substrate_center')

    placement = 'subretinal'
    technology = 'photovoltaic'
    family = 'PRIMA'

    def __init__(self, x=0, y=0, z=-100, rot=0, eye='RE', preprocess=False,
                 safe_mode=False):
        self.spacing = 100  # um, nearest-neighbor center-to-center
        self.pixel_width = 100  # um, flat-to-flat
        self.gap = self.spacing - self.pixel_width  # um, open inter-pixel gap
        elec_radius = 14  # um
        # Roughly a 19x22 grid, but edges are trimmed off:
        self.shape = (19, 22)
        self.eye = eye
        self.preprocess = preprocess
        self.safe_mode = safe_mode

        # Normalized here rather than in ElectrodeGrid, because a
        # per-electrode list of heights never reaches the grid at all -- it is
        # written onto the electrodes further down:
        z = as_value(z, um, 'z')
        # The substrate is centered on the requested position, which is not
        # the same point as the center of the pixels that survive trimming:
        self._substrate_center = (as_value(x, um, 'x'), as_value(y, um, 'y'))

        # A per-electrode `z` is one entry per surviving pixel, not one per
        # site of the untrimmed grid, so it is written on after trimming:
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

    def plot(self, annotate=False, autoscale=True, ax=None, stim=None,
             stim_cmap=False):
        """Plot the implant on its 2 x 2 mm substrate

        Takes the same arguments as
        :py:meth:`~pulse2percept.implants.ProsthesisSystem.plot`, and draws
        the substrate behind the pixels. Pixels at the rim are clipped to
        the substrate outline, since the die cuts through them.

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
        """Distance (um) between adjacent rows of pixels

        Derived geometry: ``spacing * sqrt(3) / 2``. Papers on this device
        family call this the "pixel pitch"; it is not the nearest-neighbor
        center spacing.
        """
        return self.spacing * np.sqrt(3) / 2


class Lorach2015Array(ProsthesisSystem):
    """Create the 70 um photovoltaic array of [Lorach2015]_ on the retina

    This class creates the array of 142 photovoltaic pixels described in
    [Lorach2015]_, and places it in the subretinal space, such that that the
    center of the array is located at 3D location (x,y,z), given in microns,
    and the array is rotated by rotation angle ``rot``, given in degrees.

    Each hexagonal pixel is 70 um wide (flat-to-flat) and neighboring pixel
    centers are 75 um apart, leaving a 5 um open trench between pixel bodies,
    on a nominally 1 mm substrate. Adjacent rows are therefore separated by
    ``75 * sqrt(3) / 2`` = 65 um. The active electrode at the center of each
    pixel is a disk 20 um in diameter.

    .. versionadded:: 0.7

    .. versionchanged:: 0.11.0
        Was called ``PRIMA75``, which was pulse2percept shorthand for the
        70 um array of [Lorach2015]_ rather than an official device name.

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
        counter-clock-wise (CCW) rotations in the retinal coordinate
        system.
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
    *  [Lorach2015]_ calls the 65 um row spacing the "pixel pitch". The
       nearest-neighbor center spacing, which is what ``spacing`` means here,
       is 75 um.
    *  142 whole 70 um hexagons do not fit inside a 1 mm circle at this
       spacing, so the peripheral pixels of the real device are cut through
       by the diced edge of the chip. Every pixel center is on the substrate;
       :py:meth:`plot` clips the seven rim pixels whose bodies cross it.

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
        # The substrate is centered on the requested position, which is not
        # the same point as the center of the pixels that survive trimming:
        self._substrate_center = (as_value(x, um, 'x'), as_value(y, um, 'y'))

        # A per-electrode `z` is one entry per surviving pixel, not one per
        # site of the untrimmed grid, so it is written on after trimming:
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
        """Plot the implant on its 1 mm substrate

        Takes the same arguments as
        :py:meth:`~pulse2percept.implants.ProsthesisSystem.plot`, and draws
        the substrate behind the pixels. Pixels at the rim are clipped to
        the substrate outline, since the die cuts through them.

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
        """Distance (um) between adjacent rows of pixels

        Derived geometry: ``spacing * sqrt(3) / 2``. Papers on this device
        family call this the "pixel pitch"; it is not the nearest-neighbor
        center spacing.
        """
        return self.spacing * np.sqrt(3) / 2


class Ho2019FlatArray(ProsthesisSystem):
    """Create a flat photovoltaic array of [Ho2019]_ on the retina

    The experimental flat subretinal arrays of [Ho2019]_, placed in the
    subretinal space such that the center of the array is located at 3D
    location (x,y,z), given in microns, and rotated by rotation angle
    ``rot``, given in degrees.

    Two variants are available, both on a 1 mm circular substrate:

    ===================  ======  ===========  ================
    ``pixel_size`` (um)  pixels  row spacing  active electrode
    ===================  ======  ===========  ================
    55 (F55)             250     47.6 um      14 um diameter
    40 (F40)             502     34.6 um      10 um diameter
    ===================  ======  ===========  ================

    Pixel bodies tile the lattice without an open gap, so the flat-to-flat
    pixel width equals the nearest-neighbor center spacing. The modeled pixel
    footprint is approximately 921 x 880 um (F55) and 947 x 960 um (F40).

    .. versionadded:: 0.11.0

    Parameters
    ----------
    pixel_size : {55, 40}
        Flat-to-flat width (um) of the hexagonal pixel body, which also selects
        the device variant. May be given as a unitful quantity (e.g.
        ``55 * um``); see :py:mod:`pulse2percept.units`.
    x/y/z : double
        3D location (um) of the center of the electrode array.
        The coordinate system is centered over the fovea.
        Positive ``x`` values move the electrode into the nasal retina.
        Positive ``y`` values move the electrode into the superior retina.
        Positive ``z`` values move the electrode away from the retina into the
        vitreous humor (sometimes called electrode-retina distance).
        ``z`` can either be a list with one entry per pixel (250 for F55, 502
        for F40) or a scalar that is applied to all electrodes.
        May be given as unitful quantities (e.g. ``z=100 * um``); see
        :py:mod:`pulse2percept.units`.
    rot : float or Quantity, optional
        Rotation angle of the array (deg). Positive values denote
        counter-clock-wise (CCW) rotations in the retinal coordinate
        system.
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
    *  These are experimental arrays, not clinical PRIMA implants. [Ho2019]_
       describes four devices: the flat F55/F40 modeled here and the pillar
       arrays Pil55/Pil40, which pulse2percept does not model.
    *  The later 1.5 mm vertical-junction devices, which also come in 55 um
       and 40 um pixel sizes, are a different family; see
       :py:class:`~pulse2percept.implants.Huang2021Array`.
    *  [Ho2019]_ calls the row spacing the "pixel pitch". The nearest-neighbor
       center spacing, which is what ``spacing`` means here, is the pixel size.
    *  The 1 um isolation trenches between pixels are covered by the shared
       return electrode and are not open gaps, so the pixel bodies are drawn
       the full pixel width.
    *  The F55 outline is reconstructed from Fig. 2(a) of [Ho2019]_ and stored
       as ``_HO2019_F55_AXIAL_SPANS``: the range of rows carrying a pixel in
       each of the 19 columns. Where that outline sits on the die is not
       published, so it is centered on the requested position.
    *  [Ho2019]_ publishes the F40 pixel count and substrate diameter but not
       its outline, so the 502 pixels are the lattice sites nearest the center
       of the substrate, and their bounding box therefore sits a quarter of a
       spacing below it. That is an approximation, not the fabrication mask.

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
        # The substrate is centered on the requested position, which is not
        # the same point as the center of the pixels that survive trimming:
        self._substrate_center = (as_value(x, um, 'x'), as_value(y, um, 'y'))

        # A per-electrode `z` is one entry per surviving pixel, not one per
        # site of the untrimmed grid, so it is written on after trimming:
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
        """Plot the implant on its 1 mm substrate

        Takes the same arguments as
        :py:meth:`~pulse2percept.implants.ProsthesisSystem.plot`, and draws
        the substrate behind the pixels. Pixels at the rim are clipped to
        the substrate outline, since the die cuts through them.

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
        """Distance (um) between adjacent rows of pixels

        Derived geometry: ``spacing * sqrt(3) / 2``. Papers on this device
        family call this the "pixel pitch"; it is not the nearest-neighbor
        center spacing.
        """
        return self.spacing * np.sqrt(3) / 2


class Huang2021Array(ProsthesisSystem):
    """Create a vertical-junction photovoltaic array of [Huang2021]_

    The 1.5 mm subretinal arrays of [Huang2021]_, placed in the subretinal
    space such that the center of the array is located at 3D location (x,y,z),
    given in microns, and rotated by rotation angle ``rot``, given in degrees.

    Four variants are available, all on a 1.5 mm circular substrate. Only the
    exposed (stimulating) pixels are modeled as electrodes; the peripheral
    pixels covered by the common return electrode are not:

    ===================  ================  =================  ================
    ``pixel_size`` (um)  ``n_electrodes``  fabricated pixels  active electrode
    ===================  ================  =================  ================
    55                   421               526                22 um diameter
    40                   821               1027               16 um diameter
    30                   1388              1735               12 um diameter
    20                   2806              3508               8 um diameter
    ===================  ================  =================  ================

    Pixel bodies tile the lattice without an open gap, so the flat-to-flat
    pixel width equals the nearest-neighbor center spacing. The active
    electrode at the center of each pixel is 40% of the pixel size across.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    pixel_size : {55, 40, 30, 20}
        Flat-to-flat width (um) of the hexagonal pixel body, which also selects
        the device variant. May be given as a unitful quantity (e.g.
        ``40 * um``); see :py:mod:`pulse2percept.units`.
    x/y/z : double
        3D location (um) of the center of the electrode array.
        The coordinate system is centered over the fovea.
        Positive ``x`` values move the electrode into the nasal retina.
        Positive ``y`` values move the electrode into the superior retina.
        Positive ``z`` values move the electrode away from the retina into the
        vitreous humor (sometimes called electrode-retina distance).
        ``z`` can either be a list with one entry per exposed pixel (see
        ``n_electrodes``, not ``n_total_pixels``) or a scalar that is applied
        to all electrodes.
        May be given as unitful quantities (e.g. ``z=100 * um``); see
        :py:mod:`pulse2percept.units`.
    rot : float or Quantity, optional
        Rotation angle of the array (deg). Positive values denote
        counter-clock-wise (CCW) rotations in the retinal coordinate
        system.
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
    *  These are the later 1.5 mm vertical-junction devices. The 55 um and
       40 um flat arrays of [Ho2019]_ are a different family, on a 1 mm
       substrate; see :py:class:`~pulse2percept.implants.Ho2019FlatArray`.
    *  ``n_electrodes`` counts the exposed pixels, the ones that stimulate.
       ``n_total_pixels`` counts every pixel fabricated on the die. The
       difference is the peripheral ring covered by the common return
       electrode: those pixels are not exposed as independently stimulating
       pixels, so they are not individually addressable electrodes here.
    *  The fabrication and isolation trenches between pixels are not open
       gaps, so ``gap`` is 0 and the pixel bodies are drawn the full pixel
       width.
    *  The exposed-pixel outlines are reconstructed from Fig. 7 of
       [Huang2021]_, registered to the triangular lattice, and constrained to
       reproduce the published exposed-pixel counts. Where an outline sits on
       the die is determined far less reliably than which pixels it contains,
       so each mask is centered on the requested position.
    *  Three of the 2806 exposed pixels of the 20 um variant fall outside the
       published quadrant and are inferred; only one of the three is
       meaningfully ambiguous. See ``_HUANG2021_AXIAL_SPANS``.
    *  The small flat/notch visible on some photographed dies is not modeled:
       a nominal 1.5 mm disc is used for the substrate.

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
        # 40% of the pixel size across, hence a fifth of it as a radius:
        elec_radius = self.pixel_size * 0.2
        # Just large enough to hold the reconstructed exposed-pixel mask:
        self.shape = _axial_mask_shape(spans)
        self.eye = eye
        self.preprocess = preprocess
        self.safe_mode = safe_mode

        # Normalized here rather than in ElectrodeGrid, because a
        # per-electrode list of heights never reaches the grid at all -- it is
        # written onto the electrodes further down:
        z = as_value(z, um, 'z')
        # The substrate is centered on the requested position, which is not
        # the same point as the center of the pixels that survive trimming:
        self._substrate_center = (as_value(x, um, 'x'), as_value(y, um, 'y'))

        # A per-electrode `z` is one entry per surviving pixel, not one per
        # site of the untrimmed grid, so it is written on after trimming:
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
        """Plot the implant on its 1.5 mm substrate

        Takes the same arguments as
        :py:meth:`~pulse2percept.implants.ProsthesisSystem.plot`, and draws
        the substrate behind the pixels. Pixels at the rim are clipped to
        the substrate outline, since the die cuts through them.

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
        """Distance (um) between adjacent rows of pixels

        Derived geometry: ``spacing * sqrt(3) / 2``. Papers on this device
        family call this the "pixel pitch"; it is not the nearest-neighbor
        center spacing.
        """
        return self.spacing * np.sqrt(3) / 2


@deprecated(alt_func='Ho2019FlatArray(55)', deprecated_version='0.11.0',
            removed_version='0.12.0',
            extra_msg='The name is ambiguous: 55 um arrays appear in two '
                      'device families. This one is the flat F55 array of '
                      'Ho et al. (2019); the 1.5 mm vertical-junction array '
                      'of Huang et al. (2021) is ``Huang2021Array(55)``.')
class PRIMA55(Ho2019FlatArray):
    """Create a PRIMA-55 array on the retina

    .. deprecated:: 0.11.0

        Use :py:class:`~pulse2percept.implants.Ho2019FlatArray` with
        ``pixel_size=55`` instead: the name is ambiguous, because
        [Huang2021]_ describes a different 55 um array. This wrapper builds
        the corrected [Ho2019]_ geometry, which already differs from the
        pre-0.11 ``PRIMA55``; see the v0.11.0 release notes.

    Takes the same arguments as
    :py:class:`~pulse2percept.implants.Ho2019FlatArray`, minus ``pixel_size``.
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
    """Create a PRIMA-40 array on the retina

    .. deprecated:: 0.11.0

        Use :py:class:`~pulse2percept.implants.Ho2019FlatArray` with
        ``pixel_size=40`` instead: the name is ambiguous, because
        [Huang2021]_ describes a different 40 um array. This wrapper builds
        the corrected [Ho2019]_ geometry, which already differs from the
        pre-0.11 ``PRIMA40``; see the v0.11.0 release notes.

    Takes the same arguments as
    :py:class:`~pulse2percept.implants.Ho2019FlatArray`, minus ``pixel_size``.
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
    """Create a PRIMA-100 array on the retina

    .. deprecated:: 0.11.0

        Use :py:class:`~pulse2percept.implants.PRIMAPivotal` instead. The
        geometry is unchanged; the qualified name identifies the one published
        hardware configuration this class models, and leaves ``PRIMA`` free
        for the eventual commercial device.

    Takes the same arguments as
    :py:class:`~pulse2percept.implants.PRIMAPivotal`.
    """
    __slots__ = ()


@deprecated(alt_func='Lorach2015Array', deprecated_version='0.11.0',
            removed_version='0.12.0',
            extra_msg='``PRIMA75`` was pulse2percept shorthand, not an '
                      'official device name.')
class PRIMA75(Lorach2015Array):
    """Create a PRIMA-75 array on the retina

    .. deprecated:: 0.11.0

        Use :py:class:`~pulse2percept.implants.Lorach2015Array` instead. The
        geometry is unchanged; ``PRIMA75`` was pulse2percept shorthand for the
        70 um array of [Lorach2015]_ rather than an official device name.

    Takes the same arguments as
    :py:class:`~pulse2percept.implants.Lorach2015Array`.
    """
    __slots__ = ()
