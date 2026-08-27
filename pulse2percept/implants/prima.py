""":py:class:`~pulse2percept.implants.PhotovoltaicPixel`, 
   :py:class:`~pulse2percept.implants.PRIMA`, 
   :py:class:`~pulse2percept.implants.PRIMA75`, 
   :py:class:`~pulse2percept.implants.PRIMA55`, 
   :py:class:`~pulse2percept.implants.PRIMA40`"""

from matplotlib.patches import Circle, RegularPolygon

import numpy as np
from collections.abc import Sequence

from .base import ProsthesisSystem
from .electrodes import HexElectrode
from .electrode_arrays import ElectrodeGrid
from ..units import as_value, um


def _trim_to_disc(earray, n_pixels, x, y, rot):
    """Trim a hex grid down to the ``n_pixels`` pixels nearest its center

    Photovoltaic arrays are diced from a round substrate, so the pixels that
    survive are the lattice sites closest to its center. Only the pixel count
    is published, not the exact boundary, so the outline this produces is the
    most circular one with that count.

    Ties -- a boundary shell that only partly fits -- are broken so that
    antipodal sites are kept together, leaving the array symmetric under a
    half turn about ``(x, y)``. Angles are measured in the unrotated device
    frame, so ``rot`` turns the device without changing which pixels it has.
    """
    xy = earray.coordinates()[:, :2] - np.array([x, y], dtype=float)
    th = -np.radians(rot)
    rotm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    # Rounded, so that sites that are the same distance out are recognized as
    # one shell despite the rotation's round-off:
    xy = np.round(xy @ rotm.T, 6)
    r = np.round(np.hypot(*xy.T), 6)
    # A site and its antipode map onto the same representative, so they sort
    # adjacently and a partly-kept shell stays half-turn symmetric:
    flip = np.where((xy[:, 1] > 0) | ((xy[:, 1] == 0) & (xy[:, 0] >= 0)),
                    1.0, -1.0)
    rep = xy * flip[:, np.newaxis]
    ang = np.round(np.arctan2(rep[:, 1], rep[:, 0]), 6)
    order = np.lexsort((flip, ang, r))
    names = np.asarray(list(earray.electrodes))
    for name in names[order[n_pixels:]]:
        earray.remove_electrode(name)


class PhotovoltaicPixel(HexElectrode):
    """Photovoltaic pixel

    A hexagonal pixel body with a circular active electrode at its center, as
    used by the PRIMA family of subretinal photovoltaic arrays.

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


class PRIMA(ProsthesisSystem):
    """Create a PRIMA-100 array on the retina

    This class creates a PRIMA array with 378 photovoltaic pixels as used in
    the clinical trial [Palanker2020]_, and places it in the subretinal space
    such that the center of the array is located at 3D location (x,y,z), given
    in microns, and the array is rotated by rotation angle ``rot``, given in
    degrees.

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
    __slots__ = ('shape', 'spacing', 'pixel_width', 'gap')

    placement = 'subretinal'

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

    @property
    def row_spacing(self):
        """Distance (um) between adjacent rows of pixels

        Derived geometry: ``spacing * sqrt(3) / 2``. Papers on this device
        family call this the "pixel pitch"; it is not the nearest-neighbor
        center spacing.
        """
        return self.spacing * np.sqrt(3) / 2


class PRIMA75(ProsthesisSystem):
    """Create a PRIMA-75 array on the retina

    This class creates a PRIMA array with 142 photovoltaic pixels as described
    in [Lorach2015]_, and places it in the subretinal space, such that that the
    center of the array is located at 3D location (x,y,z), given in microns,
    and the array is rotated by rotation angle ``rot``, given in degrees.

    Each hexagonal pixel is 70 um wide (flat-to-flat) and neighboring pixel
    centers are 75 um apart, leaving a 5 um open trench between pixel bodies,
    on a nominally 1 mm substrate. Adjacent rows are therefore separated by
    ``75 * sqrt(3) / 2`` = 65 um. The active electrode at the center of each
    pixel is a disk 20 um in diameter.

    .. versionadded:: 0.7

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

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape', 'spacing', 'pixel_width', 'gap')

    placement = 'subretinal'

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

    @property
    def row_spacing(self):
        """Distance (um) between adjacent rows of pixels

        Derived geometry: ``spacing * sqrt(3) / 2``. Papers on this device
        family call this the "pixel pitch"; it is not the nearest-neighbor
        center spacing.
        """
        return self.spacing * np.sqrt(3) / 2


class PRIMA55(ProsthesisSystem):
    """Create a PRIMA-55 array on the retina

    This class creates the 250 photovoltaic pixels of the experimental F55
    array of [Ho2019]_, and places it in the subretinal space, such that the
    center of the array is located at 3D location (x,y,z), given in microns,
    and the array is rotated by rotation angle ``rot``, given in degrees.

    Each hexagonal pixel is 55 um wide (flat-to-flat) and neighboring
    pixel centers are 55 um apart, on a 1 mm circular substrate: the pixel
    bodies tile the array without an open gap between them. Adjacent rows are
    therefore separated by ``55 * sqrt(3) / 2`` = 48 um. The active
    electrode at the center of each pixel is a disk 14 um in diameter.

    .. versionadded:: 0.7

    .. versionchanged:: 0.11.0
        Models the F55 array of [Ho2019]_: 250 pixels (was: 273),
        55 um wide with no open inter-pixel gap (was: 50 um pixels
        separated by 5 um trenches), and a 14 um active electrode (was:
        16 um).

    Parameters
    ----------
    x/y/z : double
        3D location (um) of the center of the electrode array.
        The coordinate system is centered over the fovea.
        Positive ``x`` values move the electrode into the nasal retina.
        Positive ``y`` values move the electrode into the superior retina.
        Positive ``z`` values move the electrode away from the retina into the
        vitreous humor (sometimes called electrode-retina distance).
        ``z`` can either be a list with 250 entries or a scalar that is applied
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
    *  The class name is kept for backwards compatibility; the device it
       models is the experimental F55 array of [Ho2019]_, not a clinical
       PRIMA implant.
    *  [Ho2019]_ calls the 48 um row spacing the "pixel pitch". The
       nearest-neighbor center spacing, which is what ``spacing`` means here,
       is 55 um.
    *  The 1 um isolation trenches between pixels are covered by the shared
       return electrode and are not open gaps, so the pixel bodies are drawn
       the full 55 um wide.
    *  [Ho2019]_ publishes the pixel count and the substrate diameter but not
       the exact outline, so the 250 pixels are the lattice sites closest to
       the center of the substrate.

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape', 'spacing', 'pixel_width', 'gap')

    placement = 'subretinal'

    def __init__(self, x=0, y=0, z=-100, rot=0, eye='RE', preprocess=False,
                 safe_mode=False):
        self.spacing = 55  # um, nearest-neighbor center-to-center
        self.pixel_width = 55  # um, flat-to-flat
        self.gap = self.spacing - self.pixel_width  # um, open inter-pixel gap
        elec_radius = 7  # um
        # Large enough to cover the substrate; trimmed to a disc below:
        self.shape = (24, 24)
        self.eye = eye
        self.preprocess = preprocess
        self.safe_mode = safe_mode

        # Normalized here rather than in ElectrodeGrid, because a
        # per-electrode list of heights never reaches the grid at all -- it is
        # written onto the electrodes further down:
        z = as_value(z, um, 'z')

        # A per-electrode `z` is one entry per surviving pixel, not one per
        # site of the untrimmed grid, so it is written on after trimming:
        overwrite_z = isinstance(z, (list, np.ndarray))
        zarr = -100 if overwrite_z else z

        self.earray = ElectrodeGrid(self.shape, self.spacing, x=x, y=y,
                                    z=zarr, rot=rot, type='hex',
                                    orientation='vertical',
                                    etype=PhotovoltaicPixel, r=elec_radius,
                                    a=self.pixel_width / 2)
        _trim_to_disc(self.earray, 250, x, y, rot)

        if overwrite_z:
            z_arr = np.asarray(z).flatten()
            if z_arr.size != self.n_electrodes:
                raise ValueError(f"If `z` is a list, it must have "
                                 f"{self.n_electrodes} entries, not "
                                 f"{z_arr.size}.")
            for elec, z_elec in zip(self.earray.electrode_objects, z):
                elec.z = z_elec

    @property
    def row_spacing(self):
        """Distance (um) between adjacent rows of pixels

        Derived geometry: ``spacing * sqrt(3) / 2``. Papers on this device
        family call this the "pixel pitch"; it is not the nearest-neighbor
        center spacing.
        """
        return self.spacing * np.sqrt(3) / 2


class PRIMA40(ProsthesisSystem):
    """Create a PRIMA-40 array on the retina

    This class creates the 502 photovoltaic pixels of the experimental F40
    array of [Ho2019]_, and places it in the subretinal space, such that the
    center of the array is located at 3D location (x,y,z), given in microns,
    and the array is rotated by rotation angle ``rot``, given in degrees.

    Each hexagonal pixel is 40 um wide (flat-to-flat) and neighboring
    pixel centers are 40 um apart, on a 1 mm circular substrate: the pixel
    bodies tile the array without an open gap between them. Adjacent rows are
    therefore separated by ``40 * sqrt(3) / 2`` = 35 um. The active
    electrode at the center of each pixel is a disk 10 um in diameter.

    .. versionadded:: 0.7

    .. versionchanged:: 0.11.0
        Models the F40 array of [Ho2019]_: 502 pixels (was: 532),
        40 um wide with no open inter-pixel gap (was: 35 um pixels
        separated by 5 um trenches), and a 10 um active electrode (was:
        16 um).

    Parameters
    ----------
    x/y/z : double
        3D location (um) of the center of the electrode array.
        The coordinate system is centered over the fovea.
        Positive ``x`` values move the electrode into the nasal retina.
        Positive ``y`` values move the electrode into the superior retina.
        Positive ``z`` values move the electrode away from the retina into the
        vitreous humor (sometimes called electrode-retina distance).
        ``z`` can either be a list with 502 entries or a scalar that is applied
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
    *  The class name is kept for backwards compatibility; the device it
       models is the experimental F40 array of [Ho2019]_, not a clinical
       PRIMA implant.
    *  [Ho2019]_ calls the 35 um row spacing the "pixel pitch". The
       nearest-neighbor center spacing, which is what ``spacing`` means here,
       is 40 um.
    *  The 1 um isolation trenches between pixels are covered by the shared
       return electrode and are not open gaps, so the pixel bodies are drawn
       the full 40 um wide.
    *  [Ho2019]_ publishes the pixel count and the substrate diameter but not
       the exact outline, so the 502 pixels are the lattice sites closest to
       the center of the substrate.

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('shape', 'spacing', 'pixel_width', 'gap')

    placement = 'subretinal'

    def __init__(self, x=0, y=0, z=-100, rot=0, eye='RE', preprocess=False,
                 safe_mode=False):
        self.spacing = 40  # um, nearest-neighbor center-to-center
        self.pixel_width = 40  # um, flat-to-flat
        self.gap = self.spacing - self.pixel_width  # um, open inter-pixel gap
        elec_radius = 5  # um
        # Large enough to cover the substrate; trimmed to a disc below:
        self.shape = (32, 32)
        self.eye = eye
        self.preprocess = preprocess
        self.safe_mode = safe_mode

        # Normalized here rather than in ElectrodeGrid, because a
        # per-electrode list of heights never reaches the grid at all -- it is
        # written onto the electrodes further down:
        z = as_value(z, um, 'z')

        # A per-electrode `z` is one entry per surviving pixel, not one per
        # site of the untrimmed grid, so it is written on after trimming:
        overwrite_z = isinstance(z, (list, np.ndarray))
        zarr = -100 if overwrite_z else z

        self.earray = ElectrodeGrid(self.shape, self.spacing, x=x, y=y,
                                    z=zarr, rot=rot, type='hex',
                                    orientation='vertical',
                                    etype=PhotovoltaicPixel, r=elec_radius,
                                    a=self.pixel_width / 2)
        _trim_to_disc(self.earray, 502, x, y, rot)

        if overwrite_z:
            z_arr = np.asarray(z).flatten()
            if z_arr.size != self.n_electrodes:
                raise ValueError(f"If `z` is a list, it must have "
                                 f"{self.n_electrodes} entries, not "
                                 f"{z_arr.size}.")
            for elec, z_elec in zip(self.earray.electrode_objects, z):
                elec.z = z_elec

    @property
    def row_spacing(self):
        """Distance (um) between adjacent rows of pixels

        Derived geometry: ``spacing * sqrt(3) / 2``. Papers on this device
        family call this the "pixel pitch"; it is not the nearest-neighbor
        center spacing.
        """
        return self.spacing * np.sqrt(3) / 2
