""":py:class:`~pulse2percept.topography.retina.Montesano2020Map`"""
from collections import namedtuple
from functools import lru_cache
from importlib import resources

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .watson2014 import Watson2014Map
from ...utils.geometry import cart2pol, pol2cart

#: Radial support of the packaged field (dva); identity at and beyond it.
SUPPORT_MAX_DVA = 15.0

_DATA_FILE = 'montesano2020.npz'

#: The two interpolators, plus the last radial node of the stored grid, which
#: sits a few float ulps below ``SUPPORT_MAX_DVA``.
_Field = namedtuple('_Field', ['forward', 'inverse', 'radial_max'])


def _as_scalars(x, y):
    """Return 0-d results as NumPy scalars."""
    return (x[()], y[()]) if np.ndim(x) == 0 else (x, y)


def _check_layout(angle_deg, r_rf_dva, r_soma_dva):
    """Catch a swapped, truncated or reordered archive.

    Scientific validation lives in ``tools/generate_montesano2020_map.py``.
    """
    if r_soma_dva.shape != (angle_deg.size, r_rf_dva.size):
        raise ValueError(f"{_DATA_FILE}: soma_eccentricity_dva has shape "
                         f"{r_soma_dva.shape}, expected "
                         f"{(angle_deg.size, r_rf_dva.size)}.")
    if (angle_deg[0] != 0.0 or angle_deg[-1] >= 360.0 or
            np.any(np.diff(angle_deg) <= 0.0)):
        raise ValueError(f"{_DATA_FILE}: retinal_angle_deg must increase from "
                         f"0 and stay below 360.")
    if (r_rf_dva[0] != 0.0 or np.any(np.diff(r_rf_dva) <= 0.0) or
            not np.isclose(r_rf_dva[-1], SUPPORT_MAX_DVA)):
        raise ValueError(f"{_DATA_FILE}: rf_eccentricity_dva must increase "
                         f"from 0 to {SUPPORT_MAX_DVA} dva.")


def _periodic_interpolator(angle_deg, radius_dva, table):
    """Bilinear interpolator over angle (deg) and radius (dva).

    A 360 deg row identical to the 0 deg one makes the angle axis periodic, so
    a query at 359.9 deg interpolates across the seam instead of clamping.
    """
    angle = np.append(angle_deg, 360.0)
    return RegularGridInterpolator((angle, radius_dva),
                                   np.vstack([table, table[:1]]),
                                   method='linear', bounds_error=True)


@lru_cache(maxsize=1)
def _displacement_field():
    """Read the packaged field; build ``F(theta, r_rf)`` and its inverse.

    Cached, so the 3.7 MB archive is read on first use, not at import.
    """
    path = resources.files(__package__).joinpath('data', _DATA_FILE)
    with path.open('rb') as f:
        with np.load(f, allow_pickle=False) as npz:
            angle = np.asarray(npz['retinal_angle_deg'], dtype=np.float64)
            r_rf = np.asarray(npz['rf_eccentricity_dva'], dtype=np.float64)
            r_soma = np.asarray(npz['soma_eccentricity_dva'],
                                dtype=np.float64)
    _check_layout(angle, r_rf, r_soma)
    # Every forward row is strictly increasing (checked by the generator), so
    # `np.interp` inverts it onto the same radial grid. Deriving the inverse
    # here avoids packaging a second field.
    r_rf_of_soma = np.array([np.interp(r_rf, row, r_rf) for row in r_soma])
    return _Field(_periodic_interpolator(angle, r_rf, r_soma),
                  _periodic_interpolator(angle, r_rf, r_rf_of_soma),
                  float(r_rf[-1]))


class Montesano2020Map(Watson2014Map):
    """Converts between visual angle and retinal eccentricity using
    two-dimensional RGC displacement [Montesano2020]_

    Retinal ganglion cell (RGC) bodies sit displaced centrifugally from the
    receptive fields (RFs) they serve, by an amount that depends on meridian
    as well as eccentricity. This map displaces RF to soma in degrees of
    visual angle (dva) using the [Montesano2020]_ field, reconstructed from
    that paper's equations and the [Curcio1990]_ ganglion-cell topography,
    then converts the displaced location to microns with
    :py:class:`Watson2014Map`.

    Displacement is radial, so the visual-field polar angle is preserved and
    the anatomical meridian only selects the profile applied. The zone reaches
    14.10 dva temporally and superiorly, 10.52 dva inferiorly and 9.54 dva
    nasally; beyond 15 dva the map is the identity. Both directions are
    implemented, so the map works with ``location_noise``.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    eye : {'right', 'left'}, optional
        Which side of the visual field is nasal retina: positive x in a right
        eye, negative x in a left one. Positive y is inferior retina in both.
        Case-insensitive on input; stored lowercase. Defaults to ``'right'``.

    Notes
    -----
    *  An independent reconstruction of [Montesano2020]_, validated against
       that paper's published figures and E2v fits rather than ported from
       author code. See ``tools/generate_montesano2020_map.py``.
    *  Population reference anatomy ([Curcio1990]_, six donor retinas, plus a
       schematic eye), not subject-specific.
    *  Tabulated on 1440 meridians by 751 radial nodes and interpolated
       linearly, costing about 0.003 dva against the directly solved model.

    """

    @property
    def eye(self):
        """Eye whose nasal/temporal anatomy is used"""
        return self._eye

    @eye.setter
    def eye(self, eye):
        """Eye setter (called upon `self.eye = eye`)"""
        if not isinstance(eye, str):
            raise TypeError(f"'eye' must be a string, not {type(eye)}.")
        eye = eye.lower()
        if eye not in ('left', 'right'):
            raise ValueError(f"'eye' must be either 'left' or 'right', not "
                             f"{eye}.")
        self._eye = eye

    def get_default_params(self):
        return {**super().get_default_params(), 'eye': 'right'}

    def _retinal_angle(self, theta_visual):
        """Anatomical retinal angle (deg) of a visual polar angle (rad).

        0 nasal, 90 superior, 180 temporal, 270 inferior. Both eyes flip the
        vertical axis; only the horizontal one flips between them.
        """
        theta_deg = np.degrees(theta_visual)
        if self.eye == 'left':
            return np.mod(180.0 + theta_deg, 360.0)
        return np.mod(-theta_deg, 360.0)

    def _remap_radius(self, theta_visual, radius_dva, inverse=False):
        """Map a radius (dva) between RF and soma space along one meridian.

        Radii at or beyond ``SUPPORT_MAX_DVA`` pass through unchanged; nothing
        is extrapolated past the modeled support. Interpolation runs in double
        precision but returns the caller's dtype, since a model grid is
        float32 and the spatial kernels require it back.
        """
        given = np.asarray(radius_dva)
        theta_ret = np.ravel(self._retinal_angle(theta_visual))
        radius = np.ravel(given.astype(np.float64, copy=True))
        inside = radius < SUPPORT_MAX_DVA
        if inside.any():
            field = _displacement_field()
            interp = field.inverse if inverse else field.forward
            # The stored grid ends a few ulps short of 15 dva, so clip rather
            # than let a radius inside the documented support go out of bounds.
            query = np.minimum(radius[inside], field.radial_max)
            radius[inside] = interp(np.column_stack([theta_ret[inside],
                                                     query]))
        return radius.reshape(given.shape).astype(given.dtype, copy=False)

    def dva_to_ret(self, xdva, ydva):
        """Converts dva to retinal coords

        Displaces RF to soma in the visual field, then converts to microns.

        Parameters
        ----------
        xdva, ydva : double or array-like
            x,y coordinates in dva

        Returns
        -------
        xret, yret : double or array-like
            Corresponding x,y coordinates in microns
        """
        theta, r_rf = cart2pol(np.asarray(xdva), np.asarray(ydva))
        r_soma = self._remap_radius(theta, r_rf)
        # Radial: rebuild from the visual polar angle, not the anatomical one.
        return _as_scalars(*super().dva_to_ret(*pol2cart(theta, r_soma)))

    def ret_to_dva(self, xret, yret):
        """Converts retinal coords to dva

        Inverse of :py:meth:`dva_to_ret`: microns become a soma location in
        dva, which is mapped back to the RF it serves.

        Parameters
        ----------
        xret, yret : double or array-like
            x,y coordinates in microns

        Returns
        -------
        xdva, ydva : double or array-like
            Corresponding x,y coordinates in dva
        """
        theta, r_soma = super().ret_to_dva(xret, yret, coords='polar')
        r_rf = self._remap_radius(theta, r_soma, inverse=True)
        return _as_scalars(*pol2cart(theta, r_rf))
