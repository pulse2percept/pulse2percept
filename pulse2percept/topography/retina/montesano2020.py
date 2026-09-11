""":py:class:`~pulse2percept.topography.retina.Montesano2020Map`"""
from collections import namedtuple
from functools import lru_cache
from importlib import resources

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .watson2014 import Watson2014Map
from ...units import Quantity, mm, um
from ...utils.geometry import cart2pol, pol2cart

#: Radial support of the packaged field (dva); identity at and beyond it.
SUPPORT_MAX_DVA = 15.0

_DATA_FILE = 'montesano2020.npz'

#: The two interpolators, plus the last radial node of the stored grid,
#: which is a few float ulps below ``SUPPORT_MAX_DVA``.
_Field = namedtuple('_Field', ['forward', 'inverse', 'radial_max'])


#: Eq. A5 of [Watson2014]_ (dva -> mm), as (linear, square, cubic) terms.
_A5_COEFF = (0.268, 3.427e-4, -8.3309e-6)


def _invert_a5(r_mm, seed_deg):
    """Solve Eq. A5 of [Watson2014]_ for eccentricity (dva).

    [Watson2014]_ fits Eq. A6 for the reverse direction separately, and the
    two fits disagree by a few percent. Inverting Eq. A5 instead keeps
    :py:meth:`Montesano2020Map.ret_to_dva` the inverse of
    :py:meth:`Montesano2020Map.dva_to_ret`, which ``location_noise`` requires.
    """
    a, b, c = _A5_COEFF
    r_deg = np.asarray(seed_deg, dtype=np.float64)
    for _ in range(3):
        residual = ((c * r_deg + b) * r_deg + a) * r_deg - r_mm
        slope = (3.0 * c * r_deg + 2.0 * b) * r_deg + a
        r_deg = r_deg - residual / slope
    return r_deg


def _as_scalars(x, y):
    """Return 0-d results as NumPy scalars."""
    return (x[()], y[()]) if np.ndim(x) == 0 else (x, y)


def _check_layout(angle_deg, r_rf_dva, r_soma_dva):
    """Catch a swapped, truncated or reordered archive.

    Scientific validation is in ``tools/generate_montesano2020_map.py``.
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

    Appending a 360 deg copy of the 0 deg row makes the angle axis periodic:
    a query at 359.9 deg interpolates across the seam instead of clamping.
    """
    angle = np.append(angle_deg, 360.0)
    return RegularGridInterpolator((angle, radius_dva),
                                   np.vstack([table, table[:1]]),
                                   method='linear', bounds_error=True)


@lru_cache(maxsize=1)
def _displacement_field():
    """Read the packaged field; build ``F(theta, r_rf)`` and its inverse.

    Cached: the 3.7 MB archive is read on first use, not at import.
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
    # `np.interp` inverts it onto the same radial grid.
    r_rf_of_soma = np.array([np.interp(r_rf, row, r_rf) for row in r_soma])
    return _Field(_periodic_interpolator(angle, r_rf, r_soma),
                  _periodic_interpolator(angle, r_rf, r_rf_of_soma),
                  float(r_rf[-1]))


class Montesano2020Map(Watson2014Map):
    """Converts between visual angle and retinal eccentricity using
    two-dimensional RGC displacement [Montesano2020]_

    Retinal ganglion cell (RGC) bodies are displaced centrifugally from the
    receptive fields (RFs) they serve. The displacement depends on meridian
    as well as eccentricity. This map applies the [Montesano2020]_ field in
    degrees of visual angle (dva), then converts the displaced location to
    microns with :py:class:`Watson2014Map`.

    Displacement is radial: the visual-field polar angle is preserved. The
    anatomical meridian only selects which profile applies. The zone
    reaches 14.10 dva temporally and superiorly, 10.52 dva inferiorly,
    9.54 dva nasally. Beyond 15 dva the map is the identity.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    eye : {'right', 'left'}, optional
        Retinal laterality. Nasal retina is the positive-x visual field in a
        right eye, the negative-x one in a left eye; positive y is inferior
        retina in both. Case-insensitive on input; stored lowercase. Defaults
        to ``'right'``.

    Notes
    -----
    *  An independent reconstruction of [Montesano2020]_ as opposed to a port
       of the original code. The implementation was validated against that
       paper's published figures and E2v fits, and cross-checked against the
       ``visualFields`` implementation by the same group, which it matches to
       0.07 dva inside the displacement zone. 
       See ``tools/generate_montesano2020_map.py``.
    *  This is based on population anatomy [Curcio1990]_ and not to be
       interpreted as a subject-specific eye.

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
        precision. The result comes back in the caller's dtype, which the
        float32 model grids require.
        """
        given = np.asarray(radius_dva)
        theta_ret = np.ravel(self._retinal_angle(theta_visual))
        radius = np.ravel(given.astype(np.float64, copy=True))
        inside = radius < SUPPORT_MAX_DVA
        if inside.any():
            field = _displacement_field()
            interp = field.inverse if inverse else field.forward
            # The stored grid ends a few ulps short of 15 dva; clip so a
            # radius inside the documented support stays in bounds.
            query = np.minimum(radius[inside], field.radial_max)
            radius[inside] = interp(np.column_stack([theta_ret[inside],
                                                     query]))
        return radius.reshape(given.shape).astype(given.dtype, copy=False)

    def dva_to_ret(self, xdva, ydva, coords='cart'):
        """Converts dva to retinal coords

        Displaces RF to soma in the visual field, then converts to microns
        with Eq. A5 of [Watson2014]_.

        Parameters
        ----------
        xdva, ydva : double or array-like
            x,y coordinates in dva
        coords : {'cart', 'polar'}
            Whether to return the result in Cartesian or polar coordinates

        Returns
        -------
        xret, yret : double or array-like
            Corresponding x,y coordinates in microns
        """
        theta, r_rf = cart2pol(np.asarray(xdva), np.asarray(ydva))
        r_soma = self._remap_radius(theta, r_rf)
        # Radial: rebuild from the visual polar angle, not the anatomical one.
        return _as_scalars(*super().dva_to_ret(*pol2cart(theta, r_soma),
                                               coords=coords))

    def ret_to_dva(self, xret, yret, coords='cart'):
        """Converts retinal coords to dva

        Inverts :py:meth:`dva_to_ret`. Eq. A5 of [Watson2014]_ is inverted
        numerically, so this does not reproduce
        :py:meth:`Watson2014Map.ret_to_dva`, which evaluates Eq. A6.

        Parameters
        ----------
        xret, yret : double or array-like
            x,y coordinates in microns
        coords : {'cart', 'polar'}
            Whether to return the result in Cartesian or polar coordinates

        Returns
        -------
        xdva, ydva : double or array-like
            Corresponding x,y coordinates in dva
        """
        # The parent supplies the flipped polar angle and the Eq. A6 value,
        # which seeds the Eq. A5 solve.
        theta, seed = super().ret_to_dva(xret, yret, coords='polar')
        r_um = np.hypot(np.asarray(xret, dtype=np.float64),
                        np.asarray(yret, dtype=np.float64))
        r_soma = _invert_a5(Quantity(r_um, um).to_value(mm), seed)
        r_soma = r_soma.astype(np.asarray(seed).dtype, copy=False)
        r_rf = self._remap_radius(theta, r_soma, inverse=True)
        if coords.lower() == 'cart':
            return _as_scalars(*pol2cart(theta, r_rf))
        elif coords.lower() == 'polar':
            return _as_scalars(theta, r_rf)
        raise ValueError(f'Unknown coordinate system "{coords}".')
