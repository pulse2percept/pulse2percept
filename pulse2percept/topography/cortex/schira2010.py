""":py:class:`~pulse2percept.topography.cortex.Schira2010Map`"""
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .base import CorticalMap
from ...units import dva, mm
from ...utils._plotting import set_mm_ticks

# Double-Sech shear constants (Schira et al. 2010, Eq. 6; Protocol S1). Fit
# for constant areal magnification with polar angle, nearly independent of a
# and b:
_ISO_POLAR_GRAD = 0.1821
_ECC_WIDTH = 0.7609

# Computational eccentricity domain (dva), as for Polimeni2006Map. The model
# was fit to data within ~12 dva:
_MAX_ECC = 90.0

# Inverse: accepted cortical residual (mm), and eccentricity (dva) below
# which a solution is the fovea:
_INVERSE_TOL_MM = 1e-3
_FOVEA_ECC = 1e-6


def _canonical(ecc, theta, upper, region, k, a, b, lambda_, alphas):
    """Returns cortical (x, y) in mm for the right visual hemifield.

    ``theta`` is the polar angle (rad) in [-pi/2, pi/2], positive upward.
    ``upper`` selects the V2/V3 quadrant; it differs from ``theta >= 0`` only
    on the horizontal meridian. The V1 foveal tip is the origin, and the
    upper visual field maps to positive y (Protocol S1 orientation).
    """
    alpha1, alpha2, alpha3 = alphas
    side = np.where(upper, 1.0, -1.0)
    # Step 1: wedge-dipole 'pacman' angle (assembleV1V3Complex.m):
    if region == 'v1':
        phi = alpha1 * theta
    elif region == 'v2':
        phi = -alpha2 * theta + side * np.pi / 2 * (alpha1 + alpha2)
    else:
        phi = alpha3 * theta + side * np.pi / 2 * (alpha1 + alpha2)
    # Step 2: banding, a shift (dva) toward +x that is full in V1 and tapers
    # linearly to 0 at |phi| = pi (bandedDoubleSech.m; Eqs. 8-10). The prose
    # says V1 shifts "to the left"; Eqs. 9-10 and Protocol S1 add +shift:
    shift = lambda_ * np.minimum(2 * (1 - np.abs(phi) / np.pi), 1)
    z = ecc * np.exp(1j * phi) + shift
    E, P = np.abs(z), np.angle(z)
    # Step 3: Double-Sech shear and dipole log (DoubleSech.m). Eq. 7 shears
    # numerator and denominator separately; Protocol S1 applies one shear
    # with the summed a- and b-exponents to both, as here. At E = 0,
    # log(E/a) = -inf and the shear exponent is 0:
    with np.errstate(divide='ignore'):
        exponent = _ISO_POLAR_GRAD * (1 / np.cosh(np.log(E / a) * _ECC_WIDTH) +
                                      1 / np.cosh(np.log(E / b) * _ECC_WIDTH))
    zs = E * np.exp(1j * P * (1 / np.cosh(P)) ** exponent)
    w = np.log((zs + a) / (zs + b)) - np.log((lambda_ + a) / (lambda_ + b))
    return k * w.real, k * w.imag


@lru_cache(maxsize=16)
def _seed_tree(region, params):
    """Returns a KD-tree of canonical cortex (mm), and its ecc/theta/upper"""
    ecc = np.concatenate(([0], np.geomspace(1e-3, _MAX_ECC, 240)))
    theta = np.linspace(-np.pi / 2, np.pi / 2, 181)
    ecc, theta = [g.ravel() for g in np.meshgrid(ecc, theta)]
    # Both V2/V3 copies of the horizontal meridian:
    ecc, theta = np.tile(ecc, 2), np.tile(theta, 2)
    upper = np.repeat([True, False], ecc.size // 2)
    keep = (upper & (theta >= 0)) | (~upper & (theta <= 0))
    ecc, theta, upper = ecc[keep], theta[keep], upper[keep]
    x, y = _canonical(ecc, theta, upper, region, *params)
    return cKDTree(np.column_stack((x, y))), ecc, theta, upper


class Schira2010Map(CorticalMap):
    """Banded Double-Sech map of V1-V3 [Schira2010]_

    A planar, population-average analytic map of the V1/V2/V3 complex.
    Compared with the wedge-dipole
    :py:class:`~pulse2percept.topography.cortex.Polimeni2006Map`, it keeps
    areal magnification constant across polar angle, and V2/V3 form bands
    around the V1 foveal tip rather than converging on one point (the
    foveal confluence). It remains a 2D model, not an anatomical cortical
    surface; for subject-specific 3D surfaces, see
    :py:class:`~pulse2percept.topography.cortex.NeuropythyMap`.

    The forward map follows the authors' MATLAB code (Protocol S1 of
    [Schira2010]_). The inverse is numerical (KD-tree seed refined by least
    squares) and returns NaN where no visual-field point maps within 1 um.

    .. versionadded:: 0.12.0

    Parameters
    ----------
    k : float, optional
        Cortical scale (mm). [Schira2010]_ report 15-26 across subjects;
        the default follows Protocol S1.
    a, b : float, optional
        Foveal and peripheral dipole eccentricities (dva). Defaults are the
        values [Schira2010]_ recommend for most subjects; the Protocol S1
        demo uses a = 0.75.
    lambda_ : float, optional
        Banding shift (dva) applied before the log transform; the default
        is the value fitted in [Schira2010]_. 0 removes the V2/V3 foveal
        bands but keeps the Double-Sech shear.
    alpha1, alpha2, alpha3 : float, optional
        Angular compression of V1, V2, V3; also their relative areas.
        Defaults follow Protocol S1.
    regions : list of str, optional
        Any of 'v1', 'v2', 'v3'.
    left_offset : float, optional
        x offset (um) of the left hemisphere's V1 foveal tip.

    Notes
    -----
    *  Inputs up to 90 dva eccentricity are mapped; beyond that, outputs are
       NaN. The model was fit to fMRI data within ~12 dva and is not
       validated across that whole range.
    *  For ``lambda_ > 0``, the V2/V3 fovea is a band (one point per polar
       angle), so ``dva_to_v2(0, 0)`` and ``dva_to_v3(0, 0)`` are NaN, and
       ``v2_to_dva``/``v3_to_dva`` map every point of the band to (0, 0).
       With ``lambda_ = 0``, all three foveas map to the V1 foveal tip.
    *  On the horizontal meridian, V2/V3 use the upper-quadrant copy
       (Protocol S1 convention). The vertical meridian (x = 0) maps to the
       left hemisphere.
    *  ``a`` and ``b`` change the Double-Sech shear, whose constants were
       optimized for the default ``a`` and ``b``.
    """

    def get_default_params(self):
        params = {
            'k': 18,
            'a': 1.05,
            'b': 90,
            'lambda_': 0.4,
            'alpha1': 1,
            'alpha2': 0.5,
            'alpha3': 0.4,
        }
        return {**super().get_default_params(), **params}

    def get_param_units(self):
        """Return a dict of the units that parameters are stored in"""
        # `lambda_` shifts the 'pacman' plane, whose radius is eccentricity:
        return {**super().get_param_units(), 'k': mm, 'a': dva, 'b': dva,
                'lambda_': dva}

    def _params(self):
        """Returns the model parameters as a hashable tuple"""
        return (float(self.k), float(self.a), float(self.b),
                float(self.lambda_),
                (float(self.alpha1), float(self.alpha2), float(self.alpha3)))

    def _dva_to_cortex(self, x, y, region):
        """Converts dva to cortical coordinates (um) of ``region``"""
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float),
                                   np.asarray(y, dtype=float))
        shape = x.shape
        x, y = x.ravel(), y.ravel()
        left = x < 0
        ecc, theta = np.hypot(x, y), np.arctan2(y, np.abs(x))
        with np.errstate(invalid='ignore'):
            xc, yc = _canonical(ecc, theta, theta >= 0, region,
                                *self._params())
        bad = ~(ecc <= _MAX_ECC)
        if region != 'v1' and self.lambda_ != 0:
            bad |= ecc == 0
        xc[bad], yc[bad] = np.nan, np.nan
        # Upper visual field to lower cortex; right hemifield to the left
        # hemisphere, mirrored:
        xc, yc = 1000 * xc, -1000 * yc
        xc = np.where(left, xc, self.left_offset - xc)
        if not shape:
            return xc[0], yc[0]
        return xc.reshape(shape), yc.reshape(shape)

    def _cortex_to_dva(self, x, y, region):
        """Converts cortical coordinates (um) of ``region`` to dva"""
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float),
                                   np.asarray(y, dtype=float))
        shape = x.shape
        x, y = x.ravel(), y.ravel()
        left = x >= self.left_offset / 2
        xc = np.where(left, x, self.left_offset - x) / 1000
        yc = -y / 1000
        params = self._params()
        tree, ecc0, theta0, upper0 = _seed_tree(region, params)
        xdva = np.full(xc.shape, np.nan)
        ydva = np.full(xc.shape, np.nan)
        for i in np.flatnonzero(np.isfinite(xc) & np.isfinite(yc)):
            target = np.array([xc[i], yc[i]])
            j = tree.query(target)[1]
            upper = upper0[j]

            def residual(p):
                return np.array(_canonical(p[0], p[1], upper, region,
                                           *params)) - target

            # The V2/V3 quadrant is fixed by the seed; the solve stays in it:
            lo = 0 if upper and region != 'v1' else -np.pi / 2
            hi = 0 if not upper and region != 'v1' else np.pi / 2
            fit = least_squares(residual, [ecc0[j], theta0[j]],
                                bounds=([0, lo], [_MAX_ECC, hi]),
                                x_scale='jac', xtol=1e-12, ftol=1e-12,
                                gtol=1e-12)
            if np.hypot(*residual(fit.x)) > _INVERSE_TOL_MM:
                continue
            ecc, theta = fit.x
            if ecc < _FOVEA_ECC:
                ecc = 0.0
            xdva[i] = ecc * np.cos(theta) * (-1 if left[i] else 1)
            ydva[i] = ecc * np.sin(theta)
        if not shape:
            return xdva[0], ydva[0]
        return xdva.reshape(shape), ydva.reshape(shape)

    def dva_to_v1(self, x, y):
        return self._dva_to_cortex(x, y, 'v1')

    def dva_to_v2(self, x, y):
        return self._dva_to_cortex(x, y, 'v2')

    def dva_to_v3(self, x, y):
        return self._dva_to_cortex(x, y, 'v3')

    def v1_to_dva(self, x, y):
        return self._cortex_to_dva(x, y, 'v1')

    def v2_to_dva(self, x, y):
        return self._cortex_to_dva(x, y, 'v2')

    def v3_to_dva(self, x, y):
        return self._cortex_to_dva(x, y, 'v3')

    def plot(self, ax=None):
        """Plots iso-eccentricity and iso-polar lines of V1-V3

        Shows the right hemisphere (left visual field).
        """
        if ax is None:
            ax = plt.gca()
        colors = {'v1': 'gray', 'v2': 'blue', 'v3': 'red'}
        rings = np.array([0.5, 1, 2, 5, 10, 20, 40, 80])
        spokes = np.linspace(-np.pi / 2, np.pi / 2, 5)
        ecc = np.geomspace(1e-3, 80, 100)
        for region, color in colors.items():
            transform = getattr(self, f'dva_to_{region}')
            # Upper and lower quadrants separately, as V2/V3 are split:
            for half in (np.linspace(0, np.pi / 2, 51),
                         np.linspace(-np.pi / 2, -1e-9, 51)):
                for i, r in enumerate(rings):
                    ax.plot(*transform(-r * np.cos(half), r * np.sin(half)),
                            color, linewidth=1,
                            label=region if i == 0 and half[0] == 0 else None)
            for t in np.append(spokes, -1e-9):
                ax.plot(*transform(-ecc * np.cos(t), ecc * np.sin(t)), color,
                        linewidth=1)
        # Coordinates are stored in um; label ticks in mm:
        set_mm_ticks(ax)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_aspect('equal')
        ax.legend()
