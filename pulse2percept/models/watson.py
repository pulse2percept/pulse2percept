import numpy as np
import scipy.stats as spst

from ..utils import cart2pol, pol2cart


class WatsonConversionMixin(object):
    """Converts dva to retinal coords using Watson (2014)"""

    def get_tissue_coords(self, xdva, ydva):
        return dva2ret(xdva), dva2ret(ydva)


class WatsonDisplacementMixin(object):
    """Converts dva to ret coords with RGC displacement using Watson (2014)"""

    @staticmethod
    def _watson_displacement(r, meridian='temporal'):
        if (not isinstance(meridian, (np.ndarray, str)) or
                not np.all([m in ['temporal', 'nasal']
                            for m in np.array([meridian]).ravel()])):
            raise ValueError("'meridian' must be either 'temporal' or 'nasal'")
        alpha = np.where(meridian == 'temporal', 1.8938, 2.4607)
        beta = np.where(meridian == 'temporal', 2.4598, 1.7463)
        gamma = np.where(meridian == 'temporal', 0.91565, 0.77754)
        delta = np.where(meridian == 'temporal', 14.904, 15.111)
        mu = np.where(meridian == 'temporal', -0.09386, -0.15933)
        scale = np.where(meridian == 'temporal', 12.0, 10.0)
        # Formula:
        rmubeta = (np.abs(r) - mu) / beta
        numer = delta * gamma * np.exp(-rmubeta ** gamma)
        numer *= rmubeta ** (alpha * gamma - 1)
        denom = beta * spst.gamma.pdf(alpha, 5)
        return numer / denom / scale

    def get_tissue_coords(self, xdva, ydva):
        if self.eye == 'LE':
            raise NotImplementedError
        # Convert x, y (dva) into polar coordinates:
        theta, rho_dva = utils.cart2pol(xdva, ydva)
        # Add RGC displacement:
        meridian = np.where(xdva < 0, 'temporal', 'nasal')
        rho_dva += self._watson_displacement(rho_dva, meridian=meridian)
        # Convert back to x, y (dva):
        x, y = utils.pol2cart(theta, rho_dva)
        # Convert to retinal coords:
        return dva2ret(x), dva2ret(y)


def ret2dva(r_um):
    """Converts retinal distances (um) to visual angles (deg)

    This function converts an eccentricity measurement on the retinal
    surface(in micrometers), measured from the optic axis, into degrees
    of visual angle.
    Source: Eq. A6 in Watson(2014), J Vis 14(7): 15, 1 - 17
    """
    sign = np.sign(r_um)
    r_mm = 1e-3 * np.abs(r_um)
    r_deg = 3.556 * r_mm + 0.05993 * r_mm ** 2 - 0.007358 * r_mm ** 3
    r_deg += 3.027e-4 * r_mm ** 4
    return sign * r_deg


def dva2ret(r_deg):
    """Converts visual angles (deg) into retinal distances (um)

    This function converts degrees of visual angle into a retinal distance from
    the optic axis (um).
    Source: Eq. A5 in Watson(2014), J Vis 14(7): 15, 1 - 17
    """
    sign = np.sign(r_deg)
    r_deg = np.abs(r_deg)
    r_mm = 0.268 * r_deg + 3.427e-4 * r_deg ** 2 - 8.3309e-6 * r_deg ** 3
    r_um = 1e3 * r_mm
    return sign * r_um
