"""`Watson2014ConversionMixin`, `Watson2014DisplacementMixin`, `ret2dva`, `dva2ret`"""

import numpy as np
import scipy.stats as spst

from ..utils import cart2pol, pol2cart


class Watson2014ConversionMixin(object):
    """Converts dva to retinal coords using [Watson2014]_

    Converts from eccentricity (defined as distance from a visual center) in
    degrees of visual angle (dva) to microns on the retina using Eqs. A5, A6
    in [Watson2014]_.

    """
    __slots__ = ()

    def get_tissue_coords(self, xdva, ydva):
        """Converts dva to retinal coords

        Parameters
        ----------
        xdva, ydva : double or array-like
            x,y coordinates in dva

        Returns
        -------
        xret, yret : double or array-like
            Corresponding x,y coordinates in microns
        """
        return dva2ret(xdva), dva2ret(ydva)

    def dva2ret(self, xdva):
        return dva2ret(xdva)


class Watson2014DisplacementMixin(object):
    """Converts dva to ret coords with RGC displacement

    Converts from eccentricity (defined as distance from a visual center) in
    degrees of visual angle (dva) to microns on the retina using Eqs. 5, A5,
    and A6 in [Watson2014]_.

    In a central retinal zone, the retinal ganglion cell (RGC) bodies are
    displaced centrifugally some distance from the inner segments of the cones
    to which they are connected through the bipolar cells, and thus from their
    receptive field. The displacement function is described in Eq. 5 of
    [Watson2014]_.

    """
    __slots__ = ()

    @staticmethod
    def _watson_displacement(r, meridian='temporal'):
        """Ganglion cell displacement function

        Implements the ganglion cell displacement function described in Eq. 5
        of [Watson2014]_.

        Parameters
        ----------
        r : double|array-like
            Eccentricity in degrees of visual angle (dva)
        meridian : 'temporal' or 'nasal'

        Returns
        -------
        The displacement in dva experienced by ganglion cells at eccentricity
        ``r``.

        """
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
        """Converts dva to retinal coords

        Parameters
        ----------
        xdva, ydva : double or array-like
            x,y coordinates in dva

        Returns
        -------
        xret, yret : double or array-like
            Corresponding x,y coordinates in microns
        """
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
    of visual angle using Eq. A6 in [Watson2014]_.

    Parameters
    ----------
    r_um : double or array-like
        Eccentricity in microns

    Returns
    -------
    r_dva : double or array-like
        Eccentricity in degrees of visual angle (dva)
    """
    sign = np.sign(r_um)
    r_mm = 1e-3 * np.abs(r_um)
    r_deg = 3.556 * r_mm + 0.05993 * r_mm ** 2 - 0.007358 * r_mm ** 3
    r_deg += 3.027e-4 * r_mm ** 4
    return sign * r_deg


def dva2ret(r_deg):
    """Converts visual angles (deg) into retinal distances (um)

    This function converts degrees of visual angle into a retinal distance from
    the optic axis (um) using Eq. A5 in [Watson2014]_.

    Parameters
    ----------
    r_dva : double or array-like
        Eccentricity in degrees of visual angle (dva)

    Returns
    -------
    r_um : double or array-like
        Eccentricity in microns


    """
    sign = np.sign(r_deg)
    r_deg = np.abs(r_deg)
    r_mm = 0.268 * r_deg + 3.427e-4 * r_deg ** 2 - 8.3309e-6 * r_deg ** 3
    r_um = 1e3 * r_mm
    return sign * r_um
