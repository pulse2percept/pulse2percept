"""
`RetinalMap`, `Curcio1990Map`, `Watson2014Map`, `Watson2014DisplaceMap`
"""
import numpy as np
from abc import abstractmethod
import scipy.stats as spst

from .base import VisualFieldMap
from ..utils.geometry import cart2pol, pol2cart


class RetinalMap(VisualFieldMap):
    """ Template class for retinal visual field maps, which only have 1 region."""
    split_map = False
    regions = ['ret']
    def __init__(self, **params):
        super().__init__(**params)

    def from_dva(self):
        return {'ret' : self.dva_to_ret}
    
    def to_dva(self):
        return {'ret' : self.ret_to_dva}
    
    @abstractmethod
    def dva_to_ret(self, x, y):
        """Convert degrees of visual angle (dva) to retinal coords (um)"""
        raise NotImplementedError
        
    def ret_to_dva(self, x, y):
        """Convert retinal coords (um) to degrees of visual angle (dva)"""
        raise NotImplementedError


class Curcio1990Map(RetinalMap):
    """Converts between visual angle and retinal eccentricity [Curcio1990]_"""

    def dva_to_ret(self, xdva, ydva):
        """Convert degrees of visual angle (dva) to retinal eccentricity (um)

        Assumes that one degree of visual angle is equal to 280 um on the
        retina [Curcio1990]_.
        """
        return 280.0 * xdva, 280.0 * ydva

    def ret_to_dva(self, xret, yret):
        """Convert retinal eccentricity (um) to degrees of visual angle (dva)

        Assumes that one degree of visual angle is equal to 280 um on the
        retina [Curcio1990]_
        """
        return xret / 280.0, yret / 280.0

    def __eq__(self, other):
        """
        Equality operator for Curcio1990Map.
        Compares two Curcio1990Map's based attribute equality

        Parameters
        ----------
        other: SpatialModel
            SpatialModel to compare with

        Returns
        -------
        bool:
            True if the compared objects have identical attributes, False otherwise.
        """
        if not isinstance(other, Curcio1990Map):
            return False
        if id(self) == id(other):
            return True
        return self.__dict__ == other.__dict__


class Watson2014Map(RetinalMap):
    """Converts between visual angle and retinal eccentricity [Watson2014]_"""

    def ret_to_dva(self, x_um, y_um, coords='cart'):
        """Converts retinal distances (um) to visual angles (deg)

        This function converts an eccentricity measurement on the retinal
        surface(in micrometers), measured from the optic axis, into degrees
        of visual angle using Eq. A6 in [Watson2014]_.

        Parameters
        ----------
        x_um, y_um : double or array-like
            Original x and y coordinates on the retina (microns)
        coords : {'cart', 'polar'}
            Whether to return the result in Cartesian or polar coordinates

        Returns
        -------
        x_dva, y_dva : double or array-like
            Transformed x and y coordinates (degrees of visual angle, dva)
        """
        phi_um, r_um = cart2pol(x_um, y_um)
        sign = np.sign(r_um)
        r_mm = 1e-3 * np.abs(r_um)
        r_deg = 3.556 * r_mm + 0.05993 * r_mm ** 2 - 0.007358 * r_mm ** 3
        r_deg += 3.027e-4 * r_mm ** 4
        r_deg *= sign
        if coords.lower() == 'cart':
            return pol2cart(phi_um, r_deg)
        elif coords.lower() == 'polar':
            return phi_um, r_deg
        raise ValueError(f'Unknown coordinate system "{coords}".')

    def dva_to_ret(self, x_deg, y_deg, coords='cart'):
        """Converts visual angles (deg) into retinal distances (um)

        This function converts degrees of visual angle into a retinal distance 
        from the optic axis (um) using Eq. A5 in [Watson2014]_.

        Parameters
        ----------
        x_dva, y_dva : double or array-like
            Original x and y coordinates (degrees of visual angle, dva)
        coords : {'cart', 'polar'}
            Whether to return the result in Cartesian or polar coordinates

        Returns
        -------
        x_ret, y_ret : double or array-like
            Transformed x and y coordinates on the retina (microns)

        """
        phi_deg, r_deg = cart2pol(x_deg, y_deg)
        sign = np.sign(r_deg)
        r_deg = np.abs(r_deg)
        r_mm = 0.268 * r_deg + 3.427e-4 * r_deg ** 2 - 8.3309e-6 * r_deg ** 3
        r_um = 1e3 * r_mm * sign
        if coords.lower() == 'cart':
            return pol2cart(phi_deg, r_um)
        elif coords.lower() == 'polar':
            return phi_deg, r_um
        raise ValueError(f'Unknown coordinate system "{coords}".')

    def __eq__(self, other):
        """
        Equality operator for Watson2014 Object.
        Compares two Watson2014 Objects based attribute equality

        Parameters
        ----------
        other: Watson2014Map
            Watson2014 Object to compare against

        Returns
        -------
        bool:
            True if the compared objects have identical attributes, False otherwise.
        """
        if not isinstance(other, Watson2014Map):
            return False
        if id(self) == id(other):
            return True
        return self.__dict__ == other.__dict__


class Watson2014DisplaceMap(Watson2014Map):
    """Converts between visual angle and retinal eccentricity using RGC
    displacement [Watson2014]_

    Converts from eccentricity (defined as distance from a visual center) in
    degrees of visual angle (dva) to microns on the retina using Eqs. 5, A5,
    and A6 in [Watson2014]_.

    In a central retinal zone, the retinal ganglion cell (RGC) bodies are
    displaced centrifugally some distance from the inner segments of the cones
    to which they are connected through the bipolar cells, and thus from their
    receptive field. The displacement function is described in Eq. 5 of
    [Watson2014]_.

    """

    def watson_displacement(self, r, meridian='temporal'):
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

    def dva_to_ret(self, xdva, ydva):
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
        # Convert x, y (dva) into polar coordinates:
        theta, rho_dva = cart2pol(xdva, ydva)
        # Add RGC displacement:
        meridian = np.where(xdva < 0, 'temporal', 'nasal')
        rho_dva += self.watson_displacement(rho_dva, meridian=meridian)
        # Convert back to x, y (dva):
        x, y = pol2cart(theta, rho_dva)
        return super(Watson2014DisplaceMap, self).dva_to_ret(x, y)

    def ret_to_dva(self, xret, yret):
        raise NotImplementedError