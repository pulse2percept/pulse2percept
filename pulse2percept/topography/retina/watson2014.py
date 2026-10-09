""":py:class:`~pulse2percept.topography.retina.Watson2014Map`"""
import numpy as np

from .base import RetinalMap
from ...units import Quantity, mm, um
from ...utils.geometry import cart2pol, pol2cart


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
        # Eq. A6 is fitted in mm; `tissue_unit` is um:
        r_mm = Quantity(np.abs(r_um), um).to_value(mm)
        r_deg = 3.556 * r_mm + 0.05993 * r_mm ** 2 - 0.007358 * r_mm ** 3
        r_deg += 3.027e-4 * r_mm ** 4
        r_deg *= sign

        # flip y axis
        phi_um *= -1

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
        # Eq. A5 gives millimeters; `tissue_unit` is microns:
        r_um = Quantity(r_mm, mm).to_value(um) * sign

        # flip y axis
        phi_deg *= -1

        if coords.lower() == 'cart':
            return pol2cart(phi_deg, r_um)
        elif coords.lower() == 'polar':
            return phi_deg, r_um
        raise ValueError(f'Unknown coordinate system "{coords}".')
