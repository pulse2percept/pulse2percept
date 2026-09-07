""":py:class:`~pulse2percept.topography.retina.Curcio1990Map`"""
from .base import RetinalMap


class Curcio1990Map(RetinalMap):
    """Converts between visual angle and retinal eccentricity [Curcio1990]_"""

    def dva_to_ret(self, xdva, ydva):
        """Convert degrees of visual angle (dva) to retinal eccentricity (um)

        Assumes that one degree of visual angle is equal to 280 um on the
        retina [Curcio1990]_.
        """
        return 280.0 * xdva, -280.0 * ydva

    def ret_to_dva(self, xret, yret):
        """Convert retinal eccentricity (um) to degrees of visual angle (dva)

        Assumes that one degree of visual angle is equal to 280 um on the
        retina [Curcio1990]_
        """
        return xret / 280.0, -yret / 280.0

