"""

Functions for transforming electrode specification into a current map

"""
import numpy as np

class Electrode(object):
    """
    Represent a circular, disc-like electrode.
    """
    def __init__(self, radius, x, y, sizex, sizey, sampling=25,
                 alpha=14000, n=1.69):
        """
        Initialize an electrode object

        Parameters
        ----------
        radius : float
            The radius of the electrode (in microns).
        x : float
            The x coordinate of the electrode, relative to ? (in microns).
        y : float
            The y location of the electrode, relative to ? (in microns).
        """
        self.radius = radius
        self.x = x
        self.y = y
        self.sizex = sizex
        self.sizey = sizey
        self.sampling = sampling
        self.set_scale(alpha=alpha, n=n)

    def scale_current(self, alpha=14000, n=1.69):
        """
        Scaling factor applied to the current through an electrode, reflecting
        the fall-off of the current as a function of distance from the
        electrode center. This is equation 2 in Nanduri et al [1]_.

        Parameters
        ----------

        alpha : a constant to do with the spatial fall-off.

        n : a constant to do with the spatial fall-off (Default: 1.69, based on
        Ahuja et al. [2]_)

        .. [1]

        .. [2] An In Vitro Model of a Retinal Prosthesis. Ashish K. Ahuja,
        Matthew R. Behrend, Masako Kuroda, Mark S. Humayun, and
        James D. Weiland (2008). IEEE Trans Biomed Eng 55.
        """
        [gridx, gridy] = np.meshgrid(np.arange(-self.sizex//2,
                                               self.sizex//2, self.sampling),
                                     np.arange(-self.sizey//2,
                                               self.sizey//2, self.sampling))

        r = np.sqrt((gridx + self.x) ** 2 + (gridy + self.y) ** 2)
        cspread = np.ones(r.shape)
        cspread[r > self.radius] = (alpha / (alpha + (r[r > self.radius] -
                                             self.radius) ** n))
        return cspread

    def set_scale(self, alpha=14000, n=1.69):
        self._scale = self.scale_current(alpha, n)

    def get_scale(self):
        return self._scale

    scale = property(get_scale, set_scale)


class ElectrodeGrid(object):
    def __init__(self):
        pass
