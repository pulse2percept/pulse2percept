"""

Functions for transforming electrode specification into a current map

"""
import numpy as np



class Retina():
    """
    Represent the retinal coordinate frame
    """
    def __init__(sizex=2000, sizey=2000, sampling=25):
        """

        """
        self.sizex_micron = sizex  # micron
        self.sizey_micron = sizey  # micron
        self.sampling_micron = sampling # microns per grid-cell
        self.size_micron = [self.sizex_micron, self.sizey_micron]
        self.sizex_deg = micron2degrees(self.sizex_micron)
        self.sizey_deg = micron2degrees(self.sizey_micron)
        self.size_deg = [self.sizex_deg, self.sizey_deg]
        self.sampling_deg = micron2degrees(sampling)

    def micron2degrees(micron):
        """
        Transform a distance from microns to degrees

        Based on http://retina.anatomy.upenn.edu/~rob/lance/units_space.html
        """
        deg = micron / 280
        return deg

    def degrees2microns(deg):
        """
        Transform a distance from degrees to microns

        Based on http://retina.anatomy.upenn.edu/~rob/lance/units_space.html
        """
        microns = 280 * deg
        return microns



def set_size_degrees(self, alpha=14000, n=1.69):
    self._scale = self.scale_current(alpha, n)

def get_scale(self):
    return self._scale

scale = property(get_scale, set_scale)


class Electrode(object):
    """
    Represent a circular, disc-like electrode.
    """
    def __init__(self, retina, radius, x, y, alpha=14000, n=1.69):
        """
        Initialize an electrode object

        Parameters
        ----------
        retina : a Retina object
            The retina on which this electrode is placed
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

        r = np.sqrt((gridx + self.x) ** 2 + (gridy + self.y) ** 2).T
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
    """
    Represent a grid of electrodes
    """
    def __init__(self, radii, xs, ys, sizex, sizey, sampling=25,
                 alpha=14000, n=1.69):

        self.electrodes = []
        for r, x, y in zip(radii, xs, ys):
            self.electrodes.append(Electrode(r, x, y, sizex, sizey,
                                             sampling=sampling, alpha=alpha,
                                             n=n))
        self.set_scale(alpha=alpha, n=n)

    def set_scale(self, alpha=14000, n=1.69):
        scale_list = []
        for e in self.electrodes:
            scale_list.append(e.scale_current(alpha, n))
        self._scale = np.array(scale_list)

    def get_scale(self):
        return self._scale

    scale = property(get_scale, set_scale)
