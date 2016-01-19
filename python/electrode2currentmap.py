"""

Functions for transforming electrode specifications into a current map

"""
import numpy as np
import oyster
import os

def micron2deg(micron):
    """
    Transform a distance from microns to degrees

    Based on http://retina.anatomy.upenn.edu/~rob/lance/units_space.html
    """
    deg = micron / 280
    return deg


def deg2microns(deg):
    """
    Transform a distance from degrees to microns

    Based on http://retina.anatomy.upenn.edu/~rob/lance/units_space.html
    """
    microns = 280 * deg
    return microns


class Retina():
    """
    Represent the retinal coordinate frame
    """
    def __init__(self, xlo=-1000, xhi=1000, ylo=-1000, yhi=1000,
                 sampling=25, axon_map=None, axon_lambda=2):
        """
        Initialize a retina

        axon_map :
        """
        [self.gridx, self.gridy] = np.meshgrid(np.arange(xlo, xhi,
                                                         sampling),
                                               np.arange(ylo, yhi,
                                                         sampling))

        if os.path.exists(axon_map):
            axon_map = np.load(axon_map)
            ## Verify that the file was created with a consistent grid:
            axon_id = axon_map['axon_id']
            axon_weight = axon_map['axon_weight']
            xlo_am = axon_map['xlo'],
            xhi_am = axon_map['xhi']
            ylo_am = axon_map['ylo']
            yhi_am = axon_map['yhi']
            sampling_am = axon_map['sampling']
            assert xlo == xlo_am
            assert xhi == xhi_am
            assert ylo == ylo_am
            assert yhi == yhi_am
            assert sampling_am == sampling

        else:
            print("Can't find file %s, generating"%axon_map)
            axon_id, axon_weight = oyster.makeAxonMap(micron2deg(self.gridx),
                                                      micron2deg(self.gridy),
                                                      axon_lambda=axon_lambda)
            ## Save the variables, together with metadata about the grid:
            fname = axon_map
            np.savez(fname, axon_id=axon_id, axon_weight=axon_weight, xlo=[xlo],
                     xhi=[xhi], ylo=[ylo], yhi=[yhi], sampling=[sampling])

        self.axon_id = axon_id
        self.axon_weight = axon_weight

    def cm2ecm(self, cm):
        '''
        Converts a current map to an 'effective' current map, by passing the
        map through a mapping of axon streaks.

        Inputs:
            cm: current map, an image in retinal space
            axon_id :
            axon_map :
            output of 'makeAxonMap'

        Output:
            ecm: effective current map, an image of the same size as the current
                map, where each pixel is the dot product of the pixel values in ecm
                along the pixels in the list in axon_map, weighted by the weights
                axon map.
        '''

        ecm = np.zeros(cm.shape)
        for id in range(0, len(cm.flat)):
            ecm.flat[id] = np.dot(cm.flat[self.axon_id[id]],
                                          self.axon_weight[id])
        return ecm


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
            The x coordinate of the electrode (in microns).
        y : float
            The y location of the electrode (in microns).
        """
        self.radius = micron2degrees(radius)
        self.x = micron2degrees(x)
        self.y = micron2degrees(y)
        self.sizex = retina.sizex_deg
        self.sizey = retina.sizey_deg
        self.sampling = retina.sampling_deg
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


class CurrentMap(object):
    """
    Represent a retina and grid of electrodes
    """
    def __init__(self, retina, radii, xs, ys, sizex, sizey, sampling=25,
                 alpha=14000, n=1.69):

        self.electrodes = []
        for r, x, y in zip(radii, xs, ys):
            self.electrodes.append(Electrode(retina, r, x, y, sizex, sizey,
                                             sampling=sampling, alpha=alpha,
                                             n=n))
