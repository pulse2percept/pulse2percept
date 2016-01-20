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


def deg2micron(deg):
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
            # Verify that the file was created with a consistent grid:
            axon_id = axon_map['axon_id']
            axon_weight = axon_map['axon_weight']
            xlo_am = axon_map['xlo']
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
            np.savez(fname,
                     axon_id=axon_id,
                     axon_weight=axon_weight,
                     xlo=[xlo],
                     xhi=[xhi],
                     ylo=[ylo],
                     yhi=[yhi],
                     sampling=[sampling])

        self.axon_id = axon_id
        self.axon_weight = axon_weight

    def cm2ecm(self, cm):
        """
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
        """

        ecm = np.zeros(cm.shape)
        for id in range(0, len(cm.flat)):
            ecm.flat[id] = np.dot(cm.flat[self.axon_id[id]],
                                  self.axon_weight[id])
        return ecm


class Electrode(object):
    """
    Represent a circular, disc-like electrode.
    """
    def __init__(self, radius, x, y):
        """
        Initialize an electrode object

        Parameters
        ----------
        radius : float
            The radius of the electrode (in microns).
        x : float
            The x coordinate of the electrode (in microns).
        y : float
            The y location of the electrode (in microns).
        """
        self.radius = radius
        self.x = x
        self.y = y

    def current_map(self, xg, yg, current_amp=1, alpha=14000, n=1.69):
        """
        The current map due to a current pulse through an electrode, reflecting
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
        r = np.sqrt((xg + self.x) ** 2 + (yg + self.y) ** 2).T
        cspread = np.ones(r.shape)
        cspread[r > self.radius] = (alpha / (alpha + (r[r > self.radius] -
                                             self.radius) ** n))
        return cspread * current_amp


class ElectrodeGrid(object):
    """
    Represent a retina and grid of electrodes
    """
    def __init__(self, radii, xs, ys):
        self.electrodes = []
        for r, x, y in zip(radii, xs, ys):
            self.electrodes.append(Electrode(r, x, y))

    def current_map(self, xg, yg, current_amps, alpha=14000, n=1.69):
        c = np.zeros((len(self.electrodes), xg.shape[0], xg.shape[1]))
        for i in range(c.shape[0]):
            c[i] = self.electrodes[i].current_map(xg, yg,
                                                  current_amp=current_amps[i],
                                                  alpha=alpha, n=n)
        return np.sum(c, 0)
