"""This module implements the model described in [#nanduri-2012]_.
References
----------
.. [#nanduri-2012] D Nanduri, I Fine, A Horsager, GM Boynton, MS Humayun,
                   RJ Greenberg, JD Weiland (2012), Frequency and amplitude
                   modulation have different effects on the percepts elicited
                   by retinal stimulation. Investigative Ophthalmology & Visual
                   Science 53:205-214, doi:`10.1167/iovs.11-8401 <https://doi.org/10.1167/iovs.11-8401>`_.
"""
import numpy as np
from .base import BaseModel
from .scoreboard import ScoreboardModel
from ._nanduri2012 import spatial_fast, temporal_fast


class Nanduri2012SpatialMixin(object):
    """Nanduri spatial mixin"""

    def _get_default_params(self):
        """Returns all settable parameters of the scoreboard model"""
        params = super(Nanduri2012SpatialMixin, self)._get_default_params()
        params.update({'atten_a': 14000, 'atten_n': 1.69})
        return params

    def dva2ret(self, xdva):
        """Convert degrees of visual angle (dva) to retinal eccentricity (um)

        Assumes that one degree of visual angle is equal to 288 um on the
        retina.
        """
        return 288.0 * xdva

    def _predict_spatial(self, implant, t):
        """Predicts the brightness at spatial locations"""
        if t is None:
            t = implant.stim.time
        if implant.stim.time is None and t is not None:
            raise ValueError("Cannot calculate spatial response at times "
                             "t=%s, because stimulus does not have a time "
                             "component." % t)
        # Interpolate stimulus at desired time points:
        if implant.stim.time is None:
            stim = implant.stim.data.astype(np.float32)
        else:
            stim = implant.stim[:, np.array([t]).ravel()].astype(np.float32)

        if stim.size == 0:
            return np.zeros((np.array([t]).size, np.prod(self.grid.x.shape)),
                            dtype=np.float32)
        # This does the expansion of a compact stimulus and a list of
        # electrodes to activation values at X,Y grid locations:
        electrodes = implant.stim.electrodes
        return spatial_fast(stim,
                            np.array([implant[e].x for e in electrodes],
                                     dtype=np.float32),
                            np.array([implant[e].y for e in electrodes],
                                     dtype=np.float32),
                            np.array([implant[e].z for e in electrodes],
                                     dtype=np.float32),
                            np.array([implant[e].r for e in electrodes],
                                     dtype=np.float32),
                            self.grid.xret.ravel(),
                            self.grid.yret.ravel(),
                            self.atten_a,
                            self.atten_n,
                            self.thresh_percept)


class Nanduri2012TemporalMixin(object):

    def _get_default_params(self):
        base_params = super()._get_default_params()
        params = {
            'has_time': True,
            'thresh_percept': 0,
            # Simulation time step:
            'dt': 0.005 / 1000,
            # Time decay for the ganglion cell impulse response:
            'tau1': 0.42 / 1000,
            # Time decay for the charge accumulation:
            'tau2': 45.25 / 1000,
            # Time decay for the slow leaky integrator:
            'tau3': 26.25 / 1000,
            # Scaling factor applied to charge accumulation:
            'eps': 8.73,
            # Asymptote of the sigmoid:
            'asymptote': 14.0,
            # Slope of the sigmoid:
            'slope': 3.0,
            # Shift of the sigmoid:
            'shift': 16.0,
        }
        # This is subtle: Rather than calling `params.update(base_params)`, we
        # call `base_params.update(params)`. This will overwrite `base_params`
        # with values from `params`, which allows us to set `thresh_percept`=0
        # rather than what the BaseModel dictates:
        base_params.update(params)
        return base_params

    def _predict_temporal(self, spatial, t_spatial, t_percept):
        # TODO what if t_percept is not a multiple of self.dt?
        # Beware of floating point errors!!! 29.999 will be rounded down to
        # 29 by np.uint:
        idx_percept = np.uint(np.round(t_percept / self.dt))
        t_percept = idx_percept * self.dt
        return temporal_fast(spatial.astype(np.float32),
                             t_spatial.astype(np.float32),
                             idx_percept,
                             self.dt, self.tau1, self.tau2, self.tau3,
                             self.asymptote, self.shift, self.slope, self.eps)


class Nanduri2012Model(Nanduri2012TemporalMixin, Nanduri2012SpatialMixin,
                       BaseModel):
    pass
