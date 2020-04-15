"""`Nanduri2012Model`"""
import numpy as np
from .base import Model, SpatialModel, TemporalModel
from ._nanduri2012 import spatial_fast, temporal_fast


class Nanduri2012Spatial(SpatialModel):
    """Spatial model"""

    def get_default_params(self):
        """Returns all settable parameters of the Nanduri model"""
        params = super().get_default_params()
        params.update({'atten_a': 14000, 'atten_n': 1.69})
        return params

    def dva2ret(self, xdva):
        """Convert degrees of visual angle (dva) to retinal eccentricity (um)

        Assumes that one degree of visual angle is equal to 288 um on the
        retina.
        """
        return 288.0 * xdva

    def ret2dva(self, xret):
        """Convert retinal eccentricity (um) to degrees of visual angle (dva)

        Assumes that one degree of visual angle is equal to 288 um on the
        retina.
        """
        return xret / 288.0

    def predict_spatial(self, implant, t):
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
        # A Stimulus could be compressed to zero:
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


class Nanduri2012Temporal(TemporalModel):
    """Temporal model"""

    def get_default_params(self):
        base_params = super().get_default_params()
        params = {
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

    def predict_temporal(self, spatial, t_spatial, t_percept):
        t_percept = np.array([t_percept]).flatten()
        # We need to make sure the requested `t_percept` are multiples of `dt`:
        remainder = np.mod(t_percept, self.dt) / self.dt
        atol = 1e-3
        within_atol = (remainder < atol) | (np.abs(1 - remainder) < atol)
        if not np.all(within_atol):
            raise ValueError("t=%s are not multiples of dt=%.2e." %
                             (t_percept[np.logical_not(within_atol)], self.dt))

        # Beware of floating point errors! 29.999 will be rounded down to 29
        # by np.uint, so we need to np.round it first:
        idx_percept = np.uint64(np.round(t_percept / self.dt))
        t_percept = idx_percept * self.dt
        if np.unique(idx_percept).size < t_percept.size:
            raise ValueError("All times 't' must be distinct multiples of "
                             "`dt`=%.2e" % self.dt)
        return temporal_fast(spatial.astype(np.float32),
                             t_spatial.astype(np.float32),
                             idx_percept,
                             self.dt, self.tau1, self.tau2, self.tau3,
                             self.asymptote, self.shift, self.slope, self.eps)


class Nanduri2012Model(Model):
    """Nanduri et al. (2012) Model

    Implements the model described in [Nanduri2012]_, where percepts are
    circular and their brightness evolves over time.

    Parameters
    ----------
    dt : float, optional, default: 5 microseconds
        Sampling time step (seconds)
    tau1: float, optional, default: 0.42 ms
        Time decay constant for the fast leaky integrater.
    tau2: float, optional, default: 45.25 ms
        Time decay constant for the charge accumulation.
    tau3: float, optional, default: 26.25 ms
        Time decay constant for the slow leaky integrator.
        Default: 26.25 / 1000 s.
    eps: float, optional, default: 8.73
        Scaling factor applied to charge accumulation.
    asymptote: float, optional, default: 14.0
        Asymptote of the logistic function used in the stationary nonlinearity
        stage.
    slope: float, optional, default: 3.0
        Slope of the logistic function in the stationary nonlinearity stage.
    shift: float, optional, default: 16.0
        Shift of the logistic function in the stationary nonlinearity stage.
    thresh_percept: float, optional, default: 0
        Below threshold, the percept has brightness zero.
    """

    def __init__(self, **params):
        super().__init__(spatial=Nanduri2012Spatial(),
                         temporal=Nanduri2012Temporal(),
                         **params)
