"""`Horsager2009Model`, `Horsager2009Temporal` [Horsager2009]_"""
import numpy as np
import copy
from .base import Model, TemporalModel
from ._horsager2009 import temporal_fast


class Horsager2009Temporal(TemporalModel):
    """Temporal model of [Horsager2009]_

    Implements the temporal response model described in [Horsager2009]_, which
    assumes that the temporal activation of retinal tissue is the output of a
    linear-nonlinear model cascade (see Fig.2 in the paper).

    .. note ::

        Use this class if you want to combine the temporal model with a spatial
        model.
        Use :py:class:`~pulse2percept.models.Horsager2009Model` if you want a
        a standalone model.

    Parameters
    ----------
    dt : float, optional
        Sampling time step (ms)
    tau1 : float, optional
        Time decay constant for the fast leaky integrater.
    tau2 : float, optional
        Time decay constant for the charge accumulation.
    tau3 : float, optional
        Time decay constant for the slow leaky integrator.
    eps : float, optional
        Scaling factor applied to charge accumulation. Common values at
        threshold: 2.25, suprathreshold: 8.73.
    beta : float, optional
        Power nonlinearity (exponent of the half-wave rectification).
        Common values at threshold: 3.43, suprathreshold: 0.83.
    thresh_percept: float, optional
        Below threshold, the percept has brightness zero.
    n_threads: int, optional
            Number of CPU threads to use during parallelization using OpenMP. Defaults to max number of user CPU cores.

    """

    def get_default_params(self):
        base_params = super(Horsager2009Temporal, self).get_default_params()
        params = {
            # Time decay for the ganglion cell impulse response:
            'tau1': 0.42,
            # Time decay for the charge accumulation:
            'tau2': 45.25,
            # Time decay for the slow leaky integrator:
            'tau3': 26.25,
            # Scaling factor applied to charge accumulation:
            'eps': 2.25,
            # Exponent:
            'beta': 3.43
        }
        # This is subtle: Rather than calling `params.update(base_params)`, we
        # call `base_params.update(params)`. This will overwrite `base_params`
        # with values from `params`, which allows us to set `thresh_percept`=0
        # rather than what the BaseModel dictates:
        base_params.update(params)
        return base_params

    def _predict_temporal(self, stim, t_percept):
        """Predict the temporal response"""
        # Pass the stimulus as a 2D NumPy array to the fast Cython function:
        stim_data = stim.data.reshape((-1, len(stim.time)))
        # Calculate at which simulation time steps we need to output a percept.
        # This is basically t_percept/self.dt, but we need to beware of
        # floating point rounding errors! 29.999 will be rounded down to 29 by
        # np.uint32, so we need to np.round it first:
        idx_percept = np.uint32(np.round(t_percept / self.dt))
        if np.unique(idx_percept).size < t_percept.size:
            raise ValueError(f"All times 't_percept' must be distinct multiples "
                             f"of `dt`={self.dt:.2e}")
        # Cython returns a 2D (space x time) NumPy array:
        return temporal_fast(stim_data.astype(np.float32),
                             stim.time.astype(np.float32),
                             idx_percept,
                             self.dt, self.tau1, self.tau2, self.tau3,
                             self.eps, self.beta, self.thresh_percept, self.n_threads)


class Horsager2009Model(Model):
    """[Horsager2009]_ Standalone model

    Implements the temporal response model described in [Horsager2009]_, which
    assumes that the temporal activation of retinal tissue is the output of a
    linear-nonlinear model cascade (see Fig.2 in the paper).

    .. note ::

        Use this class if you want a standalone model.
        Use :py:class:`~pulse2percept.models.Horsager2009Temporal` if you want
        to combine the temporal model with a spatial model.

    Parameters
    ----------
    dt : float, optional
        Sampling time step (ms)
    tau1 : float, optional
        Time decay constant for the fast leaky integrater.
    tau2 : float, optional
        Time decay constant for the charge accumulation.
    tau3 : float, optional
        Time decay constant for the slow leaky integrator.
    eps : float, optional
        Scaling factor applied to charge accumulation. Common values at
        threshold: 0.00225, suprathreshold: 0.00873.
        Power nonlinearity (exponent of the half-wave rectification).
        Common values at threshold: 3.43, suprathreshold: 0.83.
    thresh_percept: float, optional
        Below threshold, the percept has brightness zero.
    n_threads: int, optional
            Number of CPU threads to use during parallelization using OpenMP. Defaults to max number of user CPU cores.

    """

    def __init__(self, **params):
        super(Horsager2009Model, self).__init__(spatial=None,
                                                temporal=Horsager2009Temporal,
                                                **params)

    def __deepcopy__(self, memodict={}):
        """
        Perform a deep copy of the Horsager2009Model object.

        Parameters
        ----------
        memodict: dict
            Dictionary of objects already copied during the current copying pass.

        Returns
            Deep copy of the object
        -------

        """
        # Check if already been copied
        if id(self) in memodict:
            return memodict[id(self)]

        # Deep copy original object's attributes
        attributes = copy.deepcopy(self.__dict__)

        # Remove attributes set internally by Horsager2009Model
        attributes.pop('spatial')
        attributes.pop('temporal')

        # Perform the copy by creating a new object with deep copied attributes
        copied = self.__class__(**attributes)

        # Save copied
        memodict[id(copied)] = copied

        return copied