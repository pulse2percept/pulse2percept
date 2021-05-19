"""`FadingTemporal`"""
import numpy as np
from .base import TemporalModel
from ._temporal import fading_fast


class FadingTemporal(TemporalModel):
    """A generic temporal model for phosphene fading

    Implements phosphene fading using a leaky integrator:

    .. math::

        \\frac{dB}{dt} = -\\frac{A+B}{\\tau}

    where :math:`A` is the stimulus  amplitude, :math:`B` is the perceived
    brightness, and :math:`\\tau` is the exponential  decay constant (``tau``).

    The model makes the following assumptions:

    *  Cathodic currents (negative amplitudes) will increase perceived
       brightness
    *  Anodic currents (positive amplitudes) will decrease brightness
    *  Brightness is bounded in :math:`[\\theta, \\infinity[`, where 
       :math:`\\theta` (``thresh_percept``) is a nonnegative scalar

    Parameters
    ----------
    dt : float, optional
        Sampling time step of the simulation (ms)
    tau : float, optional
        Time decay constant for the exponential decay (ms).
        Larger values lead to slower decay.
        Brightness should decay to half its peak ("half-life") after
        :math:`\\ln(2) \\tau` milliseconds.
    thresh_percept: float, optional
        Below threshold, the percept has brightness zero.

    .. versionadded: 0.7.1

    """

    def get_default_params(self):
        base_params = super(FadingTemporal, self).get_default_params()
        params = {
            # Time constant for the exponential decay:
            'tau': 0.01,
        }
        # This is subtle: Rather than calling `params.update(base_params)`, we
        # call `base_params.update(params)`. This will overwrite `base_params`
        # with values from `params`, which allows us to set `thresh_percept`=0
        # rather than what the BaseModel dictates:
        base_params.update(params)
        return base_params

    def _build(self):
        if self.tau < 0:
            raise ValueError('"tau" cannot be negative.')

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
            raise ValueError("All times 't_percept' must be distinct multiples "
                             "of `dt`=%.2e" % self.dt)
        # Cython returns a 2D (space x time) NumPy array:
        return fading_fast(stim_data.astype(np.float32),
                           stim.time.astype(np.float32),
                           idx_percept, self.dt, self.tau, self.thresh_percept)
