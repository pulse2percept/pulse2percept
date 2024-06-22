"""`FadingTemporal`"""
import numpy as np
from .base import TemporalModel, TorchBaseModel
from ._temporal import fading_fast
import torch

class TorchFadingTemporal(TorchBaseModel):
    
    def __init__(self, p2pmodel):
        super().__init__(p2pmodel)
        self.dt = torch.tensor(p2pmodel.dt, device=self.device)
        self.tau = torch.tensor(p2pmodel.tau, device=self.device)

    def forward(self, stim, state=None, model_params=None):
        """_summary_

        Parameters
        ----------
        stim : torch.Tensor
            One frame of output from a spatial model. This frame will represent
            the stimulus at a given time point. Must have shape (n_pixels)
        state : torch.Tensor, optional
            The internal state of the model. state[0] corresponds to ``bright``
            Defaults to None. If None, the state is assumed to be zero.
        model_params : torch.Tensor, optional
            The model parameters to use when predicting the percept.
            model_params[0] is ``tau``.

        Returns
        -------
        percept : torch.Tensor
            One frame of the predicted response
        state : torch.Tensor
            The internal state of the model after this pass. 
            state[0] corresponds to ``bright``

        """

        tau = self.tau if model_params is None else torch.Tensor(model_params[0], device=self.device)


        if state is None:
            state = torch.unsqueeze(torch.zeros_like(stim, device=self.device), 0)
        
        # get brightness state out
        bright = state[0]

        bright = torch.relu(bright + self.dt * (-stim - bright) / tau)

        # put brightness state back in
        state[0] = bright

        return bright, state
    
    def offline_predict(self, stim, model_params=None):
        tau = self.tau if model_params is None else torch.Tensor(model_params[0], device=self.device)




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
    *  Brightness is bounded in :math:`[\\theta, \\infty]`, where
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
    n_threads: int, optional
            Number of CPU threads to use during parallelization using OpenMP. Defaults to max number of user CPU cores.

    .. versionadded:: 0.7.1

    """

    def get_default_params(self):
        base_params = super(FadingTemporal, self).get_default_params()
        params = {
            # Time constant for the exponential decay:
            'tau': 100,
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
            raise ValueError(f"All times 't_percept' must be distinct multiples "
                             f"of `dt`={self.dt:.2e}")
        # Cython returns a 2D (space x time) NumPy array:
        return fading_fast(stim_data.astype(np.float32),
                           stim.time.astype(np.float32),
                           idx_percept, self.dt, self.tau, self.thresh_percept, self.n_threads)
