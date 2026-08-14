""":py:class:`~pulse2percept.models.FadingTemporal`"""
import numpy as np
from .base import TemporalModel
from ._temporal import fading_fast


class FadingTemporal(TemporalModel):
    """A generic temporal model for phosphene fading

    Implements phosphene fading using a leaky integrator driven by the
    cathodic half of the stimulus:

    .. math::

        \\frac{dB}{dt} = \\frac{\\max(-A, 0) - B}{\\tau}

    where :math:`A` is the stimulus amplitude, :math:`B` is the perceived
    brightness, and :math:`\\tau` is the exponential decay constant (``tau``).

    The model makes the following assumptions:

    *  Cathodic currents (negative amplitudes) increase perceived brightness
    *  Anodic currents (positive amplitudes) do not, and are ignored
    *  Brightness is bounded below by zero. What is reported is then
       thresholded, so an output value is either 0 or at least
       :math:`\\theta` (``thresh_percept``, a nonnegative scalar)

    .. versionchanged:: 0.10.0

        The drive is now half-wave rectified, driven by the cathodic phase.
        A stimulus that is purely cathodic is unaffected.

    .. note::

        This is the simplest sensical temporal model, not a perceptually
        validated model of phosphene fading.

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
    reduce : {'peak', 'last'}, optional
        How a percept time point summarizes the interval since the previous
        one, when ``predict_percept`` chooses the output times itself; see
        :py:class:`~pulse2percept.models.TemporalModel`. This model tracks the
        peak inside the integrator, so it is exact at any output rate.
    n_threads: int, optional
            Number of CPU threads to use during parallelization using OpenMP. 
            Defaults to max number of user CPU cores.

    .. versionadded:: 0.7.1

    """

    #: The peak is tracked across every `dt` step inside `fading_fast`, so
    #: `predict_percept` does not have to subsample the interval itself.
    _reduces_intervals = True

    def get_default_params(self):
        base_params = super(FadingTemporal, self).get_default_params()
        params = {
            # Time constant for the exponential decay:
            'tau': 100,
            # This model is generic rather than a published fit, so it is free
            # to report the more useful summary by default. The peak is tracked
            # inside `fading_fast`, so it costs nothing and is exact:
            'reduce': 'peak',
        }
        # This is subtle: Rather than calling `params.update(base_params)`, we
        # call `base_params.update(params)`. This will overwrite `base_params`
        # with values from `params`, which allows us to set `thresh_percept`=0
        # rather than what the BaseModel dictates:
        base_params.update(params)
        return base_params

    def _build(self):
        # Zero is as unusable as a negative value: the integrator divides by
        # `tau`, so it does not decay infinitely fast, it produces inf/nan.
        if self.tau <= 0:
            raise ValueError(f'"tau" must be positive, not {self.tau}.')

    def _predict_temporal(self, stim, t_percept, reduce='last'):
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
                           idx_percept, self.dt, self.tau, self.thresh_percept,
                           self.n_threads, 1 if reduce == 'peak' else 0)
