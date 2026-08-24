""":py:class:`~pulse2percept.models.FadingTemporal`,
:py:class:`~pulse2percept.models.AlphaTemporal`"""
import numpy as np
from .base import TemporalModel
from ..units import ms
from ._temporal import alpha_fast, fading_fast


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

        It cannot be shorter than ``dt``. The integrator steps explicitly, so
        a time constant of one step already carries brightness all the way to
        its drive; anything shorter overshoots and oscillates. ``tau`` also
        sets the *rise*, not just the decay, so raising it does not make a
        percept persist -- it makes it dimmer, as :math:`1/\\tau`.
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

    def get_param_units(self):
        """Return a dict of the units that parameters are stored in"""
        return {**super().get_param_units(), 'tau': ms}

    def _build(self):
        # Zero is as unusable as a negative value: the integrator divides by
        # `tau`, so it does not decay infinitely fast, it produces inf/nan.
        if self.tau <= 0:
            raise ValueError(f'"tau" must be positive, not {self.tau}.')
        # The integrator steps explicitly, so `dt / tau` is the fraction of the
        # remaining gap it closes per step. Above 1 it overshoots and then
        # oscillates, and the overshoot is `dt / tau` -- at tau=dt/4 brightness
        # alternates between four times its drive and zero, which is not a
        # leaky integrator in any useful sense:
        if self.tau < self.dt:
            raise ValueError(
                f'"tau" must be at least dt={self.dt}, not {self.tau}. A time '
                f'constant shorter than one simulation step makes the '
                f'integrator overshoot its drive by dt/tau and oscillate. '
                f'tau=dt is the fastest meaningful setting: brightness then '
                f'reaches its drive within one step, which makes the model a '
                f'half-wave rectifier. Shorten "dt" to go faster than that.')

    def _predict_temporal(self, stim, t_percept, reduce='last'):
        """Predict the temporal response"""
        # Pass the stimulus as a 2D NumPy array to the fast Cython function:
        time = self._stim_times(stim)
        stim_data = self._stim_values(stim).reshape((-1, len(time)))
        # Calculate at which simulation time steps we need to output a percept.
        # This is basically t_percept/self.dt, but we need to beware of
        # floating point rounding errors! 29.999 will be rounded down to 29 by
        # np.uint32, so we need to np.round it first:
        idx_percept = np.uint32(np.round(t_percept / self.dt))
        if np.unique(idx_percept).size < t_percept.size:
            raise ValueError(f"All times 't_percept' must be distinct multiples "
                             f"of `dt`={self.dt:.2e}")
        # Cython returns a 2D (space x time) NumPy array. `copy=False`: a
        # spatial stage upstream already hands this over as float32, and the
        # array is the whole space x time response, so copying it to the dtype
        # it is already in is the largest allocation in the call:
        return fading_fast(stim_data.astype(np.float32, copy=False),
                           time.astype(np.float32, copy=False),
                           idx_percept, self.dt, self.tau, self.thresh_percept,
                           self.n_threads, 1 if reduce == 'peak' else 0)


class AlphaTemporal(TemporalModel):
    """A generic alpha-shaped temporal model

    Two leaky integrators in series, both with the same time constant
    :math:`\\tau`, driven by the cathodic half of the stimulus:

    .. math::

        \\tau \\frac{dx}{dt} = \\max(-A, 0) - x

    .. math::

        \\tau \\frac{dy}{dt} = x - y

    where :math:`A` is the stimulus amplitude, :math:`x` is the (hidden) first
    stage, and :math:`y` is the perceived brightness.

    :py:class:`~pulse2percept.models.FadingTemporal` has an exponentially
    decaying impulse response. Here, cascading two identical leaky integrators
    adds a finite rise time: the impulse response starts at zero, peaks, and
    then decays. Its impulse response is

    .. math::

        h(t) = \\frac{t}{\\tau^2} e^{-t/\\tau},

    proportional to the standard alpha function, peaking at :math:`t = \\tau`.

    The model makes the following assumptions:

    *  Cathodic currents (negative amplitudes) increase perceived brightness
    *  Anodic currents (positive amplitudes) do not, and are ignored
    *  ``tau`` sets the rise and the decay together; there is only the one
       constant, so a later peak is also a slower decay
    *  Brightness is bounded below by zero. What is reported is then
       thresholded, so an output value is either 0 or at least
       :math:`\\theta` (``thresh_percept``, a nonnegative scalar)

    .. note::

        This is a generic alpha-shaped temporal response, not a perceptually
        validated model. Use it where a percept should build up over tens of
        milliseconds rather than appear instantaneously.

    Parameters
    ----------
    dt : float, optional
        Sampling time step of the simulation (ms)
    tau : float, optional
        Time constant of both leaky stages (ms). The impulse response peaks
        approximately ``tau`` milliseconds after an impulse, and decays with
        the same constant afterwards.

        It cannot be shorter than ``dt``, for the same reason as in
        :py:class:`~pulse2percept.models.FadingTemporal`: the stages step
        explicitly, so a time constant of one step already carries each stage
        all the way to its input, and anything shorter overshoots and
        oscillates.
    thresh_percept: float, optional
        Below threshold, the percept has brightness zero.
    reduce : {'peak', 'last'}, optional
        How a percept time point summarizes the interval since the previous
        one, when ``predict_percept`` chooses the output times itself; see
        :py:class:`~pulse2percept.models.TemporalModel`. This model tracks the
        peak inside the cascade, so it is exact at any output rate.
    n_threads: int, optional
        Number of CPU threads to use during parallelization using OpenMP.
        Defaults to max number of user CPU cores.

    .. versionadded:: 0.10.0

    """

    #: The peak is tracked across every `dt` step inside `alpha_fast`, so
    #: `predict_percept` does not have to subsample the interval itself.
    _reduces_intervals = True

    def get_default_params(self):
        base_params = super(AlphaTemporal, self).get_default_params()
        params = {
            # Time constant of both stages:
            'tau': 100,
            # Generic rather than a published fit, so it is free to report the
            # more useful summary by default; see `FadingTemporal`:
            'reduce': 'peak',
        }
        base_params.update(params)
        return base_params

    def get_param_units(self):
        """Return a dict of the units that parameters are stored in"""
        return {**super().get_param_units(), 'tau': ms}

    def _build(self):
        if self.tau <= 0:
            raise ValueError(f'"tau" must be positive, not {self.tau}.')
        # Both stages step explicitly, so `dt / tau` is the fraction of the
        # remaining gap each closes per step. Above 1 they overshoot and
        # oscillate; see `FadingTemporal._build`:
        if self.tau < self.dt:
            raise ValueError(
                f'"tau" must be at least dt={self.dt}, not {self.tau}. A time '
                f'constant shorter than one simulation step makes each stage '
                f'overshoot its input by dt/tau and oscillate. Shorten "dt" '
                f'to go faster than that.')

    def _predict_temporal(self, stim, t_percept, reduce='last'):
        """Predict the temporal response"""
        # Pass the stimulus as a 2D NumPy array to the fast Cython function:
        time = self._stim_times(stim)
        stim_data = self._stim_values(stim).reshape((-1, len(time)))
        # Round before casting: 29.999 would otherwise truncate to 29.
        idx_percept = np.uint32(np.round(t_percept / self.dt))
        if np.unique(idx_percept).size < t_percept.size:
            raise ValueError(f"All times 't_percept' must be distinct "
                             f"multiples of `dt`={self.dt:.2e}")
        # Cython returns a 2D (space x time) NumPy array:
        return alpha_fast(stim_data.astype(np.float32),
                          time.astype(np.float32),
                          idx_percept, self.dt, self.tau, self.thresh_percept,
                          self.n_threads, 1 if reduce == 'peak' else 0)
