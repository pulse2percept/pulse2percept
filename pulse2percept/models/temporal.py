""":py:class:`~pulse2percept.models.FadingTemporal`,
:py:class:`~pulse2percept.models.AlphaTemporal`"""
import numpy as np
from .base import TemporalModel
from ..units import ms
from ._temporal import alpha_fast, fading_fast


class FadingTemporal(TemporalModel):
    r"""Generic temporal model for phosphene fading.

    The model is a leaky integrator driven by cathodic current:

    .. math::

        \frac{dB}{dt} = \frac{\max(-A, 0) - B}{\tau}

    where :math:`A` is stimulus amplitude, :math:`B` is brightness,
    and :math:`\tau` is the time constant. Anodic current is ignored.

    This is a generic response model, not a perceptually validated fit.

    .. versionchanged:: 0.10.0

        The drive is half-wave rectified, so only cathodic current
        increases brightness.

    Parameters
    ----------
    dt : float, optional
        Simulation time step (ms).
    tau : float, optional
        Leaky-integrator time constant (ms). Larger values slow both
        rise and decay; for brief pulses they also reduce peak
        brightness. Must be at least ``dt``.
    thresh_percept : float, optional
        Brightness values below threshold are set to zero.
    reduce : {'peak', 'last'}, optional
        How each output point summarizes the preceding interval.
    n_threads : int, optional
        Number of OpenMP threads.

    .. versionadded:: 0.7.1
    """

    #: The kernel tracks interval peaks directly.
    _reduces_intervals = True

    def get_default_params(self):
        base_params = super(FadingTemporal, self).get_default_params()
        params = {
            # Time constant for the exponential decay:
            'tau': 100,
            # Report the exact interval peak by default.
            'reduce': 'peak',
        }
        # Override base defaults with this model's defaults.
        base_params.update(params)
        return base_params

    def get_param_units(self):
        """Return a dict of the units that parameters are stored in"""
        return {**super().get_param_units(), 'tau': ms}

    def _build(self):
        # The integrator divides by ``tau``.
        if self.tau <= 0:
            raise ValueError(f'"tau" must be positive, not {self.tau}.')
        # Explicit Euler is unstable here when ``tau < dt``.
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
        # Round before casting so floating-point noise cannot shift a sample.
        idx_percept = np.uint32(np.round(t_percept / self.dt))
        if np.unique(idx_percept).size < t_percept.size:
            raise ValueError(f"All times 't_percept' must be distinct multiples "
                             f"of `dt`={self.dt:.2e}")
        # Cython returns a 2D (space x time) NumPy array. `copy=False`:
        # avoid copying an already-float32 spatial response.
        return fading_fast(stim_data.astype(np.float32, copy=False),
                           time.astype(np.float32, copy=False),
                           idx_percept, self.dt, self.tau, self.thresh_percept,
                           self.n_threads, 1 if reduce == 'peak' else 0)


class AlphaTemporal(TemporalModel):
    r"""Generic alpha-shaped temporal model.

    Two identical leaky integrators are cascaded:

    .. math::

        \tau \frac{dx}{dt} = \max(-A, 0) - x

    .. math::

        \tau \frac{dy}{dt} = x - y

    Their impulse response is proportional to

    .. math::

        h(t) = \frac{t}{\tau^2} e^{-t/\tau},

    which rises from zero, peaks at :math:`t=\tau`, and then decays.
    Cathodic current drives the response; anodic current is ignored.

    This is a generic response model, not a perceptually validated fit.

    Parameters
    ----------
    dt : float, optional
        Simulation time step (ms).
    tau : float, optional
        Time constant of both stages (ms). Must be at least ``dt``.
    thresh_percept : float, optional
        Brightness values below threshold are set to zero.
    reduce : {'peak', 'last'}, optional
        How each output point summarizes the preceding interval.
    n_threads : int, optional
        Number of OpenMP threads.

    .. versionadded:: 0.10.0
    """

    #: The kernel tracks interval peaks directly.
    _reduces_intervals = True

    def get_default_params(self):
        base_params = super(AlphaTemporal, self).get_default_params()
        params = {
            # Time constant of both stages:
            'tau': 100,
            # Report the exact interval peak by default.
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
        # Explicit Euler is unstable here when ``tau < dt``.
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
        # Cython returns a 2D (space x time) NumPy array. `copy=False`:
        # avoid copying an already-float32 spatial response.
        return alpha_fast(stim_data.astype(np.float32, copy=False),
                          time.astype(np.float32, copy=False),
                          idx_percept, self.dt, self.tau, self.thresh_percept,
                          self.n_threads, 1 if reduce == 'peak' else 0)
