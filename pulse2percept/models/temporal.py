""":py:class:`~pulse2percept.models.FadingTemporal`,
:py:class:`~pulse2percept.models.AlphaTemporal`"""
import numpy as np
from .base import TemporalModel
from ..units import ms
from ._temporal import alpha_fast, fading_fast


class FadingTemporal(TemporalModel):
    r"""Generic temporal model for phosphene fading.

    Cathodic current is half-wave rectified into the drive

    .. math::

        D(t) = \max[-A(t), 0],

    where :math:`A(t)` is stimulus amplitude. Brightness follows a first-order
    leaky integrator,

    .. math::

        \tau \frac{dB}{dt} = D(t) - B(t).

    For constant drive :math:`D`, the continuous-time response is

    .. math::

        B(t) = D + [B(0) - D] e^{-t/\tau}.

    Thus :math:`\tau` controls both rise and decay. Larger values produce
    slower responses and lower peaks for brief pulses. Anodic current does not
    drive brightness.

    The model is evaluated with the explicit-Euler recurrence

    .. math::

        B_{k+1} =
        B_k + \frac{\Delta t}{\tau}\left(D_k - B_k\right),

    with :math:`\Delta t =` ``dt``. The implementation requires
    :math:`\tau \geq \Delta t`, so the discrete-time pole
    :math:`1-\Delta t/\tau` is nonnegative.

    This is a generic temporal response model, not a perceptually validated
    fit.

    Parameters
    ----------
    dt : float or Quantity, optional
        Simulation time step, in milliseconds. Default: 0.005 ms.
    tau : float or Quantity, optional
        Leaky-integrator time constant, in milliseconds. Larger values slow
        both rise and decay and reduce the peak response to brief pulses.
        Must be at least ``dt``. Default: 100 ms.
    thresh_percept : float, optional
        Brightness values below this threshold are set to zero. Default: 0.
    reduce : {'peak', 'last'}, optional
        How automatically chosen output points summarize the preceding
        interval. ``'peak'`` reports the maximum brightness reached;
        ``'last'`` reports brightness at the output instant. Explicit
        ``t_percept`` values always request those instants. Default:
        ``'peak'``.
    verbose : bool, optional
        Whether to print status messages. Default: True.
    n_threads : int, optional
        Number of OpenMP threads. Defaults to all available CPU cores.
    n_jobs : int or None, optional
        Alias for ``n_threads``. ``None`` and -1 use all available CPU cores.

    .. versionchanged:: 0.10.0

        The drive is half-wave rectified, so only cathodic current increases
        brightness.

    .. versionadded:: 0.7.1
    """

    #: The kernel tracks interval peaks directly.
    _reduces_intervals = True

    def get_default_params(self):
        base_params = super(FadingTemporal, self).get_default_params()
        params = {
            'tau': 100,
            'reduce': 'peak',
        }
        base_params.update(params)
        return base_params

    def get_param_units(self):
        """Return units used to store model parameters."""
        return {**super().get_param_units(), 'tau': ms}

    def _build(self):
        if self.tau <= 0:
            raise ValueError(f'"tau" must be positive, not {self.tau}.')
        # Require a nonnegative explicit-Euler pole, 1 - dt / tau.
        if self.tau < self.dt:
            raise ValueError(
                f'"tau" must be at least dt={self.dt}, not {self.tau}. A time '
                f'constant shorter than one simulation step makes the '
                f'integrator overshoot its drive by dt/tau and oscillate. '
                f'tau=dt is the fastest meaningful setting: brightness then '
                f'reaches its drive within one step, which makes the model a '
                f'half-wave rectifier. Shorten "dt" to go faster than that.')

    def _predict_temporal(self, stim, t_percept, reduce='last'):
        """Predict the temporal response."""
        time = self._stim_times(stim)
        stim_data = self._stim_values(stim).reshape((-1, len(time)))
        # Round before casting so floating-point noise cannot shift a sample.
        idx_percept = np.uint32(np.round(t_percept / self.dt))
        if np.unique(idx_percept).size < t_percept.size:
            raise ValueError(f"All times 't_percept' must be distinct multiples "
                             f"of `dt`={self.dt:.2e}")
        return fading_fast(stim_data.astype(np.float32, copy=False),
                           time.astype(np.float32, copy=False),
                           idx_percept, self.dt, self.tau, self.thresh_percept,
                           self.n_threads, 1 if reduce == 'peak' else 0)


class AlphaTemporal(TemporalModel):
    r"""Generic alpha-shaped temporal model.

    Cathodic current is half-wave rectified into

    .. math::

        D(t) = \max[-A(t), 0].

    Two identical first-order stages are then cascaded:

    .. math::

        \tau \frac{dx}{dt} = D(t) - x(t),

    .. math::

        \tau \frac{dB}{dt} = x(t) - B(t).

    For a unit-area impulse, the continuous-time impulse response is

    .. math::

        h(t) = \frac{t}{\tau^2} e^{-t/\tau},
        \qquad t \geq 0,

    which rises from zero, peaks at :math:`t=\tau`, and then decays. Anodic
    current does not drive brightness.

    The model is evaluated with the explicit-Euler recurrences

    .. math::

        x_{k+1} =
        x_k + \frac{\Delta t}{\tau}\left(D_k - x_k\right),

    .. math::

        B_{k+1} =
        B_k + \frac{\Delta t}{\tau}\left(x_k - B_k\right),

    with :math:`\Delta t =` ``dt``. The implementation requires
    :math:`\tau \geq \Delta t`.

    This is a generic temporal response model, not a perceptually validated
    fit.

    Parameters
    ----------
    dt : float or Quantity, optional
        Simulation time step, in milliseconds. Default: 0.005 ms.
    tau : float or Quantity, optional
        Time constant of both stages, in milliseconds. Larger values delay and
        broaden the response. Must be at least ``dt``. Default: 100 ms.
    thresh_percept : float, optional
        Brightness values below this threshold are set to zero. Default: 0.
    reduce : {'peak', 'last'}, optional
        How automatically chosen output points summarize the preceding
        interval. ``'peak'`` reports the maximum brightness reached;
        ``'last'`` reports brightness at the output instant. Explicit
        ``t_percept`` values always request those instants. Default:
        ``'peak'``.
    verbose : bool, optional
        Whether to print status messages. Default: True.
    n_threads : int, optional
        Number of OpenMP threads. Defaults to all available CPU cores.
    n_jobs : int or None, optional
        Alias for ``n_threads``. ``None`` and -1 use all available CPU cores.

    .. versionadded:: 0.10.0
    """

    #: The kernel tracks interval peaks directly.
    _reduces_intervals = True

    def get_default_params(self):
        base_params = super(AlphaTemporal, self).get_default_params()
        params = {
            'tau': 100,
            'reduce': 'peak',
        }
        base_params.update(params)
        return base_params

    def get_param_units(self):
        """Return units used to store model parameters."""
        return {**super().get_param_units(), 'tau': ms}

    def _build(self):
        if self.tau <= 0:
            raise ValueError(f'"tau" must be positive, not {self.tau}.')
        # Require a nonnegative explicit-Euler pole in each stage.
        if self.tau < self.dt:
            raise ValueError(
                f'"tau" must be at least dt={self.dt}, not {self.tau}. A time '
                f'constant shorter than one simulation step makes each stage '
                f'overshoot its input by dt/tau and oscillate. Shorten "dt" '
                f'to go faster than that.')

    def _predict_temporal(self, stim, t_percept, reduce='last'):
        """Predict the temporal response."""
        time = self._stim_times(stim)
        stim_data = self._stim_values(stim).reshape((-1, len(time)))
        # Round before casting so floating-point noise cannot shift a sample.
        idx_percept = np.uint32(np.round(t_percept / self.dt))
        if np.unique(idx_percept).size < t_percept.size:
            raise ValueError(f"All times 't_percept' must be distinct "
                             f"multiples of `dt`={self.dt:.2e}")
        return alpha_fast(stim_data.astype(np.float32, copy=False),
                          time.astype(np.float32, copy=False),
                          idx_percept, self.dt, self.tau, self.thresh_percept,
                          self.n_threads, 1 if reduce == 'peak' else 0)
