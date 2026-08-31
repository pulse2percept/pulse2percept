""":py:class:`~pulse2percept.models.Horsager2009Model`,
   :py:class:`~pulse2percept.models.Horsager2009Temporal` [Horsager2009]_"""
import numpy as np
from .base import Model, TemporalModel, _thread_params
from ..units import ms
from ._horsager2009 import temporal_fast


class Horsager2009Temporal(TemporalModel):
    r"""Temporal model of [Horsager2009]_.

    Implements the linear-nonlinear cascade from Fig. 2 of [Horsager2009]_.
    With stimulus current :math:`A(t)`, the fast pathway and charge
    accumulation are

    .. math::

        \tau_1 \frac{dR_1}{dt} &= -A(t) - R_1(t), \\

        \frac{dC}{dt} &= \max[A(t), 0], \\

        \tau_2 \frac{dR_2}{dt} &= C(t) - R_2(t).

    Thus negative current drives the fast response, while positive current
    contributes to accumulated charge. The two pathways combine through a
    rectifying power nonlinearity,

    .. math::

        R_3(t) =
        \left[
            \max\left(
                R_1(t) - \epsilon_{\mathrm{ms}} R_2(t), 0
            \right)
        \right]^\beta,

    where :math:`\epsilon_{\mathrm{ms}} = \epsilon / 1000` because p2p
    integrates time in milliseconds while the original parameterization used
    microseconds.

    The result passes through three identical slow leaky integrators,

    .. math::

        \tau_3 \frac{dR_{4a}}{dt} &= R_3 - R_{4a}, \\

        \tau_3 \frac{dR_{4b}}{dt} &= R_{4a} - R_{4b}, \\

        \tau_3 \frac{dB}{dt} &= R_{4b} - B,

    and :math:`B(t)` is the predicted brightness.

    Use this class to combine the temporal model with a spatial model. Use
    :py:class:`~pulse2percept.models.Horsager2009Model` for the standalone
    temporal model.

    Parameters
    ----------
    dt : float or Quantity, optional
        Simulation time step, in milliseconds. Default: 0.005 ms.
    tau1 : float or Quantity, optional
        Time constant of the fast response :math:`R_1`, in milliseconds.
        Default: 0.42 ms.
    tau2 : float or Quantity, optional
        Time constant of the filtered charge accumulation :math:`R_2`, in
        milliseconds. Default: 45.25 ms.
    tau3 : float or Quantity, optional
        Time constant of each of the three final leaky-integrator stages, in
        milliseconds. Default: 26.25 ms.
    eps : float, optional
        Strength of the subtractive charge-accumulation pathway. The public
        value retains the original microsecond parameterization and is divided
        by 1000 internally for millisecond integration. Default: 2.25.
        [Horsager2009]_ also reports 8.73 for the suprathreshold fit.
    beta : float, optional
        Exponent of the rectifying power nonlinearity. Default: 3.43.
        [Horsager2009]_ also reports 0.83 for the suprathreshold fit.
    thresh_percept : float, optional
        Brightness values below this threshold are set to zero. Default: 0.
    reduce : {'peak', 'last'}, optional
        How automatically chosen output points summarize the preceding
        interval. ``'last'`` reports brightness at the output instant;
        ``'peak'`` approximates the interval peak by subsampling. Explicit
        ``t_percept`` values always request those instants. Default:
        ``'last'``.
    verbose : bool, optional
        Whether to print status messages. Default: True.
    n_threads : int, optional
        Number of OpenMP threads. Defaults to all available CPU cores.
    n_jobs : int or None, optional
        Alias for ``n_threads``. ``None`` and -1 use all available CPU cores.
    """

    def __init__(self, *, dt=0.005, tau1=0.42, tau2=45.25, tau3=26.25,
                 eps=2.25, beta=3.43, thresh_percept=0, reduce='last',
                 verbose=True, n_threads=None, n_jobs=None):
        super().__init__(dt=dt, tau1=tau1, tau2=tau2, tau3=tau3, eps=eps,
                         beta=beta, thresh_percept=thresh_percept,
                         reduce=reduce, verbose=verbose,
                         **_thread_params(n_threads, n_jobs))

    def get_default_params(self):
        base_params = super(Horsager2009Temporal, self).get_default_params()
        params = {
            'tau1': 0.42,
            'tau2': 45.25,
            'tau3': 26.25,
            'eps': 2.25,
            'beta': 3.43
        }
        base_params.update(params)
        return base_params

    def get_param_units(self):
        """Return units used to store model parameters."""
        return {**super().get_param_units(), 'tau1': ms, 'tau2': ms,
                'tau3': ms}

    def _predict_temporal(self, stim, t_percept):
        """Predict the temporal response."""
        time = self._stim_times(stim)
        stim_data = self._stim_values(stim).reshape((-1, len(time)))
        # Round before casting so floating-point noise cannot shift a sample.
        idx_percept = np.uint32(np.round(t_percept / self.dt))
        if np.unique(idx_percept).size < t_percept.size:
            raise ValueError(f"All times 't_percept' must be distinct multiples "
                             f"of `dt`={self.dt:.2e}")
        return temporal_fast(stim_data.astype(np.float32),
                             time.astype(np.float32),
                             idx_percept,
                             self.dt, self.tau1, self.tau2, self.tau3,
                             self.eps, self.beta, self.thresh_percept, self.n_threads)


class Horsager2009Model(Model):
    """Standalone temporal model of [Horsager2009]_.

    Uses :py:class:`~pulse2percept.models.Horsager2009Temporal` without a
    spatial component. See that class for the model equations. Use
    ``Horsager2009Temporal`` instead when combining the temporal cascade with
    a spatial model.

    Parameters
    ----------
    dt : float or Quantity, optional
        Simulation time step, in milliseconds. Default: 0.005 ms.
    tau1 : float or Quantity, optional
        Time constant of the fast response, in milliseconds. Default:
        0.42 ms.
    tau2 : float or Quantity, optional
        Time constant of the filtered charge accumulation, in milliseconds.
        Default: 45.25 ms.
    tau3 : float or Quantity, optional
        Time constant of each final leaky-integrator stage, in milliseconds.
        Default: 26.25 ms.
    eps : float, optional
        Strength of the subtractive charge-accumulation pathway. Default:
        2.25. [Horsager2009]_ also reports 8.73 for the suprathreshold fit.
    beta : float, optional
        Exponent of the rectifying power nonlinearity. Default: 3.43.
        [Horsager2009]_ also reports 0.83 for the suprathreshold fit.
    thresh_percept : float, optional
        Brightness values below this threshold are set to zero. Default: 0.
    reduce : {'peak', 'last'}, optional
        How automatically chosen output points summarize the preceding
        interval. Default: ``'last'``.
    verbose : bool, optional
        Whether to print status messages. Default: True.
    n_threads : int, optional
        Number of OpenMP threads. Defaults to all available CPU cores.
    n_jobs : int or None, optional
        Alias for ``n_threads``. ``None`` and -1 use all available CPU cores.
    """

    def __init__(self, *, dt=0.005, tau1=0.42, tau2=45.25, tau3=26.25,
                 eps=2.25, beta=3.43, thresh_percept=0, reduce='last',
                 verbose=True, n_threads=None, n_jobs=None):
        super().__init__(
            spatial=None,
            temporal=Horsager2009Temporal(
                dt=dt, tau1=tau1, tau2=tau2, tau3=tau3, eps=eps, beta=beta,
                thresh_percept=thresh_percept, reduce=reduce, verbose=verbose,
                n_threads=n_threads, n_jobs=n_jobs))
