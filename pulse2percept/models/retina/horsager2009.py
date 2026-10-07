""":py:class:`~pulse2percept.models.retina.Horsager2009Model`,
   :py:class:`~pulse2percept.models.retina.Horsager2009Temporal` [Horsager2009]_"""
import numpy as np
from ..base import Model, TemporalModel
from ..temporal import _charge_chunks, _flush_denormals, _on_torch
from ...units import ms


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
    :py:class:`~pulse2percept.models.retina.Horsager2009Model` for the standalone
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

    .. versionchanged:: 0.12.0

        Runs on Torch; ``n_threads`` and ``n_jobs`` were removed.
    """

    def __init__(self, *, dt=0.005, tau1=0.42, tau2=45.25, tau3=26.25,
                 eps=2.25, beta=3.43, thresh_percept=0, reduce='last',
                 verbose=True):
        super().__init__(dt=dt, tau1=tau1, tau2=tau2, tau3=tau3, eps=eps,
                         beta=beta, thresh_percept=thresh_percept,
                         reduce=reduce, verbose=verbose)

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
        # Torch runs on its own thread pool:
        del base_params['n_threads'], base_params['n_jobs']
        return base_params

    def get_param_units(self):
        """Return units used to store model parameters."""
        return {**super().get_param_units(), 'tau1': ms, 'tau2': ms,
                'tau3': ms}

    def _predict_temporal(self, stim, t_percept):
        """Predict the float32 NumPy temporal response."""
        return _on_torch(self._predict_horsager, self._stim_values(stim),
                         self._stim_times(stim), t_percept)

    def _predict_temporal_tensor(self, stim, t_percept, reduce='last'):
        """Predict the temporal response to tensor data, keeping its dtype
        and autograd graph. ``reduce`` is applied by the caller."""
        return self._predict_horsager(stim.data, self._stim_times(stim),
                                      t_percept)

    @_flush_denormals()
    def _predict_horsager(self, data, time, t_percept):
        """Return the Torch response to ``data`` sampled at ``time``.

        Each constant-stimulus chunk is one closed-form step of the fast
        stage and one matrix product of the slow cascade. Timing is fixed.
        """
        import torch
        beta = float(np.float32(self.beta))
        # Not simply zero: 0**beta is 1 for beta == 0 and inf for beta < 0:
        with np.errstate(divide='ignore'):
            zero_pow = float(np.float32(0) ** np.float32(beta))
        # Time-major, as in `_charge_chunks`:
        state = data.new_zeros((3, data.reshape((-1, len(time))).shape[0]))
        rows = []
        # Rectified steps can be skipped only if they contribute nothing:
        for x, carry, weights, offs in _charge_chunks(
                self, data, time, t_percept, -1, skip_quiet=zero_pow == 0):
            if x is None:
                state = carry @ state
                rows.append(state[:len(offs)])
                state = state[len(offs):]
                continue
            pos = x > 0
            # The inner `where` keeps NaN out of the gradient of rectified
            # steps:
            r3 = torch.where(pos, torch.where(pos, x, 1.0) ** beta, zero_pow)
            if beta < 0:
                resp = _inf_steps(state, r3, carry, weights, offs)
            else:
                resp = torch.addmm(carry @ state, weights, r3)
            state = resp[len(offs):]
            rows.append(resp[:len(offs)])
        resp = torch.cat(rows).T.contiguous()
        # Thresholds the output only, not the state:
        thresh = float(np.float32(self.thresh_percept))
        return torch.where(resp.abs() >= thresh, resp, 0.0)


def _inf_steps(state, r3, carry, weights, offs):
    """Return the time-major slow cascade of ``r3``, which may hold inf.

    Explicit-Euler steps make every stage inf at the first inf input and NaN
    (inf - inf) from the next step on; a non-finite state is NaN after one
    step.
    """
    import torch
    inf = torch.isinf(r3)
    n = r3.shape[0]
    # 1-based step of the first inf input, n + 1 if none:
    first = torch.where(inf.any(dim=0), inf.to(r3.dtype).argmax(dim=0) + 1,
                        n + 1)
    finite = torch.isfinite(state)
    first = torch.where(finite.all(dim=0), first, 0)
    # inf * 0 is NaN, so the matrix product sees zeros instead:
    resp = torch.addmm(carry @ torch.where(finite, state, 0.0), weights,
                       torch.where(inf, 0.0, r3))
    step = torch.tensor(list(offs) + [n] * 3, device=r3.device)[:, None]
    return torch.where(step > first, torch.nan,
                       torch.where(step == first, torch.inf, resp))


class Horsager2009Model(Model):
    """Standalone temporal model of [Horsager2009]_.

    Uses :py:class:`~pulse2percept.models.retina.Horsager2009Temporal` without a
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

    .. versionchanged:: 0.12.0

        Runs on Torch; ``n_threads`` and ``n_jobs`` were removed.
    """

    def __init__(self, *, dt=0.005, tau1=0.42, tau2=45.25, tau3=26.25,
                 eps=2.25, beta=3.43, thresh_percept=0, reduce='last',
                 verbose=True):
        super().__init__(
            spatial=None,
            temporal=Horsager2009Temporal(
                dt=dt, tau1=tau1, tau2=tau2, tau3=tau3, eps=eps, beta=beta,
                thresh_percept=thresh_percept, reduce=reduce,
                verbose=verbose))
