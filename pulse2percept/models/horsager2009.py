"""This module implements the model described in [#horsager-2009]_.

References
----------
.. [#horsager-2009] A Horsager, SH Greenwald, JD Weiland, MS Humayun, RJ
                    Greenberg, MJ McMahon, GM Boynton, I Fine (2009).
                    Predicting visual sensitivity in retinal prosthesis
                    patients. Investigative Ophthalmology & Visual Science
                    50(4): 1483, doi:`10.1167/iovs.08-2595 <https://doi.org/10.1167/iovs.08-2595>`_.
"""
import numpy as np
from .base import BaseModel
from .scoreboard import ScoreboardModel
from ._horsager2009 import step_horsager2009


class Horsager2009TemporalMixin(object):
    """Horsager temporal mixin

    Parameters
    ----------
    tau1: float, optional, default: 0.42 / 1000 seconds
        Time decay constant for the fast leaky integrater of the ganglion
        cell layer(GCL).
    tau2: float, optional, default: 45.25 / 1000 seconds
        Time decay constant for the charge accumulation, has values
        between 38 - 57 ms.
    tau3: float, optional, default: 26.25 / 1000 seconds
        Time decay constant for the slow leaky integrator.
        Default: 26.25 / 1000 s.
    epsilon: float, optional, default: 8.73
        Scaling factor applied to charge accumulation.
    beta: float, optional, default: 3.43
        Power nonlinearity applied after half - rectification. The original model
        used two different values, depending on whether an experiment is at
        threshold(`beta`=3.43) or above threshold(`beta`=0.83).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ca = None
        self.r1 = None
        self.r2 = None
        self.r4a = None
        self.r4b = None
        self.r4c = None
        self.reset_state()

    def reset_state(self):
        n_px = np.prod(self.grid.shape)
        self.ca = np.zeros(n_px, dtype=np.double)
        self.r1 = np.zeros(n_px, dtype=np.double)
        self.r2 = np.zeros(n_px, dtype=np.double)
        self.r4a = np.zeros(n_px, dtype=np.double)
        self.r4b = np.zeros(n_px, dtype=np.double)
        self.r4c = np.zeros(n_px, dtype=np.double)

    def _get_default_params(self):
        base_params = super()._get_default_params()
        params = {
            'has_time': True,
            'thresh_percept': 0,
            # Simulation time step:
            'dt': 0.005 / 1000,
            # Time decay for the ganglion cell impulse response:
            'tau1': 0.42 / 1000,
            # Time decay for the charge accumulation:
            'tau2': 45.25 / 1000,
            # Time decay for the slow leaky integrator:
            'tau3': 26.25 / 1000,
            # Scaling factor applied to charge accumulation:
            'epsilon': 2.25,
            # Power nonlinearity applied after half-rectification:
            'beta': 3.43
        }
        # This is subtle: Rather than calling `params.update(base_params)`, we
        # call `base_params.update(params)`. This will overwrite `base_params`
        # with values from `params`, which allows us to set `thresh_percept`=0
        # rather than what the BaseModel dictates:
        base_params.update(params)
        return base_params

    def _step_temporal_model(self, stim, dt):
        """Steps the temporal model"""
        i, amp = stim
        ca, r1, r2, r4a, r4b, r4c = step_horsager2009(
            dt, amp, self.ca[i], self.r1[i], self.r2[i],
            self.r4a[i], self.r4b[i], self.r4c[i],
            self.tau1, self.tau2, self.tau3, self.epsilon, self.beta
        )
        self.ca[i] = ca
        self.r1[i] = r1
        self.r2[i] = r2
        self.r4a[i] = r4a
        self.r4b[i] = r4b
        self.r4c[i] = r4c
        return r4c

        _, self.gamma1 = utils.gamma(1, self.tau1, self.tsample)
        _, self.gamma2 = utils.gamma(1, self.tau2, self.tsample)
        _, self.gamma3 = utils.gamma(3, self.tau3, self.tsample)

    def calc_layer_current(self, in_arr, pt_list, layers):
        """Calculates the effective current map of a given layer
        Parameters
        ----------
        in_arr: array - like
            A 2D array specifying the effective current values
            at a particular spatial location(pixel); one value
            per retinal layer and electrode.
            Dimensions: <  # layers x #electrodes>
        pt_list: list
            List of pulse train 'data' containers.
            Dimensions: <  # electrodes x #time points>
        layers: list
            List of retinal layers to simulate.
            Choose from:
            - 'OFL': optic fiber layer
            - 'GCL': ganglion cell layer
        """
        if 'INL' in layers:
            raise ValueError("The Horsager2009 model does not support an "
                             "inner nuclear layer.")

        if ('GCL' or 'OFL') in layers:
            ecm = np.sum(in_arr[1, :, np.newaxis] * pt_list, axis=0)
        else:
            raise ValueError("Acceptable values for `layers` are: 'GCL', "
                             "'OFL'.")
        return ecm

    def model_cascade(self, in_arr, pt_list, layers, use_jit):
        """Horsager model cascade
        Parameters
        ----------
        in_arr: array - like
            A 2D array specifying the effective current values
            at a particular spatial location(pixel); one value
            per retinal layer and electrode.
            Dimensions: <  # layers x #electrodes>
        pt_list: list
            List of pulse train 'data' containers.
            Dimensions: <  # electrodes x #time points>
        layers: list
            List of retinal layers to simulate.
            Choose from:
            - 'OFL': optic fiber layer
            - 'GCL': ganglion cell layer
        use_jit: bool
            If True, applies just - in-time(JIT) compilation to
            expensive computations for additional speed - up
            (requires Numba).
        """
        if 'INL' in layers:
            raise ValueError("The Nanduri2012 model does not support an inner "
                             "nuclear layer.")

        # Although the paper says to use cathodic-first, the code only
        # reproduces if we use what we now call anodic-first. So flip the sign
        # on the stimulus here:
        stim = -self.calc_layer_current(in_arr, pt_list, layers)

        # R1 convolved the entire stimulus (with both pos + neg parts)
        r1 = self.tsample * utils.conv(stim, self.gamma1, mode='full',
                                       method='sparse')[:stim.size]

        # It's possible that charge accumulation was done on the anodic phase.
        # It might not matter too much (timing is slightly different, but the
        # data are not accurate enough to warrant using one over the other).
        # Thus use what makes the most sense: accumulate on cathodic
        ca = self.tsample * np.cumsum(np.maximum(0, -stim))
        ca = self.tsample * utils.conv(ca, self.gamma2, mode='full',
                                       method='fft')[:stim.size]
        r2 = r1 - self.epsilon * ca

        # Then half-rectify and pass through the power-nonlinearity
        r3 = np.maximum(0.0, r2) ** self.beta

        # Then convolve with slow gamma
        r4 = self.tsample * utils.conv(r3, self.gamma3, mode='full',
                                       method='fft')[:stim.size]

        return utils.TimeSeries(self.tsample, r4)


class Horsager2009Model(Horsager2009TemporalMixin, ScoreboardModel):
    pass
