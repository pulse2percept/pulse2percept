"""This modules implements the model described in [#nanduri-2012]_.

References
----------
.. [#nanduri-2012] D Nanduri, I Fine, A Horsager, GM Boynton, MS Humayun,
                   RJ Greenberg, JD Weiland (2012), Frequency and amplitude
                   modulation have different effects on the percepts elicited
                   by retinal stimulation. Investigative Ophthalmology & Visual
                   Science 53:205-214, doi:`10.1167/iovs.11-8401 <https://doi.org/10.1167/iovs.11-8401>`_.
"""
import numpy as np
from .base import BaseModel
from .scoreboard import ScoreboardModel
from .watson2014 import Watson2014ConversionMixin
from ._nanduri2012 import fast_nanduri2012


class Nanduri2012SpatialMixin(object):
    """Nanduri spatial mixin"""

    def _get_default_params(self):
        """Returns all settable parameters of the scoreboard model"""
        params = super()._get_default_params()
        params.update({'atten_a': 14000, 'atten_n': 1.69})
        return params

    def _predict_pixel_percept(self, xygrid, implant, t=None):
        """Predicts the brightness at a particular pixel location

        Parameters
        ----------
        xygrid : tuple
            (idx, (x, y)): Stemming from an enumerate of self.grid
        implant : ProsthesisSystem
            A ProsthesisSystem object with an assigned stimulus
        t : optional
            Not yet implemented.
        """
        idx_xy, xydva = xygrid
        # Call the Cython function for fast processing:
        electrodes = implant.stim.electrodes
        bright = nanduri_spatial(implant.stim.interp(time=t).data.ravel(),
                                 np.array([implant[e].x for e in electrodes]),
                                 np.array([implant[e].y for e in electrodes]),
                                 np.array([implant[e].r for e in electrodes]),
                                 *self.get_tissue_coords(*xydva),
                                 self.atten_a,
                                 self.atten_n)
        # return utils.Percept(self.xdva, self.ydva, brightness)
        return bright


class Nanduri2012TemporalMixin(object):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ca = None
        self.r1 = None
        self.r2 = None
        # self.r3 = None
        self.r4a = None
        self.r4b = None
        self.r4c = None
        self.reset_state()

    def reset_state(self):
        n_px = np.prod(self.grid.shape)
        self.ca = np.zeros(n_px, dtype=np.double)
        self.r1 = np.zeros(n_px, dtype=np.double)
        self.r2 = np.zeros(n_px, dtype=np.double)
        # self.r3 = np.zeros(n_px, dtype=np.double)
        self.r4a = np.zeros(n_px, dtype=np.double)
        self.r4b = np.zeros(n_px, dtype=np.double)
        self.r4c = np.zeros(n_px, dtype=np.double)

    def _get_default_params(self):
        base_params = super()._get_default_params()
        params = {
            'has_time': True,
            'thresh_percept': 0,
            # Simulation time step:
            'dt': 0.01 / 1000,
            'dt_fast': 1.0 / 1000,
            # Time decay for the ganglion cell impulse response:
            'tau1': 0.42 / 1000,
            # Time decay for the charge accumulation:
            'tau2': 45.25 / 1000,
            # Time decay for the slow leaky integrator:
            'tau3': 26.25 / 1000,
            # Scaling factor applied to charge accumulation:
            'eps': 0,  # 8.73,
            # Asymptote of the sigmoid:
            'asymptote': 14.0,
            # Slope of the sigmoid:
            'slope': 3.0,
            # Shift of the sigmoid:
            'shift': 16.0,
            # Nanduri (2012) has a term in the stationary nonlinearity step
            # that depends on future values of R3: max_t(R3). Because the
            # finite difference model cannot look into the future, we need to
            # set a scaling factor here:
            'max_r3': 100.0,
            'max_out': 0,
        }
        base_params.update(params)
        return base_params

    def _step_temporal_model(self, stim, dt):
        """Steps the temporal model"""
        i, amp = stim
        ca, r1, r2, r4a, r4b, r4c = fast_nanduri2012(
            dt, amp, self.ca[i], self.r1[i], self.r2[i],
            self.r4a[i], self.r4b[i], self.r4c[i],
            self.tau1, self.tau2, self.tau3, self.eps,
            self.asymptote, self.shift, self.slope, self.max_r3
        )
        if r4c > self.max_out:
            print(r4c)
            self.max_out = r4c
        self.ca[i] = ca
        self.r1[i] = r1
        self.r2[i] = r2
        self.r4a[i] = r4a
        self.r4b[i] = r4b
        self.r4c[i] = r4c
        return r4c


class Nanduri2012Model(Nanduri2012TemporalMixin, ScoreboardModel):
    pass
