"""`DynaphosModel`"""
import numpy as np
import warnings
import multiprocessing
from copy import deepcopy, copy

from ..base import BaseModel, NotBuiltError
from ...stimuli import Stimulus
from ...percepts import Percept
from ...implants import ProsthesisSystem
from ...utils import cart2pol
from ...topography import Polimeni2006Map

class DynaphosModel(BaseModel):
    """Adaptation of the Dynaphos model from [Grinten2023]_

    Implements the Dynaphos model. Percepts from each
    electrode are Gaussian blobs, with the size dictated by a magnification factor
    M determined by the electrode's position in the visual cortex.
    
    Parameters:
    -----------
    dt : float, optional
        Sampling time step of the simulation (ms)
    regions : list of str, optional
        The visual regions to simulate. Options are 'v1', 'v2', or 'v3'.
        Default: ['v1']
    rheobase : float, optional
        Rheobase current constant (uA)
    tau_trace : float, optional
        Trace decay constant (ms)
    kappa_trace : float, optional
        Stimulus input effect modifier constant for memory trace
    excitability : float, optional
        Excitability constant for current spread (uA/mm^2)
    tau_act : float, optional
        Activation decay constant (ms)
    sig_slope : float, optional
        Slope of the sigmoidal brightness curve
    a50 : float, optional
        Activation value for which a phosphene reaches half of its maximum brightness
    freq : float, optional
        Default stimulus frequency (Hz)
    p_dur : float, optional
        Default stimulus pulse duration (ms)
    xrange : (x_min, x_max), optional
        A tuple indicating the range of x values to simulate (in degrees of
        visual angle). Negative values correspond to the right hemisphere of
        visual cortex, and positive values correspond to the left hemisphere.
    yrange : (y_min, y_max), optional
        A tuple indicating the range of y values to simulate (in degrees of
        visual angle).
    xystep : int, double, tuple, optional
        Step size for the range of (x,y) values to simulate (in degrees of
        visual angle). For example, to create a grid with x values [0, 0.5, 1]
        use ``x_range=(0, 1)`` and ``xystep=0.5``.
    grid_type : {'rectangular', 'hexagonal'}, optional
        Whether to simulate points on a rectangular or hexagonal grid.
    retinotopy : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
        An instance of a :py:class:`~pulse2percept.topography.VisualFieldMap`
        object that provides visual field mappings.
        By default, :py:class:`~pulse2percept.topography.Polimeni2006Map` is
        used.
    n_gray : int, optional
        The number of gray levels to use. If an integer is given, k-means
        clustering is used to compress the color space of the percept into
        ``n_gray`` bins. If None, no compression is performed.
    noise : float or int, optional
        Adds salt-and-pepper noise to each percept frame. An integer will be
        interpreted as the number of pixels to subject to noise in each 
        frame. A float between 0 and 1 will be interpreted as a ratio of 
        pixels to subject to noise in each frame.

    .. important ::
    
        If you change important model parameters outside the constructor (e.g.,
        by directly setting ``model.xrange = (-10, 10)``), you will have to call
        ``model.build()`` again for your changes to take effect.
    """
    @property
    def regions(self):
        return self._regions
    
    @regions.setter
    def regions(self, regions):
        
        if not isinstance(regions, list):
            regions = [regions]
        self._regions = regions

    def __init__(self, **params):
            self._regions = None
            super().__init__(**params)
            
            window_dur = 1000.0 / self.freq
            if self.p_dur*2 > window_dur:
                raise ValueError(f"Pulse (dur={self.p_dur*2:.2f} ms) does not fit into "
                                 f"pulse train window (dur={window_dur:.2f} "
                                 f"ms)")

            self.retinotopy.regions = self.regions
            self.grid = None
    
    def get_default_params(self):
            """Returns all settable parameters of the Dynaphos model"""
            params = {
                'xrange': (-5, 5),  # dva
                'yrange': (-5, 5),  # dva
                'xystep': 0.25,  # dva
                'grid_type': 'rectangular',
                # Use [Polemeni2006]_ visual field map with parameters specified in the paper
                'retinotopy': Polimeni2006Map(a=0.75,k=17.3,b=120,alpha1=0.95),
                # Number of gray levels to use in the percept:
                'n_gray': None,
                # Salt-and-pepper noise on the output:
                'noise': None,
                # True: print status messages, 0: silent
                'verbose': True,
                # Visual field regions to simulate
                'regions': ['v1'],
                # Time step in ms
                'dt': 20,
                # Activation decay constant (ms)
                'tau_act': 111.111111,
                # Rheobase current constant (uA)
                'rheobase': 23.9,
                # Trace decay constant (ms)
                'tau_trace': 1.96765520573e6,
                # Input effect modifier for memory trace
                'kappa_trace': 13.95528162,
                # Excitability constant (uA/mm^2)
                'excitability': 675,
                # Slope of the sigmoidal curve
                'sig_slope': 19152642.500946816,
                # A50 - activation for which a phosphene reaches half of its maximum brightness
                'a50': 1.057631326853325e-07,
                # Default stimulus frequency (Hz)
                'freq': 300,
                # Default stimulus pulse duration (ms)
                'p_dur': 0.170,
            }
            return {**params}
    
    def _build(self):
        pass
                
    def build(self, **build_params):
        """Build the model

        Performs expensive one-time calculations, such as building the spatial
        grid used to predict a percept. You must call ``build`` before
        calling ``predict_percept``.

        Parameters
        ----------
        build_params: additional parameters to set
            You can overwrite parameters that are listed in
            ``get_default_params``. Trying to add new class attributes outside
            of that will cause a ``FreezeError``.
            Example: ``model.build(param1=val)``

        """
        # import at runtime to avoid circular import
        from ...topography import Grid2D
        for key, val in build_params.items():
            setattr(self, key, val)
        # Build the spatial grid:
        self.grid = Grid2D(self.xrange, self.yrange, step=self.xystep,
                           grid_type=self.grid_type)
        self.grid.build(self.retinotopy)
        self._build()
        self.is_built = True
        return self
                    
    def _predict_percept(self, earray, stim, t_percept):
        """Predicts the brightness at spatial locations over time"""
        x_el = np.array([earray[e].x for e in stim.electrodes],
                                        dtype=np.float32)
        y_el = np.array([earray[e].y for e in stim.electrodes],
                                            dtype=np.float32)
        # whether to allow current to spread between hemispheres
        separate = 0
        boundary = 0
        if self.retinotopy.split_map:
            separate = 1
            boundary = self.retinotopy.left_offset/2

        phosphene_locations = {}
        for region in self.regions:
            phosphene_locations[region] = self.retinotopy.to_dva()[region](x_el, y_el)

        theta, r = cart2pol(*phosphene_locations['v1'])

        # magnification factors (mm/dva)
        M = self.retinotopy.k * (self.retinotopy.b - self.retinotopy.a) / ((r + self.retinotopy.a) * (r + self.retinotopy.b))

        # excitability constant uA/mm^2
        K = self.excitability
        
        xRange = self.grid['dva'].x[0, :]
        yRange = self.grid['dva'].y[:, 0]
        xgrid = self.grid['dva'].x.ravel()
        n_space = len(xgrid)
        n_time = len(t_percept)
        idx_percept = np.uint32(np.round(t_percept / self.dt))

        # default values
        freq = self.freq
        p_dur = self.p_dur
        
        # get from biphasic pulse train data if possible
        try:
            elec_params = []
            for e in stim.electrodes:
                amp = stim.metadata['electrodes'][str(e)]['metadata']['amp']
                freq = stim.metadata['electrodes'][str(e)]['metadata']['freq']
                pdur = stim.metadata['electrodes'][str(e)]['metadata']['phase_dur']
                elec_params.append([freq, amp, pdur])
            elec_params = np.array(elec_params)
            freq = elec_params[:,0]
            p_dur = elec_params[:,2]
        except KeyError:
            pass

        # holds current activation for each phosphene
        A = np.zeros(len(x_el))
        # holds effective current for each phosphene
        Ieff = np.zeros(len(x_el))
        # rheobase current (uA)
        I0 = self.rheobase
        # holds memory trace for each phosphene
        Q = np.zeros(len(x_el))
        # holds diameter of activated cortical tissue
        D = np.zeros(len(x_el))
        # constant for trace decay (seconds)
        tau_trace = self.tau_trace
        # input effect for trace
        kappa_trace = self.kappa_trace

        # brightness array
        # holds (n_space) x (n_time)
        bright = np.zeros((n_space,n_time), dtype=np.float32)

        n_percept = len(idx_percept)
        n_stim = len(stim.time)
        n_sim = idx_percept[n_percept - 1] + 1 # no negative indices
        stim_idx = 0
        frame_idx = 0
        for sim_idx in range(n_sim):
            t_sim = sim_idx * self.dt
            # get highest amp value over the frame
            amp = np.zeros(len(x_el))
            while stim_idx + 1 < n_stim and t_sim >= stim.time[stim_idx + 1]:
                stim_idx += 1
                amp = np.maximum(amp, stim.data[:,stim_idx])
            # Ieff = max(0, (Istim - I0 - Q) * f * Pw) (uA)
            Ieff = np.maximum(0, (amp - I0 - Q) * freq * (p_dur / 1000))
            # update memory trace (uA)
            Q = Q + ((-Q / (tau_trace / 1000)) + Ieff * kappa_trace) * (self.dt / 1000)
            # update phosphene size
            D = 2 * np.sqrt(amp / K) # mm
            P = (D / M) # dva
            sigma = np.clip(P / 2, 1e-22, None)
            # get activation (convert Ieff from uA to A)
            A = A + ((-A / (self.tau_act / 1000)) + Ieff * 1e-6) * (self.dt / 1000)
            # get brightness
            brightness = np.divide(1, 1 + np.exp(-self.sig_slope * (A - self.a50)))
            # create gaussian blobs & add to frame
            def create_gaussian(x0,y0,sigma,x_el):
                if separate:
                    if x_el < boundary:
                        cutoff = xRange <= 0
                    else:
                        cutoff = xRange > 0
                gaussX = np.where(cutoff, 0, np.exp(-(xRange - x0)**2 / (2 * sigma ** 2)))
                gaussY = np.exp(-(yRange - y0)**2 / (2 * sigma ** 2))
                gauss = np.outer(gaussY, gaussX)
                return gauss
            if sim_idx == idx_percept[frame_idx]:
                # `idx_t_percept` stores the time points at which we need to
                # output a percept. We compare `idx_sim` to `idx_t_percept`
                # rather than `t_sim` to `t_percept` because there is no good
                # (fast) way to compare two floating point numbers:
                for el_idx in range(stim.data.shape[0]):
                    gauss = np.zeros(self.grid['dva'].x.shape)
                    if A[el_idx] != 0:
                        gauss = create_gaussian(phosphene_locations['v1'][0][el_idx], 
                                                phosphene_locations['v1'][1][el_idx], 
                                                sigma[el_idx], x_el[el_idx])
                        bright[:,frame_idx] += gauss.ravel() * brightness[el_idx]
                bright[:,frame_idx] = np.clip(bright[:,frame_idx], 0, 1)
                frame_idx = frame_idx + 1
        return np.asarray(bright)
    
    def predict_percept(self, implant, t_percept=None):
        """Predict the spatiotemporal response

        Parameters
        ----------
        implant: :py:class:`~pulse2percept.implants.ProsthesisSystem`
            A valid prosthesis system. A stimulus can be passed via
            :py:meth:`~pulse2percept.implants.ProsthesisSystem.stim`.
        t_percept: float or list of floats, optional
            The time points at which to output a percept (ms).
            If None, ``implant.stim.time`` is used.

        Returns
        -------
        percept: :py:class:`~pulse2percept.models.Percept`
            A Percept object whose ``data`` container has dimensions Y x X x T.
            Will return None if ``implant.stim`` is None.

        """
        if not self.is_built:
            raise NotBuiltError("Yout must call ``build`` first.")
        if not isinstance(implant, ProsthesisSystem):
            raise TypeError(f"'implant' must be a ProsthesisSystem object, "
                            f"not {type(implant)}.")
        if implant.stim is None:
            # Nothing to see here:
            return None
        if implant.stim.time is None and t_percept is not None:
            raise ValueError(f"Cannot calculate spatial response at times "
                             f"t_percept={t_percept} because stimulus does not "
                             f"have a time component.")
        if implant.stim.time is None:
            raise ValueError(f"Cannot calculate response because stimulus does not "
                             f"have a time component.")
        # Make sure we don't change the user's Stimulus object:
        stim = deepcopy(implant.stim)
        # Make sure to operate on the compressed stim:
        if not stim.is_compressed:
            stim.compress()
        if t_percept is None:
            # If no time vector is given, output at frame rate determined by self.dt. We always
            # start at zero and include the last time point:
            t_percept = np.arange(0, np.maximum(self.dt, stim.time[-1]) + 1, self.dt)
        t_percept = np.sort([t_percept]).flatten()
        remainder = np.mod(t_percept, self.dt) / self.dt
        atol = 1e-3
        within_atol = (remainder < atol) | (np.abs(1 - remainder) < atol)
        if not np.all(within_atol):
            raise ValueError(f"t={t_percept[np.logical_not(within_atol)]} are "
                             f"not multiples of dt={self.dt:.2e}.")
        n_time = np.array([t_percept]).size
        if stim.data.size == 0:
            # Stimulus was compressed to zero:
            resp = np.zeros((self.grid.x.size, n_time), dtype=np.float32)
        else:
            resp = self._predict_percept(implant.earray, stim, t_percept)
        return Percept(resp.reshape(list(self.grid.x.shape) + [t_percept.size]),
                       space=self.grid, time=t_percept,
                       metadata={'stim': stim}, n_gray=self.n_gray, noise=self.noise)
