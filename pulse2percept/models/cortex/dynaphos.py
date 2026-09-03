""":py:class:`~pulse2percept.models.cortex.DynaphosModel`"""
import numpy as np
import warnings
from copy import deepcopy, copy

from ..base import (BaseModel, _check_implant, _electrode_offsets,
                    _latent_offsets, _location_noise_sigma,
                    _require_stim_dimension)
from ...percepts import Percept
from ...stimuli import BiphasicPulseTrain
from ...units import A, Quantity, as_value, dva, Hz, mm, ms, uA
from ...utils import cart2pol
from ...utils.constants import MS_PER_S, UM_PER_MM, ZORDER
from ...topography import Polimeni2006Map


def _pulse_train_clocks(stim):
    """``{electrode: (freq, phase_dur)}`` where the stimulus is pulse trains

    Read off the trains the stimulus is made of. ``None`` when it is not made
    of them, in which case this model simulates on its own default clock, as
    it always has.

    Has to be asked *before* the stimulus is compressed: compression installs
    a new waveform, and that is exactly what says the trains no longer
    describe it.
    """
    sources = stim._structured_sources()
    if sources is None:
        return None
    if any(type(src) is not BiphasicPulseTrain for _, src in sources):
        return None
    return {str(e): (src.freq, src.phase_dur) for e, src in sources}


#: Amperes in a microamp (1e-6). The activation cascade below is the published
#: one, which is written in SI units, while a p2p stimulus is in microamps.
#: Spelled out here at module scope because ``A`` names the activation array
#: inside ``_predict_percept``.
_A_PER_UA = Quantity(1, uA).to_value(A)


class DynaphosModel(BaseModel):
    """Adaptation of the Dynaphos model from [vanderGrinten2023]_

    The original and official implementation is available at 
    https://github.com/neuralcodinglab/dynaphos.

    Implements the Dynaphos model. Percepts from each
    electrode are Gaussian blobs, with the size dictated by a magnification factor
    M determined by the electrode's position in the visual cortex.
    
    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.Implant`
        The implant whose stimulation this model predicts.

        .. versionadded:: 0.11.0

    dt : float, optional
        Sampling time step of the simulation (ms)
    regions : list of str, optional
        The visual regions to simulate. Options are 'v1', 'v2', or 'v3'.
        Default : ['v1']
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
    a_thr : float, optional
        Activation threshold value, under which a phosphene is not generated
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
    step : int, double, tuple, optional
        Step size for the range of (x,y) values to simulate (in degrees of
        visual angle). For example, to create a grid with x values [0, 0.5, 1]
        use ``xrange=(0, 1)`` and ``step=0.5``. Pass a tuple to give the x
        and y axes different step sizes.
    grid_type : {'rect', 'hex'}, optional
        Whether to simulate points on a rectangular or hexagonal grid.
    visual_field_map : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
        An instance of a :py:class:`~pulse2percept.topography.VisualFieldMap`
        object that provides visual field mappings.
        By default, :py:class:`~pulse2percept.topography.Polimeni2006Map` is
        used.
    n_gray : int, optional
        The number of gray levels to use. If an integer is given, k-means
        clustering is used to compress the color space of the percept into
        ``n_gray`` bins. If None, no compression is performed.
    location_noise : float or None, optional
        Standard deviation of the variation in phosphene location from the
        ``visual_field_map``, in dva. Locations are fixed for a model instance.
        ``None`` or 0 disables the variation.
        
        .. versionadded:: 0.11.0

    noise : float or int, optional
        Adds salt-and-pepper noise to each percept frame. An integer will be
        interpreted as the number of pixels to subject to noise in each 
        frame. A float between 0 and 1 will be interpreted as a ratio of 
        pixels to subject to noise in each frame.

        
    .. important::
    
        Changing a model parameter outside the constructor (e.g., by directly
        setting ``model.xrange = (-10, 10)``) invalidates the build, and the
        next ``predict_percept`` builds it again.
    """

    @property
    def regions(self):
        return self._regions

    @regions.setter
    def regions(self, regions):
        
        if not isinstance(regions, list):
            regions = [regions]
        self._regions = regions

    def __init__(self, implant, *, dt=20, regions=None, rheobase=23.9,
                 tau_trace=1.96765520573e6, kappa_trace=13.95528162,
                 excitability=675, tau_act=111.111111,
                 sig_slope=19152642.500946816, a_thr=9.141886000943878e-08,
                 a50=1.057631326853325e-07, freq=300, p_dur=0.170,
                 xrange=(-5, 5), yrange=(-5, 5), step=0.25,
                 grid_type='rect', visual_field_map=None, n_gray=None,
                 noise=None,
                 location_noise=None,
                 verbose=True):
            _check_implant(implant)
            self._implant = implant
            self._regions = None
            super().__init__(
                dt=dt, rheobase=rheobase, tau_trace=tau_trace,
                kappa_trace=kappa_trace, excitability=excitability,
                tau_act=tau_act, sig_slope=sig_slope, a_thr=a_thr, a50=a50,
                freq=freq, p_dur=p_dur, xrange=xrange, yrange=yrange,
                step=step, grid_type=grid_type,
                # Published map parameters, see [vanderGrinten2023]_:
                visual_field_map=(
                    Polimeni2006Map(a=0.75, k=17.3, b=120, alpha1=0.95)
                    if visual_field_map is None else visual_field_map),
                n_gray=n_gray, noise=noise,
                location_noise=location_noise, verbose=verbose,
                regions=['v1'] if regions is None else regions)

            self.visual_field_map.regions = self.regions
            self.grid = None
            # This subject's latent electrode offsets; see `_latent_offsets`:
            self._location_noise_z = None

    @property
    def implant(self):
        """The prosthesis system this model predicts percepts for

        Model context rather than trial input: named once, and
        :py:meth:`predict_percept` is then given the stimulus. Rebinding
        invalidates the build.

        .. versionadded:: 0.11.0
        """
        return self._implant

    @implant.setter
    def implant(self, implant):
        """Implant setter (called upon ``self.implant = implant``)"""
        _check_implant(implant)
        if implant is not self._implant:
            self._is_built = False
            # A different implant is a different set of electrodes to displace:
            self._location_noise_z = None
        self._implant = implant

    
    def get_default_params(self):
            """Returns all settable parameters of the Dynaphos model"""
            params = {
                'xrange': (-5, 5),  # dva
                'yrange': (-5, 5),  # dva
                'step': 0.25,  # dva
                'grid_type': 'rect',
                # Use [Polemeni2006]_ visual field map with parameters specified in the paper
                'visual_field_map': Polimeni2006Map(a=0.75,k=17.3,b=120,
                                                    alpha1=0.95),
                # Number of gray levels to use in the percept:
                'n_gray': None,
                # Salt-and-pepper noise on the output:
                'noise': None,
                # Subject-specific phosphene displacement (dva):
                'location_noise': None,
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
                # A_Thr - activation threshold under which a phosphene is not generated
                'a_thr': 9.141886000943878e-08,
                # Default stimulus frequency (Hz)
                'freq': 300,
                # Default stimulus pulse duration (ms)
                'p_dur': 0.170,
            }
            return {**params}

    def get_param_units(self):
        """Return a dict of the units that parameters are stored in

        This model's equations mix units: they take microamps, milliseconds
        and hertz as input, and the published cascade they implement is
        written in SI (seconds and amperes). What is declared here is the
        *input* contract -- the units a caller supplies, which are the ones
        the docstring documents. ``_predict_percept`` converts them to SI once
        before its loop; see the conversion block there.
        """
        return {
            **super().get_param_units(),
            'xrange': dva,
            'yrange': dva,
            'step': dva,
            # The percept is displaced in the visual field, not on cortex:
            'location_noise': dva,
            'dt': ms,
            # Decay constants, both converted to seconds where they are used:
            'tau_act': ms,
            'tau_trace': ms,
            'rheobase': uA,
            'excitability': uA / mm ** 2,
            'freq': Hz,
            'p_dur': ms,
            # `kappa_trace`, `sig_slope`, `a50` and `a_thr` are fitted
            # constants of the activation cascade, and are left undeclared.
        }


    def _build(self):
        pass
                
    def build(self, **build_params):
        """Build the model

        Performs expensive one-time calculations, such as building the spatial
        grid used to predict a percept. ``predict_percept`` calls it for you
        when the model is not built.

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
        # See `BaseModel.build`:
        self.set_params(**build_params)
        # check that freq/pdur fit. `freq` counts cycles per second, and every
        # duration in this model is in milliseconds:
        window_dur = MS_PER_S / self.freq
        if self.p_dur*2 > window_dur:
            raise ValueError(f"Pulse (dur={self.p_dur*2:.2f} ms) does not fit into "
                            f"pulse train window (dur={window_dur:.2f} "
                            f"ms)")
        # Build the spatial grid:
        self.grid = Grid2D(self.xrange, self.yrange, step=self.step,
                           grid_type=self.grid_type)
        self.grid.build(self.visual_field_map)
        if _location_noise_sigma(self) is not None:
            # Draw the subject here so that the offsets do not depend on which
            # stimulus is predicted first:
            _latent_offsets(self)
        self._build()
        self._is_built = True
        return self
                    
    def _predict_percept(self, electrode_array, stim, t_percept, clocks=None):
        """Predicts the brightness at spatial locations over time"""
        x_el, y_el, _ = self._electrode_coords(electrode_array, stim)
        # whether to allow current to spread between hemispheres
        separate = 0
        boundary = 0
        if self.visual_field_map.split_map:
            separate = 1
            boundary = self.visual_field_map.left_offset/2

        phosphene_locations = {}
        for region in self.regions:
            phosphene_locations[region] = self.visual_field_map.to_dva()[region](x_el, y_el)

        theta, r = cart2pol(*phosphene_locations['v1'])

        # `location_noise` moves the phosphene, not the electrode: the size
        # below still follows the canonical cortical magnification.
        offsets = _electrode_offsets(self, stim.electrodes)
        if offsets is not None:
            for region, (px, py) in phosphene_locations.items():
                phosphene_locations[region] = (px + offsets[:, 0],
                                               py + offsets[:, 1])

        # magnification factors (mm/dva)
        M = self.visual_field_map.k * (self.visual_field_map.b - self.visual_field_map.a) / ((r + self.visual_field_map.a) * (r + self.visual_field_map.b))

        # excitability constant uA/mm^2
        K = self.excitability
        
        xRange = self.grid['dva'].x[0, :]
        yRange = self.grid['dva'].y[:, 0]
        xgrid = self.grid['dva'].x.ravel()
        n_space = len(xgrid)
        n_time = len(t_percept)
        idx_percept = np.uint32(np.round(t_percept / self.dt))

        # The model's own clock, unless the stimulus brought one per
        # electrode. `clocks` is keyed by name because compression drops the
        # electrodes that are driven at zero, and these have to line up with
        # the ones that survived:
        freq = self.freq
        p_dur = self.p_dur
        if clocks is not None:
            per_electrode = np.array([clocks[str(e)] for e in stim.electrodes])
            freq = per_electrode[:, 0]
            p_dur = per_electrode[:, 1]

        # holds instantaneous current for each phosphene
        amp = np.zeros(len(x_el))
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
        # holds sigma for gaussian phosphene generation
        sigma = np.zeros(len(x_el))
        # constant for trace decay (ms; converted to seconds below)
        tau_trace = self.tau_trace
        # input effect for trace
        kappa_trace = self.kappa_trace

        # brightness array
        # holds (n_space) x (n_time)
        bright = np.zeros((n_space,n_time), dtype=np.float32)

        n_percept = len(idx_percept)
        # Across the numerical boundary: microamps and milliseconds for
        # this model, whatever the stimulus happens to store:
        stim_data = self._stim_values(stim)
        stim_time = self._stim_times(stim)
        n_stim = len(stim_time)
        n_sim = idx_percept[n_percept - 1] + 1 # no negative indices
        stim_idx = 0
        frame_idx = 0
        # The cascade below is the published one, which is written in SI
        # units, while this model is handed milliseconds and microamps.
        # Convert the durations once, here, so that the loop is plain floats
        # and no factor of a thousand is left implicit inside it (see
        # `_A_PER_UA` for the current):
        p_dur_s = p_dur / MS_PER_S
        tau_trace_s = tau_trace / MS_PER_S
        tau_act_s = self.tau_act / MS_PER_S
        dt_s = self.dt / MS_PER_S
        for sim_idx in range(n_sim):
            t_sim = sim_idx * self.dt
            # get highest amp value over the frame
            # but only reset amp to 0 if stimulus is updated at all during the frame
            if stim_idx + 1 < n_stim and t_sim >= stim_time[stim_idx + 1]:
                amp = np.zeros(len(x_el))
            # or stimulus has ended but we still want to predict
            if t_sim > stim_time[-1]:
                amp = np.zeros(len(x_el))
            while stim_idx + 1 < n_stim and t_sim >= stim_time[stim_idx + 1]:
                stim_idx += 1
                amp = np.maximum(amp, stim_data[:,stim_idx])
            # Ieff = max(0, (Istim - I0 - Q) * f * Pw) (uA)
            Ieff = np.maximum(0, (amp - I0 - Q) * freq * p_dur_s)
            # update memory trace (uA)
            Q = Q + ((-Q / tau_trace_s) + Ieff * kappa_trace) * dt_s
            # update phosphene size
            D = 2 * np.sqrt(amp / K) # mm
            P = (D / M) # dva
            # calculate sigma for gaussian (only update sigma if amplitude > 0)
            sigma = np.where(amp > 0, np.clip(P / 2, 1e-22, None), sigma) 
            # get activation (Ieff converted from uA to A)
            A = A + ((-A / tau_act_s) + Ieff * _A_PER_UA) * dt_s
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
                for el_idx in range(stim_data.shape[0]):
                    gauss = np.zeros(self.grid['dva'].x.shape)
                    if A[el_idx] >= self.a_thr:
                        gauss = create_gaussian(phosphene_locations['v1'][0][el_idx], 
                                                phosphene_locations['v1'][1][el_idx], 
                                                sigma[el_idx], x_el[el_idx])
                        bright[:,frame_idx] += gauss.ravel() * brightness[el_idx]
                bright[:,frame_idx] = np.clip(bright[:,frame_idx], 0, 1)
                frame_idx = frame_idx + 1
        return np.asarray(bright)
    
    def predict_percept(self, source, t_percept=None):
        """Predict the spatiotemporal response

        .. versionchanged:: 0.11.0
            Takes the stimulus source rather than an implant; the implant is
            the one this model is bound to.

        Parameters
        ----------
        source : :py:class:`~pulse2percept.stimuli.Stimulus` source type
            What is presented to the device; see
            :py:meth:`~pulse2percept.implants.Implant.prepare_stim`.
        t_percept: float or list of floats, optional
            The time points at which to output a percept (ms). This
            model's numerical contract is fixed to milliseconds.
            If None, the prepared stimulus' own time points are used.
            May be given as a unitful quantity (e.g. ``[0, 20] * ms``);
            see :py:mod:`pulse2percept.units`.

        Returns
        -------
        percept: :py:class:`~pulse2percept.models.Percept`
            A Percept object whose ``data`` container has dimensions Y x X x T.
            Will return None if ``source`` is None or empty.

        """
        if not self.is_built:
            self.build()
        t_percept = as_value(t_percept, self.time_unit, 't_percept')
        prepared = self.implant.prepare_stim(source)
        if prepared is None:
            # Nothing to see here:
            return None
        _require_stim_dimension(self, prepared)
        if prepared.time is None and t_percept is not None:
            raise ValueError(f"Cannot calculate spatial response at times "
                             f"t_percept={t_percept} because stimulus does not "
                             f"have a time component.")
        if prepared.time is None:
            raise ValueError(f"Cannot calculate response because stimulus does not "
                             f"have a time component.")
        # Make sure we don't change the user's Stimulus object:
        stim = deepcopy(prepared)
        # The pulse clock is a question about what the stimulus is made of,
        # and compressing it answers "samples, and nothing else". So ask
        # first; the waveform below is what the time evolution runs on:
        clocks = _pulse_train_clocks(stim)
        # Make sure to operate on the compressed stim:
        if not stim.is_compressed:
            stim.compress()
        if t_percept is None:
            # If no time vector is given, output at the frame rate determined
            # by self.dt. We start at zero and stop at the last `dt` boundary
            # the stimulus reaches, including its end when that lands exactly
            # on one; `nextafter` is what makes `arange`'s half-open end
            # behave that way. The `+ 1` it replaces was one *millisecond* of
            # slack, which overshot the end of the stimulus whenever `dt` was
            # finer than that, and would not have survived a model counting in
            # anything but milliseconds. The floor at `dt` is the one case
            # that still reports past the end: a stimulus shorter than a
            # single step gets that step anyway, so that there is a percept to
            # look at. Name `t_percept` to ask for other instants.
            end = np.maximum(self.dt, self._stim_times(stim)[-1])
            t_percept = np.arange(0, np.nextafter(end, np.inf), self.dt)
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
            resp = self._predict_percept(self.implant.electrode_array, stim,
                                         t_percept, clocks)
        return Percept(resp.reshape(list(self.grid.x.shape) + [t_percept.size]),
                       space=self.grid, time=t_percept,
                       time_unit=self.time_unit,
                       metadata={'stim': stim}, n_gray=self.n_gray, noise=self.noise)

    def plot(self, use_dva=False, style=None, autoscale=True, ax=None,
             figsize=None, fc=None):
        """Plot the model
        Parameters
        ----------
        use_dva : bool, optional
            Plot points in visual field. If false, simulated points will be 
            plotted in cortex
        style : {'hull', 'scatter', 'cell'}, optional
            Grid plotting style:

            * 'hull': Show the convex hull of the grid (that is, the outline of
              the smallest convex set that contains all grid points).
            * 'scatter': Scatter plot all grid points
            * 'cell': Show the outline of each grid cell as a polygon. Note that
              this can be costly for a high-resolution grid.
              
        autoscale : bool, optional
            Whether to adjust the x,y limits of the plot to fit the implant
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            A Matplotlib axes object. If None, will either use the current axes
            (if exists) or create a new Axes object.
        figsize : (float, float), optional
            Desired (width, height) of the figure in inches
        Returns
        -------
        ax : ``matplotlib.axes.Axes``
            Returns the axis object of the plot
        """
        if style is None:
            style = 'hull' if use_dva else 'scatter'
        # Model must be built to access cortical coordinates
        if not self.is_built:
            self.build()
        ax = self.grid.plot(style=style, use_dva=use_dva, autoscale=autoscale, 
                            ax=ax, figsize=figsize, fc=fc, 
                            zorder=ZORDER['background'], 
                            legend=True if not use_dva else False)
        if use_dva:
            ax.set_xlabel('x (dva)')
            ax.set_ylabel('y (dva)')
        else:
            # Cortical coordinates are stored in microns, plotted in mm:
            ax.set_xticklabels(np.array(ax.get_xticks()) / UM_PER_MM)
            ax.set_yticklabels(np.array(ax.get_yticks()) / UM_PER_MM)
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
        return ax