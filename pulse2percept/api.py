import numpy as np
import logging

from pulse2percept import utils
from pulse2percept import retina
from pulse2percept import implants
from pulse2percept import stimuli


class Simulation(object):

    def __init__(self, implant, name=None, engine='joblib', dojit=True,
                 num_jobs=-1):
        """Generates a simulation framework

        Parameters
        ----------
        implant : implants.ElectrodeArray
            An implants.ElectrodeArray object that describes the implant.
        name : str, optional
            Name of the simulation. Default: None.
        engine : str, optional
            Which computational backend to use:
            - 'serial': Single-core computation
            - 'joblib': Parallelization via joblib (requires `pip install
                        joblib`)
            - 'dask': Parallelization via dask (requires `pip install dask`)
            Default: joblib.
        dojit : bool, optional
            Whether to use just-in-time (JIT) compilation to speed up
            computation.
            Default: True.
        num_jobs : int, optional
            Number of cores (threads) to run the model on in parallel.
            Specify -1 to use as many cores as available.
            Default: -1.
        """
        if not isinstance(implant, implants.ElectrodeArray):
            e_s = "`implant` must be of type implants.ElectrodeArray"
            raise TypeError(e_s)

        self.name = name
        self.implant = implant
        self.engine = engine
        self.dojit = dojit
        self.num_jobs = num_jobs

        # Optic fiber layer (OFL): After calling `set_optic_fiber_layer`, this
        # variable will contain a `retina.Grid` object.
        self.ofl = None

        # Ganglion cell layer (GCL): After calling `set_ganglion_cell_layer`,
        # this variable will contain a `retina.TemporalModel` object.
        self.gcl = None

    def set_optic_fiber_layer(self, sampling=100, axon_lambda=2, rot_deg=0,
                              x_range=None, y_range=None, datapath='./',
                              save_data=True):
        """Sets parameters of the optic fiber layer (OFL)

        Parameters
        ----------
        sampling : float, optional
            Microns per grid cell. Default: 100 microns.
        axon_lambda : float, optional
            Constant that determines fall-off with axonal distance.
            Default: 2.
        rot_deg : float, optional
            Rotation angle (deg). Default: 0.
        x_range : list|None
            Lower and upper bound of the retinal grid (microns) in horizontal
            dimension. Either a list [xlo, xhi] or None. If None, the generated
            grid will be just big enough to fit the implant.
            Default: None.
        y_range : list|None
            Lower and upper bound of the retinal grid (microns) in vertical
            dimension. Either a list [ylo, yhi] or None. If None, the generated
            grid will be just big enough to fit the implant.
            Default: None.
        datapath : str
            Relative path where to look for existing retina files, and where to
            store new retina files. Default: current directory.
        save_data : bool
            Flag whether to save the data to a new retina file (True) or not
            (False). The file name is automatically generated from all
            specified input arguments.
            Default: True.
        """
        # For auto-generated grids:
        round_to = 500  # round to nearest (microns)
        cspread = 500  # add padding for current spread (microns)

        if x_range is None:
            # No x ranges given: generate automatically to fit the implant
            xs = [a.x_center for a in self.implant]
            xlo = np.floor((np.min(xs) - cspread) / round_to) * round_to
            xhi = np.ceil((np.max(xs) + cspread) / round_to) * round_to
        elif isinstance(x_range, (int, float)):
            xlo = x_range
            xhi = x_range
        elif isinstance(x_range, (list, tuple, np.ndarray)):
            if len(x_range) != 2 or x_range[1] < x_range[0]:
                e_s = "x_range must be a list [xlo, xhi] where xlo <= xhi."
                raise ValueError(e_s)
            xlo = x_range[0]
            xhi = x_range[1]
        else:
            raise ValueError("x_range must be a list [xlo, xhi] or None.")

        if y_range is None:
            # No y ranges given: generate automatically to fit the implant
            ys = [a.y_center for a in self.implant]
            ylo = np.floor((np.min(ys) - cspread) / round_to) * round_to
            yhi = np.ceil((np.max(ys) + cspread) / round_to) * round_to
        elif isinstance(y_range, (int, float)):
            ylo = y_range
            yhi = y_range
        elif isinstance(y_range, (list, np.ndarray)):
            if len(y_range) != 2 or y_range[1] < y_range[0]:
                e_s = "y_range must be a list [ylo, yhi] where ylo <= yhi."
                raise ValueError(e_s)
            ylo = y_range[0]
            yhi = y_range[1]
        else:
            raise ValueError("y_range must be a list [ylo, yhi] or None.")

        # Generate the grid from the above specs
        self.ofl = retina.Grid(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi,
                               sampling=sampling,
                               axon_lambda=axon_lambda,
                               rot=np.deg2rad(rot_deg),
                               datapath=datapath,
                               save_data=save_data)

    def set_ganglion_cell_layer(self, tsample=0.005 / 1000,
                                tau_gcl=0.42 / 1000, tau_inl=18.0 / 1000,
                                tau_ca=45.25 / 1000, scale_ca=42.1,
                                tau_slow=26.25 / 1000, scale_slow=10.0,
                                lweight=0.636, aweight=0.5,
                                slope=3.0, shift=15.0):
        """Sets parameters of the ganglion cell layer (GCL)

        Parameters
        ----------
        tsample : float
            Sampling time step (seconds). Default: 0.005 / 1000 s.
        tau_gcl : float
            Time decay constant for the fast leaky integrater of the ganglion
            cell layer.
            Default: 45.25 / 1000 s.
        tau_inl : float
            Time decay constant for the fast leaky integrater of the inner
            nuclear layer (INL). It has been shown that even epiretinal arrays
            can activate bipolar cells (in the INL), which in turn influence
            GCL activity. Default: 18.0 / 1000 s.
        tau_ca : float
            Time decay constant for the charge accumulation, has values
            between 38 - 57 ms. Default: 45.25 / 1000 s.
        scale_ca : float, optional
            Scaling factor applied to charge accumulation (used to be called
            epsilon). Default: 42.1.
        tau_slow : float
            Time decay constant for the slow leaky integrator.
            Default: 26.25 / 1000 s.
        scale_slow : float
            Scaling factor applied to the output of the cascade, to make
            output values interpretable brightness values >= 0.
            Default: 1150.0
        lweight : float
            Relative weight applied to responses from bipolar cells (weight
            of ganglion cells is 1).
            Default: 0.636.
        aweight : float
            Relative weight applied to anodic charges (weight of cathodic
            charges is 1).
            Default: 0.5.
        slope : float
            Slope of the logistic function in the stationary nonlinearity
            stage. Default: 3. In normalized units of perceptual response
            perhaps should be 2.98
        shift : float
            Shift of the logistic function in the stationary nonlinearity
            stage. Default: 16. In normalized units of perceptual response
            perhaps should be 15.9
        """
        # Generate a a TemporalModel from above specs
        tm = retina.TemporalModel(tsample=tsample,
                                  tau_gcl=tau_gcl, tau_inl=tau_inl,
                                  tau_ca=tau_ca, scale_ca=scale_ca,
                                  tau_slow=tau_slow, scale_slow=scale_slow,
                                  lweight=lweight, aweight=aweight,
                                  slope=slope, shift=shift)
        self.gcl = tm

    def _set_layers(self):
        """Sets up all layers whose setters have not been called by the user

        This function makes sure all necessary parts of the simulation are
        initialized before transforming stimuli to percepts.
        Uninitialized layers (by the user) will simply be initialized with
        default argument values.
        """
        if self.ofl is None:
            self.set_optic_fiber_layer()
        if self.gcl is None:
            self.set_ganglion_cell_layer()

    def pulse2percept(self, stim, t_percept=None, tol=0.05,
                      layers=['OFL', 'GCL', 'INL']):
        """Transforms an input stimulus to a percept

        Parameters
        ----------
        stim : utils.TimeSeries|list|dict
            There are several ways to specify an input stimulus:

            - For a single-electrode array, pass a single pulse train; i.e.,
              a single utils.TimeSeries object.
            - For a multi-electrode array, pass a list of pulse trains; i.e.,
              one pulse train per electrode.
            - For a multi-electrode array, specify all electrodes that should
              receive non-zero pulse trains by name.
        t_percept : float, optional
            The desired sampling time step (seconds) of the output. If None is
            given, the output sampling time step will correspond to the time
            step of the `stim` object.
            Default: Inherit from the `stim` object.
        tol : float, optional
            Ignore pixels whose effective current is smaller than a fraction
            `tol` of the max value.
            Default: 0.05.
        layers : list, optional
            A list of retina layers to simulate:
            - 'OFL': Includes the optic fiber layer in the simulation.
                     If omitted, the tissue activation map will not account
                     for axon streaks.
            - 'GCL': Includes the ganglion cell layer in the simulation.
            - 'INL': Includes the inner nuclear layer in the simulation.
                     If omitted, bipolar cell activity does not contribute
                     to ganglion cell activity.
            Order of specified layer does not matter.
            Default: ['OFL', 'GCL', 'INL'].

        Returns
        -------
        A utils.TimeSeries object whose data container comprises the predicted
        brightness over time at each retinal location (x, y), with the last
        dimension of the container representing time (t).

        Examples
        --------
        Simulate a single-electrode array:

        >>> import pulse2percept as p2p
        >>> implant = p2p.implants.ElectrodeArray('subretinal', 0, 0, 0, 0)
        >>> stim = p2p.stimuli.PulseTrain(tsample=5e-6, freq=50, amp=20)
        >>> sim = p2p.Simulation(implant)
        >>> percept = sim.pulse2percept(stim)  # doctest: +SKIP

        Simulate an Argus I array centered on the fovea, where a single
        electrode is being stimulated ('C3'):

        >>> import pulse2percept as p2p
        >>> implant = p2p.implants.ArgusI()
        >>> stim = {'C3': stimuli.PulseTrain(tsample=5e-6, freq=50,
        ...                                              amp=20)}
        >>> sim = p2p.Simulation(implant)
        >>> resp = sim.pulse2percept(stim, implant)  # doctest: +SKIP
        """
        logging.getLogger(__name__).info("Starting pulse2percept...")

        # Get a flattened, all-uppercase list of layers
        layers = np.array([layers]).flatten()
        layers = np.array([l.upper() for l in layers])

        # Make sure all specified layers exist
        not_supported = [l not in retina.SUPPORTED_LAYERS for l in layers]
        if any(not_supported):
            msg = ', '.join(layers[not_supported])
            msg = "Specified layer %s not supported. " % msg
            msg += "Choose from %s." % ', '.join(retina.SUPPORTED_LAYERS)
            raise ValueError(msg)

        # Set up all layers that haven't been set up yet
        self._set_layers()

        # Parse `stim` (either single pulse train or a list/dict of pulse
        # trains), and generate a list of pulse trains, one for each electrode
        pt_list = stimuli.parse_pulse_trains(stim, self.implant)

        if not np.allclose([p.tsample for p in pt_list], self.gcl.tsample):
            e_s = "For now, all pulse trains must have the same sampling "
            e_s += "time step as the ganglion cell layer. In the future, "
            e_s += "this requirement might be relaxed."
            raise ValueError(e_s)

        # Tissue activation maps: If OFL is simulated, includes axon streaks.
        if 'OFL' in layers:
            ecs, _ = self.ofl.electrode_ecs(self.implant)
        else:
            _, ecs = self.ofl.electrode_ecs(self.implant)

        # Calculate the max of every current spread map
        lmax = np.zeros((2, ecs.shape[-1]))
        if 'INL' in layers:
            lmax[0, :] = ecs[:, :, 0, :].max(axis=(0, 1))
        if ('GCL' or 'OFL') in layers:
            lmax[1, :] = ecs[:, :, 1, :].max(axis=(0, 1))

        # `ecs_list` is a pixel by `n` list where `n` is the number of layers
        # being simulated. Each value in `ecs_list` is the current contributed
        # by each electrode for that spatial location
        ecs_list = []
        idx_list = []
        for xx in range(self.ofl.gridx.shape[1]):
            for yy in range(self.ofl.gridx.shape[0]):
                # If any of the used current spread maps at [yy, xx] are above
                # tolerance, we need to process that pixel
                process_pixel = False
                if 'INL' in layers:
                    # For this pixel: Check if the ecs in any layer is large
                    # enough compared to the max across pixels within the layer
                    process_pixel |= np.any(ecs[yy, xx, 0, :] >=
                                            tol * lmax[0, :])
                if ('GCL' or 'OFL') in layers:
                    process_pixel |= np.any(ecs[yy, xx, 1, :] >=
                                            tol * lmax[1, :])

                if process_pixel:
                    ecs_list.append(ecs[yy, xx])
                    idx_list.append([yy, xx])

        s_info = "tol=%.1f%%, %d/%d px selected" % (tol * 100, len(ecs_list),
                                                    np.prod(ecs.shape[:2]))
        logging.getLogger(__name__).info(s_info)

        sr_list = utils.parfor(self.gcl.model_cascade,
                               ecs_list, n_jobs=self.num_jobs,
                               engine=self.engine,
                               func_args=[pt_list, layers, self.dojit])
        bm = np.zeros(self.ofl.gridx.shape +
                      (sr_list[0].data.shape[-1], ))
        idxer = tuple(np.array(idx_list)[:, i] for i in range(2))
        bm[idxer] = [sr.data for sr in sr_list]
        percept = utils.TimeSeries(sr_list[0].tsample, bm)

        # It is possible to specify an additional sampling rate for the
        # percept: If different from the input sampling rate, need to resample.
        if t_percept != percept.tsample:
            percept = percept.resample(t_percept)

        logging.getLogger(__name__).info("Done.")

        return percept

    def plot_fundus(self, stim=None, ax=None):
        """Plot the implant on the retinal surface akin to a fundus photopgraph

        This function plots an electrode array on top of the axon streak map
        of the retina, akin to a fundus photograph. A blue rectangle highlights
        the area of the retinal surface that is being simulated.
        If `stim` is passed, activated electrodes will be highlighted.

        Parameters
        ----------
        stim : utils.TimeSeries|list|dict, optional
            An input stimulus, as passed to p2p.pulse2percept. If given,
            activated electrodes will be highlighted in the plot.
            Default: None
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            A Matplotlib axes object. If None given, a new one will be created.
            Default: None

        Returns
        -------
        Returns a handle to the created figure (`fig`) and axes element (`ax`).
        """
        from matplotlib import patches

        self._set_layers()

        fig = None
        if ax is None:
            # No axes object given: create
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, figsize=(10, 8))

        ax.set_facecolor('black')
        ax.plot(self.ofl.jan_x[:, ::5], -self.ofl.jan_y[:, ::5],
                c=(0.5, 1, 0.5))

        # Draw in the the retinal patch we're simulating.
        # This defines the size of our "percept" image below.
        dva_xmin = retina.ret2dva(self.ofl.gridx.min())
        dva_ymin = -retina.ret2dva(self.ofl.gridy.max())
        patch = patches.Rectangle((dva_xmin, dva_ymin),
                                  retina.ret2dva(self.ofl.range_x),
                                  retina.ret2dva(self.ofl.range_y),
                                  alpha=0.7)
        ax.add_patch(patch)

        # Highlight location of stimulated electrodes
        if stim is not None:
            for key in stim:
                el = self.implant[key]
                if el is not None:
                    ax.plot(retina.ret2dva(el.x_center),
                            -retina.ret2dva(el.y_center), 'oy',
                            markersize=np.sqrt(el.radius) * 2)

        # Plot all electrodes and their label
        for e in self.implant.electrodes:
            ax.text(retina.ret2dva(e.x_center + 10),
                    -retina.ret2dva(e.y_center + 5),
                    e.name, color='white', size='x-large')
            ax.plot(retina.ret2dva(e.x_center),
                    -retina.ret2dva(e.y_center), 'ow',
                    markersize=np.sqrt(e.radius))

        ax.set_aspect('equal')
        ax.set_xlim(-20, 20)
        ax.set_xlabel('visual angle (deg)')
        ax.set_ylim(-15, 15)
        ax.set_ylabel('visual angle (deg)')
        ax.set_title('Image flipped (upper retina = upper visual field)')
        ax.grid('off')

        return fig, ax


def get_brightest_frame(percept):
    """Returns the brightest frame of a percept

    This function returns the frame of a percept (brightness over time) that
    contains the brightest pixel.

    Parameters
    ----------
    percept : TimeSeries
        The brightness movie as a TimeSeries object.
    """
    _, frame = percept.max_frame()

    return frame
