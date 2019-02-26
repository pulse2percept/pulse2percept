import numpy as np
import logging
import six

from pulse2percept import (utils, retina, implants, stimuli)


class Simulation(object):

    def __init__(self, implant, engine='joblib', scheduler='threading',
                 use_jit=True, n_jobs=-1):
        """Generates a simulation framework

        Parameters
        ----------
        implant : implants.ElectrodeArray
            An implants.ElectrodeArray object that describes the implant.
        engine : str, optional, default: 'joblib'
            Which computational back end to use:
            - 'serial': Single-core computation
            - 'joblib': Parallelization via joblib (requires `pip install
                        joblib`)
            - 'dask': Parallelization via dask (requires `pip install dask`).
                      Dask backend can be specified via `threading`.
        scheduler : str, optional, default: 'threading'
            Which scheduler to use (irrelevant for 'serial' engine):
            - 'threading': a scheduler backed by a thread pool
            - 'multiprocessing': a scheduler backed by a process pool
        use_jit : bool, optional, default: True
            Whether to use just-in-time (JIT) compilation to speed up
            computation.
        n_jobs : int, optional, default: -1
            Number of cores (threads) to run the model on in parallel.
            Specify -1 to use as many cores as available.
        """
        if not isinstance(implant, implants.ElectrodeArray):
            e_s = "`implant` must be of type implants.ElectrodeArray"
            raise TypeError(e_s)

        self.implant = implant
        self.engine = engine
        self.scheduler = scheduler
        self.use_jit = use_jit
        self.n_jobs = n_jobs

        # Optic fiber layer (OFL): After calling `set_optic_fiber_layer`, this
        # variable will contain a `retina.Grid` object.
        self.ofl = None

        # Ganglion cell layer (GCL): After calling `set_ganglion_cell_layer`,
        # this variable will contain a `retina.TemporalModel` object.
        self.gcl = None

    def set_optic_fiber_layer(self, sampling=100, x_range=None, y_range=None,
                              n_axons=501, phi_range=(-180.0, 180.0),
                              n_rho=801, rho_range=(4.0, 45.0),
                              loc_od=(15.5, 1.5),
                              sensitivity_rule='decay', decay_const=1.0,
                              alpha=14000, contribution_rule='max',
                              powermean_exp=None, datapath='.', save_data=True):
        """Sets parameters of the optic fiber layer (OFL)

        Parameters
        ----------
        sampling : float, optional, default: 100 microns
            Microns per grid cell.
        x_range : (xlo, xhi)|None, default: None
            Lower and upper bound of the retinal grid (microns) in horizontal
            dimension. Either a tuple (xlo, xhi) or None. If None, the
            generated grid will be just big enough to fit the implant.
        y_range : (ylo, yhi)|None, default: None
            Lower and upper bound of the retinal grid (microns) in vertical
            dimension. Either a tuple (ylo, yhi) or None. If None, the
            generated grid will be just big enough to fit the implant.
        n_axons : int, optional, default: 501
            The number of axons to generate. Their start orientations `phi0`
            (in modified polar coordinates) will be sampled uniformly from
            `phi_range`.
        phi_range : (lophi, hiphi), optional, default: (-180, 180)
            Range of angular positions of axon fibers at their starting points
            (polar coordinates, degrees) to be sampled uniformly with `n_axons`
            samples. Must be within [-180, 180].
        n_rho: int, optional, default: 801
            Number of sampling points along the radial axis(polar coordinates).
        rho_range: (rho_min, rho_max), optional, default: (4.0, 45.0)
            Lower and upper bounds for the radial position values(polar
            coordinates).
        loc_od: (x_od, y_od), optional, default: (15.5 1.5)
            Location of the center of the optic disc (x, y) in Cartesian
            coordinates. In a right (left) eye, we should have x > 0 (x < 0).
        sensitivity_rule : {'decay', 'Jeng2011'}, optional, default: 'decay'
            This rule specifies how the activation of the axon differs as a
            function of distance from the soma. The following options are
            available:
            - 'decay':
                Axon sensitivity decays exponentially with distance. Specify
                `decay_const` to change the steepness of the fall-off with
                distance.
            - 'Jeng2011':
                Axon sensitivity peaks near the sodium band (50um from the
                soma), then plateaus on the distal axon at roughly half of the
                peak sensitivity (see Fig. 2 in [1]_).
        contribution_rule : {'max', 'sum', 'mean'}, optional, default: 'max'
            This rule specifies how the activation thresholds across all axon
            segments are combined to determine the contribution of the axon to
            the current spread. The following options are available:
            - 'max':
                The axon's contribution to the current spread is equal to the
                max. sensitivity across all axon segments.
            - 'sum':
                The axon's contribution to the current spread is equal to the
                sum sensitivity across all axon segments.
            - 'mean':
                The axon's contribution to the current spread is equal to the
                mean sensitivity across all axon segments. Specify
                `powermean_exp` to change the exponent of the generalized
                (power) mean, calculated as np.mean(x ** powermean_exp) **
                (1.0 / powermean_exp). Default is 1, which is equal to the
                arithmetic mean.
        decay_const : float, optional, default: 2.0
            When `sensitivity_rule` is set to 'decay', specifies the decay
            constant of the exponential fall-off.
        alpha : float, optional, default: 14000
                Current spread parameter for passive current spread from the electrode.
        powermean_exp : float, optional, default: 1.0
            When `sensitivity_rule` is set to 'mean', specifies the exponent of
            the generalized (power) mean function. The power mean is calculated
            as np.mean(x ** powermean_exp) ** (1.0 / powermean_exp).
        datapath : str, default: current directory
            Relative path where to look for existing retina files, and where to
            store new retina files.
        save_data : bool, default: True
            Flag whether to save the data to a new retina file (True) or not
            (False). The file name is automatically generated from all
            specified input arguments.

        Notes
        -----
        .. [1] J. Jeng, S. Tang, A. Molnar, N. J. Desai, and S. I. Fried, "The
           sodium channel band shapes the response to electric stimulation in
           retinal ganglion cells", J Neural Eng 8 (036022), 2011.
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
                e_s = "x_range must be a tuple (xlo, xhi) where xlo <= xhi."
                raise ValueError(e_s)
            xlo = x_range[0]
            xhi = x_range[1]
        else:
            raise ValueError("x_range must be a tuple (xlo, xhi) or None.")

        if y_range is None:
            # No y ranges given: generate automatically to fit the implant
            ys = [a.y_center for a in self.implant]
            ylo = np.floor((np.min(ys) - cspread) / round_to) * round_to
            yhi = np.ceil((np.max(ys) + cspread) / round_to) * round_to
        elif isinstance(y_range, (int, float)):
            ylo = y_range
            yhi = y_range
        elif isinstance(y_range, (list, tuple, np.ndarray)):
            if len(y_range) != 2 or y_range[1] < y_range[0]:
                e_s = "y_range must be a tuple (ylo, yhi) where ylo <= yhi."
                raise ValueError(e_s)
            ylo = y_range[0]
            yhi = y_range[1]
        else:
            raise ValueError("y_range must be a tuple (ylo, yhi) or None.")

        if alpha <= 0:
            raise ValueError("alpha cannot be a negative value. ")

        # Generate the grid from the above specs
        self.ofl = retina.Grid(x_range=(xlo, xhi), y_range=(ylo, yhi),
                               eye=self.implant.eye, sampling=sampling,
                               n_axons=n_axons, phi_range=phi_range,
                               n_rho=n_rho, rho_range=rho_range,
                               loc_od=loc_od,
                               sensitivity_rule=sensitivity_rule,
                               contribution_rule=contribution_rule,
                               decay_const=decay_const,
                               powermean_exp=powermean_exp, alpha=alpha,
                               datapath=datapath, save_data=save_data,
                               engine=self.engine, scheduler=self.scheduler,
                               n_jobs=self.n_jobs)

    def set_ganglion_cell_layer(self, model, **kwargs):
        """Sets parameters of the ganglion cell layer (GCL)

        Select from pre-existing ganglion cell models or specify a custom one.

        Parameters
        ----------
        model : str|retina.BaseModel
            A custom ganglion cell model can be specified by passing an
            instance of type `retina.BaseModel`. Else select from pre-existing
            models:

            - 'latest':
                The latest temporal model for epiretinal and subretinal
                arrays (experimental).

                Additional keyword arguments:

                - tau_gcl : float, optional, default: 45.25 ms
                    Time decay constant for the fast leaky integrater of the
                    ganglionc ell layer.
                - tau_inl : float, optional, default: 18 ms
                    Time decay constant for the fast leaky integrater of the
                    inner nuclear layer (INL). It has been shown that even
                    epiretinal arrays can activate bipolar cells (in the INL),
                    which in turn influence GCL activity.
                - tau_ca : float, optional, default: 45.25 ms
                    Time decay constant for the charge accumulation.
                - scale_ca : float, optional, default: 42.1
                    Scaling factor applied to charge accumulation (used to be
                    called epsilon).
                - tau_slow : float, optional, default: 26.25 ms
                    Time decay constant for the slow leaky integrator.
                - scale_slow : float, optional, default: 1150.0
                    Scaling factor applied to the output of the cascade, to
                    make output values interpretable brightness values >= 0.
                - lweight : float, optional, default: 0.636
                    Relative weight applied to responses from bipolar cells
                    (weight of ganglion cells is 1).
                - aweight : float, optional, default: 0.5
                    Relative weight applied to anodic charges (weight of
                    cathodic charges is 1).
                - slope : float, optional, default: 3.0
                    Slope of the logistic function in the stationary
                    nonlinearity stage.
                - shift : float, optional, default: 15.0
                    Shift of the logistic function in the stationary
                    nonlinearity stage.

            - 'Nanduri2012':
                A model of temporal sensitivity as described in [1]_.

                Additional keyword arguments
                ----------------------------
                tsample : float, optional, default:
                tau1 : float, optional, default: 0.42 / 1000 (seconds)
                    Time decay constant for the fast leaky integrater of
                    the ganglion cell layer (GCL).
                tau2 : float, optional, default: 45.25 / 1000 (seconds)
                    Time decay constant for the charge accumulation, has
                    values between 38 - 57 ms.
                tau3 : float, optional, default: 26.25 / 1000 (seconds)
                    Time decay constant for the slow leaky integrator.
                    Default: 26.25 / 1000 s.
                eps : float, optional, default: 8.73
                    Scaling factor applied to charge accumulation (used to
                    be called epsilon).
                asymptote : float, optional, default: 14.0
                    Asymptote of the logistic function used in the
                    stationary nonlinearity stage.
                slope : float, optional, default: 3.0
                    Slope of the logistic function in the stationary
                    nonlinearity stage.
                shift : float, optional, default: 16.0
                    Shift of the logistic function in the stationary
                    nonlinearity stage.

            - 'Horsager2009':
                A model of temporal sensitivity as described in [2]_.

                Parameters
                ----------
                tsample : float, optional, default: 0.005 / 1000 seconds
                    Sampling time step (seconds).
                tau1 : float, optional, default: 0.42 / 1000 seconds
                    Time decay constant for the fast leaky integrater of the
                    ganglion cell layer (GCL).
                tau2 : float, optional, default: 45.25 / 1000 seconds
                    Time decay constant for the charge accumulation, has values
                    between 38 - 57 ms.
                tau3 : float, optional, default: 26.25 / 1000 seconds
                    Time decay constant for the slow leaky integrator.
                    Default: 26.25 / 1000 s.
                eps : float, optional, default: 8.73
                    Scaling factor applied to charge accumulation (used to be
                    called epsilon).
                beta : float, optional, default: 3.43
                    Power nonlinearity applied after half-rectification. The
                    original model used two different values, depending on
                    whether an experiment is at threshold (`beta`=3.43) or
                    above threshold (`beta`=0.83).

        Notes
        -----
        .. [1] D. Nanduri, I. Fine, A. Horsager, G. M. Boynton, M. S. Humayun,
               R. Greenberg, J. D. Weiland, "Frequency and Amplitude Modulation
               Have Different Effects on the Percepts Elicited by Retinal
               Stimulation", Investigative Ophthalmology & Visual Science 53,
               205-214, 2012.
        .. [2] A. Horsager, S. H. Greenwald, J. D. Weiland, M. S. Humayun,
               R. J. Greenberg, M. J. McMahon, G. M. Boynton, and I. Fine,
               "Predicting visual sensitivity in retinal prosthesis patients",
               Investigative Ophthalmology & Visual Science, 50(4):1483, 2009.



        """
        model_not_found = False
        if isinstance(model, six.string_types):
            # If `model` is a string, choose from existing models
            if model.lower() == 'latest':
                logging.getLogger(__name__).debug("Setting up latest model.")
                self.gcl = retina.TemporalModel(**kwargs)
            elif model.lower() in ['nanduri', 'nanduri2012']:
                logging.getLogger(__name__).debug("Setting up Nanduri (2012) "
                                                  "model.")
                self.gcl = retina.Nanduri2012(**kwargs)
            elif model.lower() in ['horsager', 'horsager2009']:
                logging.getLogger(__name__).debug("Setting up Horsager "
                                                  "(2009) model.")
                self.gcl = retina.Horsager2009(**kwargs)
            else:
                model_not_found = True
        elif isinstance(model, retina.BaseModel):
            # If `model` is not a string, must be of type BaseModel
            debug_str = "Setting up %s." % model.__module__
            logging.getLogger(__name__).debug(debug_str)
            self.gcl = model
        else:
            model_not_found = True

        if model_not_found:
            err_str = "Model '%s' not found. Choose from: " % model
            err_str += ", ".join(retina.SUPPORTED_TEMPORAL_MODELS)
            err_str += " or provide your own retina.BaseModel instance."
            raise ValueError(err_str)

    def _set_layers(self):
        """Sets up all layers whose setters have not been called by the user

        This function makes sure all necessary parts of the simulation are
        initialized before transforming stimuli to percepts.
        Layers not initialized by the user will simply be initialized with
        default argument values.
        """
        if self.ofl is None:
            self.set_optic_fiber_layer()
        if self.gcl is None:
            self.set_ganglion_cell_layer('latest')

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
        t_percept : float, optional, default: inherit from `stim` object
            The desired time sampling of the output (seconds).
        tol : float, optional, default: 0.05
            Ignore pixels whose effective current is smaller than a fraction
            `tol` of the max value.
        layers : list, optional, default: ['OFL', 'GCL', 'INL']
            A list of retina layers to simulate (order does not matter):
            - 'OFL': Includes the optic fiber layer in the simulation.
                     If omitted, the tissue activation map will not account
                     for axon streaks.
            - 'GCL': Includes the ganglion cell layer in the simulation.
            - 'INL': Includes the inner nuclear layer in the simulation.
                     If omitted, bipolar cell activity does not contribute
                     to ganglion cell activity.

        Returns
        -------
        A utils.TimeSeries object whose data container comprises the predicted
        brightness over time at each retinal location (x, y), with the last
        dimension of the container representing time (t).

        Examples
        --------
        Simulate a single-electrode array:

        >>> import pulse2percept as p2p
        >>> implant = p2p.implants.ElectrodeArray('subretinal', 0, 0, 0)
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
        not_supported = np.array([l not in retina.SUPPORTED_LAYERS
                                  for l in layers], dtype=bool)
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
        pt_data = [pt.data for pt in pt_list]

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
                    process_pixel |= np.any(ecs[yy, xx, 0, :]

                                            >= tol * lmax[0, :])
                if ('GCL' or 'OFL') in layers:
                    process_pixel |= np.any(ecs[yy, xx, 1, :]
                                            >= tol * lmax[1, :])

                if process_pixel:
                    ecs_list.append(ecs[yy, xx])
                    idx_list.append([yy, xx])

        s_info = "tol=%.1f%%, %d/%d px selected" % (tol * 100, len(ecs_list),
                                                    np.prod(ecs.shape[:2]))
        logging.getLogger(__name__).info(s_info)

        sr_list = utils.parfor(self.gcl.model_cascade,
                               ecs_list, n_jobs=self.n_jobs,
                               engine=self.engine, scheduler=self.scheduler,
                               func_args=[pt_data, layers, self.use_jit])
        bm = np.zeros(self.ofl.gridx.shape
                     + (sr_list[0].data.shape[-1], ))
        idxer = tuple(np.array(idx_list)[:, i] for i in range(2))
        bm[idxer] = [sr.data for sr in sr_list]
        percept = utils.TimeSeries(sr_list[0].tsample, bm)

        # It is possible to specify an additional sampling rate for the
        # percept: If different from the input sampling rate, need to resample.
        if t_percept != percept.tsample:
            percept = percept.resample(t_percept)

        logging.getLogger(__name__).info("Done.")

        return percept


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
