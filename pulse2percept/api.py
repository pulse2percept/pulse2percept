import numpy as np
import logging

import pulse2percept as p2p


class Simulation(object):

    def __init__(self, name, implant, engine='joblib', dojit=True,
                 num_jobs=-1):
        """Generates a simulation framework

        Parameters
        ----------
        name : str
            Name of the simulation
        implant : implants.ElectrodeArray
            An implants.ElectrodeArray object that describes the implant.
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
        if not isinstance(implant, p2p.implants.ElectrodeArray):
            e_s = "`implant` must be of type p2p.implants.ElectrodeArray"
            raise TypeError(e_s)

        logging.getLogger(__name__).info('Create simulation "%s".' % name)
        self.name = name
        self.implant = implant
        self.engine = engine
        self.dojit = dojit
        self.num_jobs = num_jobs

        self.ofl = None
        self.ofl_streaks = True

        self.gcl = None

    def pulse2percept(self, stim, t_pulse=None, t_percept=None, tol=0.05):
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
            given, the output sampling time step will correspond to the input
            time step (as defined by `stim` and `tm`).
            Default: None.
        tol : float, optional
            Ignore pixels whose effective current is smaller than a fraction
            `tol` of the max value.
            Default: 0.05.

        Returns
        -------
        A utils.TimeSeries object whose data container comprises the predicted
        brightness over time at each retinal location (x, y), with the last
        dimension of the container representing time (t).

        Examples
        --------
        Stimulate a single-electrode array:

        >>> import pulse2percept as p2p
        >>> implant = p2p.implants.ElectrodeArray('subretinal', 0, 0, 0, 0)
        >>> stim = p2p.stimuli.Psycho2Pulsetrain(tsample=5e-6, freq=50, amp=20)
        >>> sim = p2p.Simulation("Single-electrode example", implant)
        >>> percept = sim.pulse2percept(stim)  # doctest: +SKIP

        Stimulate a single electrode ('C3') of an Argus I array centered on the
        fovea:

        >>> import pulse2percept as p2p
        >>> implant = p2p.implants.ArgusI()
        >>> stim = {'C3': p2p.stimuli.Psycho2Pulsetrain(tsample=5e-6, freq=50,
        ...                                              amp=20)}
        >>> sim = p2p.Simulation("ArgusI example", implant)
        >>> resp = sim.pulse2percept(stim, implant)  # doctest: +SKIP
        """
        self._init()

        # Parse `stim` (either single pulse train or a list/dict of pulse
        # trains), and generate a list of pulse trains, one for each electrode
        pt_list = p2p.stimuli.parse_pulse_trains(stim, self.implant)

        self.gcl.tsample = pt_list[0].tsample

        # Perform any necessary calculations per electrode
        pt_list = p2p.utils.parfor(self.gcl.calc_per_electrode,
                                   pt_list, engine=self.engine,
                                   n_jobs=self.num_jobs)

        # Which layer to simulate is given by implant type.
        # For now, both implant types process the same two layers. In the
        # future, these layers might differ. Order doesn't matter.
        if self.implant.etype == 'epiretinal':
            dolayers = ['NFL', 'INL']  # nerve fiber layer
        elif self.implant.etype == 'subretinal':
            dolayers = ['NFL', 'INL']  # inner nuclear layer
        else:
            e_s = "Supported electrode types are 'epiretinal', 'subretinal'"
            raise ValueError(e_s)

        # Derive the effective current spread
        if self.ofl_streaks:
            ecs, _ = self.ofl.electrode_ecs(self.implant)
        else:
            _, ecs = self.ofl.electrode_ecs(self.implant)

        # Calculate the max of every current spread map
        lmax = np.zeros((2, ecs.shape[-1]))
        if 'INL' in dolayers:
            lmax[0, :] = ecs[:, :, 0, :].max(axis=(0, 1))
        if 'NFL' in dolayers:
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
                if 'INL' in dolayers:
                    # For this pixel: Check if the ecs in any layer is large
                    # enough compared to the max across pixels within the layer
                    process_pixel |= np.any(ecs[yy, xx, 0, :] >=
                                            tol * lmax[0, :])
                if 'NFL' in dolayers:
                    process_pixel |= np.any(ecs[yy, xx, 1, :] >=
                                            tol * lmax[1, :])

                if process_pixel:
                    ecs_list.append(ecs[yy, xx])
                    idx_list.append([yy, xx])

        s_info = "tol=%.1f%%, %d/%d px selected" % (tol * 100, len(ecs_list),
                                                    np.prod(ecs.shape[:2]))
        logging.getLogger(__name__).info(s_info)

        logging.getLogger(__name__).info("Starting computation...")
        sr_list = p2p.utils.parfor(self.gcl.calc_per_pixel,
                                   ecs_list, n_jobs=self.num_jobs,
                                   engine=self.engine,
                                   func_args=[pt_list, dolayers, self.dojit])
        bm = np.zeros(self.ofl.gridx.shape +
                      (sr_list[0].data.shape[-1], ))
        idxer = tuple(np.array(idx_list)[:, i] for i in range(2))
        bm[idxer] = [sr.data for sr in sr_list]
        percept = p2p.utils.TimeSeries(sr_list[0].tsample, bm)

        if t_percept != percept.tsample:
            percept = percept.resample(t_percept)

        logging.getLogger(__name__).info("Done.")

        return percept

    def set_ofl(self, sampling=100, axon_lambda=2, rot_deg=0,
                x_range=None, y_range=None,
                loadpath='../', save_data=True, streaks_enabled=True):
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
        loadpath : str
            Relative path where to look for existing retina file.
            Default: '../'
        save_data : bool
            Flag whether to save the data to a new retina file (True) or not
            (False). The file name is automatically generated from all
            specified input arguments.
            Default: True.
        streaks_enabled : bool, optional
            Flag whether to use a tissue activation map that includes axon
            streaks (True) or not (False).
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
        elif isinstance(x_range, (list, tuple, np.array)):
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
        elif isinstance(y_range, (list, np.array)):
            if len(y_range) != 2 or y_range[1] < y_range[0]:
                e_s = "y_range must be a list [ylo, yhi] where ylo <= yhi."
                raise ValueError(e_s)
            ylo = y_range[0]
            yhi = y_range[1]
        else:
            raise ValueError("x_range must be a list [xlo, xhi] or None.")

        # Generate the grid from the above specs
        self.ofl = p2p.retina.Grid(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi,
                                   sampling=sampling,
                                   axon_lambda=axon_lambda,
                                   rot=np.deg2rad(rot_deg),
                                   save_data=save_data)
        self.ofl_streaks = streaks_enabled

    def set_gcl(self, t_gcl=0.005 / 1000,
                tau_gcl=0.42 / 1000, tau_inl=18.0 / 1000,
                tau_ca=45.25 / 1000, scale_ca=42.1,
                tau_slow=26.25 / 1000, scale_slow=1150.0,
                lweight=0.636, aweight=0.5,
                slope=3.0, shift=15.0):
        """Sets parameters of the ganglion cell layer (GCL)

        Parameters
        ----------
        t_gcl : float
            Sampling time step (seconds) for the ganglion cell layer.
            Default: 0.005 / 1000 s.
        tau_gcl : float
            Time decay constant for the fast leaky integrater of the ganglion
            cell layer.
            Default: 45.25 / 1000 s.
        tau_inl : float
            Time decay constant for the fast leaky integrater of the inner
            nuclear layer (INL); i.e., bipolar cell layer.
            This is only important in combination with subretinal electrode
            arrays. Default: 18.0 / 1000 s.
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
        tm = p2p.retina.TemporalModel(tsample=t_gcl, tau_gcl=tau_gcl,
                                      tau_inl=tau_inl, tau_ca=tau_ca,
                                      scale_ca=scale_ca, tau_slow=tau_slow,
                                      scale_slow=scale_slow,
                                      lweight=lweight, aweight=aweight,
                                      slope=slope, shift=shift)
        self.gcl = tm

    def plot_fundus(self, ax=None, stim=None):
        """Plot the implant on the retinal surface akin to a fundus photopgraph

        This function plots an electrode array on top of the axon streak map
        of the retina, akin to a fundus photograph. A blue rectangle highlights
        the area of the retinal surface that is being simulated.
        If `stim` is passed, activated electrodes will be highlighted.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            A Matplotlib axes object. If None given, a new one will be created.
            Default: None
        stim : utils.TimeSeries|list|dict, optional
            An input stimulus, as passed to p2p.pulse2percept. If given,
            activated electrodes will be highlighted in the plot.
            Default: None

        Returns
        -------
        Returns a handle to the created figure (`fig`) and axes element (`ax`).
        """
        from matplotlib import patches

        fig = None
        if ax is None:
            # No axes object given: create
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, figsize=(10, 8))

        ax.set_axis_bgcolor('black')
        ax.plot(self.ofl.jan_x[:, ::5], -self.ofl.jan_y[:, ::5],
                c=(0.5, 1, 0.5))

        # Draw in the the retinal patch we're simulating.
        # This defines the size of our "percept" image below.
        dva_xmin = p2p.retina.ret2dva(self.ofl.gridx.min())
        dva_ymin = -p2p.retina.ret2dva(self.ofl.gridy.max())
        patch = patches.Rectangle((dva_xmin, dva_ymin),
                                  p2p.retina.ret2dva(self.ofl.range_x),
                                  p2p.retina.ret2dva(self.ofl.range_y),
                                  alpha=0.7)
        ax.add_patch(patch)

        # Highlight location of stimulated electrodes
        if stim is not None:
            for key in stim:
                ax.plot(p2p.retina.ret2dva(self.implant[key].x_center),
                        -p2p.retina.ret2dva(self.implant[key].y_center), 'oy',
                        markersize=np.sqrt(self.implant[key].radius) * 2)

        # Plot all electrodes and their label
        for e in self.implant.electrodes:
            ax.text(p2p.retina.ret2dva(e.x_center + 10),
                    -p2p.retina.ret2dva(e.y_center + 5),
                    e.name, color='white', size='x-large')
            ax.plot(p2p.retina.ret2dva(e.x_center),
                    -p2p.retina.ret2dva(e.y_center), 'ow',
                    markersize=np.sqrt(e.radius))

        ax.set_aspect('equal')
        ax.set_xlim(-20, 20)
        ax.set_xlabel('visual angle (deg)')
        ax.set_ylim(-15, 15)
        ax.set_ylabel('visual angle (deg)')
        ax.set_title('Image flipped (upper retina = upper visual field)')
        ax.grid('off')

        return fig, ax

    def _init(self):
        """Initializes the model

        This function makes sure all necessary parts of the simulation are
        initialized before transforming stimuli. Uninitialized layers will
        simply be initialized with default argument values.
        """
        if self.ofl is None:
            self.set_ofl()
        if self.gcl is None:
            self.set_gcl()


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
