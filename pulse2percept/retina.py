import numpy as np

import scipy.special as ss
import scipy.spatial as spat
import abc
import six
import os.path
import logging

from . import fast_retina as fr
from . import utils


SUPPORTED_LAYERS = ['INL', 'GCL', 'OFL']
SUPPORTED_TEMPORAL_MODELS = ['latest', 'Nanduri2012', 'Horsager2009']


class Grid(object):
    """Represent the retinal coordinate frame"""

    def __init__(self, x_range=(-1000.0, 1000.0), y_range=(-1000.0, 1000.0),
                 eye='RE', sampling=25, n_axons=501, phi_range=(-180.0, 180.0),
                 n_rho=801, rho_range=(4.0, 45.0), loc_od=(15.5, 1.5),
                 sensitivity_rule='decay', contribution_rule='max',
                 decay_const=2.0, alpha=14000, powermean_exp=None, datapath='.',
                 save_data=True, engine='joblib', scheduler='threading',
                 n_jobs=-1):
        """Generates a spatial grid representing the retinal coordinate frame

        This function generates the coordinate system for the retina
        and an axon map. As this can take a while, the function will
        first look for an already existing file in the directory `datapath`
        that was automatically created from an earlier call to this function,
        before it attempts to generate new grid from scratch.

        Parameters
        ----------
        x_range : (xlo, xhi), optional, default: xlo=-1000, xhi=1000
           Extent of the retinal coverage (microns) in horizontal dimension.
        y_range : (ylo, yhi), optional, default: ylo=-1000, ylo=1000
           Extent of the retinal coverage (microns) in vertical dimension.
        eye : {'LE', 'RE'}, optional, default: 'RE'
            Which eye to simulate (left/right). The optic disc is at (15, 2)
            deg in a right eye, and at (-15, 2) deg in a left eye.
        sampling : float, optional, default: 25
            Spatial sampling step (microns) for the grid.
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
        loc_od: (x_od, y_od), optional, default: (15.0, 2.0)
            Location of the center of the optic disc(x, y) in Cartesian
            coordinates.
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
                peak sensitivity. See Figure 2 in Jeng, Tang, Molnar, Desai,
                and Fried (2011). The sodium channel band shapes the response
                to electric stimulation in retinal ganglion cells. J Neural Eng
                8 (036022).
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
                (1.0 / powermean_exp).
        decay_const : float, optional, default: 2.0
            When `sensitivity_rule` is set to 'decay', specifies the decay
            constant of the exponential fall-off.
        alpha : float, optional, default: 14000
            Current spread parameter for passive current spread from the electrode.
        powermean_exp : float, optional, default: None
            When `sensitivity_rule` is set to 'mean', specifies the exponent of
            the generalized (power) mean function. The power mean is calculated
            as np.mean(x ** powermean_exp) ** (1.0 / powermean_exp).
        datapath : str, optional, default: current directory
            Relative path where to look for existing retina files, and where to
            store new files.
        save_data : bool, optional, default: True
            Flag whether to save the data to a new file (True) or not (False).
            The file name is automatically generated from all specified input
            arguments.
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
        n_jobs : int, optional, default: -1
            Number of cores (threads) to run the model on in parallel.
            Specify -1 to use as many cores as available.
        """
        if not isinstance(x_range, (tuple, list, np.ndarray)):
            raise ValueError('`x_range` must be a tuple (`xlo`, `xhi`).')
        if x_range[0] > x_range[1]:
            raise ValueError('Lower bound on x cannot be larger than the '
                             'upper bound.')
        if not isinstance(y_range, (tuple, list, np.ndarray)):
            raise ValueError('`y_range` must be a tuple (`ylo`, `yhi`).')
        if y_range[0] > y_range[1]:
            raise ValueError('Lower bound on y cannot be larger than the '
                             'upper bound.')
        if n_axons < 1:
            raise ValueError('Number of axons must be >= 1.')
        if np.any(np.abs(phi_range) > 180.0):
            raise ValueError('phi must be within [-180, 180].')
        if phi_range[0] > phi_range[1]:
            raise ValueError('Lower bound on phi cannot be larger than the '
                             'upper bound.')
        self.x_range = x_range
        self.y_range = y_range
        self.sampling = sampling
        self.sensitivity_rule = sensitivity_rule
        self.contribution_rule = contribution_rule
        self.decay_const = decay_const
        self.alpha = alpha
        self.powermean_exp = powermean_exp
        self.engine = engine
        self.scheduler = scheduler
        self.n_jobs = n_jobs

        if np.abs(loc_od[1]) > 10.0:
            logging.getLogger(__name__).warn("The Jansonius model might "
                                             "misbehave if `loc_od` has "
                                             "a y value > 10.")

        if eye.upper() == 'RE':
            if loc_od[0] <= 0:
                w_s = "In a right eye, the optic disc usually has x > 0 - "
                w_s += "currently at (%.1f, %.1f)." % (loc_od[0], loc_od[1])
                logging.getLogger(__name__).warn(w_s)
        elif eye.upper() == 'LE':
            if loc_od[0] > 0:
                w_s = "In a left eye, the optic disc usually has x < 0 - "
                w_s += "currently at (%.1f, %.1f)." % (loc_od[0], loc_od[1])
                logging.getLogger(__name__).warn(w_s)
        else:
            e_s = "Unknown eye string '%s': Choose from 'LE', 'RE'." % eye
            raise ValueError(e_s)

        # Include endpoints in meshgrid
        xlo, xhi = x_range
        ylo, yhi = y_range
        num_x = int((xhi - xlo) / sampling + 1)
        num_y = int((yhi - ylo) / sampling + 1)
        self.gridx, self.gridy = np.meshgrid(np.linspace(xlo, xhi, num_x),
                                             np.linspace(ylo, yhi, num_y),
                                             indexing='xy')

        # Create descriptive filename based on input args
        filename = "retina_%s_s%d_a%d_r%d_%dx%d.npz" % (eye, sampling, n_axons,
                                                        n_rho, xhi - xlo,
                                                        yhi - ylo)
        filename = os.path.join(datapath, filename)

        # There are some variables, like `sensitivity_rule` and `decay_const`
        # that are only needed in the effective current calculation, not for
        # the grid and axon maps - so not included here:
        grid_dict = {'x_range': x_range, 'y_range': y_range, 'eye': eye,
                     'n_axons': n_axons, 'phi_range': phi_range,
                     'n_rho': n_rho, 'rho_range': rho_range,
                     'sampling': sampling, 'loc_od': loc_od}

        # Assign all elements in the dictionary to this object
        for key, value in six.iteritems(grid_dict):
            setattr(self, key, value)

        # Check if such a file already exists. If so, load parameters and
        # make sure they are the same as specified above. Else, create new.
        need_new_grid = False
        if not os.path.exists(filename):
            need_new_grid = True
        else:
            logging.getLogger(__name__).info('Loading file "%s".' % filename)
            try:
                load_grid_dict = six.moves.cPickle.load(open(filename, 'rb'))
            except six.moves.cPickle.UnpicklingError:
                msg = 'UnpicklingError: Could not load file "%s".'
                logging.getLogger(__name__).info(msg)
                need_new_grid = True
                load_grid_dict = {}

            # Make sure all relevant variables are present and have the right
            # values:
            for key, value in six.iteritems(grid_dict):
                if key not in load_grid_dict:
                    logging.getLogger(__name__).info('File out of date.')
                    need_new_grid = True
                    break
                if value != load_grid_dict[key]:
                    logging.getLogger(__name__).info('File out of date.')
                    need_new_grid = True
                    break

        # At this point we know whether we need to generate a new retina:
        if need_new_grid:
            info_str = "Generating new file '%s'." % filename
            logging.getLogger(__name__).info(info_str)

            # Grow a number `n_axons` of axon bundles with orientations in
            # `phi_range`
            phi = np.linspace(*phi_range, num=n_axons)
            func_kwargs = {'n_rho': n_rho, 'rho_range': rho_range,
                           'loc_od': loc_od, 'eye': eye}
            self.axon_bundles = utils.parfor(jansonius2009, phi,
                                             func_kwargs=func_kwargs,
                                             engine=engine, n_jobs=n_jobs,
                                             scheduler=scheduler)
            grid_dict['axon_bundles'] = self.axon_bundles

            # Assume there is a neuron at every grid location: Use the above
            # axon bundles to assign an axon to each neuron
            xg, yg = ret2dva(self.gridx), ret2dva(self.gridy)
            pos_xy = np.column_stack((xg.ravel(), yg.ravel()))
            self.axons = utils.parfor(find_closest_axon, pos_xy,
                                      func_args=[self.axon_bundles])
            grid_dict['axons'] = self.axons

            # For every axon segment, calculate distance to soma. Snap axon
            # locations to the grid using a nearest-neighbor tree structure:
            func_kwargs = {'tree': spat.cKDTree(pos_xy)}
            self.axon_distances = utils.parfor(axon_dist_from_soma, self.axons,
                                               func_args=[xg, yg],
                                               func_kwargs=func_kwargs,
                                               engine=engine, n_jobs=n_jobs,
                                               scheduler=scheduler)
            grid_dict['axon_distances'] = self.axon_distances

            # Save the variables, together with metadata about the grid:
            if save_data:
                six.moves.cPickle.dump(grid_dict, open(filename, 'wb'))
        else:
            # Assign all elements in the loaded dictionary to this object
            for key, value in six.iteritems(load_grid_dict):
                setattr(self, key, value)

    def current2effectivecurrent(self, current_spread):
        """

        Converts a current spread map to an 'effective' current spread map, by
        passing the map through a mapping of axon streaks.

        Parameters
        ----------
        cs: array
            The 2D spread map in retinal space

        Returns
        -------
        ecm: array
            The effective current spread, a time series of the same size as
            the current map, where each pixel is the dot product of the pixel
            values in ecm along the pixels in the list in axon_map, weighted
            by the weights axon map.
        """
        contrib = utils.parfor(axon_contribution, self.axon_distances,
                               func_args=[current_spread], engine=self.engine,
                               func_kwargs={
                                   'sensitivity_rule': self.sensitivity_rule,
                                   'contribution_rule': self.contribution_rule,
                                   'decay_const': self.decay_const,
                                   'powermean_exp': self.powermean_exp
                               },
                               scheduler=self.scheduler, n_jobs=self.n_jobs)

        ecs = np.zeros_like(current_spread)
        px_contrib = list(filter(None, contrib))
        for idx, value in px_contrib:
            ecs.ravel()[idx] = value

        # Normalize so that the max of `ecs` is the same as `current_spread`
        return ecs / (ecs.max() + np.finfo(float).eps) * current_spread.max()

    def electrode_ecs(self, implant, n=1.69):
        """
        Gather current spread and effective current spread for each electrode
        within both the bipolar and the ganglion cell layer

        Parameters
        ----------
        implant: implants.ElectrodeArray
            An implants.ElectrodeArray instance describing the implant.
        n: float
            Current spread parameter

        Returns
        -------
        ecs: contains n arrays containing the the effective current
            spread within various layers
            for each electrode in the array respectively.

        See also
        --------
        Electrode.current_spread
        """

        cs = np.zeros((self.gridx.shape[0], self.gridx.shape[1],
                       2, len(implant.electrodes)), dtype=float)
        ecs = np.zeros((self.gridx.shape[0], self.gridx.shape[1],
                        2, len(implant.electrodes)), dtype=float)

        for i, e in enumerate(implant.electrodes):
            cs[..., 0, i] = e.current_spread(self.gridx, self.gridy,
                                             layer='INL', alpha=self.alpha, n=n)
            ecs[..., 0, i] = cs[..., 0, i]
            cs[..., 1, i] = e.current_spread(self.gridx, self.gridy,
                                             layer='OFL', alpha=self.alpha, n=n)
            ecs[:, :, 1, i] = self.current2effectivecurrent(cs[..., 1, i])

        return ecs, cs


@six.add_metaclass(abc.ABCMeta)
class BaseModel():
    """Abstract base class for all models of temporal sensitivity.

    This class provides a standard template for all models of temporal
    sensitivity.
    """

    def set_kwargs(self, warn_inexistent, **kwargs):
        """Overwrite any given keyword arguments

        Parameters
        ----------
        warn_inexistent: bool
            If True, displays a warning message if a keyword is provided that
            is not recognized by the temporal model.
        """
        for key, value in six.iteritems(kwargs):
            if not hasattr(self, key) and warn_inexistent:
                w_s = "Unknown class attribute '%s'" % key
                logging.getLogger(__name__).warning(w_s)
            setattr(self, key, value)

    def __init__(self, **kwargs):
        self.set_kwargs(True, **kwargs)

    @abc.abstractmethod
    def model_cascade(self, in_arr, pt_list, layers, use_jit):
        """Abstract base ganglion cell model

        Parameters
        ----------
        in_arr: array - like
            A 2D array specifying the effective current values at a particular
            spatial location(pixel); one value per retinal layer and
            electrode. Dimensions: <  # layers x #electrodes>
        pt_list: list
            List of pulse train 'data' containers.
            Dimensions: <  # electrodes x #time points>
        layers: list
            List of retinal layers to simulate.
            Choose from:
            - 'OFL': optic fiber layer
            - 'GCL': ganglion cell layer
            - 'INL': inner nuclear layer
        use_jit: bool
            If True, applies just - in-time(JIT) compilation to expensive
            computations for additional speed - up(requires Numba).
        """
        pass

    # Static attribute
    tsample = 0.005 / 1000


class Horsager2009(BaseModel):
    """Model of temporal sensitivity (Horsager et al. 2009)

    This class implements the model of temporal sensitivty as described in:
    > A Horsager, SH Greenwald, JD Weiland, MS Humayun, RJ Greenberg,
    > MJ McMahon, GM Boynton, and I Fine(2009). Predicting visual sensitivity
    > in retinal prosthesis patients. Investigative Ophthalmology & Visual
    > Science, 50(4): 1483.

    Parameters
    ----------
    tsample: float, optional, default: 0.005 / 1000 seconds
        Sampling time step(seconds).
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
        Scaling factor applied to charge accumulation(used to be called
        epsilon).
    beta: float, optional, default: 3.43
        Power nonlinearity applied after half - rectification. The original model
        used two different values, depending on whether an experiment is at
        threshold(`beta`=3.43) or above threshold(`beta`=0.83).
    """

    def __init__(self, **kwargs):
        self.tsample = 0.01 / 1000
        self.tau1 = 0.42 / 1000
        self.tau2 = 45.25 / 1000
        self.tau3 = 26.25 / 1000
        self.epsilon = 2.25
        self.beta = 3.43

        # Overwrite any given keyword arguments, print warning message (True)
        # if attempting to set an unrecognized keyword
        self.set_kwargs(True, **kwargs)

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


class Nanduri2012(BaseModel):
    """Model of temporal sensitivity (Nanduri et al. 2012)

    This class implements the model of temporal sensitivity as described in:
    > Nanduri, Fine, Horsager, Boynton, Humayun, Greenberg, Weiland(2012).
    > Frequency and Amplitude Modulation Have Different Effects on the Percepts
    > Elicited by Retinal Stimulation. Investigative Ophthalmology & Visual
    > Science January 2012, Vol.53, 205 - 214. doi: 10.1167 / iovs.11 - 8401.

    Parameters
    ----------
    tsample: float, optional, default: 0.005 / 1000 seconds
        Sampling time step(seconds).
    tau1: float, optional, default: 0.42 / 1000 seconds
        Time decay constant for the fast leaky integrater of the ganglion
        cell layer(GCL).
    tau2: float, optional, default: 45.25 / 1000 seconds
        Time decay constant for the charge accumulation, has values
        between 38 - 57 ms.
    tau3: float, optional, default: 26.25 / 1000 seconds
        Time decay constant for the slow leaky integrator.
        Default: 26.25 / 1000 s.
    eps: float, optional, default: 8.73
        Scaling factor applied to charge accumulation(used to be called
        epsilon).
    asymptote: float, optional, default: 14.0
        Asymptote of the logistic function used in the stationary
        nonlinearity stage.
    slope: float, optional, default: 3.0
        Slope of the logistic function in the stationary nonlinearity
        stage.
    shift: float, optional, default: 16.0
        Shift of the logistic function in the stationary nonlinearity
        stage.
    """

    def __init__(self, **kwargs):
        # Set default values of keyword arguments
        self.tau1 = 0.42 / 1000
        self.tau2 = 45.25 / 1000
        self.tau3 = 26.25 / 1000
        self.eps = 8.73
        self.asymptote = 14.0
        self.slope = 3.0
        self.shift = 16.0

        # Nanduri (2012) has a term in the stationary nonlinearity step that
        # depends on future values of R3: max_t(R3). Because the finite
        # difference model cannot look into the future, we need to set a
        # scaling factor here:
        self.maxR3 = 100.0

        # Overwrite any given keyword arguments, print warning message (True)
        # if attempting to set an unrecognized keyword
        self.set_kwargs(True, **kwargs)

    def calc_layer_current(self, in_arr, pt_list):
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
        """
        in_flat = in_arr[1, :].reshape(-1).astype(float)
        pt_arr = np.array(pt_list, dtype=float)
        return np.array(fr.nanduri2012_calc_layer_current(in_flat, pt_arr))

    def model_cascade(self, in_arr, pt_list, layers, use_jit):
        """Nanduri model cascade

        Parameters
        ----------
        in_arr: array - like
            A 2D array specifying the effective current values
            at a particular spatial location(pixel); one value
            per retinal layer and electrode.
            Dimensions: < #layers x #electrodes>
        pt_list: list
            List of pulse train 'data' containers.
            Dimensions: < #electrodes < #time points > >
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
        if 'GCL' not in layers and 'OFL' not in layers:
            raise ValueError("Acceptable values for `layers` are: 'GCL', "
                             "'OFL'.")

        pulse = self.calc_layer_current(in_arr, pt_list)
        percept = fr.nanduri2012_model_cascade(pulse, self.tsample,
                                               self.tau1, self.tau2, self.tau3,
                                               self.asymptote, self.shift,
                                               self.slope, self.eps,
                                               self.maxR3)
        return utils.TimeSeries(self.tsample, percept)


class TemporalModel(BaseModel):
    """Latest edition of the temporal sensitivity model (experimental)

    This class implements the latest version of the temporal sensitivity
    model(experimental). As such, the model might still change from version
    to version. For more stable implementations, please refer to other,
    published models(see `p2p.retina.SUPPORTED_TEMPORAL_MODELS`).

    Parameters
    ----------
    tsample: float, optional, default: 0.005 / 1000 seconds
        Sampling time step(seconds).
    tau_gcl: float, optional, default: 45.25 / 1000 seconds
        Time decay constant for the fast leaky integrater of the ganglion
        cell layer(GCL).
        This is only important in combination with epiretinal electrode
        arrays.
    tau_inl: float, optional, default: 18.0 / 1000 seconds
        Time decay constant for the fast leaky integrater of the inner
        nuclear layer(INL); i.e., bipolar cell layer.
        This is only important in combination with subretinal electrode
        arrays.
    tau_ca: float, optional, default: 45.25 / 1000 seconds
        Time decay constant for the charge accumulation, has values
        between 38 - 57 ms.
    scale_ca: float, optional, default: 42.1
        Scaling factor applied to charge accumulation(used to be called
        epsilon).
    tau_slow: float, optional, default: 26.25 / 1000 seconds
        Time decay constant for the slow leaky integrator.
    scale_slow: float, optional, default: 1150.0
        Scaling factor applied to the output of the cascade, to make
        output values interpretable brightness values >= 0.
    lweight: float, optional, default: 0.636
        Relative weight applied to responses from bipolar cells(weight
        of ganglion cells is 1).
    aweight: float, optional, default: 0.5
        Relative weight applied to anodic charges(weight of cathodic
        charges is 1).
    slope: float, optional, default: 3.0
        Slope of the logistic function in the stationary nonlinearity
        stage.
    shift: float, optional, default: 15.0
        Shift of the logistic function in the stationary nonlinearity
        stage.
    """

    def __init__(self, **kwargs):
        # Set default values of keyword arguments
        self.tau_gcl = 0.42 / 1000
        self.tau_inl = 18.0 / 1000
        self.tau_ca = 45.25 / 1000
        self.tau_slow = 26.25 / 1000
        self.scale_ca = 42.1
        self.scale_slow = 1150.0
        self.lweight = 0.636
        self.aweight = 0.5
        self.slope = 3.0
        self.shift = 15.0

        # Overwrite any given keyword arguments, print warning message (True)
        # if attempting to set an unrecognized keyword
        self.set_kwargs(True, **kwargs)

        # perform one-time setup calculations
        _, self.gamma_inl = utils.gamma(1, self.tau_inl, self.tsample)
        _, self.gamma_gcl = utils.gamma(1, self.tau_gcl, self.tsample)

        # gamma_ca is used to calculate charge accumulation
        _, self.gamma_ca = utils.gamma(1, self.tau_ca, self.tsample)

        # gamma_slow is used to calculate the slow response
        _, self.gamma_slow = utils.gamma(3, self.tau_slow, self.tsample)

    def fast_response(self, stim, gamma, method, use_jit=True):
        """Fast response function

        Convolve a stimulus `stim` with a temporal low - pass filter `gamma`.

        Parameters
        ----------
        stim: array
           Temporal signal to process, stim(r, t) in Nanduri et al. (2012).
        use_jit: bool, optional
           If True (default), use numba just - in-time compilation.
        usefft: bool, optional
           If False (default), use sparseconv, else fftconvolve.

        Returns
        -------
        Fast response, b2(r, t) in Nanduri et al. (2012).

        Notes
        -----
        The function utils.sparseconv can be much faster than np.convolve and
        signal.fftconvolve if `stim` is sparse and much longer than the
        convolution kernel.
        The output is not converted to a TimeSeries object for speedup.
        """
        conv = utils.conv(stim, gamma, mode='full', method=method,
                          use_jit=use_jit)

        # Cut off the tail of the convolution to make the output signal
        # match the dimensions of the input signal.
        return self.tsample * conv[:stim.shape[-1]]

    def charge_accumulation(self, ecm):
        """Calculates the charge accumulation

        Charge accumulation is calculated on the effective input current
        `ecm`, as opposed to the output of the fast response stage.

        Parameters
        ----------
        ecm: array - like
            A 2D array specifying the effective current values at a particular
            spatial location(pixel); one value per retinal layer, averaged
            over all electrodes through that pixel.
            Dimensions: <  # layers x #time points>
        """
        ca = np.zeros_like(ecm)

        for i in range(ca.shape[0]):
            summed = self.tsample * np.cumsum(np.abs(ecm[i, :]))
            conved = self.tsample * utils.conv(summed, self.gamma_ca,
                                               mode='full', method='fft')
            ca[i, :] = self.scale_ca * conved[:ecm.shape[-1]]
        return ca

    def stationary_nonlinearity(self, stim):
        """Stationary nonlinearity

        Nonlinearly rescale a temporal signal `stim` across space and time,
        based on a sigmoidal function dependent on the maximum value of `stim`.
        This is Box 4 in Nanduri et al. (2012).
        The parameter values of the asymptote, slope, and shift of the logistic
        function are given by self.asymptote, self.slope, and self.shift,
        respectively.

        Parameters
        ----------
        stim: array
           Temporal signal to process, stim(r, t) in Nanduri et al. (2012).

        Returns
        -------
        Rescaled signal, b4(r, t) in Nanduri et al. (2012).

        Notes
        -----
        Conversion to TimeSeries is avoided for the sake of speedup.
        """
        # use expit (logistic) function for speedup
        sigmoid = ss.expit((stim.max() - self.shift) / self.slope)
        return stim * sigmoid

    def slow_response(self, stim):
        """Slow response function

        Convolve a stimulus `stim` with a low - pass filter(3 - stage gamma)
        with time constant self.tau_slow.
        This is Box 5 in Nanduri et al. (2012).

        Parameters
        ----------
        stim: array
           Temporal signal to process, stim(r, t) in Nanduri et al. (2012)

        Returns
        -------
        Slow response, b5(r, t) in Nanduri et al. (2012).

        Notes
        -----
        This is by far the most computationally involved part of the perceptual
        sensitivity model.
        Conversion to TimeSeries is avoided for the sake of speedup.
        """
        # No need to zero-pad: fftconvolve already takes care of optimal
        # kernel/data size
        conv = utils.conv(stim, self.gamma_slow, method='fft', mode='full')

        # Cut off the tail of the convolution to make the output signal match
        # the dimensions of the input signal.
        return self.scale_slow * self.tsample * conv[:stim.shape[-1]]

    def calc_layer_current(self, ecs_item, pt_list, layers):
        """For a given pixel, calculates the effective current for each retinal
           layer over time

        This function operates at a single - pixel level: It calculates the
        combined current from all electrodes through a spatial location
        over time. This calculation is performed per retinal layer.

        Parameters
        ----------
        ecs_item: array - like
            A 2D array specifying the effective current values at a
            particular spatial location(pixel); one value per retinal
            layer and electrode.
            Dimensions: <  # layers x #electrodes>
        pt_list: list
            A list of PulseTrain `data` containers.
            Dimensions: <  # electrodes x #time points>
        layers: list
            List of retinal layers to simulate. Choose from:
            - 'OFL': optic fiber layer
            - 'GCL': ganglion cell layer
            - 'INL': inner nuclear layer
        """
        not_supported = np.array([l not in SUPPORTED_LAYERS for l in layers],
                                 dtype=bool)
        if any(not_supported):
            raise ValueError("Acceptable values for `layers` is 'OFL', 'GCL', "
                             "'INL'.")

        ecm = np.zeros((ecs_item.shape[0], pt_list[0].shape[-1]))
        if 'INL' in layers:
            ecm[0, :] = np.sum(ecs_item[0, :, np.newaxis] * pt_list, axis=0)
        if ('GCL' or 'OFL') in layers:
            ecm[1, :] = np.sum(ecs_item[1, :, np.newaxis] * pt_list, axis=0)
        return ecm

    def model_cascade(self, ecs_item, pt_list, layers, use_jit):
        """The Temporal Sensitivity model

        This function applies the model of temporal sensitivity to a single
        retinal cell(i.e., a pixel). The model is inspired by Nanduri
        et al. (2012), with some extended functionality.

        Parameters
        ----------
        ecs_item: array - like
            A 2D array specifying the effective current values at a particular
            spatial location(pixel); one value per retinal layer and
            electrode.
            Dimensions: <  # layers x #electrodes>
        pt_list: list
            A list of PulseTrain `data` containers.
            Dimensions: <  # electrodes x #time points>
        layers: list
            List of retinal layers to simulate. Choose from:
            - 'OFL': optic fiber layer
            - 'GCL': ganglion cell layer
            - 'INL': inner nuclear layer
        use_jit: bool
            If True, applies just - in-time(JIT) compilation to expensive
            computations for additional speed - up(requires Numba).

        Returns
        -------
        Brightness response over time. In Nanduri et al. (2012), the
        maximum value of this signal was used to represent the perceptual
        brightness of a particular location in space, B(r).
        """
        # For each layer in the model, scale the pulse train data with the
        # effective current:
        ecm = self.calc_layer_current(ecs_item, pt_list, layers)

        # Calculate charge accumulation on the input
        ca = self.charge_accumulation(ecm)

        # Sparse convolution is faster if input is sparse. This is true for
        # the first convolution in the cascade, but not for subsequent ones.
        if 'INL' in layers:
            fr_inl = self.fast_response(ecm[0], self.gamma_inl,
                                        use_jit=use_jit,
                                        method='sparse')

            # Cathodic and anodic parts are treated separately: They have the
            # same charge accumulation, but anodic currents contribute less to
            # the response
            fr_inl_cath = np.maximum(0, -fr_inl)
            fr_inl_anod = self.aweight * np.maximum(0, fr_inl)
            resp_inl = np.maximum(0, fr_inl_cath + fr_inl_anod - ca[0, :])
        else:
            resp_inl = np.zeros_like(ecm[0])

        if ('GCL' or 'OFL') in layers:
            fr_gcl = self.fast_response(ecm[1], self.gamma_gcl,
                                        use_jit=use_jit,
                                        method='sparse')

            # Cathodic and anodic parts are treated separately: They have the
            # same charge accumulation, but anodic currents contribute less to
            # the response
            fr_gcl_cath = np.maximum(0, -fr_gcl)
            fr_gcl_anod = self.aweight * np.maximum(0, fr_gcl)
            resp_gcl = np.maximum(0, fr_gcl_cath + fr_gcl_anod - ca[1, :])
        else:
            resp_gcl = np.zeros_like(ecm[1])

        resp = resp_gcl + self.lweight * resp_inl
        resp = self.stationary_nonlinearity(resp)
        resp = self.slow_response(resp)
        return utils.TimeSeries(self.tsample, resp)


def ret2dva(r_um):
    """Converts retinal distances (um) to visual angles (deg)

    This function converts an eccentricity measurement on the retinal
    surface(in micrometers), measured from the optic axis, into degrees
    of visual angle.
    Source: Eq. A6 in Watson(2014), J Vis 14(7): 15, 1 - 17
    """
    sign = np.sign(r_um)
    r_mm = 1e-3 * np.abs(r_um)
    r_deg = 3.556 * r_mm + 0.05993 * r_mm ** 2 - 0.007358 * r_mm ** 3
    r_deg += 3.027e-4 * r_mm ** 4
    return sign * r_deg


def dva2ret(r_deg):
    """Converts visual angles (deg) into retinal distances (um)

    This function converts a retinal distancefrom the optic axis(um)
    into degrees of visual angle.
    Source: Eq. A5 in Watson(2014), J Vis 14(7): 15, 1 - 17
    """
    sign = np.sign(r_deg)
    r_deg = np.abs(r_deg)
    r_mm = 0.268 * r_deg + 3.427e-4 * r_deg ** 2 - 8.3309e-6 * r_deg ** 3
    r_um = 1e3 * r_mm
    return sign * r_um


def jansonius2009(phi0, n_rho=801, rho_range=(4.0, 45.0), eye='RE',
                  loc_od=(15.5, 1.5), beta_sup=-1.9, beta_inf=0.5):
    """Grows a single axon bundle based on the model by Jansonius et al. (2009)

    This function generates the trajectory of a single nerve fiber bundle
    based on the mathematical model described in [1]_.

    Parameters
    ----------
    phi0: float
        Angular position of the axon at its starting point(polar
        coordinates, degrees). Must be within[-180, 180].
    n_rho: int, optional, default: 801
        Number of sampling points along the radial axis(polar coordinates).
    rho_range: (rho_min, rho_max), optional, default: (4.0, 45.0)
        Lower and upper bounds for the radial position values(polar
        coordinates).
    loc_od: (x_od, y_od), optional, default: (15.5, 1.5)
        Location of the center of the optic disc(x, y) in Cartesian
        coordinates. In a right (left) eye, we should have x > 0 (x < 0).
    beta_sup: float, optional, default: -1.9
        Scalar value for the superior retina(see Eq. 5, `\beta_s` in the
        paper).
    beta_inf: float, optional, default: 0.5
        Scalar value for the inferior retina(see Eq. 6, `\beta_i` in the
        paper.)

    Returns
    -------
    ax_pos: Nx2 array
        Returns a two - dimensional array of axonal positions, where
        ax_pos[0, :] contains the(x, y) coordinates of the axon segment closest
        to the optic disc, and aubsequent row indices move the axon away from
        the optic disc. Number of rows is at most `n_rho`, but might be smaller
        if the axon crosses the meridian.

    Notes
    -----
    The study did not include axons with phi0 in [-60, 60] deg.

    .. [1] N. M. Jansionus, J. Nevalainen, B. Selig, L.M. Zangwill, P.A.
           Sample, W. M. Budde, J. B. Jonas, W. A. Lagrèze, P. J. Airaksinen,
           R. Vonthein, L. A. Levin, J. Paetzold, and U. Schieferd, "A
           mathematical description of nerve fiber bundle trajectories and
           their variability in the human retina. Vision Research 49:2157-2163,
           2009.

    """
    if eye.upper() not in ['LE', 'RE']:
        e_s = "Unknown eye string '%s': Choose from 'LE', 'RE'." % eye
        raise ValueError(e_s)

    if eye.upper() == 'LE':
        # The Jansonius model doesn't know about left eyes: We invert the x
        # coordinate of the optic disc here, run the model, and then invert all
        # x coordinates of all axon fibers back.
        loc_od = (-loc_od[0], loc_od[1])

    if np.abs(phi0) > 180.0:
        raise ValueError('phi0 must be within [-180, 180].')
    if n_rho < 1:
        raise ValueError('Number of radial sampling points must be >= 1.')
    if np.any(np.array(rho_range) < 0):
        raise ValueError('rho cannot be negative.')
    if rho_range[0] > rho_range[1]:
        raise ValueError('Lower bound on rho cannot be larger than the '
                         ' upper bound.')
    is_superior = phi0 > 0
    rho = np.linspace(*rho_range, num=n_rho)

    if is_superior:
        # Axon is in superior retina, compute `b` (real number) from Eq. 5:
        b = np.exp(beta_sup + 3.9 * np.tanh(-(phi0 - 121.0) / 14.0))
        # Equation 3, `c` a positive real number:
        c = 1.9 + 1.4 * np.tanh((phi0 - 121.0) / 14.0)
    else:
        # Axon is in inferior retina: compute `b` (real number) from Eq. 6:
        b = -np.exp(beta_inf + 1.5 * np.tanh(-(-phi0 - 90.0) / 25.0))
        # Equation 4, `c` a positive real number:
        c = 1.0 + 0.5 * np.tanh((-phi0 - 90.0) / 25.0)

    # Spiral as a function of `rho`:
    phi = phi0 + b * (rho - rho.min()) ** c

    # Convert to Cartesian coordinates
    xprime = rho * np.cos(np.deg2rad(phi))
    yprime = rho * np.sin(np.deg2rad(phi))

    # Find the array elements where the axon crosses the meridian
    if is_superior:
        # Find elements in inferior retina
        idx = np.where(yprime < 0)[0]
    else:
        # Find elements in superior retina
        idx = np.where(yprime > 0)[0]
    if idx.size:
        # Keep only up to first occurrence
        xprime = xprime[:idx[0]]
        yprime = yprime[:idx[0]]

    # Adjust coordinate system, having fovea=[0, 0] instead of `loc_od`=[0, 0]
    xmodel = xprime + loc_od[0]
    ymodel = yprime
    if loc_od[0] > 0:
        # If x-coordinate of optic disc is positive, use Appendix A
        idx = xprime > -loc_od[0]
    else:
        # Else we need to flip the sign
        idx = xprime < -loc_od[0]
    ymodel[idx] = yprime[idx] + loc_od[1] * (xmodel[idx] / loc_od[0]) ** 2

    # In a left eye, need to flip back x coordinates:
    if eye.upper() == 'LE':
        xmodel *= -1

    # Return as Nx2 array
    return np.vstack((xmodel, ymodel)).T


def find_closest_axon(pos_xy, axon_bundles):
    """Finds the closest axon to a 2D point

    This function finds the axon bundle closest to a 2D point `pos_xy` on the
    retina, and returns an axon that originates in `pos_xy` and projects to
    the optic disc.

    Parameters
    ----------
    pos_xy: (x, y)
        2D Cartesian coordinates of a location on the retina.
    axon_bundles: list of Nx2 arrays
        List of two - dimensional arrays containing the(x, y) coordinates of
        each axon bundle. The first row of each axon bundle is assumed to
        be closest to the optic disc, and subsequent row indices move the axon
        away from the optic disc.

    Returns
    -------
    axon: Nx2 array
        A single axon, where axon[0, :] contains the (x, y) coordinates of the
        location closest to `pos_xy`, and all subsequent rows move the axon
        closer to the optic disc.

    Notes
    -----
    The order of axonal segments in the output argument `axon` is reversed
    with respect to the axonal segments in the input argument `axon_bundles`.
    """
    xneuron, yneuron = pos_xy
    # Find the nearest axon to this pixel
    dist2 = [min((ax[:, 0] - xneuron) ** 2 + (ax[:, 1] - yneuron) ** 2)
             for ax in axon_bundles]
    axon_id = np.argmin(dist2)

    # Find the position on the axon
    ax = axon_bundles[axon_id]
    dist2 = (ax[:, 0] - xneuron) ** 2 + (ax[:, 1] - yneuron) ** 2
    pos_id = np.argmin(dist2)

    # Add all positions: from `pos_id` to the optic disc
    return axon_bundles[axon_id][pos_id:0:-1, :]


def axon_dist_from_soma(axon, xg, yg, tree=None):
    """Calculates the distance to soma for every axon segment

    For every segment of an axon, this function calculates the distance to the
    soma. The 2D coordinates of the axon are snapped to the grid using a
    nearest-neighbor tree structure.

    Parameters
    ----------
    axon: Nx2 array
        A single axon, where axon[0, :] contains the (x, y) coordinates of the
        location closest to the soma, and all subsequent rows move the axon
        away from the soma towards the optic disc.
    xg, yg: array
        meshgrid of pixel locations in units of visual angle sp
    tree : spat.cKDTree class instance, optional, default: train on `xg`, `yg`
        A kd-tree trained on `xg`, `yg` for quick nearest-neighbor lookup.

    Returns
    -------
    idx_cs : list
        Axon segment locations snapped to the grid, returned as a list of
        indices into the flat `xg`, `yg` meshgrid.
    dist : list
        Axon segment distances to the soma
    """
    if tree is None:
        # Build a nearest-neighbor tree for the coordinate system
        tree = spat.cKDTree(np.column_stack((xg.ravel(), yg.ravel())))

    # Consider only pixels within the grid, otherwise snap to grid might
    # yield unexpected results
    idx_valid = (axon[:, 0] >= xg.min()) * (axon[:, 0] <= xg.max())
    idx_valid *= (axon[:, 1] >= yg.min()) * (axon[:, 1] <= yg.max())

    # For these, find the xg, yg coordinates
    _, idx_cs = tree.query(axon[idx_valid, :])
    if len(idx_cs) == 0:
        return np.array([0]), np.array([np.inf])

    # Drop duplicates
    _, idx_cs_unique = np.unique(idx_cs, return_index=True)
    idx_cs = idx_cs[np.sort(idx_cs_unique)]
    if len(idx_cs) == 0:
        return np.array([0]), np.array([np.inf])

    # Find the location of the soma, based on the first axon segment
    _, idx_neuron = tree.query(axon[0, :])

    # Calculate the distance to soma.
    # For distance calculation, add a pixel at the location of the soma:
    idx_dist = np.insert(idx_cs, 0, idx_neuron, axis=0)

    # For every axon segment, calculate distance from soma by summing up the
    # individual distances between neighboring axon segments ("walking along
    # the axon"):
    xdiff = np.diff(xg.ravel()[idx_dist])
    ydiff = np.diff(yg.ravel()[idx_dist])
    dist = np.sqrt(np.cumsum(xdiff ** 2 + ydiff ** 2))

    return idx_cs, dist


def axon_contribution(axon_dist, current_spread, sensitivity_rule='decay',
                      contribution_rule='max', min_contribution=0.01,
                      decay_const=2.0, powermean_exp=None):
    """Determines the contribution of a single axon to the current map

    This function determines the contribution of a single axon to the current
    map based on a sensitivity rule (i.e., how the activation threshold of the
    axon differs as a function of distance from the soma), and an contribution
    rule (i.e., how the different activation thresholds along the axon are
    combined to determine the axon contribution).

    Parameters
    ----------
    axon_dist : tuple (indices, distances)
        A tuple containing a list of coordinates (indices into the retinal
        coordinates mesh grid) and distances for each axon segment.
    current_spread : 2D array
        A 2D current spread map that must have the same dimensions as the
        `xg`, `yg` meshgrid.
    sensitivity_rule : {'decay', 'Jeng2011'}, optional, default: 'decay'
        This rule specifies how the activation of the axon differs as a
        function of distance from the soma. The following options are
        available:
        - 'decay':
            Axon sensitivity decays exponentially with distance. Specify
            `decay_const` to change the steepness of the fall-off with
            distance.
        - 'Jeng2011':
            Axon sensitivity peaks near the sodium band (50um from the soma),
            then plateaus on the distal axon at roughly half of the peak
            sensitivity. See Figure 2 in Jeng, Tang, Molnar, Desai, and Fried
            (2011). The sodium channel band shapes the response to electric
            stimulation in retinal ganglion cells. J Neural Eng 8 (036022).
    contribution_rule : {'max', 'sum', 'power-mean'}, optional, default: 'max'
        This rule specifies how the activation thresholds across all axon
        segments are combined to determine the contribution of the axon to the
        current spread. The following options are available:
        - 'max':
            The axon's contribution to the current spread is equal to the max.
            sensitivity across all axon segments.
        - 'sum':
            The axon's contribution to the current spread is equal to the sum
            sensitivity across all axon segments.
        - 'mean':
            The axon's contribution to the current spread is equal to the mean
            sensitivity across all axon segments. Specify `powermean_exp` to
            change the exponent of the generalized (power) mean, calculated as
            np.mean(x ** powermean_exp) ** (1.0 / powermean_exp).
    min_contribution : float, optional, default: 0.01
        Current contributions below this value will not be counted.
    decay_const : float, optional, default: 2.0
        When `sensitivity_rule` is set to 'decay', specifies the decay constant
        of the exponential fall-off.
    powermean_exp : float, optional, default: None
        When `sensitivity_rule` is set to 'mean', specifies the exponent of the
        generalized (power) mean function. The power mean is calculated as
        np.mean(x ** powermean_exp) ** (1.0 / powermean_exp).
    """
    if contribution_rule == 'mean':
        if powermean_exp is None:
            raise ValueError("`powermean_exp` cannot be None when contribution "
                             "rule is set to 'mean'.")
        if powermean_exp <= 0.0:
            raise ValueError('`powermean_exp` must be positive.')
    else:
        if powermean_exp is not None:
            raise ValueError(("Contribution rule must be set to 'mean' in "
                              "order to change `powermean_exp` (currently "
                              "set to %s)." % contribution_rule))
    if decay_const <= 0.0:
        raise ValueError('`decay_const` must be positive.')

    # Unpack list of indices and distances for each axon segment
    idx_cs, dist = axon_dist
    idx_soma = idx_cs[0]

    # The sensitivity rule specifies how the activation thresholds differs
    # along the axon:
    if sensitivity_rule.lower() == 'decay':
        # Exponential fall-off with distance
        sensitivity = np.exp(-dist / decay_const)
    elif sensitivity_rule.lower() == 'jeng2011':
        # Roughly the inverse of Figure 2 in Jeng et al. (2011): The peak
        # sensitivity is over the sodium band, and the sensitivity of the
        # distal axon plateaus at roughly 50% of peak
        mu_gauss = ret2dva(50.0)
        std_gauss = ret2dva(20.0)
        bell = 0.7 * np.exp(-(dist - mu_gauss) ** 2 / (2 * std_gauss ** 2))
        plateau = 0.5
        soma = np.maximum(mu_gauss - dist, 0)
        sensitivity = np.maximum(0, bell - 0.001 * dist + plateau - soma)
    else:
        raise ValueError('Unknown sensitivity rule "%s"' % sensitivity_rule)

    # Effective activation of all axon segments, given by the segment's
    # sensitivity and activating current
    activation = sensitivity * current_spread.ravel()[idx_cs]

    # The contribution rule specifies how the activation values along the axon
    # are combined to determine the contribution of the axon to the current
    # spread:
    if contribution_rule.lower() == 'max':
        contribution = activation.max()
    elif contribution_rule.lower() == 'sum':
        contribution = activation.sum()
    elif contribution_rule.lower() == 'mean':
        # Generalized (power) mean
        p = powermean_exp
        contribution = np.mean(activation ** p) ** (1.0 / p)
    else:
        raise ValueError('Unknown activation rule "%s"' % contribution_rule)

    return idx_soma, contribution


@utils.deprecated(alt_func='p2p.retina.jansonius2009',
                  deprecated_version='0.3', removed_version='0.4')
def jansonius(num_cells=500, num_samples=801, center=np.array([15, 2]),
              rot=0 * np.pi / 180, scale=1, bs=-1.9, bi=.5, r0=4,
              max_samples=45, ang_range=60):
    """Implements the model of retinal axonal pathways by generating a
    matrix of(x, y) positions.

    Assumes that the fovea is at[0, 0]

    Parameters
    ----------
    num_cells: int
        Number of axons(cells).
    num_samples: int
        Number of samples per axon(spatial resolution).
    Center: 2 item array
        The location of the optic disk in dva.

    See:

    Jansonius et al., 2009, A mathematical description of nerve fiber bundle
    trajectories and their variability in the human retina, Vision Research
    """

    # Default parameters:
    #
    # r0 = 4;             %Minumum radius (optic disc size)
    #
    # center = [15,2];    %p.center of optic disc
    #
    # rot = 0*pi/180;    %Angle of rotation (clockwise)
    # scale = 1;             %Scale factor
    #
    # bs = -1.9;          %superior 'b' parameter constant
    # bi = .5;            %inferior 'c' parameter constant
    # ang_range = 60

    # sample space of superior/inferior retina, add them in a 1D array
    # superior is where ang0 > 0
    # this will be the first dimension of the meshgrid
    # inferior should go from -180 to -60? or typo in paper
    # ang0 is \phi_0
    ang0 = np.hstack([np.linspace(ang_range, 180, num_cells / 2),  # superior
                      np.linspace(-180, ang_range, num_cells / 2)])  # inferior

    # from r0=4 to max_samples=45, take num_samples=801 steps
    # this will be the second dimension of the meshgrid
    r = np.linspace(r0, max_samples, num_samples)

    # generate angle and radius matrices from vectors with meshgrid
    ang0mat, rmat = np.meshgrid(ang0, r)

    num_samples = ang0mat.shape[0]
    num_cells = ang0mat.shape[1]

    # index into axons from superior (upper) retina
    sup = ang0mat > 0

    # Set up 'b' parameter:
    b = np.zeros([num_samples, num_cells])

    # Equation 5: upper retina
    b[sup] = np.exp(
        bs + 3.9 * np.tanh(-(ang0mat[sup] - 121) / 14))

    # equation 6: lower retina
    b[~sup] = -np.exp(bi + 1.5 * np.tanh(-(-ang0mat[~sup] - 90) / 25))

    # Set up 'c' parameter:
    c = np.zeros([num_samples, num_cells])

    # equation 3 (fixed typo)
    # Paper says -(angmat-121)/14. Is the - sign the typo?
    c[sup] = 1.9 + 1.4 * np.tanh((ang0mat[sup] - 121) / 14)
    c[~sup] = 1 + .5 * np.tanh((-ang0mat[~sup] - 90) / 25)   # equation 4

    # Here's the main function: spirals as a function of r (equation 1)
    ang = ang0mat + b * (rmat - r0)**c

    # Transform to x-y coordinates
    xprime = rmat * np.cos(ang * np.pi / 180)
    yprime = rmat * np.sin(ang * np.pi / 180)

    # Find where the fibers cross the horizontal meridian
    cross = np.zeros([num_samples, num_cells])
    cross[sup] = yprime[sup] < 0
    cross[~sup] = yprime[~sup] > 0

    # Set Nans to axon paths after crossing horizontal meridian
    id = np.where(np.transpose(cross))

    curr_col = -1
    for i in range(0, len(id[0])):  # loop through axons
        if curr_col != id[0][i]:
            yprime[id[1][i]:, id[0][i]] = np.NaN
            curr_col = id[0][i]

    # Bend the image according to (the inverse) of Appendix A
    xmodel = xprime + center[0]
    ymodel = yprime
    id = xprime > -center[0]
    ymodel[id] = yprime[id] + center[1] * (xmodel[id] / center[0])**2

    #  rotate about the optic disc and scale
    x = scale * (np.cos(rot) * (xmodel - center[0]) + np.sin(rot)
                * (ymodel - center[1])) + center[0]
    y = scale * (-np.sin(rot) * (xmodel - center[0]) + np.cos(rot)
                * (ymodel - center[1])) + center[1]

    return x, y


@utils.deprecated(deprecated_version='0.3', removed_version='0.4')
def make_axon_map(xg, yg, jan_x, jan_y, axon_lambda=1, min_weight=0.001):
    """Retinal axon map

    Generates a mapping of how each pixel in the retina space is affected
    by stimulation of underlying ganglion cell axons.
    Parameters
    ----------
    xg, yg: array
        meshgrid of pixel locations in units of visual angle sp
    axon_lambda: float
        space constant for how effective stimulation(or 'weight') falls off
        with distance from the pixel back along the axon toward the optic disc
        (default 1 degree)
    min_weight: float
        minimum weight falloff.  default .001

    Returns
    -------
    axon_id: list
        a list, for every pixel, of the index into the pixel in xg, yg space,
        along the underlying axonal pathway.
    axon_weight: list
        a list, for every pixel, of the axon weight into the pixel in xg, yg
        space

    """
    axon_id = []
    axon_weight = []
    for idx, _ in enumerate(xg.ravel()):
        cur_xg = xg.ravel()[idx]
        cur_yg = yg.ravel()[idx]
        # find the nearest axon to this pixel
        d = (jan_x - cur_xg) ** 2 + (jan_y - cur_yg) ** 2
        cur_ax_id = np.nanargmin(d)  # index into the current axon

        # `ax_num`: which axon it is
        # `ax_pos_id0`: the point on that axon that is closest to `px`
        [ax_pos_id0, ax_num] = np.unravel_index(cur_ax_id, d.shape)

        dist = 0
        this_id = [idx]
        this_weight = [1.0]
        for ax_pos_id in range(ax_pos_id0 - 1, -1, -1):
            # Increment the distance from the starting point
            # The following calculation had a bug in them: squaring was done
            # twice
            ax = (jan_x[ax_pos_id + 1, ax_num] - jan_x[ax_pos_id, ax_num])
            ay = (jan_y[ax_pos_id + 1, ax_num] - jan_y[ax_pos_id, ax_num])
            dist += np.sqrt(ax ** 2 + ay ** 2)

            # weight falls off exponentially as distance from axon cell body
            weight = np.exp(-dist / axon_lambda)

            # find the nearest pixel to the current position along the axon
            dist_xg = np.abs(xg[0, :] - jan_x[ax_pos_id, ax_num])
            dist_yg = np.abs(yg[:, 0] - jan_y[ax_pos_id, ax_num])
            nearest_xg_id = dist_xg.argmin()
            nearest_yg_id = dist_yg.argmin()
            nearest_xg = xg[0, nearest_xg_id]
            nearest_yg = yg[nearest_yg_id, 0]

            # if the position along the axon has moved to a new pixel, and the
            # weight isn't too small...
            if weight > min_weight:
                if nearest_xg != cur_xg or nearest_yg != cur_yg:
                    # update the current pixel location
                    cur_xg = nearest_xg
                    cur_yg = nearest_yg

                    # Append the list
                    # The following calculation had a bug in it: `weight` was
                    # exponentiated twice
                    this_weight.append(weight)
                    this_id.append(np.ravel_multi_index((nearest_yg_id,
                                                         nearest_xg_id),
                                                        xg.shape))

        axon_id.append(this_id)
        axon_weight.append(this_weight)
    return axon_id, axon_weight
