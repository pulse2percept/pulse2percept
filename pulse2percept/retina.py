import numpy as np
import scipy.signal as signal
import scipy.special as ss
import os.path
import logging

from pulse2percept import utils


SUPPORTED_LAYERS = ['INL', 'GCL', 'OFL']


class Grid(object):
    """Represent the retinal coordinate frame"""

    def __init__(self, xlo=-1000, xhi=1000, ylo=-1000, yhi=1000,
                 sampling=25, axon_lambda=2.0, rot=0 * np.pi / 180,
                 datapath='./', save_data=True):
        """Generates a spatial grid representing the retinal coordinate frame

        This function generates the coordinate system for the retina
        and an axon map. As this can take a while, the function will
        first look for an already existing file in the directory `datapath`
        that was automatically created from an earlier call to this function,
        before it attempts to generate new grid from scratch.

        Parameters
        ----------
        xlo, xhi : float
           Extent of the retinal coverage (microns) in horizontal dimension.
           Default: xlo=-1000, xhi=1000.
        ylo, yhi : float
           Extent of the retinal coverage (microns) in vertical dimension.
           Default: ylo=-1000, ylo=1000.
        datapath : str
            Relative path where to look for existing retina files, and where to
            store new files. Default: current directory.
        save_data : bool
            Flag whether to save the data to a new file (True) or not (False).
            The file name is automatically generated from all specified input
            arguments.
            Default: True.
        """
        # Include endpoints in meshgrid
        num_x = int((xhi - xlo) / sampling + 1)
        num_y = int((yhi - ylo) / sampling + 1)
        self.gridx, self.gridy = np.meshgrid(np.linspace(xlo, xhi, num_x),
                                             np.linspace(ylo, yhi, num_y),
                                             indexing='xy')

        # Create descriptive filename based on input args
        filename = "%sretina_s%d_l%.1f_rot%.1f_%dx%d.npz" \
            % (datapath, sampling, axon_lambda, rot / np.pi * 180,
               xhi - xlo, yhi - ylo)

        # Bool whether we need to create a new grid
        need_new_grid = True

        # Check if such a file already exists. If so, load parameters and
        # make sure they are the same as specified above. Else, create new.
        if os.path.exists(filename):
            need_new_grid = False
            axon_map = np.load(filename)

            # Verify that the file was created with a consistent grid:
            ax_id = axon_map['axon_id']
            ax_wt = axon_map['axon_weight']
            xlo_am = axon_map['xlo']
            xhi_am = axon_map['xhi']
            ylo_am = axon_map['ylo']
            yhi_am = axon_map['yhi']
            sampling_am = axon_map['sampling']
            axon_lambda_am = axon_map['axon_lambda']

            if 'jan_x' in axon_map and 'jan_y' in axon_map:
                jan_x = axon_map['jan_x']
                jan_y = axon_map['jan_y']
            else:
                jan_x = jan_y = None

            # If any of the dimensions don't match, we need a new retina
            need_new_grid |= xlo != xlo_am
            need_new_grid |= xhi != xhi_am
            need_new_grid |= ylo != ylo_am
            need_new_grid |= yhi != yhi_am
            need_new_grid |= sampling != sampling_am
            need_new_grid |= axon_lambda != axon_lambda_am

            if 'rot' in axon_map:
                rot_am = axon_map['rot']
                need_new_grid |= rot != rot_am
            else:
                # Backwards compatibility for older retina object files that
                # did not have `rot`
                need_new_grid |= rot != 0

        # At this point we know whether we need to generate a new retina:
        if need_new_grid:
            info_str = "File '%s' doesn't exist " % filename
            info_str += "or has outdated parameter values, generating..."
            logging.getLogger(__name__).info(info_str)

            jan_x, jan_y = jansonius(rot=rot)
            dva_x = ret2dva(self.gridx)
            dva_y = ret2dva(self.gridy)
            ax_id, ax_wt = make_axon_map(dva_x, dva_y,
                                         jan_x, jan_y,
                                         axon_lambda=axon_lambda)

            # Save the variables, together with metadata about the grid:
            if save_data:
                np.savez(filename,
                         axon_id=ax_id,
                         axon_weight=ax_wt,
                         jan_x=jan_x,
                         jan_y=jan_y,
                         xlo=[xlo],
                         xhi=[xhi],
                         ylo=[ylo],
                         yhi=[yhi],
                         sampling=[sampling],
                         axon_lambda=[axon_lambda],
                         rot=[rot])

        self.axon_lambda = axon_lambda
        self.rot = rot
        self.sampling = sampling
        self.axon_id = ax_id
        self.axon_weight = ax_wt
        self.jan_x = jan_x
        self.jan_y = jan_y
        self.range_x = self.gridx.max() - self.gridx.min()
        self.range_y = self.gridy.max() - self.gridy.min()

    def current2effectivecurrent(self, cs):
        """

        Converts a current spread map to an 'effective' current spread map, by
        passing the map through a mapping of axon streaks.

        Parameters
        ----------
        cs : array
            The 2D spread map in retinal space

        Returns
        -------
        ecm : array
            The effective current spread, a time-series of the same size as the
            current map, where each pixel is the dot product of the pixel
            values in ecm along the pixels in the list in axon_map, weighted
            by the weights axon map.
        """
        ecs = np.zeros(cs.shape)
        for id in range(0, len(cs.flat)):
            ecs.flat[id] = np.dot(cs.flat[self.axon_id[id]],
                                  self.axon_weight[id])

        # normalize so the response under the electrode in the ecs map
        # is equal to cs
        maxloc = np.where(cs == np.max(cs))
        scFac = np.max(cs) / ecs[maxloc[0][0], maxloc[1][0]]
        ecs = ecs * scFac

        # this normalization is based on unit current on the retina producing
        # a max response of 1 based on axonal integration.
        # means that response magnitudes don't change as you increase the
        # length of axonal integration or sampling of the retina
        # Doesn't affect normalization over time, or responses as a function
        # of the anount of current,

        return ecs

    def electrode_ecs(self, implant, alpha=14000, n=1.69):
        """
        Gather current spread and effective current spread for each electrode
        within both the bipolar and the ganglion cell layer

        Parameters
        ----------
        implant : implants.ElectrodeArray
            An implants.ElectrodeArray instance describing the implant.

        alpha : float
            Current spread parameter
        n : float
            Current spread parameter

        Returns
        -------
        ecs : contains n arrays containing the the effective current
            spread within various layers
            for each electrode in the array respectively.

        See also
        --------
        Electrode.current_spread
        """

        cs = np.zeros((self.gridx.shape[0], self.gridx.shape[1],
                       2, len(implant.electrodes)))
        ecs = np.zeros((self.gridx.shape[0], self.gridx.shape[1],
                        2, len(implant.electrodes)))

        for i, e in enumerate(implant.electrodes):
            cs[..., 0, i] = e.current_spread(self.gridx, self.gridy,
                                             layer='INL', alpha=alpha, n=n)
            ecs[..., 0, i] = cs[..., 0, i]
            cs[..., 1, i] = e.current_spread(self.gridx, self.gridy,
                                             layer='OFL', alpha=alpha, n=n)
            ecs[:, :, 1, i] = self.current2effectivecurrent(cs[..., 1, i])

        return ecs, cs


class TemporalModel(object):

    def __init__(self, tsample=0.005 / 1000,
                 tau_gcl=0.42 / 1000, tau_inl=18.0 / 1000,
                 tau_ca=45.25 / 1000, scale_ca=42.1,
                 tau_slow=26.25 / 1000, scale_slow=10.0,
                 lweight=0.636, aweight=0.5,
                 slope=3.0, shift=15.0):
        """Temporal Sensitivity Model

        A model of temporal integration from retina pixels.

        Parameters
        ----------
        tsample : float
            Sampling time step (seconds). Default: 5e-6 s.
        tau_gcl : float
            Time decay constant for the fast leaky integrater of the ganglion
            cell layer (GCL).
            This is only important in combination with epiretinal electrode
            arrays. Default: 45.25 / 1000 s.
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
        self.tsample = tsample
        self.tau_gcl = tau_gcl
        self.tau_inl = tau_inl
        self.tau_ca = tau_ca
        self.scale_ca = scale_ca
        self.tau_slow = tau_slow
        self.slope = slope
        self.shift = shift
        self.lweight = lweight
        self.aweight = aweight
        self.scale_slow = scale_slow

        # perform one-time setup calculations
        # Gamma functions used as convolution kernels do not depend on input
        # data, hence can be calculated once, then re-used (trade off memory
        # for speed).
        # gamma_gcl and gamma_inl are used to calculate the fast response in
        # bipolar and ganglion cells respectively

        _, self.gamma_inl = utils.gamma(1, self.tau_inl, self.tsample)
        _, self.gamma_gcl = utils.gamma(1, self.tau_gcl, self.tsample)

        # gamma_ca is used to calculate charge accumulation
        _, self.gamma_ca = utils.gamma(1, self.tau_ca, self.tsample)

        # gamma_slow is used to calculate the slow response
        _, self.gamma_slow = utils.gamma(3, self.tau_slow, self.tsample)

    def fast_response(self, stim, gamma, dojit=True, usefft=False):
        """Fast response function (Box 2) for the bipolar layer

        Convolve a stimulus `stim` with a temporal low-pass filter (1-stage
        gamma) with time constant `self.tau_inl` ~ 14ms representing bipolars.

        Parameters
        ----------
        stim : array
           Temporal signal to process, stim(r,t) in Nanduri et al. (2012).
        dojit : bool, optional
           If True (default), use numba just-in-time compilation.
        usefft : bool, optional
           If False (default), use sparseconv, else fftconvolve.

        Returns
        -------
        Fast response, b2(r,t) in Nanduri et al. (2012).

        Notes
        -----
        The function utils.sparseconv can be much faster than np.convolve and
        signal.fftconvolve if `stim` is sparse and much longer than the
        convolution kernel.

        The output is not converted to a TimeSeries object for speedup.
        """
        # FFT is faster on non-sparse data
        if usefft:
            conv = self.tsample * signal.fftconvolve(stim, gamma, mode='full')
        else:
            conv = self.tsample * utils.sparseconv(gamma, stim,
                                                   mode='full',
                                                   dojit=dojit)
            # Cut off the tail of the convolution to make the output signal
            # match the dimensions of the input signal.
        return conv[:stim.shape[-1]]

    def charge_accumulation(self, ecm):
        """Calculates the charge accumulation

        Charge accumulation is calculcalated on the effective input current
        `ecm`, as opposed to the output of the fast response stage.

        Parameters
        ----------
        ecm : array-like
            A 2D array specifying the effective current values at a particular
            spatial location (pixel); one value per retinal layer, averaged
            over all electrodes through that pixel.
            Dimensions: <#layers x #time points>
        """
        ca = np.zeros_like(ecm)

        for i in range(ca.shape[0]):
            summed = self.tsample * np.cumsum(np.abs(ecm[i, :]))
            conved = self.tsample * signal.fftconvolve(summed, self.gamma_ca,
                                                       mode='full')
            ca[i, :] = self.scale_ca * conved[:ecm.shape[-1]]
        return ca

    def stationary_nonlinearity(self, stim):
        """Stationary nonlinearity (Box 4)

        Nonlinearly rescale a temporal signal `stim` across space and time,
        based on a sigmoidal function dependent on the maximum value of `stim`.
        This is Box 4 in Nanduri et al. (2012).

        The parameter values of the asymptote, slope, and shift of the logistic
        function are given by self.asymptote, self.slope, and self.shift,
        respectively.

        Parameters
        ----------
        stim : array
           Temporal signal to process, stim(r,t) in Nanduri et al. (2012).

        Returns
        -------
        Rescaled signal, b4(r,t) in Nanduri et al. (2012).

        Notes
        -----
        Conversion to TimeSeries is avoided for the sake of speedup.
        """
        # use expit (logistic) function for speedup
        sigmoid = ss.expit((stim.max() - self.shift) / self.slope)

        # avoid division by zero
        return stim * sigmoid

    def slow_response(self, stim):
        """Slow response function (Box 5)

        Convolve a stimulus `stim` with a low-pass filter (3-stage gamma)
        with time constant self.tau_slow.
        This is Box 5 in Nanduri et al. (2012).

        Parameters
        ----------
        stim : array
           Temporal signal to process, stim(r,t) in Nanduri et al. (2012)

        Returns
        -------
        Slow response, b5(r,t) in Nanduri et al. (2012).

        Notes
        -----
        This is by far the most computationally involved part of the perceptual
        sensitivity model.

        Conversion to TimeSeries is avoided for the sake of speedup.
        """
        # No need to zero-pad: fftconvolve already takes care of optimal
        # kernel/data size
        conv = signal.fftconvolve(stim, self.gamma_slow, mode='full')

        # Cut off the tail of the convolution to make the output signal match
        # the dimensions of the input signal.
        return self.scale_slow * self.tsample * conv[:stim.shape[-1]]

    def calc_layer_current(self, ecs_item, pt_list, layers):
        """For a given pixel, calculates the effective current for each retinal
           layer over time

            This function operates at a single-pixel level: It calculates the
            combined current from all electrodes through a spatial location
            over time. This calculation is performed per retinal layer.

            Parameters
            ----------
            ecs_item: array-like
                A 2D array specifying the effective current values at a
                particular spatial location (pixel); one value per retinal
                layer and electrode.
                Dimensions: <#layers x #electrodes>
            pt_list: list
                A list of PulseTrain `data` containers.
                Dimensions: <#electrodes x #time points>
            layers : list

                List of retinal layers to simulate. Choose from:
                - 'OFL': optic fiber layer
                - 'GCL': ganglion cell layer
                - 'INL': inner nuclear layer

        """
        ecm = np.zeros((ecs_item.shape[0], pt_list[0].shape[-1]))
        pt_data = np.array([pt.data for pt in pt_list])
        if 'INL' in layers:
            ecm[0, :] = np.sum(ecs_item[0, :, np.newaxis] * pt_data, axis=0)
        if ('GCL' or 'OFL') in layers:
            ecm[1, :] = np.sum(ecs_item[1, :, np.newaxis] * pt_data, axis=0)
        return ecm

    def model_cascade(self, ecs_item, pt_list, layers, dojit):
        """The Temporal Sensitivity model

        This function applies the model of temporal sensitivity to a single
        retinal cell (i.e., a pixel). The model is inspired by Nanduri
        et al. (2012), with some extended functionality.

        Parameters
        ----------
        ecs_item: array-like
            A 2D array specifying the effective current values at a particular
            spatial location (pixel); one value per retinal layer and
            electrode.
            Dimensions: <#layers x #electrodes>
        pt_list: list
            A list of PulseTrain `data` containers.
            Dimensions: <#electrodes x #time points>
        layers : list

            List of retinal layers to simulate. Choose from:
            - 'OFL': optic fiber layer
            - 'GCL': ganglion cell layer
            - 'INL': inner nuclear layer
        dojit : bool
            If True, applies just-in-time (JIT) compilation to expensive
            computations for additional speed-up (requires Numba).

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
                                        dojit=dojit,
                                        usefft=False)

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
                                        dojit=dojit,
                                        usefft=False)

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
    surface (in micrometers), measured from the optic axis, into degrees
    of visual angle.
    Source: Eq. A6 in Watson (2014), J Vis 14(7):15, 1-17
    """
    sign = np.sign(r_um)
    r_mm = 1e-3 * np.abs(r_um)
    r_deg = 3.556 * r_mm + 0.05993 * r_mm ** 2 - 0.007358 * r_mm ** 3
    r_deg += 3.027e-4 * r_mm ** 4
    return sign * r_deg


@utils.deprecated('p2p.retina.ret2dva')
def micron2deg(micron):
    """Transforms a distance from microns to degrees

    This function is deprecated as of v0.2, and will be completely removed
    in v0.3. Use p2p.retina.ret2dva instead.

    Based on http://retina.anatomy.upenn.edu/~rob/lance/units_space.html
    """
    deg = micron / 280.0
    return deg


@utils.deprecated('p2p.retina.dva2ret')
def deg2micron(deg):
    """Transforms a distance from degrees to microns

    This function is deprecated as of v0.2, and will be completely removed
    in v0.3. Use p2p.retina.dva2ret instead.

    Based on http://retina.anatomy.upenn.edu/~rob/lance/units_space.html
    """
    microns = 280.0 * deg
    return microns


def dva2ret(r_deg):
    """Converts visual angles (deg) into retinal distances (um)

    This function converts a retinal distancefrom the optic axis (um)
    into degrees of visual angle.
    Source: Eq. A5 in Watson (2014), J Vis 14(7):15, 1-17
    """
    sign = np.sign(r_deg)
    r_deg = np.abs(r_deg)
    r_mm = 0.268 * r_deg + 3.427e-4 * r_deg ** 2 - 8.3309e-6 * r_deg ** 3
    r_um = 1e3 * r_mm
    return sign * r_um


def jansonius(num_cells=500, num_samples=801, center=np.array([15, 2]),
              rot=0 * np.pi / 180, scale=1, bs=-1.9, bi=.5, r0=4,
              max_samples=45, ang_range=60):
    """Implements the model of retinal axonal pathways by generating a
    matrix of (x,y) positions.

    Assumes that the fovea is at [0, 0]

    Parameters
    ----------
    num_cells : int
        Number of axons (cells).
    num_samples : int
        Number of samples per axon (spatial resolution).
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

    ang0 = np.hstack([np.linspace(ang_range, 180, num_cells / 2),
                      np.linspace(-180, ang_range, num_cells / 2)])

    r = np.linspace(r0, max_samples, num_samples)
    # generate angle and radius matrices from vectors with meshgrid
    ang0mat, rmat = np.meshgrid(ang0, r)

    num_samples = ang0mat.shape[0]
    num_cells = ang0mat.shape[1]

    # index into superior (upper) axons
    sup = ang0mat > 0

    # Set up 'b' parameter:
    b = np.zeros([num_samples, num_cells])

    b[sup] = np.exp(
        bs + 3.9 * np.tanh(-(ang0mat[sup] - 121) / 14))  # equation 5
    # equation 6
    b[~sup] = -np.exp(bi + 1.5 * np.tanh(-(-ang0mat[~sup] - 90) / 25))

    # Set up 'c' parameter:
    c = np.zeros([num_samples, num_cells])

    # equation 3 (fixed typo)
    c[sup] = 1.9 + 1.4 * np.tanh((ang0mat[sup] - 121) / 14)
    c[~sup] = 1 + .5 * np.tanh((-ang0mat[~sup] - 90) / 25)   # equation 4

    # %Here's the main function: spirals as a function of r (equation 1)
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

    currCol = -1
    for i in range(0, len(id[0])):  # loop through axons
        if currCol != id[0][i]:
            yprime[id[1][i]:, id[0][i]] = np.NaN
            currCol = id[0][i]

    # Bend the image according to (the inverse) of Appendix A
    xmodel = xprime + center[0]
    ymodel = yprime
    id = xprime > -center[0]
    ymodel[id] = yprime[id] + center[1] * (xmodel[id] / center[0])**2

    #  rotate about the optic disc and scale
    x = scale * (np.cos(rot) * (xmodel - center[0]) + np.sin(rot) *
                 (ymodel - center[1])) + center[0]
    y = scale * (-np.sin(rot) * (xmodel - center[0]) + np.cos(rot) *
                 (ymodel - center[1])) + center[1]

    return x, y


def make_axon_map(xg, yg, jan_x, jan_y, axon_lambda=1, min_weight=.001):
    """Retinal axon map

    Generates a mapping of how each pixel in the retina space is affected
    by stimulation of underlying ganglion cell axons.
    Parameters
    ----------
    xg, yg : array
        meshgrid of pixel locations in units of visual angle sp
    axon_lambda : float
        space constant for how effective stimulation (or 'weight') falls off
        with distance from the pixel back along the axon toward the optic disc
        (default 1 degree)
    min_weight : float
        minimum weight falloff.  default .001

    Returns
    -------
    axon_id : list
        a list, for every pixel, of the index into the pixel in xg,yg space,
        along the underlying axonal pathway.
    axon_weight : list
        a list, for every pixel, of the axon weight into the pixel in xg,yg
        space

    """
    # initialize lists
    axon_xg = ()
    axon_yg = ()
    axon_dist = ()
    axon_weight = ()
    axon_id = ()

    # loop through pixels as indexed into a single dimension
    for px in range(0, len(xg.flat)):
        # find the nearest axon to this pixel
        d = (jan_x - xg.flat[px])**2 + (jan_y - yg.flat[px])**2
        cur_ax_id = np.nanargmin(d)  # index into the current axon
        [ax_pos_id0, ax_num] = np.unravel_index(cur_ax_id, d.shape)

        dist = 0

        cur_xg = xg.flat[px]
        cur_yg = yg.flat[px]

        # add first values to the list for this pixel
        axon_dist = axon_dist + ([0],)
        axon_weight = axon_weight + ([1],)
        axon_xg = axon_xg + ([cur_xg],)
        axon_yg = axon_yg + ([cur_yg],)
        axon_id = axon_id + ([px],)

        # now loop back along this nearest axon toward the optic disc
        for ax_pos_id in range(ax_pos_id0 - 1, -1, -1):
            # increment the distance from the starting point
            ax = (jan_x[ax_pos_id + 1, ax_num] - jan_x[ax_pos_id, ax_num])**2
            ay = (jan_y[ax_pos_id + 1, ax_num] - jan_y[ax_pos_id, ax_num])**2
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

                    # append the list
                    axon_weight[px].append(np.exp(weight))
                    axon_id[px].append(np.ravel_multi_index((nearest_yg_id,
                                                             nearest_xg_id),
                                                            xg.shape))

    return list(axon_id), list(axon_weight)
