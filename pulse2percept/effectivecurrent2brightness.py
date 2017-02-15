# -*effectivecurrent2brightness -*-
"""effectivecurrent2brightness
This transforms the effective current into brightness for a single point in
space based on the Horsager model as modified by Devyani
Inputs: a vector of effective current over time
Output: a vector of brightness over time
"""
from __future__ import print_function
import numpy as np
import scipy.signal as signal
import scipy.special
import logging

import pulse2percept.electrode2currentmap as e2cm
from pulse2percept import utils

# reload(logging)
logger = logging.getLogger(__name__)


class TemporalModel(object):

    def __init__(self, tsample=0.005 / 1000,
                 tau_nfl=0.42 / 1000, tau_inl=18.0 / 1000,
                 tau_ca=45.25 / 1000, tau_slow=26.25 / 1000,
                 scale_slow=1150.0, lweight=0.636, aweight=0.5,
                 slope=3.0, shift=15.0):
        """Temporal Sensitivity Model

        A model of temporal integration from retina pixels.

        Parameters
        ----------
        tsample : float
            Sampling time step (seconds). Default: 5e-6 s.
        tau_nfl : float
            Time decay constant for the fast leaky integrater of the nerve
            fiber layer (NFL); i.e., ganglion cell layer.
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
        self.tau_nfl = tau_nfl
        self.tau_inl = tau_inl
        self.tau_ca = tau_ca
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
        # gamma_nfl and gamma_inl are used to calculate the fast response in
        # bipolar and ganglion cells respectively

        t = np.arange(0, 8 * self.tau_inl, self.tsample)
        self.gamma_inl = e2cm.gamma(1, self.tau_inl, t)

        t = np.arange(0, 10 * self.tau_nfl, self.tsample)
        self.gamma_nfl = e2cm.gamma(1, self.tau_nfl, t)

        # gamma_ca is used to calculate charge accumulation
        t = np.arange(0, 6 * self.tau_ca, self.tsample)
        self.gamma_ca = e2cm.gamma(1, self.tau_ca, t)

        # gamma_slow is used to calculate the slow response
        t = np.arange(0, 10 * self.tau_slow, self.tsample)
        self.gamma_slow = e2cm.gamma(3, self.tau_slow, t)

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
        scipy.signals.fftconvolve if `stim` is sparse and much longer than the
        convolution kernel.

        The output is not converted to a TimeSeries object for speedup.
        """
        # FFT is faster on non-sparse data
        if usefft:
            conv = self.tsample * signal.fftconvolve(stim, gamma, mode='full')
        else:
            conv = self.tsample * utils.sparseconv(gamma, stim, mode='full',
                                                   dojit=dojit)
            # Cut off the tail of the convolution to make the output signal
            # match the dimensions of the input signal.
        return conv[:stim.shape[-1]]

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
        stimmax = stim.max()
        scale = scipy.special.expit((stimmax - self.shift) / self.slope)

        # avoid division by zero
        return stim / (stimmax + np.finfo(float).eps) * scale

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

    def model_cascade(self, ecm, dolayers, dojit):
        """Model cascade according to Nanduri et al. (2012).

        The 'Nanduri' model calculates the fast response first, followed by the
        current accumulation.

        Parameters
        ----------
        ecm : TimeSeries
            Effective current
        dolayers : list

            List of retinal layers to simulate. Choose from:
            - 'NFL': nerve fiber layer
            - 'INL': inner nuclear layer

        Returns
        -------
        Brightness response over time. In Nanduri et al. (2012), the
        maximum value of this signal was used to represent the perceptual
        brightness of a particular location in space, B(r).
        """
        # Sparse convolution is faster if input is sparse. This is true for
        # the first convolution in the cascade, but not for subsequent ones.
        usefft = False
        if 'INL' in dolayers:
            resp_inl = self.fast_response(ecm[0].data, self.gamma_inl,
                                          dojit=dojit,
                                          usefft=usefft)
            # Set `usefft` to True for subsequent stages of the cascade
            usefft = True
        else:
            resp_inl = np.zeros((ecm[0].data.shape))

        if 'NFL' in dolayers:
            resp_nfl = self.fast_response(ecm[1].data, self.gamma_nfl,
                                          dojit=dojit,
                                          usefft=usefft)
            # Set `usefft` to True for subsequent stages of the cascade
            usefft = True
        else:
            resp_nfl = np.zeros((ecm[1].data.shape))

        # Here we are converting from current  - where a cathodic (effective)
        # stimulus is negative - to a vague concept of neuronal response,
        # where positive implies a neuronal response.
        # There is a rectification here because we used to assume that the
        # anodic part of the pulse is ineffective (which is wrong).
        resp_cathodic = self.lweight * np.maximum(-resp_inl, 0) + \
            np.maximum(-resp_nfl, 0)
        resp_anodic = self.lweight * np.maximum(resp_inl, 0) + \
            np.maximum(resp_nfl, 0)
        resp = resp_cathodic + self.aweight * resp_anodic
        resp = self.stationary_nonlinearity(resp)
        resp = self.slow_response(resp)
        return utils.TimeSeries(self.tsample, resp)


def pulse2percept(stim, implant, tm=None, retina=None,
                  rsample=30, scale_charge=42.1, tol=0.05, use_ecs=True,
                  engine='joblib', dojit=True, n_jobs=-1):
    """Transforms an input stimulus to a percept

    This function passes an input stimulus `stim` to a retinal `implant`,
    which is placed on a simulated `retina`, and produces a predicted percept
    by means of the temporal model `tm`.

    Parameters
    ----------
    stim : utils.TimeSeries|list|dict
        There are several ways to specify an input stimulus:

        - For a single-electrode array, pass a single pulse train; i.e., a
          single utils.TimeSeries object.
        - For a multi-electrode array, pass a list of pulse trains; i.e., one
          pulse train per electrode.
        - For a multi-electrode array, specify all electrodes that should
          receive non-zero pulse trains by name.
    implant : e2cm.ElectrodeArray
        An ElectrodeArray object that describes the implant.
    tm : ec2b.TemporalModel, optional
        A model of temporal sensitivity.
    retina : e2cm.Retina, optional
        A Retina object specyfing the geometry of the retina.
    rsample : int, optional
        Resampling factor. For example, a resampling factor of 3 keeps
        only every third frame.
        Default: 30 frames per second.
    scale_charge : float, optional
        Scaling factor applied to charge accumulation (used to be called
        epsilon). Default: 42.1.
    tol : float, optional
        Ignore pixels whose effective current is smaller than a fraction `tol`
        of the max value.
        Default: 0.05.
    use_ecs : bool, optional
        Flag whether to use effective current spread (True) or regular
        current spread, unaffected by axon pathways (False).
        Default: True.
    engine : str, optional
        Which computational backend to use:
        - 'serial': Single-core computation
        - 'joblib': Parallelization via joblib (requires `pip install joblib`)
        - 'dask': Parallelization via dask (requires `pip install dask`)
        Default: joblib.
    dojit : bool, optional
        Whether to use just-in-time (JIT) compilation to speed up computation.
        Default: True.
    n_jobs : int, optional
        Number of cores (threads) to run the model on in parallel. Specify -1
        to use as many cores as possible.
        Default: -1.

    Returns
    -------
    A brightness movie depicting the predicted percept, running at `rsample`
    frames per second.

    Examples
    --------
    Stimulate a single-electrode array:

    >>> implant = e2cm.ElectrodeArray('subretinal', 0, 0, 0, 0)
    >>> stim = e2cm.Psycho2Pulsetrain(tsample=5e-6, freq=50, amp=20)
    >>> resp = pulse2percept(stim, implant)  # doctest: +SKIP

    Stimulate a single electrode ('C3') of an Argus I array centered on the
    fovea:

    >>> implant = e2cm.ArgusI()
    >>> stim = {'C3': e2cm.Psycho2Pulsetrain(tsample=5e-6, freq=50, amp=20)}
    >>> resp = pulse2percept(stim, implant)  # doctest: +SKIP
    """
    # Check type to avoid backwards compatibility issues
    if not isinstance(implant, e2cm.ElectrodeArray):
        raise TypeError("`implant` must be of type ec2b.ElectrodeArray")

    # Parse `stim` (either single pulse train or a list/dict of pulse trains),
    # and generate a list of pulse trains, one for each electrode
    pt_list = parse_pulse_trains(stim, implant)

    # Generate a standard temporal model if necessary
    if tm is None:
        tm = TemporalModel(pt_list[0].tsample)
    elif not isinstance(tm, TemporalModel):
        raise TypeError("`tm` must be of type ec2b.TemporalModel")

    # Generate a retina if necessary
    if retina is None:
        # Make sure implant fits on retina
        round_to = 500  # round to nearest (microns)
        cspread = 500  # expected current spread (microns)
        xs = [a.x_center for a in implant]
        ys = [a.y_center for a in implant]
        xlo = np.floor((np.min(xs) - cspread) / round_to) * round_to
        xhi = np.ceil((np.max(xs) + cspread) / round_to) * round_to
        ylo = np.floor((np.min(ys) - cspread) / round_to) * round_to
        yhi = np.ceil((np.max(ys) + cspread) / round_to) * round_to
        retina = e2cm.Retina(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi,
                             save_data=False)
    elif not isinstance(retina, e2cm.Retina):
        raise TypeError("`retina` object must be of type e2cm.Retina")

    # Which layer to simulate is given by implant type.
    # For now, both implant types process the same two layers. In the
    # future, these layers might differ. Order doesn't matter.
    if implant.etype == 'epiretinal':
        dolayers = ['NFL', 'INL']  # nerve fiber layer
    elif implant.etype == 'subretinal':
        dolayers = ['NFL', 'INL']  # inner nuclear layer
    else:
        e_s = "Supported electrode types are 'epiretinal', 'subretinal'"
        raise ValueError(e_s)

    # Derive the effective current spread
    if use_ecs:
        ecs, _ = retina.electrode_ecs(implant)
    else:
        _, ecs = retina.electrode_ecs(implant)

    # `ecs_list` is a pixel by `n` list where `n` is the number of layers
    # being simulated. Each value in `ecs_list` is the current contributed
    # by each electrode for that spatial location
    ecs_list = []
    idx_list = []
    for xx in range(retina.gridx.shape[1]):
        for yy in range(retina.gridx.shape[0]):
            # If any of the used current spread maps at [yy, xx] are above
            # tolerance, we need to process that pixel
            process_pixel = False
            if 'INL' in dolayers:
                layer = ecs[yy, xx, 0, :]
                process_pixel |= np.any(layer >= tol * layer.max())
            if 'NFL' in dolayers:
                layer = ecs[yy, xx, 1, :]
                process_pixel |= np.any(layer >= tol * layer.max())

            if process_pixel:
                ecs_list.append(ecs[yy, xx])
                idx_list.append([yy, xx])

    logger.info("tol=%.1f%%, %d/%d px selected" % (tol * 100,
                                                   len(ecs_list),
                                                   np.prod(ecs.shape[:2])))

    # Apply charge accumulation
    for i, p in enumerate(pt_list):
        ca = tm.tsample * np.cumsum(np.maximum(0, -p.data))
        tmp = signal.fftconvolve(ca, tm.gamma_ca, mode='full')
        conv_ca = scale_charge * tm.tsample * tmp[:p.data.size]

        # negative elements first
        idx = np.where(p.data <= 0)[0]
        pt_list[i].data[idx] = np.minimum(p.data[idx] + conv_ca[idx], 0)

        # then positive elements
        idx = np.where(p.data > 0)[0]
        pt_list[i].data[idx] = np.maximum(p.data[idx] - conv_ca[idx], 0)
    pt_arr = np.array([p.data for p in pt_list])

    sr_list = utils.parfor(calc_pixel, ecs_list, n_jobs=n_jobs, engine=engine,
                           func_args=[pt_arr, tm, rsample, dolayers, dojit])
    bm = np.zeros(retina.gridx.shape + (sr_list[0].data.shape[-1], ))
    idxer = tuple(np.array(idx_list)[:, i] for i in range(2))
    bm[idxer] = [sr.data for sr in sr_list]
    return utils.TimeSeries(sr_list[0].tsample, bm)


def calc_pixel(ecs_item, pt, tm, rsample, dolayers, dojit=False):
    """Calculates the temporal response of a single pixel

    Parameters
    ----------
    ecs_item : array
        Effective current spread map from the electrodes in a particular
        spatial location (pixel), one value per layer.
    pt : utils.TimeSeries
        Pulse train
    tm : ec2b.TemporalModel
        A model of temporal sensitivity.
    rsample : int
        Resampling factor. For example, a resampling factor of 3 keeps
        only every third frame.
        Default: 30 frames per second.
    dolayers : list
        List of retinal layers to simulate. Choose from:

        - 'NFL': nerve fiber layer
        - 'INL': inner nuclear layer
    dojit : bool, optional
        Whether to use just-in-time (JIT) compilation to speed up computation.
        Default: True.
    """
    # Convert the current map to one that includes axon streaks
    ecm = e2cm.ecm(ecs_item, pt, tm.tsample)

    # Apply temporal model that pixel
    sr = tm.model_cascade(ecm, dolayers, dojit=dojit)

    # Apply resampling factor
    sr.resample(rsample)

    return sr


def parse_pulse_trains(stim, implant):
    """Parse input stimulus and convert to list of pulse trains

    Parameters
    ----------
    stim : utils.TimeSeries|list|dict
        There are several ways to specify an input stimulus:

        - For a single-electrode array, pass a single pulse train; i.e., a
          single utils.TimeSeries object.
        - For a multi-electrode array, pass a list of pulse trains, where
          every pulse train is a utils.TimeSeries object; i.e., one pulse
          train per electrode.
        - For a multi-electrode array, specify all electrodes that should
          receive non-zero pulse trains by name in a dictionary. The key
          of each element is the electrode name, the value is a pulse train.
          Example: stim = {'E1': pt, 'stim': pt}, where 'E1' and 'stim' are
          electrode names, and `pt` is a utils.TimeSeries object.
    implant : e2cm.ElectrodeArray
        An ElectrodeArray object that describes the implant.

    Returns
    -------
    A list of pulse trains; one pulse train per electrode.
    """
    # Parse input stimulus
    if isinstance(stim, utils.TimeSeries):
        # `stim` is a single object: This is only allowed if the implant
        # has only one electrode
        if implant.num_electrodes > 1:
            e_s = "More than 1 electrode given, use a list of pulse trains"
            raise ValueError(e_s)
        pt = [stim]
    elif isinstance(stim, dict):
        # `stim` is a dictionary: Look up electrode names and assign pulse
        # trains, fill the rest with zeros

        # Get right size from first dict element, then generate all zeros
        idx0 = list(stim.keys())[0]
        pt_zero = utils.TimeSeries(stim[idx0].tsample,
                                   np.zeros_like(stim[idx0].data))
        pt = [pt_zero] * implant.num_electrodes

        # Iterate over dictionary and assign non-zero pulse trains to
        # corresponding electrodes
        for key, value in stim.items():
            el_idx = implant.get_index(key)
            if el_idx is not None:
                pt[el_idx] = value
            else:
                e_s = "Could not find electrode with name '%s'" % key
                raise ValueError(e_s)
    else:
        # Else, `stim` must be a list of pulse trains, one for each electrode
        if len(stim) != implant.num_electrodes:
            e_s = "Number of pulse trains must match number of electrodes"
            raise ValueError(e_s)
        pt = stim

    return pt


def get_brightest_frame(resp):
    """Return brightest frame of a brightness movie

    This function returns the frame of a brightness movie that contains the
    brightest pixel.

    Parameters
    ----------
    resp : TimeSeries
        The brightness movie as a TimeSeries object.
    """
    # Find brightest element
    idx_px = resp.data.argmax()

    # Find frame that contains brightest pixel using `unravel`, which maps
    # the flat index `idx_px` onto the high-dimensional indices (x,y,z).
    # What we want is index `z` (i.e., the frame index), given by the last
    # element in the return argument.
    idx_frame = np.unravel_index(idx_px, resp.shape)[-1]

    return utils.TimeSeries(resp.tsample, resp.data[..., idx_frame])
