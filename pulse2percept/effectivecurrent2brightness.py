# -*effectivecurrent2brightness -*-
"""effectivecurrent2brightness
This transforms the effective current into brightness for a single point in
space based on the Horsager model as modified by Devyani
Inputs: a vector of effective current over time
Output: a vector of brightness over time
"""
from __future__ import print_function
import numpy as np
from scipy.signal import fftconvolve
from scipy.special import expit

import pulse2percept.electrode2currentmap as e2cm
from pulse2percept import utils


class TemporalModel(object):

    def __init__(self, tsample=0.005 / 1000,
                 tau_nfl=0.42 / 1000, tau_inl=18.0 / 1000,
                 tau_ca=45.25 / 1000, tau_slow=26.25 / 1000,
                 lweight=0.636, scale_slow=1150.0,
                 asymptote=1.0, slope=3.0, shift=15.0):
        """Temporal Sensitivity Model

        A model of temporal integration from retina pixels.

        Parameters
        ----------
        tsample : float
            Sampling time step (seconds). Default: 5e-6 s.
        tau1 : float
            Parameter for the fast leaky integrator for each layer, tends to be
            between 0.24 - 0.65 ms for ganglion cells, 14 - 18 ms for bipolar
            cells. Default: 4.2e-4 s.
        tau_ca : float
            Parameter for the charge accumulation, has values between 38 - 57
            ms. Default: 4.525e-2 s.
        tau_slow : float
            Parameter for the slow leaky integrator. Default: 2.625e-2.
        asymptote : float
            Asymptote of the logistic function in the stationary nonlinearity
            stage. Default: 14.
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
        self.asymptote = asymptote
        self.slope = slope
        self.shift = shift
        self.lweight = lweight
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

    def fast_response(self, b1, gamma, dojit=True, usefft=False):
        """Fast response function (Box 2) for the bipolar layer

        Convolve a stimulus `b1` with a temporal low-pass filter (1-stage
        gamma) with time constant `self.tau_inl` ~ 14ms representing bipolars.

        Parameters
        ----------
        b1 : array
           Temporal signal to process, b1(r,t) in Nanduri et al. (2012).
        dojit : bool, optional
           If True (default), use numba just-in-time compilation.
        usefft : bool, optional
           If False (default), use sparseconv, else fftconvolve.

        Returns
        -------
        b2 : array
           Fast response, b2(r,t) in Nanduri et al. (2012).

        Notes
        -----
        The function utils.sparseconv can be much faster than np.convolve and
        scipy.signals.fftconvolve if `b1` is sparse and much longer than the
        convolution kernel.

        The output is not converted to a TimeSeries object for speedup.
        """

        if usefft:  # In Krishnan model, b1 is no longer sparse (run FFT)
            conv = self.tsample * fftconvolve(b1, gamma, mode='full')
        else:  # In Nanduri model, b1 is sparse. Use sparseconv.
            conv = self.tsample * utils.sparseconv(gamma, b1,
                                                   mode='full', dojit=dojit)
            # Cut off the tail of the convolution to make the output signal
            # match the dimensions of the input signal.

        # return self.tsample * conv[:, b1.shape[-1]]
        return conv[:b1.shape[-1]]

    def stationary_nonlinearity(self, b3):
        """Stationary nonlinearity (Box 4)

        Nonlinearly rescale a temporal signal `b3` across space and time, based
        on a sigmoidal function dependent on the maximum value of `b3`.
        This is Box 4 in Nanduri et al. (2012).

        The parameter values of the asymptote, slope, and shift of the logistic
        function are given by self.asymptote, self.slope, and self.shift,
        respectively.

        Parameters
        ----------
        b3 : array
           Temporal signal to process, b3(r,t) in Nanduri et al. (2012).

        Returns
        -------
        b4 : array
           Rescaled signal, b4(r,t) in Nanduri et al. (2012).

        Notes
        -----
        Conversion to TimeSeries is avoided for the sake of speedup.
        (np.sum(y) * (t[1]-t[0]))
        """

        # use expit (logistic) function for speedup
        b3max = b3.max()
        scale = expit((b3max - self.shift) / self.slope) * self.asymptote

        # avoid division by zero
        return b3 / (b3max + np.finfo(float).eps) * scale

    def slow_response(self, b4):
        """Slow response function (Box 5)

        Convolve a stimulus `b4` with a low-pass filter (3-stage gamma)
        with time constant self.tau_slow.
        This is Box 5 in Nanduri et al. (2012).

        Parameters
        ----------
        b4 : array
           Temporal signal to process, b4(r,t) in Nanduri et al. (2012)

        Returns
        -------
        b5 : array
           Slow response, b5(r,t) in Nanduri et al. (2012).

        Notes
        -----
        This is by far the most computationally involved part of the perceptual
        sensitivity model.

        Conversion to TimeSeries is avoided for the sake of speedup.
        """
        # No need to zero-pad: fftconvolve already takes care of optimal
        # kernel/data size
        conv = fftconvolve(b4, self.gamma_slow, mode='full')

        # Cut off the tail of the convolution to make the output signal match
        # the dimensions of the input signal.
        return self.scale_slow * self.tsample * conv[:b4.shape[-1]]

    def model_cascade(self, ecm, dolayer, dojit):
        """Model cascade according to Nanduri et al. (2012).

        The 'Nanduri' model calculates the fast response first, followed by the
        current accumulation.

        Parameters
        ----------
        ecm : TimeSeries
            Effective current

        Returns
        -------
        b5 : TimeSeries
            Brightness response over time. In Nanduri et al. (2012), the
            maximum value of this signal was used to represent the perceptual
            brightness of a particular location in space, B(r).
        """
        if 'INL' in dolayer:
            resp_inl = self.fast_response(ecm[0].data, self.gamma_inl,
                                          dojit=dojit,
                                          usefft=True)
        else:
            resp_inl = np.zeros((ecm[0].data.shape))

        if 'NFL' in dolayer:
            resp_nfl = self.fast_response(ecm[1].data, self.gamma_nfl,
                                          dojit=dojit,
                                          usefft=False)
        else:
            resp_nfl = np.zeros((ecm[1].data.shape))

        # here we are converting from current  - where a cathodic (effective)
        # stimulus is negative to a vague concept of neuronal response, where
        # positive implies a neuronal response
        # There is a rectification here because we assume that the anodic part
        # of the pulse is ineffective which is wrong
        respC = self.lweight * np.maximum(-resp_inl, 0) + \
            np.maximum(-resp_nfl, 0)
        respA = self.lweight * np.maximum(resp_inl, 0) + \
            np.maximum(resp_nfl, 0)
        resp = respC + 0.5 * respA
        resp = self.stationary_nonlinearity(resp)
        resp = self.slow_response(resp)
        return utils.TimeSeries(self.tsample, resp)


def pulse2percept(tm, ecs, retina, ptrain, rsample, dolayer,
                  scale_charge=42.1,
                  engine='joblib', dojit=True, n_jobs=-1, tol=0.05):
    """
    From pulses (stimuli) to percepts (spatio-temporal)

    Parameters
    ----------
    tm : TemporalModel class instance.
    ecs : ndarray
    retina : Retina class instance.
    ptrain : list
    rsample : float/int, optional
    dolayer : str
    scale_charge : float
        Scaling factor applied to charge accumulation (used to be called
        epsilon). Default: 42.1.
    dojit : bool, optional
    """

    # `ecs_list` is a pixel by `n` list where `n` is the number of layers
    # being simulated. Each value in `ecs_list` is the current contributed
    # by each electrode for that spatial location
    ecs_list = []
    idx_list = []
    for xx in range(retina.gridx.shape[1]):
        for yy in range(retina.gridx.shape[0]):
            if np.all(ecs[yy, xx] < tol):
                pass
            else:
                ecs_list.append(ecs[yy, xx])
                idx_list.append([yy, xx])

    # pulse train for each electrode
    for i, p in enumerate(ptrain):
        ca = tm.tsample * np.cumsum(np.maximum(0, -p.data))
        tmp = fftconvolve(ca, tm.gamma_ca, mode='full')
        conv_ca = scale_charge * tm.tsample * tmp[:p.shape[-1]]

        # negative elements first
        idx = np.where(p.data <= 0)[0]
        ptrain[i].data[idx] = np.minimum(p.data[idx] + conv_ca[idx], 0)

        # then positive elements
        idx = np.where(p.data > 0)[0]
        ptrain[i].data[idx] = np.maximum(p.data[idx] - conv_ca[idx], 0)

    ptrain_data = np.array([p.data for p in ptrain])

    sr_list = utils.parfor(calc_pixel, ecs_list, n_jobs=n_jobs, engine=engine,
                           func_args=[ptrain_data, tm, rsample,
                                      dolayer, dojit])
    bm = np.zeros(retina.gridx.shape + (sr_list[0].data.shape[-1], ))
    idxer = tuple(np.array(idx_list)[:, i] for i in range(2))
    bm[idxer] = [sr.data for sr in sr_list]
    return utils.TimeSeries(sr_list[0].tsample, bm)


def calc_pixel(ecs_item, ptrain_data, tm, resample, dolayer,
               dojit=False):
    ecm = e2cm.ecm(ecs_item, ptrain_data, tm.tsample)
    # converts the current map to one that includes axon streaks
    sr = tm.model_cascade(ecm, dolayer, dojit=dojit)
    sr.resample(resample)
    return sr


def get_brightest_frame(resp):
    """Return brightest frame of a brightness movie

    This function returns the frame of a brightness movie that contains the
    brightest element.

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
