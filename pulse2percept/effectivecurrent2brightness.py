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

    def __init__(self, model='Krishnan', tsample=0.005 / 1000,
                 tau_nfl=.42 / 1000, tau_inl=18. / 1000,
                 lweight=(1 / (3.16 * (10 ** 6))), tau2=45.25 / 1000,
                 tau3=26.25 / 1000, epsilon=8.78,
                 asymptote=14., slope=3., shift=16.):
        """(Updated) Perceptual Sensitivity Model.

        A model of temporal integration from retina pixels.
        The model transforms the effective current into brightness for a single
        point in space. The model is based on Horsager et al. (2009), was
        adapted by Nanduri et al. (2012), and contains some novel features as
        well.

        The model comes in two flavors: 'Nanduri' implements the model cascade
        as described in Fig. 6 of Nanduri et al. (2012). Effective current is
        first convolved with a fast gamma function (tau_nfl), before the
        response is adjusted based on accumulated cathodic charge (tau2).
        'Krishnan' inverts this logic, where effective current is first
        adjusted based on accumulated cathodic charge, and then convolved with
        a fast gamma function.
        The two model flavors are mathematically equivalent.

        Parameters
        ----------
        model : str {'Nanduri', 'Krishnan'}
            A string indicating which flavor of the model to use.
            Default: 'Nanduri'.
        ``Nanduri``
            Fast leaky integrator first, then charge accumulation.
        ``Krishnan``
            Charge accumulation first, then fast leaky integrator.
        tsample : float
            Sampling time step (seconds). Default: 5e-6 s.
        tau1 : float
            Parameter for the fast leaky integrator for each layer, tends to be
            between 0.24 - 0.65 ms for ganglion cells, 14 - 18 ms for bipolar
            cells. Default: 4.2e-4 s.
        tau2 : float
            Parameter for the charge accumulation, has values between 38 - 57
            ms. Default: 4.525e-2 s.
        tau3 : float
            Parameter for the slow leaky integrator. Default: 2.625e-2.
        epsilon : float
            Scaling factor for the effects of charge accumulation, has values
            2-3 for threshold or 8-10 for suprathreshold. Default: 8.73. If all the gammas are
            normalized gotes to 8.78
        asymptote : float
            Asymptote of the logistic function in the stationary nonlinearity
            stage. Default: 14.
        slope : float
            Slope of the logistic function in the stationary nonlinearity
            stage. Default: 3.
        shift : float
            Shift of the logistic function in the stationary nonlinearity
            stage. Default: 16.
        """
        self.model = model
        self.tsample = tsample
        self.tau_nfl = tau_nfl
        self.tau_inl = tau_inl
        self.tau2 = tau2
        self.tau3 = tau3
        self.epsilon = epsilon
        self.asymptote = asymptote
        self.slope = slope
        self.shift = shift
        self.lweight = lweight

        # perform one-time setup calculations
        # Gamma functions used as convolution kernels do not depend on input
        # data, hence can be calculated once, then re-used (trade off memory
        # for speed).
        # gamma_nfl and gamma_inl are used to calculate the fast response in
        # bipolar and ganglion cells respectively

        self.gamma_inl = []
        t = np.arange(0, 10 * self.tau_inl, self.tsample)
        self.gamma_inl = e2cm.gamma(1, self.tau_inl, t)
        
        self.gamma_nfl = []
        t = np.arange(0, 10 * self.tau_nfl, self.tsample)
        self.gamma_nfl = e2cm.gamma(1, self.tau_nfl, t)

        # gamma2 is used to calculate charge accumulation
        t = np.arange(0, 8 * self.tau2, self.tsample)
        self.gamma2 = e2cm.gamma(1, self.tau2, t)

        # gamma3 is used to calculate the slow response
        t = np.arange(0, 8 * self.tau3, self.tsample)
        self.gamma3 = e2cm.gamma(3, self.tau3, t)

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
        return self.tsample * conv[:b1.shape[-1]] * np.sum(gamma)

    def charge_accumulation(self, b2):
        """Charge accumulation step (Box 3)

        Calculate the accumulated cathodic charge of a stimulus `b2`.
        This accumulated charge is then convolved with a one-stage gamma
        function of time constant `self.tau2`.

        Parameters
        ----------
        b2 : array
            Temporal signal to process, b2(r,t) in Nanduri et al. (2012).

        Returns
        -------
        charge_accum : array
            Accumulated charge over time.

        Notes
        -----
        The output is not converted to a TimeSeries object for speedup.
        """
        # np.maximum seems to be faster than np.where
        ca = self.tsample * np.cumsum(np.maximum(0, b2), axis=-1)

        conv = fftconvolve(ca, self.gamma2, mode='full') * np.sum(self.gamma2)

        # Cut off the tail of the convolution to make the output signal match
        # the dimensions of the input signal
        return self.epsilon * self.tsample * conv[:b2.shape[-1]]

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
        """
        # rectify: np.maximum seems to be faster than np.where
        b3 = np.maximum(0, b3)

        # use expit (logistic) function for speedup
        b3max = b3.max()
        scale = expit((b3max - self.shift) / self.slope) * self.asymptote

        # avoid division by zero
        return b3 / (1e-10 + b3max) * scale

    def slow_response(self, b4):
        """Slow response function (Box 5)

        Convolve a stimulus `b4` with a low-pass filter (3-stage gamma)
        with time constant self.tau3.
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
        conv = fftconvolve(b4, self.gamma3, mode='full')

        # Cut off the tail of the convolution to make the output signal match
        # the dimensions of the input signal.
        return self.tsample * conv[:b4.shape[-1]] * np.sum(self.gamma3)

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
        ca = 0
        if 'INL' in dolayer:
            if self.model == 'Nanduri':
                ca = self.charge_accumulation(ecm[0].data)
            resp_inl = (self.fast_response(ecm[0].data, self.gamma_inl, dojit=dojit,
                                               usefft=True) - ca)
        else:
            resp_inl = np.zeros((ecm[0].data.shape))

        if 'NFL' in dolayer:
            if self.model == 'Nanduri':
                ca = self.charge_accumulation(ecm[1].data)
            resp_nfl = self.fast_response_nfl(ecm[1].data, self.gamma_nfl, dojit=dojit,
                                              usefft=False) - ca
        else:
            resp_nfl = np.zeros((ecm[1].data.shape))

        resp = (self.lweight * resp_inl) + resp_nfl

        resp = self.stationary_nonlinearity(resp)
        resp = self.slow_response(resp)
        return utils.TimeSeries(self.tsample, resp)


def pulse2percept(tm, ecs, retina, ptrain, rsample, dolayer,
                  engine='joblib', dojit=True, n_jobs=-1, tol=.05):
    """
    From pulses (stimuli) to percepts (spatio-temporal)

    Parameters
    ----------
    tm : TemporalModel class instance.
    ecs : ndarray
    retina : Retina class instance.
    stimuli : list
    subsample_factor : float/int, optional
    dojit : bool, optional
    """

    ecs_list = []
    idx_list = []
    for xx in range(retina.gridx.shape[1]):
        for yy in range(retina.gridx.shape[0]):
            if np.all(ecs[yy, xx] < tol):
                pass
            else:
                ecs_list.append(ecs[yy, xx])
                idx_list.append([yy, xx])
                # ecs_list is a pix by n list where n is the number of
                # layers being simulated
                # each value in ecs is the current contributed by
                # each electrode for that spatial location

    # pulse train for each electrode
    if tm.model == 'Krishnan':
        for p in range(len(ptrain)):
            ca = tm.tsample * np.cumsum(np.maximum(0, ptrain[p].data))
            tmp = fftconvolve(ca, tm.gamma2, mode='full')
            conv_ca = tm.epsilon * tm.tsample * tmp[:ptrain[p].shape[-1]]
            ptrain[p].data = ptrain[p].data - conv_ca

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
