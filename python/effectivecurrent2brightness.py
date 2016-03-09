# -*effectivecurrent2brightness -*-
"""effectivecurrent2brightness
This transforms the effective current into brightness for a single point in
space based on the Horsager model as modified by Devyani
Inputs: a vector of effective current over time
Output: a vector of brightness over time
"""
from __future__ import print_function
from scipy.misc import factorial
from scipy.signal import fftconvolve
import numpy as np
import utils
from utils import TimeSeries
import gc
import electrode2currentmap as e2cm

def gamma(n, tau, t):
    """
    returns a gamma function from in [0, t]:

    y = (t/theta).^(n-1).*exp(-t/theta)/(theta*factorial(n-1))

    which is the result of an n stage leaky integrator.
    """

    flag = 0
    if t[0] == 0:
        t = t[1:len(t)]
        flag = 1

    y = ((t/tau)  ** (n-1) *
        np.exp(-t / tau) /
        (tau * factorial(n-1)))

    if flag == 1:
        y = np.concatenate([[0], y])

    return y


class TemporalModel(object):
    def __init__(self, tau1=.42/1000, tau2=45.25/1000,
                 tau3=26.25/1000, e=8.73, beta=.6, asymptote=14, slope=3,
                 shift=16):
        """
        A model of temporal integration from retina pixels

        Fs : sampling rate

        tau1 = .42/1000  is a parameter for the fast leaky integrator, from
        Alan model, tends to be between .24 - .65

        tau2 = 45.25/1000  integrator for charge accumulation, has values
        between 38-57

        e = scaling factor for the effects of charge accumulation 2-3 for
        threshold or 8-10 for suprathreshold

        tau3 = ??

        parameters for a stationary nonlinearity providing a continuous
        function that nonlinearly rescales the response based on Nanduri et al
        2012, equation 3:

        asymptote = 14
        slope =.3
        shift =47
        """
        self.tau1 = tau1
        self.tau2 = tau2
        self.tau3 = tau3
        self.e = e
        self.beta = beta
        self.asymptote = asymptote
        self.slope = slope
        self.shift = shift

    def fast_response(self, stimulus, dojit=True):
        """
        Fast response function
        """
        t = np.arange(0, 20 * self.tau1, stimulus.tsample)
        g = gamma(1, self.tau1, t)
        R1 = stimulus.tsample * utils.sparseconv(g, stimulus.data, dojit)
        return TimeSeries(stimulus.tsample, R1)

    def charge_accumulation(self, fast_response, stimulus):
        t = np.arange(0, 8 * self.tau2, fast_response.tsample)

        # calculated accumulated charge
        rect_amp = np.where(stimulus.data > 0, stimulus.data, 0)  # rectify
        ca = stimulus.tsample * np.cumsum(rect_amp.astype(float), axis=-1)
        g = gamma(1, self.tau2, t)
        chargeaccumulated = (self.e * stimulus.tsample *
                             fftconvolve(g, ca))
        zero_pad = np.zeros(fast_response.shape[:-1] +
                            (chargeaccumulated.shape[-1] -
                             fast_response.shape[-1],))

        fast_response = TimeSeries(fast_response.tsample,
                                   np.concatenate([fast_response.data,
                                                   zero_pad], -1))

        R2 = fast_response.data - chargeaccumulated
        ind = R2 < 0
        R2 = np.where(R2 > 0, R2, 0)  # rectify again
        return TimeSeries(fast_response.tsample, R2)

    def stationary_nonlinearity(self, fast_response_ca):
        # now we put in the stationary nonlinearity of Devyani's:
        R2norm = fast_response_ca.data / fast_response_ca.data.max()
        scale_factor = (self.asymptote / (1 + np.exp(-(fast_response_ca.data /
                        self.slope) +
                        self.shift)))
        R3 = R2norm * scale_factor  # scaling factor varies with original
        return TimeSeries(fast_response_ca.tsample, R3)

    def slow_response(self, fast_response_ca_snl):
        # this is cropped as tightly as
        # possible for speed sake
        t = np.arange(0, self.tau3 * 8, fast_response_ca_snl.tsample)
        g = gamma(3, self.tau3, t)
        c = fftconvolve(g, fast_response_ca_snl.data)
        return TimeSeries(fast_response_ca_snl.tsample,
                          fast_response_ca_snl.tsample * c)

    def model_cascade(self, ecm, dojit):
        fr = self.fast_response(ecm, dojit=dojit)
        ca = self.charge_accumulation(fr, ecm)
        sn = self.stationary_nonlinearity(ca)
        return self.slow_response(sn)


def pulse2percept(temporal_model, ecs, retina, stimuli,
                  fps=30, dojit=True, n_jobs=-1, tol=1e-10):
    """
    From pulses (stimuli) to percepts (spatio-temporal)

    Parameters
    ----------
    temporal_model : emporalModel class instance.
    ecs : ndarray
    retina : a Retina class instance.
    stimuli : list
    subsample_factor : float/int, optional
    dojit : bool, optional
    """
    rs = int(1 / (fps*stimuli[0].tsample))
    ecs_list = []
    idx_list = []
    for xx in range(retina.gridx.shape[1]):
        for yy in range(retina.gridx.shape[0]):
            if np.all(ecs[yy, xx] < tol):
                pass
            else:
                ecs_list.append(ecs[yy, xx])
                idx_list.append([yy, xx])

    stim_data = np.array([s.data for s in stimuli])
    sr_list = utils.parfor(calc_pixel, ecs_list, n_jobs=n_jobs,
                           func_args=[stim_data, temporal_model,
                                      rs, dojit, stimuli[0].tsample])
    bm = np.zeros(retina.gridx.shape + (sr_list[0].data.shape[-1], ))
    idxer = tuple(np.array(idx_list)[:, i] for i in range(2))
    bm[idxer] = [sr.data for sr in sr_list]

    return TimeSeries(sr_list[0].tsample, bm)


def calc_pixel(ecs_vector, stim_data, temporal_model, rs, dojit, tsample):
    ecm = e2cm.ecm(ecs_vector, stim_data, tsample)
    sr = temporal_model.model_cascade(ecm, dojit=dojit)
    sr.resample(rs)
    del temporal_model, ecm
    gc.collect()
    return sr
