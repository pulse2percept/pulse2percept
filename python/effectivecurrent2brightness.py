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
import matplotlib.pyplot as plt
import utils
from utils import TimeSeries


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

    def fast_response(self, stimulus):
        """
        Fast response function
        """
        t = np.arange(0, 20 * self.tau1, stimulus.tsample)
        R1 = stimulus.tsample * np.convolve(gamma(1, self.tau1, t),
                                            stimulus.data)
        return TimeSeries(stimulus.tsample, R1)

    def charge_accumulation(self, fast_response, stimulus):
        t = np.arange(0, 8 * self.tau2, stimulus.tsample)

        # calculated accumulated charge
        rect_amp = np.where(stimulus.data > 0, stimulus.data, 0)  # rectify
        ca = stimulus.tsample * np.cumsum(rect_amp.astype(float))
        chargeaccumulated = (self.e * stimulus.tsample *
                             fftconvolve(gamma(1, self.tau2, t), ca))

        fast_response = TimeSeries(fast_response.tsample,
                                   np.concatenate([fast_response.data,
                                            np.zeros(len(chargeaccumulated) -
                                                     fast_response.shape[0])]))

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
        G3 = gamma(3, self.tau3, t)

       # conv = np.convolve(G3, fast_response_ca_snl)
        conv = fftconvolve(G3, fast_response_ca_snl.data)
        R4 = fast_response_ca_snl.tsample * conv

        return TimeSeries(fast_response_ca_snl.tsample, R4)
