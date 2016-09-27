# -*effectivecurrent2brightness -*-
"""effectivecurrent2brightness
This transforms the effective current into brightness for a single point in
space based on the Horsager model as modified by Devyani
Inputs: a vector of effective current over time
Output: a vector of brightness over time
"""
from __future__ import print_function
import numpy as np
from scipy.misc import factorial
from scipy.signal import fftconvolve
from scipy.signal import convolve2d
from scipy.special import expit

import pulse2percept.electrode2currentmap as e2cm
from pulse2percept import utils


class TemporalModel(object):
    def __init__(self, model='Nanduri', tsample=5e-6, tau1=4.2e-4,
                 tau2=4.525e-2, tau3=2.625e-2, epsilon=8.73, asymptote=14,
                 slope=3, shift=16):
        """(Updated) Perceptual Sensitivity Model.

        A model of temporal integration from retina pixels.
        The model transforms the effective current into brightness for a single
        point in space. The model is based on Horsager et al. (2009), was
        adapted by Nanduri et al. (2012), and contains some novel features as
        well.

        The model comes in two flavors: 'Nanduri' implements the model cascade
        as described in Fig. 6 of Nanduri et al. (2012). Effective current is
        first convolved with a fast gamma function (tau1), before the response
        is adjusted based on accumulated cathodic charge (tau2).
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
            Parameter for the fast leaky integrator, tends to be between 0.24 -
            0.65 ms. Default: 4.2e-4 s.
        tau2 : float
            Parameter for the charge accumulation, has values between 38 - 57
            ms. Default: 4.525e-2 s.
        tau3 : float
            Parameter for the slow leaky integrator. Default: 2.625e-2.
        epsilon : float
            Scaling factor for the effects of charge accumulation, has values
            2-3 for threshold or 8-10 for suprathreshold. Default: 8.73.
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
        self.tau1 = tau1
        self.tau2 = tau2
        self.tau3 = tau3
        self.epsilon = epsilon
        self.asymptote = asymptote
        self.slope = slope
        self.shift = shift

        # perform onte-time setup calculations
        # Gamma functions used as convolution kernels do not depend on input
        # data, hence can be calculated once, then re-used (trade off memory
        # for speed).
        # gamma1 is used to calculate the fast response
        t = np.arange(0, 20 * self.tau1, self.tsample)
        self.gamma1 = e2cm.gamma(1, self.tau1, t)

        # gamma2 is used to calculate charge accumulation
        t = np.arange(0, 8 * self.tau2, self.tsample)
        self.gamma2 = e2cm.gamma(1, self.tau2, t)

        # gamma3 is used to calculate the slow response
        t = np.arange(0, 8 * self.tau3, self.tsample)
        self.gamma3 = e2cm.gamma(3, self.tau3, t)


    def fast_response(self, b1, dojit=True, usefft=False):
        """Fast response function (Box 2)

        Convolve a stimulus `b1` with a temporal low-pass filter (1-stage
        gamma) with time constant `self.tau1`.
        This is Box 2 in Nanduri et al. (2012).

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
        if usefft:
            # In Krishnan model, b1 is no longer sparse. Run FFT instead.
            conv = fftconvolve(b1, self.gamma1, mode='full')
        else:
            # In Nanduri model, b1 is sparse. Use sparseconv.
            conv = utils.sparseconv(self.gamma1, b1, mode='full', dojit=dojit)

        # Cut off the tail of the convolution to make the output signal match
        # the dimensions of the input signal.
        return self.tsample * conv[:b1.shape[-1]]


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

        conv = fftconvolve(ca, self.gamma2, mode='full')

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
        return self.tsample * conv[:b4.shape[-1]]

    def model_cascade(self, ecm, dojit):
        """Executes the whole cascade of the perceptual sensitivity model.

        The order of the cascade stages depend on the model flavor: either
        'Nanduri' or 'Krishnan'.

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
        if self.model == 'Nanduri':
            # Nanduri: first fast response, then charge accumulation
            return self.cascade_nanduri(ecm, dojit)
        elif self.model == 'Krishnan':
            # Krishnan: first charge accumulation, then fast response
            return self.cascade_krishnan(ecm, dojit)
        else:
            raise ValueError('Acceptable values for "model" are: '
                             '{"Nanduri", "Krishnan"}')

    def cascade_nanduri(self, ecm, dojit):
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
        resp = self.fast_response(ecm.data, dojit=dojit, usefft=False)
        ca = self.charge_accumulation(ecm.data)
        resp = self.stationary_nonlinearity(resp - ca)
        resp = self.slow_response(resp)
        return utils.TimeSeries(self.tsample, resp)

    def cascade_krishnan(self, ecm, dojit):
        """Model cascade according to Krishnan et al. (2015).

        The 'Krishnan' model calculates the current accumulation first,
        followed by the fast response.

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
        ca = self.charge_accumulation(ecm.data)
        resp = self.fast_response(ecm.data - ca, dojit=dojit, usefft=True)
        resp = self.stationary_nonlinearity(resp)
        resp = self.slow_response(resp)
        return utils.TimeSeries(self.tsample, resp)


def pulse2percept(temporal_model, ecs, retina, stimuli, rs, engine='joblib',
                  dojit=True, n_jobs=-1, tol=.05):
    """
    From pulses (stimuli) to percepts (spatio-temporal)

    Parameters
    ----------
    temporal_model : temporalModel class instance.
    ecs : ndarray
    retina : a Retina class instance.
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
                # the current contributed by each electrode for that spatial
                # location

    # pulse train for each electrode
    stim_data = np.array([s.data for s in stimuli])
    sr_list = utils.parfor(calc_pixel, ecs_list, n_jobs=n_jobs, engine=engine,
                           func_args=[stim_data, temporal_model, rs,
                                      stimuli[0].tsample, dojit])
    bm = np.zeros(retina.gridx.shape + (sr_list[0].data.shape[-1], ))
    idxer = tuple(np.array(idx_list)[:, i] for i in range(2))
    bm[idxer] = [sr.data for sr in sr_list]
    return utils.TimeSeries(sr_list[0].tsample, bm)


def calc_pixel(ecs_vector, stim_data, temporal_model, resample_factor,
               tsample, dojit=False):
    ecm = e2cm.ecm(ecs_vector, stim_data, tsample)
    sr = temporal_model.model_cascade(ecm, dojit=dojit)
    sr.resample(resample_factor)
    return sr

def onoffFiltering(movie, n, sig=[.1, .25],amp=[.01, -0.005]):
    """
    From a movie to a version that is filtered by a collection on and off cells
    of sizes

    Parameters
    ----------
    movie: movie to be filtered
    n : the sizes of the retinal ganglion cells (in μm, 293 μm equals 1 degree)
    """
    onmovie = np.zeros([movie.data.shape[0], movie.data.shape[1], movie.data.shape[2]])
    offmovie = np.zeros([movie.data.shape[0], movie.data.shape[1], movie.data.shape[2]])
    newfiltImgOn=np.zeros([movie.shape[0], movie.shape[1]])
    newfiltImgOff=np.zeros([movie.shape[0], movie.shape[1]])
    pad = max(n)*2
    for xx in range(movie.shape[-1]):
        oldimg=movie[:, :, xx].data
        tmpimg=np.mean(np.mean(oldimg))*np.ones([oldimg.shape[0]+pad*2,oldimg.shape[1]+pad*2])
        img = insertImg(tmpimg, oldimg)
        filtImgOn=np.zeros([img.shape[0], img.shape[1]])
        filtImgOff=np.zeros([img.shape[0], img.shape[1]])

        for i in range(n.shape[0]):
            [x,y] = np.meshgrid(np.linspace(-1,1,n[i]),np.linspace(-1,1,n[i]))
            rsq = x**2+y**2
            dx = x[0,1]-x[0,0]
            on = np.exp(-rsq/(2*sig[0]**2))*(dx**2)/(2*np.pi*sig[0]**2)
            off = np.exp(-rsq/(2*sig[1]**2))*(dx**2)/(2*np.pi*sig[1]**2)
            filt = on-off
            tmp_on = convolve2d(img,filt,'same')/n.shape[-1]
            tmp_off=tmp_on
            tmp_on= np.where(tmp_on>0, tmp_on, 0)
            tmp_off= -np.where(tmp_off<0, tmp_off, 0)
             #   rectified = np.where(ptrain.data > 0, ptrain.data, 0)
            filtImgOn =    filtImgOn+tmp_on/n.shape[0]
            filtImgOff =   filtImgOff+tmp_off/n.shape[0]

        # Remove padding
        nopad=np.zeros([img.shape[0]-pad*2, img.shape[1]-pad*2])
        newfiltImgOn[:,:] = insertImg(nopad,filtImgOn)
        newfiltImgOff[:, :] = insertImg(nopad,filtImgOff)
        onmovie[:, :, xx]=newfiltImgOn
        offmovie[:, :, xx]=newfiltImgOff

    return (onmovie, offmovie)

def onoffRecombine(onmovie, offmovie):
    """
    From a movie as filtered by on and off cells,
    to a recombined version that is either based on an electronic
    prosthetic (on + off) or recombined as might be done by a cortical
    cell in normal vision (on-off)
    Parameters
    ----------
    movie: on and off movies to be recombined
    combination : options are 'both' returns both prosthetic and normal vision, 'normal' and 'prosthetic'
    """

    prostheticmovie=onmovie + offmovie
    normalmovie=onmovie - offmovie
    return (normalmovie, prostheticmovie)


def insertImg(out_img,in_img):
    """ insertImg(out_img,in_img)
    Inserts in_img into the center of out_img.
    if in_img is larger than out_img, in_img is cropped and centered.
    """

    if in_img.shape[0]>out_img.shape[0]:
        x0 = np.floor([(in_img.shape[0]-out_img.shape[0])/2])
        xend=x0+out_img.shape[0]
        in_img=in_img[x0:xend, :]

    if in_img.shape[1]>out_img.shape[1]:
        y0 = np.floor([(in_img.shape[1]-out_img.shape[1])/2])
        yend=y0+out_img.shape[1]
        in_img=in_img[:, y0:yend]

    x0 = np.floor([(out_img.shape[0]-in_img.shape[0])/2])
    y0 = np.floor([(out_img.shape[1]-in_img.shape[1])/2])
    out_img[x0:x0+in_img.shape[0], y0:y0+in_img.shape[1]] = in_img

    return out_img
