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
from scipy.signal import convolve2d
import numpy as np
import utils
from utils import TimeSeries
import gc
import electrode2currentmap as e2cm


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
        g = e2cm.gamma(1, self.tau1, t)
        R1 = stimulus.tsample * utils.sparseconv(g, stimulus.data, dojit)
        return TimeSeries(stimulus.tsample, R1)

    def charge_accumulation(self, fast_response, stimulus):
        t = np.arange(0, 8 * self.tau2, fast_response.tsample)

        # calculated accumulated charge
        rect_amp = np.where(stimulus.data > 0, stimulus.data, 0)  # rectify
        ca = stimulus.tsample * np.cumsum(rect_amp.astype(float), axis=-1)
        g = e2cm.gamma(1, self.tau2, t)
        chargeaccumulated = (self.e * stimulus.tsample *
                             fftconvolve(g, ca))
        zero_pad = np.zeros(fast_response.shape[:-1] +
                            (chargeaccumulated.shape[-1] -
                             fast_response.shape[-1],))

        fast_response = TimeSeries(fast_response.tsample,
                                   np.concatenate([fast_response.data,
                                                   zero_pad], -1))

        R2 = fast_response.data - chargeaccumulated
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
        g = e2cm.gamma(3, self.tau3, t)
        c = fftconvolve(g, fast_response_ca_snl.data)
        return TimeSeries(fast_response_ca_snl.tsample,
                          fast_response_ca_snl.tsample * c)

    def model_cascade(self, ecm, dojit):
        fr = self.fast_response(ecm, dojit=dojit)
        # ca = self.charge_accumulation(fr, ecm)
        # this line deleted because charge accumulation now modeled at the 
        # elecrode level as accumulated voltage
        sn = self.stationary_nonlinearity(fr)
        return self.slow_response(sn)


def pulse2percept(temporal_model, ecs, retina, stimuli, rs, dojit=True, n_jobs=-1, tol=.05):
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
                # the current contributed by each electrode for that spatial location

    stim_data = np.array([s.data for s in stimuli])  # pulse train for each electrode
    sr_list = utils.parfor(calc_pixel, ecs_list, n_jobs=n_jobs,
                           func_args=[stim_data, temporal_model,
                                      rs,  stimuli[0].tsample, dojit])
    bm = np.zeros(retina.gridx.shape + (sr_list[0].data.shape[-1], ))
    idxer = tuple(np.array(idx_list)[:, i] for i in range(2))
    bm[idxer] = [sr.data for sr in sr_list]
    return TimeSeries(sr_list[0].tsample, bm)


def calc_pixel(ecs_vector, stim_data, temporal_model, rs, tsample, dojit='False'):
    ecm = e2cm.ecm(ecs_vector, stim_data, tsample)
    sr = temporal_model.model_cascade(ecm, dojit=dojit)
    del temporal_model, ecm
    gc.collect()
    sr.resample(rs)
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
    onmovie = np.zeros([movie.shape[0], movie.shape[1], movie.shape[2]])
    offmovie = np.zeros([movie.shape[0], movie.shape[1], movie.shape[2]])
    newfiltImgOn=np.zeros([movie.shape[0], movie.shape[1]])
    newfiltImgOff=np.zeros([movie.shape[0], movie.shape[1]])
    pad = max(n)*2
    for xx in range(movie.shape[-1]):
        oldimg=movie[:, :, xx]
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