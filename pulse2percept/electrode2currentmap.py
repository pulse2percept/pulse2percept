# -*electrode2currentmap -*-
"""

Functions for transforming electrode specifications into a current map

"""
import numpy as np
import os
from scipy import interpolate
from scipy.misc import factorial
from scipy.signal import fftconvolve

from pulse2percept import oyster
from pulse2percept.utils import TimeSeries


def micron2deg(micron):
    """
    Transform a distance from microns to degrees

    Based on http://retina.anatomy.upenn.edu/~rob/lance/units_space.html
    """
    deg = micron / 280
    return deg


def deg2micron(deg):
    """
    Transform a distance from degrees to microns

    Based on http://retina.anatomy.upenn.edu/~rob/lance/units_space.html
    """
    microns = 280 * deg
    return microns


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

    y = ((t / tau) ** (n - 1) *
         np.exp(-t / tau) /
         (tau * factorial(n - 1)))

    if flag == 1:
        y = np.concatenate([[0], y])

    return y


class Electrode(object):
    """
    Represent a circular, disc-like electrode.
    """

    def __init__(self, radius, x, y, h, ptype):
        """
        Initialize an electrode object

        Parameters
        ----------
        radius : float
            The radius of the electrode (in microns).
        x : float
            The x coordinate of the electrode (in microns) from the fovea
        y : float
            The y location of the electrode (in microns) from the fovea
        h : float
            The height of the electrode from the retinal surface 
              epiretinal array - distance to the ganglion layer
             subretinal array - distance to the bipolar layer
             
        Estimates of layer thickness based on:
        LoDuca et al. Am J. Ophthalmology 2011
        Thickness Mapping of Retinal Layers by Spectral Domain Optical Coherence Tomography
        Note that this is for normal retinal, so may overestimate thickness.
        Thickness from their paper (averaged across quadrants):
          0-600 um radius (from fovea)
            Layer 1. (Nerve fiber layer) = 4
            Layer 2. (Ganglion cell bodies + inner plexiform) = 56
            Layer 3. (Bipolar bodies, inner nuclear layer) = 23
          600-1550 um radius
            Layer 1. 34
            Layer 2. 87
            Layer 3. 37.5
          1550-3000 um radius
            Layer 1. 45.5
            Layer 2. 58.2
            Layer 3. 30.75
        
        We place our ganglion axon surface on the inner side of the nerve fiber layer
        We place our bipolar surface 1/2 way through the inner nuclear layer
        So for an epiretinal array the bipolar layer is L1+L2+(.5*L3)
                
        """
        self.radius = radius
        self.x = x
        self.y = y
        fovdist=np.sqrt(x**2+y**2)
        
        if ptype=='epiretinal':            
          self.gh = h
          if fovdist<=600:
              self.bh = h + 71.5
          elif fovdist<=1550:
              self.bh = h + 139.75
          elif fovdist>1550:
              self.bh = h + 119.075
        
        elif ptype=='subretinal':            
          if fovdist<=600:
              self.bh = h + 23/2
              self.gh = h + 83
          elif fovdist<=1550:
              self.bh = h + 37.5/2
              self.gh = h + 158.5
          elif fovdist>1550:
              self.bh = h + 30.75/2
              self.gh = h + 141.45
              
    def current_spread(self, xg, yg, layer, alpha=14000, n=1.69):
        """

        The current spread due to a current pulse through an electrode,
        reflecting the fall-off of the current as a function of distance from
        the electrode center. This can be calculated for any layer in the retina
        Based on equation 2 in Nanduri et al [1]_.
        
        Parameters
        ----------
        xg and yg defining the retinal grid
        layers describing which layers of the retina are simulated
            'NFL': nerve fiber layer, ganglion axons
            'INL': inner nuclear layer, containing the bipolars
        alpha : float
            a constant to do with the spatial fall-off.

        n : float
            a constant to do with the spatial fall-off (Default: 1.69, based
            on Ahuja et al. [2]  An In Vitro Model of a Retinal Prosthesis.
            Ashish K. Ahuja, Matthew R. Behrend, Masako Kuroda, Mark S.
            Humayun, and James D. Weiland (2008). IEEE Trans Biomed Eng 55.
    
        """
        r = np.sqrt((xg - self.x) ** 2 + (yg - self.y) ** 2)
        # current values on the retina due to array being above the retinal surface
        if 'NFL' in layer: # nerve fiber layer, ganglion axons          
            h = np.ones(r.shape) * self.gh          
           # actual distance from the electrode edge
            d = ((r - self.radius)**2 + self.gh**2)**.5
        elif 'INL' in layer:  # inner nuclear layer, containing the bipolars
            h = np.ones(r.shape) * self.bh
            d = ((r - self.radius)**2 + self.bh**2)**.5
        cspread = (alpha / (alpha + h ** n))
        cspread[r > self.radius] = (alpha /
                                    (alpha + d[r > self.radius] ** n))

        return cspread


class ElectrodeArray(object):
    """
    Represent a retina and array of electrodes
    """

    def __init__(self, radii, xs, ys, hs, ptype):
        self.electrodes = []
        for r, x, y, h in zip(radii, xs, ys, hs):
            self.electrodes.append(Electrode(r, x, y, h, ptype))

#    def current_spread(self, xg, yg, layers, alpha=14000, n=1.69):
#        
#        BOOM WAH!
#        # this is the part that is broken. Somehow the the info about layers 
#        # needs to be included in the ElectrodeArray object 
#        c = np.zeros((len(self.electrodes), xg.shape[0], xg.shape[1]))
#              
#        for i in range(c.shape[0]):
#            c[i] = self.electrodes[i].current_spread(xg, yg, layers[l],
#                                                     alpha=alpha, n=n, layers[l])
#        return np.sum(c, 0)


def receptive_field(electrode, xg, yg, size):

    # creates a map of the retina for each electrode
    # where it's 1 under the electrode, 0 elsewhere
    rf = np.zeros(xg.shape)
    ind = np.where((xg > electrode.x - (size / 2)) &
                   (xg < electrode.x + (size / 2)) &
                   (yg > electrode.y - (size / 2)) &
                   (yg < electrode.y + (size / 2)))

    rf[ind] = 1
    return rf


def gaussian_receptive_field(electrode, xg, yg, sigma):
    """
    A Gaussian receptive field
    """
    amp = np.exp(-((xg - electrode.x)**2 + (yg - electrode.y) ** 2) /
                 (2 * (sigma ** 2)))
    return amp / np.sum(amp)


def retinalmovie2electrodtimeseries(rf, movie, fps=30):
    """
    calculate the luminance over time for each electrodes receptive field
    """
    rflum = np.zeros(movie.shape[-1])
    for f in range(movie.shape[-1]):
        tmp = rf * movie[:, :, f]
        rflum[f] = np.mean(tmp)

    return rflum


def get_pulse(pulse_dur, tsample, interphase_dur, pulsetype):
    """Returns a single biphasic pulse.

    A single biphasic pulse with duration `pulse_dur` per phase,
    separated by `interphase_dur` is returned.

    Parameters
    ----------
    pulse_dur : float
        Duration of single (positive or negative) pulse phase in seconds.
    tsample : float
        Sampling time step in seconds.
    interphase_dur : float
        Duration of inter-phase interval (between positive and negative
        pulse) in seconds.
    pulsetype : {'cathodicfirst', 'anodicfirst'}
        A cathodic-first pulse has the negative phase first, whereas an
        anodic-first pulse has the positive phase first.

    """
    on = np.ones(round(pulse_dur / tsample))
    gap = np.zeros(round(interphase_dur / tsample))
    off = -1 * on
    if pulsetype == 'cathodicfirst':
        # cathodicfirst has negative current first
        pulse = np.concatenate((off, gap), axis=0)
        pulse = np.concatenate((pulse, on), axis=0)
    elif pulsetype == 'anodicfirst':
        pulse = np.concatenate((on, gap), axis=0)
        pulse = np.concatenate((pulse, off), axis=0)
    else:
        raise ValueError("Acceptable values for `pulsetype` are "
                         "'anodicfirst' or 'cathodicfirst'")
    return pulse


def accumulatingvoltage(ptrain, tau=45.25 / 1000):
    """Model's accumulating voltage on the electrode

    General idea based on "On the Cause and Control of Residual Voltage
    Generated by Electrical Stimulation of Neural Tissue
    Ashwati Krishnan1 and Shawn K. Kelly2, Member, IEEE,2012
    """
    # calculate accumulated charge
    t = np.arange(0, 20 * tau, ptrain.tsample)
    rectified = np.where(ptrain.data > 0, ptrain.data, 0)  # rectify
    ca = ptrain.tsample * np.cumsum(rectified.astype(float), axis=-1)
    g = gamma(1, tau, t)
    chargeaccumulated = (ptrain.tsample * fftconvolve(g, ca))
    zero_pad = np.zeros(rectified.shape[:-1] +
                        (chargeaccumulated.shape[-1] - rectified.shape[-1],))

    ptrain_pad = TimeSeries(
        ptrain.tsample, np.concatenate([ptrain.data, zero_pad], -1))

    ptrain_ca = ptrain_pad.data - chargeaccumulated

    return TimeSeries(ptrain.tsample, ptrain_ca)


class Movie2Pulsetrain(TimeSeries):
    """
    Is used to create pulse-train stimulus based on luminance over time from
    a movie
    """

    def __init__(self, rflum, fps=30.0, amplitude_transform='linear',
                 amp_max=60, freq=20, pulse_dur=.5 / 1000.,
                 interphase_dur=.5 / 1000., tsample=.25 / 1000.,
                 pulsetype='cathodicfirst', stimtype='pulsetrain'):
        """
        Parameters
        ----------
        rflum : 1D array
           Values between 0 and 1
        """
        # set up the individual pulses
        pulse = get_pulse(pulse_dur, tsample, interphase_dur, pulsetype)
        # set up the sequence
        dur = rflum.shape[-1] / fps
        if stimtype == 'pulsetrain':
            interpulsegap = np.zeros(round((1.0 / freq) / tsample) -
                                     len(pulse))
            ppt = []
            for j in range(0, int(np.ceil(dur * freq))):
                ppt = np.concatenate((ppt, interpulsegap), axis=0)
                ppt = np.concatenate((ppt, pulse), axis=0)

        ppt = ppt[0:round(dur / tsample)]
        intfunc = interpolate.interp1d(np.linspace(0, len(rflum), len(rflum)),
                                       rflum)

        amp = intfunc(np.linspace(0, len(rflum), len(ppt)))
        data = amp * ppt * amp_max
        TimeSeries.__init__(self, tsample, data)


class Psycho2Pulsetrain(TimeSeries):
    """
    Is used to generate pulse trains to simulate psychophysical experiments.

    """

    def __init__(self, freq=20, dur=0.5, pulse_dur=0.75/1000,
                 interphase_dur=0.75/1000, delay=0., tsample=0.005/1000,
                 current_amplitude=20, pulsetype='cathodicfirst',
                 pulseorder='gapfirst'):
        """

        Parameters
        ----------
        freq : float
            Frequency of the pulse envelope in Hz.

        dur : float
            Stimulus duration in seconds.

        pulse_dur : float
            Single-pulse duration in seconds.

        interphase_duration : float
            Single-pulse interphase duration (the time between the positive
            and negative phase) in seconds.

        delay : float
            Delay until stimulus on-set in seconds.

        tsample : float
            Sampling interval in seconds.

        current_amplitude : float
            Max amplitude of the pulse train in micro-amps.

        pulsetype : string
            Pulse type {"cathodicfirst" | "anodicfirst"}, where
            'cathodicfirst' has the negative phase first.

        pulseorder : string
            Pulse order {"gapfirst" | "pulsefirst"}, where
            'pulsefirst' has the pulse first, followed by the gap.
        """
        # envelope size (single pulse + gap) given by `freq`
        envelope_size = int(np.round((1 / freq) / tsample))

        # delay given by `delay`
        delay_size = int(np.round(delay / tsample))
        delay = np.zeros(delay_size)

        # single pulse given by `pulse_dur`
        pulse = current_amplitude * get_pulse(pulse_dur, tsample,
                                              pulse_dur,
                                              pulsetype)
        pulse_size = pulse.size

        # then gap is used to fill up what's left
        gap_size = envelope_size - delay_size - pulse_size
        gap = np.zeros(gap_size)

        pulse_train = []
        for j in range(int(np.round(dur * freq))):
            if pulseorder == 'pulsefirst':
                pulse_train = np.concatenate((pulse_train, delay, pulse,
                                              gap), axis=0)
            elif pulseorder == 'gapfirst':
                pulse_train = np.concatenate((pulse_train, delay, gap,
                                              pulse), axis=0)
            else:
                raise ValueError("Acceptable values for `pulseorder` are "
                                 "'pulsefirst' or 'gapfirst'")

        TimeSeries.__init__(self, tsample, pulse_train)


class Retina(object):
    """
    Represent the retinal coordinate frame
    """

    def __init__(self, xlo=-1000, xhi=1000, ylo=-1000, yhi=1000,
                 sampling=25, axon_map=None, axon_lambda=2):
        """
        Initialize a retina

        Parameters
        ----------
        xlo, xhi : int
           Extent of the retinal coverage (microns) in horizontal dimension
        ylo, yhi :
           Extent of the retinal coverage (microns) in vertical dimension
        sampling : int
            Microns per grid cell
        axon_map : str
           Full path to a file that encodes the axon map (see :mod:`oyster`)
        axon_lambda : float
            Constant that determines fall-off with axonal distance
        """
        self.gridx, self.gridy = np.meshgrid(np.arange(xlo, xhi,
                                                       sampling),
                                             np.arange(ylo, yhi,
                                                       sampling),
                                             indexing='xy')

        if axon_map is not None and os.path.exists(axon_map):
            axon_map = np.load(axon_map)
            # Verify that the file was created with a consistent grid:
            axon_id = axon_map['axon_id']
            axon_weight = axon_map['axon_weight']
            xlo_am = axon_map['xlo']
            xhi_am = axon_map['xhi']
            ylo_am = axon_map['ylo']
            yhi_am = axon_map['yhi']
            sampling_am = axon_map['sampling']
            axon_lambda_am = axon_map['axon_lambda']
            assert xlo == xlo_am
            assert xhi == xhi_am
            assert ylo == ylo_am
            assert yhi == yhi_am
            assert sampling_am == sampling
            assert axon_lambda_am == axon_lambda

        else:
            if axon_map is None:
                axon_map = 'axons.npz'
            print("Can't find file %s, generating" % axon_map)
            axon_id, axon_weight = oyster.makeAxonMap(micron2deg(self.gridx),
                                                      micron2deg(self.gridy),
                                                      axon_lambda=axon_lambda)
            # Save the variables, together with metadata about the grid:
            fname = axon_map
            np.savez(fname,
                     axon_id=axon_id,
                     axon_weight=axon_weight,
                     xlo=[xlo],
                     xhi=[xhi],
                     ylo=[ylo],
                     yhi=[yhi],
                     sampling=[sampling],
                     axon_lambda=[axon_lambda])

        self.sampling = sampling
        self.axon_id = axon_id
        self.axon_weight = axon_weight

    def cm2ecm(self, current_spread, integrationtype):
        """

        Converts a current spread map to an 'effective' current spread map, by
        passing the map through a mapping of axon streaks.

        Parameters
        ----------
        current_spread : the 2D spread map in retinal space

        integrationtype : either 'dotproduct' or 'maxrule'

        Returns
        -------
        ecm: effective current spread, a time-series of the same size as the
        current map, where each pixel is the dot product of the pixel values in
        ecm along the pixels in the list in axon_map, weighted by the weights
        axon map.
        """
        ecs = np.zeros(current_spread.shape)
        for id in range(0, len(current_spread.flat)):
            if integrationtype is 'dotproduct':
                ecs.flat[id] = np.dot(current_spread.flat[self.axon_id[id]],
                                      self.axon_weight[id])
            elif integrationtype is 'maxrule':
                cs = current_spread.flat[self.axon_id[id]]
                ecs.flat[id] = np.max(np.multiply(cs, self.axon_weight[id]))
            else:
                print('pulse not defined')

        ecs = ecs / ecs.max()

        # this normalization is based on unit current on the retina producing
        # a max response of 1 based on axonal integration.
        # means that response magnitudes don't change as you increase the
        # length of axonal integration or sampling of the retina
        # Doesn't affect normalization over time, or responses as a function
        # of the anount of current,

        return ecs

    def electrode_ecs(self, electrode_array, layers, alpha=14000, n=1.69,
                      integrationtype='maxrule'):
        """
        Gather current spread and effective current spread for each electrode
        within both the bipolar and the ganglion cell layer

        Parameters
        ----------
        electrode_array : ElectrodeArray class instance.
        alpha : float
            Current spread parameter
        n : float
            Current spread parameter

        Returns
        -------
        g_layer, b_layer : two arrays containing the the effective current
            spread within the ganglion cell layer and the bipolar layer 
            (currently equivalent to the calculated curent spread)
            for each electrode in the array respectively.

        See also
        --------
        Electrode.current_spread
        """
        cs = np.zeros((self.gridx.shape[0], self.gridx.shape[1],
            len(layers),    
            len(electrode_array.electrodes)))
        ecs = np.zeros((self.gridx.shape[0], self.gridx.shape[1],
            len(layers),
            len(electrode_array.electrodes)))
                        
        for l in range(0,len(layers)):                
            for i, e in enumerate(electrode_array.electrodes):
                cs[..., l, i] = e.current_spread(self.gridx, self.gridy, layers[l],
                                          alpha=alpha, n=n)
                if layers[l]=='NFL':
                   ecs[..., l, i] = self.cm2ecm(cs[..., l,i], integrationtype)
                else:
                   ecs[...,l,i]=cs[..., l, i] 

        return ecs, cs 


def ecm(ecs_list, stim_data, tsample):
    """
    effective current map from the electrodes in one spatial location
    ([x, y] index) and the stimuli through these electrodes.

    Parameters
    ----------
    ecs_list: nlayer x npixels (over threshold) arrays

    stimuli : list of TimeSeries objects with the electrode stimulation
        pulse trains.

    Returns
    -------
    A TimeSeries object with the effective current for this stimulus
    """
    ecm = np.sum(ecs_list[:, None] * stim_data, 0)
    return TimeSeries(tsample, ecm)
