# -*effectivecurrent2brightness -*-
"""

Functions for transforming electrode specifications into a current map

"""
import numpy as np
import oyster
import os
from scipy import interpolate
from utils import TimeSeries


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


class Electrode(object):
    """
    Represent a circular, disc-like electrode.
    """
    def __init__(self, radius, x, y):
        """
        Initialize an electrode object

        Parameters
        ----------
        radius : float
            The radius of the electrode (in microns).
        x : float
            The x coordinate of the electrode (in microns).
        y : float
            The y location of the electrode (in microns).
        """
        self.radius = radius
        self.x = x
        self.y = y

    def current_spread(self, xg, yg, alpha=14000, n=1.69):
        """

        The current spread due to a current pulse through an electrode,
        reflecting the fall-off of the current as a function of distance from
        the electrode center. This is equation 2 in Nanduri et al [1]_.

        Parameters
        ----------

        alpha : a constant to do with the spatial fall-off.

        n : a constant to do with the spatial fall-off (Default: 1.69, based on
        Ahuja et al. [2]_)

        .. [1]

        .. [2] An In Vitro Model of a Retinal Prosthesis. Ashish K. Ahuja,
        Matthew R. Behrend, Masako Kuroda, Mark S. Humayun, and
        James D. Weiland (2008). IEEE Trans Biomed Eng 55.
        """
        r = np.sqrt((xg - self.x) ** 2 + (yg - self.y) ** 2)
        cspread = np.ones(r.shape)
        cspread[r > self.radius] = (alpha / (alpha + (r[r > self.radius] -
                                             self.radius) ** n))
        return cspread


class ElectrodeArray(object):
    """
    Represent a retina and array of electrodes
    """
    def __init__(self, radii, xs, ys):
        self.electrodes = []
        for r, x, y in zip(radii, xs, ys):
            self.electrodes.append(Electrode(r, x, y))

    def current_spread(self, xg, yg, alpha=14000, n=1.69):
        c = np.zeros((len(self.electrodes), xg.shape[0], xg.shape[1]))
        for i in range(c.shape[0]):
            c[i] = self.electrodes[i].current_spread(xg, yg,
                                                     alpha=alpha, n=n)
        return np.sum(c, 0)


def receptive_field(electrode, xg, yg, size):
    """
    # TODO currently this is in units of the grid, needs to be converted to
    microns
    """
    rf = np.zeros(xg.shape)
    ind = np.where((xg > electrode.x-(size/2)) &
                   (xg < electrode.x+(size/2)) &
                   (yg > electrode.y-(size/2)) &
                   (yg < electrode.y+(size/2)))

    rf[ind] = 1
    return rf


def gaussian_receptive_field(electrode, xg, yg, sigma):
    """ 
    A Gaussian receptive field
    """
    amp = np.exp(-((xg - electrode.x)**2 + (yg - electrode.y)**2) / (2 * (sigma ** 2)))
    return amp / np.sum(amp)


def retinalmovie2electrodtimeseries(rf, movie, fps=30):
    """

    """
    rflum = np.zeros(movie.shape[-1])
    for f in range(movie.shape[-1]):
        tmp = rf * movie[:, :, f]
        rflum[f] = np.mean(tmp)

    return rflum

def get_pulse(pulse_dur, tsample, interphase_dur, pulsetype):
    on = np.ones(round(pulse_dur / tsample))
    gap = np.zeros(round(interphase_dur / tsample))
    off = -1 * on
    if pulsetype == 'cathodicfirst':
        pulse = np.concatenate((on, gap), axis=0)
        pulse = np.concatenate((pulse, off), axis=0)

    elif pulsetype == 'anodicfirst':
        pulse = np.concatenate((off, gap), axis=0)
        pulse = np.concatenate((pulse, on), axis=0)
    else:
        print('pulse not defined')
    return pulse


class Movie2Pulsetrain(TimeSeries):
    """
    Is used to create pulse-train stimulus based on luminance over time from
    a movie
    """
    def __init__(self, rflum, fps=30.0, amplitude_transform='linear',
                 amp_max=60, freq=20, pulse_dur=.5/1000.,
                 interphase_dur=.5/1000., tsample=.25/1000.,
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
            interpulsegap = np.zeros(round((1 / freq) / tsample) - len(pulse))
            ppt = []
            for j in range(0, int(np.ceil(dur * freq))):
                ppt = np.concatenate((ppt, interpulsegap), axis=0)
                ppt = np.concatenate((ppt, pulse), axis=0)

        ppt = ppt[0:round(dur/tsample)]
        intfunc = interpolate.interp1d(np.linspace(0, len(rflum), len(rflum)),
                                       rflum)

        amp = intfunc(np.linspace(0, len(rflum), len(ppt)))
        data = amp * ppt * amp_max
        TimeSeries.__init__(self, tsample, data)


class Psycho2Pulsetrain(TimeSeries):
    """
    Is used to generate pulse trains to simulate psychophysical experiments.

    """
    def __init__(self, freq=20, dur=0.5, pulse_dur=.075/1000.,
                 interphase_dur=.075/1000., delay=0., tsample=.005/1000.,
                 current_amplitude=20, pulsetype='cathodicfirst',
                 stimtype='pulsetrain'):
        """

        Parameters
        ----------
        freq :
        dur : float
            Duration in seconds

        pulse_dur : float
            Pulse duration in seconds

        interphase_duration : float
            In seconds

        delay : float

        tsample : float
            Sampling interval in seconds

        current_amplitude : float
            In XXX units?

        pulsetype : string
            {"cathodicfirst" | "anodicfirst"}

        stimtype : string
            {"pulsetrain" | XXX other options?}
        """
        # set up the individual pulses
        pulse = get_pulse(pulse_dur, tsample, interphase_dur, pulsetype)

        # set up the sequence
        if stimtype == 'pulsetrain':
            interpulsegap = np.zeros(round((1/freq) / tsample) - len(pulse))
            ppt = []
            for j in range(0, int(np.ceil(dur * freq))):
                ppt = np.concatenate((ppt, interpulsegap), axis=0)
                ppt = np.concatenate((ppt, pulse), axis=0)

        if delay > 0:
                ppt = np.concatenate((np.zeros(round(delay / tsample)), ppt),
                                     axis=0)

        ppt = ppt[0:round(dur/tsample)]
        data = (current_amplitude * ppt)
        TimeSeries.__init__(self, tsample, data)


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

        self.axon_id = axon_id
        self.axon_weight = axon_weight

    def cm2ecm(self, current_spread):
        """

        Converts a current spread map to an 'effective' current spread map, by
        passing the map through a mapping of axon streaks.

        Parameters
        ----------
        current_spread : the 2D spread map in retinal space

        Returns
        -------
        ecm: effective current spread, a time-series of the same size as the
        current map, where each pixel is the dot product of the pixel values in
        ecm along the pixels in the list in axon_map, weighted by the weights
        axon map.
        """
        ecs = np.zeros(current_spread.shape)
        for id in range(0, len(current_spread.flat)):
            ecs.flat[id] = np.dot(current_spread.flat[self.axon_id[id]],
                                  self.axon_weight[id])

        ecs = ecs / ecs.max()

        return ecs

    def electrode_ecs(self, electrode_array, alpha=14000, n=1.69):
        """
        Gather current spread and effective current spread for each electrode

        Parameters
        ----------
        electrode_array : ElectrodeArray class instance.
        alpha : float
            Current spread parameter
        n : float
            Current spread parameter

        Returns
        -------
        ecs_list, cs_list : two lists containing the the effective current
            spread and current spread for each electrode in the array
            respectively.

        See also
        --------
        Electrode.current_spread
        """
        ecs = np.zeros((self.gridx.shape[0], self.gridx.shape[1],
                       len(electrode_array.electrodes)))

        cs =  np.zeros((self.gridx.shape[0], self.gridx.shape[1],
                       len(electrode_array.electrodes)))

        for i, e in enumerate(electrode_array.electrodes):
            cs[..., i] = e.current_spread(self.gridx, self.gridy,
                                          alpha=alpha, n=n)
            ecs[..., i] = self.cm2ecm(cs[..., i])

        return ecs, cs


def ecm(ecs_vector, stim_data, tsample):
    """
    effective current map from the electrodes in one spatial location
    ([x, y] index) and the stimuli through these electrodes.

    Parameters
    ----------
    ecs_vector : 1D arrays

    stimuli : list of TimeSeries objects with the electrode stimulation
        pulse trains.

    Returns
    -------
    A TimeSeries object with the effective current for this stimulus
    """
    ecm = np.sum(ecs_vector[:, None] * stim_data, 0)
    return TimeSeries(tsample, ecm)
