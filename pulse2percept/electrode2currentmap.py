# -*electrode2currentmap -*-
"""

Functions for transforming electrode specifications into a current map

"""
import numpy as np
from os.path import exists
from scipy import interpolate
from scipy.misc import factorial

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

    y = y / (np.sum(y) * (t[1] - t[0]))  # normalizes so area doesn't change

    return y


class Electrode(object):
    """
    Represent a circular, disc-like electrode.
    """

    def __init__(self, etype, radius, x_center, y_center, height, name=None):
        """Create an electrode on the retina

        This function creates an electrode of type `etype` and places it
        on the retina at location (`xs`, `ys`) in microns. The electrode has
        radius `radius` (microns) and sits a distance `height` away from the
        retinal surface.
        The coordinate system is anchored around the fovea at (0, 0).

        Parameters
        ----------
        etype : str
            Electrode type, {'epiretinal', 'subretinal'}
        radius : float
            The radius of the electrode (in microns).
        x_center : float
            The x coordinate of the electrode center (in microns) from the
            fovea.
        y_center : float
            The y location of the electrode (in microns) from the fovea
        height : float
            The height of the electrode from the retinal surface
              epiretinal array - distance to the ganglion layer
             subretinal array - distance to the bipolar layer
        name : string
            Electrode name

        Estimates of layer thickness based on:
        LoDuca et al. Am J. Ophthalmology 2011
        Thickness Mapping of Retinal Layers by Spectral Domain Optical
        Coherence Tomography
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

        We place our ganglion axon surface on the inner side of the nerve fiber
        layer.
        We place our bipolar surface 1/2 way through the inner nuclear layer.
        So for an epiretinal array the bipolar layer is L1+L2+(.5*L3).

        """
        assert radius >= 0
        assert height >= 0

        self.etype = etype
        self.radius = radius
        self.x_center = x_center
        self.y_center = y_center
        self.name = name

        fovdist = np.sqrt(x_center ** 2 + y_center ** 2)

        if etype == 'epiretinal':
            self.h_nfl = height
            if fovdist <= 600:
                self.h_inl = height + 71.5
            elif fovdist <= 1550:
                self.h_inl = height + 139.75
            elif fovdist > 1550:
                self.h_inl = height + 119.075
        elif etype == 'subretinal':
            if fovdist <= 600:
                self.h_inl = height + 23 / 2
                self.h_nfl = height + 83
            elif fovdist <= 1550:
                self.h_inl = height + 37.5 / 2
                self.h_nfl = height + 158.5
            elif fovdist > 1550:
                self.h_inl = height + 30.75 / 2
                self.h_nfl = height + 141.45
        else:
            e_s = "Acceptable values for `ptype` are: 'epiretinal', "
            e_s += "'subretinal'."
            raise ValueError(e_s)

    def current_spread(self, xg, yg, layer, alpha=14000, n=1.69):
        """

        The current spread due to a current pulse through an electrode,
        reflecting the fall-off of the current as a function of distance from
        the electrode center. This can be calculated for any layer in the
        retina.
        Based on equation 2 in Nanduri et al [1].

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
        r = np.sqrt((xg - self.x_center) ** 2 + (yg - self.y_center) ** 2)
        # current values on the retina due to array being above the retinal
        # surface
        if 'NFL' in layer:  # nerve fiber layer, ganglion axons
            h = np.ones(r.shape) * self.h_nfl
            # actual distance from the electrode edge
            d = ((r - self.radius)**2 + self.h_nfl**2)**.5
        elif 'INL' in layer:  # inner nuclear layer, containing the bipolars
            h = np.ones(r.shape) * self.h_inl
            d = ((r - self.radius)**2 + self.h_inl**2)**.5
        else:
            s = "Layer %s not found. Acceptable values for `layer` are " \
                "'NFL' or 'INL'." % layer
            raise ValueError(s)
        cspread = (alpha / (alpha + h ** n))
        cspread[r > self.radius] = (alpha /
                                    (alpha + d[r > self.radius] ** n))

        return cspread


class ElectrodeArray(object):

    def __init__(self, etype, radii, xs, ys, hs, names=None):
        """Create an ElectrodeArray on the retina

        This function creates an electrode array of type `etype` and places it
        on the retina. Lists should specify, for each electrode, its size
        (`radii`), location on the retina (`xs` and `ys`), distance to the
        retina (height, `hs`), and a string identifier (`names`, optional).

        Array location should be given in microns, where the fovea is located
        at (0, 0).

        Single electrodes in the array can be addressed by index (integer)
        or name.

        Parameters
        ----------
        etype : string
            Electrode type, {'epiretinal', 'subretinal'}
        radii : array_like
            List of electrode radii.
        xs : array_like
            List of x-coordinates for the center of the electrodes (microns).
        ys : array_like
            List of y-coordinates for the center of the electrodes (microns).
        hs : array_like
            List of electrode heights (distance from the retinal surface)
        names : array_like, optional
            List of names (string identifiers) for each eletrode.
            Default: None.

        Examples
        --------
        A single epiretinal electrode called 'A1', with radius 100um, sitting
        at retinal location (0, 0), 10um away from the retina:
        >>> from pulse2percept import electrode2currentmap as e2cm
        >>> implant1 = e2cm.ElectrodeArray('epiretinal', 100, 0, 0, 10, 'A1')

        An array with two electrodes of size 100um, one sitting at
        (-100, -100), the other sitting at (0, 0), with 0 distance from the
        retina, of type 'subretinal':
        >>> implant2 = e2cm.ElectrodeArray('subretinal', [100, 100], [-100, 0],
        ...                                [-100, 0], [0, 0])

        Get access to the electrode with name 'A1' in the array:
        >>> my_electrode = implant1['A1']

        Get access to the second electrode in the array:
        >>> my_electrode = implant2[1]



        """
        # Make it so the constructor can accept either floats, lists, or
        # numpy arrays, and `zip` works regardless.
        radii = np.array([radii], dtype=np.float32).flatten()
        xs = np.array([xs], dtype=np.float32).flatten()
        ys = np.array([ys], dtype=np.float32).flatten()
        hs = np.array([hs], dtype=np.float32).flatten()
        names = np.array([names], dtype=np.str).flatten()
        assert radii.size == xs.size == ys.size == hs.size

        if names.size != radii.size:
            # If not every electrode has a name, replace with None's
            names = np.array([None] * radii.size)

        self.etype = etype
        self.num_electrodes = names.size
        self.names = names
        self.electrodes = []
        for r, x, y, h, n in zip(radii, xs, ys, hs, names):
            self.electrodes.append(Electrode(etype, r, x, y, h, n))

    def __iter__(self):
        return iter(self.electrodes)

    def __getitem__(self, item):
        """Return the electrode specified by `item`

        Parameters
        ----------
        item : int|string
            If `item` is an integer, returns the `item`-th electrode in the
            array. If `item` is a string, returns the electrode with string
            identifier `item`.
        """
        try:
            # Is `item` an integer?
            return self.electrodes[item]
        except:
            # If `item` is a valid string identifier, return valid index.
            # Else return None
            try:
                return self.electrodes[self.get_index(item)]
            except:
                return None

    def get_index(self, name):
        """Returns the index of an electrode called `name`

        This function searches the electrode array for an electrode with
        string identifier `name`. If found, the index of that electrode is
        returned, else None.

        Parameters
        ----------
        name : str
            An electrode name (string identifier).

        Returns
        -------
        A valid electrode index or None.
        """
        try:
            # Is `name` a valid electrode name?
            idx = self.names.tolist().index(name)
            return idx
        except:
            # Else: `name` could not be found.
            return None


class ArgusI(ElectrodeArray):

    def __init__(self, x_center=0, y_center=0, h=0, rot=0 * np.pi / 180):
        """Create an ArgusI array on the retina

        This function creates an ArgusI array and places it on the retina
        such that the center of the array is located at
        [`x_center`, `y_center`] (microns) and the array is rotated by
        rotation angle `rot` (radians).

        The array is oriented in the visual field as shown in Fig. 1 of
        Horsager et al. (2009); that is, if placed in (0,0), the top two
        rows will lie in the lower retina (upper visual field):
        y       A1 B1 C1 D1                     260 520 260 520
        ^       A2 B2 C2 D2   where electrode   520 260 520 260
        |       A3 B3 C3 D3   diameters are:    260 520 260 520
        -->x    A4 B4 C4 D4                     520 260 520 260

        Electrode order is: A1, B1, C1, D1, A2, B2, ..., D4.
        An electrode can be addressed by index (integer) or name.

        Parameters
        ----------
        x_center : float
            x coordinate of the array center (um)
        y_center : float
            y coordinate of the array center (um)
        h : float || array_like
            Distance of the array to the retinal surface (um). Either a list
            with 16 entries or a scalar.
        rot : float
            Rotation angle of the array (rad). Positive values denote
            counter-clock-wise rotations.

        Examples
        --------
        Create an ArgusI array centered on the fovea, at 100um distance from
        the retina:
        >>> from pulse2percept import electrode2currentmap as e2cm
        >>> argus = e2cm.ArgusI(x_center=0, y_center=0, h=100, rot=0)

        Get access to electrode 'B1':
        >>> my_electrode = argus['B1']

        """
        # Alternating electrode sizes, arranged in checkerboard pattern
        r_arr = np.array([260, 520, 260, 520]) / 2.0
        r_arr = np.concatenate((r_arr, r_arr[::-1], r_arr, r_arr[::-1]),
                               axis=0)

        # Standard Argus I names: A1, B1, C1, D1, A1, B2, ..., D4
        # Shortcut: Use `chr` to go from int to char
        names = [chr(i) + str(j) for j in range(1, 5) for i in range(65, 69)]

        if isinstance(h, list):
            h_arr = np.array(h).flatten()
            if h_arr.size != len(r_arr):
                e_s = "If `h` is a list, it must have 16 entries."
                raise ValueError(e_s)
        else:
            # All electrodes have the same height
            h_arr = np.ones_like(r_arr) * h

        # Equally spaced electrodes
        e_spacing = 800  # um
        x_arr = np.arange(0, 4) * e_spacing - 1.5 * e_spacing
        x_arr, y_arr = np.meshgrid(x_arr, x_arr, sparse=False)

        # Rotation matrix
        R = np.array([np.cos(rot), np.sin(rot),
                      -np.sin(rot), np.cos(rot)]).reshape((2, 2))

        # Rotate the array
        xy = np.vstack((x_arr.flatten(), y_arr.flatten()))
        xy = np.matmul(R, xy)
        x_arr = xy[0, :]
        y_arr = xy[1, :]

        # Apply offset
        x_arr += x_center
        y_arr += y_center

        self.etype = 'epiretinal'
        self.num_electrodes = len(names)
        self.names = np.array(names, dtype=np.str)
        self.electrodes = []
        for r, x, y, h, n in zip(r_arr, x_arr, y_arr, h_arr, names):
            self.electrodes.append(Electrode(self.etype, r, x, y, h, n))


class ArgusII(ElectrodeArray):

    def __init__(self, x_center=0, y_center=0, h=0, rot=0 * np.pi / 180):
        """Create an ArgusII array on the retina

        This function creates an ArgusII array and places it on the retina
        such that the center of the array is located at
        [`x_center`, `y_center`] (microns) and the array is rotated by
        rotation angle `rot` (radians).

        The array is oriented upright in the visual field, such that an
        array with center (0,0) has the top three rows lie in the lower
        retina (upper visual field), as shown below:
                A1 A2 A3 A4 A5 A6 A7 A8 A9 A10
        y       B1 B2 B3 B4 B5 B6 B7 B8 B9 B10
        ^       C1 C2 C3 C4 C5 C6 C7 C8 C9 C10
        |       D1 D2 D3 D4 D5 D6 D7 D8 D9 D10
        -->x    E1 E2 E3 E4 E5 E6 E7 E8 E9 E10
                F1 F2 F3 F4 F5 F6 F7 F8 F9 F10

        Electrode order is: A1, A2, ..., A10, B1, B2, ..., F10.
        An electrode can be addressed by index (integer) or name.

        Parameters
        ----------
        x_center : float
            x coordinate of the array center (um)
        y_center : float
            y coordinate of the array center (um)
        h : float || array_like
            Distance of the array to the retinal surface (um). Either a list
            with 60 entries or a scalar.
        rot : float
            Rotation angle of the array (rad). Positive values denote
            counter-clock-wise rotations.

        Examples
        --------
        Create an ArgusII array centered on the fovea, at 100um distance from
        the retina:
        >>> from pulse2percept import electrode2currentmap as e2cm
        >>> argus = e2cm.ArgusII(x_center=0, y_center=0, h=100, rot=0)

        Get access to electrode 'E7':
        >>> my_electrode = argus['E7']

        """
        # Electrodes are 200um in diameter
        r_arr = np.ones(60) * 100.0

        # Standard ArgusII names: A1, A2, ..., A10, B1, ..., F10
        names = [chr(i) + str(j) for i in range(65, 71) for j in range(1, 11)]

        if isinstance(h, list):
            h_arr = np.array(h).flatten()
            if h_arr.size != len(r_arr):
                e_s = "If `h` is a list, it must have 60 entries."
                raise ValueError(e_s)
        else:
            # All electrodes have the same height
            h_arr = np.ones_like(r_arr) * h

        # Equally spaced electrodes
        e_spacing = 525  # um
        x_arr = np.arange(10) * e_spacing - 4.5 * e_spacing
        y_arr = np.arange(6) * e_spacing - 2.5 * e_spacing
        x_arr, y_arr = np.meshgrid(x_arr, y_arr, sparse=False)

        # Rotation matrix
        R = np.array([np.cos(rot), np.sin(rot),
                      -np.sin(rot), np.cos(rot)]).reshape((2, 2))

        # Rotate the array
        xy = np.vstack((x_arr.flatten(), y_arr.flatten()))
        xy = np.matmul(R, xy)
        x_arr = xy[0, :]
        y_arr = xy[1, :]

        # Apply offset
        x_arr += x_center
        y_arr += y_center

        self.etype = 'epiretinal'
        self.num_electrodes = len(names)
        self.names = np.array(names, dtype=np.str)
        self.electrodes = []
        for r, x, y, h, n in zip(r_arr, x_arr, y_arr, h_arr, names):
            self.electrodes.append(Electrode(self.etype, r, x, y, h, n))


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
    on = np.ones(int(round(pulse_dur / tsample)))
    gap = np.zeros(int(round(interphase_dur / tsample)))
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


class Movie2Pulsetrain(TimeSeries):
    """
    Is used to create pulse-train stimulus based on luminance over time from
    a movie
    """

    def __init__(self, rflum, tsample, fps=30.0, amplitude_transform='linear',
                 amp_max=60, freq=20, pulse_dur=.5 / 1000.,
                 interphase_dur=.5 / 1000.,
                 pulsetype='cathodicfirst', stimtype='pulsetrain'):
        """
        Parameters
        ----------
        rflum : 1D array
           Values between 0 and 1
        tsample : suggest TemporalModel.tsample
        """
        # set up the individual pulses
        pulse = get_pulse(pulse_dur, tsample, interphase_dur, pulsetype)
        # set up the sequence
        dur = rflum.shape[-1] / fps
        if stimtype == 'pulsetrain':
            interpulsegap = np.zeros(int(round((1.0 / freq) / tsample)) -
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

    def __init__(self, tsample, freq=20, amp=20, dur=0.5, delay=0,
                 pulse_dur=0.45 / 1000, interphase_dur=0.45 / 1000,
                 pulsetype='cathodicfirst',
                 pulseorder='pulsefirst'):
        """

        tsample : float
            Sampling interval in seconds parameters, use TemporalModel.tsample.
        ----------
        optional parameters
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


        amp : float
            Max amplitude of the pulse train in micro-amps.

        pulsetype : string
            Pulse type {"cathodicfirst" | "anodicfirst"}, where
            'cathodicfirst' has the negative phase first.

        pulseorder : string
            Pulse order {"gapfirst" | "pulsefirst"}, where
            'pulsefirst' has the pulse first, followed by the gap.
        """
        # Stimulus size given by `dur`
        stim_size = int(np.round(1.0 * dur / tsample))

        if freq == 0 or amp == 0:
            TimeSeries.__init__(self, tsample, np.zeros(stim_size))
            return

        # Envelope size (single pulse + gap) given by `freq`
        envelope_size = int(np.round(1.0 / float(freq) / tsample))

        # Delay given by `delay`
        delay_size = int(np.round(1.0 * delay / tsample))

        if delay_size < 0:
            raise ValueError("Delay must fit within 1/freq interval.")
        delay = np.zeros(delay_size)

        # Single pulse given by `pulse_dur`
        pulse = amp * get_pulse(pulse_dur, tsample,
                                interphase_dur,
                                pulsetype)
        pulse_size = pulse.size
        if pulse_size < 0:
            raise ValueError("Single pulse must fit within 1/freq interval.")

        # Then gap is used to fill up what's left
        gap_size = envelope_size - (delay_size + pulse_size)
        if gap_size < 0:
            raise ValueError("Pulse and delay must fit within 1/freq "
                             "interval.")
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

        # If `freq` is not a nice number, the resulting pulse train might not
        # have the desired length
        if pulse_train.size < stim_size:
            fill_size = stim_size - pulse_train.shape[-1]
            pulse_train = np.concatenate((pulse_train, np.zeros(fill_size)),
                                         axis=0)

        # Trim to correct length (takes care of too long arrays, too)
        pulse_train = pulse_train[:stim_size]

        TimeSeries.__init__(self, tsample, pulse_train)


class Retina(object):
    """
    Represent the retinal coordinate frame
    """

    def __init__(self, xlo=-1000, xhi=1000, ylo=-1000, yhi=1000,
                 sampling=25, axon_lambda=2.0, rot=0 * np.pi / 180,
                 loadpath='../', save_data=True, verbose=True):
        """Generates a retina

        This function generates the coordinate system for the retina
        and an axon map. As this can take a while, the function will
        first look for an already existing file in the directory `loadpath`
        that was automatically created from an earlier call to this function.

        Parameters
        ----------
        xlo, xhi : float
           Extent of the retinal coverage (microns) in horizontal dimension.
           Default: xlo=-1000, xhi=1000.
        ylo, yhi : float
           Extent of the retinal coverage (microns) in vertical dimension.
           Default: ylo=-1000, ylo=1000.
        sampling : float
            Microns per grid cell. Default: 25 microns.
        axon_lambda : float
            Constant that determines fall-off with axonal distance.
            Default: 2.
        rot : float
            Rotation angle (rad). Default: 0.
        loadpath : str
            Relative path where to look for existing retina file.
            Default: '../'
        save_data : bool
            Flag whether to save the data to a new retina file (True) or not
            (False). The file name is automatically generated from all
            specified input arguments.
            Default: True.
        verbose : bool
             Flag whether to produce output (True) or suppress it (False).
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
            % (loadpath, sampling, axon_lambda, rot / np.pi * 180,
               xhi - xlo, yhi - ylo)

        # Bool whether we need to create a new retina
        need_new_retina = True

        # Check if such a file already exists. If so, load parameters and
        # make sure they are the same as specified above. Else, create new.
        if exists(filename):
            need_new_retina = False
            axon_map = np.load(filename)

            # Verify that the file was created with a consistent grid:
            axon_id = axon_map['axon_id']
            axon_weight = axon_map['axon_weight']
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
            need_new_retina |= xlo != xlo_am
            need_new_retina |= xhi != xhi_am
            need_new_retina |= ylo != ylo_am
            need_new_retina |= yhi != yhi_am
            need_new_retina |= sampling != sampling_am
            need_new_retina |= axon_lambda != axon_lambda_am

            if 'rot' in axon_map:
                rot_am = axon_map['rot']
                need_new_retina |= rot != rot_am
            else:
                # Backwards compatibility for older retina object files that
                # did not have `rot`
                need_new_retina |= rot != 0

        # At this point we know whether we need to generate a new retina:
        if need_new_retina:
            if verbose:
                info_str = "File '%s' doesn't exist " % filename
                info_str += "or has outdated parameter values, generating..."
                print(info_str)
            jan_x, jan_y = oyster.jansonius(rot=rot)
            axon_id, axon_weight = oyster.makeAxonMap(micron2deg(self.gridx),
                                                      micron2deg(self.gridy),
                                                      jan_x, jan_y,
                                                      axon_lambda=axon_lambda)

            # Save the variables, together with metadata about the grid:
            if save_data:
                np.savez(filename,
                         axon_id=axon_id,
                         axon_weight=axon_weight,
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
        self.axon_id = axon_id
        self.axon_weight = axon_weight
        self.jan_x = jan_x
        self.jan_y = jan_y

    def cm2ecm(self, cs):
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

    def electrode_ecs(self, electrode_array, alpha=14000, n=1.69,
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
        ecs : contains n arrays containing the the effective current
            spread within various layers
            for each electrode in the array respectively.

        See also
        --------
        Electrode.current_spread
        """

        cs = np.zeros((self.gridx.shape[0], self.gridx.shape[1],
                       2, len(electrode_array.electrodes)))
        ecs = np.zeros((self.gridx.shape[0], self.gridx.shape[1],
                        2, len(electrode_array.electrodes)))

        for i, e in enumerate(electrode_array.electrodes):
            cs[..., 0, i] = e.current_spread(self.gridx, self.gridy,
                                             layer='INL', alpha=alpha, n=n)
            ecs[..., 0, i] = cs[..., 0, i]
            cs[..., 1, i] = e.current_spread(self.gridx, self.gridy,
                                             layer='NFL', alpha=alpha, n=n)
            ecs[:, :, 1, i] = self.cm2ecm(cs[..., 1, i])

        return ecs, cs


def ecm(ecs_item, ptrain_data, tsample):
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

    ecm = np.sum(ecs_item[:, :, None] * ptrain_data, 1)
    return TimeSeries(tsample, ecm)


def distance2threshold(el_dist):
    """Converts electrode distance (um) to threshold (uA)

    Based on linear regression of data presented in Fig. 7b of
    deBalthasar et al. (2008). Relationship is linear in log-log space.
    """

    slope = 1.5863261730600329
    intercept = -4.2496180725811659

    if el_dist > 0:
        return np.exp(np.log(el_dist) * slope + intercept)
    else:
        return np.exp(intercept)
