"""`ProsthesisSystem`"""
from copy import deepcopy
import matplotlib.pyplot as plt

from .electrodes import Electrode
from .electrode_arrays import ElectrodeArray
from ..stimuli import Stimulus
from ..utils import PrettyPrint


class ProsthesisSystem(PrettyPrint):
    """Visual prosthesis system

    A visual prosthesis combines an electrode array and (optionally) a
    stimulus. This is the base class for prosthesis systems such as
    :py:class:`~pulse2percept.implants.ArgusII` and
    :py:class:`~pulse2percept.implants.AlphaIMS`.

    Parameters
    ----------
    earray : :py:class:`~pulse2percept.implants.ElectrodeArray` or
             :py:class:`~pulse2percept.implants.Electrode`
        The electrode array used to deliver electrical stimuli to the retina.
    stim : :py:class:`~pulse2percept.stimuli.Stimulus` source type
        A valid source type for the :py:class:`~pulse2percept.stimuli.Stimulus`
        object (e.g., scalar, NumPy array, pulse train).
    eye : 'LE' or 'RE'
        A string indicating whether the system is implanted in the left ('LE')
        or right eye ('RE')

    Examples
    --------
    A system in the left eye made from a single
    :py:class:`~pulse2percept.implants.DiskElectrode` with radius
    r=100um sitting at x=200um, y=-50um, z=10um:

    >>> from pulse2percept.implants import DiskElectrode, ProsthesisSystem
    >>> implant = ProsthesisSystem(DiskElectrode(200, -50, 10, 100), eye='LE')

    .. note::

        A stimulus can also be assigned later (see
        :py:attr:`~pulse2percept.implants.ProsthesisSystem.stim`).
    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('_earray', '_stim', '_eye')

    def __init__(self, earray, stim=None, eye='RE'):
        self.earray = earray
        self.stim = stim
        self.eye = eye

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        return {'earray': self.earray, 'stim': self.stim, 'eye': self.eye}

    def check_stim(self, stim):
        """Quality-check the stimulus

        This method is executed every time a new value is assigned to ``stim``.

        No checks are performed by default, but the user can define their own
        checks in implants that inherit from
        :py:class:`~pulse2percept.implants.ProsthesisSystem`.

        Parameters
        ----------
        stim : :py:class:`~pulse2percept.stimuli.Stimulus` source type
            A valid source type for the
            :py:class:`~pulse2percept.stimuli.Stimulus` object (e.g., scalar,
            NumPy array, pulse train).
        """
        pass

    def plot(self, ax=None, annotate=False, upside_down=False, xlim=None,
             ylim=None, pad=None, step=None):
        """Plot

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot, optional, default: None
            A Matplotlib axes object. If None given, a new one will be created.
        annotate : bool, optional, default: False
            Flag whether to label electrodes in the implant.
        upside_down : bool, optional, default: False
            Flag whether to plot the retina upside-down, such that the upper
            half of the plot corresponds to the upper visual field. In general,
            inferior retina == upper visual field (and superior == lower).
        xlim : (xmin, xmax), optional, default: None
            Range of x values to plot. If None, the plot will be centered over
            the implant.
        ylim : (ymin, ymax), optional, default: None
            Range of y values to plot. If None, the plot will be centered over
            the implant.

        Returns
        -------
        ax : ``matplotlib.axes.Axes``
            Returns the axis object of the plot
        """

        if ax is None:
            _, ax = plt.subplots(figsize=(15, 8))

        for name, electrode in self.items():
            electrode.plot(ax=ax)
            if annotate:
                ax.text(electrode.x, electrode.y, name, ha='center',
                        va='center',  color='black', size='large',
                        bbox={'boxstyle': 'square,pad=-0.2', 'ec': 'none',
                              'fc': (1, 1, 1, 0.7)},
                        zorder=3)

        # Determine xlim, ylim: Allow for some padding `pad` and round to the
        # nearest `step`:
        xpos = [el.x for el in self.values()]
        ypos = [el.y for el in self.values()]
        xmin, xmax = np.min(xpos), np.max(xpos)
        ymin, ymax = np.min(ypos), np.max(ypos)
        if pad is None:
            pad = 100 * np.ceil((xmax - xmin) / 1000)
            pad = np.maximum(pad, 100 * np.ceil((xmax - xmin) / 1000))
        if step is None:
            step = 2 * pad
        xlim, ylim = None, None
        if xlim is None:
            xlim = (step * np.floor((xmin - pad) / step),
                    step * np.ceil((ymax + pad) / step))
        if ylim is None:
            ylim = (step * np.floor((ymin - pad) / step),
                    step * np.ceil((ymax + pad) / step))
        ax.set_xlim(xlim)
        ax.set_xticks(np.linspace(*xlim, num=5))
        ax.set_xlabel('x (microns)')
        ax.set_ylim(ylim)
        ax.set_yticks(np.linspace(*ylim, num=5))
        ax.set_ylabel('y (microns)')
        ax.set_aspect('equal')

        # Need to flip y axis to have upper half == upper visual field
        if upside_down:
            ax.invert_yaxis()

        return ax

    @property
    def earray(self):
        """Electrode array

        """
        return self._earray

    @earray.setter
    def earray(self, earray):
        """Electrode array setter (called upon ``self.earray = earray``)"""
        # Assign the electrode array:
        if isinstance(earray, Electrode):
            # For convenience, build an array from a single electrode:
            earray = ElectrodeArray(earray)
        if not isinstance(earray, ElectrodeArray):
            raise TypeError("'earray' must be an ElectrodeArray object, not "
                            "%s." % type(earray))
        self._earray = earray

    @property
    def stim(self):
        """Stimulus

        A stimulus can be created from many source types, such as scalars,
        NumPy arrays, and dictionaries (see
        :py:class:`~pulse2percept.stimuli.Stimulus` for a complete list).

        A stimulus can be assigned either in the
        :py:class:`~pulse2percept.implants.ProsthesisSystem` constructor
        or later by assigning a value to `stim`.

        .. note::
           Unless when using dictionary notation, the number of stimuli must
           equal the number of electrodes in ``earray``.

        Examples
        --------
        Send a biphasic pulse (30uA, 0.45ms phase duration) to an implant made
        from a single :py:class:`~pulse2percept.implants.DiskElectrode`:

        >>> from pulse2percept.implants import DiskElectrode, ProsthesisSystem
        >>> from pulse2percept.stimuli import BiphasicPulse
        >>> implant = ProsthesisSystem(DiskElectrode(0, 0, 0, 100))
        >>> implant.stim = BiphasicPulse(30, 0.45)

        Stimulate Electrode B7 in Argus II with 13 uA:

        >>> from pulse2percept.implants import ArgusII
        >>> implant = ArgusII(stim={'B7': 13})

        """
        return self._stim

    @stim.setter
    def stim(self, data):
        """Stimulus setter (called upon ``self.stim = data``)"""
        if data is None:
            self._stim = None
        else:
            if isinstance(data, Stimulus):
                # Already a stimulus object:
                stim = Stimulus(data, extrapolate=True)
            elif isinstance(data, dict):
                # Electrode names already provided by keys:
                stim = Stimulus(data, extrapolate=True)
            else:
                # Use electrode names as stimulus coordinates:
                stim = Stimulus(data, electrodes=list(self.earray.keys()),
                                extrapolate=True)

            # Make sure all electrode names are valid:
            for electrode in stim.electrodes:
                # Invalid index will return None:
                if not self.earray[electrode]:
                    raise ValueError("Electrode '%s' not found in "
                                     "implant." % electrode)
            # Perform safety checks, etc.:
            self.check_stim(stim)
            # Store stimulus:
            self._stim = deepcopy(stim)

    @property
    def eye(self):
        """Implanted eye

        A :py:class:`~pulse2percept.implants.ProsthesisSystem` can be implanted
        either in a left eye ('LE') or right eye ('RE'). Models such as
        :py:class:`~pulse2percept.models.AxonMapModel` will treat left and
        right eyes differently (for example, adjusting the location of the
        optic disc).

        Examples
        --------
        Implant Argus II in a left eye:

        >>> from pulse2percept.implants import ArgusII
        >>> implant = ArgusII(eye='LE')
        """
        return self._eye

    @eye.setter
    def eye(self, eye):
        """Eye setter (called upon `self.eye = eye`)"""
        if not isinstance(eye, str):
            raise TypeError("'eye' must be a string, not %s." % type(eye))
        eye = eye.upper()
        if eye != 'LE' and eye != 'RE':
            raise ValueError("'eye' must be either 'LE' or 'RE', not "
                             "%s." % eye)
        self._eye = eye

    @property
    def n_electrodes(self):
        """Number of electrodes in the array

        This is equivalent to calling ``earray.n_electrodes``.
        """
        return self.earray.n_electrodes

    def __getitem__(self, item):
        return self.earray[item]

    def __iter__(self):
        return iter(self.earray)

    def keys(self):
        """Return all electrode names in the electrode array"""
        return self.earray.keys()

    def values(self):
        """Return all electrode objects in the electrode array"""
        return self.earray.values()

    def items(self):
        """Return all electrode names and objects in the electrode array

        Internally, electrodes are stored in a dictionary in
        ``earray.electrodes``. For convenience, electrodes can also be accessed
        via ``items``.

        Examples
        --------
        Save the x-coordinates of all electrodes of Argus I in a dictionary:

        >>> from pulse2percept.implants import ArgusI
        >>> xcoords = {}
        >>> for name, electrode in ArgusI().items():
        ...     xcoords[name] = electrode.x

        """
        return self.earray.items()
