"""`ProsthesisSystem`"""
import numpy as np
from copy import deepcopy

from .electrodes import Electrode
from .electrode_arrays import ElectrodeArray
from ..stimuli import Stimulus, ImageStimulus, VideoStimulus
from ..utils import PrettyPrint


class ProsthesisSystem(PrettyPrint):
    """Visual prosthesis system

    A visual prosthesis combines an electrode array and (optionally) a
    stimulus. This is the base class for prosthesis systems such as
    :py:class:`~pulse2percept.implants.ArgusII` and
    :py:class:`~pulse2percept.implants.AlphaIMS`.

    .. versionadded:: 0.6

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
    preprocess : bool or callable, optional
        Either True/False to indicate whether to execute the implant's default
        preprocessing method whenever a new stimulus is assigned, or a custom
        function (callable).
    safe_mode : bool, optional
        If safe mode is enabled, only charge-balanced stimuli are allowed.

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
    __slots__ = ('_earray', '_stim', '_eye', 'safe_mode', 'preprocess')

    def __init__(self, earray, stim=None, eye='RE', preprocess=False,
                 safe_mode=False):
        self.earray = earray
        self.eye = eye
        self.safe_mode = safe_mode
        self.preprocess = preprocess
        self.stim = stim

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        return {'earray': self.earray, 'stim': self.stim, 'eye': self.eye,
                'safe_mode': self.safe_mode, 'preprocess': self.preprocess}

    @staticmethod
    def _require_charge_balanced(stim):
        # Stimuli without a time component return None, others return True/False
        if stim.is_charge_balanced is False:
            raise ValueError("Safety check: Stimulus must be charge-balanced.")

    def check_stim(self, stim):
        """Quality-check the stimulus

        This method is executed every time a new value is assigned to ``stim``.

        If ``safe_mode`` is set to True, this function will only allow stimuli
        that are charge-balanced.

        The user can define their own checks in implants that inherit from
        :py:class:`~pulse2percept.implants.ProsthesisSystem`.

        Parameters
        ----------
        stim : :py:class:`~pulse2percept.stimuli.Stimulus` source type
            A valid source type for the
            :py:class:`~pulse2percept.stimuli.Stimulus` object (e.g., scalar,
            NumPy array, pulse train).
        """
        if self.safe_mode:
            self._require_charge_balanced(stim)

    def preprocess_stim(self, stim):
        """Preprocess the stimulus

        This methods is executed every time a new value is assigned to ``stim``.

        No preprocessing is performed by default, but the user can define their
        own method in implants that inherit from
        return stim
        :py:class:`~pulse2percept.implants.ProsthesisSystem`.

        A custom method must return a 
        :py:class:`~pulse2percept.stimuli.Stimulus` object with the correct
        number of electrodes for the implant.

        Parameters
        ----------
        stim : :py:class:`~pulse2percept.stimuli.Stimulus` source type
            A valid source type for the
            :py:class:`~pulse2percept.stimuli.Stimulus` object (e.g., scalar,
            NumPy array, pulse train).

        Returns
        ----------
        stim_out : :py:class:`~pulse2percept.stimuli.Stimulus` object
        """
        return stim

    def plot(self, annotate=False, autoscale=True, ax=None):
        """Plot

        Parameters
        ----------
        annotate : bool, optional
            Whether to scale the axes view to the data
        autoscale : bool, optional
            Whether to adjust the x,y limits of the plot to fit the implant
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            A Matplotlib axes object. If None, will either use the current axes
            (if exists) or create a new Axes object.

        Returns
        -------
        ax : ``matplotlib.axes.Axes``
            Returns the axis object of the plot
        """
        return self.earray.plot(annotate=annotate, autoscale=autoscale, ax=ax)

    def activate(self, electrodes):
        self.earray.activate(electrodes)

    def deactivate(self, electrodes):
        self.earray.deactivate(electrodes)

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
            # Preprocess can be a function (callable) or True/False:
            if callable(self.preprocess):
                data = self.preprocess(data)
            elif self.preprocess:
                data = self.preprocess_stim(data)
            # Convert to stimulus object:
            if isinstance(data, Stimulus):
                # Already a stimulus object:
                stim = data
            elif isinstance(data, dict):
                # Electrode names already provided by keys:
                stim = Stimulus(data)
            else:
                # Use electrode names as stimulus coordinates:
                stim = Stimulus(data, electrodes=self.electrode_names)

            if len(stim.electrodes) > self.n_electrodes:
                if (isinstance(stim, (ImageStimulus, VideoStimulus)) and
                        hasattr(self.earray, 'shape')):
                    # Convenience function for electrode grids:
                    stim = stim.rgb2gray().resize(self.earray.shape)
                else:
                    err_str = ("Number of electrodes in the stimulus (%d) "
                               "does not match the number of electrodes in "
                               "the implant (%d)." % (len(stim.electrodes),
                                                      self.n_electrodes))
                    raise ValueError(err_str)
            # Make sure all electrode names are valid:
            for electrode in stim.electrodes:
                # Invalid index will return None:
                if not self.earray[electrode]:
                    raise ValueError('Electrode "%s" not found in '
                                     'implant.' % electrode)
                if not self.earray[electrode].activated:
                    raise ValueError('Cannot assign stimulus to deactivated '
                                     'Electrode "%s".' % electrode)
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

    @property
    def electrodes(self):
        """Return all electrode names and objects in the electrode array

        Internally, electrodes are stored in an ordered dictionary.
        You can iterate over different electrodes in the array as follows:

        .. code::

            for name, electrode in implant.electrodes.items():
                print(name, electrode)

        You can access an individual electrode by indexing directly into the
        prosthesis system object, e.g. ``implant['A1']`` or ``implant[0]``.

        """
        return self.earray.electrodes

    @property
    def electrode_names(self):
        """Return a list of all electrode names in the electrode array"""
        return self.earray.electrode_names

    @property
    def electrode_objects(self):
        """Return a list of all electrode objects in the array"""
        return self.earray.electrode_objects
