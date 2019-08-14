import numpy as np
np.set_printoptions(precision=2, threshold=5, edgeitems=2)
from copy import deepcopy as cp
from collections import OrderedDict as ODict
from scipy.interpolate import interp1d

from .pulse_trains import TimeSeries
from ..utils import PrettyPrint


class Stimulus(PrettyPrint):
    """Stimulus

    .. versionadded:: 0.6

    A stimulus is comprised of a labeled 2-D NumPy array that contains the
    data, where the rows denoted electrodes and the columns denote points in
    time.

    A stimulus can be created from a variety of source types (see below),
    including lists and dictionaries. Depending on the source type, a stimulus
    might have a time component or not.

    Parameters
    ----------
    source : source type
        A valid source type is one of the following:

        * Scalar value: interpreted as the current amplitude delivered to a
          single electrode (no time component).
        * NumPy array:
           * Nx1 array: interpreted as N current amplitudes delivered to N
             electrodes (no time component).
           * NxM array: interpreted as N electrodes each receiving M current
             amplitudes in time.
        * `TimeSeries`: interpreted as the stimulus in time for a single
          electrode (e.g., `BiphasicPulse`, `PulseTrain`).

        In addition, you can also pass a collection of source types.
        Each element must be a valid source type for a single electrode (e.g.,
        scalar, 1-D array, TimeSeries).

        * List or tuple: List elements will be assigned to electrodes in order.
        * Dictionary: Dictionary keys are used to address electrodes by name.

    electrodes : int, string or list thereof; optional, default: None
        Optionally, you can provide your own electrode names. If none are
        given, electrode names will be extracted from the source type (e.g.,
        the keys from a dictionary). If a scalar or NumPy array is passed,
        electrode names will be numbered 0..N.

        .. note::
           The number of electrode names provided must match the number of
           electrodes extracted from the source type (i.e., N).

    time : int, float or list thereof; optional, default: None
        Optionally, you can provide the time points of the source data.
        If none are given, time steps will be numbered 0..M.

        .. note::
           The number of time points provided must match the number of time
           points extracted from the source type (i.e., M).

           Stimuli created from scalars or 1-D NumPy arrays will have no time
           componenet, in which case you cannot provide your own time points.

    metadata : dict, optional, default: None
        Additional stimulus metadata can be stored in a dictionary.

    compress : bool, optional, default: False
        If True, will compress the source data in two ways:
        * Remove electrodes with all-zero activation.
        * Retain only the time points at which the stimulus changes.
        For example, in a pulse train, only the signal edges are saved. This
        drastically reduces the memory footprint of the stimulus.

    interp_method : str or int, optional, default: 'linear'
        For SciPy's `interp1` method, specifies the kind of interpolation as a
        string ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
        'previous', 'next') or as an integer specifying the order of the spline
        interpolator to use.
        Here, 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline
        interpolation of zeroth, first, second or third order; 'previous' and
        'next' simply return the previous or next value of the point.
        By default, points outside the data range will be extrapolated.

    Examples
    --------
    Stimulate a single electrode with -13uA:

    >>> from pulse2percept.stimuli import Stimulus
    >>> stim = Stimulus(-13)

    Stimulate Electrode 'B9' with a 10Hz cathodic-first pulse train lasting
    0.2 seconds, with 0.45ms pulse duration, 40uA current amplitude, and the
    time series resolved at 0.1ms resolution:

    >>> from pulse2percept.stimuli import Stimulus, PulseTrain
    >>> stim = Stimulus({'B9': PulseTrain(0.0001, freq=10, amp=40, dur=0.2)})

    Compress an existing Stimulus:

    >>> from pulse2percept.stimuli import Stimulus, PulseTrain
    >>> stim = Stimulus({'B9': PulseTrain(0.0001, freq=10, amp=40, dur=0.5)})
    >>> stim.shape
    (1, 5000)
    >>> stim.compress()
    >>> stim.shape
    (1, 40)

    Stimulate ten electrodes with 0uA:

    >>> from pulse2percept.stimuli import Stimulus
    >>> stim = Stimulus(np.zeros(10))

    Provide new electrode names for an existing Stimulus object:

    >>> from pulse2percept.stimuli import Stimulus
    >>> old_stim = Stimulus([3, 5])
    >>> new_stim = Stimulus(old_stim, electrodes=['new0', 'new1'])

    Interpolate the stimulus value at some point in time. Here, the stimulus
    is a single-electrode ramp stimulus (stimulus value == point in time):

    >>> from pulse2percept.stimuli import Stimulus
    >>> stim = Stimulus(np.arange(10).reshape((1, -1)))
    >>> stim.interp(time=3.45) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    Stimulus(compressed=False, data=[[...3.45]], electrodes=[0],
             interp_method='linear', shape=(1, 1), time=[...3.45])

    """

    def __init__(self, source, electrodes=None, time=None, metadata=None,
                 compress=False, interp_method='linear'):
        # Extract the data and coordinates (electrodes, time) from the source:
        data, electrodes, time, compress = self._factory(source, electrodes,
                                                         time, compress)
        # Save all attributes:
        self.data = data
        self.shape = data.shape
        self.electrodes = electrodes
        self.time = time
        self.metadata = metadata
        self.compressed = False
        # Compress the data if necessary (will set compressed to True):
        if compress:
            self.compress()
        # Set up the interpolator:
        self.interp_method = interp_method
        self._set_interp()

    def get_params(self):
        """Return a dictionary of class attributes"""
        return {'data': self.data, 'electrodes': self.electrodes,
                'time': self.time, 'shape': self.shape,
                'compressed': self.compressed,
                'interp_method': self.interp_method}

    def _from_source(self, source):
        """Extract the data container and time information from source data
        This private method converts input data from allowable source types
        into a 2D NumPy array, where the first dimension denotes electrodes
        and the second dimension denotes points in time.
        Some stimuli don't have a time component (such as a stimulus created
        from a scalar or 1D NumPy array. In this case, times=None.
        """
        if np.isscalar(source) and not isinstance(source, str):
            # Scalar: 1 electrode, no time component
            time = None
            data = np.array([source], dtype=float).reshape((1, -1))
        elif isinstance(source, (list, tuple)):
            # List or touple with N elements: 1 electrode, N time points
            time = np.arange(len(source))
            data = np.array(source, dtype=float).reshape((1, -1))
        elif isinstance(source, np.ndarray):
            if source.ndim > 1:
                raise ValueError("Cannot create Stimulus object from a %d-D "
                                 "NumPy array. Must be 1-D." % source.ndim)
            # 1D NumPy array with N elements: 1 electrode, N time points
            time = np.arange(len(source))
            data = source.astype(float).reshape((1, -1))
        elif isinstance(source, TimeSeries):
            # TimeSeries with NxM time points: N electrodes, M time points
            time = np.arange(source.shape[-1]) * source.tsample
            data = source.data.astype(float).reshape((-1, len(time)))
        else:
            raise TypeError("Cannot create Stimulus object from %s. Choose "
                            "from: scalar, tuple, list, NumPy array, or "
                            "TimeSeries." % type(source))
        return time, data

    def _factory(self, source, electrodes, time, compress):
        if isinstance(source, self.__class__):
            # Stimulus: We're done. This might happen in ProsthesisSystem if
            # the user builds the stimulus themselves. It can also be used to
            # overwrite the time axis or provide new electrode names.
            _data = source.data
            _time = source.time
            _electrodes = source.electrodes
            _compress = source.compressed
        else:
            # Input is either be a valid source type (see `self._from_source`)
            # or a collection thereof. Thus treat everything as a collection,
            # and iterate:
            if isinstance(source, dict):
                iterator = source.items()
            elif isinstance(source, (list, tuple, np.ndarray)):
                iterator = enumerate(source)
            else:
                iterator = enumerate([source])
            _time = []
            _electrodes = []
            _data = []
            _compress = compress
            for e, s in iterator:
                # Extract times and data from source:
                t, d = self._from_source(s)
                _time.append(t)
                _electrodes.append(e)
                _data.append(d)
            # All elements of `times` must be the same, but they can either be
            # None or a NumPy array, so comparing with == will fail. Therefore,
            # convert all elements to NumPy float arrays, which will convert
            # None to NaN. Then you can compare two entries with np.allclose,
            # making sure the `equal_nan` option is set to True so that two NaNs
            # are considered equal:
            for e, t in enumerate(_time):
                if not np.allclose(np.array(t, dtype=float),
                                   np.array(_time[0], dtype=float),
                                   equal_nan=True):
                    raise ValueError("All stimuli must have the same time axis, "
                                     "but electrode %s has t=%s and electrode %s "
                                     "has t=%s." % (_electrodes[0], _time[0],
                                                    _electrodes[e], t))
            _time = _time[0] if _time else None
            # _time = _time[0]
            # Now make `data` a 2-D NumPy array, with `electrodes` as rows and
            # `times` as columns (except sometimes `times` is None).
            _data = np.vstack(_data) if _data else np.array([])
        # User can overwrite the names of the electrodes:
        if electrodes is not None:
            electrodes = np.array(electrodes).flatten()
            if len(electrodes) != _data.shape[0]:
                raise ValueError("Number of electrodes provided (%d) does not "
                                 "match the number of electrodes in the data "
                                 "(%d)." % (len(electrodes), _data.shape[0]))
            _electrodes = electrodes
        else:
            _electrodes = np.array(_electrodes)
            assert len(_electrodes) == _data.shape[0]
        # User can overwrite time:
        if time is not None:
            if _time is None:
                raise ValueError("Cannot set times=%s, because stimulus does "
                                 "not have a time component." % time)
            time = np.array(time).flatten()
            if len(time) != _data.shape[1]:
                raise ValueError("Number of time steps provided (%d) does not "
                                 "match the number of time steps in the data "
                                 "(%d)." % (len(time), _data.shape[1]))
            _time = time
        return _data, _electrodes, _time, _compress

    def compress(self):
        """Compress the source data

        Returns
        -------
        compressed : Stimulus

        """
        data = self.data
        electrodes = self.electrodes
        time = self.time
        # Remove rows (electrodes) with all zeros:
        keep_el = [not np.allclose(row, 0) for row in data]
        data = data[keep_el]
        electrodes = electrodes[keep_el]

        if time is not None:
            # In time, we can't just remove empty columns. We need to walk
            # through each column and save all the "state transitions" along
            # with the points in time when they happened. For example, a
            # digital signal:
            # data = [0 0 1 1 1 1 0 0 0 1], time = [0 1 2 3 4 5 6 7 8 9]
            # becomes
            # data = [0 0 1 1 0 0 1],       time = [0 1 2 5 6 8 9].
            # You always need the first and last element. You also need the
            # high and low value (along with the time stamps) for every signal
            # edge.
            ticks = []  # sparsified time stamps
            signal = []  # sparsified signal values
            for t in range(data.shape[-1]):
                if t == 0 or t == data.shape[-1] - 1:
                    # Always need the first and last element:
                    ticks.append(time[t])
                    signal.append(data[:, t])
                else:
                    if not np.allclose(data[:, t], data[:, t - 1]):
                        ticks.append(time[t - 1])
                        signal.append(data[:, t - 1])
                        ticks.append(time[t])
                        signal.append(data[:, t])
            # NumPy made the slices row vectors instead of column vectors, so
            # now we need to vertically stack them and transpose:
            data = np.vstack(signal).T
            time = np.array(ticks)
        self.data = data
        self.shape = data.shape
        self.electrodes = electrodes
        self.time = time
        self.compressed = True

    def _need_interp(self):
        """Returns True if new interpolator needs to be set up"""
        if self.time is None:
            return False
        if len(self.time) <= 1:
            return False
        return True

    def _set_interp(self):
        """Set up the interpolator"""
        self._interp = None
        if self._need_interp():
            self._interp = [interp1d(self.time, row, kind=self.interp_method,
                                     assume_sorted=True, bounds_error=False,
                                     fill_value='extrapolate')
                            for row in self.data]

    def interp(self, time=None):
        """Interpolate along the time axis

        SciPy's `interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_ method is used to interpolate stimulus values along
        the time axis. An interpolation method can be set in the Stimulus
        constructor.

        Parameters
        ----------
        time : int, float or list thereof
            The time point(s) at which to interpolate the stimulus

        Returns
        -------
        interpolated: Stimulus
            New Stimulus object with new time coordinates

        Examples
        --------
        Interpolate the stimulus value at some point in time. Here, the
        stimulus is a single-electrode ramp stimulus (stimulus value == point
        in time):

        >>> from pulse2percept.stimuli import Stimulus
        >>> stim = Stimulus(np.arange(10).reshape((1, -1)))
        >>> stim.interp(time=3.45) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Stimulus(compressed=False, data=[[...3.45]], electrodes=[0],
                 interp_method='linear', shape=(1, 1), time=[...3.45])

        Use the 'nearest' interpolation method instead
        (see `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_):

        >>> from pulse2percept.stimuli import Stimulus
        >>> stim = Stimulus(np.arange(10).reshape((1, -1)),
        ...                 interp_method='nearest')
        >>> stim.interp(time=3.45) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Stimulus(compressed=False, data=[[...3.]], electrodes=[0],
                 interp_method='linear', shape=(1, 1), time=[...3.45])

        Extrapolate to a time point outside the provided data range:

        >>> from pulse2percept.stimuli import Stimulus
        >>> stim = Stimulus(np.arange(10).reshape((1, -1)))
        >>> stim.interp(time=123.45) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Stimulus(compressed=False, data=[[...123.45]], electrodes=[0],
                 interp_method='linear', shape=(1, 1), time=[...123.45])

        """
        if not self._need_interp():
            # This includes the special case of a single time point (where
            # `interp1d` would not work):
            return Stimulus(self.data[:, 0], electrodes=self.electrodes,
                            time=None, compress=self.compressed)

        # Should work with scalars and lists:
        time = np.array([time]).flatten()
        data = np.array([[self._interp[e](t) for t in time]
                         for e, _ in enumerate(self.electrodes)])
        return Stimulus(data, electrodes=self.electrodes, time=time,
                        compress=self.compressed)

    def __eq__(self, other):
        """Returns True if two Stimulus objects are identical

        Two Stimulus objects are considered identical if they have the same
        electrode names, time steps, and data points.

        Parameters
        ----------
        other : any
            Another object or variable to which the current object should be
            compared.

        Examples
        --------
        >>> from pulse2percept.stimuli import Stimulus
        >>> stim1 = Stimulus(np.ones(3))
        >>> stim2 = Stimulus(np.zeros(5))
        >>> stim1 == stim2
        False

        Compare a Stimulus with something else entirely:

        >>> from pulse2percept.stimuli import Stimulus
        >>> stim1 = Stimulus(np.ones(3))
        >>> stim1 == 1
        False

        """
        if not isinstance(other, Stimulus):
            return False
        if self.time is None:
            if other.time is not None:
                return False
        else:
            if other.time is None:
                return False
            if len(self.time) != len(other.time):
                return False
            if not np.allclose(self.time, other.time):
                return False
        if len(self.electrodes) != len(other.electrodes):
            return False
        if not np.all(self.electrodes == other.electrodes):
            return False
        if self.shape != other.shape:
            return False
        if not np.allclose(self.data, other.data):
            return False
        return True

    def __ne__(self, other):
        """Returns True if two Stimulus objects are different

        Two Stimulus objects are considered different if they store different
        electrode names, time steps, or data points.

        Parameters
        ----------
        other : any
            Another object or variable to which the current object should be
            compared.

        Examples
        --------
        Compare two Stimulus objects:

        >>> from pulse2percept.stimuli import Stimulus
        >>> stim1 = Stimulus(np.ones(3))
        >>> stim2 = Stimulus(np.zeros(5))
        >>> stim1 != stim2
        True

        """
        return not self.__eq__(other)
