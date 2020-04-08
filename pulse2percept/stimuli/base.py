"""`Stimulus`"""
import numpy as np
np.set_printoptions(precision=2, threshold=5, edgeitems=2)
from copy import deepcopy as cp
from collections import OrderedDict as ODict
from scipy.interpolate import interp1d

from .pulse_trains import TimeSeries
from ..utils import PrettyPrint


class Stimulus(PrettyPrint):
    """Stimulus

    A stimulus is comprised of a labeled 2-D NumPy array that contains the
    data, where the rows denote electrodes and the columns denote points in
    time.

    A stimulus can be created from a variety of source types (see below),
    including lists and dictionaries. Depending on the source type, a stimulus
    might have a time component or not.

    You can access the stimulus applied to electrode ``e`` at time ``t``
    by directly indexing into ``Stimulus[e, t]``. In this case, ``t`` is not
    a column index but a time point. If the time point is not explicitly stored
    in the ``data`` container, its value will be automatically interpolated
    from neighboring values.

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
        * :py:class:`~pulse2percept.stimuli.TimeSeries`: interpreted as the
          stimulus in time for a single electrode (e.g.,
          :py:class:`~pulse2percept.stimuli.BiphasicPulse`,
          :py:class:`~pulse2percept.stimuli.PulseTrain`).

        In addition, you can also pass a collection of source types.
        Each element must be a valid source type for a single electrode (e.g.,
        scalar, 1-D array, :py:class:`~pulse2percept.stimuli.TimeSeries`).

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
        For SciPy's ``interp1`` method, specifies the kind of interpolation as
        a string ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
        'previous', 'next') or as an integer specifying the order of the spline
        interpolator to use.
        Here, 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline
        interpolation of zeroth, first, second or third order; 'previous' and
        'next' simply return the previous or next value of the point.

    extrapolate : bool, optional, default: False
        Whether to extrapolate data points outside the given range.

    .. versionadded:: 0.6

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
    >>> stim[:, 3.45] # doctest: +ELLIPSIS
    3.45...

    """
    # Frozen class: Only the following class attributes are allowed
    __slots__ = ('metadata', '_interp', '_interp_method', '_extrapolate',
                 '__stim')

    def __init__(self, source, electrodes=None, time=None, metadata=None,
                 compress=False, interp_method='linear', extrapolate=False):
        self.metadata = metadata
        # Private: User is not supposed to overwrite these later on:
        self._interp_method = interp_method
        self._extrapolate = extrapolate
        # Extract the data and coordinates (electrodes, time) from the source:
        self._factory(source, electrodes, time, compress)

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        return {'data': self.data, 'electrodes': self.electrodes,
                'time': self.time, 'shape': self.shape,
                'metadata': self.metadata}

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
            data = np.array([source], dtype=np.float32).reshape((1, -1))
        elif isinstance(source, (list, tuple)):
            # List or touple with N elements: 1 electrode, N time points
            time = np.arange(len(source))
            data = np.array(source, dtype=np.float32).reshape((1, -1))
        elif isinstance(source, np.ndarray):
            if source.ndim > 1:
                raise ValueError("Cannot create Stimulus object from a %d-D "
                                 "NumPy array. Must be 1-D." % source.ndim)
            # 1D NumPy array with N elements: 1 electrode, N time points
            time = np.arange(len(source))
            data = source.astype(np.float32).reshape((1, -1))
        elif isinstance(source, TimeSeries):
            # TimeSeries with NxM time points: N electrodes, M time points
            time = np.arange(source.shape[-1]) * source.tsample
            data = source.data.astype(np.float32).reshape((-1, len(time)))
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
        else:
            # Input is either be a valid source type (see `self._from_source`)
            # or a collection thereof. Thus treat everything as a collection
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
                if not np.allclose(np.array(t, dtype=np.float32),
                                   np.array(_time[0], dtype=np.float32),
                                   equal_nan=True):
                    raise ValueError("All stimuli must have the same time axis, "
                                     "but electrode %s has t=%s and electrode %s "
                                     "has t=%s." % (_electrodes[0], _time[0],
                                                    _electrodes[e], t))
            _time = _time[0] if _time else None
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
        # Store the data in the private container. Setting all elements at once
        # enforces consistency; e.g., between shape of electrodes and time:
        self._stim = {
            'data': _data,
            'electrodes': _electrodes,
            'time': _time,
        }
        # Compress the data upon request:
        if compress:
            self.compress()

    def compress(self):
        """Compress the source data

        Returns
        -------
        compressed : :py:class:`~pulse2percept.stimuli.Stimulus`

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
        self._stim = {
            'data': data,
            'electrodes': electrodes,
            'time': time,
        }

    def __getitem__(self, item):
        """Returns an item from the data array, interpolated if necessary"""
        # NumPy handles most indexing and slicing. However, we need to prevent
        # cases like stim[:, [0, 1]] which ask for time=[0.0, 1.0] and not for
        # column index 0 and 1:
        if isinstance(item, tuple):
            if isinstance(item[1], slice):
                # Currently the only supported slice is ':'
                if item[1].start or item[1].stop or item[1].step:
                    raise NotImplementedError("Slicing the time axis not yet "
                                              "supported.")
            else:
                if not np.any(item[1] == Ellipsis):
                    # Convert to float so time is not mistaken for column index
                    item = (item[0], np.float32(item[1]))

        try:
            # NumPy handles most indexing and slicing:
            return self._stim['data'][item]
        except IndexError as e:
            # IndexErrors must still be thrown except when `item` is a tuple,
            # in which case we might want to interpolate time:
            if not isinstance(item, tuple):
                raise IndexError(e)
        # Handle the special case where we interpolate time.
        # First of all, if time=None, then _interp=None, and we won't interp:
        if self._interp is None:
            raise ValueError("Cannot interpolate time if time=None.")
        # Electrodes cannot be interpolated, so convert from slice, ellipsis or
        # indices into a list:
        if item[0] == Ellipsis:
            interp = np.array(self._interp)
        elif isinstance(item[0], list):
            interp = np.array([self._interp[i] for i in item[0]])
        else:
            interp = np.array([self._interp[item[0]]]).flatten()
        # Time might be a single index or a list of indices. Convert all to
        # list so we can iterate:
        time = np.array([item[1]]).flatten()
        data = np.array([[ip(t) for t in time] for ip in interp])
        # Return a single element as scalar:
        if data.size <= 1:
            data = data.ravel()[0]
        return data

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
        >>> Stimulus([1, 2, 3]) == Stimulus([1, 2, 3])
        True
        >>> Stimulus(np.ones(3)) == Stimulus(np.zeros(5))
        False

        Compare a Stimulus with something else entirely:

        >>> Stimulus(np.ones(3)) == 1
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
        if not np.array_equal(self.electrodes, other.electrodes):
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

    @property
    def _stim(self):
        """A dictionary containing all the stimulus data"""
        return self.__stim

    @_stim.setter
    def _stim(self, stim):
        # Check stimulus data for consistency:
        if not isinstance(stim, dict):
            raise TypeError("Stimulus data must be stored in a dictionary, "
                            "not %s." % type(stim))
        for field in ['data', 'electrodes', 'time']:
            if field not in stim:
                raise AttributeError("Stimulus dict must contain a field "
                                     "'%s'." % field)
        if len(stim['data']) > 0:
            if not isinstance(stim['data'], np.ndarray):
                raise TypeError("Stimulus data must be a NumPy array, not "
                                "%s." % type(stim['data']))
            if stim['data'].ndim != 2:
                raise ValueError("Stimulus data must be a 2-D NumPy array, not "
                                 "%d-D." % stim['data'].ndim)
        if len(stim['electrodes']) != stim['data'].shape[0]:
            raise ValueError("Number of electrodes (%d) must match the number "
                             "of rows in the data array "
                             "(%d)." % (len(stim['electrodes']),
                                        stim['data'].shape[0]))
        if len(stim['electrodes']) != stim['data'].shape[0]:
            raise ValueError("Number of electrodes (%d) must match the number "
                             "of rows in the data array "
                             "(%d)." % (len(stim['electrodes']),
                                        stim['data'].shape[0]))
        if stim['time'] is not None:
            if len(stim['time']) != stim['data'].shape[1]:
                raise ValueError("Number of time points (%d) must match the "
                                 "number of columns in the data array "
                                 "(%d)." % (len(stim['time']),
                                            stim['data'].shape[1]))
        elif len(stim['data']) > 0:
            if stim['data'].shape[1] > 1:
                raise ValueError("Number of columns in the data array must be "
                                 "1 if time=None.")
        # All checks passed, store the data:
        self.__stim = stim
        # Set up the interpolator:
        self._interp = None
        if self.time is None:
            return
        if len(self.time) == 1:
            # Special case: Duplicate data with slightly different time points
            # so we can set up an interp1d:
            eps = np.finfo(np.float32).eps
            time = np.array([self.time - eps, self.time + eps]).flatten()
            data = np.repeat(self.data, 2, axis=1)
        else:
            time = self.time
            data = self.data
        if self._extrapolate:
            bounds_error = False
            fill_value = 'extrapolate'
        else:
            bounds_error = True
            fill_value = None
        self._interp = [interp1d(time, row, kind=self._interp_method,
                                 assume_sorted=True,
                                 bounds_error=bounds_error,
                                 fill_value=fill_value)
                        for row in data]

    @property
    def data(self):
        """Stimulus data container

        A 2-D NumPy array that contains the stimulus data, where the rows
        denote electrodes and the columns denote points in time.
        """
        return self._stim['data']

    @property
    def shape(self):
        """Data container shape"""
        return self.data.shape

    @property
    def electrodes(self):
        """Electrode names

        A list of electrode names, corresponding to the rows in the data
        container.
        """
        return self._stim['electrodes']

    @property
    def time(self):
        """Time steps

        A list of time steps, corresponding to the columns in the data
        container.
        """
        return self._stim['time']
