import numpy as np
np.set_printoptions(precision=2, threshold=5, edgeitems=2)
from copy import deepcopy as cp
from collections import OrderedDict as ODict
from scipy import interpolate as spi

from .pulse_trains import TimeSeries
from ..utils import PrettyPrint


class Stimulus(PrettyPrint):
    """Stimulus

    A stimulus is comprised of a labeled 2-D NumPy array that contains the
    data, where the rows denoted electrodes and the columns denote points in
    time.

    A stimulus can be created from a variety of source types (see below),
    including lists and dictionaries. Depending on the source type, a stimulus
    might have a time component or not.

    .. note::
       A compression method is used to `sparsify` large and redundant data
       sources such as `TimeSeries`.

       This also means that all-zero source data and electrodes with all-zero
       activation will be trimmed from the stimulus.

       Set `sparsify=False` to deactivate this behavior.

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

    electrodes : int, string or list thereof
        Optionally, you can provide your own electrode names. If none are
        given, electrode names will be extracted from the source type (e.g.,
        the keys from a dictionary). If a scalar or NumPy array is passed,
        electrode names will be numbered 0..N.

        .. note::
           The number of electrode names provided must match the number of
           electrodes extracted from the source type (i.e., N).

    time : int, float or list thereof
        Optionally, you can provide the time points of the source data.
        If none are given, time steps will be numbered 0..M.

        .. note::
           The number of time points provided must match the number of time
           points extracted from the source type (i.e., M).

           Stimuli created from scalars or 1-D NumPy arrays will have no time
           componenet, in which case you cannot provide your own time points.

    metadata : dict
        Additional stimulus metadata can be stored in a dictionary.

    sparsify : bool
        If True, will compress the source data in two ways:
        * Remove electrodes with all-zero activation.
        * Retain only the time points at which the stimulus changes.

        .. note::
           If set to False, most models will evaluate the percept at every time
           step of the stimulus. For pulse trains, this is incredibly
           inefficient.

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

    Stimulate ten electrodes with 0uA (make sure to deactivate sparsify,
    otherwise your data container will be empty):

    >>> from pulse2percept.stimuli import Stimulus
    >>> stim = Stimulus(np.zeros(10), sparsify=False)

    Provide new electrode names for an existing Stimulus object:

    >>> from pulse2percept.stimuli import Stimulus
    >>> old_stim = Stimulus([3, 5])
    >>> new_stim = Stimulus(old_stim, electrodes=['new0', 'new1'])

    """

    def __init__(self, source, electrodes=None, time=None, metadata=None,
                 sparsify=True):
        # Extract the data and coordinates (electrodes, time) from the source:
        data, electrodes, time = self._factory(source, electrodes, time)
        if sparsify:
            # Sparsify the data. For example, in a pulse train, only the signal
            # edges are saved. This drastically reduces the memory footprint of
            # the stimulus:
            data, electrodes, time = self.sparsify(data, electrodes, time)
        # Save all attributes:
        self.data = data
        self.shape = data.shape
        self.electrodes = electrodes
        self.time = time
        self.metadata = metadata

    def get_params(self):
        return {'data': self.data, 'electrodes': self.electrodes,
                'time': self.time, 'shape': self.shape}

    def _from_source(self, source):
        """Extract the data container and time information from source data

        This private method converts input data from allowable source types
        into a 2D NumPy array, where the first dimension denotes electrodes
        and the second dimension denotes points in time.

        Some stimuli don't have a time component (such as a stimulus created
        from a scalar or 1D NumPy array. In this case, times=None.
        """
        if isinstance(source, self.__class__):
            # Stimulus: We're done. This might happen in ProsthesisSystem if
            # the user builds the stimulus themselves. It can also be used to
            # overwrite the time axis or provide new electrode names.
            time = source.time
            data = source.data
        elif np.isscalar(source) and not isinstance(source, str):
            print('scalar')
            # Scalar: 1 electrode, no time component
            time = None
            data = np.array([source]).reshape((1, -1))
        elif isinstance(source, (list, tuple)):
            print("list/tuple")
            # List or touple with N elements: 1 electrode, N time points
            time = np.arange(len(source))
            data = np.array(source).reshape((1, -1))
        elif isinstance(source, np.ndarray):
            print("ndarray", source, source.ndim)
            if source.ndim > 1:
                raise ValueError("Cannot create Stimulus object from a %d-D "
                                 "NumPy array. Must be 1-D." % source.ndim)
            # 1D NumPy array with N elements: 1 electrode, N time points
            time = np.arange(len(source))
            data = source.reshape((1, -1))
        elif isinstance(source, TimeSeries):
            print('timeseries')
            # TimeSeries with NxM time points: N electrodes, M time points
            time = np.arange(source.shape[-1]) * source.tsample
            data = source.data.reshape((-1, len(time)))
        else:
            raise TypeError("Cannot create Stimulus object from %s. Choose "
                            "from: scalar, tuple, list, NumPy array, or "
                            "TimeSeries." % type(source))
        return time, data

    def _factory(self, source, electrodes, time):
        # Input is either be a valid source type (see ``self._from_source``) or
        # a collection thereof. Thus treat everything as a collection, and
        # iterate:
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
        # Now make `data` a 2-D NumPy array, with `electrodes` as rows and
        # `times` as columns (except sometimes `times` is None).
        _data = np.vstack(_data)
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
        # All elements of `times` must be the same, but they can either be None
        # or a NumPy array, so comparing with == will fail. Therefore, convert
        # all elements to NumPy float arrays, which will convert None to NaN.
        # Then you can compare two entries with np.allclose, making sure the
        # `equal_nan` option is set to True so that two NaNs are considered
        # equal:
        for e, t in enumerate(_time):
            if not np.allclose(np.array(t, dtype=float),
                               np.array(_time[0], dtype=float),
                               equal_nan=True):
                raise ValueError("All stimuli must have the same time axis, "
                                 "but electrode %s has t=%s and electrode %s "
                                 "has t=%s." % (_electrodes[0], _time[0],
                                                _electrodes[e], t))
                break
        _time = _time[0]
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
        return _data, _electrodes, _time

    @staticmethod
    def sparsify(data, electrodes, time):
        """Compress the source data"""
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
        return data, electrodes, time
