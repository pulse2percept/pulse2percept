import numpy as np
np.set_printoptions(precision=2, threshold=5, edgeitems=2)
from copy import deepcopy as cp
from collections import OrderedDict as ODict
from scipy import interpolate as spi

from .pulse_trains import TimeSeries
from ..utils import PrettyPrint


class Stimulus(PrettyPrint):

    def __init__(self, source, electrode=None, time=None, metadata=None):
        # Input is either be a valid source type (see ``self._from_source``) or
        # a collection thereof. Thus treat everything as a collection, and
        # iterate:
        if isinstance(source, dict):
            iterator = source.items()
        elif isinstance(source, (list, tuple, np.ndarray)):
            iterator = enumerate(source)
        else:
            iterator = enumerate([source])
        _times = []
        _electrodes = []
        data = []
        for e, s in iterator:
            # Extract times and data from source:
            t, d = self._from_source(s)
            _times.append(t)
            _electrodes.append(e)
            data.append(d)
        # Now make `data` a 2-D NumPy array, with `electrodes` as rows and
        # `times` as columns (except sometimes `times` is None).
        data = np.vstack(data)
        # User can overwrite the names of the electrodes:
        if electrode is not None:
            electrode = np.array(electrode).flatten()
            if len(electrode) != data.shape[0]:
                raise ValueError("Number of electrodes provided (%d) does not "
                                 "match the number of electrodes in the data "
                                 "(%d)." % (len(electrode), data.shape[0]))
            _electrodes = electrode
        else:
            _electrodes = np.array(_electrodes)
        # All elements of `times` must be the same, but they can either be None
        # or a NumPy array, so comparing with == will fail. Therefore, convert
        # all elements to NumPy float arrays, which will convert None to NaN.
        # Then you can compare two entries with np.allclose, making sure the
        # `equal_nan` option is set to True so that two NaNs are considered
        # equal:
        for e, t in enumerate(_times):
            if not np.allclose(np.array(t, dtype=float),
                               np.array(_times[0], dtype=float),
                               equal_nan=True):
                raise ValueError("All stimuli must have the same time axis, "
                                 "but electrode %s has t=%s and electrode %s "
                                 "has t=%s." % (_electrodes[0], _times[0],
                                                _electrodes[e], t))
                break
        _times = _times[0]
        # User can overwrite time:
        if time is not None:
            if _times is None:
                raise ValueError("Cannot set times=%s, because stimulus does "
                                 "not have a time component." % time)
            time = np.array(time).flatten()
            if len(time) != data.shape[1]:
                raise ValueError("Number of time steps provided (%d) does not "
                                 "match the number of time steps in the data "
                                 "(%d)." % (len(time), data.shape[1]))
            _times = time

        # Last step is to sparsify the data. For example, in a pulse train,
        # only the signal edges are saved. This drastically reduces the memory
        # footprint of the stimulus:
        data, _electrodes, _times = self.sparsify(data, _electrodes, _times)

        self.data = data
        self.shape = data.shape
        self.electrode = _electrodes
        self.time = _times
        self.metadata = metadata

    def get_params(self):
        return {'data': self.data, 'electrode': self.electrode,
                'time': self.time, 'shape': self.shape}

    def _from_source(self, source):
        """Extract the data container and time information from source data

        This private method converts input data from allowable source types
        into a 2D NumPy array, where the first dimension denotes electrodes
        and the second dimension denotes points in time.

        Some stimuli don't have a time component (such as a stimulus created
        from a scalar or 1D NumPy array. In this case, times=None.
        """
        if np.isscalar(source) and not isinstance(source, str):
            print('scalar')
            # Scalar: 1 electrode, no time component
            times = None
            data = np.array([source]).reshape((1, -1))
        elif isinstance(source, (list, tuple)):
            print("list/tuple")
            # List or touple with N elements: 1 electrode, N time points
            times = np.arange(len(source))
            data = np.array(source).reshape((1, -1))
        elif isinstance(source, np.ndarray):
            print("ndarray", source, source.ndim)
            if source.ndim > 1:
                raise ValueError("Cannot create Stimulus object from a %d-D "
                                 "NumPy array. Must be 1-D." % source.ndim)
            # 1D NumPy array with N elements: 1 electrode, N time points
            times = np.arange(len(source))
            data = source.reshape((1, -1))
        elif isinstance(source, TimeSeries):
            print('timeseries')
            # TimeSeries with NxM time points: N electrodes, M time points
            times = np.arange(source.shape[-1]) * source.tsample
            data = source.data.reshape((-1, len(times)))
        else:
            raise TypeError("Cannot create Stimulus object from %s. Choose "
                            "from: scalar, tuple, list, NumPy array, or "
                            "TimeSeries." % type(source))
        return times, data

    @staticmethod
    def sparsify(data, electrodes, times):
        # Remove rows (electrodes) with all zeros:
        keep_el = [not np.allclose(row, 0) for row in data]
        data = data[keep_el]
        electrodes = electrodes[keep_el]

        if times is not None:
            # In time, we can't just remove empty columns. We need to walk
            # through each column and save all the "state transitions" along
            # with the points in time when they happened. For example, a
            # digital signal:
            # data = [0 0 1 1 1 1 0 0 0 1], times = [0 1 2 3 4 5 6 7 8 9]
            # becomes
            # data = [0 0 1 1 0 0 1],       times = [0 1 2 5 6 8 9].
            # You always need the first and last element. You also need the
            # high and low value (along with the time stamps) for every signal
            # edge.
            time = []  # sparsified time stamps
            signal = []  # sparsified signal values
            for t in range(data.shape[-1]):
                if t == 0 or t == data.shape[-1] - 1:
                    # Always need the first and last element:
                    time.append(times[t])
                    signal.append(data[:, t])
                else:
                    if not np.allclose(data[:, t], data[:, t - 1]):
                        time.append(times[t - 1])
                        signal.append(data[:, t - 1])
                        time.append(times[t])
                        signal.append(data[:, t])
            # NumPy made the slices row vectors instead of column vectors, so
            # now we need to vertically stack them and transpose:
            data = np.vstack(signal).T
            times = np.array(time)
        return data, electrodes, times


#     def sel(self, electrode=None, t=None):
#         pass

#     def interp(self, electrode=None, t=None, fill_values='extrapolate'):
#         pass

#     def plot(self, electrode=None, t=None):
#         pass
