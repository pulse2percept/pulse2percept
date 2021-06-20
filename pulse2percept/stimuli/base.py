"""`Stimulus`, `ImageStimulus`"""
import warnings
from ..utils import PrettyPrint, parfor, unique, is_strictly_increasing
from ..utils.constants import DT, MIN_AMP
from ._base import fast_compress_space, fast_compress_time

import logging
from sys import _getframe
from matplotlib.axes import Subplot
import matplotlib.pyplot as plt
from copy import deepcopy
import operator as ops
from math import isclose
import numpy as np
np.set_printoptions(precision=2, threshold=5, edgeitems=2)

# Log all warnings.warn() at the WARNING level:
logging.captureWarnings(True)


def merge_time_axes(data, time):
    """Merge time axes

    When a collection of source types is passed, it is possible that they
    have different time axes (e.g., different time steps, or a different
    stimulus duration). In this case, we need to merge all time axes into a
    single, coherent one. This is expensive, because of interpolation.
    """
    # We can skip the costly interpolation if all `time` vectors are
    # identical:
    identical = True
    for t in time:
        if len(t) != len(time[0]) or not np.allclose(t, time[0]):
            identical = False
            break
    if identical:
        return data, [time[0]]
    # Otherwise, we need to interpolate. Keep only the unique time points
    # across stimuli:
    new_time = unique(np.concatenate(time), tol=DT)
    # Now we need to interpolate the data values at each of these
    # new time points.
    new_data = []
    for t, d in zip(time, data):
        # t is a 1D vector, d is a 2D data matrix and might have more than
        # one row:
        new_rows = [np.interp(new_time, t, row) for row in d]
        new_rows = np.array(new_rows).reshape((-1, len(new_time)))
        new_data.append(new_rows)
    return new_data, [new_time]


class Stimulus(PrettyPrint):
    """Stimulus

    A stimulus is comprised of a labeled 2D NumPy array that contains the data,
    where the rows denote electrodes and the columns denote points in time.
    A stimulus can be created from a variety of source types (e.g., scalars,
    lists, NumPy arrays, and dictionaries).

    .. seealso ::

        *  `Basic Concepts > Electrical Stimuli <topics-stimuli>`

    .. versionadded:: 0.6

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

        In addition, you can also pass a collection of source types.
        Each element must be a valid source type for a single electrode (e.g.,
        scalar, 1-D array, :py:class:`~pulse2percept.stimuli.Stimulus`).

        * List or tuple: List elements will be assigned to electrodes in order.
        * Dictionary: Dictionary keys are used to address electrodes by name.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. If none are
        given, electrode names will be extracted from the source type (e.g.,
        the keys from a dictionary). If a scalar or NumPy array is passed,
        electrode names will be numbered 0..N.

        .. note::

           The number of electrode names provided must match the number of
           electrodes extracted from the source type (i.e., N).

    time : int, float or list thereof; optional
        Optionally, you can provide the time points of the source data.
        If none are given, time steps will be numbered 0..M.

        .. note::

           The number of time points provided must match the number of time
           points extracted from the source type (i.e., M).
           Stimuli created from scalars or 1-D NumPy arrays will have no time
           componenet, in which case you cannot provide your own time points.

    metadata : dict, optional
        Additional stimulus metadata can be stored in a dictionary.

    compress : bool, optional
        If True, will compress the source data in two ways:

        * Remove electrodes with all-zero activation.
        * Retain only the time points at which the stimulus changes.

        For example, in a pulse train, only the signal edges are saved. This
        drastically reduces the memory footprint of the stimulus.

    Notes
    -----
    *  Depending on the source type, a stimulus might have a time component or
       not (e.g., scalars: time=None).
    *  You can access the stimulus applied to electrode ``e`` at time ``t``
       by directly indexing into ``Stimulus[e, t]``. In this case, ``t`` is not
       a column index but a time point.
    *  If the time point is not explicitly stored in the ``data`` container,
       its value will be automatically interpolated from neighboring values.
    *  If a requested time point lies outside the range of stored data,
       the value of its closest end point will be returned.

    Examples
    --------
    Stimulate a single electrode with -13uA:

    >>> from pulse2percept.stimuli import Stimulus
    >>> stim = Stimulus(-13)

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
    __slots__ = ('metadata', '_is_compressed', '__stim')

    def __init__(self, source, electrodes=None, time=None, metadata=None,
                 compress=False):
        if isinstance(metadata, dict) and 'electrodes' in metadata.keys():
            self.metadata = metadata
        else:
            self.metadata = {'electrodes': {}, 'user': metadata}
        # Flag will be flipped in the compress method:
        self.is_compressed = False
        # Extract the data and coordinates (electrodes, time) from the source:
        self._factory(source, electrodes, time, compress)

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        return {'data': self.data, 'electrodes': self.electrodes,
                'time': self.time, 'shape': self.shape, 'dt': self.dt,
                'is_charge_balanced': self.is_charge_balanced,
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
            data = np.array([source], dtype=np.float32).reshape((1, -1))
            time = None
            electrodes = None
        elif isinstance(source, (list, tuple)):
            # List or touple with N elements: 1 electrode, N time points
            data = np.array(source, dtype=np.float32).reshape((1, -1))
            time = np.arange(data.shape[-1], dtype=np.float32)
            electrodes = None
        elif isinstance(source, np.ndarray):
            if source.ndim > 1:
                raise ValueError("Cannot create Stimulus object from a %d-D "
                                 "NumPy array. Must be 1-D." % source.ndim)
            # 1D NumPy array with N elements: 1 electrode, N time points
            data = source.astype(np.float32).reshape((1, -1))
            time = np.arange(data.shape[-1], dtype=np.float32)
            electrodes = None
        elif isinstance(source, Stimulus):
            # e.g. from a dictionary of Stimulus objects
            data = source.data
            time = source.time
            electrodes = source.electrodes
        else:
            raise TypeError("Cannot create Stimulus object from %s. Choose "
                            "from: scalar, tuple, list, NumPy array, or "
                            "Stimulus." % type(source))
        return time, data, electrodes

    def _factory(self, source, electrodes, time, compress):
        """Build the Stimulus object from the specified source type"""
        if isinstance(source, self.__class__):
            # Stimulus: We're done. This might happen in ProsthesisSystem if
            # the user builds the stimulus themselves. It can also be used to
            # overwrite the time axis or provide new electrode names:
            _data = source.data
            _time = source.time
            _electrodes = source.electrodes
        elif isinstance(source, np.ndarray):
            # A NumPy array is either 1-D (list of electrodes, time=None) or
            # 2-D (electrodes x time points):
            if source.ndim == 1:
                _data = source.reshape((-1, 1))
                _time = None
                _electrodes = np.arange(_data.shape[0])
            elif source.ndim == 2:
                _data = source
                _time = np.arange(_data.shape[-1], dtype=np.float32)
                _electrodes = np.arange(_data.shape[0])
            else:
                raise ValueError("Cannot create Stimulus object from a %d-D "
                                 "NumPy array. Must be < 2-D." % source.ndim)
        else:
            # Input is either a scalar or (more likely) a collection of source
            # types. Easiest to tream them all as a collection and iterate:
            if isinstance(source, dict):
                iterator = source.items()
            elif isinstance(source, (list, tuple)):
                iterator = enumerate(source)
            else:
                iterator = enumerate([source])
            _time = []
            _electrodes = []
            _data = []
            for ele, src in iterator:
                # Extract times and data from source:
                t, d, e = self._from_source(src)
                _time.append(t)
                _data.append(d)
                if isinstance(source, dict):
                    # Special case, electrode names are specified in a dict:
                    _electrodes.append(ele)
                else:
                    # In all other cases, use the electrode names specified by
                    # the source (unless they're None):
                    _electrodes.append(e if e is not None else ele)
                try:
                    self.metadata['electrodes'][str(ele)] = {
                        'metadata': src.metadata,
                        'type': type(src)
                    }
                except AttributeError:
                    pass
            # Make sure all stimuli have time=None or none of them do:
            if len(np.unique([t is None for t in _time])) > 1:
                raise ValueError("If one stimulus has time=None, all others "
                                 "must have time=None as well.")
            # When none of the stimuli have time=None, we need to merge the
            # time axes (this is expensive because of interpolation):
            if len(_time) > 1 and _time[0] is not None:
                _data, _time = merge_time_axes(_data, _time)
            # Now make `_data` a 2-D NumPy array, with `_electrodes` as rows
            # and `_time` as columns (except sometimes `_time` is None).
            _data = np.vstack(_data) if _data else np.array([])
            _time = _time[0] if _time else None

        # User can overwrite the names of the electrodes:
        if electrodes is not None:
            _electrodes = np.array([electrodes]).flatten()
        else:
            if not isinstance(_electrodes, np.ndarray):
                # Could be a list of NumPy arrays, need to flatten:
                try:
                    _electrodes = np.concatenate(_electrodes)
                except ValueError:
                    _electrodes = np.array(_electrodes)
        if len(_electrodes) != _data.shape[0]:
            raise ValueError("Number of electrodes provided (%d) does not "
                             "match the number of electrodes in the data "
                             "(%d)." % (len(_electrodes), _data.shape[0]))
        unq, nunq = np.unique(_electrodes, return_index=True)
        if len(unq) != _data.shape[0]:
            # We found duplicate names: replace them by integer index
            idx = np.delete(np.arange(len(_electrodes)), nunq)
            msg = ("Duplicate electrode names detected %s, and replaced with "
                   "integer values" % _electrodes[idx])
            warnings.warn(msg)
            _electrodes[idx] = idx

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
            'data': np.ascontiguousarray(_data, dtype=np.float32),
            'electrodes': _electrodes,
            'time': _time if _time is None else _time.astype(np.float32),
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
        keep_el = fast_compress_space(data)
        data = data[keep_el]
        electrodes = electrodes[keep_el]

        if time is not None:
            idx_time = fast_compress_time(data, time)
            data = data[:, idx_time]
            time = time[idx_time]

        self._stim = {
            'data': data,
            'electrodes': electrodes,
            'time': time,
        }
        self.is_compressed = True

    def append(self, other):
        """Append another stimulus

        This method appends another stimulus (with matching electrodes) in
        time. The combined stimulus duration will be the sum of the two
        individual stimuli.

        .. versionadded:: 0.7

        Parameters
        ----------
        other : :py:class:`~pulse2percept.stimuli.Stimulus`
            Another stimulus with matching electrodes.

        Returns
        -------
        comb : :py:class:`~pulse2percept.stimuli.Stimulus`
            A combined stimulus with the same number of electrodes and new
            stimulus duration equal to the sum of the two individual stimuli.

        """
        if not isinstance(other, Stimulus):
            raise TypeError("Other object must be a Stimulus, not "
                            "%s." % type(other))
        if self.time is None or other.time is None:
            raise ValueError("Cannot append another stimulus if time=None.")
        if not np.all(other.electrodes == self.electrodes):
            raise ValueError("Both stimuli must have the same electrodes.")
        if other.time[0] < 0:
            raise NotImplementedError("Appending a stimulus with a negative "
                                      "time axis is currently not supported.")
        stim = deepcopy(self)
        # Last time point of `self` can be merged with first point of `other`
        # but only if they have the same amplitude(s):
        if isclose(other.time[0], 0, abs_tol=DT):
            if not np.allclose(other.data[:, 0], self.data[:, -1]):
                err_str = ("Data mismatch: Cannot append other stimulus "
                           "because other[t=0] != this[t=%fms]. You may need "
                           "to shift the other stimulus in time by at least "
                           "%.1e ms." % (this.time[-1], DT))
                raise ValueError(err_str)
            time = np.hstack((self.time, other.time[1:] + self.time[-1]))
            data = np.hstack((self.data, other.data[:, 1:]))
        else:
            time = np.hstack((self.time, other.time + self.time[-1]))
            data = np.hstack((self.data, other.data))
        # Append the data points. If there's something wrong with the
        # concatenated list of time points, the stim setter will catch it:
        stim._stim = {'data': data,
                      'electrodes': self.electrodes,
                      'time': time}
        return stim

    def plot(self, electrodes=None, time=None, fmt='k-', ax=None):
        """Plot the stimulus

        .. versionadded:: 0.7

        Parameters
        ----------
        electrodes : int, string, or list thereof; optional, default: None
            The electrodes for which to plot the stimulus. If None, all
            electrodes are plotted.
        time : (t_min, t_max) tuple, slice, or list of exact time points
            The time points at which to plot the stimulus. Specify a range of
            time points with a tuple or a slice, or specify the exact time
            points to interpolate.
            If None, all time points are plotted.
        fmt : str, optional, default: 'k-'
            A Matplotlib format string; e.g., 'ro' for red circles.
        ax : matplotlib.axes.Axes or list thereof; optional, default: None
            A Matplotlib Axes object or a list thereof (one per electrode to
            plot). If None, a new Axes object will be created.

        Returns
        -------
        axes : matplotlib.axes.Axes or np.ndarray of them
            Returns one matplotlib.axes.Axes per electrode
        """
        if self.time is None:
            # Cannot plot stimulus with single time point:
            raise NotImplementedError
        if electrodes is None:
            # Plot all electrodes:
            electrodes = self.electrodes
        elif isinstance(electrodes, (int, str)):
            # Convert to list so we can iterate over it:
            electrodes = [electrodes]
        # The user can ask for a range, slice, or list of time points, which
        # are either interpolated or loaded directly.
        if time is None:
            # Ask for a slice instead of `self.time` to avoid interpolation,
            # which can be time-consuming for an uncompressed stimulus:
            time = slice(None)
        if isinstance(time, tuple):
            # Return a range of time points:
            t_idx = (self.time > time[0]) & (self.time < time[1])
            # Include the end points (might have to be interpolated):
            t_vals = [time[0]] + list(self.time[t_idx]) + [time[1]]
            t_idx = t_vals
        elif isinstance(time, (list, np.ndarray)):
            # Return list of exact time points:
            t_idx = time
            t_vals = time
        elif isinstance(time, slice) or time == Ellipsis:
            # Return a slice of time points:
            t_idx = time
            t_vals = self.time[t_idx]
        else:
            raise TypeError('"time" must be a tuple, slice, list, or NumPy '
                            'array, not %s.' % type(time))
        axes = ax
        if axes is None:
            if len(electrodes) == 1:
                axes = plt.gca()
            else:
                _, axes = plt.subplots(nrows=len(electrodes),
                                       figsize=(8, 1.2 * len(electrodes)))
        if not isinstance(axes, (list, np.ndarray)):
            # Convert to list so w can iterate over it:
            axes = [axes]
        for i, ax in enumerate(axes):
            if not isinstance(ax, Subplot):
                raise TypeError("'axes' must be a list of subplots, but "
                                "axes[%d] is %s." % (i, type(ax)))
        if len(axes) != len(electrodes):
            raise ValueError("Number of subplots (%d) must be equal to the "
                             "number of electrodes (%d)." % (len(axes),
                                                             len(electrodes)))
        # Plot each electrode in its own subplot:
        for ax, electrode in zip(axes, electrodes):
            # Slice or interpolate stimulus:
            slc = self.__getitem__((electrode, t_idx))
            ax.plot(t_vals, np.squeeze(slc), fmt, linewidth=2)
            # Turn off the ugly box spines:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            # Annotate the subplot:
            ax.set_xticks([])
            ax.set_yticks([slc.min(), 0, slc.max()])
            x_pad = 0.02 * (t_vals[-1] - t_vals[0])
            ax.set_xlim(t_vals[0] - x_pad, t_vals[-1] + x_pad)
            y_pad = np.maximum(1, 0.02 * (slc.max() - slc.min()))
            ax.set_ylim(slc.min() - y_pad, slc.max() + y_pad)
            ax.set_ylabel(electrode)
        # Show x-ticks only on last subplot:
        axes[-1].set_xticks(np.linspace(t_vals[0], t_vals[-1], num=5))
        # Labels are common to all subplots:
        axes[-1].figure.subplots_adjust(bottom=0.2)
        axes[-1].figure.text(0.5, 0, 'Time (ms)', va='top', ha='center')
        axes[-1].figure.text(0, 0.5, r'Amplitude ($\mu$A)', va='center',
                             ha='center', rotation='vertical')
        if len(axes) == 1:
            return axes[0]
        return axes

    def __getitem__(self, item):
        """Returns an item from the data array, interpolated if necessary

        There are many potential use cases:

        *  ``stim[i]`` or ``stim[i, :]``: access electrode ``i`` (int or str)
        *  ``stim[[i0,i1]]`` or ``stim[[i0, i1], :]``
        *  ``stim[stim.electrodes != 'A1', :]``
        *  ``stim[:, 1]``: always interpreted as t=1.0, not index=1
        *  ``stim[:, 1.234]``: interpolated time
        *  ``stim[:, stim.time < 0.4]``, ``stim[:, 0.3:1.9:0.001]``

        """
        # STEP 1: AVOID CONFUSING TIME POINTS WITH COLUMN INDICES
        # NumPy handles most indexing and slicing. However, we need to prevent
        # cases like stim[:, [0, 1]] which ask for time=[0.0, 1.0] and not for
        # column index 0 and 1:
        if isinstance(item, tuple):
            electrodes = item[0]
            time = item[1]
            if isinstance(time, slice):
                if not time.step:
                    # We can't interpolate if we don't know the step size, so
                    # the only allowed option is slice(None, None, None), which
                    # is the same as ':'
                    if time.start or time.stop:
                        raise ValueError("You must provide a step size when "
                                         "slicing the time axis.")
                else:
                    start = self.time[0] if time.start is None else time.start
                    stop = self.time[-1] if time.stop is None else time.stop
                    time = np.arange(start, stop, time.step, dtype=np.float32)
            else:
                if not np.any(time == Ellipsis):
                    # Convert to float so time is not mistaken for column index
                    if np.array(time).dtype != bool:
                        time = np.float32(time)
        else:
            electrodes = item
            time = None

        # STEP 2: ELECTRODES COULD BE SPECIFIED AS INT OR STR
        if isinstance(electrodes, (list, np.ndarray)) or np.isscalar(electrodes):
            # Electrodes cannot be interpolated, so convert from slice,
            # ellipsis or indices into a list:
            parsed_electrodes = []
            for e in np.array([electrodes]).ravel():
                if isinstance(e, str):
                    # Use string as index into the list of electrode names:
                    parsed_electrodes.append(list(self.electrodes).index(e))
                else:
                    # Most likely an integer index:
                    parsed_electrodes.append(e)
            if not isinstance(electrodes, (list, np.ndarray)):
                # If a scalar was passed, return a scalar:
                electrodes = parsed_electrodes[0]
            else:
                # Otherwise return an array:
                electrodes = np.array(parsed_electrodes)
        # Make sure electrode index is valid:
        try:
            self._stim['data'][electrodes]
        except IndexError:
            raise IndexError("Invalid electrode index", electrodes)

        # STEP 2: NUMPY HANDLES MOST INDEXING AND SLICING:
        # Rebuild original index from ``electrodes`` and ``time``:
        if time is None:
            item = electrodes
        else:
            item = (electrodes, time)
        try:
            return self._stim['data'][item]
        except IndexError as e:
            # IndexErrors must still be thrown except when `item` is a tuple,
            # in which case we might want to interpolate time:
            if not isinstance(item, tuple):
                raise IndexError(e)

        # STEP 3: INTERPOLATE TIME
        # From here on out, we know that ``item`` is a tuple, otherwise we
        # would have raised an IndexError above.
        # First of all, if time=None, we won't interp:
        if self.time is None:
            raise ValueError("Cannot interpolate time if time=None.")
        time = np.array([time]).flatten()
        if (not isinstance(electrodes, (list, np.ndarray)) and
                electrodes == Ellipsis):
            data = self.data
        else:
            data = self.data[electrodes, :].reshape(-1, len(self.time))
        data = [np.interp(time, self.time, row) for row in data]
        data = np.array(data).reshape((-1, len(time)))
        # Return a single element as scalar:
        if data.size == 1:
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
            if not np.allclose(self.time, other.time, atol=DT):
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

    def _apply_operator(self, a, op, b, field='data'):
        """Template for all arithmetic operators"""
        # One of the arguments must be a scalar (the other being self.data):
        a_supported = np.isscalar(a) and not isinstance(a, str)
        b_supported = np.isscalar(b) and not isinstance(b, str)
        if not a_supported and not b_supported:
            raise TypeError("Unsupported operand for types %s and "
                            "%s" % (type(a), type(b)))
        # Return a copy of the current object with the new data:
        stim = deepcopy(self)
        stim._stim = {'data': op(a, b) if field == 'data' else stim.data,
                      'electrodes': stim.electrodes,
                      'time': op(a, b) if field == 'time' else stim.time}
        return stim

    def __add__(self, scalar):
        """Add a scalar to every data point in the stimulus"""
        return self._apply_operator(self.data, ops.add, scalar)

    def __radd__(self, scalar):
        """Add a scalar to every data point in the stimulus"""
        return self.__add__(scalar)

    def __sub__(self, scalar):
        """Subtract a scalar from every data point in the stimulus"""
        return self._apply_operator(self.data, ops.sub, scalar)

    def __rsub__(self, scalar):
        """Subtract every data point in the stimulus from a scalar"""
        return self._apply_operator(scalar, ops.sub, self.data)

    def __mul__(self, scalar):
        """Multiply every data point in the stimulus with a scalar"""
        return self._apply_operator(self.data, ops.mul, scalar)

    def __rmul__(self, scalar):
        """Multiply every data point in the stimulus with a scalar"""
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        """Divide every data point in the stimulus by a scalar"""
        return self._apply_operator(self.data, ops.truediv, scalar)

    def __neg__(self):
        """Flip the sign of every data point in the stimulus"""
        return self.__mul__(-1)

    def __rshift__(self, scalar):
        """Shift every time point in the stimulus some ms into the future"""
        return self._apply_operator(self.time, ops.add, scalar, field='time')

    def __lshift__(self, scalar):
        """Shift every time point in the stimulus some ms into the past"""
        return self.__rshift__(-scalar)

    def _check_stim(self, stim):
        # Check stimulus data for consistency:
        for field in ['data', 'electrodes', 'time']:
            if field not in stim:
                raise AttributeError("Stimulus dict must contain a field "
                                     "'%s'." % field)
        data_shape = stim['data'].shape
        if data_shape[0] > 0 and stim['data'].ndim != 2:
            raise ValueError("Stimulus data must be a 2-D NumPy array, not "
                             "%d-D." % stim['data'].ndim)
        n_electrodes = len(stim['electrodes'])
        if n_electrodes != data_shape[0]:
            raise ValueError("Number of electrodes (%d) must match the number "
                             "of rows in the data array "
                             "(%d)." % (n_electrodes, data_shape[0]))
        if stim['time'] is not None:
            n_time = len(stim['time'])
            if n_time != data_shape[1]:
                raise ValueError("Number of time points (%d) must match the "
                                 "number of columns in the data array "
                                 "(%d)." % (n_time, data_shape[1]))
            if not is_strictly_increasing(stim['time'], tol=0.95*DT):
                msg = ("Time points must be strictly monotonically ",
                       "increasing: %s" % list(stim['time']))
                warnings.warn(msg)
        elif data_shape[0] > 0:
            if data_shape[1] > 1:
                raise ValueError("Number of columns in the data array must be "
                                 "1 if time=None.")

    @property
    def _stim(self):
        """A dictionary containing all the stimulus data"""
        return self.__stim

    @_stim.setter
    def _stim(self, stim):
        self._check_stim(stim)
        # All checks passed, store the data:
        self.__stim = stim

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

    @property
    def is_compressed(self):
        """Flag indicating whether the stimulus has been compressed"""
        return self._is_compressed

    @is_compressed.setter
    def is_compressed(self, val):
        """This flag can only be set in ``compress``"""
        # getframe(0) is 'is_compressed'
        # getframe(1) is the one we are looking for:
        f_caller = _getframe(1).f_code.co_name
        if f_caller in ["__init__", "compress"]:
            self._is_compressed = val
        else:
            err_s = ("The attribute `is_compressed` can only be set in the "
                     "constructor or in `compress`, not in `%s`." % f_caller)
            raise AttributeError(err_s)

    @property
    def dt(self):
        """Sampling time step (ms)

        Defines the duration of the signal edge transitions.

        .. versionadded:: 0.7

        """
        return DT

    @property
    def is_charge_balanced(self):
        """Flag indicating whether the stimulus is charge-balanced

        A stimulus with a time component is considered charge-balanced if its
        net current is smaller than 10 pico Amps.
        For the whole stimulus to be charge-balanced, every electrode must be
        charge-balanced as well.
        """
        if self.time is None:
            return np.allclose(self.data, 0, atol=MIN_AMP)
        return np.allclose(np.trapz(self.data, x=self.time), 0, atol=MIN_AMP)

    @property
    def duration(self):
        """Stimulus duration (ms)"""
        return self.time[-1]
