""":py:class:`~pulse2percept.utils.PrettyPrint`,
   :py:class:`~pulse2percept.utils.Frozen`,
   :py:class:`~pulse2percept.utils.Parametrized`,
   :py:class:`~pulse2percept.utils.Data`,
   :py:class:`~pulse2percept.utils.bijective26_name`,
   :py:class:`~pulse2percept.utils.cached`,
   :py:class:`~pulse2percept.utils.gamma`,
   :py:class:`~pulse2percept.utils.unique`"""
import numpy as np
import sys
import abc
from copy import copy, deepcopy
from scipy.integrate import trapezoid
from scipy.special import factorial
from collections import OrderedDict as ODict
from functools import wraps
from string import ascii_uppercase

from .deprecation import warn_deprecated_params


class PrettyPrint(object, metaclass=abc.ABCMeta):
    """PrettyPrint

    An abstract class that provides a way to prettyprint all class attributes,
    inspired by scikit-learn.

    Classes deriving from PrettyPrint are required to implement a
    ``_pprint_params`` method that returns a dictionary containing all the
    attributes to prettyprint.

    Examples
    --------
    >>> from pulse2percept.utils import PrettyPrint
    >>> class MyClass(PrettyPrint):
    ...     def __init__(self, a, b):
    ...         self.a = a
    ...         self.b = b
    ...
    ...     def _pprint_params(self):
    ...         return {'a': self.a, 'b': self.b}
    >>> MyClass(1, 2)
    MyClass(a=1, b=2)

    """
    __slots__ = ()

    @abc.abstractmethod
    def _pprint_params(self):
        """Return a dictionary of class attributes"""
        raise NotImplementedError

    def __repr__(self):
        """Pretty print class as: ClassName(arg1=val1, arg2=val2)"""
        # Line width:
        lwidth = 60
        # Sort list of parameters alphabetically:
        sorted_params = ODict(sorted(self._pprint_params().items()))
        # Start string with class name, followed by all arguments:
        str_params = self.__class__.__name__ + '('
        # New line indent (align with class name on first line):
        lindent = len(str_params)
        # Keep track of number of chars on current line:
        lc = len(str_params)
        for key, val in sorted_params.items():
            # Attribute string:
            if isinstance(val, str):
                # Need extra '' around strings for repr:
                sparam = key + '=\'' + str(val) + '\', '
            else:
                if isinstance(val, np.ndarray):
                    # Print NumPy arrays without line breaks. Pass the
                    # shortening options to `array2string` rather than calling
                    # `np.set_printoptions`, which would change how arrays
                    # print globally for anyone who reprs a p2p object.
                    strobj = np.array2string(val, precision=3, threshold=7,
                                             edgeitems=3).replace('\n', ',')
                    # If still too long, show shape:
                    if len(strobj) > lwidth - lindent:
                        strobj = f'<{str(val.shape)} np.ndarray>'
                else:
                    strobj = str(val)
                    is_type = isinstance(val, type)
                    if len(strobj) > lwidth - lindent:
                        # Too long, just show the type:
                        strobj = str(type(val))
                        is_type = True
                    if is_type:
                        # of the form <class 'X'>, only retain X:
                        strobj = strobj.replace("<class '", "")
                        strobj = strobj.replace("'>", "")
                        # For X.Y.Z, only retain Z:
                        strobj = strobj.split('.')[-1]
                sparam = key + '=' + strobj + ', '
            # If adding `sparam` puts line over `lwidth`, start a new line:
            if lc + len(sparam) > lwidth:
                # But only do so if this is not the first param to be added
                # (check last character of previously written string):
                if str_params[-1] != '(':
                    str_params += '\n' + ' ' * lindent
                    lc = lindent
            str_params += sparam
            lc += len(sparam)
        if len(sorted_params) > 0:
            # Delete last comma:
            str_params = str_params[:-2]
        # Add ')':
        str_params += ')'
        return str_params


class FreezeError(AttributeError):
    """Exception class used to raise when trying to add attributes to Frozen
    Classes of type Frozen do not allow for new attributes to be set outside
    the constructor.
    """


def freeze_class(set):
    """Freezes a class
    Raise an error when trying to set an undeclared name, or when calling from
    a method other than ``Frozen.__init__`` or the ``__init__`` method of a
    class derived from Frozen
    """

    def set_attr(self, name, value):
        if hasattr(self, name):
            # If attribute already exists, simply set it
            set(self, name, value)
            return
        elif sys._getframe(1).f_code.co_name == '__init__' or \
             sys._getframe(2).f_code.co_name == '__init__':
            # Allow __setattr__ calls in __init__ calls of proper object types
            if isinstance(sys._getframe(1).f_locals['self'], self.__class__):
                set(self, name, value)
                return
        err_str = (f"'{name}' not found. You cannot add attributes "
                   f"to {self.__class__.__name__} outside the constructor.")
        raise FreezeError(err_str)
    return set_attr


class Frozen(object):
    """Frozen
    "Frozen" classes (and subclasses) do not allow for new class attributes to
    be set outside the constructor. On attempting to add a new attribute, the
    class will raise a FreezeError.
    """
    __slots__ = ()

    __setattr__ = freeze_class(object.__setattr__)

    class __metaclass__(type):
        __setattr__ = freeze_class(type.__setattr__)


class Parametrized(Frozen, PrettyPrint, metaclass=abc.ABCMeta):
    """Abstract base class for objects with user-settable parameters

    Provides the following functionality:

    *  Pretty-print class attributes (via ``_pprint_params`` and
       ``PrettyPrint``)
    *  User-settable parameters must be listed in ``get_default_params``
    *  New class attributes can only be added in the constructor
       (enforced via ``Frozen`` and ``FreezeError``)
    *  Value-based equality and deep copying that understand NumPy arrays

    .. versionadded:: 0.10.0

    """

    # Parameters that are still accepted, but that nothing reads any more.
    # Maps a name to the ``deprecate_parameter`` describing it; subclasses
    # override this (see ``SpatialModel``).
    _deprecated_params = {}

    def __init__(self, **params):
        """Parametrized constructor

        Parameters
        ----------
        **params : optional keyword arguments
            All keyword arguments must be listed in ``get_default_params``
        """
        # Set all default arguments:
        defaults = self.get_default_params()
        for key, val in defaults.items():
            setattr(self, key, val)
        # Warn on the ones the user named explicitly, before they are set:
        # applying a default must stay silent.
        warn_deprecated_params(type(self).__name__, params,
                               self._deprecated_params)
        # Then overwrite any arguments also given in `params`:
        for key, val in params.items():
            if key in defaults:
                setattr(self, key, val)
            else:
                err_str = (f"'{key}' is not a valid parameter. Choose "
                           f"from: {', '.join(defaults.keys())}.")
                raise AttributeError(err_str)

    @abc.abstractmethod
    def get_default_params(self):
        """Return a dict of user-settable parameters"""
        raise NotImplementedError

    def set_params(self, **params):
        """Set the parameters of this object"""
        warn_deprecated_params(type(self).__name__, params,
                               self._deprecated_params)
        for key, value in params.items():
            setattr(self, key, value)

    def _pprint_params(self):
        """Return a dict of class attributes to display when pretty-printing"""
        return {key: getattr(self, key)
                for key, _ in self.get_default_params().items()}

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        if id(self) in memodict:
            return memodict[id(self)]
        copied = copy(self)
        # Register before recursing, and pass `memodict` down, so that shared
        # references are copied once and reference cycles terminate:
        memodict[id(self)] = copied
        for attr in self.__dict__:
            copied.__setattr__(attr,
                               deepcopy(self.__getattribute__(attr), memodict))
        return copied

    def __eq__(self, other):
        """
        Equality operator for Parametrized.

        Parameters
        ----------
        other: Parametrized
            Object to compare against

        Returns
        -------
        bool:
            True if the compared objects have identical attributes, False
            otherwise.
        """
        if not isinstance(other, self.__class__):
            return False
        if id(self) == id(other):
            return True
        if self.__dict__.keys() != other.__dict__.keys():
            return False
        for key in self.__dict__.keys():
            mine, theirs = self.__dict__[key], other.__dict__[key]
            # If either side is an array, compare as arrays: `!=` between an
            # array and anything else is elementwise, and the result cannot be
            # coerced to a bool.
            if isinstance(mine, np.ndarray) or isinstance(theirs, np.ndarray):
                if not np.array_equal(mine, theirs):
                    return False
            elif mine != theirs:
                return False
        return True

    def __hash__(self):
        # Default python 2.6+ implementation
        return id(self) // 16


class Data(PrettyPrint):
    """N-dimensional data container

    .. versionadded:: 0.6

    Parameters
    ----------
    data : np.ndarray
        An N-dimensional NumPy array containing the data to store
    axes : dict or tuple, optional
        For each dimension in ``data``, specify axis name and labels.
    metadata : dict, optional
        A dictionary that can store arbitrary metadata

    """

    def __init__(self, data, axes=None, metadata=None):
        self._internal = {
            'data': data,
            'axes': axes,
            'metadata': metadata
        }

    def _pprint_params(self):
        """Return a dictionary of class attributes to pretty-print"""
        return {key: getattr(self, key)
                for key in self._internal['pprint_params']}

    @property
    def _internal(self):
        """Return the internal data structure"""
        return self.__internal

    @_internal.setter
    def _internal(self, source):
        # Error check
        data = np.asarray(source['data'])
        if data.ndim == 0:
            # Convert scalar to 1-dim array:
            data = np.array([data])
        if source['axes'] is None:
            # Automatic axis labels and values: 'axis0', 'axis1', etc.
            axes = ODict([(f'axis{d}', np.arange(data.shape[d]))
                          for d in np.arange(data.ndim)])
        else:
            # Build an ordered dictionary from the provided axis labels/values
            # and make sure it lines up with the dimensions of the NumPy array:
            try:
                axes = ODict(source['axes'])
            except TypeError:
                raise TypeError("'axes' must be either an ordered dictionary "
                                "or a list of tuples (label, values).")
            if len(axes) != data.ndim:
                raise ValueError(f"Number of axis labels ({len(axes)}) does not match "
                                 f"number of dimensions in the NumPy array "
                                 f" ({data.ndim}).")
            if len(np.unique(list(axes.keys()))) < data.ndim:
                raise ValueError("All axis labels must be unique.")
            for i, (key, values) in enumerate(axes.items()):
                if values is None:
                    if data.shape[i] > 1:
                        # If there's 1 data point, then None is None. If
                        # there's > 1 data points, it's an omitted axis we need
                        # to fill in:
                        axes[key] = np.arange(data.shape[i])
                        continue
                else:
                    if data.shape[i] == 1 and np.isscalar(values):
                        values = np.array([values])
                    if len(values) != data.shape[i]:
                        err_str = (f"Number of values for axis '{key}' ({len(values)}) does "
                                   f"not match data.shape[{i}] "
                                   f"({data.shape[i]})")
                        raise ValueError(err_str)
                    axes[key] = values

        # Create a property for each of the following:
        pprint_params = ['data', 'dtype', 'shape', 'metadata']
        for param in pprint_params:
            setattr(self.__class__, param,
                    property(fget=self._fget_prop(param)))

        # Also add axis labels as properties:
        for axis, values in axes.items():
            setattr(self.__class__, axis, property(fget=self._fget_axes(axis)))
        pprint_params += list(axes.keys())

        # Internal data structure is a dictionary that stores the actual data
        # container as an N-dim array alongside axis labels and metadata.
        # Setting all elements at once enforces consistency; e.g. between shape
        # and axes:
        self.__internal = {
            'data': data,
            'dtype': data.dtype,
            'shape': data.shape,
            'axes': axes,
            'metadata': source['metadata'],
            'pprint_params': pprint_params
        }

    def _fget_prop(self, name):
        """Generic property getter"""

        def fget(self):
            try:
                return self._internal[name]
            except KeyError as e:
                raise AttributeError(e)
        return fget

    def _fget_axes(self, name):
        """Axis property getter"""

        def fget(self):
            try:
                return self._internal['axes'][name]
            except KeyError as e:
                raise AttributeError(e)
        return fget


def gamma(n, tau, tsample, tol=0.01):
    """Returns the impulse response of ``n`` cascaded leaky integrators

    This function calculates the impulse response of ``n`` cascaded
    leaky integrators with constant of proportionality 1/``tau``:
    y = (t/theta).^(n-1).*exp(-t/theta)/(theta*factorial(n-1))

    Parameters
    ----------
    n : int
        Number of cascaded leaky integrators
    tau : float
        Decay constant of leaky integration (seconds).
        Equivalent to the inverse of the constant of proportionality.
    tsample : float
        Sampling time step (seconds).
    tol : float
        Cut the kernel to size by ignoring function values smaller
        than a fraction ``tol`` of the peak value.
    """
    n = int(n)
    tau = float(tau)
    tsample = float(tsample)
    if n <= 0 or tau <= 0 or tsample <= 0:
        raise ValueError("`n`, `tau`, and `tsample` must be nonnegative.")
    if tau <= tsample:
        raise ValueError("`tau` cannot be smaller than `tsample`.")

    # Allocate a time vector that is long enough for sure.
    # Trim vector later on.
    t = np.arange(0, 5 * n * tau, tsample)

    # Calculate gamma
    y = (t / tau) ** (n - 1) * np.exp(-t / tau)
    y /= (tau * factorial(n - 1))

    # Normalize to unit area
    y /= trapezoid(np.abs(y), dx=tsample)

    # Cut off tail where values are smaller than `tol`.
    # Make sure to start search on the right-hand side of the peak.
    peak = y.argmax()
    small_vals = np.where(y[peak:] < tol * y.max())[0]
    if small_vals.size:
        t = t[:small_vals[0] + peak]
        y = y[:small_vals[0] + peak]

    return t, y


def cached(f):
    """Cached property decorator

    Decorator can be added to the property of a class to maintain a cache.
    This is useful when computing the property is computationall expensive.
    The property will only be computed on first call, and subsequent calls will
    refer to the cached result.

    .. important ::

        When making use of a cached property, the class should also maintain
        a ``_cache_active`` flag set to True or False.

    .. versionadded:: 0.7

    """
    @wraps(f)
    def wrapper(obj):
        cache = obj._cache
        prop = f.__name__

        if not ((prop in cache) and obj._cache_active):
            cache[prop] = f(obj)

        return cache[prop]

    return wrapper


def bijective26_name(i):
    """Bijective base-26 numeration

    Creates the "alphabetic number" for a given integer i following bijective
    base-26 numeration: A-Z, AA-AZ, BA-BZ, ... ZA-ZZ, AAA-AAZ, ABA-ABZ, ...

    Parameters
    ----------
    i : int
        Regular number to be translated into an alphabetic number

    Returns
    -------
    name : string
        Alphabetic number

    Examples
    --------

    >>> bijective26_name(0)
    'A'

    >>> bijective26_name(26)
    'AA'

    """
    n_ascii = len(ascii_uppercase)
    repeat = i // n_ascii
    letter = i % n_ascii
    name = ""
    if repeat > 0:
        name = bijective26_name(repeat - 1)
    return f"{name}{ascii_uppercase[letter]}"