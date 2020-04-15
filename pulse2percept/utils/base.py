"""`PrettyPrint`, `Frozen`, `gamma`"""
import numpy as np
import sys
import abc
from scipy.special import factorial
from collections import OrderedDict


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
        # Shorten NumPy array output:
        np.set_printoptions(precision=3, threshold=7, edgeitems=3)
        # Line width:
        lwidth = 60
        # Sort list of parameters alphabetically:
        sorted_params = OrderedDict(sorted(self._pprint_params().items()))
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
                    # Print NumPy arrays without line breaks:
                    strobj = np.array2string(val).replace('\n', ',')
                    # If still too long, show shape:
                    if len(strobj) > lwidth - lindent:
                        strobj = '<%s np.ndarray>' % str(val.shape)
                else:
                    strobj = str(val)
                    if len(strobj) > lwidth - lindent:
                        # Too long, just show the type:
                        strobj = str(type(val)).replace("<class '", "")
                        strobj = strobj.replace("'>", "")
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
        elif sys._getframe(1).f_code.co_name == '__init__':
            # Allow __setattr__ calls in __init__ calls of proper object types
            if isinstance(sys._getframe(1).f_locals['self'], self.__class__):
                set(self, name, value)
                return
        raise FreezeError("You cannot add attributes to "
                          "%s" % self.__class__.__name__)
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
    y /= np.trapz(np.abs(y), dx=tsample)

    # Cut off tail where values are smaller than `tol`.
    # Make sure to start search on the right-hand side of the peak.
    peak = y.argmax()
    small_vals = np.where(y[peak:] < tol * y.max())[0]
    if small_vals.size:
        t = t[:small_vals[0] + peak]
        y = y[:small_vals[0] + peak]

    return t, y
