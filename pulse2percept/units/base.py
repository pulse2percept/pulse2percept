""":py:class:`~pulse2percept.units.Dimension`,
   :py:class:`~pulse2percept.units.Unit`,
   :py:class:`~pulse2percept.units.Quantity`,
   :py:class:`~pulse2percept.units.DimensionMismatchError`,
   :py:func:`~pulse2percept.units.as_value`"""
import math
from copy import deepcopy
import numpy as np

# Angle, visual angle and threshold ratio are three distinct dimensions: an
# ordinary geometric angle is a pure ratio, whereas converting visual angle or
# threshold ratio requires model- or subject-specific calibration.
BASE_DIMENSIONS = ('time', 'length', 'current', 'voltage', 'angle',
                   'visual_angle', 'threshold_ratio')

# Human-readable names used in error messages:
_BASE_LABELS = {
    'time': 'time',
    'length': 'length',
    'current': 'electric current',
    'voltage': 'voltage',
    'angle': 'angle',
    'visual_angle': 'visual angle',
    'threshold_ratio': 'threshold ratio',
}

# Names for common derived dimensions:
_DERIVED_LABELS = {
    (0, 0, 0, 0, 0, 0, 0): 'dimensionless',
    (-1, 0, 0, 0, 0, 0, 0): 'frequency',
    (1, 0, 1, 0, 0, 0, 0): 'charge',
    (0, -2, 1, 0, 0, 0, 0): 'current density',
    (1, -2, 1, 0, 0, 0, 0): 'charge density',
}


def _snap_scale(scale):
    """Snap near-exact decimal scale factors to powers of ten.

    This removes floating-point noise from equivalent compound units.
    """
    if not math.isfinite(scale) or scale <= 0:
        return scale
    rounded = round(math.log10(scale))
    if abs(rounded) < 300:
        candidate = float(f'1e{rounded:d}')
        if math.isclose(scale, candidate, rel_tol=1e-12, abs_tol=0.0):
            return candidate
    return scale


#: Relative tolerance used when comparing converted quantity magnitudes.
_EQ_RTOL = 1e-12


def _isclose(a, b):
    """Compare magnitudes up to floating-point conversion noise.

    Return ``NotImplemented`` for values that cannot be compared
    numerically.
    """
    try:
        result = np.isclose(a, b, rtol=_EQ_RTOL, atol=0.0)
    except (TypeError, ValueError):
        return NotImplemented
    # Scalars in, scalar out; arrays stay elementwise:
    return bool(result) if np.ndim(result) == 0 else result


class DimensionMismatchError(TypeError):
    """Raised when quantities have incompatible physical dimensions.

    .. versionadded:: 0.10.0
    """


def _mismatch(expected, got, name=None):
    """Build a :py:class:`DimensionMismatchError` for an API boundary"""
    got_str = f'{got.dimension.name} ({got})'
    if expected.dimension.is_dimensionless and not expected.symbol:
        # "expects dimensionless ()" reads badly, and a parameter that takes a
        # plain number is worth saying so about directly:
        if name is None:
            return DimensionMismatchError(f"Expected a plain number, got "
                                          f"{got_str}.")
        return DimensionMismatchError(f"Parameter '{name}' is dimensionless, "
                                      f"got {got_str}.")
    exp_str = f'{expected.dimension.name} ({expected})'
    if name is None:
        return DimensionMismatchError(f"Expected {exp_str}, got {got_str}.")
    return DimensionMismatchError(f"Parameter '{name}' expects {exp_str}, got "
                                  f"{got_str}.")


class Dimension(object):
    """Physical dimensionality of a unit or quantity.

    Dimensions are immutable vectors of integer exponents over
    ``BASE_DIMENSIONS``.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    **exponents : int
        Exponents of the primitive dimensions. Omitted dimensions have
        exponent zero.

    Examples
    --------
    >>> from pulse2percept.units import Dimension
    >>> Dimension(current=1) * Dimension(time=1)
    Dimension('charge')
    """
    __slots__ = ('_exponents',)

    def __init__(self, **exponents):
        exps = [0] * len(BASE_DIMENSIONS)
        for key, exp in exponents.items():
            if key not in BASE_DIMENSIONS:
                raise ValueError(f"Unknown base dimension '{key}'. Choose "
                                 f"from: {', '.join(BASE_DIMENSIONS)}.")
            if int(exp) != exp:
                raise ValueError(f"Exponent for '{key}' must be an integer, "
                                 f"not {exp}.")
            exps[BASE_DIMENSIONS.index(key)] = int(exp)
        object.__setattr__(self, '_exponents', tuple(exps))

    def __setattr__(self, name, value):
        raise AttributeError("Dimension objects are immutable.")

    @property
    def exponents(self):
        """Tuple of exponents, aligned with ``BASE_DIMENSIONS``"""
        return self._exponents

    @property
    def is_dimensionless(self):
        """Whether all exponents are zero"""
        return not any(self._exponents)

    @property
    def name(self):
        """Human-readable name, e.g. ``'electric current'``"""
        try:
            return _DERIVED_LABELS[self._exponents]
        except KeyError:
            pass
        num, den = [], []
        for base, exp in zip(BASE_DIMENSIONS, self._exponents):
            if exp == 0:
                continue
            label = _BASE_LABELS[base]
            if abs(exp) > 1:
                label = f'{label}^{abs(exp)}'
            (num if exp > 0 else den).append(label)
        name = ' * '.join(num) if num else '1'
        if den:
            name += ' / ' + ' / '.join(den)
        return name

    def _combine(self, other, sign):
        if not isinstance(other, Dimension):
            return NotImplemented
        exps = {base: a + sign * b
                for base, a, b in zip(BASE_DIMENSIONS, self._exponents,
                                      other._exponents)}
        return Dimension(**exps)

    def __mul__(self, other):
        return self._combine(other, 1)

    def __truediv__(self, other):
        return self._combine(other, -1)

    def __pow__(self, exp):
        if int(exp) != exp:
            raise ValueError(f"Dimensions can only be raised to integer "
                             f"powers, not {exp}.")
        return Dimension(**{base: e * int(exp)
                            for base, e in zip(BASE_DIMENSIONS,
                                               self._exponents)})

    def __eq__(self, other):
        if not isinstance(other, Dimension):
            return NotImplemented
        return self._exponents == other._exponents

    def __ne__(self, other):
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def __hash__(self):
        return hash(self._exponents)

    # Immutable value objects can be shared across copies.
    def __copy__(self):
        return self

    def __deepcopy__(self, memodict=None):
        return self

    # Restore state without going through the immutability guard.
    def __getstate__(self):
        return {'_exponents': self._exponents}

    def __setstate__(self, state):
        for key, value in state.items():
            object.__setattr__(self, key, value)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Dimension('{self.name}')"


#: The dimension of a plain number
DIMENSIONLESS = Dimension()

#: Canonical display symbols, keyed by the ``(dimension, scale)`` pair that
#: :py:meth:`Unit.__eq__` compares. Filled in at the bottom of this module,
#: once the predefined units exist.
_CANONICAL_SYMBOLS = {}


def _symbol_mul(a, b):
    """Compose the symbol of a product of two units"""
    if not a:
        return b
    if not b:
        return a
    return f'{a}*{b}'


def _symbol_group(sym):
    """Parenthesize a compound unit symbol when needed for composition."""
    if any(c in sym for c in '*/'):
        return f'({sym})'
    return sym


class Unit(object):
    """A physical unit.

    A unit combines a :class:`Dimension`, a scale relative to its base
    unit, and a display symbol. Multiplying a value by a unit produces a
    :class:`Quantity`; units may also be multiplied, divided, and raised
    to integer powers.

    Units are immutable.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    dimension : :class:`~pulse2percept.units.Dimension`
        Physical dimension.
    scale : float
        Scale relative to the base unit of that dimension.
    symbol : str
        Display symbol, e.g. ``'uA'``.
    """
    __slots__ = ('_dimension', '_scale', '_symbol')

    # NumPy must not try to broadcast a unit into an object array: with these
    # two attributes, ``np.array([1, 2]) * uA`` defers to ``Unit.__rmul__``
    # and produces a single Quantity wrapping the array.
    __array_priority__ = 1000
    __array_ufunc__ = None

    def __init__(self, dimension, scale, symbol):
        if not isinstance(dimension, Dimension):
            raise TypeError(f"'dimension' must be a Dimension object, not "
                            f"{type(dimension)}.")
        scale = float(scale)
        # Unit scales must be positive and finite.
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"'scale' must be a positive, finite number, not "
                             f"{scale}.")
        object.__setattr__(self, '_dimension', dimension)
        object.__setattr__(self, '_scale', _snap_scale(scale))
        object.__setattr__(self, '_symbol', str(symbol))

    def __setattr__(self, name, value):
        raise AttributeError("Unit objects are immutable.")

    @property
    def dimension(self):
        """The :py:class:`~pulse2percept.units.Dimension` of this unit"""
        return self._dimension

    @property
    def scale(self):
        """Size of this unit relative to the base unit of its dimension"""
        return self._scale

    @property
    def symbol(self):
        """Short symbol this unit was built with, e.g. ``'uA*ms'``"""
        return self._symbol

    @property
    def _display_symbol(self):
        """Display symbol, preferring an exactly equivalent predefined unit."""
        return _CANONICAL_SYMBOLS.get((self._dimension, self._scale),
                                      self._symbol)

    def __mul__(self, other):
        if isinstance(other, Unit):
            return Unit(self.dimension * other.dimension,
                        self.scale * other.scale,
                        _symbol_mul(self.symbol, other.symbol))
        if isinstance(other, Quantity):
            return Quantity(other.magnitude, self * other.unit)
        return Quantity(other, self)

    def __rmul__(self, other):
        # Reached by ``5 * uA``, ``[1, 2] * uA``, and ``np.array([1, 2]) * uA``:
        return Quantity(other, self)

    def __truediv__(self, other):
        if isinstance(other, Unit):
            symbol = self.symbol or '1'
            return Unit(self.dimension / other.dimension,
                        self.scale / other.scale,
                        f'{symbol}/{_symbol_group(other.symbol)}')
        if isinstance(other, Quantity):
            return Quantity(1.0 / other.magnitude, self / other.unit)
        return Quantity(1.0 / other, self)

    def __rtruediv__(self, other):
        # Reached by ``1 / ms`` and ``20 / ms``:
        return Quantity(other, self ** -1)

    def __pow__(self, exp):
        if int(exp) != exp:
            raise ValueError(f"Units can only be raised to integer powers, "
                             f"not {exp}.")
        exp = int(exp)
        symbol = f'{_symbol_group(self.symbol)}^{exp}' if self.symbol else ''
        return Unit(self.dimension ** exp, self.scale ** exp, symbol)

    def __eq__(self, other):
        # Exact, because `_snap_scale` has already canonicalized the scale:
        # `uA * ms` and `nC` are the same float, not merely close ones. Two
        # units are equal if they measure the same thing at the same size,
        # whatever they are spelled.
        if not isinstance(other, Unit):
            return NotImplemented
        return (self.dimension == other.dimension and
                self.scale == other.scale)

    def __ne__(self, other):
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def __hash__(self):
        # Hashes exactly what `__eq__` compares:
        return hash((self.dimension, self.scale))

    # Immutable, so a copy is the object itself; see `Dimension.__copy__`.
    # This matters here: every stimulus carries two units, and the model
    # pipeline deep-copies stimuli constantly.
    def __copy__(self):
        return self

    def __deepcopy__(self, memodict=None):
        return self

    def __getstate__(self):
        return {'_dimension': self._dimension, '_scale': self._scale,
                '_symbol': self._symbol}

    def __setstate__(self, state):
        for key, value in state.items():
            object.__setattr__(self, key, value)

    def __str__(self):
        return self._display_symbol

    def __repr__(self):
        return self._display_symbol if self._display_symbol else 'dimensionless'


class Quantity(object):
    """A number (or array of numbers) with a unit

    Quantities are what users build by multiplying a number by a unit, and
    they exist to be checked and converted at p2p's public API boundaries.
    They are deliberately *not* NumPy arrays: p2p strips units before any
    numerical work, so quantities never reach a Cython kernel and never
    impose per-element overhead on a simulation.

    For the same reason, ``np.asarray(5 * uA)`` does not silently yield ``5``.
    Removing a unit is something you write down, using
    :py:meth:`~pulse2percept.units.Quantity.to_value`.

    Equivalent unit choices convert consistently up to floating-point
    precision, and quantities compare accordingly: ``0.0041 * mA == 4.1 * uA``
    is True even though rescaling the former gives ``4.1000000000000005``.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    magnitude : float or array_like
        The numerical value(s), expressed in ``unit``.
    unit : :py:class:`~pulse2percept.units.Unit`
        The unit of ``magnitude``.

    Examples
    --------
    >>> from pulse2percept.units import uA, mA
    >>> 500 * uA == 0.5 * mA
    True
    >>> (500 * uA).to(mA)
    0.5 mA
    >>> (500 * uA).to_value(mA)
    0.5

    """
    __slots__ = ('_magnitude', '_unit')

    __array_priority__ = 1000
    __array_ufunc__ = None

    # There is deliberately no ``__array__``: it would make
    # ``np.asarray(5 * uA)`` return ``5``, which is exactly the silent unit
    # stripping this class exists to prevent.
    #
    # ``__len__``, ``__getitem__`` and ``__iter__`` are absent for a weaker
    # reason: nothing needs them yet. They could be added safely -- indexing
    # would have to return another Quantity, never a bare number, or NumPy
    # would strip units through the sequence protocol instead.

    def __init__(self, magnitude, unit):
        if not isinstance(unit, Unit):
            raise TypeError(f"'unit' must be a Unit object, not {type(unit)}.")
        if isinstance(magnitude, Quantity):
            raise TypeError("'magnitude' must be a number or array, not a "
                            "Quantity. Use quantity * unit instead.")
        if isinstance(magnitude, (list, tuple)):
            magnitude = np.asarray(magnitude, dtype=float)
        object.__setattr__(self, '_magnitude', magnitude)
        object.__setattr__(self, '_unit', unit)

    def __setattr__(self, name, value):
        raise AttributeError("Quantity objects are immutable.")

    @property
    def magnitude(self):
        """The numerical value(s), expressed in ``self.unit``"""
        return self._magnitude

    @property
    def unit(self):
        """The :py:class:`~pulse2percept.units.Unit` of this quantity"""
        return self._unit

    @property
    def dimension(self):
        """The :py:class:`~pulse2percept.units.Dimension` of this quantity"""
        return self._unit.dimension

    def to(self, unit, name=None):
        """Convert to another unit of the same dimension

        Parameters
        ----------
        unit : :py:class:`~pulse2percept.units.Unit`
            The target unit.
        name : str, optional
            Name of the parameter being converted, used to make the error
            message point at the offending argument.

        Returns
        -------
        quantity : :py:class:`~pulse2percept.units.Quantity`
            The same physical quantity expressed in ``unit``.

        """
        return Quantity(self.to_value(unit, name=name), unit)

    def to_value(self, unit, name=None):
        """Convert to another unit and return the bare number(s)

        This is the explicit way to remove a unit: after calling it you have
        an ordinary float or NumPy array, expressed in ``unit``.

        Parameters
        ----------
        unit : :py:class:`~pulse2percept.units.Unit`
            The target unit.
        name : str, optional
            Name of the parameter being converted, used to make the error
            message point at the offending argument.

        Returns
        -------
        value : float or np.ndarray
            The magnitude of this quantity expressed in ``unit``.

        """
        if isinstance(unit, Quantity):
            raise TypeError("Convert to a Unit, not a Quantity.")
        if not isinstance(unit, Unit):
            raise TypeError(f"'unit' must be a Unit object, not {type(unit)}.")
        if self.unit.dimension != unit.dimension:
            raise _mismatch(unit, self.unit, name=name)
        if self.unit.scale == unit.scale:
            return self.magnitude
        return self.magnitude * _snap_scale(self.unit.scale / unit.scale)

    def _check_compatible(self, other, verb):
        if self.dimension != other.dimension:
            raise DimensionMismatchError(
                f"Cannot {verb} {self.unit.dimension.name} ({self.unit}) and "
                f"{other.unit.dimension.name} ({other.unit}).")

    # A bare number combined with a dimensionless quantity is a quantity in the
    # canonical `dimensionless` unit -- not "the magnitude in whatever compound
    # dimensionless unit this one happens to carry". The distinction is invisible
    # for `dimensionless` itself (scale 1) and decisive for anything composed:
    # a duty cycle of ``0.45 * ms * 50 * Hz`` is 0.0225, though its ms*Hz
    # magnitude reads 22.5. So these branches rescale before they touch a
    # number, and hand back a result in `dimensionless`.
    def __add__(self, other):
        if isinstance(other, Unit):
            other = Quantity(1, other)
        if not isinstance(other, Quantity):
            if self.dimension.is_dimensionless:
                return Quantity(self.to_value(dimensionless) + other,
                                dimensionless)
            return NotImplemented
        self._check_compatible(other, 'add')
        return Quantity(self.magnitude + other.to_value(self.unit), self.unit)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Unit):
            other = Quantity(1, other)
        if not isinstance(other, Quantity):
            if self.dimension.is_dimensionless:
                return Quantity(self.to_value(dimensionless) - other,
                                dimensionless)
            return NotImplemented
        self._check_compatible(other, 'subtract')
        return Quantity(self.magnitude - other.to_value(self.unit), self.unit)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __mul__(self, other):
        if isinstance(other, Unit):
            return Quantity(self.magnitude, self.unit * other)
        if isinstance(other, Quantity):
            return Quantity(self.magnitude * other.magnitude,
                            self.unit * other.unit)
        return Quantity(self.magnitude * other, self.unit)

    def __rmul__(self, other):
        if isinstance(other, Unit):
            return Quantity(self.magnitude, other * self.unit)
        return Quantity(other * self.magnitude, self.unit)

    def __truediv__(self, other):
        if isinstance(other, Unit):
            return Quantity(self.magnitude, self.unit / other)
        if isinstance(other, Quantity):
            return Quantity(self.magnitude / other.magnitude,
                            self.unit / other.unit)
        return Quantity(self.magnitude / other, self.unit)

    def __rtruediv__(self, other):
        if isinstance(other, Unit):
            return Quantity(1.0 / self.magnitude, other / self.unit)
        return Quantity(other / self.magnitude, self.unit ** -1)

    def __pow__(self, exp):
        return Quantity(self.magnitude ** exp, self.unit ** exp)

    def __neg__(self):
        return Quantity(-self.magnitude, self.unit)

    def __pos__(self):
        return self

    def __abs__(self):
        return Quantity(abs(self.magnitude), self.unit)

    def _compare(self, other, op, verb):
        if isinstance(other, Unit):
            other = Quantity(1, other)
        if not isinstance(other, Quantity):
            if self.dimension.is_dimensionless:
                return op(self.to_value(dimensionless), other)
            raise DimensionMismatchError(
                f"Cannot {verb} {self.unit.dimension.name} ({self.unit}) and "
                f"a plain number. Multiply the number by a unit first.")
        self._check_compatible(other, verb)
        return op(self.magnitude, other.to_value(self.unit))

    def __lt__(self, other):
        return self._compare(other, lambda a, b: a < b, 'compare')

    def __le__(self, other):
        return self._compare(other, lambda a, b: a <= b, 'compare')

    def __gt__(self, other):
        return self._compare(other, lambda a, b: a > b, 'compare')

    def __ge__(self, other):
        return self._compare(other, lambda a, b: a >= b, 'compare')

    def __eq__(self, other):
        # Compared up to floating-point conversion noise: `0.0041 * mA` and
        # `4.1 * uA` are the same current, but rescaling the first one yields
        # 4.1000000000000005. See `_EQ_RTOL`.
        if isinstance(other, Unit):
            other = Quantity(1, other)
        if not isinstance(other, Quantity):
            if self.dimension.is_dimensionless:
                return _isclose(self.to_value(dimensionless), other)
            # Equality must not raise: a quantity simply is not equal to a
            # bare number of a different dimension.
            return NotImplemented
        if self.dimension != other.dimension:
            return False
        return _isclose(self.magnitude, other.to_value(self.unit))

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return np.logical_not(result) if isinstance(result, np.ndarray) \
            else not result

    # Quantities wrap mutable magnitudes, so they are not hashable:
    __hash__ = None

    def __copy__(self):
        return Quantity(self.magnitude, self.unit)

    def __deepcopy__(self, memodict=None):
        # Unlike a Dimension or a Unit, the magnitude may be a mutable array,
        # so this one really does have something to copy:
        return Quantity(deepcopy(self.magnitude, memodict), self.unit)

    def __getstate__(self):
        return {'_magnitude': self._magnitude, '_unit': self._unit}

    def __setstate__(self, state):
        for key, value in state.items():
            object.__setattr__(self, key, value)

    def __str__(self):
        symbol = self.unit._display_symbol
        if not symbol:
            return f'{self.magnitude}'
        return f'{self.magnitude} {symbol}'

    def __repr__(self):
        return self.__str__()


def as_value(value, unit, name=None):
    """Convert a value to a bare number expressed in ``unit``

    This is p2p's standard Python-to-numerics boundary. A
    :py:class:`~pulse2percept.units.Quantity` is dimension-checked and
    rescaled to ``unit``; a bare number is assumed to already be expressed in
    ``unit`` and is passed through untouched (including ``None``).

    Parameters
    ----------
    value : float, array_like, Quantity, or None
        The value to normalize.
    unit : :py:class:`~pulse2percept.units.Unit`
        The unit the numerical code expects.
    name : str, optional
        Name of the parameter, used to make the error message point at the
        offending argument.

    Returns
    -------
    value : float, np.ndarray, or None
        The bare numerical value, expressed in ``unit``.

    Examples
    --------
    >>> from pulse2percept.units import as_value, ms, s
    >>> as_value(20, ms)
    20
    >>> as_value(0.02 * s, ms)
    20.0

    """
    # Unconditionally, before the bare-value fast path: a bad ``unit`` is a bug
    # in the calling API, and it must not go unnoticed just because this
    # particular caller happened to pass a plain number.
    if not isinstance(unit, Unit):
        raise TypeError(f"'unit' must be a Unit object, not {type(unit)}.")
    if isinstance(value, Unit):
        # A bare unit is the quantity 1 of that unit, e.g. ``as_value(ms, ms)``:
        value = Quantity(1, value)
    if isinstance(value, Quantity):
        return value.to_value(unit, name=name)
    if isinstance(value, (list, tuple)) and has_units(value):
        # A sequence built one element at a time, e.g. `(-15 * dva, 15 * dva)`
        # for a parameter that takes an (x_min, x_max) pair. Converted
        # elementwise, keeping the sequence type, so the caller still gets the
        # tuple it expects:
        return type(value)(as_value(v, unit, name=name) for v in value)
    return value


def has_units(value):
    """Whether a value carries a physical unit

    True for a :py:class:`~pulse2percept.units.Quantity` or
    :py:class:`~pulse2percept.units.Unit`, and for a list or tuple containing
    one. Cheap enough to call before every attribute assignment, which is what
    :py:class:`~pulse2percept.utils.Parametrized` does.
    """
    if isinstance(value, (Quantity, Unit)):
        return True
    if isinstance(value, (list, tuple)):
        return any(isinstance(v, (Quantity, Unit)) for v in value)
    return False


# -----------------------------------------------------------------------------
# The public unit vocabulary
#
# Deliberately small and neuroscience-oriented: these are the units that appear
# in p2p's own APIs, spelled the terse Brian way. There is no registry, no
# string parsing, and no automatic prefix generation; anything else users need
# they build with unit algebra (e.g. ``uA / mm ** 2``).
# -----------------------------------------------------------------------------

TIME = Dimension(time=1)
LENGTH = Dimension(length=1)
CURRENT = Dimension(current=1)
VOLTAGE = Dimension(voltage=1)
ANGLE = Dimension(angle=1)
VISUAL_ANGLE = Dimension(visual_angle=1)
THRESHOLD_RATIO = Dimension(threshold_ratio=1)
FREQUENCY = TIME ** -1
CHARGE = CURRENT * TIME

#: The unit of a plain number, used for image intensities and other
#: dimensionless data
dimensionless = Unit(DIMENSIONLESS, 1, '')

#: Second
s = Unit(TIME, 1, 's')
#: Millisecond
ms = Unit(TIME, 1e-3, 'ms')
#: Microsecond
us = Unit(TIME, 1e-6, 'us')
#: Nanosecond
ns = Unit(TIME, 1e-9, 'ns')

#: Hertz
Hz = Unit(FREQUENCY, 1, 'Hz')
#: Kilohertz
kHz = Unit(FREQUENCY, 1e3, 'kHz')

#: Meter
m = Unit(LENGTH, 1, 'm')
#: Centimeter
cm = Unit(LENGTH, 1e-2, 'cm')
#: Millimeter
mm = Unit(LENGTH, 1e-3, 'mm')
#: Micrometer (micron)
um = Unit(LENGTH, 1e-6, 'um')
#: Nanometer
nm = Unit(LENGTH, 1e-9, 'nm')

#: Ampere
A = Unit(CURRENT, 1, 'A')
#: Milliampere
mA = Unit(CURRENT, 1e-3, 'mA')
#: Microampere
uA = Unit(CURRENT, 1e-6, 'uA')
#: Nanoampere
nA = Unit(CURRENT, 1e-9, 'nA')

#: Volt
V = Unit(VOLTAGE, 1, 'V')
#: Millivolt
mV = Unit(VOLTAGE, 1e-3, 'mV')
#: Microvolt
uV = Unit(VOLTAGE, 1e-6, 'uV')

#: Coulomb
C = Unit(CHARGE, 1, 'C')
#: Millicoulomb
mC = Unit(CHARGE, 1e-3, 'mC')
#: Microcoulomb
uC = Unit(CHARGE, 1e-6, 'uC')
#: Nanocoulomb
nC = Unit(CHARGE, 1e-9, 'nC')

#: Radian, the base scale of ordinary angle
rad = Unit(ANGLE, 1, 'rad')
#: Degree of ordinary (geometric) angle. p2p's angle-valued APIs are spelled in
#: degrees, so this is the unit their bare numbers are read in.
deg = Unit(ANGLE, np.pi / 180, 'deg')

#: Degree of visual angle. Not an ordinary angle: converting dva to a distance
#: on the retina or cortex requires a visual field map, not a scale factor, so
#: ``dva`` and ``deg`` are deliberately incompatible.
dva = Unit(VISUAL_ANGLE, 1, 'dva')

#: Multiple of perceptual threshold ("times threshold"). Becomes a current
#: only once a threshold is known; see
#: :py:class:`~pulse2percept.stimuli.BiphasicPulseTrain`.
xTh = Unit(THRESHOLD_RATIO, 1, 'xTh')


# The canonical spelling of each predefined unit, for `Unit._display_symbol`.
# Written out here rather than harvested from this module's namespace: should
# two units ever share a (dimension, scale) -- an alias such as ``sec`` for
# ``s`` -- the one listed here is the one that wins, and that ought to be a
# decision rather than an accident of declaration order. An alias simply does
# not go in this tuple; listing both is a mistake, so it raises rather than
# letting declaration order pick a winner.
_CANONICAL_UNITS = (dimensionless,
                    s, ms, us, ns,
                    Hz, kHz,
                    m, cm, mm, um, nm,
                    A, mA, uA, nA,
                    V, mV, uV,
                    C, mC, uC, nC,
                    rad, deg,
                    dva, xTh)

for _unit in _CANONICAL_UNITS:
    _key = (_unit.dimension, _unit.scale)
    if _key in _CANONICAL_SYMBOLS:
        raise RuntimeError(
            f"Two canonical units claim {_unit.dimension.name} at scale "
            f"{_unit.scale:g}: '{_CANONICAL_SYMBOLS[_key]}' and "
            f"'{_unit.symbol}'. Only one spelling can be canonical: drop the "
            f"alias from _CANONICAL_UNITS.")
    _CANONICAL_SYMBOLS[_key] = _unit.symbol
del _unit, _key
