""":py:class:`~pulse2percept.units.Dimension`,
   :py:class:`~pulse2percept.units.Unit`,
   :py:class:`~pulse2percept.units.Quantity`,
   :py:class:`~pulse2percept.units.DimensionMismatchError`,
   :py:func:`~pulse2percept.units.as_value`"""
import math
import numpy as np

# Primitive dimensions. This is deliberately *not* the seven-dimensional SI
# system: p2p only ever needs these five, and every other dimension we care
# about is derived from them (frequency = time^-1, charge = current * time,
# current density = current / length^2).
#
# ``visual_angle`` is a p2p-specific primitive dimension, not an ordinary
# physical angle: converting degrees of visual angle to retinal or cortical
# distance is a coordinate transformation owned by a visual field map, not a
# unit conversion.
BASE_DIMENSIONS = ('time', 'length', 'current', 'voltage', 'visual_angle')

# How the primitive dimensions are spelled in error messages:
_BASE_LABELS = {
    'time': 'time',
    'length': 'length',
    'current': 'electric current',
    'voltage': 'voltage',
    'visual_angle': 'visual angle',
}

# Names for the handful of derived dimensions that have one:
_DERIVED_LABELS = {
    (0, 0, 0, 0, 0): 'dimensionless',
    (-1, 0, 0, 0, 0): 'frequency',
    (1, 0, 1, 0, 0): 'charge',
    (0, -2, 1, 0, 0): 'current density',
}


def _snap_scale(scale):
    """Snap a scale factor to an exact power of ten

    Every unit p2p exposes is a decimal multiple of its base unit, so scale
    factors and conversion ratios are always powers of ten in exact
    arithmetic. In floating point they are not: ``1e-6 * 1e-3`` is
    ``1.0000000000000002e-09``, which would make ``1 * uA * ms == 1 * nC``
    false by one ulp. Snapping restores the exact decimal value, which is what
    keeps equivalent unit choices numerically identical.

    Ratios that are not powers of ten are returned unchanged.
    """
    if not math.isfinite(scale) or scale <= 0:
        return scale
    exponent = math.log10(scale)
    rounded = round(exponent)
    if abs(exponent - rounded) < 1e-9 and abs(rounded) < 300:
        # Build the float from its decimal literal rather than with ``**``, so
        # that the result is the correctly rounded power of ten:
        return float(f'1e{rounded:d}')
    return scale


class DimensionMismatchError(TypeError):
    """Raised when quantities of incompatible dimensions are combined

    Subclasses ``TypeError`` because a dimension mismatch is a type error in
    the physical sense: microamps are simply not a kind of millisecond.

    .. versionadded:: 0.10.0

    """


def _mismatch(expected, got, name=None):
    """Build a :py:class:`DimensionMismatchError` for an API boundary"""
    exp_str = f'{expected.dimension.name} ({expected})'
    got_str = f'{got.dimension.name} ({got})'
    if name is None:
        return DimensionMismatchError(f"Expected {exp_str}, got {got_str}.")
    return DimensionMismatchError(f"Parameter '{name}' expects {exp_str}, got "
                                  f"{got_str}.")


class Dimension(object):
    """Physical dimensionality of a unit or quantity

    A dimension is a vector of integer exponents over the primitive dimensions
    in ``BASE_DIMENSIONS``. Dimensions are immutable, hashable, and support
    multiplication, division, and integer powers.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    **exponents : int
        Exponent for each primitive dimension, e.g. ``Dimension(current=1,
        length=-2)`` for a current density. Omitted dimensions have exponent 0.

    Examples
    --------
    >>> from pulse2percept.units import Dimension
    >>> Dimension(current=1) * Dimension(time=1)
    Dimension('charge')
    >>> Dimension(time=-1).name
    'frequency'

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

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Dimension('{self.name}')"


#: The dimension of a plain number
DIMENSIONLESS = Dimension()


def _symbol_mul(a, b):
    """Compose the symbol of a product of two units"""
    if not a:
        return b
    if not b:
        return a
    return f'{a}*{b}'


def _symbol_group(sym):
    """Parenthesize a compound symbol so it composes unambiguously

    Exponents bind tighter than products and quotients, so ``mm^2`` needs no
    parentheses; ``uA*ms`` does.
    """
    if any(c in sym for c in '*/'):
        return f'({sym})'
    return sym


class Unit(object):
    """A unit of measurement

    A unit is a dimension plus a scale factor relative to the base unit of
    that dimension (seconds, meters, amperes, volts, or degrees of visual
    angle) plus a symbol used for display.

    Multiplying a number, list, or NumPy array by a unit produces a
    :py:class:`~pulse2percept.units.Quantity`. Multiplying, dividing, or
    exponentiating units produces another unit, so derived units such as
    ``uA / mm ** 2`` need not be predefined.

    Units are immutable. p2p does not maintain a unit registry, parse unit
    strings, or generate SI prefixes automatically: the vocabulary exported by
    :py:mod:`pulse2percept.units` is the whole of it.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    dimension : :py:class:`~pulse2percept.units.Dimension`
        The dimensionality of the unit.
    scale : float
        Size of the unit relative to the base unit of its dimension. For
        example, ``ms`` has ``scale=1e-3`` because the base unit of time is
        the second.
    symbol : str
        Short symbol used when printing quantities, e.g. ``'uA'``.

    Examples
    --------
    >>> from pulse2percept.units import uA, mm, ms
    >>> uA / mm ** 2
    uA/mm^2
    >>> 50 * uA
    50 uA

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
        object.__setattr__(self, '_dimension', dimension)
        object.__setattr__(self, '_scale', _snap_scale(float(scale)))
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
        """Short symbol used for display"""
        return self._symbol

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
        if not isinstance(other, Unit):
            return NotImplemented
        return (self.dimension == other.dimension
                and math.isclose(self.scale, other.scale, rel_tol=1e-12))

    def __ne__(self, other):
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def __hash__(self):
        return hash((self.dimension, round(math.log10(self.scale), 9)))

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return self.symbol if self.symbol else 'dimensionless'


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

    # Note the absence of ``__array__``, ``__len__``, ``__getitem__``, and
    # ``__iter__``: any of them would let NumPy quietly turn a Quantity into
    # an array of bare numbers, which is exactly the silent unit stripping
    # this class exists to prevent.

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

    def __add__(self, other):
        if isinstance(other, Unit):
            other = Quantity(1, other)
        if not isinstance(other, Quantity):
            if self.dimension.is_dimensionless:
                return Quantity(self.magnitude + other, self.unit)
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
                return Quantity(self.magnitude - other, self.unit)
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
                return op(self.magnitude, other)
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
        if isinstance(other, Unit):
            other = Quantity(1, other)
        if not isinstance(other, Quantity):
            if self.dimension.is_dimensionless:
                return self.magnitude == other
            # Equality must not raise: a quantity simply is not equal to a
            # bare number of a different dimension.
            return NotImplemented
        if self.dimension != other.dimension:
            return False
        return self.magnitude == other.to_value(self.unit)

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return np.logical_not(result) if isinstance(result, np.ndarray) \
            else not result

    # Quantities wrap mutable magnitudes, so they are not hashable:
    __hash__ = None

    def __str__(self):
        if not self.unit.symbol:
            return f'{self.magnitude}'
        return f'{self.magnitude} {self.unit.symbol}'

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
    if isinstance(value, Unit):
        # A bare unit is the quantity 1 of that unit, e.g. ``as_value(ms, ms)``:
        value = Quantity(1, value)
    if isinstance(value, Quantity):
        return value.to_value(unit, name=name)
    return value


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
VISUAL_ANGLE = Dimension(visual_angle=1)
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

#: Degree of visual angle. Not an ordinary angle: converting dva to a distance
#: on the retina or cortex requires a visual field map, not a scale factor.
dva = Unit(VISUAL_ANGLE, 1, 'dva')
