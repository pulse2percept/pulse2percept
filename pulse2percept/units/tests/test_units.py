import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.units import (Dimension, Unit, Quantity,
                                 DimensionMismatchError, as_value,
                                 dimensionless, DIMENSIONLESS,
                                 s, ms, us, ns, Hz, kHz, m, cm, mm, um, nm,
                                 A, mA, uA, nA, V, mV, uV, C, mC, uC, nC, dva)


def test_Dimension():
    npt.assert_equal(Dimension(), Dimension())
    npt.assert_equal(Dimension().is_dimensionless, True)
    npt.assert_equal(Dimension(time=1).is_dimensionless, False)
    # Multiplication, division, powers:
    npt.assert_equal(Dimension(current=1) * Dimension(time=1),
                     Dimension(current=1, time=1))
    npt.assert_equal(Dimension(length=1) / Dimension(time=1),
                     Dimension(length=1, time=-1))
    npt.assert_equal(Dimension(length=1) ** 2, Dimension(length=2))
    npt.assert_equal(Dimension(length=1) ** 0, Dimension())
    npt.assert_equal(Dimension(time=1) / Dimension(time=1), Dimension())
    # Equality is by exponent vector, and dimensions are hashable:
    npt.assert_equal(Dimension(time=1) != Dimension(length=1), True)
    npt.assert_equal(len({Dimension(time=1), Dimension(time=1)}), 1)
    # Names:
    npt.assert_equal(Dimension(time=1).name, 'time')
    npt.assert_equal(Dimension(current=1).name, 'electric current')
    npt.assert_equal(Dimension(visual_angle=1).name, 'visual angle')
    npt.assert_equal(Dimension(time=-1).name, 'frequency')
    npt.assert_equal(Dimension(current=1, time=1).name, 'charge')
    npt.assert_equal(Dimension(current=1, length=-2).name, 'current density')
    npt.assert_equal(Dimension().name, 'dimensionless')
    npt.assert_equal(Dimension(voltage=1, time=2).name, 'time^2 * voltage')
    # Immutable:
    with pytest.raises(AttributeError):
        Dimension(time=1).exponents = (1, 0, 0, 0, 0)
    # Only known base dimensions, only integer exponents:
    with pytest.raises(ValueError):
        Dimension(temperature=1)
    with pytest.raises(ValueError):
        Dimension(time=0.5)
    with pytest.raises(ValueError):
        Dimension(time=1) ** 0.5


def test_Unit():
    npt.assert_equal(ms.dimension, Dimension(time=1))
    npt.assert_equal(ms.scale, 1e-3)
    npt.assert_equal(ms.symbol, 'ms')
    npt.assert_equal(str(uA / mm ** 2), 'uA/mm^2')
    npt.assert_equal(str(uA * ms), 'uA*ms')
    npt.assert_equal(str(mm ** 2), 'mm^2')
    npt.assert_equal(str(ms / (uA * ms)), 'ms/(uA*ms)')
    # Unit algebra produces units, not quantities:
    for unit in (uA * ms, uA / mm ** 2, mm ** 2, Hz * s):
        npt.assert_equal(isinstance(unit, Unit), True)
    # Derived units line up with the predefined ones:
    npt.assert_equal(uA * ms, nC)
    npt.assert_equal(1 / s == Hz, True)
    npt.assert_equal(mA * s, mC)
    npt.assert_equal(uA / mm ** 2 == mA / m ** 2, False)
    npt.assert_equal((uA / mm ** 2).dimension, Dimension(current=1, length=-2))
    # Scales are exact powers of ten even after unit algebra:
    npt.assert_equal((uA * ms).scale, 1e-9)
    npt.assert_equal((um ** 2).scale, 1e-12)
    npt.assert_equal((uA / mm ** 2).scale, 1.0)
    # Immutable:
    with pytest.raises(AttributeError):
        ms.scale = 1
    with pytest.raises(TypeError):
        Unit('time', 1, 's')
    with pytest.raises(ValueError):
        ms ** 0.5


def test_unit_vocabulary():
    # Every exported unit has the expected scale relative to its base unit:
    for unit, scale, dim in [(s, 1, 'time'), (ms, 1e-3, 'time'),
                             (us, 1e-6, 'time'), (ns, 1e-9, 'time'),
                             (Hz, 1, 'frequency'), (kHz, 1e3, 'frequency'),
                             (m, 1, 'length'), (cm, 1e-2, 'length'),
                             (mm, 1e-3, 'length'), (um, 1e-6, 'length'),
                             (nm, 1e-9, 'length'),
                             (A, 1, 'electric current'),
                             (mA, 1e-3, 'electric current'),
                             (uA, 1e-6, 'electric current'),
                             (nA, 1e-9, 'electric current'),
                             (V, 1, 'voltage'), (mV, 1e-3, 'voltage'),
                             (uV, 1e-6, 'voltage'),
                             (C, 1, 'charge'), (mC, 1e-3, 'charge'),
                             (uC, 1e-6, 'charge'), (nC, 1e-9, 'charge'),
                             (dva, 1, 'visual angle')]:
        npt.assert_almost_equal(unit.scale, scale)
        npt.assert_equal(unit.dimension.name, dim)
    npt.assert_equal(dimensionless.dimension, DIMENSIONLESS)
    npt.assert_equal(dimensionless.symbol, '')


def test_Quantity():
    q = 50 * uA
    npt.assert_equal(isinstance(q, Quantity), True)
    npt.assert_equal(q.magnitude, 50)
    npt.assert_equal(q.unit, uA)
    npt.assert_equal(q.dimension, Dimension(current=1))
    npt.assert_equal(str(q), '50 uA')
    npt.assert_equal(repr(q), '50 uA')
    npt.assert_equal(str(5 * dimensionless), '5')
    # Both operand orders work:
    npt.assert_equal(uA * 50, 50 * uA)
    # Immutable, unhashable (the magnitude may be a mutable array):
    with pytest.raises(AttributeError):
        q.magnitude = 3
    with pytest.raises(TypeError):
        {q: 'x'}
    with pytest.raises(TypeError):
        Quantity(1, 'ms')
    with pytest.raises(TypeError):
        Quantity(50 * uA, ms)


def test_Quantity_conversion():
    # Equivalent unit choices are numerically identical, not merely close:
    npt.assert_equal(500 * uA == 0.5 * mA, True)
    npt.assert_equal(1000 * ms == 1 * s, True)
    npt.assert_equal((500 * uA).to_value(mA), 0.5)
    npt.assert_equal((0.5 * mA).to_value(uA), 500.0)
    npt.assert_equal((0.02 * s).to_value(ms), 20.0)
    npt.assert_equal((450 * us).to_value(ms), 0.45)
    npt.assert_equal((15 * mm).to_value(um), 15000.0)
    npt.assert_equal((1 * kHz).to_value(Hz), 1000.0)
    # to() keeps the unit, to_value() strips it:
    converted = (500 * uA).to(mA)
    npt.assert_equal(isinstance(converted, Quantity), True)
    npt.assert_equal(converted.unit, mA)
    npt.assert_equal(converted.magnitude, 0.5)
    npt.assert_equal(str(converted), '0.5 mA')
    # Converting to the same unit is a no-op:
    npt.assert_equal((50 * uA).to_value(uA), 50)
    # Derived dimensions:
    npt.assert_equal(1 * uA * ms == 1 * nC, True)
    npt.assert_equal((1 * uA * ms).to_value(nC), 1.0)
    npt.assert_equal((1 * mA * s).to_value(uC), 1000.0)
    npt.assert_equal((2 * uA / mm ** 2).to_value(mA / m ** 2), 2000.0)
    with pytest.raises(DimensionMismatchError):
        (5 * uA).to(ms)
    with pytest.raises(TypeError):
        (5 * uA).to(3)


def test_Quantity_arithmetic():
    npt.assert_equal(500 * uA + 0.5 * mA, 1000 * uA)
    npt.assert_equal(500 * uA - 0.1 * mA, 400 * uA)
    npt.assert_equal((500 * uA + 0.5 * mA).unit, uA)
    npt.assert_equal(0.5 * mA + 500 * uA, 1 * mA)
    npt.assert_equal(2 * (500 * uA), 1 * mA)
    npt.assert_equal((500 * uA) * 2, 1 * mA)
    npt.assert_equal((500 * uA) / 2, 250 * uA)
    npt.assert_equal(-(500 * uA), -0.5 * mA)
    npt.assert_equal(abs(-500 * uA), 0.5 * mA)
    npt.assert_equal(+(500 * uA), 500 * uA)
    # Multiplication and division build derived dimensions:
    npt.assert_equal((2 * uA) * (3 * ms), 6 * nC)
    npt.assert_equal((6 * nC) / (3 * ms), 2 * uA)
    npt.assert_equal((10 * uA) / (2 * mm ** 2), 5 * uA / mm ** 2)
    npt.assert_equal((2 * mm) ** 2, 4 * mm ** 2)
    npt.assert_equal(1 / (2 * ms), 0.5 / ms)
    npt.assert_equal((1 * s) / ms, 1000 * dimensionless)
    # Adding incompatible dimensions fails:
    with pytest.raises(DimensionMismatchError):
        (5 * uA) + (2 * ms)
    with pytest.raises(DimensionMismatchError):
        (5 * uA) - (2 * ms)
    # As does mixing a quantity with a bare number:
    with pytest.raises(TypeError):
        (5 * uA) + 2
    with pytest.raises(TypeError):
        2 - (5 * uA)
    # Except when the quantity is dimensionless:
    npt.assert_equal(2 * dimensionless + 3, 5 * dimensionless)
    npt.assert_equal(3 - 2 * dimensionless, 1 * dimensionless)


def test_Quantity_comparison():
    npt.assert_equal(500 * uA > 0.1 * mA, True)
    npt.assert_equal(500 * uA >= 0.5 * mA, True)
    npt.assert_equal(500 * uA < 1 * mA, True)
    npt.assert_equal(500 * uA <= 0.5 * mA, True)
    npt.assert_equal(500 * uA != 0.5 * mA, False)
    npt.assert_equal(500 * uA != 0.4 * mA, True)
    # A quantity is never equal to something of another dimension, but does
    # not raise: equality is asked, not asserted.
    npt.assert_equal(500 * uA == 500 * ms, False)
    npt.assert_equal(500 * uA == 500, False)
    npt.assert_equal(500 * uA == 'foo', False)
    # Ordering, on the other hand, is meaningless across dimensions:
    with pytest.raises(DimensionMismatchError):
        (5 * uA) < (2 * ms)
    with pytest.raises(DimensionMismatchError):
        (5 * uA) < 2


def test_Quantity_arrays():
    # A list or array times a unit is ONE quantity wrapping an array, not an
    # array of quantities:
    for magnitude in ([1, 2], (1, 2), np.array([1, 2])):
        q = magnitude * uA
        npt.assert_equal(isinstance(q, Quantity), True)
        npt.assert_equal(isinstance(q.magnitude, np.ndarray), True)
        npt.assert_equal(q.magnitude.dtype != object, True)
        npt.assert_almost_equal(q.magnitude, [1, 2])
    # Conversion is elementwise and returns a plain array:
    values = ([500, 1000] * uA).to_value(mA)
    npt.assert_equal(isinstance(values, np.ndarray), True)
    npt.assert_almost_equal(values, [0.5, 1.0])
    npt.assert_almost_equal((np.array([1., 2.]) * s).to_value(ms), [1000, 2000])
    # Comparisons are elementwise:
    npt.assert_equal([500, 1000] * uA == [0.5, 1.0] * mA, [True, True])
    npt.assert_equal([500, 1000] * uA > 0.6 * mA, [False, True])
    # 2D magnitudes survive too:
    q = np.ones((2, 3)) * mA
    npt.assert_equal(q.magnitude.shape, (2, 3))
    npt.assert_almost_equal(q.to_value(uA), 1000 * np.ones((2, 3)))


def test_no_silent_unit_stripping():
    # np.asarray must not quietly turn a quantity into its magnitude. It is
    # allowed to produce a useless object array; it is not allowed to produce
    # a float array that has forgotten the unit.
    npt.assert_equal(np.asarray(5 * uA).dtype, object)
    npt.assert_equal(np.asarray([1, 2] * uA).dtype, object)
    # Nor may a quantity sneak into a ufunc:
    with pytest.raises(TypeError):
        np.sqrt(4 * uA)
    with pytest.raises(TypeError):
        np.array([1.0, 2.0]) + (5 * uA)
    # Stripping is spelled out instead:
    npt.assert_equal((5 * uA).to_value(uA), 5)


def test_dva_is_not_a_length():
    # A visual field map owns the dva <-> distance relationship; it is a
    # coordinate transformation, not a unit conversion.
    with pytest.raises(DimensionMismatchError):
        (5 * dva).to(mm)
    with pytest.raises(DimensionMismatchError):
        (5 * dva).to_value(um)
    with pytest.raises(DimensionMismatchError):
        (5 * dva) + (5 * mm)
    with pytest.raises(DimensionMismatchError):
        as_value(5 * dva, um)
    npt.assert_equal(5 * dva == 5 * mm, False)
    # dva is still a perfectly good unit on its own:
    npt.assert_equal((5 * dva).to_value(dva), 5)
    npt.assert_equal(dva.dimension.name, 'visual angle')


def test_as_value():
    # Bare numbers are assumed to already be in the target unit:
    npt.assert_equal(as_value(20, ms), 20)
    npt.assert_equal(as_value(0, ms), 0)
    npt.assert_equal(as_value(None, ms), None)
    npt.assert_almost_equal(as_value([1, 2], uA), [1, 2])
    # Quantities are checked and rescaled:
    npt.assert_equal(as_value(0.020 * s, ms), 20)
    npt.assert_equal(as_value(20 * ms, ms), 20)
    npt.assert_equal(as_value(0.05 * mA, uA), 50)
    npt.assert_almost_equal(as_value([500, 1000] * uA, mA), [0.5, 1.0])
    # A bare unit is the quantity 1:
    npt.assert_equal(as_value(ms, ms), 1)
    npt.assert_equal(as_value(s, ms), 1000)
    # Mismatches name the offending parameter:
    with pytest.raises(DimensionMismatchError):
        as_value(3 * uA, ms)
    with pytest.raises(DimensionMismatchError) as excinfo:
        as_value(3 * uA, ms, 'dt')
    npt.assert_equal("Parameter 'dt' expects time (ms), got electric current "
                     "(uA)." in str(excinfo.value), True)
    with pytest.raises(DimensionMismatchError) as excinfo:
        as_value(3 * uA, ms)
    npt.assert_equal("Expected time (ms), got electric current (uA)."
                     in str(excinfo.value), True)


def test_DimensionMismatchError():
    # Catchable as a TypeError, because that is what it is:
    npt.assert_equal(issubclass(DimensionMismatchError, TypeError), True)
    with pytest.raises(TypeError):
        as_value(3 * uA, ms)
