import copyreg
import pickle
from copy import copy, deepcopy

import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.units import (Dimension, Unit, Quantity,
                                 DimensionMismatchError, as_value,
                                 dimensionless,
                                 s, ms, us, ns, Hz, kHz, m, cm, mm, um, nm,
                                 A, mA, uA, nA, V, mV, uV, C, mC, uC, nC, deg,
                                 rad, dva, xTh)
from pulse2percept.units.base import (TIME, _CANONICAL_SYMBOLS,
                                      _CANONICAL_UNITS)


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
    npt.assert_equal(Dimension(angle=1).name, 'angle')
    npt.assert_equal(Dimension(visual_angle=1).name, 'visual angle')
    npt.assert_equal(Dimension(time=-1).name, 'frequency')
    npt.assert_equal(Dimension(current=1, time=1).name, 'charge')
    npt.assert_equal(Dimension(current=1, length=-2).name, 'current density')
    npt.assert_equal(Dimension(current=1, time=1, length=-2).name,
                     'charge density')
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
    npt.assert_equal(str(uA * ms), 'nC')  # exactly nC; see test_canonical_display
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
    # Equality is exact (scales are canonical) and consistent with the hash:
    npt.assert_equal(len({uA * ms, nC}), 1)
    npt.assert_equal(hash(s ** -1) == hash(Hz), True)
    npt.assert_equal(Unit(TIME, 1e-3, 'msec') == ms, True)
    npt.assert_equal(Unit(TIME, 1.5e-3, 'x') == ms, False)
    # Immutable:
    with pytest.raises(AttributeError):
        ms.scale = 1
    with pytest.raises(TypeError):
        Unit('time', 1, 's')
    with pytest.raises(ValueError):
        ms ** 0.5
    # A unit's scale is how big it is: positive and finite, or it is not a unit
    for bad_scale in (0, -1, np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            Unit(TIME, bad_scale, 'bad')


def test_canonical_display():
    # A composed unit that is EXACTLY a predefined one is spelled that way,
    # because it is that unit and not merely convertible to it:
    npt.assert_equal(str(uA * ms), 'nC')
    npt.assert_equal(repr(uA * ms), 'nC')
    npt.assert_equal(str(200 * uA * 0.45 * ms), '90.0 nC')
    npt.assert_equal(repr(1 / ms), '1 kHz')
    npt.assert_equal(str(mA * s), 'mC')
    npt.assert_equal(str(s ** -1), 'Hz')
    npt.assert_equal(str(1 * ms / ms), '1')  # dimensionless has no symbol
    # ... and the magnitude is never touched, so a composed unit that is not
    # exactly a predefined one keeps its composed spelling rather than being
    # rescaled into one. `.to()` is the only thing that changes scale.
    # Canonicalization happens at display time, so the pieces a compound symbol
    # is built from are the ones it was composed from, not their canonical
    # spellings: uA*ms/um^2, not nC/um^2.
    charge_density = (200 * uA * 0.45 * ms) / (200 * um) ** 2
    npt.assert_equal(str(charge_density), '0.00225 uA*ms/um^2')
    npt.assert_equal(str(charge_density.to(uC / mm ** 2)), '2.25 uC/mm^2')
    npt.assert_equal(str(uA / mm ** 2), 'uA/mm^2')
    npt.assert_equal(str(uA * ms / mm ** 2), 'uA*ms/mm^2')
    # The lookup is on the same (dimension, scale) pair that __eq__ compares,
    # so display agrees with equality:
    for composed, predefined in [(uA * ms, nC), (s ** -1, Hz), (mA * s, mC),
                                 (ms ** -1, kHz), (A / A, dimensionless)]:
        npt.assert_equal(composed == predefined, True)
        npt.assert_equal(str(composed), str(predefined))
    # `symbol` still reports how the unit was built; only display is canonical:
    npt.assert_equal((uA * ms).symbol, 'uA*ms')
    # An alias is not canonical unless it is listed as such:
    npt.assert_equal(str(Unit(TIME, 1e-3, 'msec')), 'ms')


def test_canonical_units_are_unambiguous():
    # Each canonical unit claims a distinct (dimension, scale), so no predefined
    # unit's spelling is decided by declaration order:
    npt.assert_equal(len(_CANONICAL_SYMBOLS), len(_CANONICAL_UNITS))
    for unit in _CANONICAL_UNITS:
        npt.assert_equal(_CANONICAL_SYMBOLS[(unit.dimension, unit.scale)],
                         unit.symbol)


def test_snap_scale():
    # Unit algebra lands exactly on the predefined units, rather than one ulp
    # away from them:
    npt.assert_equal((uA * ms).scale == nC.scale, True)
    npt.assert_equal((um ** 3).scale == 1e-18, True)
    npt.assert_equal((mA * s / uA).scale == 1e3, True)
    # But a scale that is merely near a power of ten is left alone:
    npt.assert_equal(Unit(TIME, 1.0000001, 'almost').scale, 1.0000001)


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
                             (rad, 1, 'angle'), (deg, np.pi / 180, 'angle'),
                             (dva, 1, 'visual angle'),
                             (xTh, 1, 'threshold ratio')]:
        npt.assert_almost_equal(unit.scale, scale)
        npt.assert_equal(unit.dimension.name, dim)
    npt.assert_equal(dimensionless.dimension, Dimension())
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
    # Equivalent unit choices convert consistently:
    npt.assert_equal(500 * uA == 0.5 * mA, True)
    npt.assert_equal(1000 * ms == 1 * s, True)
    # ... including when the conversion cannot be exact in binary floating
    # point: 0.0041 * 1000 is 4.1000000000000005, not 4.1.
    npt.assert_equal(0.0041 * mA == 4.1 * uA, True)
    npt.assert_equal((0.0041 * mA).to_value(uA) == 4.1, False)
    npt.assert_equal(0.1 * s + 0.05 * s == 150 * ms, True)
    npt.assert_equal([0.0041, 0.0082] * mA == [4.1, 8.2] * uA, [True, True])
    # But a difference in the 10th significant digit is a real difference:
    npt.assert_equal(4.1 * uA == 4.100000001 * uA, False)
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
    # Including when the quantity is dimensionless, and the comparison would
    # otherwise be handed straight to np.isclose:
    npt.assert_equal(5 * dimensionless == 'foo', False)
    for nothing in (None,):
        npt.assert_equal(500 * uA == nothing, False)
        npt.assert_equal(5 * dimensionless == nothing, False)
    npt.assert_equal(5 * dimensionless != 'foo', True)
    npt.assert_equal(5 * dimensionless == 5, True)
    # Ordering, on the other hand, is meaningless across dimensions:
    with pytest.raises(DimensionMismatchError):
        (5 * uA) < (2 * ms)
    with pytest.raises(DimensionMismatchError):
        (5 * uA) < 2


def test_dimensionless_compound_units():
    # A bare number combined with a dimensionless quantity means a quantity in
    # the canonical `dimensionless` unit -- NOT the magnitude in whatever
    # compound dimensionless unit the quantity happens to carry. The two differ
    # for every compound whose scale is not 1, which is most of them, and
    # `5 * dimensionless` is exactly the case that hides the difference.
    duty = 0.45 * ms * 50 * Hz  # a duty cycle: 0.0225, spelled 22.5 ms*Hz
    npt.assert_equal(duty.magnitude, 22.5)
    npt.assert_equal(duty.to_value(dimensionless), 0.0225)
    npt.assert_equal(duty == 0.0225, True)
    npt.assert_equal(duty == 22.5, False)
    npt.assert_equal(duty < 0.05, True)
    npt.assert_equal(duty > 1.0, False)
    npt.assert_equal(duty <= 0.0225, True)
    npt.assert_equal(duty >= 0.0225, True)
    npt.assert_equal(duty != 0.0225, False)
    npt.assert_equal(duty + 1 == 1.0225, True)
    npt.assert_equal(1 + duty == 1.0225, True)
    npt.assert_equal(duty - 1 == -0.9775, True)
    npt.assert_equal(1 - duty == 0.9775, True)  # __rsub__ takes its own path
    ratio = (1 * s) / ms
    npt.assert_equal(ratio.magnitude, 1)
    npt.assert_equal(ratio == 1000, True)
    npt.assert_equal(ratio > 999, True)
    npt.assert_equal(ratio < 1001, True)
    npt.assert_equal(ratio + 1 == 1001, True)
    npt.assert_equal(1 + ratio == 1001, True)
    npt.assert_equal(ratio - 1 == 999, True)
    npt.assert_equal(1 - ratio == -999, True)
    # The result of mixing in a bare number is in `dimensionless`, so it does
    # not silently inherit a compound spelling:
    npt.assert_equal((duty + 1).unit, dimensionless)
    npt.assert_equal((1 - duty).unit, dimensionless)
    # Quantity-to-quantity comparison already converted, and still does:
    npt.assert_equal(duty == 0.0225 * dimensionless, True)
    npt.assert_equal(ratio == 1000 * dimensionless, True)


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


def test_deg_and_rad_are_ordinary_angles():
    # Radians are the base scale, so the two convert by a plain factor:
    npt.assert_almost_equal((180 * deg).to_value(rad), np.pi)
    npt.assert_almost_equal((np.pi * rad).to_value(deg), 180)
    npt.assert_equal(180 * deg == np.pi * rad, True)
    npt.assert_equal(str(45 * deg), '45 deg')
    # An ordinary angle is not a visual angle, and neither converts to the
    # other or to anything else:
    npt.assert_equal(deg.dimension == dva.dimension, False)
    npt.assert_equal(45 * deg == 45 * dva, False)
    for bad in (dva, um, ms, dimensionless):
        with pytest.raises(DimensionMismatchError):
            (45 * deg).to_value(bad)
        with pytest.raises(DimensionMismatchError):
            (1 * rad).to_value(bad)
        with pytest.raises(DimensionMismatchError):
            as_value(1 * bad, deg)
    with pytest.raises(DimensionMismatchError):
        (45 * deg) + (45 * dva)


def test_xTh_is_not_dimensionless():
    # A multiple of threshold is not a plain number: turning it into a current
    # takes a calibration, so nothing may convert between the two silently.
    npt.assert_equal(xTh == dimensionless, False)
    npt.assert_equal(2 * xTh == 2, False)
    with pytest.raises(DimensionMismatchError):
        (2 * xTh).to_value(uA)
    with pytest.raises(DimensionMismatchError):
        as_value(2 * xTh, dimensionless)
    with pytest.raises(DimensionMismatchError):
        (2 * xTh) + (2 * uA)
    # It is still a perfectly good unit on its own:
    npt.assert_equal((2 * xTh).to_value(xTh), 2)
    npt.assert_equal(str(2 * xTh), '2 xTh')
    npt.assert_equal(xTh.dimension.name, 'threshold ratio')


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
    # A dimensionless target says so rather than showing an empty symbol:
    with pytest.raises(DimensionMismatchError) as excinfo:
        as_value(3 * uA, dimensionless, 'thresh_percept')
    npt.assert_equal("Parameter 'thresh_percept' is dimensionless, got "
                     "electric current (uA)." in str(excinfo.value), True)
    npt.assert_equal(as_value(3 * dimensionless, dimensionless), 3)
    # A bad target unit is a bug in the calling API, and is caught even when
    # the value passed is a bare number:
    with pytest.raises(TypeError):
        as_value(20, 'ms')
    with pytest.raises(TypeError):
        as_value(20 * ms, 'ms')


def test_DimensionMismatchError():
    # Catchable as a TypeError, because that is what it is:
    npt.assert_equal(issubclass(DimensionMismatchError, TypeError), True)
    with pytest.raises(TypeError):
        as_value(3 * uA, ms)


def test_units_copy_and_pickle():
    # A Dimension and a Unit are immutable value objects: copying one hands
    # back the very same object, which is also what keeps a deep-copied
    # stimulus from allocating units.
    for obj in (Dimension(current=1), uA, uA / mm ** 2, dimensionless):
        npt.assert_equal(copy(obj) is obj, True)
        npt.assert_equal(deepcopy(obj) is obj, True)
    # A Quantity does have something to copy, because its magnitude may be a
    # mutable array:
    q = np.array([1.0, 2.0]) * uA
    for copied in (copy(q), deepcopy(q)):
        npt.assert_equal(copied == q, [True, True])
        npt.assert_equal(copied.unit, uA)
    copied = deepcopy(q)
    copied.magnitude[0] = 99
    npt.assert_almost_equal(q.magnitude, [1.0, 2.0])
    # `copy` shares the magnitude, as a shallow copy should:
    copied = copy(q)
    copied.magnitude[0] = 99
    npt.assert_almost_equal(q.magnitude, [99.0, 2.0])
    # All three survive a pickle round trip. They define __slots__ and refuse
    # ordinary attribute assignment, so this only works because they say how
    # to restore themselves:
    for obj in (Dimension(current=1, length=-2), ms, uA * ms, dimensionless):
        restored = pickle.loads(pickle.dumps(obj))
        npt.assert_equal(restored, obj)
        npt.assert_equal(hash(restored), hash(obj))
    for obj in (5 * uA, [1, 2] * ms, 0.5 * uA / mm ** 2):
        restored = pickle.loads(pickle.dumps(obj))
        npt.assert_equal(np.all(restored == obj), True)
        npt.assert_equal(restored.unit, obj.unit)
    # A Unit restored from a pickle is still usable in unit algebra and in
    # conversions, i.e. its dimension came back intact:
    restored = pickle.loads(pickle.dumps(mA))
    npt.assert_almost_equal((1 * restored).to_value(uA), 1000)
    npt.assert_equal(restored * s, mC)
    npt.assert_equal(restored * ms, uC)


class StaleDimension(object):
    """Pickles as a Dimension whose exponent tuple has the wrong length"""

    def __reduce__(self):
        return (copyreg._reconstructor, (Dimension, object, None),
                {'_exponents': (0, 0, 1, 0, 0, 0)})


def test_Dimension_rejects_stale_pickle():
    # An exponent tuple written against a different set of base dimensions
    # would restore into the wrong dimensions:
    with pytest.raises(ValueError):
        pickle.loads(pickle.dumps(StaleDimension()))

    # A Unit is protected through the Dimension it carries, and so is a
    # Quantity through its Unit:
    class StaleUnit(object):
        def __reduce__(self):
            return (copyreg._reconstructor, (Unit, object, None),
                    {'_dimension': StaleDimension(), '_scale': 1e-6,
                     '_symbol': 'uA'})

    class StaleQuantity(object):
        def __reduce__(self):
            return (copyreg._reconstructor, (Quantity, object, None),
                    {'_magnitude': 5, '_unit': StaleUnit()})

    for stale in (StaleUnit(), StaleQuantity()):
        with pytest.raises(ValueError):
            pickle.loads(pickle.dumps(stale))
