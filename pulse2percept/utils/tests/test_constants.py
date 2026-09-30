import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.units import Hz, Quantity, mm, ms, s, um
from pulse2percept.utils.constants import (DT, MIN_AMP, MS_PER_S, UM_PER_MM,
                                           ZORDER)


def test_zorder():
    npt.assert_equal(ZORDER['foreground'] > ZORDER['background'], True)
    npt.assert_equal(ZORDER['annotate'] > ZORDER['foreground'], True)

# These are silly, but hey - completeness


def test_min_amp():
    npt.assert_equal(MIN_AMP < 1e-4, True)


def test_dt():
    npt.assert_equal(DT < 1e-2, True)


def test_conversion_factors():
    """MS_PER_S and UM_PER_MM are exactly 1000.0, derived from the unit
    system"""
    npt.assert_equal(MS_PER_S, 1000.0)
    npt.assert_equal(UM_PER_MM, 1000.0)
    npt.assert_equal(MS_PER_S == Quantity(1, s).to_value(ms), True)
    npt.assert_equal(UM_PER_MM == Quantity(1, mm).to_value(um), True)
    # Plain floats, so inner loops get no Quantity:
    npt.assert_equal(isinstance(MS_PER_S, float), True)
    npt.assert_equal(isinstance(UM_PER_MM, float), True)
    # Hz to ms period, and um to mm:
    npt.assert_allclose(MS_PER_S / 50, (1 / (50 * Hz)).to_value(ms),
                        rtol=1e-12)
    npt.assert_allclose(15000 / UM_PER_MM, Quantity(15000, um).to_value(mm),
                        rtol=1e-12)
