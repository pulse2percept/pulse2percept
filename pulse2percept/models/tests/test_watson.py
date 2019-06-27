import numpy as np
import pytest
import numpy.testing as npt
from pulse2percept import models
from pulse2percept.models.watson import dva2ret, ret2dva


def test_WatsonConversionMixin():
    params = {'xrange': (-2, 2), 'yrange': (-1, 1), 'xystep': 1}
    trafo = models.WatsonConversionMixin()
    npt.assert_almost_equal(trafo.get_tissue_coords(0, 0), 0)
    npt.assert_almost_equal(trafo.get_tissue_coords(6, 6), 1618.53, decimal=2)


def test_WatsonDisplacementMixin():
    trafo = models.WatsonDisplacementMixin()
    with pytest.raises(ValueError):
        trafo._watson_displacement(0, meridian='invalid')
    npt.assert_almost_equal(trafo._watson_displacement(0), 0.4957506)
    npt.assert_almost_equal(trafo._watson_displacement(100), 0)

    # Check the max of the displacement function for the temporal meridian:
    radii = np.linspace(0, 30, 100)
    all_displace = trafo._watson_displacement(radii, meridian='temporal')
    npt.assert_almost_equal(np.max(all_displace), 2.153532)
    npt.assert_almost_equal(radii[np.argmax(all_displace)], 1.8181818)

    # Check the max of the displacement function for the nasal meridian:
    all_displace = trafo._watson_displacement(radii, meridian='nasal')
    npt.assert_almost_equal(np.max(all_displace), 1.9228664)
    npt.assert_almost_equal(radii[np.argmax(all_displace)], 2.1212121)


def test_ret2dva():
    # Below 15mm eccentricity, relationship is linear with slope 3.731
    npt.assert_equal(ret2dva(0.0), 0.0)
    for sign in [-1, 1]:
        for exp in [2, 3, 4]:
            ret = sign * 10 ** exp  # mm
            dva = 3.731 * sign * 10 ** (exp - 3)  # dva
            npt.assert_almost_equal(ret2dva(ret), dva,
                                    decimal=3 - exp)  # adjust precision


def test_dva2ret():
    # Below 50deg eccentricity, relationship is linear with slope 0.268
    npt.assert_equal(dva2ret(0.0), 0.0)
    for sign in [-1, 1]:
        for exp in [-2, -1, 0]:
            dva = sign * 10 ** exp  # deg
            ret = 0.268 * sign * 10 ** (exp + 3)  # mm
            npt.assert_almost_equal(dva2ret(dva), ret,
                                    decimal=-exp)  # adjust precision
