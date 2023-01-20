import copy

import numpy as np
import pytest
import numpy.testing as npt
from matplotlib.axes import Axes

from pulse2percept.utils import (VisualFieldMap, RetinalMap, 
                                 CorticalMap, Curcio1990Map,
                                 Grid2D, Watson2014Map,
                                 Watson2014DisplaceMap,
                                 cart2pol, pol2cart, delta_angle)


@pytest.mark.parametrize('x_range', [(0, 0), (-3, 3), (4, -2), (1, -1)])
@pytest.mark.parametrize('y_range', [(0, 0), (0, 7), (-3, 3), (2, -2)])
def test_Grid2D(x_range, y_range):
    grid = Grid2D(x_range, y_range, step=1, grid_type='rectangular')
    npt.assert_equal(grid.x_range, x_range)
    npt.assert_equal(grid.y_range, y_range)
    npt.assert_equal(grid.step, 1)
    npt.assert_equal(grid.type, 'rectangular')

    # Grid is created with indexing='xy', so check coordinates:
    npt.assert_equal(grid.x.shape,
                     (np.abs(np.diff(y_range)) + 1,
                      np.abs(np.diff(x_range)) + 1))
    npt.assert_equal(grid.x.shape, grid.y.shape)
    npt.assert_equal(grid.x.shape, grid.shape)
    npt.assert_almost_equal(grid.x[0, 0], x_range[0])
    npt.assert_almost_equal(grid.x[0, -1], x_range[1])
    npt.assert_almost_equal(grid.x[-1, 0], x_range[0])
    npt.assert_almost_equal(grid.x[-1, -1], x_range[1])
    npt.assert_almost_equal(grid.y[0, 0], y_range[0])
    npt.assert_almost_equal(grid.y[0, -1], y_range[0])
    npt.assert_almost_equal(grid.y[-1, 0], y_range[1])
    npt.assert_almost_equal(grid.y[-1, -1], y_range[1])


def test_Grid2D_make_rectangular_grid():
    # Range is a multiple of step size:
    grid = Grid2D((-1, 1), (0, 0), step=1)
    npt.assert_almost_equal(grid.x, [[-1, 0, 1]])
    npt.assert_almost_equal(grid.y, [[0, 0, 0]])
    for mlt in [0.01, 0.1, 1, 10, 100]:
        grid = Grid2D((-10 * mlt, 10 * mlt), (-10 * mlt, 10 * mlt),
                      step=5 * mlt)
        npt.assert_almost_equal(grid.x[0], mlt * np.array([-10, -5, 0, 5, 10]))
        npt.assert_almost_equal(grid.y[:, 0],
                                mlt * np.array([-10, -5, 0, 5, 10]))

    # Another way to verify this is to manually check the step size:
    for step in [0.25, 0.5, 1, 2]:
        grid = Grid2D((-20, 20), (-40, 40), step=step)
        npt.assert_equal(len(np.unique(np.diff(grid.x[0, :]))), 1)
        npt.assert_equal(len(np.unique(np.diff(grid.y[:, 0]))), 1)
        npt.assert_almost_equal(np.unique(np.diff(grid.x[0, :]))[0], step)
        npt.assert_almost_equal(np.unique(np.diff(grid.y[:, 0]))[0], step)

    # Step size just a little too small/big to fit into range. In this case,
    # the step size gets adjusted so that the range is as specified by the
    # user:
    grid = Grid2D((-1, 1), (0, 0), step=0.33)
    npt.assert_almost_equal(grid.x, [[-1, -2 / 3, -1 / 3, 0, 1 / 3, 2 / 3, 1]])
    npt.assert_almost_equal(grid.y, [[0, 0, 0, 0, 0, 0, 0]])
    grid = Grid2D((-1, 1), (0, 0), step=0.34)
    npt.assert_almost_equal(grid.x, [[-1, -2 / 3, -1 / 3, 0, 1 / 3, 2 / 3, 1]])
    npt.assert_almost_equal(grid.y, [[0, 0, 0, 0, 0, 0, 0]])

    # Different step size for x and y:
    grid = Grid2D((-1, 1), (0, 0), step=(0.5, 1))
    npt.assert_almost_equal(grid.x, [[-1, -0.5, 0, 0.5, 1]])
    npt.assert_almost_equal(grid.y, [[0, 0, 0, 0, 0]])
    grid = Grid2D((0, 0), (-1, 1), step=(2, 0.5))
    npt.assert_almost_equal(grid.x, [[0], [0], [0], [0], [0]])
    npt.assert_almost_equal(grid.y[:, 0], [-1, -0.5, 0, 0.5, 1])

    # Same step size, but given explicitly:
    npt.assert_almost_equal(Grid2D((-3, 3), (8, 12), step=0.123).x,
                            Grid2D((-3, 3), (8, 12), step=(0.123, 0.123)).x)


def test_Grid2D_plot():
    # This test is slow
    grid = Grid2D((-20, 20), (-40, 40), step=0.5)
    ax = grid.plot()
    npt.assert_equal(isinstance(ax, Axes), True)
    npt.assert_almost_equal(ax.get_xlim(), (-22, 22))

    # You can change the scaling:
    ax = grid.plot(transform=lambda x, y: (2*x, 2*y))
    npt.assert_equal(isinstance(ax, Axes), True)
    npt.assert_almost_equal(ax.get_xlim(), (-44, 44))

    # You can change the figure size
    ax = grid.plot(figsize=(9, 7))
    npt.assert_almost_equal(ax.figure.get_size_inches(), (9, 7))

    # You can change the style (smoke test):
    ax = grid.plot(style='hull')
    ax = grid.plot(style='cell')
    ax = grid.plot(style='scatter')

    # Step might be a tuple (smoke test):
    Grid2D((-5, 5), (-5, 5), step=(0.5, 1)).plot(style='cell')


class ValidCoordTransform(RetinalMap):

    def dva_to_ret(self, x_dva, y_dva):
        return x_dva, y_dva

    def ret_to_dva(self, x_ret, y_ret):
        return x_ret, y_ret


class ValidCorticalTransform(CorticalMap):
    def dva_to_v1(self, x, y):
        return x, y

    def dva_to_v2(self, x, y):
        return x, y

    def dva_to_v3(self, x, y):
        return x, y

    def v1_to_dva(self, x, y):
        return x, y

    def v2_to_dva(self, x, y):
        return x, y

    def v3_to_dva(self, x, y):
        return x, y


class NewRegionTransform(VisualFieldMap):

    def newlayer_transform(self, x, y):
        return x, y

    def from_dva(self):
        return {"newlayer" : self.newlayer_transform}


def test_grid_regions():
    # this also implicitly tests Cortical/RetinalMap

    grid = Grid2D((-2, 2), (-2, 2), step=1)
    # x is alias for dva.x. Test properties
    npt.assert_equal(grid.x, grid.dva.x)
    npt.assert_equal(grid.x, grid._grid['dva'].x)

    retinotopy = ValidCoordTransform()
    grid.build(retinotopy)
    # Make sure xret gets populated
    npt.assert_equal(grid.dva.x, grid.ret.x)

    grid = Grid2D((-2, 2), (-2, 2), step=1)
    retinotopy = ValidCorticalTransform(regions=['v1', 'v2', 'v3'])
    grid.build(retinotopy)
    npt.assert_equal(grid.x, grid.v1.x)
    npt.assert_equal(grid.x, grid.v2.x)
    npt.assert_equal(grid.x, grid.v3.x)

    # make sure that new layers are registered
    grid = Grid2D((-2, 2), (-2, 2), step=1)
    grid.build(NewRegionTransform())
    npt.assert_equal(grid.newlayer.x, grid.x)
    npt.assert_equal('newlayer' in grid.regions, True)


def test_Curcio1990Map():
    # Curcio1990 uses a linear dva_to_ret conversion factor:
    for factor in [0.0, 1.0, 2.0]:
        npt.assert_almost_equal(Curcio1990Map().dva_to_ret(factor, factor),
                                (280.0 * factor, 280.0 * factor))
    for factor in [0.0, 1.0, 2.0]:
        npt.assert_almost_equal(Curcio1990Map().ret_to_dva(280.0 * factor,
                                                      280.0 * factor),
                                (factor, factor))


def test_eq_Curcio19990Map():
    curcio_map = Curcio1990Map()

    # Assert not equal for differing classes
    npt.assert_equal(curcio_map == int, False)

    # Assert equal to itself
    npt.assert_equal(curcio_map == curcio_map, True)

    # Assert equal for shallow references
    copied = curcio_map
    npt.assert_equal(curcio_map == copied, True)

    # Assert deep copies are equal
    copied = copy.deepcopy(curcio_map)
    npt.assert_equal(curcio_map == copied, True)

    # Assert differing objects aren't equal
    differing_map = Watson2014Map()
    differing_map.a = 5
    npt.assert_equal(curcio_map == differing_map, False)


def test_Watson2014Map():
    trafo = Watson2014Map()
    with pytest.raises(ValueError):
        trafo.ret_to_dva(0, 0, coords='invalid')
    with pytest.raises(ValueError):
        trafo.dva_to_ret(0, 0, coords='invalid')

    # Below 15mm eccentricity, relationship is linear with slope 3.731
    npt.assert_equal(trafo.ret_to_dva(0.0, 0.0), (0.0, 0.0))
    for sign in [-1, 1]:
        for exp in [2, 3, 4]:
            ret = sign * 10 ** exp  # mm
            dva = 3.731 * sign * 10 ** (exp - 3)  # dva
            npt.assert_almost_equal(trafo.ret_to_dva(0, ret)[1], dva,
                                    decimal=3 - exp)  # adjust precision
    # Below 50deg eccentricity, relationship is linear with slope 0.268
    npt.assert_equal(trafo.dva_to_ret(0.0, 0.0), (0.0, 0.0))
    for sign in [-1, 1]:
        for exp in [-2, -1, 0]:
            dva = sign * 10 ** exp  # deg
            ret = 0.268 * sign * 10 ** (exp + 3)  # mm
            npt.assert_almost_equal(trafo.dva_to_ret(0, dva)[1], ret,
                                    decimal=-exp)  # adjust precision


def test_eq_Watson2014Map():
    map = Watson2014Map()

    # Assert not equal for differing classes
    npt.assert_equal(map == int, False)

    # Assert equal to itself
    npt.assert_equal(map == map, True)

    # Assert equal for shallow references
    copied = map
    npt.assert_equal(map == copied, True)

    # Assert deep copies are equal
    copied = copy.deepcopy(map)
    npt.assert_equal(map == copied, True)

    # Assert differing objects aren't equal
    differing_map = Watson2014Map()
    differing_map.a = 5
    npt.assert_equal(map == differing_map, False)


def test_Watson2014DisplaceMap():
    trafo = Watson2014DisplaceMap()
    with pytest.raises(ValueError):
        trafo.watson_displacement(0, meridian='invalid')
    npt.assert_almost_equal(trafo.watson_displacement(0), 0.4957506)
    npt.assert_almost_equal(trafo.watson_displacement(100), 0)

    # Check the max of the displacement function for the temporal meridian:
    radii = np.linspace(0, 30, 100)
    all_displace = trafo.watson_displacement(radii, meridian='temporal')
    npt.assert_almost_equal(np.max(all_displace), 2.153532)
    npt.assert_almost_equal(radii[np.argmax(all_displace)], 1.8181818)

    # Check the max of the displacement function for the nasal meridian:
    all_displace = trafo.watson_displacement(radii, meridian='nasal')
    npt.assert_almost_equal(np.max(all_displace), 1.9228664)
    npt.assert_almost_equal(radii[np.argmax(all_displace)], 2.1212121)
    # Smoke test
    trafo.dva_to_ret(0, 0)


def test_cart2pol():
    npt.assert_almost_equal(cart2pol(0, 0), (0, 0))
    npt.assert_almost_equal(cart2pol(10, 0), (0, 10))
    npt.assert_almost_equal(cart2pol(3, 4), (np.arctan(4 / 3.0), 5))
    npt.assert_almost_equal(cart2pol(4, 3), (np.arctan(3 / 4.0), 5))


def test_pol2cart():
    npt.assert_almost_equal(pol2cart(0, 0), (0, 0))
    npt.assert_almost_equal(pol2cart(0, 10), (10, 0))
    npt.assert_almost_equal(pol2cart(np.arctan(4 / 3.0), 5), (3, 4))
    npt.assert_almost_equal(pol2cart(np.arctan(3 / 4.0), 5), (4, 3))


def test_delta_angle():
    npt.assert_almost_equal(delta_angle(0.1, 0.2), 0.1)
    npt.assert_almost_equal(delta_angle(0.1, 0.2 + 2 * np.pi), 0.1)
    npt.assert_almost_equal(delta_angle(0.1, 0.2 - 2 * np.pi), 0.1)
    npt.assert_almost_equal(delta_angle(0.1 + 2 * np.pi, 0.2), 0.1)
    npt.assert_almost_equal(delta_angle(0.1 - 2 * np.pi, 0.2), 0.1)

    npt.assert_almost_equal(delta_angle(0.2, 0.1), -0.1)
    npt.assert_almost_equal(delta_angle(0.2, 0.1 + 2 * np.pi), -0.1)
    npt.assert_almost_equal(delta_angle(0.2, 0.1 - 2 * np.pi), -0.1)
    npt.assert_almost_equal(delta_angle(0.2 + 2 * np.pi, 0.1), -0.1)
    npt.assert_almost_equal(delta_angle(0.2 - 2 * np.pi, 0.1), -0.1)

    npt.assert_almost_equal(delta_angle(0, 2 * np.pi), 0)
    npt.assert_almost_equal(delta_angle(-np.pi, np.pi), 0)

    npt.assert_almost_equal(delta_angle(0, np.pi / 2), np.pi / 2)
    npt.assert_almost_equal(delta_angle(-np.pi / 2, 0), np.pi / 2)
    npt.assert_almost_equal(delta_angle(4*np.pi, np.pi, np.pi/2), 0)
