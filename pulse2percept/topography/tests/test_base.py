import numpy as np
import pytest
import numpy.testing as npt
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from pulse2percept.topography import (VisualFieldMap, RetinalMap, 
                                 CorticalMap, Grid2D, Polimeni2006Map,
                                 Watson2014Map)

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
    npt.assert_almost_equal(grid.y[0, 0], y_range[1])
    npt.assert_almost_equal(grid.y[0, -1], y_range[1])
    npt.assert_almost_equal(grid.y[-1, 0], y_range[0])
    npt.assert_almost_equal(grid.y[-1, -1], y_range[0])


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
                                mlt * np.array([-10, -5, 0, 5, 10])[::-1])

    # Another way to verify this is to manually check the step size:
    for step in [0.25, 0.5, 1, 2]:
        grid = Grid2D((-20, 20), (-40, 40), step=step)
        npt.assert_equal(len(np.unique(np.diff(grid.x[0, :]))), 1)
        npt.assert_equal(len(np.unique(np.diff(grid.y[:, 0]))), 1)
        npt.assert_almost_equal(np.unique(np.diff(grid.x[0, :]))[0], step)
        npt.assert_almost_equal(np.unique(np.diff(grid.y[:, 0]))[0], -step)

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
    npt.assert_almost_equal(grid.y[:, 0], [-1, -0.5, 0, 0.5, 1][::-1])

    # Same step size, but given explicitly:
    npt.assert_almost_equal(Grid2D((-3, 3), (8, 12), step=0.123).x,
                            Grid2D((-3, 3), (8, 12), step=(0.123, 0.123)).x)

class TestMapDouble(VisualFieldMap):
    def from_dva(self):
        return {
            "double": lambda x, y: (2*x, 2*y)
        }

@pytest.mark.parametrize('vfmap', [Watson2014Map(), Polimeni2006Map(regions=['v1', 'v2', 'v3'])])
def test_Grid2D_plot(vfmap):
    plt.figure()
    # This test is slow
    grid = Grid2D((-20, 20), (-40, 40), step=0.5)
    ax = grid.plot(use_dva=True)
    npt.assert_equal(isinstance(ax, Axes), True)
    npt.assert_almost_equal(ax.get_xlim(), (-22, 22))

    # You can change the scaling:
    grid.build(TestMapDouble())
    ax = grid.plot()
    npt.assert_equal(isinstance(ax, Axes), True)
    npt.assert_almost_equal(ax.get_xlim(), (-44, 44))

    # You can change the figure size
    ax = grid.plot(figsize=(9, 7))
    npt.assert_almost_equal(ax.figure.get_size_inches(), (9, 7))

    # Step might be a tuple (smoke test):
    grid = Grid2D((-5, 5), (-5, 5), step=(2, 1))
    grid.plot(style='cell', use_dva=True)

    plt.figure()
    grid = Grid2D((-5, 5), (-5, 5), step=1)
    grid.build(vfmap=vfmap)
    # You can change the style (smoke test):
    ax = grid.plot(style='hull')
    if isinstance(vfmap, Polimeni2006Map):
        npt.assert_equal(len(ax.patches), 6)
    elif isinstance(vfmap, Watson2014Map):
        npt.assert_equal(len(ax.patches), 1)
    ax = grid.plot(style='cell')
    ax = grid.plot(style='scatter')


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

    vfmap = ValidCoordTransform()
    grid.build(vfmap)
    # Make sure xret gets populated
    npt.assert_equal(grid.dva.x, grid.ret.x)

    grid = Grid2D((-2, 2), (-2, 2), step=1)
    vfmap = ValidCorticalTransform(regions=['v1', 'v2', 'v3'])
    grid.build(vfmap)
    npt.assert_equal(grid.x, grid.v1.x)
    npt.assert_equal(grid.x, grid.v2.x)
    npt.assert_equal(grid.x, grid.v3.x)

    # make sure that new layers are registered
    grid = Grid2D((-2, 2), (-2, 2), step=1)
    grid.build(NewRegionTransform())
    npt.assert_equal(grid.newlayer.x, grid.x)
    npt.assert_equal('newlayer' in grid.regions, True)