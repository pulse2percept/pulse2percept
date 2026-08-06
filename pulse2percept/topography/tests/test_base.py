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

# Parametrize over a factory, not over instances: arguments to `parametrize`
# are built at import time (on every pytest run, even when this test is
# deselected) and are shared across invocations.
@pytest.mark.parametrize('make_vfmap', [
    pytest.param(Watson2014Map, id='Watson2014Map'),
    pytest.param(lambda: Polimeni2006Map(regions=['v1', 'v2', 'v3']),
                 id='Polimeni2006Map'),
])
def test_Grid2D_plot(make_vfmap):
    vfmap = make_vfmap()
    plt.figure()
    # This test is slow
    grid = Grid2D((-20, 20), (-40, 40), step=1)
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


class Valid3DTransform(RetinalMap):

    def dva_to_ret(self, x_dva, y_dva):
        return x_dva, y_dva, np.ones_like(x_dva)

    def ret_to_dva(self, x_ret, y_ret, z_ret=None):
        return x_ret, y_ret
    
def test_3D_transform():
    grid = Grid2D((-2, 2), (-2, 2), step=1)
    vfmap = Valid3DTransform()
    grid.build(vfmap)
    npt.assert_equal(hasattr(grid.ret, 'z'), True)
    npt.assert_equal(grid.ret.x.shape, (5, 5))
    npt.assert_equal(grid.ret.y.shape, (5, 5))
    npt.assert_equal(grid.ret.z.shape, (5, 5))
    npt.assert_equal(grid.ret.x[0, 0], -2)
    npt.assert_equal(grid.ret.y[0, 0], 2)
    npt.assert_equal(grid.ret.z[0, 0], 1)
    npt.assert_equal(grid.ret.x[0, -1], 2)
    npt.assert_equal(grid.ret.y[0, -1], 2)
    npt.assert_equal(grid.ret.z[0, -1], 1)
    npt.assert_equal(grid.ret.x[-1, 0], -2)
    npt.assert_equal(grid.ret.y[-1, 0], -2)
    npt.assert_equal(grid.ret.z[-1, 0], 1)
    npt.assert_equal(grid.ret.x[-1, -1], 2)
    npt.assert_equal(grid.ret.y[-1, -1], -2)
    npt.assert_equal(grid.ret.z[-1, -1], 1)

def test_Grid2D_deepcopy_memo():
    import copy
    grid = Grid2D((-2, 2), (-2, 2), step=1)

    # Called directly, without a memo dict:
    copied = grid.__deepcopy__()
    npt.assert_equal(copied == grid, True)
    npt.assert_equal(id(copied) != id(grid), True)

    # The default memo must not persist between calls (a shared mutable
    # default would leak copies from one call into the next):
    other = Grid2D((-1, 1), (-1, 1), step=1)
    npt.assert_equal(other.__deepcopy__() == other, True)

    # An object already in the memo is returned as-is, not re-copied:
    sentinel = 'already copied'
    npt.assert_equal(grid.__deepcopy__({id(grid): sentinel}), sentinel)

    # Copies are independent of the original:
    copied = copy.deepcopy(grid)
    copied.x_range = (-9, 9)
    npt.assert_equal(grid.x_range != copied.x_range, True)


def test_Grid2D_plot3D_validation():
    grid = Grid2D((-2, 2), (-2, 2), step=1)
    grid.build(Watson2014Map())

    # A 2D axis cannot be used for a 3D plot:
    _, ax2d = plt.subplots()
    with pytest.raises(ValueError):
        grid.plot3D(ax=ax2d)

    # A 2D visual field map has nothing to plot in 3D:
    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection='3d')
    with pytest.raises(ValueError):
        grid.plot3D(ax=ax3d)

    # A 3D map gets past the ndim check, so style and surface are validated:
    grid3d = Grid2D((-2, 2), (-2, 2), step=1)
    grid3d.build(Valid3DTransform())
    with pytest.raises(ValueError):
        grid3d.plot3D(style='invalid', ax=ax3d)
    with pytest.raises(ValueError):
        grid3d.plot3D(surface='invalid', ax=ax3d)
    plt.close('all')


def test_CoordinateGrid():
    from pulse2percept.topography.base import CoordinateGrid
    x, y = np.arange(4.0), np.arange(4.0) * 2
    grid = CoordinateGrid(x, y)

    # 2D grid has no z:
    npt.assert_equal(grid.z, None)
    npt.assert_equal('CoordinateGrid(x=' in repr(grid), True)
    npt.assert_equal('z=' in repr(grid), False)
    npt.assert_equal(str(grid), repr(grid))

    # 3D grid reports z:
    grid3d = CoordinateGrid(x, y, np.ones(4))
    npt.assert_equal('z=' in repr(grid3d), True)
    npt.assert_equal('z=' in str(grid3d), True)

    # Equality: identity, equal values, different values, different type,
    # and a differing set of attributes:
    npt.assert_equal(grid == grid, True)
    npt.assert_equal(grid == CoordinateGrid(x.copy(), y.copy()), True)
    npt.assert_equal(grid == CoordinateGrid(x, y + 1), False)
    npt.assert_equal(grid == 'not a grid', False)
    npt.assert_equal(grid == grid3d, False)

    # Non-array attributes are compared directly:
    npt.assert_equal(CoordinateGrid(1, 2) == CoordinateGrid(1, 2), True)
    npt.assert_equal(CoordinateGrid(1, 2) == CoordinateGrid(1, 3), False)

    # Hashable, so grids can go in sets/dicts:
    npt.assert_equal(isinstance(hash(grid), int), True)
    npt.assert_equal(len({grid, grid}), 1)


class Valid3DMap(RetinalMap):
    """A 3D retinal map, so `plot3D` can be exercised without neuropythy."""

    def get_default_params(self):
        params = super().get_default_params()
        params.update({'ndim': 3, 'jitter_boundary': False})
        return params

    def dva_to_ret(self, x_dva, y_dva):
        x = np.asarray(x_dva, dtype=float)
        return x_dva, y_dva, np.ones_like(x)

    def ret_to_dva(self, x_ret, y_ret, z_ret=None):
        return x_ret, y_ret


def test_Grid2D_plot3D():
    grid = Grid2D((-2, 2), (-2, 2), step=1)
    grid.build(Valid3DMap())

    def new_ax():
        fig = plt.figure()
        return fig.add_subplot(111, projection='3d')

    # Both styles (smoke tests):
    for style in ['scatter', 'cell']:
        npt.assert_equal(grid.plot3D(style=style, ax=new_ax()) is not None, True)

    # All colorings:
    for color_by in ['region', 'eccentricity', 'angle']:
        npt.assert_equal(
            grid.plot3D(color_by=color_by, ax=new_ax()) is not None, True)
    with pytest.raises(ValueError):
        grid.plot3D(color_by='unknown', ax=new_ax())

    # An explicit color overrides `color_by`:
    npt.assert_equal(grid.plot3D(ax=new_ax(), c='blue') is not None, True)

    # Without an `ax`, a 3D axis is created. Close any open figures first:
    # if a 3D axis is already current, `plot3D` reuses it as-is.
    plt.close('all')
    npt.assert_equal(grid.plot3D() is not None, True)
    plt.close('all')
    ax = grid.plot3D(figsize=(9, 7))
    npt.assert_almost_equal(ax.figure.get_size_inches(), (9, 7))
    plt.close('all')
