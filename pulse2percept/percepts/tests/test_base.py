import os
import numpy as np
import pytest
import numpy.testing as npt
import matplotlib.pyplot as plt
from matplotlib.axes import Subplot
from matplotlib.animation import FuncAnimation
from imageio import mimread
from skimage import img_as_float
from skimage.io import imread

from pulse2percept.percepts import Percept
from pulse2percept.utils import Grid2D


def test_Percept():
    # Automatic axes:
    ndarray = np.arange(15).reshape((3, 5, 1))
    percept = Percept(ndarray, metadata='meta')
    npt.assert_equal(percept.shape, ndarray.shape)
    npt.assert_equal(percept.metadata, 'meta')
    npt.assert_equal(hasattr(percept, 'xdva'), True)
    npt.assert_almost_equal(percept.xdva, np.arange(ndarray.shape[1]))
    npt.assert_equal(hasattr(percept, 'ydva'), True)
    npt.assert_almost_equal(percept.ydva, np.arange(ndarray.shape[0]))
    # Singleton dimensions can be None:
    npt.assert_equal(hasattr(percept, 'time'), True)
    npt.assert_equal(percept.time, None)

    # Specific labels:
    percept = Percept(ndarray, time=0.4)
    npt.assert_almost_equal(percept.time, [0.4])
    percept = Percept(ndarray, time=[0.4])
    npt.assert_almost_equal(percept.time, [0.4])

    # Labels from a grid.
    y_range = (-1, 1)
    x_range = (-2, 2)
    grid = Grid2D(x_range, y_range)
    percept = Percept(ndarray, space=grid)
    npt.assert_almost_equal(percept.xdva, grid._xflat)
    npt.assert_almost_equal(percept.ydva, grid._yflat)
    npt.assert_equal(percept.time, None)
    grid = Grid2D(x_range, y_range)
    percept = Percept(ndarray, space=grid, time=0)
    npt.assert_almost_equal(percept.xdva, grid._xflat)
    npt.assert_almost_equal(percept.ydva, grid._yflat)
    npt.assert_almost_equal(percept.time, [0])

    with pytest.raises(TypeError):
        Percept(ndarray, space={'x': [0, 1, 2], 'y': [0, 1, 2, 3, 4]})


def test_Percept__iter__():
    ndarray = np.zeros((2, 4, 3))
    ndarray[..., 1] = 1
    ndarray[..., 2] = 2
    percept = Percept(ndarray)
    for i, frame in enumerate(percept):
        npt.assert_equal(frame.shape, (2, 4))
        npt.assert_almost_equal(frame, i)


def test_Percept_argmax():
    percept = Percept(np.arange(30).reshape((3, 5, 2)))
    npt.assert_almost_equal(percept.argmax(), 29)
    npt.assert_almost_equal(percept.argmax(axis="frames"), 1)
    with pytest.raises(TypeError):
        percept.argmax(axis=(0, 1))
    with pytest.raises(ValueError):
        percept.argmax(axis='invalid')


def test_Percept_max():
    percept = Percept(np.arange(30).reshape((3, 5, 2)))
    npt.assert_almost_equal(percept.max(), 29)
    npt.assert_almost_equal(percept.max(axis="frames"),
                            percept.data[..., 1])
    npt.assert_almost_equal(percept.max(),
                            percept.data.ravel()[percept.argmax()])
    npt.assert_almost_equal(percept.max(axis='frames'),
                            percept.data[..., percept.argmax(axis='frames')])
    with pytest.raises(TypeError):
        percept.max(axis=(0, 1))
    with pytest.raises(ValueError):
        percept.max(axis='invalid')


def test_Percept_get_brightest_frame():
    percept = Percept(np.arange(30).reshape((3, 5, 2)))
    npt.assert_almost_equal(percept.get_brightest_frame(),
                            percept.data[..., 1])


def test_Percept_plot():
    y_range = (-1, 1)
    x_range = (-2, 2)
    grid = Grid2D(x_range, y_range)
    percept = Percept(np.arange(15).reshape((3, 5, 1)), space=grid)

    # Basic usage of pcolor:
    ax = percept.plot(kind='pcolor')
    npt.assert_equal(isinstance(ax, Subplot), True)
    npt.assert_almost_equal(ax.axis(), [*x_range, *y_range])
    frame = percept.get_brightest_frame()
    npt.assert_almost_equal(ax.collections[0].get_clim(),
                            [frame.min(), frame.max()])

    # Basic usage of hex:
    ax = percept.plot(kind='hex')
    npt.assert_equal(isinstance(ax, Subplot), True)
    npt.assert_almost_equal(ax.axis(), [percept.xdva[0], percept.xdva[-1],
                                        percept.ydva[0], percept.ydva[-1]])
    npt.assert_almost_equal(ax.collections[0].get_clim(),
                            [percept.data[..., 0].min(),
                             percept.data[..., 0].max()])

    # Verify color map:
    npt.assert_equal(ax.collections[0].cmap, plt.cm.gray)

    # Specify figsize:
    ax = percept.plot(kind='pcolor', figsize=(6, 4))
    npt.assert_almost_equal(ax.figure.get_size_inches(), (6, 4))

    # Test vmin and vmax
    ax.clear()
    ax = percept.plot(vmin=2, vmax=4)
    npt.assert_equal(ax.collections[0].get_clim(), (2., 4.))

    # Invalid calls:
    with pytest.raises(ValueError):
        percept.plot(kind='invalid')
    with pytest.raises(TypeError):
        percept.plot(ax='invalid')


@pytest.mark.parametrize('n_frames', (2, 3, 10, 14))
def test_Percept_play(n_frames):
    ndarray = np.random.rand(2, 4, n_frames)
    percept = Percept(ndarray)
    ani = percept.play()
    npt.assert_equal(isinstance(ani, FuncAnimation), True)
    npt.assert_equal(len(list(ani.frame_seq)), n_frames)


@pytest.mark.parametrize('dtype', (np.float32, np.uint8))
def test_Percept_save(dtype):
    ndarray = np.arange(256, dtype=dtype).repeat(31).reshape((-1, 16, 16))
    percept = Percept(ndarray.transpose((2, 0, 1)))

    # Save multiple frames as a gif or movie:
    for fname in ['test.mp4', 'test.avi', 'test.mov', 'test.wmv', 'test.gif']:
        print(fname)
        percept.save(fname)
        npt.assert_equal(os.path.isfile(fname), True)
        # Normalized to [0, 255] with some loss of precision:
        mov = mimread(fname)
        npt.assert_equal(np.min(mov) <= 2, True)
        npt.assert_equal(np.max(mov) >= 250, True)
        os.remove(fname)

    # Cannot save multiple frames image:
    fname = 'test.jpg'
    with pytest.raises(ValueError):
        percept.save(fname)

    # But, can save single frame as image:
    percept = Percept(ndarray[..., :1])
    for fname in ['test.jpg', 'test.png', 'test.tif', 'test.gif']:
        percept.save(fname)
        npt.assert_equal(os.path.isfile(fname), True)
        img = img_as_float(imread(fname))
        npt.assert_almost_equal(np.min(img), 0, decimal=3)
        npt.assert_almost_equal(np.max(img), 1.0, decimal=3)
        os.remove(fname)
