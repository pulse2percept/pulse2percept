import numpy as np
import collections as coll
import pytest
import numpy.testing as npt
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

from pulse2percept.implants import (PointSource, ElectrodeArray, ElectrodeGrid,
                                    ProsthesisSystem, RectangleImplant)
from pulse2percept.stimuli import Stimulus, ImageStimulus, VideoStimulus
from pulse2percept.models import ScoreboardModel


def test_ProsthesisSystem():
    # Invalid instantiations:
    with pytest.raises(ValueError):
        ProsthesisSystem(ElectrodeArray(PointSource(0, 0, 0)),
                         eye='both')
    with pytest.raises(TypeError):
        ProsthesisSystem(Stimulus)

    # Iterating over the electrode array:
    implant = ProsthesisSystem(PointSource(0, 0, 0))
    npt.assert_equal(implant.n_electrodes, 1)
    npt.assert_equal(implant[0], implant.earray[0])
    npt.assert_equal(implant.electrode_names, implant.earray.electrode_names)
    for i, e in zip(implant, implant.earray):
        npt.assert_equal(i, e)

    # Set a stimulus after the constructor:
    npt.assert_equal(implant.stim, None)
    implant.stim = 3
    npt.assert_equal(isinstance(implant.stim, Stimulus), True)
    npt.assert_equal(implant.stim.shape, (1, 1))
    npt.assert_equal(implant.stim.time, None)
    npt.assert_equal(implant.stim.electrodes, [0])

    plt.cla()
    ax = implant.plot()
    npt.assert_equal(len(ax.texts), 0)
    npt.assert_equal(len(ax.collections), 1)

    with pytest.raises(ValueError):
        # Wrong number of stimuli
        implant.stim = [1, 2]
    with pytest.raises(TypeError):
        # Invalid stim type:
        implant.stim = "stim"
    # Invalid electrode names:
    with pytest.raises(ValueError):
        implant.stim = {'A1': 1}
    with pytest.raises(ValueError):
        implant.stim = Stimulus({'A1': 1})
    # Safe mode requires charge-balanced pulses:
    with pytest.raises(ValueError):
        implant = ProsthesisSystem(PointSource(0, 0, 0), safe_mode=True)
        implant.stim = 1

    # Slots:
    npt.assert_equal(hasattr(implant, '__slots__'), True)
    npt.assert_equal(hasattr(implant, '__dict__'), False)


def test_ProsthesisSystem_stim():
    implant = ProsthesisSystem(ElectrodeGrid((13, 13), 20))
    stim = Stimulus(np.ones((13 * 13 + 1, 5)))
    with pytest.raises(ValueError):
        implant.stim = stim

    # make sure empty stimulus causes None stim
    implant.stim = []
    npt.assert_equal(implant.stim, None)
    implant.stim = {}
    npt.assert_equal(implant.stim, None)
    implant.stim = np.array([])
    npt.assert_equal(implant.stim, None)

    # color mapping
    stim = np.zeros((13*13, 5))
    stim[84, 0] = 1
    stim[98, 2] = 2
    implant.stim = stim
    plt.cla()
    ax = implant.plot(stim_cmap='hsv')
    plt.colorbar()
    npt.assert_equal(len(ax.collections), 1)
    npt.assert_equal(ax.collections[0].colorbar.vmax, 2)
    npt.assert_equal(ax.collections[0].cmap(ax.collections[0].norm(1)),
                     (0.0, 1.0, 0.9647031631761764, 1))
    # make sure default behaviour unchanged
    plt.cla()
    ax = implant.plot()
    plt.colorbar()
    npt.assert_equal(len(ax.collections), 1)
    npt.assert_equal(ax.collections[0].colorbar.vmax, 1)
    npt.assert_equal(ax.collections[0].cmap(ax.collections[0].norm(1)),
                     (0.993248, 0.906157, 0.143936, 1))  

    # Deactivated electrodes cannot receive stimuli:
    implant.deactivate('H4')
    npt.assert_equal(implant['H4'].activated, False)
    implant.stim = {'H4': 1}
    npt.assert_equal('H4' in implant.stim.electrodes, False)

    implant.deactivate('all')
    npt.assert_equal(implant.stim.data.size == 0, True)
    implant.activate('all')
    implant.stim = {'H4': 1}
    npt.assert_equal('H4' in implant.stim.electrodes, True)


@pytest.mark.parametrize('rot', (0, 30, 92))
@pytest.mark.parametrize('gtype', ('hex', 'rect'))
@pytest.mark.parametrize('n_frames', (1, 3, 4))
def test_ProsthesisSystem_reshape_stim(rot, gtype, n_frames):
    implant = ProsthesisSystem(ElectrodeGrid((10, 10), 30, rot=rot, type=gtype))
    # Smoke test the automatic reshaping:
    n_px = 21
    implant.stim = ImageStimulus(np.ones((n_px, n_px, n_frames)).squeeze())
    npt.assert_equal(implant.stim.data.shape, (implant.n_electrodes, 1))
    npt.assert_equal(implant.stim.time, None)
    implant.stim = VideoStimulus(np.ones((n_px, n_px, 3 * n_frames)),
                                 time=2 * np.arange(3 * n_frames))
    npt.assert_equal(implant.stim.data.shape,
                     (implant.n_electrodes, 3 * n_frames))
    npt.assert_equal(implant.stim.time, 2 * np.arange(3 * n_frames))

    # Verify that a horizontal stimulus will always appear horizontally, even if
    # the device is rotated:
    data = np.zeros((50, 50))
    data[20:-20, 10:-10] = 1
    implant.stim = ImageStimulus(data)
    model = ScoreboardModel(xrange=(-1, 1), yrange=(-1, 1), rho=30, xystep=0.02)
    model.build()
    percept = label(model.predict_percept(implant).data.squeeze().T > 0.2)
    npt.assert_almost_equal(regionprops(percept)[0].orientation, 0, decimal=1)


def test_ProsthesisSystem_deactivate():
    implant = ProsthesisSystem(ElectrodeGrid((10, 10), 30))
    implant.stim = np.ones(implant.n_electrodes)
    electrode = 'A3'
    npt.assert_equal(electrode in implant.stim.electrodes, True)
    implant.deactivate(electrode)
    npt.assert_equal(implant[electrode].activated, False)
    npt.assert_equal(electrode in implant.stim.electrodes, False)

@pytest.mark.parametrize('ztype', ('float', 'list'))
@pytest.mark.parametrize('x', (-100, 200))
@pytest.mark.parametrize('y', (-200, 400))
@pytest.mark.parametrize('rot', (-45, 60))
def test_rectangle_implant(ztype, x, y, rot):
    # Create an argus like implant and make sure location is correct
    z = 100 if ztype == 'float' else np.ones(60) * 20
    implant = RectangleImplant(x=x, y=y, z=z, rot=rot, shape=(6, 10), r=112.5, spacing=575.0)

    # Slots:
    npt.assert_equal(hasattr(implant, '__slots__'), True)

    # Coordinates of first electrode
    xy = np.array([-2587.5, -1437.5]).T

    # Rotate
    rot_rad = np.deg2rad(rot)
    R = np.array([np.cos(rot_rad), -np.sin(rot_rad),
                  np.sin(rot_rad), np.cos(rot_rad)]).reshape((2, 2))
    xy = np.matmul(R, xy)

    # Then off-set: Make sure first electrode is placed
    # correctly
    npt.assert_almost_equal(implant['A1'].x, xy[0] + x)
    npt.assert_almost_equal(implant['A1'].y, xy[1] + y)

    # Make sure array center is still (x,y)
    y_center = implant['F1'].y + (implant['A10'].y - implant['F1'].y) / 2
    npt.assert_almost_equal(y_center, y)
    x_center = implant['A1'].x + (implant['F10'].x - implant['A1'].x) / 2
    npt.assert_almost_equal(x_center, x)

    # Make sure radius is correct
    for e in ['A1', 'B3', 'C5', 'D7', 'E9', 'F10']:
        npt.assert_almost_equal(implant[e].r, 112.5)

    # Indexing must work for both integers and electrode names
    for idx, (name, electrode) in enumerate(implant.electrodes.items()):
        npt.assert_equal(electrode, implant[idx])
        npt.assert_equal(electrode, implant[name])
    npt.assert_equal(implant["unlikely name for an electrode"], None)

    # Right-eye implant:
    xc, yc = 500, -500
    implant = RectangleImplant(eye='RE', x=xc, y=yc)
    npt.assert_equal(implant['A10'].x > implant['A1'].x, True)
    npt.assert_almost_equal(implant['A10'].y, implant['A1'].y)

    # Left-eye implant:
    implant = RectangleImplant(eye='LE', x=xc, y=yc)
    npt.assert_equal(implant['A1'].x > implant['A10'].x, True)
    npt.assert_almost_equal(implant['A10'].y, implant['A1'].y)

    # In both left and right eyes, rotation with positive angle should be
    # counter-clock-wise (CCW): for (x>0,y>0), decreasing x and increasing y
    for eye, el in zip(['LE', 'RE'], ['O1', 'O15']):
        # By default, electrode 'F1' in a left eye has the same coordinates as
        # 'F10' in a right eye (because the columns are reversed). Thus both
        # cases are testing an electrode with x>0, y>0:
        before = RectangleImplant(eye=eye)
        after = RectangleImplant(eye=eye, rot=20)
        npt.assert_equal(after[el].x < before[el].x, True)
        npt.assert_equal(after[el].y > before[el].y, True)

    # Set a stimulus via dict:
    implant = RectangleImplant(stim={'B7': 13})
    npt.assert_equal(implant.stim.shape, (1, 1))
    npt.assert_equal(implant.stim.electrodes, ['B7'])

    # Set a stimulus via array:
    implant = RectangleImplant(stim=np.ones(225))
    npt.assert_equal(implant.stim.shape, (225, 1))
    npt.assert_almost_equal(implant.stim.data, 1)

    # test different shapes
    for shape in [(6, 10), (5, 12), (15, 15)]:
        implant = RectangleImplant(shape=shape)
        npt.assert_equal(implant.earray.shape, shape)
